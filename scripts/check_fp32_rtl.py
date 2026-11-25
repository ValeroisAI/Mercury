"""Check rtl/mercury_fp32_add_rne.sv against an exact rational IEEE-754 oracle.

The checks are:
1. Elaboration plus ``check -assert`` and a cell count (``stat``).
2. Directed and seeded random vectors evaluated by Yosys ``eval``. Each result
   is compared bit for bit with an exact ``fractions.Fraction`` RNE rounding
   to binary32 that does not use the host FPU. A second oracle, host numpy
   float32 addition, is cross-checked against the first.
3. SAT proof (a few seconds) of the algebraic properties in
   tests/rtl/fp32_add_tb.sv over all 2^64 input pairs; skip with ``--no-formal``.

Run:  python scripts/check_fp32_rtl.py [--random 20000] [--seed 1] [--no-formal]
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile

_ROOT = Path(__file__).resolve().parents[1]
_RTL = "rtl/mercury_fp32_add_rne.sv"
_TB = "tests/rtl/fp32_add_tb.sv"
QNAN = 0x7FC00000
_BATCH = 4000

_ELABORATE = (
    f"read_verilog -sv {_RTL}; hierarchy -check -top mercury_fp32_add_rne; "
    "proc; opt; check -assert; stat"
)
_FORMAL = (
    f"read_verilog -formal -sv {_RTL} {_TB}; prep -top fp32_add_tb -flatten; opt; "
    "chformal -lower; sat -verify -prove-asserts -set-def-inputs"
)


# ---------------------------------------------------------------- exact oracle

def _decode(bits: int) -> tuple[str, int, Fraction | None]:
    sign = bits >> 31
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        return ("nan" if frac else "inf"), sign, None
    value = Fraction(frac, 1 << 23) * Fraction(2) ** -126 if exp == 0 else \
        Fraction((1 << 23) | frac, 1 << 23) * Fraction(2) ** (exp - 127)
    return "finite", sign, -value if sign else value


def _round_binary32(value: Fraction) -> int:
    """Round a nonzero exact value to binary32 with RNE, subnormals and overflow."""
    sign = 0x80000000 if value < 0 else 0
    mag = abs(value)
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    if Fraction(2) ** e > mag:
        e -= 1
    q = max(e, -126) - 23                       # exponent of one unit in the last place
    scaled = mag / Fraction(2) ** q
    n, rem = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rem
    if twice > scaled.denominator or (twice == scaled.denominator and n & 1):
        n += 1
    if n == 1 << 24:                            # rounding carried into the next binade
        n, q = n >> 1, q + 1
    if n < 1 << 23:                             # subnormal (q == -149)
        return sign | n
    biased = q + 23 + 127
    if biased >= 255:
        return sign | 0x7F800000
    return sign | (biased << 23) | (n & 0x7FFFFF)


def oracle_add(a: int, b: int) -> int:
    ka, sa, va = _decode(a)
    kb, sb, vb = _decode(b)
    if "nan" in (ka, kb) or (ka == kb == "inf" and sa != sb):
        return QNAN
    if ka == "inf":
        return a
    if kb == "inf":
        return b
    total = va + vb
    if total == 0:
        return 0x80000000 if (sa and sb) else 0
    return _round_binary32(total)


def host_add(a: int, b: int) -> int | None:
    """Host float32 addition through numpy, or None if numpy is missing."""
    try:
        import numpy as np
    except ImportError:
        return None
    with np.errstate(all="ignore"):
        x = np.array([a, b], dtype=np.uint32).view(np.float32)
        r = int((x[:1] + x[1:]).view(np.uint32)[0])
    return QNAN if (r & 0x7F800000) == 0x7F800000 and r & 0x7FFFFF else r


# ---------------------------------------------------------------- vectors

def _f(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", x))[0]


def directed_vectors() -> list[tuple[int, int]]:
    specials = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
                0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
                0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FBFFFFF, 0x3F800000, 0xBF800000,
                0x3F800001, 0xBF7FFFFF, 0x33800000, 0x33800001, 0x34000000, 0x4B000000,
                0x4B800000, 0x00400000, 0x00000003]
    vec = [(a, b) for a in specials for b in specials]
    vec += [
        (_f(1.0), 0x33800000),          # 1 + 2^-24: exact tie, rounds to even (1.0)
        (0x3F800001, 0x33800000),       # odd + tie: rounds up
        (_f(1.0), 0x33800001),          # just above the tie: rounds up
        (_f(1.0), 0xB3800000),          # 1 - 2^-24: exact in binary32 (0x3F7FFFFF)
        (_f(1.0), 0xB3000000),          # 1 - 2^-25: tie on the lower binade
        (0x7F7FFFFF, 0x73800000),       # max + half ulp: tie -> even = overflow to inf
        (0x7F7FFFFF, 0x737FFFFF),       # max + just under half ulp: stays max
        (0x00800000, 0x80000001),       # min normal - min subnormal -> max subnormal
        (0x007FFFFF, 0x00000001),       # max subnormal + min subnormal -> min normal
        (_f(1.5), _f(-1.5)), (_f(-0.0), _f(-0.0)), (_f(0.0), _f(-0.0)),
        (0x4B800001, 0xCB800000),       # catastrophic cancellation
        (0x3F800000, 0x0D800000),       # large exponent gap: sticky only
    ]
    return vec


def random_vectors(count: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    out: list[tuple[int, int]] = []
    for i in range(count):
        kind = i % 8
        a = rng.getrandbits(32)
        if kind == 0:                                     # raw bits (any class)
            b = rng.getrandbits(32)
        elif kind == 1:                                   # same exponent, opposite sign: cancellation
            b = (a ^ 0x80000000) ^ rng.getrandbits(rng.randint(1, 23))
        elif kind == 2:                                   # both subnormal / tiny
            a &= 0x80FFFFFF
            a &= ~0x7F000000
            b = rng.getrandbits(32) & 0x80FFFFFF & ~0x7F000000
        elif kind == 3:                                   # exponent gap 22..28 (guard/round/sticky)
            ea = rng.randint(30, 250)
            a = (a & 0x807FFFFF) | (ea << 23)
            b = (rng.getrandbits(32) & 0x807FFFFF) | ((ea - rng.randint(22, 28)) << 23)
        elif kind == 4:                                   # constructed ties: b = half ulp of a times odd
            ea = rng.randint(30, 250)
            a = (a & 0x807FFFFF) | (ea << 23)
            b = (rng.getrandbits(1) << 31) | ((ea - 24) << 23) | (rng.getrandbits(1) << 22)
        elif kind == 5:                                   # near overflow
            a = (a & 0x807FFFFF) | (rng.randint(252, 254) << 23)
            b = (rng.getrandbits(32) & 0x807FFFFF) | (rng.randint(250, 254) << 23)
        elif kind == 6:                                   # exponent gap 0..3, mixed signs
            ea = rng.randint(4, 254)
            a = (a & 0x807FFFFF) | (ea << 23)
            b = (rng.getrandbits(32) & 0x807FFFFF) | ((ea - rng.randint(0, 3)) << 23)
        else:                                             # normal/subnormal boundary
            a = (a & 0x807FFFFF) | (rng.randint(0, 2) << 23)
            b = (rng.getrandbits(32) & 0x807FFFFF) | (rng.randint(0, 2) << 23)
        out.append((a & 0xFFFFFFFF, b & 0xFFFFFFFF))
    return out


# ---------------------------------------------------------------- yosys plumbing

def _yosys_command(explicit: str | None) -> list[str]:
    exe = explicit or shutil.which("yosys") or shutil.which("yowasp-yosys")
    if exe:
        found = shutil.which(exe)
        if not found:
            raise SystemExit(f"Yosys executable not found: {exe}")
        return [str(Path(found).resolve())]
    try:
        import yowasp_yosys  # noqa: F401
    except ImportError:
        raise SystemExit("Install Yosys or YoWASP Yosys (pip install yowasp-yosys), or pass --yosys")
    return [sys.executable, "-c", "import sys, yowasp_yosys; sys.exit(yowasp_yosys.run_yosys(sys.argv[1:]))"]


def _run(cmd: list[str], script: str, log: Path | None = None) -> int:
    """Run a Yosys script from a file (vector batches exceed the Windows command-line limit)."""
    env = {**os.environ, "YOWASP_CACHE_DIR": str(_ROOT / "build" / "rtl-cache")}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "job.ys"
        path.write_text(script.replace("; ", "\n"), encoding="utf-8")
        args = cmd + (["-q", "-l", log.as_posix()] if log else []) + ["-s", path.as_posix()]
        return subprocess.run(args, cwd=_ROOT, check=False, env=env).returncode


def _parse_eval(text: str) -> int:
    """Yosys prints decimal for values below 2^31 and 32'<binary> otherwise; x/z bits fail."""
    if text.startswith("32'"):
        bits = text[3:]
        if set(bits) - {"0", "1"}:
            raise SystemExit(f"undefined RTL output bits: {text}")
        return int(bits, 2)
    return int(text)


def evaluate(cmd: list[str], vectors: list[tuple[int, int]]) -> list[int]:
    results: list[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        for start in range(0, len(vectors), _BATCH):
            chunk = vectors[start:start + _BATCH]
            log = Path(tmp) / "eval.log"
            script = (f"read_verilog -sv {_RTL}; hierarchy -top mercury_fp32_add_rne; proc; opt; "
                      + "; ".join(f"eval -set a 32'h{a:08x} -set b 32'h{b:08x} -show y" for a, b in chunk))
            if _run(cmd, script, log):
                raise SystemExit("Yosys eval failed")
            values = [_parse_eval(m) for m in re.findall(r"Eval result: \\y = ([^\s]+)\.\n", log.read_text())]
            if len(values) != len(chunk):
                raise SystemExit(f"expected {len(chunk)} eval results, parsed {len(values)}")
            results += values
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yosys")
    parser.add_argument("--random", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-formal", action="store_true", help="skip the SAT property proof")
    opts = parser.parse_args()
    cmd = _yosys_command(opts.yosys)

    if _run(cmd, _ELABORATE):
        return 1
    vectors = directed_vectors() + random_vectors(opts.random, opts.seed)
    got = evaluate(cmd, vectors)
    failures = host_mismatch = 0
    for (a, b), y in zip(vectors, got):
        want = oracle_add(a, b)
        host = host_add(a, b)
        if host is not None and host != want:
            host_mismatch += 1
        if y != want:
            failures += 1
            if failures <= 10:
                print(f"MISMATCH a={a:08x} b={b:08x} rtl={y:08x} oracle={want:08x}")
    print(f"{len(vectors)} vectors, {failures} RTL mismatches, {host_mismatch} oracle-vs-host disagreements")
    if failures or host_mismatch:
        return 1
    if not opts.no_formal and _run(cmd, _FORMAL):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
