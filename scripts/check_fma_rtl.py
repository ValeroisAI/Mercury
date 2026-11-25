"""Check the binary32 fused group-boundary datapath against the exact Python FMA."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mercury_sim.compute_isa import _f32, _fma_f32

_READ = "read_verilog -sv rtl/mercury_fp32_fma_rne.sv; prep -top mercury_fp32_fma_rne"
_BOUNDARY_CHECK = (
    "read_verilog -formal -sv rtl/mercury_fp32_fma_rne.sv rtl/mercury_stme_boundary_accumulator.sv "
    "tests/rtl/stme_boundary_tb.sv; prep -top stme_boundary_tb -flatten; opt; chformal -lower; "
    "sat -verify -prove-asserts -seq 8 -set rst 0 -set-at 1 rst 1 -set-def-inputs"
)
_EVAL_RESULT = re.compile(r"Eval result: \\y = (?:32'([01]{32})|([0-9]+))\.")
_DIRECTED = (
    (0x3F800001, 0x3F7FFFFE, 0xBF800000),  # fused cancellation: -2**-46
    (0x7F7FFFFF, 0x40000000, 0xFF7FFFFF),  # intermediate overflow can cancel
    (0x00000001, 0x3F000000, 0),          # half a minimum subnormal ties to zero
    (0x80000001, 0x3F000000, 0),          # negative underflow preserves sign
    (0x00000001, 0x3F000000, 1),          # subnormal tie rounds to even
    (0x00800000, 0x3F7FFFFF, 0),          # min normal / max subnormal boundary
    (0x00800000, 0x3F7FFFFE, 0),
    (0x7F7FFFFF, 0x40000000, 0),
    (0x7F800000, 0, 0x3F800000),
    (0xFF800000, 0xBF800000, 0xFF800000),
    (0x7F800000, 0xBF800000, 0),
    (0x3F800000, 0x40000000, 0x7F800000),
    (0x7F800001, 0x3F800000, 0x3F800000),
    (0x3F800000, 0xFFC00001, 0x3F800000),
    (0x3F800000, 0x40000000, 0x7F800001),
    (0x80000000, 0x3F800000, 0x80000000),
    (0x80000000, 0x3F800000, 0),
)


def _float(bits: int) -> float:
    return struct.unpack(">f", bits.to_bytes(4, "big"))[0]


def _bits(value: float) -> int:
    return 0x7FC00000 if math.isnan(value) else int.from_bytes(struct.pack(">f", value), "big")


def _vectors(count: int, seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    cases = list(_DIRECTED)
    for _ in range(count):
        a, b, c = (rng.getrandbits(32) for _ in range(3))
        cases.append((a, b, c))
        fp_a, fp_b = _float(a), _float(b)
        if math.isfinite(fp_a) and math.isfinite(fp_b):
            product_rounded = _f32(fp_a * fp_b)
            if math.isfinite(product_rounded):
                # Large cancellation stresses exact preservation of the product tail.
                cases.append((a, b, _bits(-product_rounded)))
    return cases


def check_vectors(
    command: list[str], env: dict[str, str], read: str, count: int, seed: int,
) -> int:
    """Compare Yosys evaluations of a binary32 FMA datapath to the exact reference."""
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    codes = _vectors(count, seed)
    with tempfile.TemporaryDirectory(prefix="fma-rtl-", dir=build) as folder:
        script = Path(folder) / "vectors.ys"
        commands = [read, "check -assert"]
        commands.extend(
            f"eval -set a 32'h{a:08x} -set b 32'h{b:08x} -set c 32'h{c:08x} -show y" for a, b, c in codes
        )
        script.write_text("\n".join(commands) + "\n", encoding="ascii")
        run = subprocess.run(
            [*command, "-s", script.relative_to(ROOT).as_posix()],
            cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE, text=True,
        )
    values = [int(bits, 2) if bits else int(decimal) for bits, decimal in _EVAL_RESULT.findall(run.stdout)]
    if len(values) != len(codes):
        raise AssertionError(f"Expected {len(codes)} binary32 outputs, got {len(values)}")
    for (a, b, c), actual in zip(codes, values, strict=True):
        expected = _bits(_fma_f32(_float(a), _float(b), _float(c)))
        if actual != expected:
            raise AssertionError(
                f"FMA a={a:#010x} b={b:#010x} c={c:#010x}: RTL={actual:#010x} Python={expected:#010x}"
            )
    return len(codes)


def main() -> int:
    """Run Yosys elaboration and compare directed and seeded FMA results bitwise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yosys", default=shutil.which("yosys") or shutil.which("yowasp-yosys"))
    parser.add_argument("--random", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=127)
    options = parser.parse_args()
    path = shutil.which(options.yosys) if options.yosys else None
    if path is None:
        parser.error("Install Yosys or pass its path with --yosys")
    if options.random < 0:
        parser.error("--random must be nonnegative")
    command = [str(Path(path).resolve()), "-Q", "-T"]
    env = {**os.environ, "YOWASP_CACHE_DIR": str(ROOT / "build" / "rtl-cache")}
    count = check_vectors(command, env, _READ, options.random, options.seed)
    print(f"FMA RTL matched the exact Python reference for {count:,} vectors.", flush=True)
    return subprocess.run(
        [*command, "-p", _BOUNDARY_CHECK],
        cwd=ROOT, env=env, check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
