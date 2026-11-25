"""Compare FP8 RTL products with the Python oracle and check group-sum handshakes."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mercury_sim.compute_isa import decode_fp8

_READ_PRODUCT = "read_verilog -sv rtl/mercury_fp8_product.sv; prep -top mercury_fp8_product; "
_PRODUCT_ROW = re.compile(r"^\s*(?:8'([01]{8})\s+)?8'([01]{8})\s*\|\s*32'([01]{32})\s*$")
_SAMPLE_B = (0x00, 0x01, 0x02, 0x07, 0x38, 0x40, 0x7B, 0x7C, 0x7E, 0x7F, 0x80, 0x81, 0xFB, 0xFF)
_READ_GROUP = (
    "read_verilog -formal -sv rtl/mercury_fp8_product.sv rtl/mercury_fp32_add_rne.sv "
    "rtl/mercury_stme_group_sum.sv tests/rtl/stme_group_tb.sv; "
)
_GROUP_CHECK = (
    _READ_GROUP + "prep -top stme_group_tb -flatten; memory_map; opt; chformal -lower; "
    "sat -verify -prove-asserts -seq 14 -set-at 1 rst 1 -set-def-inputs"
)
_STRUCTURE_CHECK = (
    _READ_GROUP + "hierarchy -check -top mercury_stme_group_sum; proc; opt; check -assert"
)
_FULL_GROUP_CHECK = (
    _READ_GROUP + "prep -top stme_group_128_tb -flatten; memory_map; opt; chformal -lower; "
    "sat -verify -prove-asserts -seq 134 -set rst 0 -set-at 1 rst 1 -set-def-inputs"
)


def _expected(lhs: float, rhs: float) -> int:
    value = lhs * rhs
    if math.isnan(value):
        return 0x7FC00000
    return int.from_bytes(struct.pack(">f", value), "big")


def _products(command: list[str], env: dict[str, str], exhaustive: bool) -> int:
    formats = [[decode_fp8(i, fmt) for i in range(256)] for fmt in ("e4m3", "e5m2")]
    expected_count = 256 * (256 if exhaustive else len(_SAMPLE_B))
    for left, right in itertools.product(range(2), repeat=2):
        mode = f"-set a_e5m2 {left} -set b_e5m2 {right}"
        jobs = ["-table a -table b"] if exhaustive else [f"-set b 8'h{b:02x} -table a" for b in _SAMPLE_B]
        yosys = _READ_PRODUCT + "; ".join(f"eval {mode} {job} -show y" for job in jobs)
        seen = set()
        with subprocess.Popen([*command, "-p", yosys], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                row = _PRODUCT_ROW.match(line)
                if row is None:
                    continue
                b_column, a_column, y_column = row.groups()
                a = int(a_column, 2)
                b = int(b_column, 2) if exhaustive else _SAMPLE_B[len(seen) // 256]
                actual = int(y_column, 2)
                expected = _expected(formats[left][a], formats[right][b])
                if actual != expected:
                    raise AssertionError(
                        f"FP8 product mismatch: formats={left},{right} a={a:#04x} b={b:#04x} "
                        f"RTL={actual:#010x} Python={expected:#010x}"
                    )
                seen.add((a, b))
            if proc.wait():
                raise RuntimeError("Yosys could not evaluate the FP8 RTL")
        if len(seen) != expected_count:
            raise AssertionError(f"Expected {expected_count} unique FP8 pairs, found {len(seen)}")
    return 4 * expected_count


def main() -> int:
    """Run the numerical product checks and optional sequential proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yosys", default=shutil.which("yosys") or shutil.which("yowasp-yosys"))
    parser.add_argument("--exhaustive", action="store_true", help="Check all 262,144 FP8 operand/format combinations")
    parser.add_argument("--product-only", action="store_true", help="Skip the group controller and adder")
    parser.add_argument("--full-group", action="store_true", help="Also check the full 128-term boundary")
    options = parser.parse_args()
    executable = shutil.which(options.yosys) if options.yosys else None
    if executable is None:
        parser.error("Install Yosys or pass its path with --yosys")
    env = {**os.environ, "YOWASP_CACHE_DIR": str(ROOT / "build" / "rtl-cache")}
    command = [str(Path(executable).resolve()), "-Q", "-T"]
    count = _products(command, env, options.exhaustive)
    print(f"FP8 RTL matched the Python reference for {count:,} input combinations.", flush=True)
    if not options.product_only:
        jobs = (_STRUCTURE_CHECK, _GROUP_CHECK)
        if options.full_group:
            jobs += (_FULL_GROUP_CHECK,)
        for job in jobs:
            status = subprocess.run([*command, "-p", job], cwd=ROOT, env=env, check=False).returncode
            if status:
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
