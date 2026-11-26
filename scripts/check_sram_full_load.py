"""Simulate the full-size SRAM payload path and the FPGA BRAM integration."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

TESTBENCHES = {
    "load": ("sram_full_load_tb", ()),
    "bram": ("sram_bram_tb", ("rtl/mercury_sram_bram.sv",)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=TESTBENCHES, help="Run only the selected testbench.")
    args = parser.parse_args()
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        parser.error("Install Icarus Verilog and add both iverilog and vvp to PATH")
    root = Path(__file__).resolve().parents[1]
    build = root / "build"
    build.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(build), "TMP": str(build), "TEMP": str(build)}
    for case in (args.case,) if args.case else TESTBENCHES:
        top, extra_sources = TESTBENCHES[case]
        simulation = build / f"{top}.vvp"
        result = subprocess.run([
            iverilog, "-g2012", "-s", top, "-o", str(simulation),
            "rtl/mercury_sram_arbiter.sv", "rtl/mercury_sram_payload.sv", *extra_sources,
            f"tests/rtl/{top}.sv",
        ], cwd=root, env=env, check=False)
        if result.returncode:
            return result.returncode
        result = subprocess.run([vvp, "-i", str(simulation)], cwd=root, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
