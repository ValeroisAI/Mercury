"""Simulate the full 8-requester, 64-bank, 512-bit SRAM payload path."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        parser.error("Install Icarus Verilog and add both iverilog and vvp to PATH")
    root = Path(__file__).resolve().parents[1]
    build = root / "build"
    build.mkdir(exist_ok=True)
    simulation = build / "sram-full-load.vvp"
    env = {**os.environ, "TMPDIR": str(build), "TMP": str(build), "TEMP": str(build)}
    result = subprocess.run([
        iverilog, "-g2012", "-s", "sram_full_load_tb", "-o", str(simulation),
        "rtl/mercury_sram_arbiter.sv", "rtl/mercury_sram_payload.sv",
        "tests/rtl/sram_full_load_tb.sv",
    ], cwd=root, env=env, check=False)
    if result.returncode:
        return result.returncode
    return subprocess.run([vvp, "-i", str(simulation)], cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
