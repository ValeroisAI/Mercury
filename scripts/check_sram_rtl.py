"""Elaborate the Shared SRAM arbiter and run its formal checks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

_ROOT = Path(__file__).resolve().parents[1]
_CHECK = (
    "read_verilog -formal -sv rtl/mercury_sram_arbiter.sv tests/rtl/sram_arbiter_tb.sv; "
    "prep -top sram_arbiter_tb -flatten; memory_map; opt; chformal -lower; "
    "sat -verify -prove-asserts -seq 7 -set-at 1 rst 1 -set-def-inputs"
)
_ELABORATE = (
    "read_verilog -sv rtl/mercury_sram_arbiter.sv; "
    "hierarchy -check -top mercury_sram_arbiter; proc; opt; check -assert; stat"
)
_MAP_CHECK = (
    "read_verilog -formal -sv rtl/mercury_sram_arbiter.sv tests/rtl/sram_arbiter_tb.sv; "
    "prep -top sram_bank_map_tb -flatten; chformal -lower; "
    "sat -verify -prove-asserts -set-def-inputs"
)


def run_yosys_jobs(description: str, jobs: tuple[str, ...]) -> int:
    """Resolve Yosys and return its verification exit status."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--yosys", default=shutil.which("yosys") or shutil.which("yowasp-yosys"))
    options = parser.parse_args()
    if not options.yosys:
        parser.error("Install Yosys or YoWASP Yosys, add it to PATH, or use --yosys")
    command = shutil.which(options.yosys)
    if command is None:
        parser.error(f"Yosys executable not found: {options.yosys}")
    executable = Path(command).resolve()
    env = {**os.environ, "YOWASP_CACHE_DIR": str(_ROOT / "build" / "rtl-cache")}
    for job in jobs:
        code = subprocess.run([str(executable), "-p", job], cwd=_ROOT, check=False, env=env).returncode
        if code:
            return code
    return 0


def main() -> int:
    """Run structural and formal checks of the SRAM control block."""
    return run_yosys_jobs(__doc__ or "", (_ELABORATE, _MAP_CHECK, _CHECK))


if __name__ == "__main__":
    raise SystemExit(main())
