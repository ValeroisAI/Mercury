"""Run the complete Mercury RTL regression with one command."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    "check_sram_rtl.py",
    "check_dma_rtl.py",
    "check_fp32_rtl.py",
    "check_fp32_pipe_rtl.py",
    "check_stme_rtl.py",
    "check_stme_interleaved_rtl.py",
    "check_fma_rtl.py",
    "check_stme_fp8_dot_rtl.py",
    "check_stme_fp8_dot_reset_rtl.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yosys", default=shutil.which("yosys") or shutil.which("yowasp-yosys"))
    parser.add_argument(
        "--exhaustive", action="store_true",
        help="Check every FP8 product and the full 128-term boundary. Use 16,384 seeded FMA samples.",
    )
    parser.add_argument("--fpga", action="store_true", help="Also map the prototype blocks onto UltraScale+.")
    args = parser.parse_args()
    executable = shutil.which(args.yosys) if args.yosys else None
    if executable is None:
        parser.error("Install Yosys or pass its path with --yosys")
    yosys = Path(executable).resolve()
    coverage = {
        "check_stme_rtl.py": ("--exhaustive", "--full-group"),
        "check_fma_rtl.py": ("--random", "16384"),
    }
    checks = (*CHECKS, "map_rtl_fpga.py") if args.fpga else CHECKS
    for check in checks:
        flags = coverage.get(check, ()) if args.exhaustive else ()
        print(f"\n{check}", flush=True)
        code = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / check), "--yosys", str(yosys), *flags],
            cwd=ROOT, check=False,
        ).returncode
        if code:
            return code
    print(f"\nAll {len(checks)} RTL commands passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
