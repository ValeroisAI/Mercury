"""Bounded formal check of rtl/mercury_sram_payload.sv (banked Shared SRAM payload routing).

The harness tests/rtl/sram_payload_tb.sv uses 4 requesters, 4 banks, 8-bit
line addresses and 32-bit data, with all requester and bank inputs free. A
macro model only returns data for reads accepted earlier since the last reset.
1. ``sat -seq 6`` from reset proves the scoreboard assertions.
2. Each reachability probe (``-DPROBE_*``, run with ``sat -falsify``) must find a counterexample. That shows the
   scenario is reachable within the bound, so the main proof is not vacuous:
   two banks issuing in one cycle, a held response through a stall, a write
   right after a consumed read, and a reset flush followed by a new read.

Run:  python scripts/check_sram_payload_rtl.py --yosys build/rtl-env/Scripts/yowasp-yosys.exe
Expect a few minutes per SAT job.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

_ROOT = Path(__file__).resolve().parents[1]
_SOURCES = "rtl/mercury_sram_arbiter.sv rtl/mercury_sram_payload.sv tests/rtl/sram_payload_tb.sv"
_PROBES = ("PROBE_TWO_BANKS", "PROBE_HELD_STALL", "PROBE_WRITE_AFTER_READ", "PROBE_RESET_FLUSH")


def _job(define: str = "", seq: int = 6) -> str:
    # Probes use -falsify: exit 0 only when SAT finds the scenario; parse errors and
    # timeouts are nonzero, so they can never count as reached. -defer skips
    # elaborating the default 8x64x512 parameterisation.
    flag = f"-D{define} " if define else ""
    mode = "-falsify" if define else "-verify"
    return (
        f"read_verilog -defer -formal -sv {flag}{_SOURCES}; "
        "prep -top sram_payload_tb -flatten; memory_map; opt; chformal -lower; "
        f"sat {mode} -prove-asserts -seq {seq} -set-at 1 rst 1 -set-def-inputs"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yosys", default=shutil.which("yosys") or shutil.which("yowasp-yosys"))
    parser.add_argument("--seq", type=int, default=6)
    parser.add_argument("--skip-probes", action="store_true")
    opts = parser.parse_args()
    exe = opts.yosys and shutil.which(opts.yosys)
    if not exe:
        parser.error("Install Yosys or YoWASP Yosys, or pass --yosys")
    env = {**os.environ, "YOWASP_CACHE_DIR": str(_ROOT / "build" / "rtl-cache")}

    def run(script: str) -> int:
        return subprocess.run([str(Path(exe).resolve()), "-q", "-p", script], cwd=_ROOT,
                              check=False, env=env, stdout=subprocess.DEVNULL).returncode

    if run(_job(seq=opts.seq)):
        print("FAIL: scoreboard assertions")
        return 1
    print(f"PASS: scoreboard assertions (seq {opts.seq})")
    if opts.skip_probes:
        return 0
    unreached = [p for p in _PROBES if run(_job(p, opts.seq)) != 0]
    for probe in _PROBES:
        print(f"{'UNREACHED' if probe in unreached else 'reached'}: {probe}")
    return 1 if unreached else 0


if __name__ == "__main__":
    raise SystemExit(main())
