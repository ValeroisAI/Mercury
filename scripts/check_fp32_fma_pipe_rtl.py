"""Check FMA pipeline latency and bit parity with the scalar reference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from check_fma_rtl import ROOT, _DIRECTED, check_vectors

_PIPE = (
    "read_verilog -sv rtl/mercury_fp32_fma_pipe.sv; hierarchy -top mercury_fp32_fma_pipe; "
    "proc; techmap -map tests/rtl/fp32_fma_pipe_passthrough.v; opt; select -assert-none t:$dff"
)
_READ_TB = (
    "read_verilog -formal -sv rtl/mercury_fp32_fma_rne.sv rtl/mercury_fp32_fma_pipe.sv "
    "tests/rtl/fp32_fma_pipe_tb.sv; prep -top fp32_fma_pipe_tb -flatten; "
    "memory_map; opt; chformal -lower; "
)


def _timing_job() -> str:
    # Two operations are flushed by reset. The subsequent stream has both
    # consecutive valid inputs and a bubble, with different arithmetic cases.
    inputs = [(2, _DIRECTED[0]), (3, _DIRECTED[1])]
    inputs += [(5 + i + (i >= 4), v) for i, v in enumerate(_DIRECTED)]
    constraints = " ".join(
        f"-set-at {step} in_valid 1 -set-at {step} a 32'h{a:08x} "
        f"-set-at {step} b 32'h{b:08x} -set-at {step} c 32'h{c:08x}"
        for step, (a, b, c) in inputs
    )
    return (
        _READ_TB + f"sat -verify -prove-asserts -seq {inputs[-1][0] + 7} "
        "-set rst 0 -set-at 1 rst 1 -set-at 4 rst 1 "
        "-set in_valid 0 -set-def-inputs " + constraints
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yosys", default=shutil.which("yosys") or shutil.which("yowasp-yosys"))
    parser.add_argument("--random", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=127)
    args = parser.parse_args()
    executable = shutil.which(args.yosys) if args.yosys else None
    if executable is None:
        parser.error("Install Yosys or pass its path with --yosys")
    if args.random < 0:
        parser.error("--random must be nonnegative")
    command = [str(Path(executable).resolve()), "-Q", "-T"]
    env = {**os.environ, "YOWASP_CACHE_DIR": str(ROOT / "build" / "rtl-cache")}
    count = check_vectors(command, env, _PIPE, args.random, args.seed)
    print(f"FMA pipeline arithmetic matched the exact Python reference for {count:,} vectors.", flush=True)
    return subprocess.run([*command, "-p", _timing_job()], cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
