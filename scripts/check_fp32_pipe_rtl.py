"""Verify and measure rtl/mercury_fp32_add_pipe.sv (FP32 RNE adder, latency 4, II = 1).

1. Elaboration and ``check -assert``.
2. Bounded formal equivalence from reset (``sat -seq 7``) against
   rtl/mercury_fp32_add_rne.sv on operands delayed by four cycles
   (tests/rtl/fp32_add_pipe_tb.sv). It covers every operand value and every
   in_valid pattern, because the pipeline holds no state beyond in-flight data.
3. Structural measurement of both designs with two flows, each applied to
   both designs the same way. Each gives the longest generic 2-input-gate path
   between flops and ports (``ltp -noff``) plus total cell and flop counts:
     default  ``synth -flatten; abc -g AND,NAND,OR,NOR,XOR,XNOR,MUX``
     single   ``synth -flatten -noabc; abc -g ...``. The default flow runs ABC
              twice, and its area-oriented rewriting re-ripples explicit
              parallel-prefix carry logic; this flow keeps that structure.
   These are structural levels only, not a timing claim for any process node.

Run:  python scripts/check_fp32_pipe_rtl.py
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fp32_rtl import _ROOT, _yosys_command  # noqa: E402

_SOURCES = "rtl/mercury_fp32_add_rne.sv rtl/mercury_fp32_add_pipe.sv"
_ELABORATE = (
    "read_verilog -sv rtl/mercury_fp32_add_pipe.sv; hierarchy -check -top mercury_fp32_add_pipe; "
    "proc; opt; check -assert"
)
_EQUIV = (
    f"read_verilog -formal -sv {_SOURCES} tests/rtl/fp32_add_pipe_tb.sv; "
    "prep -top fp32_add_pipe_tb -flatten; memory_map; opt; chformal -lower; "
    "sat -verify -prove-asserts -seq 7 -set-at 1 rst 1 -set-def-inputs"
)


def _run(cmd: list[str], script: str, log: Path) -> int:
    env = {**os.environ, "YOWASP_CACHE_DIR": str(_ROOT / "build" / "rtl-cache")}
    job = log.with_suffix(".ys")
    job.write_text(script.replace("; ", "\n"), encoding="utf-8")
    args = cmd + ["-q", "-l", log.as_posix(), "-s", job.as_posix()]
    return subprocess.run(args, cwd=_ROOT, check=False, env=env).returncode


def measure(cmd: list[str], top: str, tmp: Path, flow: str) -> dict[str, int]:
    log = tmp / f"synth_{top}_{flow}.log"
    noabc = " -noabc" if flow == "single" else ""
    script = (f"read_verilog -sv {_SOURCES}; synth -top {top} -flatten{noabc}; "
              "abc -g AND,NAND,OR,NOR,XOR,XNOR,MUX; opt_clean; stat; ltp -noff")
    if _run(cmd, script, log):
        raise SystemExit(f"synthesis of {top} failed")
    text = log.read_text()
    text = text[text.rfind("Printing statistics"):]   # the final stat only
    cells = int(re.findall(r"^\s+(\d+) cells$", text, re.M)[-1])
    flops = sum(int(n) for n in re.findall(r"^\s+(\d+)\s+\$_S?DFFE?_\w+_$", text, re.M))
    depth = int(re.search(r"Longest topological path in \S+ \(length=(\d+)\)", text).group(1))
    start = re.search(r"\(length=\d+\):\n\s+0: (\S+ ?\S*)", text)
    return {"levels": depth, "cells": cells, "flops": flops, "path_start": start.group(1) if start else "?"}


def main() -> int:
    cmd = _yosys_command(None if len(sys.argv) < 3 or sys.argv[1] != "--yosys" else sys.argv[2])
    with tempfile.TemporaryDirectory(dir=_ROOT / "build") as name:
        tmp = Path(name)
        if _run(cmd, _ELABORATE, tmp / "elab.log"):
            return 1
        if _run(cmd, _EQUIV, tmp / "equiv.log"):
            print("pipeline is NOT equivalent to mercury_fp32_add_rne")
            return 1
        print("formal equivalence vs mercury_fp32_add_rne (latency 4, seq 7): PASS")
        for flow in ("default", "single"):
            for top in ("mercury_fp32_add_rne", "mercury_fp32_add_pipe"):
                m = measure(cmd, top, tmp, flow)
                print(f"[{flow}] {top}: {m['levels']} levels, {m['cells']} cells ({m['flops']} flops), "
                      f"longest path starts at {m['path_start']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
