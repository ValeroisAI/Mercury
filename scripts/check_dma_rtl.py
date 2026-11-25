"""Run a bounded formal check of the Mercury DMA token scoreboard."""

from __future__ import annotations

from check_sram_rtl import run_yosys_jobs

_CHECK = (
    "read_verilog -formal -sv rtl/mercury_dma_tokens.sv tests/rtl/dma_tokens_tb.sv; "
    "prep -top dma_tokens_tb -flatten; memory_map; opt; chformal -lower; "
    "sat -verify -prove-asserts -seq 8 -set-at 1 rst 1 -set-def-inputs"
)


if __name__ == "__main__":
    raise SystemExit(run_yosys_jobs(__doc__ or "", (_CHECK,)))
