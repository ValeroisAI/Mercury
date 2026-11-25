"""Check scaled scalar FP8 dots through the complete group handshake."""

from __future__ import annotations

from check_sram_rtl import run_yosys_jobs

_CHECK = (
    "read_verilog -formal -sv rtl/mercury_fp8_product.sv rtl/mercury_fp32_add_rne.sv "
    "rtl/mercury_fp32_fma_rne.sv rtl/mercury_stme_group_sum.sv "
    "rtl/mercury_stme_boundary_accumulator.sv rtl/mercury_stme_fp8_dot.sv "
    "tests/rtl/stme_fp8_dot_tb.sv; "
    "prep -top stme_fp8_dot_tb -flatten; opt; chformal -lower; "
    "sat -verify -prove-asserts -seq 24 -set rst 0 -set-at 1 rst 1 -set-def-inputs"
)


if __name__ == "__main__":
    raise SystemExit(run_yosys_jobs(__doc__ or "", (_CHECK,)))
