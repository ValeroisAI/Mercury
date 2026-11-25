"""Check in-flight reset and back-to-back dots of mercury_stme_fp8_dot at GROUP_TERMS=1.

The harness tests/rtl/stme_fp8_dot_reset_tb.sv runs a fixed, handshake-driven
op list. It has no free data inputs, so the SAT run is a bounded check of one
complete scenario from reset (42 cycles), not a proof over all inputs.

Run:  python scripts/check_stme_fp8_dot_reset_rtl.py [--yosys PATH]
"""

from __future__ import annotations

from check_sram_rtl import run_yosys_jobs

_CHECK = (
    "read_verilog -formal -sv rtl/mercury_fp8_product.sv rtl/mercury_fp32_add_rne.sv "
    "rtl/mercury_fp32_fma_rne.sv rtl/mercury_stme_group_sum.sv "
    "rtl/mercury_stme_boundary_accumulator.sv rtl/mercury_stme_fp8_dot.sv "
    "tests/rtl/stme_fp8_dot_reset_tb.sv; "
    "prep -top stme_fp8_dot_reset_tb -flatten; memory_map; opt; chformal -lower; "
    "sat -verify -prove-asserts -seq 42 -set rst 0 -set-at 1 rst 1 -set-def-inputs"
)


if __name__ == "__main__":
    raise SystemExit(run_yosys_jobs(__doc__ or "", (_CHECK,)))
