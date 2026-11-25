# Fused FP32 scale and accumulation

The [boundary accumulator](../../rtl/mercury_stme_boundary_accumulator.sv) implements one group-boundary operation from the Mercury `K=128` block-FP8 contract. It computes `acc = RN32(group_sum × scale_product + acc)`. The upstream caller supplies `scale_product = RN32(scale_A × scale_B)`. The first marker resets the accumulator to +0, and the last marker publishes the result. The output is held while backpressured.

```sh
python scripts/check_fma_rtl.py --random 16384
```

Use `--yosys` to specify the executable if needed. The [local setup](./shared-sram-rtl.md) covers Windows.

The [binary32 FMA](../../rtl/mercury_fp32_fma_rne.sv) retains the exact significand product before aligning it with the accumulator. It then performs a single round-to-nearest-even operation. Nearby exponents preserve cancellation bits. For larger gaps, a sticky bit preserves the direction of a discarded nonzero tail. Directed and seeded inputs check cancellation, ties, subnormal results, signed zero, overflow and canonical NaN. **30,632 vectors** matched the exact Python reference.

The [boundary harness](../../tests/rtl/stme_boundary_tb.sv) checks a two-group case that returns `-2^-46`. Rounding the product before adding would yield zero. The same harness checks output stability under stalls and the start of a new dot.

## Generic mapping comparison

Yosys 0.71 `synth -noabc; ltp -noff; stat` reported the following for the standalone FMA:

| Implementation | Generic cells | Longest topological path |
| --- | ---: | ---: |
| Sequential-priority leading-bit encoders, wide exponent expressions | 9,832 | 281 levels |
| Balanced encoders, wide exponent expressions | 9,214 | 261 levels |
| Balanced encoders, 10-bit signed exponent arithmetic | 8,336 | 233 levels |

The baseline critical path passed through the product and result priority chains. The balanced searches replace those chains with hierarchical selection. Narrow exponent arithmetic reduces alignment and rounding-control logic. The width change passed Yosys equivalence against the preceding wide implementation.

## Six-stage FMA prototype

The [registered FMA](../../rtl/mercury_fp32_fma_pipe.sv) accepts one triple
each cycle. An input sampled at edge N is visible after edge N+5. Reset
clears all valid stages. The consumer needs an available output slot each
cycle.

```sh
python scripts/check_fp32_fma_pipe_rtl.py --random 16384
```

The checker separately evaluates the arithmetic stages with the registers
replaced by wires and checks the real registered sequence. The arithmetic
matched the exact Python oracle on **30,632 vectors**. The 29-step directed
sequence checks consecutive triples, an input bubble, reset flushing, and
output parity against delayed scalar FMA results.

With the same Yosys command as above, the longest path fell from **233 to
64 levels** on the first six-stage cut. Moving rounding-bit selection
into the final stage reduced it to **55 levels**, with the same external
latency. The final pipeline mapped to 9,003 generic cells including 695
flip-flops. The longest stage normalizes the product and addend and selects
the larger magnitude. [FPGA primitive mapping](./fpga-mapping.md) counts
1,727 LUT primitives, 528 flip-flops and two DSP blocks for this top.

The scalar dot uses the combinational FMA. Routing the registered FMA
through the tiled pipeline, buffering its outputs, and measuring physical
delay remain integration gates.
