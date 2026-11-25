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

Generic cells and topological levels do not determine GHz, area, or power on a process node. The combinational FMA, scale-product generation, and connection to the tiled pipeline have no physical timing sign-off.
