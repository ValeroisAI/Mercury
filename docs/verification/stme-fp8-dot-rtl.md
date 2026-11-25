# Scaled FP8 scalar dot

The [scalar dot prototype](../../rtl/mercury_stme_fp8_dot.sv) connects the
[ordered FP8 group sum](./stme-group-rtl.md) to the
[FP32 fused boundary accumulator](./stme-boundary-rtl.md). Submit one
descriptor per group with the dot's first/last flags and the rounded FP32
scale product. Then submit `GROUP_TERMS` operand pairs in K order. The default
is 128 pairs. Pad a short final group with zero products.

The descriptor remains live until the fused boundary accumulator takes the
sum. It stays associated with that group even if the previous dot's output
stalls. The next dot can collect one group during that stall. Reset clears
the descriptor, unscaled partial sum, scaled accumulator, and pending output.

```sh
python scripts/check_stme_fp8_dot_rtl.py
```

The [24-step Yosys harness](../../tests/rtl/stme_fp8_dot_tb.sv) uses
four-term groups to check mixed FP8 operand formats, an input bubble,
descriptor latching, and a stalled output while another dot collects its
products. The first dot evaluates
`fma(1 + 2^-23, 1 - 2^-23, -1) = -2^-46` exactly. The next dot returns
`2.0` after the first result is consumed. The group sum has a separate
134-step check at the full 128-term configuration.

The scalar integration uses the combinational adder. The separate
[interleaved prototype](./stme-interleaved-rtl.md) tests the pipelined
alternative. Scale-product generation, FMA pipelining, banked operands,
and the complete tiled STME remain integration work.
