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

The [separate-scale wrapper](../../rtl/mercury_stme_scaled_fp8_dot.sv) takes
both binary32 scales and computes their rounded product at descriptor
acceptance. It uses the same exact FMA with a matching-sign zero addend.
The product rounds to binary32 before the accumulator's fused operation.

```sh
python scripts/check_stme_fp8_dot_rtl.py
python scripts/check_stme_fp8_dot_reset_rtl.py
```

The [24-step Yosys harness](../../tests/rtl/stme_fp8_dot_tb.sv) runs on
both interfaces. Four-term groups exercise mixed FP8 operand formats, an
input bubble, descriptor latching, and a stalled output while another
dot collects its products. The first dot evaluates
`fma(1 + 2^-23, 1 - 2^-23, -1) = -2^-46` exactly. The next dot returns
`2.0` after the first result is consumed. The group sum has a separate
134-step check at the full 128-term configuration.

The [42-step reset harness](../../tests/rtl/stme_fp8_dot_reset_tb.sv)
checks single-term groups through resets with an in-flight descriptor
and a held result. It then checks successive outputs and a two-group
dot without resuming state from before the reset.

The scalar integration uses the combinational adder. The separate
[interleaved prototype](./stme-interleaved-rtl.md) tests the pipelined
alternative. The scaled blocks use combinational scale multiplication
and FMA. Pipelining, banked operands, and the complete tiled STME remain
integration work.
