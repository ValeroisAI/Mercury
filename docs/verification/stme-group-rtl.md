# Ordered FP8 group sum

The [group-sum prototype](../../rtl/mercury_stme_group_sum.sv) accepts FP8 operand pairs in logical K order and accumulates them into one binary32 partial. The default group has 128 accepted terms. Operand formats are supplied independently. An incomplete group can be padded with zero products.

Run the product and group checks from the repository root:

```sh
python scripts/check_stme_rtl.py
python scripts/check_stme_rtl.py --exhaustive --full-group
python scripts/check_fp32_rtl.py
```

If Yosys is outside your PATH, pass its path with `--yosys`. See the [local Windows setup](./shared-sram-rtl.md).

Each finite, nonzero OCP E4M3FN or E5M2 product fits exactly in binary32. The [multiplier](../../rtl/mercury_fp8_product.sv) handles signed zero and E5M2 infinities, and produces canonical NaN for invalid products and NaN operands. The [binary32 adder](../../rtl/mercury_fp32_add_rne.sv) rounds each addition to nearest, ties to even. It supports gradual underflow and emits no exception flags. [OFP8 specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)

The product check evaluated all **262,144** combinations of two operand bytes and their two format selectors against the Python reference. The sequential harness checks cancellation, input bubbles, output backpressure, and the default 128-term boundary. For example, `[large, 1, -large, 3]` produces 3 under ordered binary32 rounding because the second term rounds away. Summing the same terms under exact arithmetic gives 4.

This is a reference datapath for one partial. It runs decode, multiply, and add on one combinational path. [Interleaved accumulation](./stme-interleaved-rtl.md) and [boundary scale FMA](./stme-boundary-rtl.md) have separate prototypes. A systolic array, instruction integration, and physical timing still require implementation and verification.
