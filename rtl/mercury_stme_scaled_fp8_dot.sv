// Same scalar interface as mercury_stme_fp8_dot, with separate binary32
// operand scales. The scale product rounds before the boundary FMA.
module mercury_stme_scaled_fp8_dot #(
    parameter integer GROUP_TERMS = 128
) (
    input wire clk,
    input wire rst,
    input wire group_valid,
    output wire group_ready,
    input wire group_first,
    input wire group_last,
    input wire [31:0] scale_a,
    input wire [31:0] scale_b,
    input wire in_valid,
    output wire in_ready,
    input wire [7:0] a,
    input wire [7:0] b,
    input wire a_e5m2,
    input wire b_e5m2,
    output wire out_valid,
    input wire out_ready,
    output wire [31:0] out_acc
);
    wire [31:0] signed_zero = {scale_a[31] ^ scale_b[31], 31'b0};
    wire [31:0] scale_product;

    // A fused product with a matching-sign zero preserves multiplication's
    // signed zero result and keeps the same single-rounding arithmetic.
    mercury_fp32_fma_rne scale_multiply(scale_a, scale_b, signed_zero, scale_product);

    mercury_stme_fp8_dot #(.GROUP_TERMS(GROUP_TERMS)) dot (
        .clk(clk), .rst(rst), .group_valid(group_valid), .group_ready(group_ready),
        .group_first(group_first), .group_last(group_last), .scale_product(scale_product),
        .in_valid(in_valid), .in_ready(in_ready), .a(a), .b(b), .a_e5m2(a_e5m2), .b_e5m2(b_e5m2),
        .out_valid(out_valid), .out_ready(out_ready), .out_acc(out_acc)
    );
endmodule
