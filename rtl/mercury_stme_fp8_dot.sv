// Ordered scalar FP8 dot. Accept one descriptor before the products in each
// group. Flush a partial final group by padding it with zero products.
module mercury_stme_fp8_dot #(
    parameter integer GROUP_TERMS = 128
) (
    input wire clk,
    input wire rst,
    input wire group_valid,
    output wire group_ready,
    input wire group_first,
    input wire group_last,
    input wire [31:0] scale_product,
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
    reg descriptor_valid;
    reg saved_first, saved_last;
    reg [31:0] saved_scale;
    wire sum_in_ready, sum_valid, boundary_ready;
    wire [31:0] sum;

    assign group_ready = !rst && !descriptor_valid;
    assign in_ready = descriptor_valid && !sum_valid && sum_in_ready;

    mercury_stme_group_sum #(.GROUP_TERMS(GROUP_TERMS)) unscaled_group (
        .clk(clk), .rst(rst),
        .in_valid(in_valid && descriptor_valid && !sum_valid), .in_ready(sum_in_ready),
        .a(a), .b(b), .a_e5m2(a_e5m2), .b_e5m2(b_e5m2),
        .out_valid(sum_valid), .out_ready(boundary_ready), .out_sum(sum)
    );
    mercury_stme_boundary_accumulator scaled_dot (
        .clk(clk), .rst(rst), .in_valid(sum_valid), .in_ready(boundary_ready),
        .in_first(saved_first), .in_last(saved_last), .group_sum(sum),
        .scale_product(saved_scale), .out_valid(out_valid), .out_ready(out_ready),
        .out_acc(out_acc)
    );

    always @(posedge clk) begin
        if (rst) begin
            descriptor_valid <= 0;
            saved_first <= 0;
            saved_last <= 0;
            saved_scale <= 0;
        end else if (group_valid && group_ready) begin
            descriptor_valid <= 1;
            saved_first <= group_first;
            saved_last <= group_last;
            saved_scale <= scale_product;
        end else if (sum_valid && boundary_ready) begin
            descriptor_valid <= 0;
        end
    end
endmodule
