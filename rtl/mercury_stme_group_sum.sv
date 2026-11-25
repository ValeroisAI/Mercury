// One dot-product stream, with an ordered binary32 accumulation per group.
// Input beats are accepted in logical K order. Feed zero products for a tail.
// Scaling and accumulation across groups remain outside this block.
module mercury_stme_group_sum #(
    parameter integer GROUP_TERMS = 128,
    localparam integer COUNT_BITS = GROUP_TERMS > 1 ? $clog2(GROUP_TERMS) : 1
) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    output wire in_ready,
    input wire [7:0] a,
    input wire [7:0] b,
    input wire a_e5m2,
    input wire b_e5m2,
    output wire out_valid,
    input wire out_ready,
    output reg [31:0] out_sum
);
    // GROUP_TERMS is positive. The software contract uses 128.
    reg [COUNT_BITS-1:0] terms;
    reg [31:0] partial_sum;
    reg result_valid;
    wire [31:0] product, next_sum;

    mercury_fp8_product decode_and_multiply(a, b, a_e5m2, b_e5m2, product);
    mercury_fp32_add_rne accumulate(partial_sum, product, next_sum);

    assign out_valid = !rst && result_valid;
    assign in_ready = !rst && (!result_valid || out_ready);

    always @(posedge clk) begin
        if (rst) begin
            terms <= 0;
            partial_sum <= 0;
            result_valid <= 0;
            out_sum <= 0;
        end else begin
            if (result_valid && out_ready)
                result_valid <= 0;
            if (in_valid && in_ready) begin
                if (terms == COUNT_BITS'(GROUP_TERMS - 1)) begin
                    terms <= 0;
                    partial_sum <= 0;
                    result_valid <= 1;
                    out_sum <= next_sum;
                end else begin
                    terms <= terms + 1'b1;
                    partial_sum <= next_sum;
                end
            end
        end
    end
endmodule
