// One stream of group-boundary accumulation. The caller supplies
// scale_product = RN32(scale_A * scale_B) for the corresponding group.
module mercury_stme_boundary_accumulator (
    input wire clk,
    input wire rst,
    input wire in_valid,
    output wire in_ready,
    input wire in_first,
    input wire in_last,
    input wire [31:0] group_sum,
    input wire [31:0] scale_product,
    output wire out_valid,
    input wire out_ready,
    output reg [31:0] out_acc
);
    reg [31:0] accumulated;
    reg result_valid;
    wire [31:0] next_acc;

    // The first-group marker restarts the dot at +0.
    mercury_fp32_fma_rne boundary_fma(
        group_sum, scale_product, in_first ? 32'b0 : accumulated, next_acc
    );
    assign in_ready = !rst && (!result_valid || out_ready);
    assign out_valid = !rst && result_valid;

    always @(posedge clk) begin
        if (rst) begin
            accumulated <= 0;
            out_acc <= 0;
            result_valid <= 0;
        end else begin
            if (result_valid && out_ready)
                result_valid <= 0;
            if (in_valid && in_ready) begin
                accumulated <= next_acc;
                if (in_last) begin
                    out_acc <= next_acc;
                    result_valid <= 1;
                end
            end
        end
    end
endmodule
