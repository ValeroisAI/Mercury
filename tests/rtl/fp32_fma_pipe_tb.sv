module fp32_fma_pipe_tb (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [31:0] c
);
    wire [31:0] reference_result, y;
    wire out_valid;
    reg [31:0] delayed_result [0:5];
    reg [5:0] delayed_valid;
    reg initialized = 0;
    integer i;

    mercury_fp32_fma_rne reference(a, b, c, reference_result);
    mercury_fp32_fma_pipe dut(clk, rst, in_valid, a, b, c, out_valid, y);

    always @(posedge clk) begin
        delayed_result[0] <= reference_result;
        for (i = 1; i < 6; i = i + 1)
            delayed_result[i] <= delayed_result[i-1];
        if (rst) begin
            initialized <= 1;
            delayed_valid <= 0;
        end else begin
            delayed_valid <= {delayed_valid[4:0], in_valid};
        end
    end

    always @* begin
        if (initialized) begin
            assert(out_valid == (!rst && delayed_valid[5]));
            if (out_valid)
                assert(y == delayed_result[5]);
        end
    end
endmodule
