// Detect rounding before the add, result corruption under backpressure,
// and failure to restart the accumulator when a new dot begins.
module stme_boundary_tb(input wire clk, input wire rst);
    reg [2:0] phase;
    reg initialized = 0;
    wire in_valid = phase <= 4;
    wire in_first = phase != 1;
    wire in_last = phase != 0;
    wire [31:0] group_sum = phase == 0 ? 32'hbf800000 :
                           phase == 1 ? 32'h3f800001 : 32'h40000000;
    wire [31:0] scale_product = phase == 0 ? 32'h3f800000 :
                               phase == 1 ? 32'h3f7ffffe : 32'h40400000;
    wire out_ready = phase == 4 || phase == 5;
    wire in_ready, out_valid;
    wire [31:0] out_acc;

    mercury_stme_boundary_accumulator dut(
        clk, rst, in_valid, in_ready, in_first, in_last, group_sum, scale_product,
        out_valid, out_ready, out_acc
    );
    always @(posedge clk) begin
        if (rst) begin
            phase <= 0;
            initialized <= 1;
        end else if (phase < 6)
            phase <= phase + 1'b1;
    end
    always @* begin
        if (initialized && !rst) begin
            assert(out_valid == (phase >= 2 && phase <= 5));
            assert(in_ready == (phase != 2 && phase != 3));
            if (phase >= 2 && phase <= 4)
                assert(out_acc == 32'ha8800000);  // -2**-46
            if (phase == 5)
                assert(out_acc == 32'h40c00000);  // 6.0
        end
    end
endmodule
