// Formal equivalence: mercury_fp32_add_pipe (latency 4, II = 1) against the
// combinational reference mercury_fp32_add_rne applied to the same operands
// delayed by four cycles. A bounded proof from reset covers every behaviour:
// the pipeline holds no state beyond the in-flight operands and valid bits.
module fp32_add_pipe_tb (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [31:0] a,
    input wire [31:0] b
);
    wire        out_valid;
    wire [31:0] y, y_ref;

    mercury_fp32_add_pipe dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a), .b(b), .out_valid(out_valid), .y(y)
    );

    // Independent 4-deep delay line for the reference.
    reg [31:0] a_d [0:3];
    reg [31:0] b_d [0:3];
    reg [3:0]  v_d;
    reg        initialized = 0;
    integer    k;
    always @(posedge clk) begin
        a_d[0] <= a; b_d[0] <= b;
        for (k = 1; k < 4; k = k + 1) begin
            a_d[k] <= a_d[k-1];
            b_d[k] <= b_d[k-1];
        end
        if (rst) begin
            v_d <= 4'b0;
            initialized <= 1;
        end else
            v_d <= {v_d[2:0], in_valid};
    end

    mercury_fp32_add_rne ref_add (.a(a_d[3]), .b(b_d[3]), .y(y_ref));

    always @* begin
        if (initialized) begin
            // Fixed latency 4, II = 1; v_d is cleared by the same synchronous reset,
            // so this also proves that reset clears every in-flight valid bit.
            assert(out_valid == v_d[3]);
            if (out_valid) assert(y == y_ref);         // bit-identical numerics
        end
    end
endmodule
