// Two consecutive rounds of eight context-tagged inputs with independent
// two-term groups. Results are drained in a second, backpressured phase.
module stme_interleaved_tb(input wire clk, input wire rst);
    reg [4:0] phase;
    reg initialized = 0;
    wire in_valid = phase <= 16 || phase == 20 || phase == 21;
    wire [2:0] in_context = phase == 16 ? 3'd7 : phase >= 20 ? 3'd0 : phase[2:0];
    wire [2:0] read_context = phase >= 23 && phase <= 29 ? 3'(phase - 22) : 3'd0;
    wire out_ready = phase >= 22 && phase <= 29;
    wire in_ready, out_valid;
    wire [31:0] out_sum;
    reg [7:0] a;
    reg [31:0] expected_sum;

    mercury_stme_interleaved_sum #(.GROUP_TERMS(2)) dut (
        clk, rst, in_valid, in_ready, in_context, a, 8'h38, 1'b0, 1'b0,
        read_context, out_valid, out_ready, out_sum
    );

    always @* begin
        case (in_context)
            0: a = 8'h38;
            1: a = 8'h40;
            2: a = 8'h44;
            3: a = 8'h48;
            4: a = 8'h4a;
            5: a = 8'h4c;
            6: a = 8'h4e;
            7: a = 8'h50;
        endcase
        case (read_context)
            0: expected_sum = 32'h40000000;  // 2
            1: expected_sum = 32'h40800000;  // 4
            2: expected_sum = 32'h40c00000;  // 6
            3: expected_sum = 32'h41000000;  // 8
            4: expected_sum = 32'h41200000;  // 10
            5: expected_sum = 32'h41400000;  // 12
            6: expected_sum = 32'h41600000;  // 14
            7: expected_sum = 32'h41800000;  // 16
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            phase <= 0;
            initialized <= 1;
        end else if (phase < 30)
            phase <= phase + 1'b1;
    end

    always @* begin
        if (initialized && !rst) begin
            assert(out_valid == (phase >= 13 && phase <= 29));
            if (out_valid)
                assert(out_sum == expected_sum);
            if (phase < 16 || phase == 30)
                assert(in_ready);
            if (phase == 16 || phase == 20 || phase == 21)
                assert(!in_ready);
        end
    end
endmodule
