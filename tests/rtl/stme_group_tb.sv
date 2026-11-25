// Two 4-term groups exercise exact K order, input bubbles, backpressure,
// and simultaneous result retirement and next-group input acceptance.
module stme_group_tb(input wire clk, input wire rst);
    reg [3:0] phase;
    reg initialized = 0;
    wire in_valid = phase == 0 || phase >= 2 && phase <= 10;
    wire out_ready = phase == 7 || phase == 11;
    reg [7:0] a, b;
    reg a_e5m2, b_e5m2;
    wire in_ready, out_valid;
    wire [31:0] out_sum;

    mercury_stme_group_sum #(.GROUP_TERMS(4)) dut (
        clk, rst, in_valid, in_ready, a, b, a_e5m2, b_e5m2,
        out_valid, out_ready, out_sum
    );

    always @* begin
        // The second group is four times (2 × 1).
        a = 8'h40;
        b = 8'h38;
        a_e5m2 = 0;
        b_e5m2 = 0;
        case (phase)
            0, 3: begin
                a = phase == 0 ? 8'h7b : 8'hfb;
                b = 8'h7b;
                a_e5m2 = 1;
                b_e5m2 = 1;
            end
            2: a = 8'h38;
            4: a = 8'h44;
            default: begin end
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            phase <= 0;
            initialized <= 1;
        end else if (phase < 13)
            phase <= phase + 1'b1;
    end

    always @* begin
        if (rst) begin
            assert(!in_ready);
            assert(!out_valid);
        end
        if (initialized && !rst) begin
            assert(in_ready == (phase != 5 && phase != 6));
            assert(out_valid == (phase == 5 || phase == 6 || phase == 7 || phase == 11));
            // The large first term absorbs +1 before cancellation. Exact
            // reordering would yield 4, which is observably incorrect here.
            if (phase >= 5 && phase <= 7)
                assert(out_sum == 32'h40400000);  // 3.0
            if (phase == 11)
                assert(out_sum == 32'h41000000);  // 8.0
        end
    end
endmodule

// Check the default group boundary with a full 128 consecutive input beats.
module stme_group_128_tb(input wire clk, input wire rst);
    reg [7:0] phase;
    reg initialized = 0;
    wire in_ready, out_valid;
    wire [31:0] out_sum;
    wire in_valid = phase < 129 && phase != 23;
    wire out_ready = phase == 131;

    mercury_stme_group_sum dut (
        clk, rst, in_valid, in_ready, 8'h38, 8'h38, 1'b0, 1'b0,
        out_valid, out_ready, out_sum
    );

    always @(posedge clk) begin
        if (rst) begin
            phase <= 0;
            initialized <= 1;
        end else if (phase < 132)
            phase <= phase + 1'b1;
    end

    always @* begin
        if (initialized && !rst) begin
            assert(in_ready == (phase < 129 || phase >= 131));
            assert(out_valid == (phase >= 129 && phase <= 131));
            if (out_valid)
                assert(out_sum == 32'h43000000);  // 128.0
        end
    end
endmodule
