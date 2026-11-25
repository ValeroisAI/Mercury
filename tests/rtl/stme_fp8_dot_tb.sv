module stme_fp8_dot_tb #(
    parameter integer TERMS = 4  // At least two terms in the fused-cancellation group.
) (
    input wire clk,
    input wire rst
);
    localparam integer SECOND = TERMS + 3;
    localparam integer THIRD = 2 * TERMS + 5;
    localparam integer DRAIN = 3 * TERMS + 8;
    reg [$clog2(DRAIN+3)-1:0] phase;
    reg initialized = 0;
    wire group_live, in_ready, out_valid;
    wire [31:0] out_acc;
    wire first_terms = phase >= 1 && phase <= TERMS+1;
    wire second_terms = phase > SECOND && phase <= SECOND+TERMS;
    wire third_terms = phase > THIRD && phase <= THIRD+TERMS;
    wire send_descriptor = phase == 0 || phase == SECOND || phase == THIRD;
    wire send_term = first_terms && phase != 2 || second_terms || third_terms || phase == THIRD+TERMS+1;
    wire special_product = phase == SECOND+2;
    wire [7:0] a = phase == 1 ? 8'hb8 : phase == SECOND+1 ? 8'h38 :
                   special_product ? 8'h01 : phase == THIRD+1 ? 8'h40 : 8'h00;
    wire [7:0] b = special_product ? 8'h20 : 8'h38;
    wire [31:0] scale = phase == 0 || phase == THIRD ? 32'h3f800000 :
                        phase == SECOND ? 32'h3f7ffffe : 32'h7fc00000;

    mercury_stme_fp8_dot #(.GROUP_TERMS(TERMS)) dut (
        .clk(clk), .rst(rst),
        .group_valid(send_descriptor), .group_ready(group_live), .group_first(phase != SECOND),
        .group_last(phase != 0), .scale_product(scale), .in_valid(send_term), .in_ready(in_ready),
        .a(a), .b(b), .a_e5m2(special_product), .b_e5m2(special_product),
        .out_valid(out_valid), .out_ready(phase == DRAIN || phase == DRAIN+1), .out_acc(out_acc)
    );

    always @(posedge clk) begin
        if (rst) begin
            phase <= 0;
            initialized <= 1;
        end else if (phase < DRAIN+2)
            phase <= phase + 1'b1;
    end

    always @* begin
        if (rst)
            assert(!group_live && !in_ready && !out_valid);
        if (initialized && !rst) begin
            assert(group_live == (send_descriptor || phase >= DRAIN+1));
            assert(in_ready == (first_terms || second_terms || third_terms));
            assert(out_valid == (phase >= THIRD && phase <= DRAIN+1));
            // The first dot computes fma(1+2^-23, 1-2^-23, -1) exactly.
            if (phase >= THIRD && phase <= DRAIN)
                assert(out_acc == 32'ha8800000);  // -2^-46
            if (phase == DRAIN+1)
                assert(out_acc == 32'h40000000);  // 2.0
        end
    end
endmodule
