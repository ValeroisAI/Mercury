// Reset and back-to-back checks for mercury_stme_fp8_dot at GROUP_TERMS = 1.
//
// The driver walks a fixed op list. Each descriptor or term waits for its
// ready signal, so timing comes from the DUT handshakes, not from assumed
// latencies. A synchronous reset is injected twice in flight:
//   1. after a descriptor is accepted but before its term arrives;
//   2. while a finished dot result (2.0) is held at out_valid with out_ready=0.
// The scoreboard then requires exactly four output beats, in order:
// 1.0, 2.0, -1.0 (three single-group dots issued back to back) and 1.5
// (a two-group dot: 1.0*1.0 + 2.0*0.25). The discarded 2.0 must never appear
// as the first beat.
module stme_fp8_dot_reset_tb (
    input wire clk,
    input wire rst
);
    localparam [7:0] E4M3_P1 = 8'h38, E4M3_P2 = 8'h40, E4M3_M1 = 8'hb8;
    localparam [31:0] F_ONE = 32'h3f800000, F_QUARTER = 32'h3e800000;
    localparam integer LAST_CYCLE = 40;

    // Op kinds.
    localparam [2:0] DESC = 3'd0, TERM = 3'd1, RESET = 3'd2, WAITOUT = 3'd3, DONE = 3'd4;

    reg [4:0] op;
    reg [5:0] cycle;
    reg [2:0] beats;
    reg initialized = 0;

    reg [2:0] kind;
    reg first, last_g;
    reg [31:0] scale;
    reg [7:0] term_a;
    always @* begin
        kind = DONE; first = 1; last_g = 1; scale = F_ONE; term_a = E4M3_P1;
        case (op)
            5'd0:  kind = DESC;                                    // in-flight descriptor ...
            5'd1:  kind = RESET;                                   // ... killed before its term
            5'd2:  kind = DESC;
            5'd3:  begin kind = TERM; term_a = E4M3_P2; end         // dot = 2.0
            5'd4:  kind = WAITOUT;                                 // held at out_valid, out_ready = 0
            5'd5:  kind = RESET;                                   // discard the held 2.0
            5'd6:  kind = DESC;
            5'd7:  begin kind = TERM; term_a = E4M3_P1; end         // beat 0: 1.0
            5'd8:  kind = DESC;
            5'd9:  begin kind = TERM; term_a = E4M3_P2; end         // beat 1: 2.0
            5'd10: kind = DESC;
            5'd11: begin kind = TERM; term_a = E4M3_M1; end         // beat 2: -1.0
            5'd12: begin kind = DESC; last_g = 0; end               // two-group dot, group 1
            5'd13: begin kind = TERM; term_a = E4M3_P1; end
            5'd14: begin kind = DESC; first = 0; scale = F_QUARTER; end  // group 2, scale 0.25
            5'd15: begin kind = TERM; term_a = E4M3_P2; end         // beat 3: 1.0 + 2.0*0.25 = 1.5
            default: kind = DONE;
        endcase
    end

    wire dut_rst = rst || (initialized && kind == RESET);
    wire group_ready, in_ready, out_valid;
    wire [31:0] out_acc;
    wire out_ready = op >= 5'd6;               // nothing drains before the second reset
    wire group_valid = kind == DESC;
    wire in_valid = kind == TERM;

    mercury_stme_fp8_dot #(.GROUP_TERMS(1)) dut (
        .clk(clk), .rst(dut_rst),
        .group_valid(group_valid), .group_ready(group_ready), .group_first(first),
        .group_last(last_g), .scale_product(scale),
        .in_valid(in_valid), .in_ready(in_ready), .a(term_a), .b(E4M3_P1),
        .a_e5m2(1'b0), .b_e5m2(1'b0),
        .out_valid(out_valid), .out_ready(out_ready), .out_acc(out_acc)
    );

    wire advance = (kind == DESC && group_ready) || (kind == TERM && in_ready) ||
                   kind == RESET || (kind == WAITOUT && out_valid);
    wire beat = out_valid && out_ready;

    always @(posedge clk) begin
        if (rst) begin
            initialized <= 1;
            op <= 0;
            cycle <= 0;
            beats <= 0;
        end else begin
            if (cycle != 6'h3f) cycle <= cycle + 1'b1;
            if (advance) op <= op + 1'b1;
            if (beat) beats <= beats + 1'b1;
        end
    end

    reg [31:0] expected;
    always @* begin
        case (beats)
            3'd0: expected = 32'h3f800000;   // 1.0
            3'd1: expected = 32'h40000000;   // 2.0
            3'd2: expected = 32'hbf800000;   // -1.0
            3'd3: expected = 32'h3fc00000;   // 1.5
            default: expected = 32'hxxxxxxxx;
        endcase
    end

    always @* begin
        if (initialized && !rst) begin
            // Reset gates every handshake output in the reset cycle itself.
            if (dut_rst) assert(!group_ready && !in_ready && !out_valid);
            // Ordered scoreboard: no extra beat, no stale result, exact values.
            if (beat) begin
                assert(beats < 3'd4);
                assert(out_acc == expected);
            end
            // The first reset leaves no result behind.
            if (op <= 5'd4) assert(!out_valid || op == 5'd4);
            // Progress: within the bound, all four beats arrive and the op list completes.
            if (cycle == LAST_CYCLE[5:0]) begin
                assert(beats == 3'd4);
                assert(kind == DONE);
            end
        end
    end
endmodule
