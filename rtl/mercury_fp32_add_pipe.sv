// Pipelined IEEE-754 binary32 RNE adder: fixed latency 4 cycles, II = 1.
//
// Timing contract
//   * An operand pair presented with in_valid=1 on clock edge N comes out on
//     y with out_valid=1 right after edge N+3 (four register stages: S1..S4).
//     The result is visible in the cycle after edge N+3, i.e. LATENCY = 4
//     cycles counted as "accept cycle + 3 more".
//   * A new pair can be accepted every cycle. There is no stall or
//     backpressure input; the consumer must accept every out_valid beat.
//   * Synchronous active-high rst clears every valid bit. Data registers are
//     not reset, and y is only meaningful when out_valid=1.
//
// Numerics are bit-identical to mercury_fp32_add_rne (same RNE, gradual
// underflow, overflow to infinity, signed zeros, canonical qNaN 0x7fc00000,
// no flags). This is checked by tests/rtl/fp32_add_pipe_tb.sv and
// scripts/check_fp32_pipe_rtl.py.
//
// Stage cut
//   S1  unpack, special-case decode, magnitude compare and swap, exponent diff
//   S2  alignment shift with sticky, 28-bit add or subtract
//   S3  leading-zero count and exponent-bounded normalise shift
//   S4  RNE increment, overflow/subnormal select, pack
// This file is RTL only and makes no timing, area or power claim for any
// process node.
module mercury_fp32_add_pipe (
    input  wire        clk,
    input  wire        rst,
    input  wire        in_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire        out_valid,
    output wire [31:0] y
);
    localparam integer LATENCY = 4;
    localparam [31:0] QNAN = 32'h7fc00000;

    // ---------------- S1: decode, specials, swap ----------------
    wire        sa = a[31], sb = b[31];
    wire [7:0]  ea = a[30:23], eb = b[30:23];
    wire [22:0] fa = a[22:0], fb = b[22:0];
    wire a_nan = (&ea) && (|fa);
    wire b_nan = (&eb) && (|fb);
    wire a_inf = (&ea) && !(|fa);
    wire b_inf = (&eb) && !(|fb);
    wire        swap = {ea, fa} < {eb, fb};
    wire        sl_c = swap ? sb : sa;
    wire        ss_c = swap ? sa : sb;
    wire [7:0]  el_c = swap ? eb : ea;
    wire [7:0]  es_c = swap ? ea : eb;
    wire [22:0] fl_c = swap ? fb : fa;
    wire [22:0] fs_c = swap ? fa : fb;
    wire [7:0]  el_eff_c = (el_c == 8'd0) ? 8'd1 : el_c;
    wire [7:0]  es_eff_c = (es_c == 8'd0) ? 8'd1 : es_c;
    wire        special_c = a_nan || b_nan || a_inf || b_inf;
    wire [31:0] special_val_c = (a_nan || b_nan || (a_inf && b_inf && (sa ^ sb))) ? QNAN
                              : {a_inf ? sa : sb, 8'hff, 23'd0};

    reg         v1, v2, v3, v4;
    reg         s1_sl, s1_sub, s1_special, s1_zero_sign;
    reg  [31:0] s1_special_val;
    reg  [7:0]  s1_el_eff, s1_diff;
    reg  [23:0] s1_ml, s1_ms;

    always @(posedge clk) begin
        s1_sl          <= sl_c;
        s1_sub         <= sl_c ^ ss_c;
        s1_special     <= special_c;
        s1_special_val <= special_val_c;
        s1_zero_sign   <= sa & sb;
        s1_el_eff      <= el_eff_c;
        s1_diff        <= el_eff_c - es_eff_c;
        s1_ml          <= {(el_c != 8'd0), fl_c};
        s1_ms          <= {(es_c != 8'd0), fs_c};
    end

    // ---------------- S2: align + add/subtract ----------------
    // Kogge-Stone parallel-prefix adder (log2 carry depth). Generic `+`/`-`
    // map to a ripple chain in Yosys, which was the longest path between flops.
    function [27:0] ks_add28(input [27:0] x, input [27:0] y, input cin);
        reg [27:0] g, p, gn, pn, p0;
        integer i, d;
        begin
            p0 = x ^ y;
            g = x & y;
            p = p0;
            g[0] = g[0] | (p0[0] & cin);
            for (d = 1; d < 28; d = d * 2) begin
                gn = g; pn = p;
                for (i = d; i < 28; i = i + 1) begin
                    gn[i] = g[i] | (p[i] & g[i-d]);
                    pn[i] = p[i] & p[i-d];
                end
                g = gn; p = pn;
            end
            // g[i] is now the carry out of bit i (including cin).
            ks_add28 = p0 ^ {g[26:0], cin};
        end
    endfunction

    reg  [26:0] big, small_ext, aligned;
    reg         sticky;
    reg  [27:0] sum_c;
    always @* begin
        big = {s1_ml, 3'b000};
        small_ext = {s1_ms, 3'b000};
        if (s1_diff >= 8'd27) begin
            aligned = 27'd0;
            sticky  = |small_ext;
        end else begin
            aligned = small_ext >> s1_diff;
            // Bits shifted out: low `s1_diff` bits. ~(all-ones << d) avoids a ripple subtract.
            sticky  = |(small_ext & ~({27{1'b1}} << s1_diff));
        end
        aligned[0] = aligned[0] | sticky;
        // big - aligned = big + ~aligned + 1 (28-bit, no borrow because |big| >= |aligned|).
        sum_c = ks_add28({1'b0, big}, s1_sub ? ~{1'b0, aligned} : {1'b0, aligned}, s1_sub);
    end

    reg         s2_sl, s2_special, s2_zero_sign;
    reg  [31:0] s2_special_val;
    reg  [7:0]  s2_el_eff, s2_el_m1;
    reg  [27:0] s2_sum;
    always @(posedge clk) begin
        s2_sl <= s1_sl; s2_special <= s1_special; s2_special_val <= s1_special_val;
        s2_zero_sign <= s1_zero_sign; s2_el_eff <= s1_el_eff; s2_sum <= sum_c;
        s2_el_m1 <= s1_el_eff - 8'd1;   // precomputed: the S3 subnormal shift bound
    end

    // ---------------- S3: LZC + normalise ----------------
    // Log-depth leading-zero count. A count equal to the width means all zeros.
    function [1:0] lz2(input [1:0] x);
        lz2 = {~x[1] & ~x[0], ~x[1] & x[0]};
    endfunction
    function [2:0] lz4(input [3:0] x);
        reg [1:0] h, l;
        begin h = lz2(x[3:2]); l = lz2(x[1:0]); lz4 = h[1] ? 3'd2 + {1'b0, l} : {1'b0, h}; end
    endfunction
    function [3:0] lz8(input [7:0] x);
        reg [2:0] h, l;
        begin h = lz4(x[7:4]); l = lz4(x[3:0]); lz8 = h[2] ? 4'd4 + {1'b0, l} : {1'b0, h}; end
    endfunction
    function [4:0] lz16(input [15:0] x);
        reg [3:0] h, l;
        begin h = lz8(x[15:8]); l = lz8(x[7:0]); lz16 = h[3] ? 5'd8 + {1'b0, l} : {1'b0, h}; end
    endfunction
    function [5:0] lz32(input [31:0] x);
        reg [4:0] h, l;
        begin h = lz16(x[31:16]); l = lz16(x[15:0]); lz32 = h[4] ? 6'd16 + {1'b0, l} : {1'b0, h}; end
    endfunction

    reg  [5:0]  lz6;
    reg  [4:0]  lz, shl;
    reg  [26:0] norm_c;
    reg  [9:0]  e_norm_c;
    always @* begin
        // Pad with ones so an all-zero sum[26:0] counts 27, matching the reference.
        lz6 = lz32({s2_sum[26:0], 5'b11111});
        lz  = lz6[4:0];
        shl = 5'd0;
        if (s2_sum[27]) begin
            norm_c   = {s2_sum[27:2], s2_sum[1] | s2_sum[0]};
            e_norm_c = {2'b00, s2_el_eff} + 10'd1;
        end else begin
            shl      = ({3'd0, lz} < s2_el_m1) ? lz : 5'(s2_el_m1);
            norm_c   = s2_sum[26:0] << shl;
            e_norm_c = {2'b00, s2_el_eff} - {5'd0, shl};
        end
    end

    reg         s3_sl, s3_special, s3_zero_sign, s3_is_zero;
    reg  [31:0] s3_special_val;
    reg  [26:0] s3_norm;
    reg  [9:0]  s3_e_norm;
    always @(posedge clk) begin
        s3_sl <= s2_sl; s3_special <= s2_special; s3_special_val <= s2_special_val;
        s3_zero_sign <= s2_zero_sign; s3_is_zero <= (s2_sum == 28'd0);
        s3_norm <= norm_c; s3_e_norm <= e_norm_c;
    end

    // ---------------- S4: round + pack ----------------
    reg  [23:0] mant;
    reg         round_up;
    reg  [24:0] rounded;
    reg  [9:0]  e_round;
    reg  [31:0] result_c;
    always @* begin
        mant     = s3_norm[26:3];
        round_up = s3_norm[2] && (s3_norm[1] || s3_norm[0] || mant[0]);
        rounded  = {1'b0, mant} + {24'd0, round_up};
        if (rounded[24]) begin
            mant    = rounded[24:1];
            e_round = s3_e_norm + 10'd1;
        end else begin
            mant    = rounded[23:0];
            e_round = s3_e_norm;
        end
        if (s3_special)
            result_c = s3_special_val;
        else if (s3_is_zero)
            result_c = {s3_zero_sign, 31'd0};
        else if (e_round >= 10'd255)
            result_c = {s3_sl, 8'hff, 23'd0};
        else if (!mant[23])
            result_c = {s3_sl, 8'd0, mant[22:0]};
        else
            result_c = {s3_sl, e_round[7:0], mant[22:0]};
    end

    reg [31:0] s4_y;
    always @(posedge clk) s4_y <= result_c;

    always @(posedge clk) begin
        if (rst) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
        end else begin
            v1 <= in_valid; v2 <= v1; v3 <= v2; v4 <= v3;
        end
    end

    assign out_valid = v4;
    assign y = s4_y;
endmodule
