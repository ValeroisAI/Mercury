// Combinational IEEE-754 binary32 adder, round-to-nearest-even (RNE).
//
// Behaviour
//   * Gradual underflow: subnormal inputs and outputs (no flush-to-zero).
//   * Overflow rounds to +/-infinity (the correct RNE result).
//   * Signed zeros: an exact zero sum is +0, except (-0) + (-0) = -0.
//   * Any NaN input, and inf + (-inf), returns the canonical qNaN 0x7fc00000.
//   * No exception flags.
//
// Structure (one combinational path, no pipeline registers):
//   unpack -> magnitude compare/swap -> align (right shift + sticky) ->
//   27-bit add/subtract -> carry or leading-zero normalise (exponent-bounded
//   for subnormals) -> RNE increment -> pack and special-case select.
// The three extra bits (guard, round, sticky) are enough for exact RNE: a
// left shift of more than one place happens only when the exponent
// difference is <= 1, and then no bits were shifted out.
//
// This is RTL only. Nothing here claims timing, area or power on any process
// node; the critical path is the align shifter, then the adder, the LZC and
// the normalise shifter, then the rounding incrementer, all in series.
module mercury_fp32_add_rne (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] y
);
    localparam [31:0] QNAN = 32'h7fc00000;

    wire        sa = a[31], sb = b[31];
    wire [7:0]  ea = a[30:23], eb = b[30:23];
    wire [22:0] fa = a[22:0], fb = b[22:0];

    wire a_nan = (&ea) && (|fa);
    wire b_nan = (&eb) && (|fb);
    wire a_inf = (&ea) && !(|fa);
    wire b_inf = (&eb) && !(|fb);

    // Order by magnitude: |l| >= |s|.
    wire        swap = {ea, fa} < {eb, fb};
    wire        sl   = swap ? sb : sa;
    wire        ss   = swap ? sa : sb;
    wire [7:0]  el   = swap ? eb : ea;
    wire [7:0]  es   = swap ? ea : eb;
    wire [22:0] fl   = swap ? fb : fa;
    wire [22:0] fs   = swap ? fa : fb;

    // Subnormals use exponent 1 and a zero hidden bit.
    wire [7:0]  el_eff = (el == 8'd0) ? 8'd1 : el;
    wire [7:0]  es_eff = (es == 8'd0) ? 8'd1 : es;
    wire [23:0] ml = {(el != 8'd0), fl};
    wire [23:0] ms = {(es != 8'd0), fs};
    wire [7:0]  diff = el_eff - es_eff;  // >= 0 because |l| >= |s|
    wire        subtract = sl ^ ss;

    reg  [26:0] big, small_ext, aligned;
    reg         sticky;
    reg  [27:0] sum;
    reg  [4:0]  lz;
    reg  [9:0]  e_norm, e_round;
    reg  [4:0]  shl;
    reg  [26:0] norm;
    reg  [24:0] rounded;
    reg  [23:0] mant;
    reg         round_up;
    reg  [31:0] result;
    integer     i;

    always @* begin
        big = {ml, 3'b000};
        small_ext = {ms, 3'b000};
        if (diff >= 8'd27) begin
            aligned = 27'd0;
            sticky  = |small_ext;
        end else begin
            aligned = small_ext >> diff;
            sticky  = |(small_ext & ((27'd1 << diff) - 27'd1));
        end
        aligned[0] = aligned[0] | sticky;

        sum = subtract ? ({1'b0, big} - {1'b0, aligned}) : ({1'b0, big} + {1'b0, aligned});

        // Leading zeros of sum[26:0] (27 when the sum is zero).
        lz = 5'd27;
        for (i = 0; i <= 26; i = i + 1)
            if (sum[i]) lz = 5'(26 - i);

        if (sum[27]) begin
            // Carry out: shift right one place, keeping the dropped bit in sticky.
            norm   = {sum[27:2], sum[1] | sum[0]};
            e_norm = {2'b00, el_eff} + 10'd1;
        end else begin
            // Shift left, but never below exponent 1 (the result becomes subnormal).
            shl    = ({5'd0, lz} < ({2'b00, el_eff} - 10'd1)) ? lz : 5'(el_eff - 8'd1);
            norm   = sum[26:0] << shl;
            e_norm = {2'b00, el_eff} - {5'd0, shl};
        end

        mant     = norm[26:3];
        round_up = norm[2] && (norm[1] || norm[0] || mant[0]);
        rounded  = {1'b0, mant} + {24'd0, round_up};
        if (rounded[24]) begin
            mant    = rounded[24:1];
            e_round = e_norm + 10'd1;
        end else begin
            mant    = rounded[23:0];
            e_round = e_norm;
        end

        if (a_nan || b_nan || (a_inf && b_inf && (sa ^ sb)))
            result = QNAN;
        else if (a_inf || b_inf)
            result = {a_inf ? sa : sb, 8'hff, 23'd0};
        else if (sum == 28'd0)
            result = {sa & sb, 31'd0};
        else if (e_round >= 10'd255)
            result = {sl, 8'hff, 23'd0};
        else if (!mant[23])
            result = {sl, 8'd0, mant[22:0]};      // subnormal (e_round is 1 here)
        else
            result = {sl, e_round[7:0], mant[22:0]};
    end

    assign y = result;
endmodule
