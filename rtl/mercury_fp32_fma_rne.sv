// Binary32 fused multiply-add with one round-to-nearest-even rounding.
// Combinational numerical prototype. No exception flags or timing target.
module mercury_fp32_fma_rne (
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [31:0] c,
    output reg [31:0] y
);
    localparam [31:0] QNAN = 32'h7fc00000;
    wire [7:0] ea = a[30:23], eb = b[30:23], ec = c[30:23];
    wire [23:0] ma = {ea != 0, a[22:0]};
    wire [23:0] mb = {eb != 0, b[22:0]};
    wire [23:0] mc = {ec != 0, c[22:0]};
    wire a_inf = ea == 255 && a[22:0] == 0;
    wire b_inf = eb == 255 && b[22:0] == 0;
    wire c_inf = ec == 255 && c[22:0] == 0;
    wire any_nan = (ea == 255 && !a_inf) || (eb == 255 && !b_inf) || (ec == 255 && !c_inf);
    wire product_sign = a[31] ^ b[31];
    wire zero_product = ma == 0 || mb == 0;
    wire [47:0] product = ma * mb;

    function automatic [5:0] highest_index(input [63:0] value);
        reg [31:0] half;
        reg [15:0] quarter;
        reg [7:0] eighth;
        reg [3:0] nibble;
        reg [1:0] pair;
        begin
            // Search only the highest nonzero half at each level. For
            // an all-zero input this returns zero, as the caller expects.
            highest_index[5] = |value[63:32];
            half = highest_index[5] ? value[63:32] : value[31:0];
            highest_index[4] = |half[31:16];
            quarter = highest_index[4] ? half[31:16] : half[15:0];
            highest_index[3] = |quarter[15:8];
            eighth = highest_index[3] ? quarter[15:8] : quarter[7:0];
            highest_index[2] = |eighth[7:4];
            nibble = highest_index[2] ? eighth[7:4] : eighth[3:0];
            highest_index[1] = |nibble[3:2];
            pair = highest_index[1] ? nibble[3:2] : nibble[1:0];
            highest_index[0] = pair[1];
        end
    endfunction

    function automatic [54:0] shift_jam(input [54:0] value, input signed [9:0] distance);
        reg [54:0] aligned;
        begin
            if (distance <= 0)
                aligned = value;
            else if (distance >= 55)
                aligned = {54'b0, |value};
            else begin
                aligned = value >> distance[5:0];
                aligned[0] = aligned[0] | (|(value << (6'd55 - distance[5:0])));
            end
            shift_jam = aligned;
        end
    endfunction

    reg [5:0] p_msb, c_msb, sum_msb;
    // Exact finite product exponents lie in [-298, 255]. The ten-bit
    // signed domain also holds the largest alignment gap and shifts.
    reg signed [9:0] p_exp, c_exp, large_exp, small_exp, target_exp, leading_exp, round_shift;
    reg [54:0] p_norm, c_norm, large, small, aligned_small;
    reg [55:0] magnitude, shifted;
    reg [24:0] rounded;
    reg large_sign, small_sign, guard_bit, sticky_bit;

    always @* begin
        p_msb = highest_index({16'b0, product});
        c_msb = highest_index({40'b0, mc});

        p_exp = $signed({2'b0, ea == 0 ? 8'd1 : ea}) +
                $signed({2'b0, eb == 0 ? 8'd1 : eb}) - 10'sd300 + $signed({4'b0, p_msb});
        c_exp = $signed({2'b0, ec == 0 ? 8'd1 : ec}) - 10'sd150 + $signed({4'b0, c_msb});
        p_norm = {7'b0, product} << (6'd54 - p_msb);
        c_norm = {31'b0, mc} << (6'd54 - c_msb);
        if (mc == 0 || p_exp > c_exp || (p_exp == c_exp && p_norm >= c_norm)) begin
            large = p_norm;
            small = c_norm;
            large_exp = p_exp;
            small_exp = c_exp;
            large_sign = product_sign;
            small_sign = c[31];
        end else begin
            large = c_norm;
            small = p_norm;
            large_exp = c_exp;
            small_exp = p_exp;
            large_sign = c[31];
            small_sign = product_sign;
        end

        // Seven low bits below even a 48-bit normalized product. Severe
        // cancellation requires nearby exponents, where nothing is lost.
        // In the far path a discarded nonzero tail jams into an odd LSB,
        // preserving which side of an even rounding boundary it occupies.
        aligned_small = shift_jam(small, large_exp - small_exp);
        if (large_sign == small_sign)
            magnitude = {1'b0, large} + {1'b0, aligned_small};
        else
            magnitude = {1'b0, large} - {1'b0, aligned_small};
        sum_msb = highest_index({8'b0, magnitude});

        leading_exp = large_exp + $signed({4'b0, sum_msb}) - 10'sd54;
        target_exp = leading_exp < -10'sd126 ? -10'sd126 : leading_exp;
        round_shift = target_exp - large_exp + 10'sd31;
        shifted = 0;
        guard_bit = 0;
        sticky_bit = 0;
        if (round_shift <= 0)
            shifted = magnitude << (6'd0 - round_shift[5:0]);
        else if (round_shift <= 56) begin
            shifted = magnitude >> round_shift[5:0];
            guard_bit = magnitude[round_shift[5:0] - 6'd1];
            sticky_bit = round_shift > 1 && (|(magnitude << (6'd57 - round_shift[5:0])));
        end
        rounded = shifted[24:0] + 25'(guard_bit && (sticky_bit || shifted[0]));

        if (any_nan || ((a_inf || b_inf) && zero_product) ||
            ((a_inf || b_inf) && c_inf && product_sign != c[31]))
            y = QNAN;
        else if (a_inf || b_inf)
            y = {product_sign, 8'hff, 23'b0};
        else if (c_inf)
            y = c;
        else if (zero_product)
            y = mc != 0 ? c : {product_sign & c[31], 31'b0};
        else if (magnitude == 0)
            y = 0;
        else if (target_exp > 127 || (target_exp == 127 && rounded[24]))
            y = {large_sign, 8'hff, 23'b0};
        else if (rounded[24])
            y = {large_sign, 8'(target_exp + 128), rounded[23:1]};
        else if (target_exp == -126 && !rounded[23])
            y = {large_sign, 8'b0, rounded[22:0]};
        else
            y = {large_sign, 8'(target_exp + 127), rounded[22:0]};
    end
endmodule
