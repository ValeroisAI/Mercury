// Six-stage binary32 FMA. Accepts one triple per cycle without backpressure.
// A triple accepted at edge N reaches out_valid and y after edge N+5.
module mercury_fp32_fma_pipe (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [31:0] c,
    output wire out_valid,
    output reg [31:0] y
);
    reg [5:0] valid;
    assign out_valid = !rst && valid[5];
    always @(posedge clk)
        valid <= rst ? 6'b0 : {valid[4:0], in_valid};

    function automatic [5:0] highest_index(input [63:0] value);
        reg [31:0] half;
        reg [15:0] quarter;
        reg [7:0] eighth;
        reg [3:0] nibble;
        reg [1:0] pair;
        begin
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

    // Stage 1. Decode special values and form the exact 48-bit product.
    wire [7:0] ea = a[30:23], eb = b[30:23], ec = c[30:23];
    wire [23:0] ma = {ea != 0, a[22:0]}, mb = {eb != 0, b[22:0]}, mc = {ec != 0, c[22:0]};
    wire a_inf = a[30:0] == 31'h7f800000;
    wire b_inf = b[30:0] == 31'h7f800000;
    wire c_inf = c[30:0] == 31'h7f800000;
    wire any_nan = (ea == 255 && !a_inf) || (eb == 255 && !b_inf) || (ec == 255 && !c_inf);
    wire product_sign = a[31] ^ b[31];
    wire zero_product = ma == 0 || mb == 0;
    wire invalid = any_nan || ((a_inf || b_inf) && (zero_product || (c_inf && product_sign != c[31])));
    wire exceptional = invalid || a_inf || b_inf || c_inf || zero_product;
    wire [31:0] exceptional_y =
        invalid ? 32'h7fc00000 :
        (a_inf || b_inf) ? {product_sign, 8'hff, 23'b0} :
        (c_inf || mc != 0) ? c : {product_sign & c[31], 31'b0};

    reg [47:0] s1_product;
    reg [23:0] s1_mc;
    reg [32:0] s1_special;
    reg signed [9:0] s1_product_exp, s1_c_exp;
    reg s1_product_sign, s1_c_sign;
    always @(posedge clk) begin
        s1_product <= ma * mb;
        s1_mc <= mc;
        s1_product_exp <= $signed({2'b0, ea == 0 ? 8'd1 : ea}) +
                          $signed({2'b0, eb == 0 ? 8'd1 : eb}) - 10'sd300;
        s1_c_exp <= $signed({2'b0, ec == 0 ? 8'd1 : ec}) - 10'sd150;
        s1_product_sign <= product_sign;
        s1_c_sign <= c[31];
        s1_special <= {exceptional, exceptional_y};
    end

    // Stage 2. Normalize both operands and select the larger magnitude.
    wire [5:0] p_msb = highest_index({16'b0, s1_product});
    wire [5:0] c_msb = highest_index({40'b0, s1_mc});
    wire signed [9:0] p_exp = s1_product_exp + $signed({4'b0, p_msb});
    wire signed [9:0] c_exp = s1_c_exp + $signed({4'b0, c_msb});
    wire [54:0] p_norm = {7'b0, s1_product} << (6'd54 - p_msb);
    wire [54:0] c_norm = {31'b0, s1_mc} << (6'd54 - c_msb);
    wire product_larger = s1_mc == 0 || p_exp > c_exp || (p_exp == c_exp && p_norm >= c_norm);

    reg [54:0] s2_large, s2_small;
    reg signed [9:0] s2_exp, s2_gap;
    reg s2_sign, s2_same_sign;
    reg [32:0] s2_special;
    always @(posedge clk) begin
        s2_large <= product_larger ? p_norm : c_norm;
        s2_small <= product_larger ? c_norm : p_norm;
        s2_exp <= product_larger ? p_exp : c_exp;
        s2_gap <= product_larger ? p_exp - c_exp : c_exp - p_exp;
        s2_sign <= product_larger ? s1_product_sign : s1_c_sign;
        s2_same_sign <= s1_product_sign == s1_c_sign;
        s2_special <= s1_special;
    end

    // Stage 3. Align the smaller magnitude, jamming any discarded tail.
    reg [54:0] aligned;
    always @* begin
        if (s2_gap <= 0)
            aligned = s2_small;
        else if (s2_gap >= 55)
            aligned = {54'b0, |s2_small};
        else begin
            aligned = s2_small >> s2_gap[5:0];
            aligned[0] = aligned[0] | (|(s2_small << (6'd55 - s2_gap[5:0])));
        end
    end
    reg [54:0] s3_large, s3_aligned;
    reg signed [9:0] s3_exp;
    reg s3_sign, s3_same_sign;
    reg [32:0] s3_special;
    always @(posedge clk) begin
        s3_large <= s2_large;
        s3_aligned <= aligned;
        s3_exp <= s2_exp;
        s3_sign <= s2_sign;
        s3_same_sign <= s2_same_sign;
        s3_special <= s2_special;
    end

    // Stage 4. Add or subtract aligned magnitudes without rounding.
    reg [55:0] s4_mag;
    reg signed [9:0] s4_exp;
    reg s4_sign;
    reg [32:0] s4_special;
    always @(posedge clk) begin
        s4_mag <= s3_same_sign ? {1'b0, s3_large} + {1'b0, s3_aligned} :
                                 {1'b0, s3_large} - {1'b0, s3_aligned};
        s4_exp <= s3_exp;
        s4_sign <= s3_sign;
        s4_special <= s3_special;
    end

    // Stage 5. Determine the result exponent and alignment for rounding.
    wire [5:0] sum_msb = highest_index({8'b0, s4_mag});
    wire signed [9:0] leading_exp = s4_exp + $signed({4'b0, sum_msb}) - 10'sd54;
    wire signed [9:0] target_exp = leading_exp < -10'sd126 ? -10'sd126 : leading_exp;
    wire signed [9:0] round_shift = target_exp - s4_exp + 10'sd31;
    reg [55:0] s5_mag;
    reg signed [9:0] s5_exp, s5_shift;
    reg s5_sign, s5_zero;
    reg [32:0] s5_special;
    always @(posedge clk) begin
        s5_mag <= s4_mag;
        s5_exp <= target_exp;
        s5_shift <= round_shift;
        s5_sign <= s4_sign;
        s5_zero <= s4_mag == 0;
        s5_special <= s4_special;
    end

    // Stage 6. Extract rounding bits, round to nearest-even, and pack.
    reg [55:0] shifted;
    reg guard_bit, sticky_bit;
    always @* begin
        shifted = 0;
        guard_bit = 0;
        sticky_bit = 0;
        if (s5_shift <= 0)
            shifted = s5_mag << (6'd0 - s5_shift[5:0]);
        else if (s5_shift <= 56) begin
            shifted = s5_mag >> s5_shift[5:0];
            guard_bit = s5_mag[s5_shift[5:0] - 6'd1];
            sticky_bit = s5_shift > 1 && (|(s5_mag << (6'd57 - s5_shift[5:0])));
        end
    end
    wire [24:0] rounded = shifted[24:0] + 25'(guard_bit && (sticky_bit || shifted[0]));
    always @(posedge clk) begin
        if (s5_special[32])
            y <= s5_special[31:0];
        else if (s5_zero)
            y <= 0;
        else if (s5_exp > 127 || (s5_exp == 127 && rounded[24]))
            y <= {s5_sign, 8'hff, 23'b0};
        else if (rounded[24])
            y <= {s5_sign, 8'(s5_exp + 128), rounded[23:1]};
        else if (s5_exp == -126 && !rounded[23])
            y <= {s5_sign, 8'b0, rounded[22:0]};
        else
            y <= {s5_sign, 8'(s5_exp + 127), rounded[22:0]};
    end
endmodule
