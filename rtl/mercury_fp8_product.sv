// Exact, unscaled OCP E4M3FN or E5M2 product as binary32.
// The finite nonzero products are all normal in binary32 and need no rounding.
module mercury_fp8_product (
    input wire [7:0] a,
    input wire [7:0] b,
    input wire a_e5m2,
    input wire b_e5m2,
    output reg [31:0] y
);
    function automatic [14:0] unpack(input [7:0] value, input e5m2);
        reg [4:0] exponent;
        reg [3:0] significand;
        reg signed [8:0] power;
        reg is_inf, is_nan;
        begin
            if (e5m2) begin
                exponent = value[6:2];
                significand = {1'b0, exponent != 0, value[1:0]};
                power = (exponent == 0 ? 9'sd1 : $signed({4'b0, exponent})) - 9'sd17;
                is_inf = exponent == 31 && value[1:0] == 0;
                is_nan = exponent == 31 && value[1:0] != 0;
            end else begin
                exponent = {1'b0, value[6:3]};
                significand = {exponent != 0, value[2:0]};
                power = (exponent == 0 ? 9'sd1 : $signed({4'b0, exponent})) - 9'sd10;
                is_inf = 0;
                is_nan = value[6:0] == 7'h7f;
            end
            unpack = {is_nan, is_inf, power, significand};
        end
    endfunction

    wire a_nan, b_nan, a_inf, b_inf;
    wire signed [8:0] a_power, b_power;
    wire [3:0] a_significand, b_significand;
    assign {a_nan, a_inf, a_power, a_significand} = unpack(a, a_e5m2);
    assign {b_nan, b_inf, b_power, b_significand} = unpack(b, b_e5m2);

    wire sign = a[7] ^ b[7];
    wire [7:0] product = a_significand * b_significand;
    reg [2:0] leading_bit;
    reg [23:0] normalized;
    reg [7:0] biased_exponent;
    integer bit_index;

    always @* begin
        leading_bit = 0;
        for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
            if (product[bit_index])
                leading_bit = 3'(bit_index);
        biased_exponent = 8'(a_power + b_power + $signed({6'b0, leading_bit}) + 9'sd127);
        normalized = {16'b0, product} << (5'd23 - {2'b0, leading_bit});
        if (a_nan || b_nan || (a_inf && b_significand == 0) || (b_inf && a_significand == 0))
            y = 32'h7fc00000;
        else if (a_inf || b_inf)
            y = {sign, 8'hff, 23'b0};
        else if (product == 0)
            y = {sign, 31'b0};
        else
            y = {sign, biased_exponent, normalized[22:0]};
    end
endmodule
