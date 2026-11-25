// Formal properties for mercury_fp32_add_rne over all 2^64 input pairs.
// These are algebraic identities of IEEE-754 RNE addition, so they need no
// reference implementation. The numeric oracle comparison (directed and
// random vectors against host IEEE float32) lives in scripts/check_fp32_rtl.py.
module fp32_add_tb (
    input wire [31:0] a,
    input wire [31:0] b
);
    localparam [31:0] QNAN = 32'h7fc00000;
    wire [31:0] y_ab, y_ba, y_neg, y_pz, y_nz;

    mercury_fp32_add_rne add_ab (.a(a), .b(b), .y(y_ab));
    mercury_fp32_add_rne add_ba (.a(b), .b(a), .y(y_ba));
    mercury_fp32_add_rne add_neg (.a({~a[31], a[30:0]}), .b({~b[31], b[30:0]}), .y(y_neg));
    mercury_fp32_add_rne add_pz (.a(a), .b(32'h00000000), .y(y_pz));
    mercury_fp32_add_rne add_nz (.a(a), .b(32'h80000000), .y(y_nz));

    wire a_nan = (&a[30:23]) && (|a[22:0]);
    wire b_nan = (&b[30:23]) && (|b[22:0]);
    wire y_nan = (&y_ab[30:23]) && (|y_ab[22:0]);
    wire a_zero = a[30:0] == 31'd0;

    always @* begin
        // Commutativity, bit for bit (including the sign of zero and the NaN encoding).
        assert(y_ab == y_ba);
        // NaN inputs give the canonical qNaN; any NaN output is canonical.
        if (a_nan || b_nan) assert(y_ab == QNAN);
        if (y_nan) assert(y_ab == QNAN);
        // RNE is sign-symmetric: (-a) + (-b) = -(a + b), except NaN and exact zero (+0 rule).
        if (!y_nan && y_ab[30:0] != 31'd0) assert(y_neg == {~y_ab[31], y_ab[30:0]});
        // Additive identity: a + (-0) = a for every non-NaN a, including -0.
        if (!a_nan) assert(y_nz == a);
        // a + (+0) = a, except -0 + +0 = +0.
        if (!a_nan) assert(y_pz == (a_zero ? 32'h00000000 : a));
        // x + (-x) = +0 for finite x.
        if (!(&a[30:23]) && b == {~a[31], a[30:0]}) assert(y_ab == 32'h00000000);
    end
endmodule
