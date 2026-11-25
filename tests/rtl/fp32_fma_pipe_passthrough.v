// Oracle-only mapping. Replaces pipeline registers with wires, keeping the
// arithmetic phases available for combinational evaluation by Yosys.
module \$dff (CLK, D, Q);
    parameter WIDTH = 1;
    parameter CLK_POLARITY = 1;
    input CLK;
    input [WIDTH-1:0] D;
    output [WIDTH-1:0] Q;
    assign Q = D;
endmodule
