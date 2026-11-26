`timescale 1ns/1ps

module sram_bram_tb;
    reg clk = 0, rst = 1;
    reg [3:0] req_valid = 0, req_write = 0, resp_ready = 0;
    reg [43:0] req_line = 0;
    reg [255:0] req_wdata = 0;
    reg [31:0] req_wstrb = 0;
    wire [3:0] req_grant, resp_valid;
    wire [255:0] resp_rdata;

    mercury_sram_bram dut (.*);

    function automatic [10:0] line_for(input [8:0] row, input [1:0] bank);
        line_for = {row, bank ^ row[1:0] ^ row[4:3]};
    endfunction

    function automatic [63:0] pattern(input integer port_id);
        pattern = 64'h0123456789abcdef ^ (64'h1111111111111111 * port_id);
    endfunction

    task automatic tick(input [3:0] grant, input [3:0] response);
        #1;
        if (req_grant !== grant || resp_valid !== response)
            $fatal(1, "expected grants/responses %h/%h, got %h/%h", grant, response, req_grant, resp_valid);
        clk = 1;
        #1; clk = 0;
    endtask

    task automatic check_data(input integer port_id, input [63:0] expected);
        if (resp_rdata[port_id*64 +: 64] !== expected)
            $fatal(1, "requester %0d expected %h got %h", port_id, expected, resp_rdata[port_id*64 +: 64]);
    endtask

    initial begin
        tick(0, 0);
        rst = 0;

        // First and last row in all four banks. Include row bits from both XOR folds.
        req_valid = '1;
        req_write = '1;
        req_wstrb = '1;
        for (integer r = 0; r < 4; r++) begin
            req_line[r*11 +: 11] = line_for(0, 2'(r));
            req_wdata[r*64 +: 64] = ~pattern(r);
        end
        tick('1, 0);
        for (integer r = 0; r < 4; r++) begin
            req_line[r*11 +: 11] = line_for(511, 2'(r));
            req_wdata[r*64 +: 64] = pattern(r);
        end
        tick('1, 0);

        req_write = 0;
        tick('1, 0);
        req_valid = 0;
        tick(0, 0);
        tick(0, '1);
        tick(0, '1);
        for (integer r = 0; r < 4; r++) check_data(r, pattern(r));

        // Response consumption and masked write happen on the same edge.
        resp_ready = '1;
        req_valid = 1;
        req_write = 1;
        req_wstrb = 32'h00000081;
        req_wdata[63:0] = 64'hfedcba9876543210;
        tick(1, '1);
        req_wstrb = 0;
        req_wdata[63:0] = 0;
        tick(1, 0);

        // Two reads use the same bank on consecutive edges. Reset drops both.
        req_valid = 3;
        req_write = 0;
        req_line[21:11] = line_for(0, 0);
        tick(2, 0);
        req_valid = 1;
        tick(1, 0);
        rst = 1;
        tick(0, 0);
        rst = 0;
        req_valid = 3;
        tick(1, 0);
        req_valid = 2;
        tick(2, 0);
        req_valid = 0;
        tick(0, 1);
        check_data(0, 64'hfe23456789abcd10);
        tick(0, 2);
        check_data(1, ~pattern(0));
        tick(0, 0);

        $display("PASS: SRAM BRAM integration, masked and zero-mask writes, refill, and reset");
        $finish;
    end
endmodule
