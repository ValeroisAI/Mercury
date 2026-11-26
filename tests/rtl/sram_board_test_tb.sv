`timescale 1ns/1ps

module sram_board_test_tb;
    reg clk = 0, rst = 1;
    wire done, pass;
    integer cycles;
    always #1 clk = ~clk;
    mercury_sram_board_test dut (.*);

    task automatic restart;
        rst = 1;
        repeat (2) @(negedge clk);
        if ({done, pass} !== 0) $fatal(1, "reset did not clear result");
        rst = 0;
    endtask

    task automatic expect_result(input expected, input integer budget);
        cycles = 0;
        while (!done && cycles < budget) begin
            @(negedge clk);
            cycles++;
        end
        if (done !== 1 || pass !== expected)
            $fatal(1, "expected pass=%b, observed done=%b pass=%b after %0d cycles", expected, done, pass, cycles);
    endtask

    initial begin
        restart();
        repeat (1500) @(negedge clk);
        if (done) $fatal(1, "test should still be active before restart");
        restart();
        expect_result(1, 6000);
        $display("PASS: all 2,048 SRAM lines and four contention cases in %0d cycles", cycles);

        restart();
        force dut.resp_rdata = 0;
        expect_result(0, 6000);
        release dut.resp_rdata;
        $display("PASS: injected read corruption detected");

        restart();
        force dut.req_grant = 0;
        expect_result(0, 16384);
        release dut.req_grant;
        $display("PASS: missing grant detected by watchdog");
        $finish;
    end
endmodule
