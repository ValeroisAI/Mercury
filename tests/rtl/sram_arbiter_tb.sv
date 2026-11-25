// Formal harness. Arbitrary traffic is checked alongside a persistently
// contended ready bank and an identical bank with intermittent backpressure.
module sram_arbiter_tb (
    input wire clk,
    input wire rst,
    input wire [3:0] valid,
    input wire [31:0] lines,
    input wire [3:0] ready,
    input wire conflict_ready
);
    wire [3:0] grant, issue;
    wire [7:0] owner;
    wire [23:0] row;
    reg [3:0] expected_grant;
    reg [3:0] eligible;
    integer bank, requester;
    integer issued_owner;

    mercury_sram_arbiter #(.REQUESTERS(4), .BANKS(4), .LINE_ADDR_BITS(8)) arbitrary_traffic (
        clk, rst, valid, lines, ready, grant, issue, owner, row
    );

    function [1:0] reference_bank(input [7:0] address);
        // Independent bit slices for an 8-bit line address and four banks.
        reference_bank = address[1:0] ^ address[3:2] ^ address[6:5];
    endfunction

    always @* begin
        expected_grant = 0;
        eligible = 0;
        issued_owner = 0;
        for (bank = 0; bank < 4; bank = bank + 1) begin
            for (requester = 0; requester < 4; requester = requester + 1)
                eligible[requester] = valid[requester] &&
                    reference_bank(lines[requester*8 +: 8]) == 2'(bank);
            assert(issue[bank] == (!rst && ready[bank] && |eligible));
            if (issue[bank]) begin
                issued_owner = int'(owner[bank*2 +: 2]);
                assert(eligible[issued_owner]);
                assert(row[bank*6 +: 6] == lines[issued_owner*8+2 +: 6]);
                assert(!expected_grant[issued_owner]);
                expected_grant[issued_owner] = 1;
            end
        end
        assert(grant == expected_grant);
    end

    wire [3:0] conflict_grant, conflict_issue;
    wire [7:0] conflict_owner;
    wire [23:0] conflict_row;
    reg [3:0] rotation;
    reg initialized = 0;

    mercury_sram_arbiter #(.REQUESTERS(4), .BANKS(4), .LINE_ADDR_BITS(8)) conflict (
        clk, rst, 4'b1111, 32'b0, {3'b000, conflict_ready},
        conflict_grant, conflict_issue, conflict_owner, conflict_row
    );

    always @(posedge clk) begin
        if (rst) begin
            initialized <= 1;
            rotation <= 4'b0001;
        end else if (conflict_ready)
            rotation <= {rotation[2:0], rotation[3]};
    end

    always @* begin
        if (initialized && !rst) begin
            assert(conflict_issue == {3'b000, conflict_ready});
            assert(conflict_grant == (conflict_ready ? rotation : 4'b0000));
        end
    end
endmodule

// Prove the full-size mapping and its inverse for arbitrary line addresses.
module sram_bank_map_tb(input wire [21:0] line_addr);
    wire [5:0] bank, linear_bank;
    wire [15:0] row, linear_row;
    mercury_sram_bank_map swizzled(line_addr, bank, row);
    mercury_sram_bank_map #(.SWIZZLE(0)) linear(line_addr, linear_bank, linear_row);

    always @* begin
        assert(bank == (line_addr[5:0] ^ line_addr[11:6] ^ line_addr[18:13]));
        assert({row, bank ^ row[5:0] ^ row[12:7]} == line_addr);
        assert({linear_row, linear_bank} == line_addr);
    end
endmodule
