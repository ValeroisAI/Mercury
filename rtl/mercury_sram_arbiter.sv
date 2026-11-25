// Shared Tensor SRAM request selection. Each ready bank accepts at most
// one 64-byte line address per cycle. SRAM macros and data routing are external.
module mercury_sram_arbiter #(
    parameter integer REQUESTERS = 8,
    parameter integer BANKS = 64,
    parameter integer LINE_ADDR_BITS = 22,
    parameter SWIZZLE = 1,
    localparam integer BANK_BITS = $clog2(BANKS),
    localparam integer OWNER_BITS = $clog2(REQUESTERS),
    localparam integer ROW_BITS = LINE_ADDR_BITS - BANK_BITS
) (
    input wire clk,
    input wire rst,
    input wire [REQUESTERS-1:0] req_valid,
    input wire [REQUESTERS*LINE_ADDR_BITS-1:0] req_line,
    input wire [BANKS-1:0] bank_ready,
    output reg [REQUESTERS-1:0] req_grant,
    output reg [BANKS-1:0] bank_issue,
    output reg [BANKS*OWNER_BITS-1:0] bank_owner,
    output reg [BANKS*ROW_BITS-1:0] bank_row
);
    // REQUESTERS and BANKS are powers of two >= 2, and ROW_BITS > 0.
    // Truncation of pointer + offset implements round-robin wrap.
    reg [OWNER_BITS-1:0] next_owner [0:BANKS-1];
    wire [BANK_BITS-1:0] target_bank [0:REQUESTERS-1];
    wire [ROW_BITS-1:0] target_row [0:REQUESTERS-1];
    reg [OWNER_BITS-1:0] owner;
    integer bank, offset, index;
    genvar i;

    for (i = 0; i < REQUESTERS; i = i + 1) begin: decode
        mercury_sram_bank_map #(
            .BANKS(BANKS), .LINE_ADDR_BITS(LINE_ADDR_BITS), .SWIZZLE(SWIZZLE)
        ) mapping (
            .line_addr(req_line[i*LINE_ADDR_BITS +: LINE_ADDR_BITS]),
            .bank(target_bank[i]),
            .row(target_row[i])
        );
    end

    always @* begin
        req_grant = 0;
        bank_issue = 0;
        bank_owner = 0;
        bank_row = 0;
        owner = 0;
        for (bank = 0; bank < BANKS; bank = bank + 1) begin
            for (offset = 0; offset < REQUESTERS; offset = offset + 1) begin
                owner = next_owner[bank] + OWNER_BITS'(offset);
                if (!rst && bank_ready[bank] && !bank_issue[bank] &&
                    req_valid[owner] && target_bank[owner] == BANK_BITS'(bank)) begin
                    req_grant[owner] = 1;
                    bank_issue[bank] = 1;
                    bank_owner[bank*OWNER_BITS +: OWNER_BITS] = owner;
                    // Keeping the upper address bits makes the XOR map
                    // bijective. The inverse can recover the low bank bits.
                    bank_row[bank*ROW_BITS +: ROW_BITS] = target_row[owner];
                end
            end
        end
    end

    always @(posedge clk) begin
        for (index = 0; index < BANKS; index = index + 1) begin
            if (rst)
                next_owner[index] <= 0;
            else if (bank_issue[index])
                next_owner[index] <= bank_owner[index*OWNER_BITS +: OWNER_BITS] + 1'b1;
        end
    end
endmodule

module mercury_sram_bank_map #(
    parameter integer BANKS = 64,
    parameter integer LINE_ADDR_BITS = 22,
    parameter SWIZZLE = 1,
    localparam integer BANK_BITS = $clog2(BANKS)
) (
    input wire [LINE_ADDR_BITS-1:0] line_addr,
    output wire [BANK_BITS-1:0] bank,
    output wire [LINE_ADDR_BITS-BANK_BITS-1:0] row
);
    wire [LINE_ADDR_BITS-1:0] folded =
        line_addr ^ (line_addr >> BANK_BITS) ^ (line_addr >> (2*BANK_BITS + 1));
    assign bank = SWIZZLE ? folded[BANK_BITS-1:0] : line_addr[BANK_BITS-1:0];
    assign row = line_addr[LINE_ADDR_BITS-1:BANK_BITS];
endmodule
