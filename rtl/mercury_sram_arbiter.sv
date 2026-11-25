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
    output wire [REQUESTERS-1:0] req_grant,
    output wire [BANKS-1:0] bank_issue,
    output wire [BANKS*OWNER_BITS-1:0] bank_owner,
    output wire [BANKS*ROW_BITS-1:0] bank_row
);
    // REQUESTERS and BANKS are powers of two >= 2, and ROW_BITS > 0.
    reg [OWNER_BITS-1:0] next_owner [0:BANKS-1];
    wire [BANK_BITS-1:0] target_bank [0:REQUESTERS-1];
    wire [ROW_BITS-1:0] target_row [0:REQUESTERS-1];
    wire [BANKS-1:0] wins [0:REQUESTERS-1];
    genvar i, bank, requester, bit_index;

    for (i = 0; i < REQUESTERS; i = i + 1) begin: decode
        mercury_sram_bank_map #(
            .BANKS(BANKS), .LINE_ADDR_BITS(LINE_ADDR_BITS), .SWIZZLE(SWIZZLE)
        ) mapping (
            .line_addr(req_line[i*LINE_ADDR_BITS +: LINE_ADDR_BITS]),
            .bank(target_bank[i]),
            .row(target_row[i])
        );
        assign req_grant[i] = |wins[i];
    end

    for (bank = 0; bank < BANKS; bank = bank + 1) begin: select_bank
        wire [REQUESTERS-1:0] contenders, at_or_after, eligible, winner;
        for (requester = 0; requester < REQUESTERS; requester = requester + 1) begin: select_request
            assign contenders[requester] = req_valid[requester] &&
                target_bank[requester] == BANK_BITS'(bank);
            assign at_or_after[requester] = contenders[requester] &&
                OWNER_BITS'(requester) >= next_owner[bank];
            if (requester == 0) begin
                assign winner[requester] = eligible[requester];
            end else begin
                assign winner[requester] = eligible[requester] && !(|eligible[requester-1:0]);
            end
            assign wins[requester][bank] = winner[requester];
        end

        // First scan IDs at or above the pointer, then wrap to the lowest
        // contender. Prefix reductions select a one-hot winner in parallel.
        assign eligible = rst || !bank_ready[bank] ? 0 :
            |at_or_after ? at_or_after : contenders;
        assign bank_issue[bank] = |winner;

        for (bit_index = 0; bit_index < OWNER_BITS; bit_index = bit_index + 1) begin: encode_owner
            wire [REQUESTERS-1:0] owner_bits;
            for (requester = 0; requester < REQUESTERS; requester = requester + 1) begin: encode_bit
                assign owner_bits[requester] = winner[requester] && ((requester >> bit_index) & 1) != 0;
            end
            assign bank_owner[bank*OWNER_BITS + bit_index] = |owner_bits;
        end

        for (bit_index = 0; bit_index < ROW_BITS; bit_index = bit_index + 1) begin: select_row
            wire [REQUESTERS-1:0] row_bits;
            for (requester = 0; requester < REQUESTERS; requester = requester + 1) begin: select_bit
                assign row_bits[requester] = winner[requester] && target_row[requester][bit_index];
            end
            assign bank_row[bank*ROW_BITS + bit_index] = |row_bits;
        end

        always @(posedge clk) begin
            if (rst)
                next_owner[bank] <= 0;
            else if (bank_issue[bank])
                next_owner[bank] <= bank_owner[bank*OWNER_BITS +: OWNER_BITS] + 1'b1;
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
