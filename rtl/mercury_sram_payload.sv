// Bank-macro interface. bank_issue is an acceptance strobe, not a
// separately held valid. Each read reply arrives once, at least one
// cycle after issue. Reset must also cancel pending macro replies.
module mercury_sram_payload #(
    parameter integer REQUESTERS = 8,
    parameter integer BANKS = 64,
    parameter integer LINE_ADDR_BITS = 22,
    parameter integer DATA_BITS = 512,
    parameter SWIZZLE = 1,
    localparam integer OWNER_BITS = $clog2(REQUESTERS),
    localparam integer ROW_BITS = LINE_ADDR_BITS - $clog2(BANKS),
    localparam integer STROBE_BITS = DATA_BITS / 8
) (
    input wire clk,
    input wire rst,
    input wire [REQUESTERS-1:0] req_valid,
    input wire [REQUESTERS-1:0] req_write,
    input wire [REQUESTERS*LINE_ADDR_BITS-1:0] req_line,
    input wire [REQUESTERS*DATA_BITS-1:0] req_wdata,
    input wire [REQUESTERS*STROBE_BITS-1:0] req_wstrb,
    output wire [REQUESTERS-1:0] req_grant,
    input wire [BANKS-1:0] bank_ready,
    output wire [BANKS-1:0] bank_issue,
    output wire [BANKS-1:0] bank_write,
    output wire [BANKS*ROW_BITS-1:0] bank_row,
    output wire [BANKS*DATA_BITS-1:0] bank_wdata,
    output wire [BANKS*STROBE_BITS-1:0] bank_wstrb,
    input wire [BANKS-1:0] bank_read_valid,
    input wire [BANKS*DATA_BITS-1:0] bank_rdata,
    output wire [REQUESTERS-1:0] resp_valid,
    output wire [REQUESTERS*DATA_BITS-1:0] resp_rdata,
    input wire [REQUESTERS-1:0] resp_ready
);
    // Parameters follow the arbiter's power-of-two contract. DATA_BITS
    // is a positive multiple of eight. Lower slice indices are port 0.
    wire [BANKS*OWNER_BITS-1:0] bank_owner;
    reg [BANKS-1:0] bank_pending;
    reg [REQUESTERS-1:0] requester_pending;
    wire [BANKS-1:0] bank_reply = bank_pending & bank_read_valid;
    wire [REQUESTERS-1:0] response_consumed = resp_valid & resp_ready;
    wire [BANKS-1:0] replies [0:REQUESTERS-1];
    genvar b, r, node;

    mercury_sram_arbiter #(
        .REQUESTERS(REQUESTERS), .BANKS(BANKS), .LINE_ADDR_BITS(LINE_ADDR_BITS), .SWIZZLE(SWIZZLE)
    ) arbiter (
        .clk(clk), .rst(rst),
        .req_valid(req_valid & (~requester_pending | response_consumed)), .req_line(req_line),
        .bank_ready(bank_ready & (~bank_pending | bank_reply)),
        .req_grant(req_grant), .bank_issue(bank_issue),
        .bank_owner(bank_owner), .bank_row(bank_row)
    );

    for (b = 0; b < BANKS; b = b + 1) begin: bank_port
        wire [OWNER_BITS-1:0] owner = bank_owner[b*OWNER_BITS +: OWNER_BITS];
        reg [OWNER_BITS-1:0] read_owner;
        wire read_issued = bank_issue[b] && !bank_write[b];
        wire capture_reply = bank_reply[b];

        assign bank_write[b] = bank_issue[b] && req_write[owner];
        assign bank_wdata[b*DATA_BITS +: DATA_BITS] =
            bank_write[b] ? req_wdata[owner*DATA_BITS +: DATA_BITS] : 0;
        assign bank_wstrb[b*STROBE_BITS +: STROBE_BITS] =
            bank_write[b] ? req_wstrb[owner*STROBE_BITS +: STROBE_BITS] : 0;

        for (r = 0; r < REQUESTERS; r = r + 1) begin: route_reply
            assign replies[r][b] = capture_reply && read_owner == OWNER_BITS'(r);
        end

        always @(posedge clk) begin
            if (rst) begin
                bank_pending[b] <= 0;
                read_owner <= 0;
            end else if (read_issued) begin
                bank_pending[b] <= 1;
                read_owner <= owner;
            end else if (capture_reply) begin
                bank_pending[b] <= 0;
            end
        end
    end

    for (r = 0; r < REQUESTERS; r = r + 1) begin: requester_port
        wire capture_reply = |replies[r];
        wire read_issued = req_grant[r] && !req_write[r];
        wire [DATA_BITS-1:0] reply_tree [1:2*BANKS-1];
        reg held_valid;
        reg [DATA_BITS-1:0] held_data;

        // An accepted read owns this slot through delivery. This makes
        // the bank selection one-hot even with simultaneous bank replies.
        for (b = 0; b < BANKS; b = b + 1) begin: select_bank_data
            assign reply_tree[BANKS+b] = {DATA_BITS{replies[r][b]}} &
                bank_rdata[b*DATA_BITS +: DATA_BITS];
        end
        for (node = 1; node < BANKS; node = node + 1) begin: reduce_replies
            assign reply_tree[node] = reply_tree[2*node] | reply_tree[2*node+1];
        end

        assign resp_valid[r] = !rst && held_valid;
        assign resp_rdata[r*DATA_BITS +: DATA_BITS] = held_data;

        always @(posedge clk) begin
            if (rst) begin
                requester_pending[r] <= 0;
                held_valid <= 0;
                held_data <= 0;
            end else begin
                // A new read retains the slot when the old response retires.
                if (read_issued || response_consumed[r])
                    requester_pending[r] <= read_issued;
                if (capture_reply) begin
                    held_valid <= 1;
                    held_data <= reply_tree[1];
                end else if (response_consumed[r]) begin
                    held_valid <= 0;
                end
            end
        end
    end
endmodule
