// Single-clock FPGA memory behind the registered SRAM payload protocol.
// Address width describes the implemented capacity. Memory survives reset.
module mercury_sram_bram #(
    parameter integer REQUESTERS = 4,
    parameter integer BANKS = 4,
    parameter integer ROW_BITS = 9,
    parameter integer DATA_BITS = 64,
    parameter SWIZZLE = 1,
    localparam integer LINE_ADDR_BITS = ROW_BITS + $clog2(BANKS),
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
    output wire [REQUESTERS-1:0] resp_valid,
    output wire [REQUESTERS*DATA_BITS-1:0] resp_rdata,
    input wire [REQUESTERS-1:0] resp_ready
);
    wire [BANKS-1:0] issue, write_line, read_valid;
    wire [BANKS*ROW_BITS-1:0] row;
    wire [BANKS*DATA_BITS-1:0] wdata, rdata;
    wire [BANKS*STROBE_BITS-1:0] wstrb;

    mercury_sram_payload #(
        .REQUESTERS(REQUESTERS), .BANKS(BANKS), .LINE_ADDR_BITS(LINE_ADDR_BITS),
        .DATA_BITS(DATA_BITS), .SWIZZLE(SWIZZLE)
    ) bridge (
        .clk(clk), .rst(rst), .req_valid(req_valid), .req_write(req_write),
        .req_line(req_line), .req_wdata(req_wdata), .req_wstrb(req_wstrb),
        .req_grant(req_grant), .bank_ready({BANKS{1'b1}}), .bank_issue(issue),
        .bank_write(write_line), .bank_row(row), .bank_wdata(wdata),
        .bank_wstrb(wstrb), .bank_read_valid(read_valid), .bank_rdata(rdata),
        .resp_valid(resp_valid), .resp_rdata(resp_rdata), .resp_ready(resp_ready)
    );

    for (genvar bank = 0; bank < BANKS; bank++) begin: banks
        (* ram_style = "block" *) reg [DATA_BITS-1:0] memory [0:(1 << ROW_BITS)-1];
        wire [ROW_BITS-1:0] address = row[bank*ROW_BITS +: ROW_BITS];
        wire write_enable = issue[bank] && write_line[bank];
        wire read_enable = issue[bank] && !write_line[bank];
        reg [DATA_BITS-1:0] read_data;
        reg reply_valid;

        assign rdata[bank*DATA_BITS +: DATA_BITS] = read_data;
        assign read_valid[bank] = reply_valid;

        always @(posedge clk) begin
            // Reset cancels an in-flight read without resetting or clearing RAM.
            reply_valid <= !rst && read_enable;
            if (read_enable)
                read_data <= memory[address];
            if (write_enable)
                for (integer byte_index = 0; byte_index < STROBE_BITS; byte_index++)
                    if (wstrb[bank*STROBE_BITS + byte_index])
                        memory[address][byte_index*8 +: 8] <= wdata[bank*DATA_BITS + byte_index*8 +: 8];
        end
    end
endmodule
