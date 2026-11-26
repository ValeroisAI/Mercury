// Board-independent destructive self-test of the dedicated FPGA BRAM wrapper.
// Drive rst synchronously for at least one clock before observing done/pass.
module mercury_sram_board_test (
    input wire clk,
    input wire rst,
    output reg done,
    output reg pass
);
    localparam [1:0] WRITE = 0, MASK = 1, VERIFY = 2, CONTEND = 3;
    localparam [63:0] MASK_BITS = 64'hff00ff0000ff00ff;
    reg [1:0] phase, hot_bank;
    reg [8:0] row;
    reg [13:0] elapsed;
    reg [3:0] submitted, received;
    wire [3:0] req_valid = {4{!done}} & ~submitted;
    wire [3:0] req_write = {4{phase == WRITE || phase == MASK}};
    wire [3:0] resp_ready = {4{!done && elapsed[1]}};
    wire [3:0] req_grant, resp_valid;
    wire [43:0] req_line;
    wire [255:0] req_wdata, resp_rdata;
    wire [31:0] req_wstrb = {4{phase == MASK ? 8'ha5 : 8'hff}};
    wire [3:0] consumed = resp_ready & resp_valid;
    wire [3:0] mismatch;

    mercury_sram_bram memory (.*);

    for (genvar port_id = 0; port_id < 4; port_id++) begin: lanes
        wire [1:0] bank = phase == CONTEND ? hot_bank : 2'(port_id);
        wire [10:0] address = {row, bank ^ row[1:0] ^ row[4:3]};
        wire [15:0] token = {5'b0, row, bank};
        wire [63:0] pattern = {token ^ 16'ha5fc, ~token, token ^ 16'hca5f, token};
        assign req_line[port_id*11 +: 11] = address;
        assign req_wdata[port_id*64 +: 64] = phase == MASK ? ~pattern : pattern;
        assign mismatch[port_id] = consumed[port_id] &&
            resp_rdata[port_id*64 +: 64] !== (pattern ^ MASK_BITS);
    end

    wire phase_complete = |req_write ? (&(submitted | req_grant)) : (&(received | consumed));

    always @(posedge clk) begin
        if (rst) begin
            phase <= WRITE;
            hot_bank <= 0;
            row <= 0;
            elapsed <= 0;
            submitted <= 0;
            received <= 0;
            done <= 0;
            pass <= 0;
        end else if (!done) begin
            elapsed <= elapsed + 1'b1;
            submitted <= submitted | req_grant;
            received <= received | consumed;
            if ((|mismatch) || (|(consumed & (received | ~submitted))) || (&elapsed)) begin
                done <= 1;
            end else if (phase_complete) begin
                submitted <= 0;
                received <= 0;
                if (phase == CONTEND) begin
                    if (&hot_bank) begin
                        done <= 1;
                        pass <= 1;
                    end else begin
                        hot_bank <= hot_bank + 1'b1;
                    end
                end else if (&row) begin
                    phase <= phase + 1'b1;
                    // Keep the last initialized row for the contention phase.
                    if (phase != VERIFY) row <= 0;
                end else begin
                    row <= row + 1'b1;
                end
            end
        end
    end
endmodule
