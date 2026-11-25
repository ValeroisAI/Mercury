// Token scoreboard for one DMA stream. The engine owns memory visibility
// and must post completion only after its destination writes are visible.
module mercury_dma_tokens #(
    parameter integer SLOTS = 4,
    parameter integer GENERATION_BITS = 3,
    localparam integer SLOT_BITS = $clog2(SLOTS),
    localparam integer TOKEN_BITS = SLOT_BITS + GENERATION_BITS
) (
    input wire clk,
    input wire rst,
    input wire issue_valid,
    output reg issue_ready,
    output reg [TOKEN_BITS-1:0] issue_token,
    input wire complete_valid,
    input wire [TOKEN_BITS-1:0] complete_token,
    output reg complete_accepted,
    input wire wait_valid,
    input wire [TOKEN_BITS-1:0] wait_token,
    output reg wait_ready,
    output reg wait_stale
);
    // SLOTS >= 2 and GENERATION_BITS >= 1. The generation increases on
    // acquisition. Clients must not keep stale tokens across a full wrap,
    // and the DMA engine must flush outstanding messages on reset.
    reg [SLOTS-1:0] allocated;
    reg [SLOTS-1:0] finished;
    reg [GENERATION_BITS-1:0] generation [0:SLOTS-1];
    wire [SLOT_BITS-1:0] complete_slot = complete_token[SLOT_BITS-1:0];
    wire [SLOT_BITS-1:0] wait_slot = wait_token[SLOT_BITS-1:0];
    wire [SLOT_BITS-1:0] issue_slot = issue_token[SLOT_BITS-1:0];
    integer slot, reset_slot;

    always @* begin
        issue_ready = 0;
        issue_token = 0;
        complete_accepted = 0;
        wait_ready = 0;
        wait_stale = 0;
        if (!rst) begin
            for (slot = 0; slot < SLOTS; slot = slot + 1) begin
                if (!allocated[slot] && !issue_ready) begin
                    issue_ready = 1;
                    issue_token = {generation[slot], SLOT_BITS'(slot)};
                end
            end
            if (complete_valid && int'(complete_slot) < SLOTS) begin
                complete_accepted = allocated[complete_slot] && !finished[complete_slot] &&
                    generation[complete_slot] == complete_token[TOKEN_BITS-1:SLOT_BITS];
            end
            if (wait_valid) begin
                wait_stale = 1;
                if (int'(wait_slot) < SLOTS && allocated[wait_slot] &&
                    generation[wait_slot] == wait_token[TOKEN_BITS-1:SLOT_BITS]) begin
                    wait_ready = finished[wait_slot];
                    wait_stale = 0;
                end
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            allocated <= 0;
            finished <= 0;
            for (reset_slot = 0; reset_slot < SLOTS; reset_slot = reset_slot + 1)
                generation[reset_slot] <= 0;
        end else begin
            if (issue_valid && issue_ready) begin
                allocated[issue_slot] <= 1;
                finished[issue_slot] <= 0;
            end
            if (complete_accepted)
                finished[complete_slot] <= 1;
            if (wait_ready) begin
                allocated[wait_slot] <= 0;
                finished[wait_slot] <= 0;
                generation[wait_slot] <= generation[wait_slot] + 1'b1;
            end
        end
    end
endmodule
