// Formal harness for mercury_dma_tokens (SLOTS=4, GENERATION_BITS=3).
//
// An independent abstract scoreboard (allocated / completed / generation per
// slot) is driven only by the handshakes the DUT accepted. Every cycle after
// the first synchronous reset, every DUT output must match the scoreboard's
// prediction, which is computed from state *before* the clock edge.
// Token encoding: {generation[2:0], slot[1:0]}.
//
// Scope (bounded): sat -seq N from reset covers sequences of N cycles. A
// generation wrap (8 retirements of one slot) needs more than 24 cycles, so it
// is outside a short bound. The modulo-8 increment is still checked on every
// retirement that falls inside the bound.
module dma_tokens_tb (
    input wire clk,
    input wire rst,
    input wire issue_valid,
    input wire complete_valid,
    input wire [4:0] complete_token,
    input wire wait_valid,
    input wire [4:0] wait_token
);
    wire issue_ready, complete_accepted, wait_ready, wait_stale;
    wire [4:0] issue_token;

    mercury_dma_tokens #(.SLOTS(4), .GENERATION_BITS(3)) dut (
        .clk(clk), .rst(rst),
        .issue_valid(issue_valid), .issue_ready(issue_ready), .issue_token(issue_token),
        .complete_valid(complete_valid), .complete_token(complete_token),
        .complete_accepted(complete_accepted),
        .wait_valid(wait_valid), .wait_token(wait_token),
        .wait_ready(wait_ready), .wait_stale(wait_stale)
    );

    // ---------------- abstract scoreboard state ----------------
    reg initialized = 0;
    reg [3:0] alloc;
    reg [3:0] done;
    reg [2:0] gen0, gen1, gen2, gen3;

    function [2:0] gen_of(input [1:0] slot, input [2:0] g0, input [2:0] g1,
                          input [2:0] g2, input [2:0] g3);
        case (slot)
            2'd0: gen_of = g0;
            2'd1: gen_of = g1;
            2'd2: gen_of = g2;
            default: gen_of = g3;
        endcase
    endfunction

    // ---------------- predicted (pre-clock) responses ----------------
    reg exp_issue_ready;
    reg [1:0] free_slot;
    reg [4:0] exp_issue_token;
    reg exp_complete_accepted, exp_wait_ready, exp_wait_stale;
    wire [1:0] c_slot = complete_token[1:0];
    wire [2:0] c_gen = complete_token[4:2];
    wire [1:0] w_slot = wait_token[1:0];
    wire [2:0] w_gen = wait_token[4:2];
    wire c_match = gen_of(c_slot, gen0, gen1, gen2, gen3) == c_gen;
    wire w_match = gen_of(w_slot, gen0, gen1, gen2, gen3) == w_gen;

    always @* begin
        // Lowest-index free slot, from the current state only (no same-cycle reuse).
        free_slot = !alloc[0] ? 2'd0 : !alloc[1] ? 2'd1 : !alloc[2] ? 2'd2 : 2'd3;
        exp_issue_ready = !rst && !(&alloc);
        exp_issue_token = {gen_of(free_slot, gen0, gen1, gen2, gen3), free_slot};
        // Complete: matching, allocated, not yet completed (rejects stale and duplicate).
        exp_complete_accepted = !rst && complete_valid && alloc[c_slot] && !done[c_slot] && c_match;
        // Wait: ready only on a completion registered before this edge (no same-cycle bypass).
        exp_wait_ready = !rst && wait_valid && alloc[w_slot] && done[w_slot] && w_match;
        // Stale: free slot or generation mismatch. A pending (allocated, matching,
        // not completed) token is neither ready nor stale.
        exp_wait_stale = !rst && wait_valid && !(alloc[w_slot] && w_match);
    end

    wire issue_fire = issue_valid && exp_issue_ready;
    wire wait_fire = wait_valid && exp_wait_ready;

    always @(posedge clk) begin
        if (rst) begin
            initialized <= 1;
            alloc <= 0;
            done <= 0;
            gen0 <= 0; gen1 <= 0; gen2 <= 0; gen3 <= 0;
        end else begin
            // The three updates target disjoint slots: issue takes a free slot,
            // complete needs allocated && !done, wait needs allocated && done.
            if (issue_fire) begin
                alloc[free_slot] <= 1;
                done[free_slot] <= 0;
            end
            if (exp_complete_accepted)
                done[c_slot] <= 1;
            if (wait_fire) begin
                alloc[w_slot] <= 0;
                done[w_slot] <= 0;
                case (w_slot)
                    2'd0: gen0 <= gen0 + 3'd1;
                    2'd1: gen1 <= gen1 + 3'd1;
                    2'd2: gen2 <= gen2 + 3'd1;
                    default: gen3 <= gen3 + 3'd1;
                endcase
            end
        end
    end

    // ---------------- checks ----------------
    always @* begin
        if (initialized) begin
            assert(issue_ready == exp_issue_ready);
            if (issue_valid && issue_ready)
                assert(issue_token == exp_issue_token);
            assert(complete_accepted == exp_complete_accepted);
            assert(wait_ready == exp_wait_ready);
            assert(wait_stale == exp_wait_stale);
            // Structural properties that follow from the contract.
            assert(!(wait_ready && wait_stale));                 // never both
            if (!wait_valid) assert(!wait_ready && !wait_stale);  // gated by wait_valid
            if (&alloc) assert(!issue_ready);                     // backpressure when full
            if (rst) assert(!issue_ready && !complete_accepted && !wait_ready && !wait_stale);
        end
    end
endmodule
