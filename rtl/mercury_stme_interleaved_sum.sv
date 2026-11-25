// Interleave independent dot products through the 4-stage binary32 adder.
// Each context preserves its own logical K order and reserves result capacity.
module mercury_stme_interleaved_sum #(
    parameter integer CONTEXTS = 8,
    parameter integer GROUP_TERMS = 128,
    localparam integer CTX_BITS = $clog2(CONTEXTS),
    localparam integer COUNT_BITS = GROUP_TERMS > 1 ? $clog2(GROUP_TERMS) : 1
) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    output wire in_ready,
    input wire [CTX_BITS-1:0] in_context,
    input wire [7:0] a,
    input wire [7:0] b,
    input wire a_e5m2,
    input wire b_e5m2,
    input wire [CTX_BITS-1:0] read_context,
    output wire out_valid,
    input wire out_ready,
    output wire [31:0] out_sum
);
    // CONTEXTS is a power of two >= 2. GROUP_TERMS must be positive.
    localparam integer ADD_STAGES = 4;
    reg [CONTEXTS-1:0] busy;
    reg [CONTEXTS-1:0] result_pending;
    reg [31:0] partial [0:CONTEXTS-1];
    reg [31:0] result [0:CONTEXTS-1];
    reg [COUNT_BITS-1:0] terms [0:CONTEXTS-1];
    reg [CTX_BITS-1:0] tag [0:ADD_STAGES-1];
    wire [CTX_BITS-1:0] completed_context = tag[ADD_STAGES-1];
    wire [31:0] product, new_partial;
    wire add_valid;
    wire accept = in_valid && in_ready;
    integer ctx, stage;

    assign in_ready = !rst && !busy[in_context] && !result_pending[in_context];
    assign out_valid = !rst && result_pending[read_context];
    // Keep read_context stable while backpressuring a result.
    assign out_sum = result[read_context];
    mercury_fp8_product decode_and_multiply(a, b, a_e5m2, b_e5m2, product);
    mercury_fp32_add_pipe accumulate(clk, rst, accept, partial[in_context], product, add_valid, new_partial);

    always @(posedge clk) begin
        if (rst) begin
            busy <= 0;
            result_pending <= 0;
            for (ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                partial[ctx] <= 0;
                result[ctx] <= 0;
                terms[ctx] <= 0;
            end
        end else begin
            // Tagged pipeline outputs are never stalled. A result slot
            // for the context is reserved until the consumer drains it.
            tag[0] <= in_context;
            for (stage = 1; stage < ADD_STAGES; stage = stage + 1)
                tag[stage] <= tag[stage - 1];
            if (accept)
                busy[in_context] <= 1;
            if (out_valid && out_ready)
                result_pending[read_context] <= 0;
            if (add_valid) begin
                busy[completed_context] <= 0;
                if (terms[completed_context] == COUNT_BITS'(GROUP_TERMS - 1)) begin
                    partial[completed_context] <= 0;
                    terms[completed_context] <= 0;
                    result[completed_context] <= new_partial;
                    result_pending[completed_context] <= 1;
                end else begin
                    partial[completed_context] <= new_partial;
                    terms[completed_context] <= terms[completed_context] + 1'b1;
                end
            end
        end
    end
endmodule
