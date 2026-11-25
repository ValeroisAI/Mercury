// Bounded formal harness for mercury_sram_payload (4 requesters, 4 banks,
// 8-bit line addresses, 32-bit data, swizzled bank map).
//
// Inputs are free every cycle except bank_read_valid. A tiny macro model
// gates it: a bank may return read data only for a read it accepted at an
// earlier edge since the last reset, and only once. So pre-reset replies are
// cancelled and each accepted read gets at most one reply, with arbitrary,
// independent per-bank latency >= 1 cycle. An abstract scoreboard tracks
// per-requester "read outstanding", per-bank lock and owner, and captured
// data, and checks the DUT every cycle after the first reset.
//
// Two contract points are left open on purpose and checked with bounds:
//   * whether a requester may be granted in the cycle its response is
//     consumed, and whether a bank may issue in the cycle its read data is
//     captured. Both are allowed, never required;
//   * whether a captured response is shown in the capture cycle
//     (combinational) or from the next cycle (registered). Both are allowed.
//     After that it must stay visible and stable until consumed.
//
// `PROBE_* defines turn one assertion into a reachability probe. Each probe
// must FAIL, which proves the scenario happens within the bound (no vacuity).
module sram_payload_tb (
    input wire clk,
    input wire rst,
    input wire [3:0] req_valid,
    input wire [3:0] req_write,
    input wire [31:0] req_line,
    input wire [127:0] req_wdata,
    input wire [15:0] req_wstrb,
    input wire [3:0] bank_ready,
    input wire [3:0] bank_read_valid_in,
    input wire [127:0] bank_rdata,
    input wire [3:0] resp_ready
);
    localparam integer R = 4, B = 4;
    wire [3:0] req_grant, bank_issue, bank_write, resp_valid;
    wire [23:0] bank_row;
    wire [127:0] bank_wdata, resp_rdata;
    wire [15:0] bank_wstrb;

    // ---------------- abstract state ----------------
    reg initialized = 0;
    reg [3:0] busy;            // requester has an accepted, unconsumed read
    reg [3:0] lock;            // bank has an accepted read awaiting data
    reg [1:0] owner [0:3];     // requester of the outstanding read per bank
    reg [3:0] pending;         // captured response waiting for the requester
    reg [31:0] pdata [0:3];
    reg [3:0] prev_stall;      // resp_valid && !resp_ready last cycle
    reg [31:0] prev_rdata [0:3];
    reg prev_beat0 = 0, seen_flush = 0;   // reachability-probe bookkeeping

    wire [3:0] bank_read_valid = bank_read_valid_in & lock & {4{!rst}};

    mercury_sram_payload #(
        .REQUESTERS(4), .BANKS(4), .LINE_ADDR_BITS(8), .DATA_BITS(32), .SWIZZLE(1)
    ) dut (
        .clk(clk), .rst(rst),
        .req_valid(req_valid), .req_write(req_write), .req_line(req_line),
        .req_wdata(req_wdata), .req_wstrb(req_wstrb), .req_grant(req_grant),
        .bank_ready(bank_ready), .bank_issue(bank_issue), .bank_write(bank_write),
        .bank_row(bank_row), .bank_wdata(bank_wdata), .bank_wstrb(bank_wstrb),
        .bank_read_valid(bank_read_valid), .bank_rdata(bank_rdata),
        .resp_valid(resp_valid), .resp_rdata(resp_rdata), .resp_ready(resp_ready)
    );

    // Independent reference bank map: bank = (l ^ l>>2 ^ l>>5)[1:0], row = l[7:2].
    function [1:0] bank_of(input [7:0] l);
        bank_of = l[1:0] ^ l[3:2] ^ l[6:5];
    endfunction

    // ---------------- combinational reference ----------------
    reg [3:0] target_hit [0:3];   // [bank][requester]
    reg [3:0] strict_ok, lenient_ok;  // requester may be granted (strict: not busy)
    reg [1:0] winner [0:3];
    reg [3:0] beat, capture_now;
    integer r, b;

    always @* begin
        beat = resp_valid & resp_ready;
        capture_now = bank_read_valid;
        for (r = 0; r < R; r = r + 1) begin
            strict_ok[r]  = req_valid[r] && !busy[r];
            lenient_ok[r] = req_valid[r] && (!busy[r] || beat[r]);
        end
        for (b = 0; b < B; b = b + 1) begin
            target_hit[b] = 0;
            winner[b] = 0;
            for (r = 0; r < R; r = r + 1) begin
                target_hit[b][r] = bank_of(req_line[r*8 +: 8]) == 2'(b);
                if (req_grant[r] && target_hit[b][r]) winner[b] = 2'(r);
            end
        end
    end

    // ---------------- checks ----------------
    reg [2:0] n, m;   // small counters keep the SAT problem narrow
    reg [1:0] k;
    integer cr, cb;   // checker-only loop variables (separate drivers from the reference block)
    always @* begin
        // Default assignments: keep the checker purely combinational (no inferred latches).
        n = 0; m = 0; k = 0; cb = 0; cr = 0;
        if (initialized) begin
            if (rst) assert(req_grant == 0 && bank_issue == 0 && resp_valid == 0);
            for (cr = 0; cr < R; cr = cr + 1) begin
                // A grant needs a valid request from a requester without an unconsumed read.
                if (req_grant[cr]) assert(lenient_ok[cr]);
                // A response belongs to an accepted read: captured earlier, or being captured now.
                if (resp_valid[cr]) begin
                    assert(busy[cr]);
                    if (pending[cr])
                        assert(resp_rdata[cr*32 +: 32] == pdata[cr]);
                    else begin
                        n = 0;
                        for (cb = 0; cb < B; cb = cb + 1)
                            if (capture_now[cb] && owner[cb] == 2'(cr)) begin
                                n = n + 3'd1;
                                assert(resp_rdata[cr*32 +: 32] == bank_rdata[cb*32 +: 32]);
                            end
                        assert(n == 3'd1);
                    end
                end
                // Captured data must be visible (registered at the latest) and held.
                if (!rst && pending[cr]) assert(resp_valid[cr]);
                if (!rst && prev_stall[cr]) begin
                    assert(resp_valid[cr]);
                    assert(resp_rdata[cr*32 +: 32] == prev_rdata[cr]);
                end
            end
            for (cb = 0; cb < B; cb = cb + 1) begin
                // Grants route to the bank they map to; one per bank; issue == some grant there.
                n = 0;
                for (cr = 0; cr < R; cr = cr + 1)
                    if (req_grant[cr] && target_hit[cb][cr]) n = n + 3'd1;
                assert(n <= 3'd1);
                assert(bank_issue[cb] == (n == 3'd1));
                if (bank_issue[cb]) begin
                    assert(!rst && bank_ready[cb]);
                    assert(!lock[cb] || capture_now[cb]);             // locked until capture
                    k = winner[cb];
                    assert(bank_row[cb*6 +: 6] == req_line[k*8+2 +: 6]);
                    assert(bank_write[cb] == req_write[k]);
                    if (req_write[k]) begin
                        assert(bank_wdata[cb*32 +: 32] == req_wdata[k*32 +: 32]);
                        assert(bank_wstrb[cb*4 +: 4] == req_wstrb[k*4 +: 4]);
                    end
                end
                // Work conserving: a ready, unlocked bank with a free, valid requester must issue.
                n = 0;
                for (cr = 0; cr < R; cr = cr + 1)
                    if (strict_ok[cr] && target_hit[cb][cr]) n = n + 3'd1;
                if (!rst && bank_ready[cb] && !lock[cb] && n != 3'd0) assert(bank_issue[cb]);
            end
            // Every grant shows up as exactly one bank issue.
            n = 0;
            for (cr = 0; cr < R; cr = cr + 1) if (req_grant[cr]) n = n + 3'd1;
            m = 0;
            for (cb = 0; cb < B; cb = cb + 1) if (bank_issue[cb]) m = m + 3'd1;
            assert(n == m);
`ifdef PROBE_TWO_BANKS
            assert(!(bank_issue[0] && bank_issue[1] && !bank_write[0] && bank_write[1]));
`endif
`ifdef PROBE_HELD_STALL
            assert(!(prev_stall[2]));
`endif
`ifdef PROBE_WRITE_AFTER_READ
            // Requester 0 is granted a write the cycle after its read was consumed.
            assert(!(req_grant[0] && req_write[0] && prev_beat0));
`endif
`ifdef PROBE_RESET_FLUSH
            // A reset arrives while a read is outstanding, then a new read completes.
            assert(!(seen_flush && beat[1]));
`endif
        end
    end

    integer j;
    always @(posedge clk) begin
        prev_beat0 <= !rst && beat[0];
        if (rst) begin
            initialized <= 1;
            seen_flush <= initialized && (|lock || |busy);
            busy <= 0; lock <= 0; pending <= 0; prev_stall <= 0;
        end else begin
            for (j = 0; j < R; j = j + 1) begin
                prev_stall[j] <= resp_valid[j] && !resp_ready[j];
                prev_rdata[j] <= resp_rdata[j*32 +: 32];
            end
            // Order: capture (unlocks), consume (frees the requester), issue (relocks).
            for (j = 0; j < B; j = j + 1)
                if (capture_now[j]) begin
                    lock[j] <= 0;
                    pending[owner[j]] <= 1;
                    pdata[owner[j]] <= bank_rdata[j*32 +: 32];
                end
            for (j = 0; j < R; j = j + 1)
                if (beat[j]) begin
                    pending[j] <= 0;
                    busy[j] <= 0;
                end
            // A read granted in the cycle the previous result is consumed stays outstanding.
            for (j = 0; j < B; j = j + 1)
                if (bank_issue[j] && !bank_write[j]) begin
                    lock[j] <= 1;
                    owner[j] <= winner[j];
                    busy[winner[j]] <= 1;
                end
        end
    end
endmodule
