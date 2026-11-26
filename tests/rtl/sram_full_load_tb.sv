`timescale 1ns/1ps

module sram_full_load_tb;
    localparam integer R = 8, B = 64, A = 22, D = 512, S = D/8, ROW = A-6;
    localparam integer CYCLES = 8192, TEST_ROWS = 16;
    reg clk = 0, rst;
    reg [R-1:0] req_valid, req_write, resp_ready;
    reg [R*A-1:0] req_line;
    reg [R*D-1:0] req_wdata;
    reg [R*S-1:0] req_wstrb;
    reg [B-1:0] bank_ready, bank_read_valid;
    reg [B*D-1:0] bank_rdata;
    wire [R-1:0] req_grant, resp_valid;
    wire [R*D-1:0] resp_rdata;
    wire [B-1:0] bank_issue, bank_write;
    wire [B*ROW-1:0] bank_row;
    wire [B*D-1:0] bank_wdata;
    wire [B*S-1:0] bank_wstrb;

    mercury_sram_payload dut (.*);

    reg [D-1:0] contents [0:B-1][0:TEST_ROWS-1];
    reg [D-1:0] reply [0:B-1];
    reg [D-1:0] held [0:R-1];
    reg [R-1:0] busy, waiting, previous_grant;
    integer pointer [0:B-1], due [0:B-1], owner [0:B-1];
    integer grants [0:R-1], wait_cycles [0:R-1];
    integer total, longest_wait, bank_refills, requester_refills, reset_reads;
    reg [31:0] state;

    function automatic [31:0] draw;
        state = state ^ (state << 13);
        state = state ^ (state >> 17);
        state = state ^ (state << 5);
        draw = state;
    endfunction

    function automatic integer bank_of(input reg [A-1:0] address);
        bank_of = int'(address[5:0] ^ address[11:6] ^ address[18:13]);
    endfunction

    task automatic tick;
        #1; clk = 1;
        #1; clk = 0;
    endtask

    task automatic clear_scoreboard;
        busy = 0;
        waiting = 0;
        previous_grant = 0;
        for (integer b = 0; b < B; b++) begin
            due[b] = -1;
            pointer[b] = 0;
        end
    endtask

    task automatic run_case(input integer target_banks, input integer ready_period,
                            input reg [31:0] seed, input integer expected_total);
        reg [R-1:0] expected_grant, consumed;
        integer selected [0:B-1];
        integer row, bank, requester;
        state = seed;
        total = 0;
        longest_wait = 0;
        bank_refills = 0;
        requester_refills = 0;
        reset_reads = 0;
        req_valid = 0;
        for (integer r = 0; r < R; r++) begin
            grants[r] = 0;
            wait_cycles[r] = 0;
        end
        for (integer b = 0; b < B; b++)
            for (integer i = 0; i < TEST_ROWS; i++)
                contents[b][i] = {16{32'(b * TEST_ROWS + i)}};

        for (integer cycle = -1; cycle < CYCLES+32; cycle++) begin
            rst = cycle == -1 || (seed != 0 && cycle == CYCLES/2);
            resp_ready = '1;
            bank_ready = '1;
            bank_read_valid = 0;
            bank_rdata = 0;
            if (rst) begin
                if (cycle >= 0) reset_reads = $countones(busy);
                #1;
                if ({req_grant, bank_issue, resp_valid} !== 0) $fatal(1, "reset outputs");
                clear_scoreboard();
                req_valid = 0;
                for (integer r = 0; r < R; r++) wait_cycles[r] = 0;
                tick();
            end else begin
                for (integer b = 0; b < B; b++) begin
                    bank_ready[b] = seed == 0 ? cycle % ready_period == 0 : draw() % 5 != 0;
                    if (cycle >= CYCLES) bank_ready[b] = 1;
                    bank_read_valid[b] = due[b] == cycle;
                    if (bank_read_valid[b]) bank_rdata[b*D +: D] = reply[b];
                end
                for (integer r = 0; r < R; r++) begin
                    resp_ready[r] = seed == 0 || cycle >= CYCLES || draw() % 4 != 0;
                    // Keep the entire pending request stable until accepted.
                    if (!req_valid[r] || previous_grant[r]) begin
                        req_valid[r] = cycle < CYCLES;
                        bank = seed == 0 ? r % target_banks : int'(draw() % target_banks);
                        row = seed == 0 ? grants[r] % TEST_ROWS : int'(draw() % TEST_ROWS);
                        req_line[r*A +: A] = A'((row << 6) | (bank ^ row));
                        req_write[r] = seed != 0 && draw() % 4 == 0;
                        for (integer word_index = 0; word_index < D/32; word_index++)
                            req_wdata[r*D + word_index*32 +: 32] = draw();
                        req_wstrb[r*S +: S] = {draw(), draw()};
                    end
                end
                #1;
                if (resp_valid !== waiting) $fatal(1, "response valid at cycle %0d", cycle);
                for (integer r = 0; r < R; r++)
                    if (waiting[r] && resp_rdata[r*D +: D] !== held[r])
                        $fatal(1, "response data at cycle %0d requester %0d", cycle, r);
                consumed = waiting & resp_ready;
                expected_grant = 0;
                for (integer b = 0; b < B; b++) begin
                    selected[b] = -1;
                    if (bank_ready[b] && (due[b] == -1 || bank_read_valid[b]))
                        for (integer offset = 0; offset < R; offset++) begin
                            requester = (pointer[b] + offset) % R;
                            if (selected[b] == -1 && req_valid[requester] &&
                                (!busy[requester] || consumed[requester]) &&
                                bank_of(req_line[requester*A +: A]) == b)
                                selected[b] = requester;
                        end
                    requester = selected[b];
                    if (bank_issue[b] !== (requester >= 0))
                        $fatal(1, "issue at cycle %0d bank %0d", cycle, b);
                    if (requester >= 0) begin
                        expected_grant[requester] = 1;
                        row = int'(req_line[requester*A+6 +: ROW]);
                        if (bank_row[b*ROW +: ROW] !== ROW'(row) || bank_write[b] !== req_write[requester])
                            $fatal(1, "command at cycle %0d bank %0d", cycle, b);
                        if (req_write[requester] && (bank_wdata[b*D +: D] !== req_wdata[requester*D +: D] ||
                            bank_wstrb[b*S +: S] !== req_wstrb[requester*S +: S]))
                            $fatal(1, "write payload at cycle %0d bank %0d", cycle, b);
                    end
                end
                if (req_grant !== expected_grant) $fatal(1, "grant at cycle %0d", cycle);
                previous_grant = expected_grant;

                waiting = waiting & ~consumed;
                busy = busy & ~consumed;
                bank_refills += $countones(bank_issue & bank_read_valid);
                requester_refills += $countones(expected_grant & consumed);
                for (integer b = 0; b < B; b++) begin
                    if (bank_read_valid[b]) begin
                        if (!busy[owner[b]] || waiting[owner[b]]) $fatal(1, "reply ownership");
                        waiting[owner[b]] = 1;
                        held[owner[b]] = reply[b];
                        due[b] = -1;
                    end
                    requester = selected[b];
                    if (requester >= 0) begin
                        row = int'(req_line[requester*A+6 +: ROW]);
                        pointer[b] = (requester + 1) % R;
                        if (req_write[requester]) begin
                            for (integer octet = 0; octet < S; octet++)
                                if (req_wstrb[requester*S + octet])
                                    contents[b][row][octet*8 +: 8] = req_wdata[requester*D + octet*8 +: 8];
                        end else begin
                            busy[requester] = 1;
                            owner[b] = requester;
                            due[b] = cycle + (seed == 0 ? 1 : 1 + int'(draw() % 5));
                            reply[b] = contents[b][row];
                        end
                    end
                end
                if (cycle < CYCLES)
                    for (integer r = 0; r < R; r++) begin
                        if (expected_grant[r]) begin
                            total++;
                            grants[r]++;
                            wait_cycles[r] = 0;
                        end else begin
                            wait_cycles[r]++;
                            if (wait_cycles[r] > longest_wait) longest_wait = wait_cycles[r];
                        end
                    end
                tick();
            end
        end
        if (busy != 0 || waiting != 0 || req_valid != 0) $fatal(1, "scoreboard failed to drain");
        if (seed == 0) begin
            if (total != expected_total) $fatal(1, "expected %0d issues, got %0d", expected_total, total);
            if (longest_wait != (target_banks == R ? 1 : (R/target_banks)*ready_period-1))
                $fatal(1, "unexpected maximum wait: %0d", longest_wait);
            for (integer r = 0; r < R; r++)
                if (grants[r] != total/R) $fatal(1, "unfair fixed workload at requester %0d", r);
        end else if (bank_refills == 0 || requester_refills == 0 || reset_reads == 0) begin
            $fatal(1, "seed %0d missed turnover or reset coverage", seed);
        end
        $display("banks=%0d period=%0d seed=%0d cycles=%0d issues=%0d max_ungranted=%0d bank_refills=%0d requester_refills=%0d reset_reads=%0d",
                 target_banks, ready_period, seed, CYCLES, total, longest_wait, bank_refills, requester_refills, reset_reads);
    endtask

    initial begin
        run_case(8, 1, 0, 32768);
        run_case(4, 1, 0, 32768);
        run_case(2, 1, 0, 16384);
        run_case(1, 1, 0, 8192);
        run_case(1, 4, 0, 2048);
        run_case(2, 1, 781, 0);
        run_case(8, 1, 31415, 0);
        run_case(64, 1, 271828, 0);
        $display("PASS: 8 full-size SRAM load scenarios");
        $finish;
    end
endmodule
