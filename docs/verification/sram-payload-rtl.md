# Banked SRAM payload interface

The [payload prototype](../../rtl/mercury_sram_payload.sv) connects the
[parallel bank arbiter](./shared-sram-rtl.md) to synchronous SRAM macro
ports. Defaults are eight requesters, 64 banks, and 512 bits per line.
Packed buses place port 0 in their low bits.

| Interface event | Contract |
| --- | --- |
| Request | Keep a request, address, write data, and byte mask stable until its `req_grant` strobe. |
| Macro acceptance | A bank with asserted `bank_ready` accepts on `bank_issue`. Derive readiness independently of the issue strobe. A write carries its byte mask and commits at that edge under the macro contract. |
| Macro read reply | Return exactly one `bank_read_valid` at least one cycle after accepting a read. The bridge always captures a reply to a pending read. Derive reply validity independently of the current `bank_issue`. |
| Requester response | The bridge holds `resp_valid` and the read data until `resp_ready`. The requester can accept its response and issue its next request at the same edge. Derive `resp_ready` independently of the current `req_grant`. |
| Reset | The bridge drops outstanding work. The macro adaptor must cancel or drain old replies before admitting new work. |

A ready bank can accept its next request at the edge that captures the old
read reply. A requester can replace its outstanding read at the edge that
consumes the held response. The bank and requester each still track one
read at a time. A one-cycle macro can sustain one read per cycle on one
bank with at least two continuously active requesters and no backpressure.
An individual requester can issue one read every two cycles. Both turnover
signals feed arbitration combinationally and require timing validation in
the target implementation.

```sh
python scripts/check_sram_payload_directed_rtl.py
python scripts/check_sram_payload_rtl.py
python scripts/check_sram_full_load.py
```

The 12-step check uses four requesters, four banks and 32-bit data. It
checks simultaneous bank issue, byte masks, two independent reply
latencies, a stalled response, bank refill on reply, requester refill on
consumption, and reset
with a read in flight. Reachability probes exercise the key states in
the same sequence. The full-size module passed Yosys structural checks.

The second command checks arbitrary traffic for six steps from reset on
the reduced configuration. Its macro model can choose read reply data
and timing independently per bank after acceptance. It enforces the
one-reply and reset contract. Five reachability probes cover concurrent
bank issue, response stall, both turnover paths, and reset.

## Full-load simulation

The third command uses Icarus Verilog (`iverilog` and `vvp` on `PATH`) to
run the default eight-requester, 64-bank, 512-bit RTL against an independent
round-robin and SRAM scoreboard. In each fixed case all eight requesters
offer reads for 8,192 cycles. Each macro replies in one cycle. The wait
column counts consecutive cycles a requester offers a request without a
grant, including time while its previous read remains outstanding.

| Target banks | Bank ready | Reads accepted | Reads/cycle | Max cycles without a grant |
| ---: | --- | ---: | ---: | ---: |
| 8 | Every cycle | 32,768 | 4 | 1 |
| 4 | Every cycle | 32,768 | 4 | 1 |
| 2 | Every cycle | 16,384 | 2 | 3 |
| 1 | Every cycle | 8,192 | 1 | 7 |
| 1 | Every fourth cycle | 2,048 | 0.25 | 31 |

Three additional seeds exercise mixed masked writes and reads, independent
bank and response stalls, one- to five-cycle macro latency, and reset with
reads outstanding. The simulation checks arbitration order, address and
payload routing, data stability under stalls, and completion after drain.

The next integration gates are SRAM macros, row bounds, ECC, clock-domain
crossings, and a pipelined payload path. The DMA completion scoreboard
must observe final write visibility at the macro interface.
