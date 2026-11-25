# Banked SRAM payload interface

The [payload prototype](../../rtl/mercury_sram_payload.sv) connects the
[parallel bank arbiter](./shared-sram-rtl.md) to synchronous SRAM macro
ports. Defaults are eight requesters, 64 banks, and 512 bits per line.
Packed buses place port 0 in their low bits.

| Interface event | Contract |
| --- | --- |
| Request | Keep a request, address, write data, and byte mask stable until its `req_grant` strobe. |
| Macro acceptance | A bank with asserted `bank_ready` accepts on `bank_issue`. Derive readiness independently of the issue strobe. A write carries its byte mask and commits at that edge under the macro contract. |
| Macro read reply | Return exactly one `bank_read_valid` at least one cycle after accepting a read. The bridge always captures a reply to a pending read. |
| Requester response | The bridge holds `resp_valid` and the read data until `resp_ready`. The requester can send its next request after consumption. |
| Reset | The bridge drops outstanding work. The macro adaptor must cancel or drain old replies before admitting new work. |

A bank can serve another requester after its reply is captured. A requester
remains busy until its read result is consumed, even if its previous bank
is free. Keeping one read outstanding per bank and per requester makes
return routing one-hot and avoids reordering. For a one-cycle macro, a bank
can accept its next request two cycles after accepting a read. Consecutive
writes to a ready bank can be accepted on consecutive cycles.

```sh
python scripts/check_sram_payload_directed_rtl.py
python scripts/check_sram_payload_rtl.py
```

The 12-step check uses four requesters, four banks and 32-bit data. It
checks simultaneous bank issue, byte masks, two independent reply
latencies, a stalled response, write retry after consumption, and reset
with a read in flight. Reachability probes exercise the key states in
the same sequence. The full-size module passed Yosys structural checks.

The second command checks arbitrary traffic for six steps from reset on
the reduced configuration. Its macro model can choose read reply data
and timing independently per bank after acceptance. It enforces the
one-reply and reset contract. Four reachability probes check the
two-bank, stall, retry and reset scenarios.

The next integration gates are SRAM macros, row bounds, ECC, clock-domain
crossings, and a pipelined payload path. The DMA completion scoreboard
must observe final write visibility at the macro interface.
