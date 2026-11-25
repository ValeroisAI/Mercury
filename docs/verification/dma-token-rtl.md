# DMA token scoreboard prototype

The [SystemVerilog scoreboard](../../rtl/mercury_dma_tokens.sv) covers the lifetime of one DMA stream's asynchronous tokens. An issue reserves a slot. A matching completion marks that slot ready for acquisition. A matching wait frees a completed slot and increments its generation. This is the control path behind the proposed `m.dma.copy` and `m.dma.wait` semantics.

From the repository root, check it with Yosys:

```sh
python scripts/check_dma_rtl.py
```

For Windows, supply `--yosys build/rtl-env/Scripts/yowasp-yosys.exe` after creating the [project-local Yosys environment](./shared-sram-rtl.md).

| Event | Acceptance rule |
| --- | --- |
| Issue | `issue_valid && issue_ready` reserves the lowest currently free slot. |
| Complete | `complete_accepted` requires a live, matching, pending token. |
| Wait | `wait_ready` accepts a matching completed token. A matching pending token stalls. |
| Stale wait | `wait_stale` reports a free slot, invalid index, or generation mismatch. |

Each token packs the slot index in its low bits and the generation above it. The default four slots and three generation bits yield five-bit tokens. Completion and acquisition observe pre-edge state, and a slot acquired this cycle can first be reserved again next cycle. The caller must consume or discard old tokens before their generation wraps. Reset requires the producer to flush old completions.

The [formal harness](../../tests/rtl/dma_tokens_tb.sv) checks eight sequential steps from reset against an independent reference scoreboard. It checks free-slot allocation, full-queue backpressure, matching completion, pending and stale waits, and generation increments on acquisition. Recycling one slot through a complete generation wrap takes longer than the proof bound.

The DMA engine must publish destination data before it asserts completion. The scoreboard has no address ranges, payload storage, cache ordering, or clock-domain crossing logic. The RTL models token control for one stream, while the Python VM also checks memory overlap hazards.
