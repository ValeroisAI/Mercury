# Interleaved FP8 accumulation

The [interleaved prototype](../../rtl/mercury_stme_interleaved_sum.sv) uses a fixed four-stage binary32 adder and eight independent partial-sum contexts. An input carries a context ID. A context waits for its preceding add to write back before its next term can issue. The controller keeps a separate group counter and completed result for each context, with a default of 128 terms per group.

```sh
python scripts/check_stme_interleaved_rtl.py
```

The [bounded harness](../../tests/rtl/stme_interleaved_tb.sv) accepts 16 inputs on consecutive cycles over two rounds of eight contexts. It then checks all results independently, including a pending-add retry and an input blocked by a stalled output. The result consumer keeps its context selection stable while stalled.

Interleaving independent dot products fills the adder stages while maintaining the original rounding order within each dot. It does not change the accumulation dependency of a single context. The front-end FP8 product and context multiplexers still contribute to the cycle path. No result here establishes ASIC clock frequency or the throughput of a full STME array.
