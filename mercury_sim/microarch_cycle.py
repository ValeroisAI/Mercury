"""Cycle-level architectural model of a Mercury-1 SCC and its die memory controller.

This is a Python cycle-level architectural model, not a cycle-accurate
simulator. The term "cycle-accurate" is kept for future RTL (SystemVerilog
or Chisel run under Verilator). Latencies and structures are architectural
assumptions; they have not been validated against silicon or RTL.

One SCC runs an FP8 ``m.mma.fp8.blk`` GEMM tile loop. The default
output tile is 128 x 128, K moves in groups of 128, and 16 warps each own
a 32 x 32 sub-tile (four 16 x 16 STME accumulators). ``tile_m=256`` gives
each warp 64 x 32 (eight accumulators). ``multicast_a`` reads each A-operand
line once from the Shared SRAM banks and fans it out over a NoC tree to the
19 SCCs in the same die grid row. The model includes:

* Two RV64 warp schedulers (8 warps each). Each issues at most one
  instruction per cycle, using greedy-then-oldest selection. An explicit
  scoreboard tracks RAW, WAW and WAR on fragment registers, DMA tokens,
  STME accumulator RAW and STME queue structural hazards.
* Eight 16 x 16 STME wavefront pipelines. Each has a 16-cycle fill when
  it starts from idle, 64 FP8 streaming cycles per K=128 group, a
  16-cycle drain, and a 2-cycle ``fma_F32_RNE`` scale epilogue at every
  K=128 boundary.
* The 64-bank Shared Tensor SRAM crossbar. All contending SCCs' line
  addresses go through the bank-mapping function. The hottest bank limits
  die line throughput, and each SCC gets a fair share. Unswizzled
  stride-64 tiles put every SCC on two banks. XOR swizzling spreads
  them across all 64.
* A 12-channel GDDR7 controller (32-bit channels, 16 banks, 2 KiB rows).
  It models row-buffer hits and misses (tRP+tRCD), staggered all-bank
  refresh every tREFI with tRFC blocking (tREFI is halved at >= 85 C or
  in ``hot_2x_trefi`` mode), and double-buffered K-panel back-pressure.
* An optional UCIe-S die-to-die spill on ``MERCURY_1D``. A fraction of
  lines crosses the link at a fair-share bandwidth plus a hop latency.

Timing is event-skipping: when no scheduler can issue, time jumps to the
earliest scoreboard wake-up. Skipped stall cycles are still counted, so
skipping does not change the model's cycle counts.

Assumed DRAM timings (JEDEC-class, not vendor-qualified): tRCD+tRP 30 ns,
tREFI 1.9 us, tRFC 160 ns, 20 ns controller pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any

from .specs import MERCURY_1D, MercurySKU


WARPS_PER_SCC = 16
SCHEDULERS = 2
TILE = 128                     # SCC output tile edge and K-group length
STME_DIM = 16
STME_FILL_DRAIN = 16
STME_EPILOGUE = 2              # K=128 boundary fma_F32_RNE scale/accumulate
STME_QUEUE_DEPTH = 2
LDS_LATENCY = 6
STORE_LATENCY = 4
XBAR_LATENCY = 12
UCIE_HOP_CYCLES = 40           # ~21.6 ns PHY + adapter + far-die crossbar
LINE = 64
DMA_PORT_LINES_PER_CYCLE = 2   # 128 B/clk L1 fill port
GDDR7_CHANNEL_BITS = 32
DRAM_CHUNK = 256
DRAM_BANKS = 16
DRAM_ROW_BYTES = 2048
T_RCD_RP_NS = 30.0
T_REFI_NS = 1900.0
T_RFC_NS = 160.0
DRAM_PIPE_NS = 20.0
GRID_M = 4                     # SCC grid rows per die (76 SCCs -> 4 x 19)
NOC_HOP_CYCLES = 2             # per multicast-tree level
WARP_GRID = 4                  # 4 x 4 warps cover the SCC output tile
TILE_M_OPTIONS = (128, 256)
PANEL_BUFFERS = 3              # K-panel ring depth in Shared SRAM

HAZARD_VARIANTS = (None, "dma_before_wait")
STALL_REASONS = ("raw", "waw", "war", "dma_token", "dram_fill", "stme_acc_raw", "stme_structural")


class ScoreboardHazardError(RuntimeError):
    """A warp consumed a DMA buffer before waiting on its token (strict mode)."""


@dataclass(frozen=True)
class CycleTraceResult:
    sku: str
    k_groups: int
    swizzle: bool
    hot_2x_trefi: bool
    ucie_d2d_fraction: float
    total_cycles: int
    stme_active_cycles: float
    scoreboard_stall_cycles: int
    bank_conflict_cycles: float
    dram_stall_cycles: float
    achieved_tflops: float
    achieved_sram_tbs: float
    achieved_dram_tbs: float
    dual_issue_ipc: float
    swizzle_speedup: float
    instructions_issued: int
    stme_utilization: float
    effective_sram_banks: float
    refresh_events: int
    refresh_blocked_cycles: float
    dram_row_hit_rate: float
    hazards_detected: int
    stall_breakdown: tuple[tuple[str, int], ...]
    tile_m: int = TILE
    tile_n: int = TILE
    multicast_a: bool = False
    l1_fill_tbs: float = 0.0
    sram_bytes_per_clk_per_scc: float = 0.0
    rf_accumulator_kib: float = 0.0
    multicast_fanout: int = 1
    multicast_tree_latency_cycles: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stall_breakdown"] = dict(self.stall_breakdown)
        return data


# --------------------------------------------------------------------------
# Shared Tensor SRAM crossbar bank mapping
# --------------------------------------------------------------------------

def sram_bank(line_addr: int, banks: int, swizzle: bool) -> int:
    """Map a 64 B line address to a Shared SRAM bank (XOR-fold when swizzled)."""
    if not swizzle:
        return line_addr % banks
    shift = banks.bit_length() - 1
    return (line_addr ^ (line_addr >> shift) ^ (line_addr >> (2 * shift + 1))) % banks


def _tile_line_addresses(scc: int, banks: int) -> list[int]:
    """A then B operand tile of one SCC: 128 rows x 128 B at a stride-64-line pitch."""
    lines_per_row = TILE // LINE
    addrs = []
    for operand in (0, 1):
        base = (scc * 2 + operand) * banks * TILE
        for row in range(TILE):
            for col in range(lines_per_row):
                addrs.append(base + row * banks + col)
    return addrs


@lru_cache(maxsize=None)
def effective_sram_banks(contending_sccs: int, banks: int, swizzle: bool) -> float:
    """Die-wide line throughput per cycle: total lines / hottest-bank load."""
    load = [0] * banks
    total = 0
    for scc in range(contending_sccs):
        for addr in _tile_line_addresses(scc, banks):
            load[sram_bank(addr, banks, swizzle)] += 1
            total += 1
    return total / max(load)


# --------------------------------------------------------------------------
# GDDR7 controller
# --------------------------------------------------------------------------

class _GDDR7Controller:
    def __init__(self, sku: MercurySKU, clock_ghz: float, hot: bool) -> None:
        self.channels = sku.gddr7_bus_bits_per_die // GDDR7_CHANNEL_BITS
        bytes_per_cycle = GDDR7_CHANNEL_BITS * sku.gddr7_gbps_per_pin / 8.0 / clock_ghz
        self.chunk_cycles = DRAM_CHUNK / bytes_per_cycle
        self.act_cycles = T_RCD_RP_NS * clock_ghz
        self.trefi = T_REFI_NS * clock_ghz * (0.5 if hot else 1.0)
        self.trfc = T_RFC_NS * clock_ghz
        self.pipe = DRAM_PIPE_NS * clock_ghz
        self.bus_free = [0.0] * self.channels
        self.open_row = [[-1] * DRAM_BANKS for _ in range(self.channels)]
        self.bank_ready = [[0.0] * DRAM_BANKS for _ in range(self.channels)]
        self.last_refresh = [-1] * self.channels
        self.chunk_index = 0
        self.hits = 0
        self.misses = 0
        self.refresh_blocked = 0.0
        self.refresh_windows: set[tuple[int, int]] = set()

    def _refresh_push(self, ch: int, t: float) -> tuple[float, bool]:
        """Delay a transfer that overlaps channel ``ch``'s refresh window."""
        offset = ch * self.trefi / self.channels
        n = math.floor((t - offset) / self.trefi)
        for k in (n, n + 1):
            if k < 0:
                continue
            start = offset + k * self.trefi
            end = start + self.trfc
            if t < end and t + self.chunk_cycles > start:
                self.refresh_windows.add((ch, k))
                self.refresh_blocked += end - t
                return end, True
        return t, False

    def fetch(self, start: float, nbytes: int) -> float:
        """Stream ``nbytes`` from ``start``; return the cycle the last byte lands."""
        chunks = math.ceil(nbytes / DRAM_CHUNK)
        chunks_per_row = DRAM_ROW_BYTES // DRAM_CHUNK
        done = start
        for _ in range(chunks):
            idx = self.chunk_index
            self.chunk_index += 1
            ch = idx % self.channels
            local = idx // self.channels
            bank = (local // chunks_per_row) % DRAM_BANKS
            row = local // (chunks_per_row * DRAM_BANKS)
            t = max(start, self.bus_free[ch])
            t, refreshed = self._refresh_push(ch, t)
            if refreshed:  # all-bank refresh precharges every bank
                self.open_row[ch] = [-1] * DRAM_BANKS
                self.bank_ready[ch] = [max(r, t) for r in self.bank_ready[ch]]
            if self.open_row[ch][bank] == row:
                self.hits += 1
            else:
                self.misses += 1
                act = max(start, self.bank_ready[ch][bank])
                t = max(t, act + self.act_cycles)
                self.open_row[ch][bank] = row
            end = t + self.chunk_cycles
            self.bus_free[ch] = end
            self.bank_ready[ch][bank] = max(self.bank_ready[ch][bank], end)
            done = max(done, end)
        return done + self.pipe


# --------------------------------------------------------------------------
# Warp program and scoreboard
# --------------------------------------------------------------------------

def build_warp_program(k_groups: int, inject_hazard: str | None = None,
                       acc_rows: int = 2, acc_cols: int = 2) -> list[tuple]:
    """Double-buffered DMA -> LDS -> (acc_rows x acc_cols) MMAs per K group, then stores."""
    if inject_hazard not in HAZARD_VARIANTS:
        raise ValueError(f"Unknown hazard variant: {inject_hazard}")
    rows, cols = range(acc_rows), range(acc_cols)
    prog: list[tuple] = [("DMA", 0)]
    for g in range(k_groups):
        if inject_hazard != "dma_before_wait":
            prog.append(("WAIT", g))
        for reg in [f"a{i}" for i in rows] + [f"b{j}" for j in cols]:
            prog.append(("LDS", reg, g))
        for i in rows:
            for j in cols:
                prog.append(("MMA", f"acc{i}{j}", f"a{i}", f"b{j}"))
        if g + 1 < k_groups:  # prefetch next K group while the STMEs stream
            prog.append(("DMA", g + 1))
    for i in rows:
        for j in cols:
            prog.append(("STORE", f"acc{i}{j}"))
    return prog


class _Warp:
    __slots__ = ("wid", "prog", "pc", "reg_ready", "reg_read_until", "acc_ready",
                 "tokens", "waited", "flagged")

    def __init__(self, wid: int, prog: list[tuple]) -> None:
        self.wid = wid
        self.prog = prog
        self.pc = 0
        self.reg_ready: dict[str, float] = {}
        self.reg_read_until: dict[str, float] = {}
        self.acc_ready: dict[str, float] = {}
        self.tokens: dict[int, float] = {}
        self.waited: set[int] = set()
        self.flagged: set[int] = set()

    @property
    def done(self) -> bool:
        return self.pc >= len(self.prog)


class _SCCSim:
    def __init__(self, sku: MercurySKU, k_groups: int, swizzle: bool, hot: bool,
                 ucie_fraction: float, contending: int, inject_hazard: str | None,
                 strict: bool, ideal_dram: bool = False, tile_m: int = TILE,
                 multicast_a: bool = False) -> None:
        self.sku = sku
        self.tile_m = tile_m
        self.multicast_a = multicast_a
        self.ideal_dram = ideal_dram
        self.k_groups = k_groups
        self.clock = sku.nominal_clock_ghz
        self.strict = strict
        self.ucie_fraction = ucie_fraction
        banks = sku.shared_sram_banks_per_die
        self.eff_banks = effective_sram_banks(contending, banks, swizzle)
        ideal_banks = effective_sram_banks(contending, banks, True)
        grid_n = sku.active_sccs_per_die // GRID_M
        # A-operand lines: with multicast, one bank read serves the whole grid row.
        self.mcast_fanout = min(grid_n, contending) if multicast_a else 1
        self.mcast_latency = (NOC_HOP_CYCLES * math.ceil(math.log2(self.mcast_fanout))
                              if self.mcast_fanout > 1 else 0)
        port = 1.0 / DMA_PORT_LINES_PER_CYCLE

        def service(banks_avail: float, cost: float) -> float:
            return max(port, cost * contending / banks_avail)

        self.service_b = service(self.eff_banks, 1.0)
        self.service_a = service(self.eff_banks, 1.0 / self.mcast_fanout)
        self.ideal_b = service(ideal_banks, 1.0)
        self.ideal_a = service(ideal_banks, 1.0 / self.mcast_fanout)
        if ucie_fraction:
            link_bytes_per_cycle = sku.ucie_s_bidir_tbs * 1000.0 / 2.0 / self.clock
            self.link_service = LINE / (link_bytes_per_cycle / contending)
        else:
            self.link_service = 0.0
        self.dram = _GDDR7Controller(sku, self.clock, hot)
        self.panel_bytes = (GRID_M * tile_m + grid_n * TILE) * TILE  # FP8 A + B panels
        self.a_lines_per_dma = tile_m * TILE // LINE // WARPS_PER_SCC
        self.b_lines_per_dma = TILE * TILE // LINE // WARPS_PER_SCC
        self.lines_per_dma = self.a_lines_per_dma + self.b_lines_per_dma
        self.l1_bytes = 0
        stmes = sku.systolic_arrays_per_scc
        self.stme_free = [0.0] * stmes
        self.stme_starts: list[list[float]] = [[] for _ in range(stmes)]
        fp8_macs_per_stme = 2 * STME_DIM * STME_DIM
        self.stream_cycles = TILE * STME_DIM * STME_DIM // fp8_macs_per_stme
        self.port_free = 0.0
        self.link_free = 0.0
        self.line_counter = 0
        self.panels: list[float] = []
        self.first_req: dict[int, float] = {}
        self.group_done: dict[int, float] = {}
        self.dma_issued: dict[int, int] = {}
        self.sram_bytes = 0
        self.ucie_bytes = 0
        self.bank_conflict = 0.0
        self.stme_stream_total = 0.0
        self.hazards = 0
        self.final = 0.0
        prog = build_warp_program(k_groups, inject_hazard,
                                  acc_rows=tile_m // WARP_GRID // STME_DIM,
                                  acc_cols=TILE // WARP_GRID // STME_DIM)
        self.warps = [_Warp(w, prog) for w in range(WARPS_PER_SCC)]

    # -- memory system ------------------------------------------------------
    def _panel_ready(self, g: int, now: float) -> float:
        while len(self.panels) <= g:
            p = len(self.panels)
            if self.ideal_dram:
                self.panels.append(0.0)
                continue
            old = p - PANEL_BUFFERS
            if old < 0:
                start = 0.0
            else:  # ring buffer: wait for group `old` to leave Shared SRAM
                start = self.group_done.get(old, now)
                if self.dma_issued.get(old, 0) < WARPS_PER_SCC:
                    start = max(start, now)
            self.panels.append(self.dram.fetch(start, self.panel_bytes))
        return self.panels[g]

    def _issue_dma(self, warp: _Warp, g: int, t: float) -> None:
        self.dma_issued[g] = self.dma_issued.get(g, 0) + 1
        avail = t
        token = avail
        f = self.ucie_fraction
        for i in range(self.lines_per_dma):
            idx = self.line_counter
            self.line_counter += 1
            is_a = i < self.a_lines_per_dma
            self.l1_bytes += LINE
            if f and math.floor((idx + 1) * f) > math.floor(idx * f):
                grant = max(avail, self.link_free)
                self.link_free = grant + self.link_service
                done = self.link_free + UCIE_HOP_CYCLES + XBAR_LATENCY
                self.ucie_bytes += LINE
            else:
                svc, ideal = (self.service_a, self.ideal_a) if is_a else (self.service_b, self.ideal_b)
                grant = max(avail, self.port_free)
                self.port_free = grant + svc
                done = self.port_free + XBAR_LATENCY + (self.mcast_latency if is_a else 0)
                self.bank_conflict += svc - ideal
                # Bank-side bytes: a multicast A line costs 1/fanout of a bank read per SCC.
                self.sram_bytes += LINE / self.mcast_fanout if is_a else LINE
            token = max(token, done)
        warp.tokens[g] = token
        self.group_done[g] = max(self.group_done.get(g, 0.0), token)

    # -- scoreboard ---------------------------------------------------------
    def _check(self, warp: _Warp, t: float) -> tuple[bool, float, str]:
        inst = warp.prog[warp.pc]
        op = inst[0]
        if op == "DMA":  # descriptor waits on the K-panel's GDDR7 fill semaphore
            g = inst[1]
            self.first_req[g] = min(self.first_req.get(g, math.inf), t)
            ready = self._panel_ready(g, t)
            return (True, t, "") if ready <= t else (False, ready, "dram_fill")
        if op == "WAIT":
            tok = warp.tokens[inst[1]]
            return (True, t, "") if tok <= t else (False, tok, "dma_token")
        if op == "LDS":
            reg, g = inst[1], inst[2]
            if g not in warp.waited and g not in warp.flagged:
                warp.flagged.add(g)
                self.hazards += 1
                if self.strict:
                    raise ScoreboardHazardError(
                        f"warp {warp.wid}: LDS {reg} reads DMA buffer of K-group {g} before WAIT")
            tok = warp.tokens[g]
            if tok > t:
                return False, tok, "dma_token"
            if warp.reg_ready.get(reg, 0.0) > t:
                return False, warp.reg_ready[reg], "waw"
            if warp.reg_read_until.get(reg, 0.0) > t:
                return False, warp.reg_read_until[reg], "war"
            return True, t, ""
        if op == "MMA":
            acc, ra, rb = inst[1], inst[2], inst[3]
            src = max(warp.reg_ready.get(ra, 0.0), warp.reg_ready.get(rb, 0.0))
            if src > t:
                return False, src, "raw"
            if warp.acc_ready.get(acc, 0.0) > t:
                return False, warp.acc_ready[acc], "stme_acc_raw"
            s = warp.wid % len(self.stme_free)
            pending = [x for x in self.stme_starts[s] if x > t]
            self.stme_starts[s] = pending
            if len(pending) >= STME_QUEUE_DEPTH:
                return False, min(pending), "stme_structural"
            return True, t, ""
        if op == "STORE":
            ready = warp.acc_ready.get(inst[1], 0.0)
            return (True, t, "") if ready <= t else (False, ready, "stme_acc_raw")
        raise ValueError(f"Unknown op {op}")

    def _issue(self, warp: _Warp, t: float) -> None:
        inst = warp.prog[warp.pc]
        op = inst[0]
        if op == "DMA":
            self._issue_dma(warp, inst[1], t)
        elif op == "WAIT":
            warp.waited.add(inst[1])
        elif op == "LDS":
            warp.reg_ready[inst[1]] = t + LDS_LATENCY
        elif op == "MMA":
            acc, ra, rb = inst[1], inst[2], inst[3]
            s = warp.wid % len(self.stme_free)
            start = max(t, self.stme_free[s])
            stream_start = start + (STME_FILL_DRAIN if self.stme_free[s] < t else 0)
            stream_end = stream_start + self.stream_cycles
            self.stme_free[s] = stream_end + STME_EPILOGUE
            self.stme_starts[s].append(stream_start)
            self.stme_stream_total += self.stream_cycles
            warp.acc_ready[acc] = stream_end + STME_FILL_DRAIN + STME_EPILOGUE
            for reg in (ra, rb):
                warp.reg_read_until[reg] = max(warp.reg_read_until.get(reg, 0.0), stream_end)
        elif op == "STORE":
            self.final = max(self.final, t + STORE_LATENCY)
        warp.pc += 1

    # -- main loop ----------------------------------------------------------
    def run(self) -> dict[str, Any]:
        groups = [[w for w in self.warps if w.wid % SCHEDULERS == s] for s in range(SCHEDULERS)]
        last = [None] * SCHEDULERS
        stalls = dict.fromkeys(STALL_REASONS, 0)
        issued = 0
        t = 0
        while any(not w.done for w in self.warps):
            wake = math.inf
            stalled: list[str] = []
            any_issue = False
            for s, members in enumerate(groups):
                live = [w for w in members if not w.done]
                if not live:
                    continue
                order = sorted(live, key=lambda w: (w is not last[s], w.wid))
                reason = ""
                for w in order:
                    ok, wk, why = self._check(w, t)
                    if ok:
                        self._issue(w, t)
                        last[s] = w
                        issued += 1
                        any_issue = True
                        break
                    wake = min(wake, wk)
                    reason = reason or why
                else:
                    stalled.append(reason)
            step = 1 if any_issue else max(1, math.ceil(wake - t))
            for why in stalled:
                stalls[why] += step
            t += step
        total = max(int(math.ceil(self.final)), 1)
        return {
            "total_cycles": total,
            "issued": issued,
            "stalls": stalls,
        }


@lru_cache(maxsize=128)
def _simulate_raw(sku: MercurySKU, k_groups: int, swizzle: bool, hot: bool,
                  ucie_fraction: float, contending: int, inject_hazard: str | None,
                  strict: bool, ideal_dram: bool = False, tile_m: int = TILE,
                  multicast_a: bool = False) -> dict[str, Any]:
    sim = _SCCSim(sku, k_groups, swizzle, hot, ucie_fraction, contending, inject_hazard,
                  strict, ideal_dram, tile_m, multicast_a)
    out = sim.run()
    total = out["total_cycles"]
    clk = sim.clock
    flops = 2 * tile_m * TILE * TILE * k_groups
    if ideal_dram:
        stall_dram = 0.0
    else:  # counterfactual attribution: same run with zero-latency, infinite GDDR7
        ideal = _simulate_raw(sku, k_groups, swizzle, hot, ucie_fraction, contending,
                              inject_hazard, strict, True, tile_m, multicast_a)
        stall_dram = float(max(0, total - ideal["total_cycles"]))
    dram_bytes = sim.panel_bytes * len(sim.panels)
    stme_active = sim.stme_stream_total / sku.systolic_arrays_per_scc
    accesses = sim.dram.hits + sim.dram.misses
    out.update({
        "stme_active_cycles": stme_active,
        "bank_conflict_cycles": sim.bank_conflict,
        "dram_stall_cycles": stall_dram,
        "achieved_tflops": flops / total * clk * sku.total_active_sccs / 1000.0,
        "achieved_sram_tbs": sim.sram_bytes / total * clk * sku.total_active_sccs / 1000.0,
        "achieved_dram_tbs": dram_bytes / total * clk * sku.num_dies / 1000.0,
        "effective_sram_banks": sim.eff_banks,
        "refresh_events": len(sim.dram.refresh_windows),
        "refresh_blocked_cycles": sim.dram.refresh_blocked,
        "dram_row_hit_rate": sim.dram.hits / accesses if accesses else 0.0,
        "hazards": sim.hazards,
        "l1_fill_tbs": sim.l1_bytes / total * clk * sku.total_active_sccs / 1000.0,
        "sram_bytes_per_clk_per_scc": sim.sram_bytes / total,
        "mcast_fanout": sim.mcast_fanout,
        "mcast_latency": sim.mcast_latency,
    })
    return out


def simulate_scc_cycle_trace(
    sku: MercurySKU = MERCURY_1D,
    *,
    k_groups: int = 16,
    swizzle: bool = True,
    hot_2x_trefi: bool = False,
    dram_temp_c: float = 70.0,
    ucie_d2d_fraction: float = 0.0,
    contending_sccs: int | None = None,
    inject_hazard: str | None = None,
    strict: bool = False,
    tile_m: int = TILE,
    multicast_a: bool = False,
) -> CycleTraceResult:
    """Trace one SCC's FP8 GEMM tile loop (tile_m x 128 output, ``k_groups`` x K=128).

    The defaults (``tile_m=128``, ``multicast_a=False``) reproduce the
    original baseline exactly. ``tile_m=256`` doubles the FP32 accumulator
    footprint (64 -> 128 KiB of the 256 KiB RF) and halves B re-reads per
    FLOP. ``multicast_a`` adds a NoC fan-out tree per die grid row.

    ``contending_sccs`` (default: all active SCCs on the die) share the
    Shared SRAM crossbar and the UCIe-S link. With ``strict=True``, a DMA
    buffer read before its WAIT raises ``ScoreboardHazardError``. Otherwise
    the scoreboard interlocks the read and counts it in ``hazards_detected``.
    """
    if type(k_groups) is not int or k_groups <= 0:
        raise ValueError("k_groups must be a positive integer")
    if not 0.0 <= ucie_d2d_fraction <= 1.0:
        raise ValueError("ucie_d2d_fraction must be in [0, 1]")
    if ucie_d2d_fraction and sku.num_dies < 2:
        raise ValueError("UCIe-S die-to-die spill needs a multi-die SKU (MERCURY_1D)")
    contending = sku.active_sccs_per_die if contending_sccs is None else contending_sccs
    if not 1 <= contending <= sku.active_sccs_per_die:
        raise ValueError("contending_sccs must be within the active SCCs of one die")
    if tile_m not in TILE_M_OPTIONS:
        raise ValueError(f"tile_m must be one of {TILE_M_OPTIONS}")
    acc_kib = tile_m * TILE * 4 / 1024
    if acc_kib > sku.rf_kib_per_scc:
        raise ValueError("FP32 accumulator tile exceeds the SCC register file")
    hot = bool(hot_2x_trefi or dram_temp_c >= sku.tmem_double_refresh_c)
    args = (sku, k_groups, hot, float(ucie_d2d_fraction), contending, inject_hazard, strict)
    extra = (False, tile_m, bool(multicast_a))
    raw = _simulate_raw(sku, k_groups, swizzle, *args[2:], *extra)
    sw = raw if swizzle else _simulate_raw(sku, k_groups, True, *args[2:], *extra)
    unsw = raw if not swizzle else _simulate_raw(sku, k_groups, False, *args[2:], *extra)
    stalls = raw["stalls"]
    total = raw["total_cycles"]
    return CycleTraceResult(
        sku=sku.name,
        k_groups=k_groups,
        swizzle=swizzle,
        hot_2x_trefi=hot,
        ucie_d2d_fraction=float(ucie_d2d_fraction),
        total_cycles=total,
        stme_active_cycles=raw["stme_active_cycles"],
        scoreboard_stall_cycles=sum(stalls.values()),
        bank_conflict_cycles=raw["bank_conflict_cycles"],
        dram_stall_cycles=raw["dram_stall_cycles"],
        achieved_tflops=raw["achieved_tflops"],
        achieved_sram_tbs=raw["achieved_sram_tbs"],
        achieved_dram_tbs=raw["achieved_dram_tbs"],
        dual_issue_ipc=raw["issued"] / total,
        swizzle_speedup=unsw["total_cycles"] / sw["total_cycles"],
        instructions_issued=raw["issued"],
        stme_utilization=raw["stme_active_cycles"] / total,
        effective_sram_banks=raw["effective_sram_banks"],
        refresh_events=raw["refresh_events"],
        refresh_blocked_cycles=raw["refresh_blocked_cycles"],
        dram_row_hit_rate=raw["dram_row_hit_rate"],
        hazards_detected=raw["hazards"],
        stall_breakdown=tuple(sorted(stalls.items())),
        tile_m=tile_m,
        tile_n=TILE,
        multicast_a=bool(multicast_a),
        l1_fill_tbs=raw["l1_fill_tbs"],
        sram_bytes_per_clk_per_scc=raw["sram_bytes_per_clk_per_scc"],
        rf_accumulator_kib=acc_kib,
        multicast_fanout=raw["mcast_fanout"],
        multicast_tree_latency_cycles=raw["mcast_latency"],
    )


def run_microarch_benchmarks(sku: MercurySKU = MERCURY_1D, k_groups: int = 16) -> dict[str, Any]:
    """JSON-serializable comparison of memory-system scenarios and one tiling optimization."""
    scenarios = {
        "swizzled_baseline": simulate_scc_cycle_trace(sku, k_groups=k_groups),
        "unswizzled_bank_conflict": simulate_scc_cycle_trace(sku, k_groups=k_groups, swizzle=False),
        "hot_2x_trefi_85c": simulate_scc_cycle_trace(
            sku, k_groups=k_groups, hot_2x_trefi=True, dram_temp_c=85.0),
    }
    if sku.num_dies > 1:
        scenarios["cross_die_ucie_spill"] = simulate_scc_cycle_trace(
            sku, k_groups=k_groups, ucie_d2d_fraction=0.10)
    # Same K extent (k_groups x 128), twice the output rows per SCC.
    scenarios["multicast_256x128"] = simulate_scc_cycle_trace(
        sku, k_groups=k_groups, tile_m=256, multicast_a=True)
    tile_only = simulate_scc_cycle_trace(sku, k_groups=k_groups, tile_m=256)
    base_r, opt_r = scenarios["swizzled_baseline"], scenarios["multicast_256x128"]
    base = base_r.achieved_tflops
    peak = sku.peak_tflops("fp8")
    grid_rows = GRID_M * sku.num_dies

    def summary(r: CycleTraceResult) -> dict[str, float]:
        return {"total_cycles": r.total_cycles, "tflops": r.achieved_tflops,
                "stme_utilization": r.stme_utilization, "sram_bank_tbs": r.achieved_sram_tbs,
                "l1_fill_tbs": r.l1_fill_tbs, "sram_bytes_per_clk_per_scc": r.sram_bytes_per_clk_per_scc,
                "dram_tbs": r.achieved_dram_tbs}

    optimization = {
        "change": "SCC tile 128x128 -> 256x128 plus NoC A-operand multicast per die grid row",
        "metric": "achieved FP8 TFLOPS (whole SKU) over k_groups x K=128",
        "baseline_128x128": summary(base_r),
        "ablation_256x128_no_multicast": summary(tile_only),
        "optimized_256x128_multicast": summary(opt_r),
        "speedup_tflops": opt_r.achieved_tflops / base,
        "complexity_cost": {
            "rf_fp32_accumulator_kib_per_scc": [base_r.rf_accumulator_kib, opt_r.rf_accumulator_kib],
            "rf_budget_kib_per_scc": sku.rf_kib_per_scc,
            "accumulators_per_warp": [4, 8],
            "l1_operand_double_buffer_kib_per_scc": [2 * 2 * TILE * TILE / 1024,
                                                     2 * (256 + TILE) * TILE / 1024],
            "multicast_trees": grid_rows,
            "multicast_fanout": opt_r.multicast_fanout,
            "multicast_tree_levels": math.ceil(math.log2(opt_r.multicast_fanout)),
            "multicast_tree_latency_cycles": opt_r.multicast_tree_latency_cycles,
            "noc_tree_links_total": grid_rows * (opt_r.multicast_fanout - 1),
            "unmodeled": "NoC replication buffers, router area/power, multicast flow control",
        },
    }
    return {
        "sku": sku.name,
        "k_groups": k_groups,
        "tile": [TILE, TILE, TILE],
        "peak_fp8_tflops": peak,
        "scenarios": {name: r.to_dict() for name, r in scenarios.items()},
        "relative_to_baseline": {name: r.achieved_tflops / base for name, r in scenarios.items()},
        "fraction_of_fp8_peak": {name: r.achieved_tflops / peak for name, r in scenarios.items()},
        "optimization": optimization,
    }
