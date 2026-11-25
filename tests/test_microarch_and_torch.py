"""Cycle-trace microarchitecture and PyTorch Transformer parity tests."""

import json
import math

import pytest

from mercury_sim.microarch_cycle import (
    CycleTraceResult,
    ScoreboardHazardError,
    build_warp_program,
    effective_sram_banks,
    run_microarch_benchmarks,
    simulate_scc_cycle_trace,
)
from mercury_sim.pytorch_bridge import (
    MercuryTransformerBlock,
    compare_gemm_paths,
    correlated_outlier_activations,
    run_pytorch_bridge_benchmark,
)
from mercury_sim.specs import MERCURY_1D, MERCURY_1S


# ---------------------------------------------------------------- microarch

def test_trace_result_is_frozen_and_physical():
    r = simulate_scc_cycle_trace(MERCURY_1D)
    assert isinstance(r, CycleTraceResult)
    with pytest.raises(Exception):
        r.total_cycles = 1
    # 16 groups x 16 warps x 4 MMAs x 64 stream cycles over 8 STMEs
    assert r.stme_active_cycles == 16 * 16 * 4 * 64 / 8
    assert 0 < r.stme_utilization <= 1
    assert r.achieved_tflops < MERCURY_1D.peak_tflops("fp8")
    assert r.achieved_sram_tbs <= MERCURY_1D.shared_sram_bandwidth_tbs() + 1e-9
    assert r.achieved_dram_tbs <= MERCURY_1D.vram_bandwidth_tbs + 1e-9
    assert 0 < r.dual_issue_ipc <= 2
    assert 0.5 < r.dram_row_hit_rate < 1


def test_xor_swizzle_removes_stride64_bank_conflicts():
    sw = simulate_scc_cycle_trace(MERCURY_1D, swizzle=True)
    unsw = simulate_scc_cycle_trace(MERCURY_1D, swizzle=False)
    assert effective_sram_banks(76, 64, True) == 64
    assert effective_sram_banks(76, 64, False) == 2
    assert sw.bank_conflict_cycles == 0
    assert unsw.bank_conflict_cycles > 100 * 16
    assert unsw.total_cycles > 10 * sw.total_cycles
    assert sw.swizzle_speedup == unsw.swizzle_speedup > 10
    # A lone SCC has no die-level contention: the stride-64 layout costs nothing.
    lone = simulate_scc_cycle_trace(MERCURY_1D, swizzle=False, contending_sccs=1)
    assert lone.bank_conflict_cycles == 0


def test_hot_2x_trefi_increases_dram_stalls():
    cool = simulate_scc_cycle_trace(MERCURY_1D)
    hot = simulate_scc_cycle_trace(MERCURY_1D, hot_2x_trefi=True)
    by_temp = simulate_scc_cycle_trace(MERCURY_1D, dram_temp_c=85.0)
    assert hot.hot_2x_trefi and by_temp.hot_2x_trefi and not cool.hot_2x_trefi
    assert by_temp == hot
    assert hot.refresh_events > 1.8 * cool.refresh_events
    assert hot.refresh_blocked_cycles > cool.refresh_blocked_cycles
    assert hot.dram_stall_cycles > cool.dram_stall_cycles
    assert hot.total_cycles > cool.total_cycles


def test_cross_die_ucie_spill_lowers_throughput():
    local = simulate_scc_cycle_trace(MERCURY_1D)
    spill = simulate_scc_cycle_trace(MERCURY_1D, ucie_d2d_fraction=0.10)
    assert spill.achieved_tflops < local.achieved_tflops
    assert spill.total_cycles > local.total_cycles
    with pytest.raises(ValueError):
        simulate_scc_cycle_trace(MERCURY_1S, ucie_d2d_fraction=0.1)


def test_scoreboard_detects_dma_before_wait_and_stme_raw():
    good = simulate_scc_cycle_trace(MERCURY_1D, k_groups=4)
    stalls = dict(good.stall_breakdown)
    assert good.hazards_detected == 0
    assert stalls["stme_acc_raw"] > 0      # accumulator chains + final STOREs
    assert stalls["dma_token"] > 0         # WAIT on in-flight DMA tokens
    assert good.scoreboard_stall_cycles == sum(stalls.values())

    prog = build_warp_program(2, "dma_before_wait")
    assert not any(op[0] == "WAIT" for op in prog)
    bad = simulate_scc_cycle_trace(MERCURY_1D, k_groups=4, inject_hazard="dma_before_wait")
    assert bad.hazards_detected == 16 * 4  # every warp, every K group
    assert dict(bad.stall_breakdown)["dma_token"] > 0  # interlocked, not corrupted
    with pytest.raises(ScoreboardHazardError):
        simulate_scc_cycle_trace(MERCURY_1D, k_groups=4, inject_hazard="dma_before_wait", strict=True)


def test_microarch_benchmark_hook_is_json_clean():
    report = run_microarch_benchmarks()
    data = json.loads(json.dumps(report))
    assert set(data["scenarios"]) == {
        "swizzled_baseline", "unswizzled_bank_conflict", "hot_2x_trefi_85c", "cross_die_ucie_spill"}
    rel = data["relative_to_baseline"]
    assert rel["swizzled_baseline"] == 1.0
    assert all(rel[k] < 1.0 for k in rel if k != "swizzled_baseline")
    assert 0.5 < data["fraction_of_fp8_peak"]["swizzled_baseline"] < 0.9


# ---------------------------------------------------------------- torch

def test_transformer_block_shapes_and_outliers():
    import torch
    block = MercuryTransformerBlock(64, 4, 128).double()
    x = correlated_outlier_activations(1, 4, 64, seed=3)
    assert block(x).shape == x.shape
    assert x.abs().max() == 24.0
    with pytest.raises(ValueError):
        MercuryTransformerBlock(30, 4)
    small = compare_gemm_paths(torch.randn(2, 128, dtype=torch.float64), torch.randn(128, 4, dtype=torch.float64))
    assert small["shape_mnk"] == [2, 4, 128]


def test_torch_forward_and_gradient_parity():
    r = run_pytorch_bridge_benchmark()
    assert r["autograd_alignment"]["forward_max_abs"] < 1e-12
    assert r["autograd_alignment"]["dW_max_abs"] < 1e-12
    for name, g in r["gemms"].items():
        assert g["mercury"]["cos_sim"] > 0.999, name
        # Same FP8 bytes: the accumulator alone separates the two paths.
        assert g["mercury_accumulator_only"]["rel_l2"] < 1e-6, name
        assert g["hopper_accumulator_only"]["rel_l2"] > 10 * g["mercury_accumulator_only"]["rel_l2"], name
    # Outlier-heavy forward GEMM (K = hidden): Mercury beats Hopper end to end too.
    fwd = r["gemms"]["forward_Y"]
    assert fwd["mercury"]["rel_l2"] < fwd["hopper_14bit"]["rel_l2"]
    assert r["summary"]["mercury_preserves_0p999"]
    assert r["summary"]["mercury_beats_hopper_accumulator_rel_l2"]


def test_pytorch_bridge_hook_is_json_clean():
    r = run_pytorch_bridge_benchmark()
    data = json.loads(json.dumps(r))
    assert data == r
    assert all(math.isfinite(v) for g in data["gemms"].values() for p in g.values()
               if isinstance(p, dict) for v in p.values())
