"""Verify Mercury PCB geometry and production-volume assumptions."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from mercury_sim.cluster_bench import simulate_1t_token_pretraining_tco
from mercury_sim.physical_bom import (
    DEFAULT_VOLUMES,
    EXTENDED_OAM_48,
    OAM_R2_48,
    PCIE_24,
    STANDARD_OAM_32,
    STANDARD_OAM_48,
    analyze_nre_and_tco,
    analyze_pcb_courtyard,
    run_physical_bom_analysis,
)
from mercury_sim.specs import MERCURY_1D, MERCURY_1S


@pytest.mark.parametrize(("case", "depth"), [(STANDARD_OAM_48, 165), (OAM_R2_48, 170)])
def test_standard_envelopes_reject_48_chip_placement(case, depth):
    layout = analyze_pcb_courtyard(MERCURY_1D, case)
    expected_demand = 24 * 15 * 17 + 4_800 + 62 * 42 + 2 * 15 * 84
    assert layout.board_mm == (102, depth)
    assert layout.bottom_used_pct == pytest.approx(100 * expected_demand / (102 * depth))
    assert layout.bottom_used_pct > (95 if depth == 165 else 90)
    assert layout.fits_physically is False
    assert "bottom fill" in layout.blocking_reason
    assert "edge clearance" in layout.blocking_reason


def test_extended_48_chip_layout_passes_placement_and_trace_screen():
    layout = analyze_pcb_courtyard(MERCURY_1D, EXTENDED_OAM_48)
    assert layout.board_mm == (140, 185)
    assert layout.top_used_pct == pytest.approx(100 * (24 * 255 + 2604 + 3800 + 1200) / 25_900)
    assert layout.bottom_used_pct < 75 and layout.top_used_pct < 75
    assert layout.max_gddr7_trace_mm == 32
    assert layout.fits_physically is True
    assert layout.blocking_reason is None
    assert layout == analyze_pcb_courtyard()
    with pytest.raises(FrozenInstanceError):
        layout.fits_physically = False


@pytest.mark.parametrize(
    ("sku", "case", "size", "bottom_mm2"),
    [
        (MERCURY_1D, STANDARD_OAM_32, (102, 165), 16 * 255 + 2604 + 4800),
        (MERCURY_1S, PCIE_24, (111, 267), 12 * 255 + 1600 + 900),
    ],
)
def test_96_gb_fallback_and_72_gb_pcie_layouts_pass(sku, case, size, bottom_mm2):
    layout = analyze_pcb_courtyard(sku, case)
    assert layout.board_mm == size
    assert layout.bottom_used_pct == pytest.approx(100 * bottom_mm2 / (size[0] * size[1]))
    assert layout.top_used_pct < 75 and layout.bottom_used_pct < 75
    assert layout.max_gddr7_trace_mm <= 35
    assert layout.fits_physically is True


def test_rectangle_overlap_rejects_even_when_total_fill_is_safe():
    larger_vrm = analyze_pcb_courtyard(replace(MERCURY_1D, tdp_w=1100), EXTENDED_OAM_48)
    assert larger_vrm.top_used_pct < 75
    assert larger_vrm.fits_physically is False
    assert "top courtyard overlap" in larger_vrm.blocking_reason
    assert "VRM bank" in larger_vrm.blocking_reason


def test_mismatched_or_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="Unknown PCB form factor"):
        analyze_pcb_courtyard(MERCURY_1D, "invalid")
    with pytest.raises(ValueError, match="requires a 2-die SKU"):
        analyze_pcb_courtyard(MERCURY_1S, EXTENDED_OAM_48)


def test_nre_burden_and_45_percent_margin_fall_with_volume():
    rows = analyze_nre_and_tco()
    assert tuple(row.volume_units for row in rows) == DEFAULT_VOLUMES
    assert all(row.hardware_bom_usd == 2456 for row in rows)
    assert [row.nre_per_unit_usd for row in rows] == [8500, 3400, 1700, 850, 340]
    for lower, upper in zip(rows, rows[1:]):
        assert lower.fully_loaded_unit_cost_usd > upper.fully_loaded_unit_cost_usd
        assert lower.target_price_45gm_usd > upper.target_price_45gm_usd
        assert lower.campaign_1t_cost_per_1m_tokens_usd > upper.campaign_1t_cost_per_1m_tokens_usd
    assert rows[0].target_price_45gm_usd == pytest.approx(19_920)
    assert rows[1].target_price_45gm_usd == pytest.approx(10_647.27, abs=0.005)
    assert rows[2].target_price_45gm_usd == pytest.approx(7556.36, abs=0.005)
    assert rows[3].fully_loaded_unit_cost_usd == 3306
    assert rows[0].nre_per_unit_usd == 10 * rows[3].nre_per_unit_usd


def test_campaign_reprices_same_runtime_power_and_depreciation():
    mercury, h100, b200 = simulate_1t_token_pretraining_tco()
    small, medium, large = analyze_nre_and_tco(volumes=[10_000, 25_000, 50_000])
    assert mercury.training_days == pytest.approx(76.0, abs=0.1)
    assert medium.capex_64gpu_usd == pytest.approx(64 * (medium.target_price_45gm_usd + 800))
    expected = (medium.capex_64gpu_usd * mercury.training_days / (3 * 365) + mercury.energy_cost_usd) / 1e6
    assert medium.campaign_1t_cost_per_1m_tokens_usd == pytest.approx(expected)
    assert medium.h100_campaign_cost_per_1m_tokens_usd == h100.cost_per_million_tokens_usd
    assert medium.b200_campaign_cost_per_1m_tokens_usd == b200.cost_per_million_tokens_usd
    assert medium.campaign_1t_cost_per_1m_tokens_usd < b200.cost_per_million_tokens_usd
    assert large.campaign_1t_cost_per_1m_tokens_usd < medium.campaign_1t_cost_per_1m_tokens_usd
    assert b200.cost_per_million_tokens_usd < small.campaign_1t_cost_per_1m_tokens_usd < h100.cost_per_million_tokens_usd


def test_price_thresholds_at_exact_production_volumes():
    before_10k, below_10k, before_8k, below_8k = analyze_nre_and_tco(volumes=[27_923, 27_924, 43_724, 43_725])
    assert before_10k.target_price_45gm_usd >= 10_000 > below_10k.target_price_45gm_usd
    assert before_8k.target_price_45gm_usd >= 8_000 > below_8k.target_price_45gm_usd


def test_configurable_nre_and_1s_bom_do_not_inherit_1d_campaign():
    (row,) = analyze_nre_and_tco(MERCURY_1S, [10_000], nre_usd=0)
    assert row.nre_per_unit_usd == 0
    assert row.hardware_bom_usd == row.fully_loaded_unit_cost_usd == 1238
    assert row.gpu_capex_64gpu_usd == pytest.approx(64 * 1238 / 0.55)
    assert row.campaign_1t_cost_per_1m_tokens_usd is None


@pytest.mark.parametrize("volume", [0, -5, True, 10.0])
def test_invalid_production_volume_rejected(volume):
    with pytest.raises(ValueError, match="positive integers"):
        analyze_nre_and_tco(volumes=[volume])


@pytest.mark.parametrize("nre", [-1, float("inf"), float("nan")])
def test_invalid_nre_rejected(nre):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        analyze_nre_and_tco(nre_usd=nre)


def test_json_hook_round_trip():
    payload = run_physical_bom_analysis()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert set(payload["pcb_layouts"]) == {STANDARD_OAM_48, EXTENDED_OAM_48, STANDARD_OAM_32, PCIE_24, OAM_R2_48}
    assert [row["volume_units"] for row in payload["nre_volume_table"]] == list(DEFAULT_VOLUMES)
    assert payload["nre_breakdown_usd"]["total"] == 85_000_000
