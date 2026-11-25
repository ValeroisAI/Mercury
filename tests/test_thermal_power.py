"""Tests for Mercury-1 electro-thermal RC simulator, PMU controller, and ATE/server qualification."""

from mercury_sim.specs import MERCURY_1D, MERCURY_1S
from mercury_sim.thermal_power import (
    monte_carlo_wafer_binning,
    qualify_manufactured_card,
    simulate_thermal_run,
)


def test_mercury_1s_workstation_air_25c_holds_full_boost():
    res = simulate_thermal_run(MERCURY_1S, "workstation_air_25c", "training_fp8", duration_s=300.0)
    assert res.steady_clock_ghz == 1.85
    assert res.throttle_fraction == 0.0
    assert not res.thermal_trip_triggered
    assert res.steady_tj_c < 85.0
    assert res.steady_tmem_c < 85.0
    assert 330.0 <= res.steady_chip_power_w <= 360.0


def test_mercury_1d_oam_liquid_35c_sustains_full_training_no_double_refresh():
    res = simulate_thermal_run(MERCURY_1D, "oam_liquid_35c", "training_fp8", duration_s=300.0)
    assert res.steady_clock_ghz == 1.85
    assert res.throttle_fraction == 0.0
    assert not res.thermal_trip_triggered
    assert not res.double_refresh_triggered
    assert res.effective_vram_bw_factor == 1.0
    # Cooler junction (54.6C vs 85C reference) reduces subthreshold leakage (~44W vs 86W)
    assert 620.0 <= res.steady_chip_power_w <= 705.0
    assert res.steady_tj_c < 70.0
    assert res.steady_tmem_c < 75.0
    hot_air = simulate_thermal_run(MERCURY_1D, "server_4u_air_25c", "training_fp8", duration_s=300.0)
    assert res.trace[-1].leakage_w < hot_air.trace[-1].leakage_w


def test_mercury_1d_hot_4u_air_35c_exposes_thermal_throttling_and_double_refresh():
    res = simulate_thermal_run(MERCURY_1D, "server_4u_air_35c", "training_fp8", duration_s=400.0)
    assert res.double_refresh_triggered
    assert res.effective_vram_bw_factor == 0.975
    assert res.throttle_fraction > 0.0
    assert res.steady_clock_ghz < 1.85
    assert not res.thermal_trip_triggered  # PMU throttling stabilizes before 105C trip


def test_mid_run_cooling_fault_clamps_power_and_protects_silicon():
    res = simulate_thermal_run(
        MERCURY_1D,
        "oam_liquid_35c",
        "training_fp8",
        duration_s=360.0,
        mid_run_cooling_fault_s=120.0,
        fault_cooling="degraded_fan_fault",
    )
    assert res.throttle_fraction > 0.1
    assert res.peak_tj_c <= 106.0
    assert res.steady_clock_ghz < 1.5


def test_power_virus_respects_board_power_cap():
    res = simulate_thermal_run(MERCURY_1D, "oam_liquid_35c", "power_virus", duration_s=120.0)
    assert res.steady_board_power_w <= MERCURY_1D.board_power_limit_w
    assert res.throttle_fraction > 0.0  # Power-cap DVFS engages on synthetic >100% power virus


def test_ate_binning_and_server_post_fault_detection():
    # 8 spare SCCs survive up to 8 defects per die
    ok_card = qualify_manufactured_card(MERCURY_1D, defective_sccs_per_die=(6, 8))
    assert ok_card.card_passed
    assert ok_card.die_results[0].active_sccs == 76
    assert ok_card.die_results[1].active_sccs == 76

    bad_card = qualify_manufactured_card(MERCURY_1D, defective_sccs_per_die=(0, 9))
    assert not bad_card.card_passed
    assert any("ATE binning" in r for r in bad_card.failure_reasons)

    ube_card = qualify_manufactured_card(MERCURY_1S, inject_ecc_ube_count=1)
    assert not ube_card.card_passed

    degraded_net = qualify_manufactured_card(MERCURY_1D, pcie_lanes_trained=8, eth_ports_up=3)
    assert degraded_net.card_passed  # Still boots, flags degraded link telemetry
    assert degraded_net.eth_unidir_gbs == 150.0
    assert len(degraded_net.failure_reasons) == 2


def test_monte_carlo_wafer_binning_matches_salvage_gain():
    stats = monte_carlo_wafer_binning(MERCURY_1D, num_wafers=30, seed=7)
    assert stats["salvaged_yield_pct"] > stats["perfect_yield_pct"]
    assert 60.0 < stats["salvaged_yield_pct"] < 90.0
