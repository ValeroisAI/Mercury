"""Coupled electro-thermal RC simulator, PMU DVFS controller, and manufacturing/server fault injector."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .specs import COOLING_PROFILES, DVFS_TABLE, MERCURY_1D, MERCURY_1S, CoolingProfile, DVFSState, MercurySKU


@dataclass(frozen=True)
class WorkloadPowerProfile:
    """Normalized workload activity factors relative to nominal sustained BF16/FP8 training."""

    name: str
    compute_activity: float  # 1.0 = nominal 85% AI training utilization; 1.25 = synthetic power virus
    mem_activity: float      # 1.0 = active GDDR7 training streaming
    io_activity: float       # 1.0 = active 400GbE + PCIe + D2D
    active_scc_fraction: float = 1.0


WORKLOAD_PROFILES: dict[str, WorkloadPowerProfile] = {
    "idle": WorkloadPowerProfile("idle", compute_activity=0.05, mem_activity=0.10, io_activity=0.15),
    "inference_b1": WorkloadPowerProfile("inference_b1", compute_activity=0.28, mem_activity=0.95, io_activity=0.25),
    "training_bf16": WorkloadPowerProfile("training_bf16", compute_activity=0.98, mem_activity=0.95, io_activity=0.95),
    "training_fp8": WorkloadPowerProfile("training_fp8", compute_activity=1.00, mem_activity=1.00, io_activity=1.00),
    "power_virus": WorkloadPowerProfile("power_virus", compute_activity=1.28, mem_activity=1.12, io_activity=1.10),
}


@dataclass(frozen=True)
class ThermalTracePoint:
    time_s: float
    pstate: str
    clock_ghz: float
    vcore_v: float
    tj_c: float
    tcase_c: float
    tmem_c: float
    chip_power_w: float
    board_power_w: float
    leakage_w: float
    throttled: bool
    double_refresh_active: bool
    thermal_trip: bool


@dataclass(frozen=True)
class ThermalSimulationResult:
    sku_name: str
    cooling_name: str
    workload_name: str
    duration_s: float
    steady_tj_c: float
    peak_tj_c: float
    steady_tcase_c: float
    steady_tmem_c: float
    peak_tmem_c: float
    steady_clock_ghz: float
    mean_clock_ghz: float
    steady_chip_power_w: float
    steady_board_power_w: float
    peak_board_power_w: float
    throttle_fraction: float
    double_refresh_triggered: bool
    thermal_trip_triggered: bool
    trip_time_s: float | None
    effective_vram_bw_factor: float
    trace: tuple[ThermalTracePoint, ...]


@dataclass(frozen=True)
class DieBinningResult:
    die_index: int
    defective_sccs: int
    active_sccs: int
    passed_binning: bool
    murphy_yield_pct: float
    salvaged_yield_pct: float
    ecc_single_bit_scrubs: int
    ecc_uncorrectable_errors: int


@dataclass(frozen=True)
class CardQualificationResult:
    sku_name: str
    card_passed: bool
    die_results: tuple[DieBinningResult, ...]
    pcie_link_width: int
    pcie_link_gen: int
    pcie_h2d_goodput_gbs: float
    eth_ports_up: int
    eth_unidir_gbs: float
    failure_reasons: tuple[str, ...]


def _murphy_yield(area_cm2: float, d0: float) -> float:
    x = area_cm2 * d0
    return 1.0 if x == 0 else ((1.0 - math.exp(-x)) / x) ** 2


def qualify_manufactured_card(
    sku: MercurySKU,
    defective_sccs_per_die: tuple[int, ...] | None = None,
    d0_per_cm2: float = 0.10,
    pcie_lanes_trained: int | None = None,
    eth_ports_up: int | None = None,
    inject_ecc_sbe_count: int = 2,
    inject_ecc_ube_count: int = 0,
) -> CardQualificationResult:
    """Simulate ATE die binning (84->76 SCC redundancy) and customer server POST bring-up."""
    if defective_sccs_per_die is None:
        defective_sccs_per_die = tuple(0 for _ in range(sku.num_dies))
    if len(defective_sccs_per_die) != sku.num_dies:
        raise ValueError("defective_sccs_per_die length must match sku.num_dies")

    area_cm2 = sku.die_area_mm2 / 100.0
    y_perf = _murphy_yield(area_cm2, d0_per_cm2) * 100.0
    y_salv = min(
        95.0,
        (y_perf + (_murphy_yield(area_cm2 * 0.77, d0_per_cm2) * 100.0)) * 0.5 + 3.0,
    )

    die_results: list[DieBinningResult] = []
    reasons: list[str] = []

    for idx, defects in enumerate(defective_sccs_per_die):
        if defects < 0 or defects > sku.fab_sccs_per_die:
            raise ValueError(f"Invalid defect count {defects} for die {idx}")
        surviving = sku.fab_sccs_per_die - defects
        passed_die = surviving >= sku.active_sccs_per_die
        active = sku.active_sccs_per_die if passed_die else surviving
        if not passed_die:
            reasons.append(
                f"Die {idx} failed ATE binning: {surviving}/{sku.fab_sccs_per_die} SCCs survived (< {sku.active_sccs_per_die} required)"
            )
        die_results.append(
            DieBinningResult(
                die_index=idx,
                defective_sccs=defects,
                active_sccs=active,
                passed_binning=passed_die,
                murphy_yield_pct=y_perf,
                salvaged_yield_pct=y_salv,
                ecc_single_bit_scrubs=inject_ecc_sbe_count,
                ecc_uncorrectable_errors=inject_ecc_ube_count,
            )
        )

    if inject_ecc_ube_count > 0:
        reasons.append(f"GDDR7 ECC uncorrectable error count = {inject_ecc_ube_count}")

    trained_lanes = sku.pcie_gen5_lanes if pcie_lanes_trained is None else pcie_lanes_trained
    if trained_lanes <= 0 or trained_lanes > sku.pcie_gen5_lanes:
        raise ValueError("Invalid trained PCIe lane count")
    if trained_lanes < sku.pcie_gen5_lanes:
        reasons.append(f"PCIe degraded link width: Gen5 x{trained_lanes} (expected x{sku.pcie_gen5_lanes})")
    # PCIe Gen5 raw 32 GT/s 128b/130b = ~3.94 GB/s/lane, DMA goodput ~88% -> ~3.47 GB/s/lane (~55.5 GB/s x16)
    pcie_goodput_gbs = trained_lanes * 3.47

    ports_up = sku.eth_400g_ports if eth_ports_up is None else eth_ports_up
    if ports_up < 0 or ports_up > sku.eth_400g_ports:
        raise ValueError("Invalid Ethernet port count")
    if ports_up < sku.eth_400g_ports:
        reasons.append(f"Scale-out port degradation: {ports_up}/{sku.eth_400g_ports} x 400GbE links active")
    eth_bw_gbs = (ports_up * 400.0) / 8.0

    # Card is considered deployable even if 1 eth port or x8 PCIe degrades?
    # Hardware POST strictly fails on ATE binning failure or uncorrectable ECC; link degradation is flagged.
    hard_fail = (not all(d.passed_binning for d in die_results)) or (inject_ecc_ube_count > 0)
    return CardQualificationResult(
        sku_name=sku.name,
        card_passed=not hard_fail,
        die_results=tuple(die_results),
        pcie_link_width=trained_lanes,
        pcie_link_gen=5,
        pcie_h2d_goodput_gbs=pcie_goodput_gbs,
        eth_ports_up=ports_up,
        eth_unidir_gbs=eth_bw_gbs,
        failure_reasons=tuple(reasons),
    )


def compute_instantaneous_power_w(
    sku: MercurySKU,
    dvfs: DVFSState,
    workload: WorkloadPowerProfile,
    tj_c: float,
    tmem_c: float,
) -> tuple[float, float, float, float, float]:
    """Return (chip_total_w, die_compute_plus_leak_w, mem_w, io_w, leak_w)."""
    # Baseline calibrated at 1.85 GHz, 0.78 V, Tj=85C, nominal training (activity=1.0):
    # 1S: 179W compute + 43W leak + 75W mem + 44W io = 341W (~340W TDP)
    # 1D: 357W compute + 86W leak + 150W mem + 88W io = 681W (~680W TDP)
    base_compute_w = 178.5 * sku.num_dies
    base_leak_85_w = 43.0 * sku.num_dies
    base_mem_w = 75.0 * sku.num_dies
    base_io_w = 44.0 * sku.num_dies

    v_ratio = dvfs.vcore_v / sku.nominal_vcore_v
    f_ratio = dvfs.clock_ghz / sku.nominal_clock_ghz

    compute_dyn_w = base_compute_w * workload.compute_activity * workload.active_scc_fraction * (v_ratio**2) * f_ratio
    # Exponential subthreshold leakage sensitivity (~2.2%/C -> doubles every ~31.5 C)
    leak_w = base_leak_85_w * v_ratio * math.exp(0.022 * (tj_c - 85.0))
    # GDDR7 power scales with activity; above 85C JEDEC 2x tREFI refresh adds ~8% DRAM overhead
    refresh_mult = 1.08 if tmem_c >= sku.tmem_double_refresh_c else 1.0
    mem_w = base_mem_w * (0.25 + 0.75 * workload.mem_activity) * refresh_mult
    io_w = base_io_w * (0.30 + 0.70 * workload.io_activity)

    die_power_w = compute_dyn_w + leak_w
    chip_total_w = die_power_w + mem_w + io_w
    return chip_total_w, die_power_w, mem_w, io_w, leak_w


def simulate_thermal_run(
    sku: MercurySKU,
    cooling: CoolingProfile | str,
    workload: WorkloadPowerProfile | str = "training_fp8",
    duration_s: float = 300.0,
    dt_s: float = 0.5,
    mid_run_cooling_fault_s: float | None = None,
    fault_cooling: CoolingProfile | str = "degraded_fan_fault",
    seed_ambient_jitter_c: float = 0.0,
) -> ThermalSimulationResult:
    """Integrate 2-node thermal RC ODEs with closed-loop DVFS power & thermal PMU control."""
    cool = COOLING_PROFILES[cooling] if isinstance(cooling, str) else cooling
    wl = WORKLOAD_PROFILES[workload] if isinstance(workload, str) else workload
    fault_cool = COOLING_PROFILES[fault_cooling] if isinstance(fault_cooling, str) else fault_cooling

    if dt_s <= 0 or duration_s <= 0:
        raise ValueError("duration_s and dt_s must be positive")

    steps = int(round(duration_s / dt_s))
    tj = cool.ambient_c + seed_ambient_jitter_c
    tcase = cool.ambient_c + seed_ambient_jitter_c
    tmem = cool.ambient_c + seed_ambient_jitter_c

    pstate_idx = 0
    thermal_trip = False
    trip_time: float | None = None
    double_refresh_ever = False
    trace: list[ThermalTracePoint] = []

    peak_tj = tj
    peak_tmem = tmem
    peak_board_w = 0.0
    throttled_steps = 0
    clock_sum = 0.0

    for step in range(steps + 1):
        t = step * dt_s
        active_cool = (
            fault_cool
            if (mid_run_cooling_fault_s is not None and t >= mid_run_cooling_fault_s)
            else cool
        )
        amb = active_cool.ambient_c

        if thermal_trip:
            dvfs = DVFS_TABLE[-1]  # P5_IDLE emergency clamp
            eff_wl = WORKLOAD_PROFILES["idle"]
        else:
            dvfs = DVFS_TABLE[pstate_idx]
            eff_wl = wl

        chip_w, die_w, mem_w, io_w, leak_w = compute_instantaneous_power_w(
            sku, dvfs, eff_wl, tj, tmem
        )
        board_w = chip_w / sku.vrm_efficiency

        # Closed-loop PMU check: enforce board power limit AND Tj / Tmem thermal thresholds
        if not thermal_trip:
            if tj >= sku.tj_trip_c or tmem >= sku.tmem_trip_c:
                thermal_trip = True
                trip_time = t
                pstate_idx = len(DVFS_TABLE) - 1
                dvfs = DVFS_TABLE[pstate_idx]
                eff_wl = WORKLOAD_PROFILES["idle"]
                chip_w, die_w, mem_w, io_w, leak_w = compute_instantaneous_power_w(
                    sku, dvfs, eff_wl, tj, tmem
                )
                board_w = chip_w / sku.vrm_efficiency
            elif (
                tj >= sku.tj_throttle_c
                or tmem >= sku.tmem_throttle_c
                or board_w > sku.board_power_limit_w
            ):
                if pstate_idx < len(DVFS_TABLE) - 2:
                    pstate_idx += 1
                    dvfs = DVFS_TABLE[pstate_idx]
                    chip_w, die_w, mem_w, io_w, leak_w = compute_instantaneous_power_w(
                        sku, dvfs, eff_wl, tj, tmem
                    )
                    board_w = chip_w / sku.vrm_efficiency
            elif (
                pstate_idx > 0
                and tj < (sku.tj_throttle_c - 6.0)
                and tmem < (sku.tmem_throttle_c - 6.0)
            ):
                # Hysteresis check: only step back up if candidate higher P-state fits under board power cap
                cand = DVFS_TABLE[pstate_idx - 1]
                cand_w, *_ = compute_instantaneous_power_w(sku, cand, eff_wl, tj, tmem)
                if (cand_w / sku.vrm_efficiency) <= sku.board_power_limit_w * 0.97:
                    pstate_idx -= 1
                    dvfs = cand
                    chip_w, die_w, mem_w, io_w, leak_w = compute_instantaneous_power_w(
                        sku, dvfs, eff_wl, tj, tmem
                    )
                    board_w = chip_w / sku.vrm_efficiency

        double_refresh = tmem >= sku.tmem_double_refresh_c
        if double_refresh:
            double_refresh_ever = True

        throttled = (pstate_idx > 0) or thermal_trip
        if throttled:
            throttled_steps += 1
        clock_sum += dvfs.clock_ghz
        peak_tj = max(peak_tj, tj)
        peak_tmem = max(peak_tmem, tmem)
        peak_board_w = max(peak_board_w, board_w)

        if step % 10 == 0 or step == steps or thermal_trip:
            trace.append(
                ThermalTracePoint(
                    time_s=t,
                    pstate=dvfs.name,
                    clock_ghz=dvfs.clock_ghz,
                    vcore_v=dvfs.vcore_v,
                    tj_c=tj,
                    tcase_c=tcase,
                    tmem_c=tmem,
                    chip_power_w=chip_w,
                    board_power_w=board_w,
                    leakage_w=leak_w,
                    throttled=throttled,
                    double_refresh_active=double_refresh,
                    thermal_trip=thermal_trip,
                )
            )

        if step == steps:
            break

        # 2-node + VRAM thermal RC integration
        # Effective steady target for case/coldplate: T_case_eq = ambient + chip_w * R_ca
        # Die junction sits above case by (die_w_per_die + io_w_per_die * 0.5) * R_jc_die
        per_die_heat_w = (die_w + 0.5 * io_w) / sku.num_dies
        t_case_target = amb + chip_w * active_cool.r_theta_ca
        # Die RC tau ~ c_th_die * R_eq; integrate case and junction cleanly
        tau_case_s = max(5.0, active_cool.c_th_die_j_per_c * active_cool.r_theta_ca * 2.2)
        dt_case = ((t_case_target - tcase) / tau_case_s) * dt_s
        tcase_next = tcase + dt_case

        t_j_target = tcase_next + per_die_heat_w * active_cool.r_theta_jc_die
        tau_die_s = max(1.0, active_cool.c_th_die_j_per_c * active_cool.r_theta_jc_die * 0.7)
        tj_next = tj + ((t_j_target - tj) / tau_die_s) * dt_s

        # Backside clamshell GDDR7 thermal node couples to case + local DRAM self-heating
        # Clamshell PCB impedance ~0.16 C/W effective on 1D (distributed across 48 packages) / ~0.22 C/W on 1S
        r_mem_local = 0.21 if sku.num_dies == 1 else 0.145
        t_mem_target = tcase_next + mem_w * r_mem_local * (0.65 + active_cool.mem_coupling_factor)
        tau_mem_s = max(2.5, active_cool.c_th_mem_j_per_c * r_mem_local * 1.4)
        tmem_next = tmem + ((t_mem_target - tmem) / tau_mem_s) * dt_s

        tcase, tj, tmem = tcase_next, tj_next, tmem_next

    last = trace[-1]
    bw_factor = 0.975 if last.double_refresh_active else 1.0
    return ThermalSimulationResult(
        sku_name=sku.name,
        cooling_name=cool.name,
        workload_name=wl.name,
        duration_s=duration_s,
        steady_tj_c=last.tj_c,
        peak_tj_c=peak_tj,
        steady_tcase_c=last.tcase_c,
        steady_tmem_c=last.tmem_c,
        peak_tmem_c=peak_tmem,
        steady_clock_ghz=last.clock_ghz,
        mean_clock_ghz=clock_sum / (steps + 1),
        steady_chip_power_w=last.chip_power_w,
        steady_board_power_w=last.board_power_w,
        peak_board_power_w=peak_board_w,
        throttle_fraction=throttled_steps / (steps + 1),
        double_refresh_triggered=double_refresh_ever,
        thermal_trip_triggered=thermal_trip,
        trip_time_s=trip_time,
        effective_vram_bw_factor=bw_factor,
        trace=tuple(trace),
    )


def monte_carlo_wafer_binning(
    sku: MercurySKU,
    num_wafers: int = 20,
    dies_per_wafer: int = 117,
    d0_per_cm2: float = 0.10,
    seed: int = 42,
) -> dict[str, float]:
    """Simulate stochastic wafer defect clustering across SCC logic vs non-repairable die area."""
    rng = random.Random(seed)
    area_cm2 = sku.die_area_mm2 / 100.0
    non_repair_area_cm2 = area_cm2 * 0.494  # SRAM/PHY/IO/NoC unrepairable outside spare SCCs
    repair_area_cm2 = area_cm2 * 0.506      # 84 SCC array area

    total_dies = num_wafers * dies_per_wafer
    perfect_dies = 0
    salvaged_dies = 0
    rejected_dies = 0

    for _ in range(total_dies):
        # Negative-binomial-like clustered defect count via Gamma-Poisson
        # Shape parameter alpha=2.5 matches realistic semiconductor defect clustering
        alpha = 2.5
        rate_nr = rng.gammavariate(alpha, (non_repair_area_cm2 * d0_per_cm2) / alpha)
        nr_defects = _poisson(rng, rate_nr)
        if nr_defects > 0:
            rejected_dies += 1
            continue

        rate_scc = rng.gammavariate(alpha, (repair_area_cm2 * d0_per_cm2) / alpha)
        scc_defects = _poisson(rng, rate_scc)
        if scc_defects == 0:
            perfect_dies += 1
            salvaged_dies += 1
        elif (sku.fab_sccs_per_die - scc_defects) >= sku.active_sccs_per_die:
            salvaged_dies += 1
        else:
            rejected_dies += 1

    return {
        "total_dies": float(total_dies),
        "perfect_yield_pct": 100.0 * perfect_dies / total_dies,
        "salvaged_yield_pct": 100.0 * salvaged_dies / total_dies,
        "good_dies_per_wafer": salvaged_dies / num_wafers,
        "rejected_dies": float(rejected_dies),
    }


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0.0:
        return 0
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return k - 1
