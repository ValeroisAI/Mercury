"""End-to-end customer server, cluster parallelism, Roofline, MoE, fault, and TCO qualification models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .specs import MERCURY_1D, MERCURY_1S, MercurySKU
from .thermal_power import (
    ThermalSimulationResult,
    monte_carlo_wafer_binning,
    qualify_manufactured_card,
    simulate_thermal_run,
)


@dataclass(frozen=True)
class ClusterTrainingResult:
    scenario_name: str
    sku_name: str
    num_gpus: int
    parallelism: str
    precision: str
    bytes_per_param: int
    vram_per_gpu_gib: float
    static_mem_per_gpu_gib: float
    activation_headroom_gib: float
    fits_in_vram: bool
    steady_clock_ghz: float
    effective_vram_bw_tbs: float
    eth_unidir_gbs: float
    compute_time_s: float
    exposed_comm_time_s: float
    step_time_s: float
    achieved_mfu: float
    cluster_tokens_per_sec: float
    effective_tflops_per_gpu: float


@dataclass(frozen=True)
class DecodeHonestyResult:
    sku_name: str
    model_name: str
    batch_size: int
    active_weights_gb: float
    effective_bw_tbs: float
    tokens_per_sec: float
    bottleneck: str
    h200_ref_tok_s: float
    b200_ref_tok_s: float


@dataclass(frozen=True)
class PretrainTCOResult:
    system_name: str
    num_gpus: int
    capex_usd: float
    cluster_wall_power_kw: float
    training_days: float
    energy_kwh: float
    energy_cost_usd: float
    total_campaign_cost_usd: float
    cost_per_million_tokens_usd: float


def simulate_llm_training_cluster(
    scenario_name: str,
    sku: MercurySKU,
    num_gpus: int,
    model_params_b: float = 70.0,
    hidden_dim: int = 8192,
    num_layers: int = 80,
    seq_len: int = 4096,
    microbatch_per_gpu: int = 2,
    grad_accum_steps: int = 16,
    tp_degree: int = 2,
    precision: str = "fp8",
    bytes_per_param: int = 11,
    cooling: str = "oam_liquid_35c",
    eth_ports_up: int | None = None,
    sram_tile_hit_boost: float = 1.0,
) -> ClusterTrainingResult:
    """Simulate realistic multi-GPU training step combining thermal clock, GDDR7 BW, SRAM hit, and RoCEv2 overlap."""
    if tp_degree not in (1, 2, 4, 8) or num_gpus < tp_degree or num_gpus % tp_degree != 0:
        raise ValueError("Invalid tp_degree for num_gpus")

    wl_name = "training_fp8" if precision.lower() == "fp8" else "training_bf16"
    thermal: ThermalSimulationResult = simulate_thermal_run(sku, cooling, wl_name, duration_s=300.0)

    # Effective clock and VRAM BW after any thermal throttling & GDDR7 double refresh
    eff_clock = thermal.steady_clock_ghz
    eff_bw_tbs = sku.vram_bandwidth_tbs * thermal.effective_vram_bw_factor
    ports_up = sku.eth_400g_ports if eth_ports_up is None else eth_ports_up
    eth_bw_gbs = (ports_up * 400.0) / 8.0

    # Memory residency in binary GiB
    static_gib = (model_params_b * 1e9 * bytes_per_param) / (1024**3) / num_gpus
    vram_gib = sku.vram_gib_total
    headroom_gib = vram_gib - static_gib
    # Need at least ~12 GiB for checkpointed activations + RoCEv2 comm buffers + workspace
    fits = headroom_gib >= 12.0

    # Peak hardware TFLOPS at steady thermal clock
    peak_tflops = sku.peak_tflops(precision, clock_ghz=eff_clock)

    # Compute FLOPs per GPU per optimizer step: 6 * N * tokens_per_gpu
    tokens_per_mb_per_gpu = seq_len * microbatch_per_gpu
    tokens_per_step_per_gpu = tokens_per_mb_per_gpu * grad_accum_steps
    flops_per_gpu = 6.0 * (model_params_b * 1e9) * tokens_per_step_per_gpu

    # Roofline + SRAM efficiency model anchored to FlashAttention-3 / SwiGLU tiling
    sram_factor = min(1.0, (0.76 + 0.16 * (sku.shared_sram_mb_total / 256.0)) * sram_tile_hit_boost)
    if precision.lower() == "fp8":
        base_kernel_eff = 0.48
        ridge_scale = 680.0
    else:
        base_kernel_eff = 0.56
        ridge_scale = 385.0
    mem_eff = min(1.0, (eff_bw_tbs * sram_factor * ridge_scale) / max(1.0, peak_tflops))
    single_gpu_eff = base_kernel_eff * mem_eff
    compute_time_s = (flops_per_gpu / 1e12) / (peak_tflops * single_gpu_eff)

    # Communication model:
    # 1) Intra-card TP=2 on Mercury-1D uses 600 GB/s/dir UCIe-S; inter-card TP uses RoCEv2 eth_bw_gbs
    bytes_act_elem = 2.0  # BF16 activation residuals across TP AllReduce
    if tp_degree == 1:
        tp_comm_s = 0.0
    else:
        act_tensor_gb = (tokens_per_mb_per_gpu * hidden_dim * bytes_act_elem) / 1e9
        # 2 AllReduces per layer (fwd+bwd = 4x), Ring volume = 2*(tp-1)/tp
        tp_vol_per_mb_gb = num_layers * 4.0 * (2.0 * (tp_degree - 1) / tp_degree) * act_tensor_gb
        if tp_degree == 2 and sku.num_dies == 2 and sku.ucie_s_bidir_tbs > 0:
            tp_link_gbs = (sku.ucie_s_bidir_tbs * 1000.0) / 2.0  # 600 GB/s unidirectional
            overlap_tp = 0.88
        else:
            tp_link_gbs = max(1e-6, eth_bw_gbs)
            overlap_tp = 0.65
        tp_comm_s = grad_accum_steps * (tp_vol_per_mb_gb / tp_link_gbs) * (1.0 - overlap_tp)

    # 2) FSDP across data-parallel domain (dp_degree = num_gpus // tp_degree)
    dp_degree = num_gpus // tp_degree
    if dp_degree <= 1:
        fsdp_comm_s = 0.0
    else:
        # FSDP ReduceScatter (2B grads) + AllGather (1B/2B params) across dp shard per step
        param_comm_bytes = 3.0 if precision.lower() == "fp8" else 4.0
        fsdp_vol_per_gpu_gb = (model_params_b * 1e9 * param_comm_bytes / tp_degree) * ((dp_degree - 1) / dp_degree) / 1e9
        fsdp_raw_s = fsdp_vol_per_gpu_gb / max(1e-6, eth_bw_gbs)
        # FSDP overlaps cleanly behind backward pass & gradient accumulation
        fsdp_comm_s = fsdp_raw_s * 0.15

    exposed_comm_s = tp_comm_s + fsdp_comm_s
    step_time_s = compute_time_s + exposed_comm_s
    achieved_mfu = (flops_per_gpu / 1e12) / (step_time_s * peak_tflops)
    eff_tflops = peak_tflops * achieved_mfu
    total_tokens_step = tokens_per_step_per_gpu * num_gpus
    tok_per_sec = total_tokens_step / step_time_s

    return ClusterTrainingResult(
        scenario_name=scenario_name,
        sku_name=sku.name,
        num_gpus=num_gpus,
        parallelism=f"TP={tp_degree}+FSDP={dp_degree}",
        precision=precision.upper(),
        bytes_per_param=bytes_per_param,
        vram_per_gpu_gib=vram_gib,
        static_mem_per_gpu_gib=static_gib,
        activation_headroom_gib=headroom_gib,
        fits_in_vram=fits,
        steady_clock_ghz=eff_clock,
        effective_vram_bw_tbs=eff_bw_tbs,
        eth_unidir_gbs=eth_bw_gbs,
        compute_time_s=compute_time_s,
        exposed_comm_time_s=exposed_comm_s,
        step_time_s=step_time_s,
        achieved_mfu=achieved_mfu,
        cluster_tokens_per_sec=tok_per_sec,
        effective_tflops_per_gpu=eff_tflops,
    )


def simulate_moe_ep8_training(
    sku: MercurySKU = MERCURY_1D,
    num_gpus: int = 16,
    ep_degree: int = 8,
    total_params_b: float = 141.0,   # Mixtral 8x22B class (fits cleanly across 16x 1D = 2,145.8 GiB)
    active_params_b: float = 39.0,
    tokens_per_step_per_gpu: int = 65536,
    cooling: str = "oam_liquid_35c",
) -> ClusterTrainingResult:
    """Simulate Mixture-of-Experts (EP=8 + FSDP) All-to-All dispatch/combine over 4x400GbE."""
    if num_gpus < ep_degree or num_gpus % ep_degree != 0:
        raise ValueError("num_gpus must be a multiple of ep_degree")
    thermal = simulate_thermal_run(sku, cooling, "training_fp8", duration_s=300.0)
    eff_clock = thermal.steady_clock_ghz
    eff_bw_tbs = sku.vram_bandwidth_tbs * thermal.effective_vram_bw_factor
    eth_bw_gbs = sku.eth_unidir_gbs
    peak_tflops = sku.peak_tflops("fp8", clock_ghz=eff_clock)

    static_gib = (total_params_b * 1e9 * 11) / (1024**3) / num_gpus
    headroom_gib = sku.vram_gib_total - static_gib

    # MoE active compute FLOPs ~ 6 * active_params_b * tokens
    flops_per_gpu = 6.0 * (active_params_b * 1e9) * tokens_per_step_per_gpu
    mem_eff = min(1.0, (eff_bw_tbs * 0.90 * 680.0) / peak_tflops)
    compute_s = (flops_per_gpu / 1e12) / (peak_tflops * 0.45 * mem_eff)

    # All-to-All dispatch/combine across ep_degree + FSDP across dp_degree
    dp_degree = num_gpus // ep_degree
    act_dispatch_gb = (tokens_per_step_per_gpu * 6144 * 2 * 1.5 * 4 * (ep_degree - 1) / ep_degree) / 1e9
    fsdp_gb = ((total_params_b * 1e9 * 3.0 / ep_degree) * ((dp_degree - 1) / max(1, dp_degree)) / 1e9) if dp_degree > 1 else 0.0
    # Dual-phase pipeline (DeepSeek DualPipe style) hides ~75% of EP All-to-All and ~85% of FSDP
    exposed_comm_s = (act_dispatch_gb / eth_bw_gbs) * 0.25 + (fsdp_gb / eth_bw_gbs) * 0.15
    step_s = compute_s + exposed_comm_s
    mfu = (flops_per_gpu / 1e12) / (step_s * peak_tflops)

    return ClusterTrainingResult(
        scenario_name=f"moe_{int(total_params_b)}b_ep{ep_degree}_fp8",
        sku_name=sku.name,
        num_gpus=num_gpus,
        parallelism=f"EP={ep_degree}+FSDP={dp_degree}",
        precision="FP8",
        bytes_per_param=11,
        vram_per_gpu_gib=sku.vram_gib_total,
        static_mem_per_gpu_gib=static_gib,
        activation_headroom_gib=headroom_gib,
        fits_in_vram=headroom_gib >= 12.0,
        steady_clock_ghz=eff_clock,
        effective_vram_bw_tbs=eff_bw_tbs,
        eth_unidir_gbs=eth_bw_gbs,
        compute_time_s=compute_s,
        exposed_comm_time_s=exposed_comm_s,
        step_time_s=step_s,
        achieved_mfu=mfu,
        cluster_tokens_per_sec=(tokens_per_step_per_gpu * num_gpus) / step_s,
        effective_tflops_per_gpu=peak_tflops * mfu,
    )


def simulate_batch1_decode_honesty(
    sku: MercurySKU = MERCURY_1D,
    active_weights_gb: float = 70.0,
    cooling: str = "oam_liquid_35c",
) -> DecodeHonestyResult:
    """Explicitly check batch=1 autoregressive decode ceiling (GDDR7 memory-bandwidth bottleneck)."""
    thermal = simulate_thermal_run(sku, cooling, "inference_b1", duration_s=180.0)
    eff_bw_tbs = sku.vram_bandwidth_tbs * thermal.effective_vram_bw_factor
    # Batch=1 memory-bound decode hits ~82% sustained DRAM controller efficiency
    tok_s = (eff_bw_tbs * 1000.0 * 0.82) / active_weights_gb
    h200_tok_s = (4.8 * 1000.0 * 0.82) / active_weights_gb
    b200_tok_s = (8.0 * 1000.0 * 0.82) / active_weights_gb
    return DecodeHonestyResult(
        sku_name=sku.name,
        model_name="Llama-3-70B-FP8 (Batch=1)",
        batch_size=1,
        active_weights_gb=active_weights_gb,
        effective_bw_tbs=eff_bw_tbs,
        tokens_per_sec=tok_s,
        bottleneck="GDDR7 memory bandwidth bound (~3.07 TB/s vs H200 4.8 TB/s & B200 8.0 TB/s)",
        h200_ref_tok_s=h200_tok_s,
        b200_ref_tok_s=b200_tok_s,
    )


def simulate_1t_token_pretraining_tco(
    num_gpus: int = 64,
    total_tokens: float = 1e12,
    model_params_b: float = 70.0,
    electricity_usd_per_kwh: float = 0.08,
    pue: float = 1.25,
) -> tuple[PretrainTCOResult, ...]:
    """Compare 1-Trillion-Token 70B FP8 pretraining cost across Mercury-1D, H100, and B200 clusters."""
    total_flops = 6.0 * (model_params_b * 1e9) * total_tokens
    m1d_res = simulate_llm_training_cluster(
        "tco_64gpu", MERCURY_1D, num_gpus=num_gpus, tp_degree=2, precision="fp8", bytes_per_param=11
    )
    m1d_thermal = simulate_thermal_run(MERCURY_1D, "oam_liquid_35c", "training_fp8", duration_s=300.0)

    # Per-GPU system capex including switches/host share:
    # Mercury-1D: $4,700 card + $800 Ethernet/server share = $5,500/GPU
    # H100: $30,000/GPU incl NVSwitch/InfiniBand share
    # B200: $40,000/GPU incl NVLink5/IB share
    candidates = [
        (
            "64x Mercury-1D (400GbE OAM Liquid)",
            5500.0 * num_gpus,
            m1d_res.effective_tflops_per_gpu,
            m1d_thermal.steady_board_power_w + 110.0,  # + host/switch share per GPU
        ),
        (
            "64x NVIDIA H100 SXM (NVLink4 + IB)",
            30000.0 * num_gpus,
            1979.0 * 0.42,  # 42% FP8 MFU
            700.0 / 0.94 + 140.0,
        ),
        (
            "64x NVIDIA B200 (NVLink5 + IB)",
            40000.0 * num_gpus,
            4500.0 * 0.45,  # 45% FP8 MFU
            1000.0 / 0.94 + 160.0,
        ),
    ]

    results: list[PretrainTCOResult] = []
    for name, capex, eff_tflops_gpu, wall_w_per_gpu in candidates:
        cluster_tflops = num_gpus * eff_tflops_gpu
        seconds = total_flops / (cluster_tflops * 1e12)
        days = seconds / 86400.0
        hours = seconds / 3600.0
        cluster_kw = (num_gpus * wall_w_per_gpu * pue) / 1000.0
        kwh = cluster_kw * hours
        energy_cost = kwh * electricity_usd_per_kwh
        # Depreciate 3-year hardware capex over actual training campaign duration (plus electricity)
        amortized_hw = capex * (days / (3.0 * 365.0))
        total_cost = amortized_hw + energy_cost
        cost_per_1m = total_cost / (total_tokens / 1e6)
        results.append(
            PretrainTCOResult(
                system_name=name,
                num_gpus=num_gpus,
                capex_usd=capex,
                cluster_wall_power_kw=cluster_kw,
                training_days=days,
                energy_kwh=kwh,
                energy_cost_usd=energy_cost,
                total_campaign_cost_usd=total_cost,
                cost_per_million_tokens_usd=cost_per_1m,
            )
        )
    return tuple(results)


def run_all_qualification_scenarios() -> dict[str, Any]:
    """Execute all 10 virtual chip, card, server, and cluster qualification scenarios."""
    # 1. ATE & Server POST
    post_1s = qualify_manufactured_card(MERCURY_1S, defective_sccs_per_die=(4,))
    post_1d = qualify_manufactured_card(MERCURY_1D, defective_sccs_per_die=(6, 8))
    post_reject = qualify_manufactured_card(MERCURY_1D, defective_sccs_per_die=(2, 9))
    wafer_mc = monte_carlo_wafer_binning(MERCURY_1D, num_wafers=30, seed=42)

    # 2. Workstation Air (Mercury-1S)
    ws_25c = simulate_thermal_run(MERCURY_1S, "workstation_air_25c", "training_fp8", duration_s=300.0)
    ws_35c = simulate_thermal_run(MERCURY_1S, "workstation_air_35c", "training_fp8", duration_s=300.0)

    # 3. OAM Air vs Liquid (Mercury-1D)
    oam_air_25 = simulate_thermal_run(MERCURY_1D, "server_4u_air_25c", "training_fp8", duration_s=300.0)
    oam_air_35 = simulate_thermal_run(MERCURY_1D, "server_4u_air_35c", "training_fp8", duration_s=400.0)
    oam_liq_35 = simulate_thermal_run(MERCURY_1D, "oam_liquid_35c", "training_fp8", duration_s=300.0)

    # 4. Cooling Failure Survivability
    cooling_fault = simulate_thermal_run(
        MERCURY_1D,
        "oam_liquid_35c",
        "training_fp8",
        duration_s=360.0,
        mid_run_cooling_fault_s=120.0,
        fault_cooling="degraded_fan_fault",
    )

    # 5. Power-Virus Board Power Cap
    power_virus = simulate_thermal_run(MERCURY_1D, "oam_liquid_35c", "power_virus", duration_s=120.0)

    # 6. Cluster 70B Parallelism & VRAM Residency
    pod8_tp8 = simulate_llm_training_cluster("8x1D_TP8_naive", MERCURY_1D, 8, tp_degree=8, bytes_per_param=11)
    pod8_tp2 = simulate_llm_training_cluster("8x1D_TP2_FSDP4", MERCURY_1D, 8, tp_degree=2, bytes_per_param=11)
    pod16_fp32_adam = simulate_llm_training_cluster("16x1D_TP2_FSDP8_16B", MERCURY_1D, 16, tp_degree=2, bytes_per_param=16)

    # 7. MoE & Batch=1 Decode Honesty
    moe_ep8 = simulate_moe_ep8_training()
    decode_b1 = simulate_batch1_decode_honesty()

    # 8. Degraded Port / PCIe Fault Impact
    degraded_card = qualify_manufactured_card(MERCURY_1D, pcie_lanes_trained=8, eth_ports_up=3)
    degraded_cluster = simulate_llm_training_cluster(
        "8x1D_3port_degraded", MERCURY_1D, 8, tp_degree=2, bytes_per_param=11, eth_ports_up=3
    )

    # 9. Compute & ISA Numerical / Tile Diagnostics (from GPT's module)
    from .compute_isa import run_compute_isa_diagnostics

    isa_diag = run_compute_isa_diagnostics(sku=MERCURY_1D)

    # 10. 1T-Token Pretraining TCO
    tco_results = simulate_1t_token_pretraining_tco()

    return {
        "s1_ate_post": {
            "1s_pass": post_1s,
            "1d_pass": post_1d,
            "1d_reject_9defects": post_reject,
            "wafer_mc": wafer_mc,
        },
        "s2_workstation_1s": {"air_25c": ws_25c, "air_35c": ws_35c},
        "s3_oam_1d_cooling": {"air_25c": oam_air_25, "air_35c": oam_air_35, "liquid_35c": oam_liq_35},
        "s4_cooling_fault": cooling_fault,
        "s5_power_virus": power_virus,
        "s6_cluster_70b": {"pod8_tp8": pod8_tp8, "pod8_tp2": pod8_tp2, "pod16_16b": pod16_fp32_adam},
        "s7_moe_and_decode": {"moe_ep8": moe_ep8, "decode_b1": decode_b1},
        "s8_link_degradation": {"card": degraded_card, "cluster": degraded_cluster},
        "s9_compute_isa": isa_diag,
        "s10_pretrain_tco": tco_results,
    }
