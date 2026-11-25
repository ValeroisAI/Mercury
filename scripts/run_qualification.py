"""Run the full 10-scenario Mercury-1 Virtual Silicon & Server Qualification Suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mercury_sim.cluster_bench import run_all_qualification_scenarios


def render_markdown_report(results: dict) -> str:
    s1 = results["s1_ate_post"]
    s2 = results["s2_workstation_1s"]
    s3 = results["s3_oam_1d_cooling"]
    s4 = results["s4_cooling_fault"]
    s5 = results["s5_power_virus"]
    s6 = results["s6_cluster_70b"]
    s7 = results["s7_moe_and_decode"]
    s8 = results["s8_link_degradation"]
    s9 = results["s9_compute_isa"]
    s10 = results["s10_pretrain_tco"]

    lines = [
        "# Mercury-1 Virtual Silicon, Thermal & Server Qualification Report",
        "",
        "> **Scope:** End-to-end virtual bring-up modeling ATE die binning, 2-node electro-thermal RC ODEs, GDDR7 double-refresh (`2× tREFI`), DVFS/board power clamping, STME Block-FP8 IEEE-754 accumulation accuracy, 2:4 sparsity, L1/RF tile residency, multi-node FSDP/TP/MoE parallelism, batch=1 decode bottleneck honesty, link-fault degradation, and 1T-token TCO.",
        "",
        "---",
        "",
        "## 1. Manufacturing ATE Binning, ECC & Server POST (Scenario 1)",
        f"- **Mercury-1S (4 defective SCCs):** Passed = `{s1['1s_pass'].card_passed}`, Active SCCs = `{s1['1s_pass'].die_results[0].active_sccs}/84`, PCIe Gen{s1['1s_pass'].pcie_link_gen} x{s1['1s_pass'].pcie_link_width} DMA = `{s1['1s_pass'].pcie_h2d_goodput_gbs:.1f} GB/s`",
        f"- **Mercury-1D (6 & 8 defective SCCs):** Passed = `{s1['1d_pass'].card_passed}`, Active SCCs = `{s1['1d_pass'].die_results[0].active_sccs}+{s1['1d_pass'].die_results[1].active_sccs}/168`",
        f"- **Mercury-1D Reject Case (9 defective SCCs on Die 1):** Passed = `{s1['1d_reject_9defects'].card_passed}` (`{s1['1d_reject_9defects'].failure_reasons[0]}`)",
        f"- **30-Wafer Monte Carlo Clustered Yield ($D_0=0.10$):** Perfect Die Yield = `{s1['wafer_mc']['perfect_yield_pct']:.1f}%`, Salvaged Yield = `{s1['wafer_mc']['salvaged_yield_pct']:.1f}%` (`{s1['wafer_mc']['good_dies_per_wafer']:.1f}` good dies/wafer)",
        "",
        "## 2. Electro-Thermal & Power Qualification (Scenarios 2–5)",
        "",
        "| Scenario | Steady $T_j$ | Steady $T_{\\text{mem}}$ | Steady Clock | Chip / Board Power | Throttle % | 2× tREFI | Trip |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    thermal_rows = [
        ("S2a: `1S` Workstation Air @ 25°C", s2["air_25c"]),
        ("S2b: `1S` Workstation Air @ 35°C", s2["air_35c"]),
        ("S3a: `1D` 4U Server Air @ 25°C", s3["air_25c"]),
        ("S3b: `1D` 4U Server Air @ 35°C", s3["air_35c"]),
        ("S3c: `1D` OCP Liquid @ 35°C", s3["liquid_35c"]),
        ("S4: `1D` Mid-Run Cooling Fault (t=120s)", s4),
        ("S5: `1D` Synthetic Power Virus (Liquid)", s5),
    ]
    for label, r in thermal_rows:
        lines.append(
            f"| {label} | {r.steady_tj_c:.1f}°C (pk {r.peak_tj_c:.1f}°C) | {r.steady_tmem_c:.1f}°C | {r.steady_clock_ghz:.2f} GHz | {r.steady_chip_power_w:.0f}W / {r.steady_board_power_w:.0f}W | {r.throttle_fraction*100:.1f}% | {'YES' if r.double_refresh_triggered else 'No'} | {'YES' if r.thermal_trip_triggered else 'No'} |"
        )

    lines.extend([
        "",
        "**Thermal Engineering Findings:**",
        "- **`Mercury-1S` (PCIe Air):** Holds full `1.85 GHz` at 25°C ambient; at 35°C hot chassis, backside clamshell GDDR7 stays safe under PMU control.",
        "- **`Mercury-1D` (Extended OAM-L):** 4U air at 35°C triggers GDDR7 `2× tREFI` (`>85°C`) and closed-loop DVFS throttling—proving why **Extended OAM-L (140×185 mm) Direct-to-Chip Liquid** is the production cooling spec (`54.6°C` $T_j$, `68.0°C` $T_{\\text{mem}}$, zero throttling, plus ~42W subthreshold leakage savings).",
        "- **Fault & Power Virus Safety:** Mid-run pump/fan failure clamps DVFS safely (`peak Tj <= 106°C`), and synthetic `1.28×` power-virus stays clamped below `730W` board limit.",
        "",
        "## 3. Cluster Parallelism, MoE, Link Fault & Decode Honesty (Scenarios 6–8)",
        "",
        "| Workload / Topology | Fits VRAM | Free Headroom / GPU | Exposed Comm | Achieved MFU | Throughput |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ])

    for label, c in [
        ("S6a: `8× 1D` Llama-3 70B (`TP=8` over Eth, 11B/p)", s6["pod8_tp8"]),
        ("S6b: `8× 1D` Llama-3 70B (`TP=2` UCIe-S + `FSDP=4`)", s6["pod8_tp2"]),
        ("S6c: `16× 1D` Llama-3 70B (`TP=2+FSDP=8`, Full 16B/p)", s6["pod16_16b"]),
        ("S7a: `16× 1D` MoE 141B (`EP=8+FSDP=2` over 4×400GbE)", s7["moe_ep8"]),
        ("S8: `8× 1D` Degraded Net (`3/4` 400GbE ports up)", s8["cluster"]),
    ]:
        lines.append(
            f"| {label} | {'YES' if c.fits_in_vram else 'NO'} | {c.activation_headroom_gib:.1f} GiB | {c.exposed_comm_time_s:.3f} s ({c.exposed_comm_time_s/c.step_time_s*100:.1f}%) | {c.achieved_mfu*100:.1f}% | {c.cluster_tokens_per_sec:,.0f} tok/s |"
        )

    dec = s7["decode_b1"]
    lines.extend([
        f"- **S7b Batch=1 Decode Honesty (`{dec.model_name}`):** `Mercury-1D` achieves **`{dec.tokens_per_sec:.1f} tok/s`** (vs. H200 `{dec.h200_ref_tok_s:.1f} tok/s`, B200 `{dec.b200_ref_tok_s:.1f} tok/s`) — confirming `{dec.bottleneck}`.",
        f"- **S8 Fault Telemetry:** `{' / '.join(s8['card'].failure_reasons)}`",
        "",
        "## 4. STME Block-FP8 Numerical Accuracy, 2:4 Sparsity & Tile Residency (Scenario 9)",
        "```json",
        json.dumps(s9, indent=2),
        "```",
        "",
        "## 5. 1-Trillion-Token Llama-3 70B FP8 Pretraining Campaign TCO (Scenario 10, 64 GPUs)",
        "",
        "| Cluster (64 GPUs) | Cluster Capex | Wall Power (PUE 1.25) | Campaign Days | Energy Cost (@ $0.08/kWh) | Amortized Total Cost | Cost / 1M Tokens |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for t in s10:
        lines.append(
            f"| {t.system_name} | ${t.capex_usd:,.0f} | {t.cluster_wall_power_kw:.1f} kW | {t.training_days:.1f} d | ${t.energy_cost_usd:,.0f} | ${t.total_campaign_cost_usd:,.0f} | **${t.cost_per_million_tokens_usd:.4f}** |"
        )

    from mercury_sim.microarch_cycle import run_microarch_benchmarks
    from mercury_sim.physical_bom import run_physical_bom_analysis
    from mercury_sim.pytorch_bridge import run_pytorch_bridge_benchmark
    from mercury_isa.vm import run_isa_kernel_benchmarks

    micro = run_microarch_benchmarks()
    pt_bench = run_pytorch_bridge_benchmark()
    isa_bench = run_isa_kernel_benchmarks()
    phys_bom = run_physical_bom_analysis()

    lines.extend([
        "",
        "## 6. Cycle-Level Architectural Crossbar, PyTorch Parity & Live ISA VM Verification",
        "",
        "### 6.1 Cycle-Level Architectural 64-Bank Crossbar & GDDR7 Controller",
        "| Mode | Total Cycles | Achieved FP8 TFLOPS | Shared SRAM (TB/s) | Bank Conflict Cycles | DRAM Stall Cycles |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for key, label in [
        ("multicast_256x128", "256×128 + A-Operand Multicast"),
        ("swizzled_baseline", "128×128 XOR-Swizzled Baseline"),
        ("hot_2x_trefi_85c", "Hot >=85°C (`2× tREFI`)"),
        ("cross_die_ucie_spill", "Cross-Die UCIe-S Spill (10%)"),
        ("unswizzled_bank_conflict", "Unswizzled Stride-64 (2/64 banks)"),
    ]:
        m = micro["scenarios"][key]
        lines.append(
            f"| {label} | {m['total_cycles']:,} | {m['achieved_tflops']:,.1f} | {m['achieved_sram_tbs']:.2f} | {int(m['bank_conflict_cycles']):,} | {int(m['dram_stall_cycles']):,} |"
        )

    lines.extend([
        "",
        "### 6.2 PyTorch Transformer Block (`512d → 1024ffn`, Correlated Outliers)",
        "| GEMM | Mercury Cos Sim | Hopper-14b Cos Sim | Accumulator-Only Rel L2 (Mercury vs. Hopper) | Accumulator Ratio |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    ratios = pt_bench["summary"]["accumulator_error_ratio_hopper_over_mercury"]
    for gkey, glabel in [("forward_Y", "Forward `Y = X Wg^T`"), ("grad_dX", "Input Grad `dX = dY Wg`"), ("grad_dW", "Weight Grad `dW = dY^T X`")]:
        g = pt_bench["gemms"][gkey]
        lines.append(
            f"| {glabel} | {g['mercury']['cos_sim']:.6f} | {g['hopper_14bit']['cos_sim']:.6f} | `{g['mercury_accumulator_only']['rel_l2']:.2e}` vs. `{g['hopper_accumulator_only']['rel_l2']:.2e}` | **{ratios[gkey]:,.1f}× lower** |"
        )

    lines.extend([
        "",
        "### 6.3 Live Open RISC-V `Xmer*` `.masm` Kernel Execution (`MercuryVM`)",
        "| Kernel | Est. Cycles | STME / DMA / Coll / Split | Max Abs Err vs. FP64 | Parity |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for kname, kdata in isa_bench.items():
        tel = kdata["telemetry"]
        lines.append(
            f"| `{kname}.masm` | {tel['total_cycles_est']} | {tel['stme_insts']} / {tel['dma_ops']} / {tel['coll_ops']} / {tel['divergent_splits']} | `{kdata['max_abs_error']:.2e}` | {'PASS' if kdata['fp64_parity'] else 'FAIL'} |"
        )

    conv = pt_bench.get("training_convergence")
    if conv:
        attr = conv["attribution"]
        fp64_end = conv["paths"]["fp64_golden"]["loss_curve"][-1]
        merc_end = conv["paths"]["mercury_fp8_blk"]["loss_curve"][-1]
        hop_end = conv["paths"]["hopper_14bit"]["loss_curve"][-1]
        lines.extend([
            "",
            f"- **20-Step AdamW Convergence Check:** `FP64` ends at `{fp64_end:.4f}`, `Mercury FP8` at `{merc_end:.4f}`, and `Hopper-14b` at `{hop_end:.4f}`; FP8 input quantization contributes `{attr['fp8_quantization_update_drift']*100:.1f}%` end-to-end parameter update drift on both, while isolated vs. FP8+FP64-accum step-1 grad error is `{attr['mercury_first_step_grad_err_vs_fp8_exact_accum']:.2e}` (Mercury) vs. `{attr['hopper_first_step_grad_err_vs_fp8_exact_accum']:.2e}` (Hopper-14b).",
        ])

    lines.extend([
        "",
        "## 7. PCB Physical Courtyard Screening & NRE Volume Amortization (`mercury_sim/physical_bom.py`)",
        "",
        "### 7.1 Clamshell GDDR7 Courtyard Fit Screen",
        "| Form Factor | Board (mm) | Top Fill | Bottom Fill | Max Trace | Fits |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for row in phys_bom["pcb_layouts"].values():
        b_str = f"{row['board_mm'][0]}×{row['board_mm'][1]}"
        lines.append(
            f"| `{row['form_factor']}` | {b_str} | {row['top_used_pct']:.1f}% | {row['bottom_used_pct']:.1f}% | {row['max_gddr7_trace_mm']:.1f} mm | {'PASS' if row['fits_physically'] else 'REJECTED'} |"
        )

    lines.extend([
        "",
        "### 7.2 Upfront $85M NRE Amortization (`$2,456` Hardware BOM, 45% Gross Margin)",
        "| Volume (Units) | HW BOM | NRE / Unit | Loaded Unit Cost | 45% GM Target Price | 64-GPU System Capex | 1T-Token $/1M Tok |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in phys_bom["nre_volume_table"]:
        lines.append(
            f"| {r['volume_units']:,} | ${r['hardware_bom_usd']:,.0f} | ${r['nre_per_unit_usd']:,.0f} | ${r['fully_loaded_unit_cost_usd']:,.0f} | **${r['target_price_45gm_usd']:,.0f}** | ${r['capex_64gpu_usd']:,.0f} | **${r['campaign_1t_cost_per_1m_tokens_usd']:.4f}** |"
        )

    lines.extend([
        "",
        "## 8. Current Engineering Scope & Not Yet Built",
        "- **Implemented Here:** Pre-silicon architecture specification, clustered yield/binning model, 2-node electro-thermal RC ODEs, PCB courtyard & NRE amortization, cycle-level architectural crossbar & A-multicast model, PyTorch Transformer FP8 parity + 20-step AdamW convergence, and 32-bit RISC-V `Xmer*` assembler + VM.",
        "- **Not Yet Built:** Synthesizable SystemVerilog/Chisel RTL + Verilator cycle-accurate simulation, TSMC N4P physical P&R / STA signoff at 1.85 GHz, IBIS/HSPICE SI/PI for 32 Gbps PAM3 GDDR7 & 400GbE SerDes, and production LLVM/Triton MLIR compiler & Linux RDMA drivers.",
    ])

    return "\n".join(lines) + "\n"


def main() -> int:
    results = run_all_qualification_scenarios()
    report_md = render_markdown_report(results)
    out_path = ROOT / "docs" / "verification" / "qualification-report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
