# Mercury-1 Virtual Silicon, Thermal & Server Qualification Report

> **Scope:** End-to-end virtual bring-up modeling ATE die binning, 2-node electro-thermal RC ODEs, GDDR7 double-refresh (`2× tREFI`), DVFS/board power clamping, STME Block-FP8 IEEE-754 accumulation accuracy, 2:4 sparsity, L1/RF tile residency, multi-node FSDP/TP/MoE parallelism, batch=1 decode bottleneck honesty, link-fault degradation, and 1T-token TCO.

---

## 1. Manufacturing ATE Binning, ECC & Server POST (Scenario 1)
- **Mercury-1S (4 defective SCCs):** Passed = `True`, Active SCCs = `76/84`, PCIe Gen5 x16 DMA = `55.5 GB/s`
- **Mercury-1D (6 & 8 defective SCCs):** Passed = `True`, Active SCCs = `76+76/168`
- **Mercury-1D Reject Case (9 defective SCCs on Die 1):** Passed = `False` (`Die 1 failed ATE binning: 75/84 SCCs survived (< 76 required)`)
- **30-Wafer Monte Carlo Clustered Yield ($D_0=0.10$):** Perfect Die Yield = `65.3%`, Salvaged Yield = `80.9%` (`94.7` good dies/wafer)

## 2. Electro-Thermal & Power Qualification (Scenarios 2–5)

| Scenario | Steady $T_j$ | Steady $T_{\text{mem}}$ | Steady Clock | Chip / Board Power | Throttle % | 2× tREFI | Trip |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| S2a: `1S` Workstation Air @ 25°C | 76.3°C (pk 76.3°C) | 79.0°C | 1.85 GHz | 333W / 358W | 0.0% | No | No |
| S2b: `1S` Workstation Air @ 35°C | 82.2°C (pk 83.9°C) | 87.8°C | 1.70 GHz | 311W / 334W | 89.9% | YES | No |
| S3a: `1D` 4U Server Air @ 25°C | 75.7°C (pk 81.3°C) | 91.3°C | 1.70 GHz | 612W / 651W | 81.0% | YES | No |
| S3b: `1D` 4U Server Air @ 35°C | 67.9°C (pk 85.4°C) | 90.2°C | 0.90 GHz | 394W / 419W | 93.6% | YES | No |
| S3c: `1D` OCP Liquid @ 35°C | 54.6°C (pk 54.6°C) | 68.0°C | 1.85 GHz | 639W / 680W | 0.0% | No | No |
| S4: `1D` Mid-Run Cooling Fault (t=120s) | 54.3°C (pk 96.1°C) | 60.6°C | 0.40 GHz | 117W / 125W | 64.4% | YES | YES |
| S5: `1D` Synthetic Power Virus (Liquid) | 55.8°C (pk 55.8°C) | 70.6°C | 1.70 GHz | 679W / 722W | 100.0% | No | No |

**Thermal Engineering Findings:**
- **`Mercury-1S` (PCIe Air):** Holds full `1.85 GHz` at 25°C ambient; at 35°C hot chassis, backside clamshell GDDR7 stays safe under PMU control.
- **`Mercury-1D` (OAM):** 4U air at 35°C triggers GDDR7 `2× tREFI` (`>85°C`) and closed-loop DVFS throttling—proving why **OCP OAM 2.0 Direct-to-Chip Liquid** is the production cooling spec (`54.6°C` $T_j$, `68.0°C` $T_{\text{mem}}$, zero throttling, plus ~42W subthreshold leakage savings).
- **Fault & Power Virus Safety:** Mid-run pump/fan failure clamps DVFS safely (`peak Tj <= 106°C`), and synthetic `1.28×` power-virus stays clamped below `730W` board limit.

## 3. Cluster Parallelism, MoE, Link Fault & Decode Honesty (Scenarios 6–8)

| Workload / Topology | Fits VRAM | Free Headroom / GPU | Exposed Comm | Achieved MFU | Throughput |
| :--- | :---: | :---: | :---: | :---: | :--- |
| S6a: `8× 1D` Llama-3 70B (`TP=8` over Eth, 11B/p) | YES | 44.5 GiB | 2.105 s (3.7%) | 41.9% | 18,394 tok/s |
| S6b: `8× 1D` Llama-3 70B (`TP=2` UCIe-S + `FSDP=4`) | YES | 44.5 GiB | 0.197 s (0.4%) | 43.4% | 19,031 tok/s |
| S6c: `16× 1D` Llama-3 70B (`TP=2+FSDP=8`, Full 16B/p) | YES | 68.9 GiB | 0.206 s (0.4%) | 43.4% | 38,055 tok/s |
| S7a: `16× 1D` MoE 141B (`EP=8+FSDP=2` over 4×400GbE) | YES | 43.8 GiB | 0.025 s (0.1%) | 36.7% | 57,768 tok/s |
| S8: `8× 1D` Degraded Net (`3/4` 400GbE ports up) | YES | 44.5 GiB | 0.216 s (0.4%) | 43.4% | 19,024 tok/s |
- **S7b Batch=1 Decode Honesty (`Llama-3-70B-FP8 (Batch=1)`):** `Mercury-1D` achieves **`36.0 tok/s`** (vs. H200 `56.2 tok/s`, B200 `93.7 tok/s`) — confirming `GDDR7 memory bandwidth bound (~3.07 TB/s vs H200 4.8 TB/s & B200 8.0 TB/s)`.
- **S8 Fault Telemetry:** `PCIe degraded link width: Gen5 x8 (expected x16) / Scale-out port degradation: 3/4 x 400GbE links active`

## 4. STME Block-FP8 Numerical Accuracy, 2:4 Sparsity & Tile Residency (Scenario 9)
```json
{
  "numerical": {
    "1024": {
      "mercury_rel_l2_err": 5.825500580793454e-05,
      "hopper_rel_l2_err": 0.0010491065344983587,
      "mercury_max_abs_err": 4.261276487755822,
      "hopper_max_abs_err": 37.72612023775582,
      "mercury_beats_hopper": true
    },
    "4096": {
      "mercury_rel_l2_err": 2.45618866358012e-05,
      "hopper_rel_l2_err": 0.001030423395272189,
      "mercury_max_abs_err": 6.634055647737114,
      "hopper_max_abs_err": 142.19507916891598,
      "mercury_beats_hopper": true
    }
  },
  "sparsity": {
    "dense_parity": true,
    "sparse_result": [
      28.0
    ]
  },
  "schedules": {
    "flashattention_3": {
      "kernel": "flashattention_3",
      "tile_m": 64,
      "tile_n": 128,
      "rf_used_bytes": 164352,
      "l1_used_bytes": 106760,
      "shared_sram_bytes": 409636,
      "gddr7_bytes_transferred": 811044,
      "naive_gddr7_bytes_transferred": 13664292,
      "gddr7_traffic_reduction": 16.847781378075666,
      "bank_conflict_penalty": 1.0,
      "achieved_ai_gddr7": 703.3223154354141,
      "effective_mfu": 0.68
    },
    "swiglu_mlp": {
      "kernel": "swiglu_mlp",
      "tile_m": 64,
      "tile_n": 128,
      "rf_used_bytes": 163840,
      "l1_used_bytes": 82448,
      "shared_sram_bytes": 65544,
      "gddr7_bytes_transferred": 462856,
      "naive_gddr7_bytes_transferred": 2035720,
      "gddr7_traffic_reduction": 4.398171353509515,
      "bank_conflict_penalty": 1.0,
      "achieved_ai_gddr7": 217.48296662460896,
      "effective_mfu": 0.7500000000000001
    }
  }
}
```

## 5. 1-Trillion-Token Llama-3 70B FP8 Pretraining Campaign TCO (Scenario 10, 64 GPUs)

| Cluster (64 GPUs) | Cluster Capex | Wall Power (PUE 1.25) | Campaign Days | Energy Cost (@ $0.08/kWh) | Amortized Total Cost | Cost / 1M Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 64x Mercury-1D (400GbE OAM Liquid) | $352,000 | 63.2 kW | 76.0 d | $9,226 | $33,672 | **$0.0337** |
| 64x NVIDIA H100 SXM (NVLink4 + IB) | $1,920,000 | 70.8 kW | 91.4 d | $12,418 | $172,649 | **$0.1726** |
| 64x NVIDIA B200 (NVLink5 + IB) | $2,560,000 | 97.9 kW | 37.5 d | $7,051 | $94,742 | **$0.0947** |
