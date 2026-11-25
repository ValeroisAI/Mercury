# Mercury-1: Open-Source RISC-V AI Training GPU Architecture Specification

> **Status:** Phase 1 Architectural Specification (Pre-RTL Verification)  
> **Recommended Flagship SKU:** `Mercury-1D` (Dual Unified Die OAM Module, 144 GB GDDR7 ECC, 1,152 BF16 / 2,304 FP8 Dense TFLOPS)  
> **Companion SKU (Same Mask Set):** `Mercury-1S` (Single Die PCIe Gen5 x16 Card, 72 GB GDDR7 ECC, 576 BF16 / 1,152 FP8 Dense TFLOPS)  
> **Companion Docs:** [RISC-V Open-GPU Research & Claims Audit](../research/riscv-open-gpu-landscape.md) · [ISA, Microarchitecture & Software Stack](./isa-and-software-stack.md)

---

## 1. Design Thesis & Problem Statement

Commercial AI training GPUs (NVIDIA H100/H200/B200) charge **$25,000–$40,000+ per accelerator** not primarily because logic silicon is expensive, but because of four compounding cost drivers:
1. **HBM3e + CoWoS-L Packaging Tax:** On an NVIDIA B200 (~$6,400–$6,750 estimated BOM), HBM3e (~$2,900–$3,250) and CoWoS-L (~$1,100) represent **>60% of manufacturing cost**, whereas dual ~800 mm² logic dies cost ~$900.
2. **80%+ Gross Margin & Proprietary Lock-in:** Enterprise pricing extracts 4×–6× over manufacturing BOM, reinforced by closed CUDA/NVLink/InfiniBand fabrics.
3. **Oversized Monolithic Dies & Legacy Graphics Silicon:** Reticle-limit ~814 mm² dies suffer steep yield losses without aggressive redundancy, while carrying graphics rasterization or heavy FP64 blocks irrelevant to BF16/FP8 AI training.
4. **Hopper FP8 Software Penalty:** Existing Tensor Cores truncate internal FP8 accumulators and lack hardware block-wise ($1\times 128$ / $128\times 128$) dequantization scaling, forcing custom Triton/CUDA scaling loops.

**Mercury-1's Core Thesis:** Fabricate **one compact, high-yield TSMC N4P RISC-V compute die (`453.8 mm²`)**, replace HBM3e + CoWoS-L with **organic-package UCIe-S + wide-bus Clamshell GDDR7 ECC**, defeat GDDR7's lower bandwidth with **massive on-die Shared Tensor SRAM (384 MB)**, bake **native block-FP8 + IEEE FP32 systolic accumulation** into hardware, and integrate **4 × 400GbE / RoCEv2** directly onto the die to eliminate NVSwitch/InfiniBand taxes.

---

## 2. Architectural Trade-Off Options Evaluated

To ensure honest engineering rigor, we sized three distinct physical configurations on TSMC N4P ($18,000/300mm wafer, mature $D_0 = 0.10\text{ defects/cm}^2$, Murphy yield with spare-SCC redundancy):

| Parameter | Option A: `Mercury-1S` | **Option B: `Mercury-1D` (Recommended)** | Option C: `Mercury-1H` (Rejected) |
| :--- | :--- | :--- | :--- |
| **Package & Die Topology** | 1 × Unified Compute Die (Organic BGA, PCIe Gen5 x16) | **2 × Unified Compute Dies (Organic UCIe-S MCM, OAM)** | 2 × HBM-Variant Dies (TSMC CoWoS-L 2.5D Interposer) |
| **Die Area (per die)** | 453.8 mm² (21.3 × 21.3 mm) | **453.8 mm² (21.3 × 21.3 mm)** *(same mask set as 1S)* | 496.5 mm² (22.3 × 22.3 mm) |
| **SCCs (Fabricated / Active)** | 84 / 76 | **168 / 152** (84 / 76 per die) | 208 / 192 (104 / 96 per die) |
| **Sustained Training Clock** | 1.85 GHz | **1.85 GHz** | 1.85 GHz |
| **Dense BF16 / Sparse BF16** | 575.9 / 1,151.8 TFLOPS | **1,151.8 / 2,303.6 TFLOPS** | 1,454.9 / 2,909.8 TFLOPS |
| **Dense FP8 / Sparse FP8** | 1,151.8 / 2,303.6 TFLOPS | **2,303.6 / 4,607.2 TFLOPS** | 2,909.8 / 5,819.6 TFLOPS |
| **Dense MXFP4 / TF32 / FP32 Vec** | 2,303.6 / 287.9 / 36.0 TFLOPS | **4,607.2 / 575.9 / 72.0 TFLOPS** | 5,819.6 / 727.4 / 90.9 TFLOPS |
| **Shared Tensor SRAM (+Local L1+RF)** | 192 MB (+38.0 MiB L1+RF) | **384 MB (+76.0 MiB L1+RF)** | 384 MB (+48 MiB L1/RF) |
| **External VRAM (ECC)** | 72 GB / 67.1 GiB GDDR7 (384-bit @ 32 Gbps) | **144 GB / 134.1 GiB GDDR7 (2×384-bit = 768-bit @ 32 Gbps)** | 144 GB / 134.1 GiB HBM3e (6×1024-bit @ 9.2 Gbps) |
| **External Memory Bandwidth** | 1.54 TB/s | **3.07 TB/s** *(upgradeable to 3.46 TB/s @ 36 Gbps)* | 7.07 TB/s |
| **Scale-Out Networking** | 2 × 400GbE (800 Gbps) + PCIe Gen5 | **4 × 400GbE (1.6 Tbps = 200 GB/s/dir)** + PCIe Gen5 | 4 × 400GbE (1.6 Tbps) + PCIe Gen5 |
| **Estimated TDP** | ~340 W | **~680 W** *(OCP OAM 2.0 48V)* | ~750 W |
| **Estimated Manufacturing BOM** | **~$1,238** | **~$2,456** | **~$7,263** |
| **BF16 TFLOPS / $1,000 BOM** | **465 TFLOPS/$k** | **469 TFLOPS/$k** | 200 TFLOPS/$k |

### Why We Recommend `Mercury-1D` + `Mercury-1S` (Single-Mask Family)
- **Why reject Option C (`Mercury-1H`)?** Adding HBM3e ($5,040) + CoWoS-L ($1,050) inflates BOM by **2.96×** ($7,263 vs. $2,456) for only +26% peak TFLOPS while re-introducing TSMC CoWoS-L and HBM3e allocation bottlenecks.
- **Why `Mercury-1D` + `Mercury-1S` together?** Both SKUs use the **exact same 453.8 mm² N4P compute die** (saving ~$35M–$50M in duplicate reticle mask NRE). `Mercury-1S` fits standard 350W workstation/server PCIe slots (72 GB / 67.1 GiB ECC—2.25× RTX 5090's 32 GB), while `Mercury-1D` stitches two dies over organic UCIe-S with symmetric left/right 384-bit GDDR7 routing to deliver **144 GB / 134.1 GiB VRAM + 1,152 BF16 TFLOPS**—exceeding NVIDIA H100/H200 on peak dense FLOPS and matching/exceeding H200 capacity at **5.45× lower 8-GPU pod capex ($44k vs. $240k) and 6.35× lower capex per BF16 TFLOP ($4.78 vs. $30.33/TFLOP)** (Section 6.2).

---

## 3. Physical Silicon Architecture & Floorplan (`Mercury-1 Compute Die`)

### 3.1 Area Budget & Yield Model (TSMC N4P, $300\text{ mm}$ Wafer)

| Die Block | Area (mm²) | Share | Engineering Basis (Known / Inferred) |
| :--- | :---: | :---: | :--- |
| **84 × Streaming Compute Clusters (SCCs)** | 229.4 mm² | 50.6% | ~2.73 mm²/SCC (`8×16×16` STME + `128` FP32/INT32 `Zve64f` RVV lanes + dedicated `256 KiB` Register File + dedicated `256 KiB` L1 Tensor Scratchpad + `2` RV64 schedulers; stripped of vector FP64 `Zve64d` & raster units). **76 active / 8 spare** (~9.5% redundancy). |
| **192 MB Shared Tensor SRAM (64 × 3 MB)** | 58.2 mm² | 12.8% | ~3.30 MB/mm² effective density (bridging TSMC N4P HP 26.7 Mb/mm² and HD 31.8 Mb/mm² plus ECC/bank periphery). |
| **12 × 32-bit GDDR7 PHYs & MCs (384-bit)** | 33.6 mm² | 7.4% | ~2.8 mm² per 32-bit PAM3 PHY + ECC controller along three perimeter edges. |
| **NoC / PCIe Gen5 / 2×400GbE / UCIe-S** | 84.0 mm² | 18.5% | PCIe Gen5 x16 (~28 mm²), Global RISC-V Command/DMA (~18 mm²), 2×400GbE PAM4 SerDes (~16 mm²), 12 mm UCIe-S D2D shoreline (~22 mm²). |
| **Global Routing, Clock Tree & Power Grid** | 48.6 mm² | 10.7% | ~12% physical implementation overhead across top-level NoC mesh. |
| **Total Die Size (21.3 × 21.3 mm)** | **453.8 mm²** | **100%** | **117 Gross Dies/Wafer** (3 mm edge exclusion, 0.1 mm kerf). |

- **Yield Math ($D_0 = 0.10\text{ defects/cm}^2$):**
  - Perfect-die Murphy yield: $Y_{\text{perf}} = \left(\frac{1 - e^{-4.538 \times 0.10}}{4.538 \times 0.10}\right)^2 = \mathbf{64.6\%}$
  - Effective yield with 8/84 spare SCC defect salvage: **$Y_{\text{eff}} \approx 70.9\%$** ($\approx \mathbf{82.9}$ good dies per $18,000 wafer = **~$217 silicon cost per die**).

### 3.2 Logical Floorplan & Dual-Die Organic MCM (`Mercury-1D`)

```
                    400GbE #0   400GbE #1                     400GbE #2   400GbE #3
                        │           │                             │           │
     ┌──────────────────┴───────────┴──────────┐   ┌──────────────┴───────────┴──────────────────┐
     │        GDDR7 Ch 0-3 (128-bit, 24GB)     │   │         GDDR7 Ch 12-15 (128-bit, 24GB)      │
     ├──────────┬───────────────────┬──────────┤   ├──────────┬───────────────────────┬──────────┤
     │          │  84 SCCs (76 act) │          │ U │          │   84 SCCs (76 act)    │          │
     │ GDDR7    │  + 2D Mesh NoC    │  UCIe-S  │ C │  UCIe-S  │   + 2D Mesh NoC       │ GDDR7    │
     │ Ch 4-7   │  + 192 MB Shared  │   PHY    │═I═│   PHY    │   + 192 MB Shared     │ Ch 16-19 │
     │ (128-b,  │    Tensor SRAM    │ (1.2TB/s │ e │ (1.2TB/s │     Tensor SRAM       │ (128-b,  │
     │  24GB)   │  + RV64 Cmd Proc  │  bidir)  │-S │  bidir)  │   + RV64 Cmd Proc     │  24GB)   │
     ├──────────┴───────────────────┴──────────┤   ├──────────┴───────────────────────┴──────────┤
     │       GDDR7 Ch 8-11 (128-bit, 24GB)     │   │        GDDR7 Ch 20-23 (128-bit, 24GB)       │
     └──────────────────┬──────────────────────┘   └──────────────────────┬──────────────────────┘
                        │                                                 │
                  PCIe Gen5 x16                                   PCIe Gen5 x16 (Host/Aux)
```

- **Symmetric PCB Layout:** Die 0 drives 12 GDDR7 channels (`384-bit`, 24 × 24Gb clamshell packages = 72 GB) on the left perimeter; Die 1 drives 12 GDDR7 channels (`384-bit`, 24 × 24Gb clamshell packages = 72 GB) on the right perimeter. No PAM3 memory signals cross between left and right halves.
- **Organic UCIe-S D2D Bridge:** 12 mm facing beachfront at a conservative **100 GB/s/mm bidirectional** (50 GB/s/mm/dir, standard organic laminate bump pitch 110 µm) delivers **1.2 TB/s bidirectional (600 GB/s each way)** between Die 0 and Die 1 without a silicon interposer.

---

## 4. Memory & Bandwidth Hierarchy: Solving the GDDR7 Wall

| Tier | Capacity (`1S` / `1D`) | Aggregate Bandwidth (`1S` / `1D`) | Role in AI Training |
| :--- | :--- | :--- | :--- |
| **L0: SCC Register File & L1 Scratchpad** | 38.0 MiB / **76.0 MiB** (`256 KiB` RF + `256 KiB` L1 per SCC) | 18.0 TB/s / **36.0 TB/s** (`128 B/clk`) | Separate `256 KiB` RF (`16` warps scalar + `Zve64f` SIMT + STME accumulators) and `256 KiB` L1 Tensor Scratchpad (`19.0 MiB/die` L1 holds double-buffered FlashAttention tiles cleanly). |
| **L1: Distributed Shared Tensor SRAM** | 192 MB / **384 MB** (`64×3 MB/die`) | 7.58 TB/s / **15.16 TB/s** (`64 B/clk/bank`) | Holds FlashAttention-3 $Q/K/V$ tiles, fused RMSNorm/SwiGLU intermediates, FP8 $1\times 128$ block scales, and reduction buffers completely off GDDR7. |
| **L2: External GDDR7 ECC VRAM** | 72 GB (67.1 GiB) / **144 GB (134.1 GiB)** (24Gb clamshell; **192 GB** w/ 32Gb) | 1.54 TB/s / **3.07 TB/s** (`32 Gbps PAM3`) | Model weights, FP32 AdamW master/optimizer states, gradients, checkpointed activations. Inline hardware LZ4-block/zero-run + FP8 scale compression yields ~1.3×–1.5× effective BW on sparse/checkpointed tensors. |

### Roofline Analysis (Why 384 MB SRAM Changes Everything)
- **External GDDR7 Ridge Point (`Mercury-1D`):**
  - BF16: $\frac{1,151.8\text{ TFLOPS}}{3.072\text{ TB/s}} = \mathbf{375\text{ FLOPs/Byte}}$ (vs. H100's 295 F/B and B200's 281 F/B).
  - FP8: $\frac{2,303.6\text{ TFLOPS}}{3.072\text{ TB/s}} = \mathbf{750\text{ FLOPs/Byte}}$ (vs. H100's 590 F/B).
- **On-Die Shared Tensor SRAM Ridge Point (`Mercury-1D`):**
  - BF16: $\frac{1,151.8}{15.16} = \mathbf{76\text{ FLOPs/Byte}}$ | FP8: $\frac{2,303.6}{15.16} = \mathbf{152\text{ FLOPs/Byte}}$.
- **Training Kernel Consequence:** Unfused attention or small-batch elementwise ops would stall at 375–750 F/B on GDDR7 alone. With **384 MB on-die Tensor SRAM** (vs. ~83 MB total SRAM on H100), Async Tensor DMA (`m.dma.*`) prefetches weight tiles once while $Q K^T V$ softmax blocks and SwiGLU/RMSNorm residuals stay inside the 15.16 TB/s SRAM fabric, keeping transformer GEMMs compute-bound above **>50% BF16 MFU**.

---

## 5. Power Budget & Manufacturing BOM Breakdown

### 5.1 Bottom-Up TDP Model (`Mercury-1S` vs. `Mercury-1D`)
*(Modeled at 1.85 GHz TSMC N4P sustained training load: ~0.31 W/BF16-TFLOP dynamic compute, ~48 W leakage per 500 mm² die, ~40 W/TB/s GDDR7 subsystem power, ~26 W per 800G SerDes + PCIe.)*

| Subsystem | `Mercury-1S` (Single Die PCIe) | `Mercury-1D` (Dual Die OAM) |
| :--- | :---: | :---: |
| Compute & Clock/NoC Dynamic | 179 W | 357 W |
| Static Leakage (TSMC N4P) | 43 W | 86 W |
| GDDR7 DRAM + PAM3 PHYs | 75 W | 150 W |
| 400GbE SerDes + PCIe + D2D | 44 W | 88 W |
| **Rounded Thermal Design Power (TDP)** | **340 W** (Air-cooled PCIe) | **680 W** (OCP OAM 2.0 48V Direct-to-Chip Liquid / High-CFM Air) |

### 5.2 Manufacturing Bill of Materials (BOM) vs. Commercial Incumbents

| Cost Line Item | `Mercury-1S` (72 GB / 67.1 GiB) | `Mercury-1D` (144 GB / 134.1 GiB) | NVIDIA H100 (80 GB) | NVIDIA B200 (192 GB) |
| :--- | :---: | :---: | :---: | :---: |
| **Logic Silicon (Foundry Wafer / Yield)** | $217 | $434 | ~$350–$450 (est.) | ~$900 (2×800 mm²) |
| **VRAM (GDDR7 @ ~$8/GB vs. HBM3e)** | $576 | $1,152 | ~$1,800–$2,000 (est.) | ~$2,900–$3,250 |
| **Package Substrate / Interposer** | $90 (Organic) | $220 (Organic UCIe-S) | ~$450 (CoWoS-S est.) | ~$1,100 (CoWoS-L) |
| **PCB, 600–750W VRM, Cooler/OAM** | $240 | $420 | ~$500 | ~$1,100–$1,400 |
| **OSAT Assembly, Burn-In & Test** | $115 | $230 | ~$150–$200 | ~$300 |
| **Total Estimated Manufacturing BOM** | **~$1,238** | **~$2,456** | **~$3,320** *(Known est.)* | **~$6,400–$6,750** *(Known est.)* |
| **Target Street / Cluster Price** | **~$2,400** *(~48% GM)* | **~$4,700** *(~48% GM)* | ~$25,000–$30,000+ | ~$30,000–$40,000+ |

---

## 6. AI Training Cluster Topology & Llama-3 70B Economics

### 6.1 Parallelism Strategy on `Mercury-1D`
1. **Intra-Package `TP=2` (Optional High-Speed Tensor Parallelism):** Each `Mercury-1D` naturally exposes either a single 144 GB unified device or two 72 GB NUMA chiplets connected by **600 GB/s/dir UCIe-S**—ideal for low-overhead `TP=2`.
2. **Inter-Node Scale-Out (`4 × 400GbE` = `200 GB/s TX + 200 GB/s RX`):** Eliminates proprietary NVSwitch and InfiniBand switches using flat standard leaf-spine 400GbE / RoCEv2 Ethernet switches plus Mercury's Hardware Collective AllReduce Engine.
   - **Communication-to-Compute Check (Llama-3 70B):**
     - **Pure `TP=8` across Ethernet (`mb=4, seq=4096`):** Raw AllReduce volume is 150.3 GB/step (~0.75 s vs. 1.49 s compute), which would bottleneck un-overlapped TP=8.
     - **Recommended `TP=2 (intra-card) + FSDP` (or `PP=2 + FSDP`):** Gradient ReduceScatter + AllGather for FSDP with `131K tokens/GPU` global batch transfers ~35 GB/GPU (**0.175 s** over 200 GB/s Ethernet vs. **95.6 s** compute = **<0.2% exposed overhead!**).
3. **70B Parameter Memory Residency (Verified in Binary GiB):**
   - **8 × `Mercury-1D` Pod (1,152 GB decimal = `1,072.9 GiB` total VRAM, `134.1 GiB/GPU`):** Full 16 B/param FP32 AdamW needs `1,043.1 GiB` (`130.4 GiB/GPU`), leaving only **`3.7 GiB/GPU`**—insufficient for large-batch activations. Using **compact FP8/BF16 optimizer states (11 B/param: FP32 master + BF16 m/v + BF16 grad + FP8 weights)** uses **`717.1 GiB` total (`89.6 GiB/GPU`)**, leaving **`44.5 GiB/GPU`** for activations!
   - **16 × `Mercury-1D` Pod (2,304 GB decimal = `2,145.8 GiB` total VRAM, ~$88k pod capex):** Runs full uncompromised **16 B/param FP32 AdamW** at **`65.2 GiB/GPU`**, leaving **`68.9 GiB/GPU`** for large-batch activations—at **~2.7× lower capex than a single 8×H100 node**!

### 6.2 8-GPU Training Pod Comparison (Inferred from Sizing & Street Benchmarks)

| Metric | `8 × Mercury-1D` Pod | `8 × NVIDIA H100` Node | `8 × NVIDIA B200` Node |
| :--- | :---: | :---: | :---: |
| **Aggregate BF16 Dense PFLOPS** | **9.21 PFLOPS** | 7.91 PFLOPS | 18.00 PFLOPS |
| **Aggregate FP8 Dense PFLOPS** | **18.43 PFLOPS** | 15.83 PFLOPS | 36.00 PFLOPS |
| **Aggregate VRAM Capacity** | **1,152 GB** (GDDR7 ECC) | 640 GB (HBM3) | 1,440 GB (HBM3e) |
| **Estimated Pod Capex (incl. switches/host)** | **~$44,000** | ~$240,000 | ~$320,000 |
| **Capex per BF16 TFLOP** | **$4.78 / TFLOP** | $30.33 / TFLOP | $17.78 / TFLOP |
| **Projected Llama-3 70B MFU (BF16 / FP8)** | **~54.3% / ~40.0%** | ~52% / ~42% | ~50% / ~45% |
| **Projected 70B Training Throughput (FP8)** | **~2,191 tok/s/GPU** *(17.5k pod)* | ~1,980 tok/s/GPU | ~4,820 tok/s/GPU |

---

## 7. Open Risks & Phase 2 Implementation Roadmap

| Risk Area | Root Cause | Mitigation & Phase 2 Verification Plan |
| :--- | :--- | :--- |
| **GDDR7 SI / Thermal Density** | 48 clamshell PAM3 chips @ 32 Gbps near 680W TDP | Enforce strict symmetric 2×384-bit edge routing (≤ 35 mm channels); target **OCP OAM 2.0 48V DC** power rails with direct-to-chip cold-plate liquid cooling (standard for 700W–1000W+ OAM trays). |
| **Triton / PyTorch Compiler Maturity** | Non-CUDA ecosystems lose 20–40% real MFU to kernel gaps | Build **Phase 2 Behavioral/Roofline Simulator + MLIR/Triton Kernel Lowering Prototype** + reference FlashAttention/Block-FP8 GEMM microbenchmarks before RTL freeze. |
| **Small-Batch Inference Decode Ceiling** | 3.07 TB/s GDDR7 lags H200 (4.8 TB/s) and B200 (8.0 TB/s) on batch=1 decode | `Mercury-1` targets **high-arithmetic-intensity AI training & batched inference**, where 384 MB on-die SRAM + 144 GB capacity keeps compute units saturated. |

### Proposed Phase 2 Modules (Pending User Review & Approval)
1. **Executable Physics, Roofline & Training Simulator (`python`):** Parameterized cycle-approximate workload simulator modeling FlashAttention, Block-FP8 GEMM, FSDP/TP/PP overlap, and BOM sensitivity.
2. **RISC-V `Mercury-1` SCC & Systolic RTL Prototype (`SystemVerilog` or `Chisel`):** Synthesizable single-SCC + `16×16` Block-FP8 STME with full FP32 accumulator + hardware verification testbench.
3. **Interactive Architecture Explorer Dashboard (Web UI):** Live Roofline, Die Floorplan, Memory Hierarchy, and Cluster $/TFLOP TCO comparator backed by the simulator outputs.
