# Research Report: RISC-V Open-Source GPU Architectures & AI Training Claims

> **Methodology (`Researcher` standard):** Every factual claim below separates **Known** (verified from public repositories, papers, or technical specifications retrieved during research), **Inferred** (engineering calculations or architectural extrapolations derived from verified data), and **Unknown** (unverified proprietary or unreleased parameters). Source URLs/domains and verbatim evidence snippets from retrieved search summaries are cited throughout.

---

## 1. Executive Verdict: Claim vs. Reality

A widespread claim circulating in open-hardware discussions is that *"RISC-V based Open-Source GPU Architecture"* outperforms existing commercial GPUs. **Our evidence-based verdict is mixed—true on economics and programmability potential, false on raw out-of-the-box silicon throughput today:**

| Dimension | Claim ("Beats current chips") | Verified Reality | Confidence |
| :--- | :--- | :--- | :--- |
| **Raw TFLOPS (Existing OSS RTL)** | Beats NVIDIA/AMD GPUs | **FALSE today.** Existing open-source RISC-V GPUs (Vortex, Ventus, e-GPU) run on FPGAs or academic test chips at ~200–300 MHz, achieving **25.6–128 GFLOPS** (0.025–0.128 TFLOPS) vs. **989–2,250 BF16 TFLOPS** on NVIDIA H100/B200. | **High** |
| **ISA & Licensing Freedom** | Open, royalty-free, no CUDA lock-in | **TRUE.** Built on modular `RV32/RV64` + `RVV 1.0` + open SIMT/matrix extensions with Apache/MIT RTL and open LLVM/PoCL/VOLT/Triton/chipStar compilers. | **High** |
| **BOM & TCO Cost Potential** | Much cheaper than H100/B200 | **TRUE (if fabricated on commercial 4nm/5nm with GDDR7 + Ethernet).** Nearly **48–50% of NVIDIA B200's ~$6,400–$6,750 BOM is HBM3e (~$2,900–$3,250) + CoWoS-L (~$1,100)**, while the dual ~800 mm² logic dies cost ~$900, and NVIDIA sells systems at ~$30k–$40k+ (~80–83% gross margin). | **High** |
| **AI Training Suitability** | Ready to replace CUDA clusters | **PARTIALLY TRUE (Architectural, Not Turnkey).** Vortex 3.0 adds WGMMA + 2:4 sparsity + FP8/BF8/TF32; Ventus adds RVV + tensor cores; Tenstorrent Blackhole ships 664–745 FP8 TFLOPS at $1,399/card on GDDR6 + Ethernet. Commercial AI training requires commercial ASIC tape-out, PyTorch/Triton compiler maturity, and hardware FP8/BF16 numerical precision. | **High** |

**What evidence would change this verdict:** Public silicon tape-out benchmarks on a sub-7nm commercial node (e.g., TSMC N5/N4P) from an open-source RISC-V GPU RTL project (Vortex 3.0, Ventus, or OpenRVGPU) demonstrating >500 dense BF16 TFLOPS and sustained >40% MFU on multi-node PyTorch LLM training.

---

## 2. Survey of Key RISC-V Open-Source GPU & Accelerator Architectures

### 2.1 `"RISC-V based Open-Source GPU Architecture and Design Exploration"` (`OpenRVGPUCourse`)
- **Known:** The phrase matches the open course/project repository **`github.com/chenweiphd/OpenRVGPUCourse`**, titled *"RISC-V based Open-Source GPU Architecture and Design Exploration"*.
  - Quoted evidence (`github.com` search summary): *"The primary GitHub repository associated with this topic is: GitHub Repository: `chenweiphd/OpenRVGPUCourse` (titled 'RISC-V based Open-Source GPU Architecture and Design Exploration')."* and *"Key architecture exploration projects mentioned in the course: `OpenRVGPU`, `OpenTensorCore`, `Open3DNoC`, `OpenLLMRoad`."*
  - Quoted evidence on toolchain: *"Utilizing C-Model simulation, FPGA prototyping, and open-source EDA (OpenEDA / OpenROAD flows) from architectural modeling to physical layout"* and *"Examining GPU compilers, Triton, and optimization strategies for workloads like DeepSeek and Transformer-based models."*
- **Engineering Assessment:** `OpenRVGPUCourse` is an **architectural exploration and education system** covering LLM/DeepSeek workloads, Tensor Cores, 3D NoCs, and Triton compilers—not a fabricated commercial chip beating H100/B200 in silicon today.

### 2.2 Vortex GPGPU (`vortexgpgpu/vortex`, Georgia Tech)
- **Known:**
  - Full-stack open-source SIMT RISC-V GPGPU supporting OpenCL 1.2, Vulkan (`vortexpipe`/Mesa), HIP (`chipStar`), and the **VOLT** LLVM-based SIMT compiler (`github.com/vortexgpgpu/vortex`, `arxiv.org`, `gatech.edu`).
  - **Vortex 3.0** added hardware-accelerated command processing, async barriers, **WGMMA (Warp-Group Matrix Multiply-Accumulate)**, **2:4 structured sparsity**, and mixed-precision formats including **FP8, BF8, and TF32** (`phoronix.com`, `github.com`, `gatech.edu`).
  - Quoted evidence (`riscvschool.com` / `emergentmind.com` / `arxiv.org` search summaries): *"Academic and baseline configurations typically scale from tens to over a hundred GFLOPS. For instance, baseline or mid-scale FPGA setups register roughly 128 GFLOPS (e.g., configurations running 1024 total threads at ~250 MHz)."* and *"Vortex 3.0 introduced: WGMMA (Warp Group-Level Matrix Multiply-Accumulate) and tensor core structured sparsity (such as 2:4 sparsity support)."*
- **Design Lesson for Mercury:** Vortex proves minimal RISC-V SIMT extensions (`vx_wspawn`, `vx_tmc`, divergence split/join) + WGMMA + 2:4 sparsity work cleanly with LLVM/OpenCL/HIP, but FPGA clocks (~250 MHz) bottleneck throughput by ~8×–10× relative to a TSMC N4P ASIC (~1.85 GHz) and lack deep systolic tensor arrays per core.

### 2.3 Ventus GPGPU (`THU-DSP-LAB/ventus-gpgpu`, Tsinghua University)
- **Known:**
  - Open-source GPGPU written in Chisel HDL built directly on the **RISC-V Vector Extension (RVV)** rather than inventing an unrelated SIMD ISA (`riscv-europe.org`, `github.com`).
  - Quoted evidence (`riscv-europe.org` / `scribd.com` search summaries): *"Ventus adopts and extends the RISC-V Vector Extension (RVV) as its base SIMT/vector-thread architecture"* and *"Under a typical 32-thread configuration, its tensor core packs 128 multipliers and 128 accumulators for 4x8x4 matrix operations."*
  - Scaled on Xilinx VU19P FPGA clusters up to 16 SMs / 256 warps with a Terapines/LLVM OpenCL-to-RVV compiler and PoCL driver (`semanticscholar.org`, `gitlink.org.cn`).
- **Design Lesson for Mercury:** Mapping SIMT warp lanes directly onto standard **RVV 1.0 vector registers** avoids reinventing compiler vectorization and enables dual-use FP32/INT32 vector + tensor execution inside the same cluster.

### 2.4 Commercial & Embedded RISC-V GPU/AI Reference Points
- **X-Silicon (`NanoTile C-GPU™`):** Fuses a RISC-V Vector CPU core with custom graphics/ML instructions and native Vulkan support (`riscv.org`, `design-reuse.com`), targeting low-power edge/Android rather than multi-hundred-watt datacenter training.
- **EPFL `e-GPU`:** Configurable TinyAI RISC-V GPU in TSMC 16nm @ 300 MHz (~28 mW), targeting IoT/bio-signals (`arxiv.org`, `themoonlight.io`)—not datacenter training.
- **Tenstorrent (`Wormhole` / `Blackhole`):** Most commercially relevant proof-point of RISC-V AI economics. Uses a 2D mesh of **Tensix cores** (each embedding RISC-V control cores + matrix/vector engine + large local SRAM), open `TT-Metalium`/`TT-Forge` software, **GDDR6** instead of HBM, and **4×800G Ethernet** ports instead of NVLink/InfiniBand (`tenstorrent.com`, `gpusmith.com`, `spheron.network`).
  - Quoted evidence (`gpusmith.com`, `spheron.network`, `hotchips.org` search summaries): *"~664 – 745 TFLOPS (Block FP8)"* and *"~373 TFLOPS (BF16 Matrix)"*, *"GDDR6 (32 GB p150 card, ~512 GB/s)"*, *"4x 800 Gbps QSFP-DD (No host PCIe bottleneck for scale)"*, *"A single Blackhole p150 developer card runs around $1,399."*

---

## 3. Why Current Datacenter GPUs Are So Expensive: Silicon, Memory & Interconnect Benchmarks

### 3.1 Commercial Baseline Specifications & BOM Breakdown (Known)

| Accelerator | Node & Die | Peak Dense Compute | Memory & Bandwidth | Interconnect | Est. BOM / Street Price |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA H100 SXM** | TSMC 4N, ~814 mm², 80B Tr | 989 BF16 / 1,979 FP8 TFLOPS | 80 GB HBM3 (3.35 TB/s) | NVLink 4 (900 GB/s), 700W | ~$3,320 BOM / ~$25k–$30k+ |
| **NVIDIA H200 SXM** | TSMC 4N, ~814 mm² | 989 BF16 / 1,979 FP8 TFLOPS | 141 GB HBM3e (4.8 TB/s) | NVLink 4 (900 GB/s), 700W | >$30k+ street |
| **NVIDIA B200** | TSMC 4NP, 2×~800 mm², 208B Tr | 2,250 BF16 / 4,500 FP8 / 9,000 FP4 TFLOPS | 180–192 GB HBM3e (8.0 TB/s) | NVLink 5 (1.8 TB/s), 1,000W | ~$6,400–$6,750 BOM / ~$30k–$40k+ |
| **AMD MI300X** | TSMC 5nm chiplets | ~1,307 BF16 / 2,615 FP8 TFLOPS | 192 GB HBM3 (5.3 TB/s) | Infinity Fabric, 750W | ~$2.00–$4.85/hr cloud |
| **NVIDIA RTX 5090** | TSMC 4NP | ~210 BF16 dense (419 sparse) / 1,676 FP4 dense | 32 GB GDDR7 (1.79 TB/s, 512-bit) | PCIe Gen5 x16, 575W | $1,999 MSRP |
| **Tenstorrent Blackhole p150** | Advanced TSMC node | ~373 BF16 / 664–745 Block-FP8 TFLOPS | 32 GB GDDR6 (~512 GB/s) | 4×800G Ethernet, 300W | $1,399 card |

**Key Economic Evidence:**
- Quoted (`siliconanalysts.com` / `substack.com` search summaries): *"NVIDIA B200 Production Cost: Estimated between $6,400 to $6,750 per GPU module: HBM3e Memory accounts for nearly 48% to 50% of the total manufacturing cost (~$2,900 to $3,250)... CoWoS-L bridge packaging... costing roughly $1,100... The dual ~800 mm² compute dies account for less than 15–20% of the raw production expense (~$900)."* and *"NVIDIA... preserving high gross margins (~80%–83%)."*
- Quoted (`amcompute.com` / `lovechip.com` / `gigagpu.com` search summaries): *"HBM3e (Very Premium — ~$30 to $45+ per GB)"* vs. *"GDDR7... typically costing a fraction of HBM (roughly one-third to one-fourth the price per GB)"* with *"PAM3 encoding... lower operating voltage (1.1V, down from 1.35V on GDDR6X)... 32 to 40+ Gbps."*
- Quoted (`runpod.io` search summary on RTX PRO 6000 Blackwell): *"96 GB GDDR7 (with ECC support)... Clamshell Mode / Memory Interface: 512-bit memory bus"* proving high-capacity GDDR7 clamshell ECC boards work in production.

### 3.2 Foundry, Packaging, & AI Training Precision Facts (Known)
- **TSMC N5 / N4P Wafer & Density:** ~$17,500–$18,500 per 300mm wafer (`siliconanalysts.com`, `bits-chips.com`); N4P offers ~6% density uplift over N5 (`eeworld.com.cn`), effective full-chip density ~130–135 MTr/mm² (`angstronomics.com`), HD 6T SRAM ~31.8 Mb/mm² and HP 6T SRAM ~26.7 Mb/mm² (`semianalysis.com`), and mature N5/N4 defect density $D_0 \approx 0.09\text{–}0.15\text{ defects/cm}^2$ (`siliconanalysts.com`).
- **UCIe Standard Organic Package (UCIe-S):** Avoids CoWoS-L silicon interposers entirely, using organic laminate with 100–130 µm bump pitch, up to 25 mm reach, and **22–125 GB/s/mm** baseline beachfront bandwidth density (scaling up to 448 GB/s/mm in advanced 3nm demonstrations) (`hotchips.org`, `insidedeeptech.com`, `researchgate.net`).
- **FP8 AI Training Physics (DeepSeek-V3):** Standard Hopper Tensor Cores suffer precision truncation in internal FP8 accumulators and lack native hardware support for block-wise ($1\times 128$ activation, $128\times 128$ weight) dequantization scaling factors (`arxiv.org`, `colfax-intl.com`, `deepwiki.com`). DeepSeek-V3 had to write custom Triton/CUDA kernels fusing scaling loops and promoting to FP32 accumulation to achieve stable training accuracy.

---

## 4. Architectural Conclusion: Where **Mercury-1** Can Genuinely Beat Incumbents

An open-source RISC-V GPU **cannot** beat an H100/B200 in raw per-chip TFLOPS simply by putting Vortex/Ventus on an FPGA. However, an **open RISC-V ASIC (`Mercury-1`)** CAN beat commercial GPUs decisively on **Training TFLOPS per Dollar ($/PFLOP-day), TCO, Supply-Chain Freedom, and FP8 Training Efficiency** by combining five proven architectural principles:

1. **Eliminate the HBM3e + CoWoS-L Bottleneck:** Replace ~$4,000+ of HBM3e + CoWoS-L with **dual-die organic UCIe-S MCM + 768-bit Clamshell GDDR7 ECC (144 GB @ 3.07 TB/s)**—cutting memory + packaging BOM by **~77%**.
2. **Defeat the GDDR7 Memory Wall with Massive On-Die Tensor SRAM (384 MB):** Because N4P logic/SRAM dies are cheap (~$217/die at ~453.8 mm² vs. ~$4,000 HBM+CoWoS), allocate **192 MB/die (384 MB total)** of distributed Shared Tensor SRAM (~15.16 TB/s aggregate) to keep FlashAttention-3 tiles and fused MLP states on-chip—achieving **~4.6× more total Shared Tensor SRAM than NVIDIA H100 (~83 MB total SRAM)**.
3. **Strip Non-Training Deadweight:** Remove legacy graphics pipelines (RASTER/ROP/RT cores) and heavy FP64 matrix arrays; invest that area into **8 × (16×16) Systolic Tensor Matrix Engines (`STME`) + 128 RVV FP32 vector lanes per SCC**, hitting **1,152 TFLOPS BF16 / 2,304 TFLOPS FP8 dense** on Mercury-1D (surpassing H100's 989 / 1,979 TFLOPS).
4. **Fix Hopper's FP8 Training Hardware Flaw:** Build **native hardware $1\times 128$ / $128\times 128$ block-wise FP8 scale-factor fusion + full IEEE-754 FP32 accumulation** directly into the systolic MAC pipeline, saving Triton/CUDA software overhead during LLM training.
5. **Kill the NVLink / InfiniBand Tax:** Integrate **4 × 400GbE / UALink-over-Ethernet RoCEv2 SerDes (1.6 Tbps = 200 GB/s full-duplex)** plus a hardware collective AllReduce engine directly on the OAM chiplets, enabling flat Ethernet AI training pods.

---

## 5. Known / Inferred / Unknown Summary Matrix

| Category | Items | Confidence |
| :--- | :--- | :--- |
| **Known (Source-Backed)** | Vortex 3.0 (WGMMA, 2:4 sparsity, VOLT, ~128 GFLOPS FPGA); Ventus (RVV+Chisel, 4×8×4 tensor core); OpenRVGPUCourse (OpenRVGPU/OpenTensorCore/Open3DNoC/OpenLLMRoad); Tenstorrent Blackhole ($1,399, GDDR6, 4×800G); NVIDIA H100/H200/B200/RTX5090/RTX PRO 6000 specs & BOM ($6.4k–$6.75k B200 BOM, 48–50% HBM3e); TSMC N4P wafer ($17.5k–$18.5k) & SRAM density (26.7–31.8 Mb/mm²); UCIe-S organic density; DeepSeek-V3 block-FP8 + FP32 accumulation findings. | **High** |
| **Inferred (Derived in Mercury Sizing Model)** | Exact Mercury-1 die area (~453.8 mm²/die), Murphy yield with SCC redundancy (~70.9%), bottom-up TDP (~340W 1S / ~680W 1D), BOM estimates ($1,238 1S / $2,456 1D), and projected Llama-3 70B MFU (~40%–54%). | **Medium-High** (sensitive to foundry contract discounts & memory market pricing) |
| **Unknown (Requires RTL Synthesis & Silicon Validation)** | Exact place-and-route clock closure on TSMC N4P standard-cell libraries, GDDR7 clamshell board SI margin at >32 Gbps under sustained 680W thermal load, actual PyTorch/Triton compiler pass overhead prior to RTL tape-out. | **Low / Open** (to be tested in Phase 2 behavioral/cycle simulator & RTL) |
