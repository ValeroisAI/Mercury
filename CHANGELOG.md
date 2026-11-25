# Changelog

All notable changes to the **Mercury** open-source RISC-V AI Training GPU project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- A leaf/spine FSDP sensitivity model for ring placement, port loss, and the memory cost of deferred synchronization or parameter resharding.
- Multi-seed, multi-distribution CPU validation of FP8 training with a shared quantizer and an exact-accumulation control.
- A SystemVerilog Shared Tensor SRAM bank mapper and round-robin arbiter, with generic elaboration and bounded formal checks.
- A SystemVerilog DMA token scoreboard with a generation-tagged issue, completion, and wait protocol.
- Exact FP8 products, ordered FP32 group sums, an interleaved four-stage FP32 adder, a six-stage FP32 FMA, and a fused group-boundary accumulator, with numerical and bounded RTL checks.
- Scaled scalar FP8 dot prototypes with rounded products of binary32 scales, group descriptors, backpressure, and reset checks.
- An evidence checklist for FPGA integration, N4P timing and area, Extended OAM-L board validation, and cluster qualification.
- A single RTL regression command and reproducible UltraScale+ primitive mapping for ten standalone RTL tops.
- An SRAM payload prototype for 512-bit lines with byte strobes, tagged readback, and requester backpressure.

### Changed

- Reorganized the README around reproducible evidence, physical constraints, and the pre-silicon scope.
- Replaced the SRAM arbiter's serial scan with parallel round-robin selection. The fixed-cycle interface and physical bank mapping stay the same.
- Clarified the Workbench model data and concept render, added keyboard navigation to the qualification tabs, and improved active-control contrast in dark mode.

## [0.4.0] - 2026-11-25

### Added
- **PCB Physical Courtyard & NRE Volume Economics (`mercury_sim/physical_bom.py`):**
  - Quantified backside + topside courtyard fill across `102×165 mm` OAM v1.5 (`95.3%` overflow), `102×170 mm` OAM r2.0 (`92.5%` overflow), `140×185 mm` **Extended OAM-L** (`53.0% / 61.9%` pass, `32.0 mm` traces), `96 GB / 32-chip` OAM v1.5 fallback (`69.4% / 68.2%` pass), and `111×267 mm` PCIe (`27.2% / 18.8%` pass) (`19` tests in `tests/test_physical_bom.py`).
  - Added `$85M` NRE (`$35M` masks + `$22M` SerDes/GDDR7/UCIe IP + `$28M` software) amortization curves at 45% gross margin (`$19,920` at `10k` -> `<$10k` at `27,924` -> `<$8k` at `43,725` -> `$6,011` at `100k` -> `$5,084` at `250k`).
- **Cycle-Level `256×128` A-Multicast & 20-Step PyTorch `AdamW` Convergence (`mercury_sim/`):**
  - `microarch_cycle.py`: Added `multicast_256x128` (`2,022.0` FP8 TFLOPS, `87.8%` peak, Shared SRAM traffic `11.62 -> 4.36 TB/s`) and aligned terminology to `"cycle-level architectural model"`.
  - `pytorch_bridge.py`: Added 20-step `torch.optim.AdamW` convergence isolating FP8 quantization (`~27%` update drift) from accumulator rounding (`3.96e-8` vs. `1.74e-3` step-1 grad error; `15` tests in `tests/test_microarch_and_torch.py`).
- **Interactive Engineering Workbench (`dashboard/`):**
  - Static HTML/CSS/SVG/JS workbench (`DESIGN.md` token contract) displaying live data from `scripts/export_dashboard_data.py` across light/dark themes and `1440px`/`390px` responsive layouts.

### Changed
- **Honest Engineering & Scope Boundaries:** Replaced `"Target Street Price"` framing with explicit Hardware-Only BOM vs. NRE amortization, updated `Mercury-1D` form factor to `Extended OAM-L (140×185 mm)`, corrected `0.3.0` `"Cycle-Accurate"` wording to `"Cycle-Level Architectural"`, and documented `"Current Scope & Not Yet Built"` across `README.md`, `docs/`, `scripts/generate_charts.py`, and `scripts/run_qualification.py`.

## [0.3.0] - 2026-11-25

### Added
- **Open RISC-V `Xmer*` ISA Toolchain & SIMT Virtual Machine (`mercury_isa/`):**
  - `encoding.py`: Bit-exact 32-bit encoder/decoder for `RV64I/M` scalar, `Zve64f`/`Zvfbfmin` vector, and custom `Xmersimt` (`0x0b`), `Xmerstme` (`0x2b`), `Xmerdma` (`0x5b`), and `Xmercoll` (`0x7b`) opcodes.
  - `assembler.py`: Two-pass `.masm` assembler with label resolution and round-trip disassembler.
  - `vm.py`: Multi-warp `MercuryVM` enforcing SIMT `m.split`/`m.join` reconvergence stacks, `m.wspawn`/`m.bar`, strict async `m.dma.wait`/`m.mma.wait` hazard checks, and `m.mma.fp8.blk` execution.
  - Runnable sample kernels: `examples/kernels/divergent_relu_norm.masm` and `examples/kernels/fused_fp8_gemm_flash_tile.masm` with FP64 parity verification (`63` tests in `tests/test_mercury_isa.py`).
- **Cycle-Accurate Microarchitecture & PyTorch Bridge (`mercury_sim/`):**
  - `microarch_cycle.py`: Cycle-level dual-scheduler SCC scoreboard, 64-bank Shared Tensor SRAM crossbar (demonstrating `28.5×` speedup via XOR bank swizzling), 12-channel GDDR7 `tREFI`/`2× tREFI` controller, and UCIe-S spill simulator.
  - `pytorch_bridge.py`: Real `torch.nn` pre-norm Transformer block (`RMSNorm` + causal MHA + `SwiGLU`) comparing FP64 golden, Mercury `m.mma.fp8.blk`, and Hopper 14-bit truncated accumulators (`9` tests in `tests/test_microarch_and_torch.py`).
- **Publication Benchmark Visuals & Hero Render (`assets/`):**
  - `scripts/generate_charts.py`: Reproducible 2× supersampled Pillow generator outputting 5 live data-backed PNG charts in `assets/charts/` plus `assets/mercury-1-hero.jpg`.

## [0.2.0] - 2026-11-25

### Added
- **Virtual Silicon, Electro-Thermal & Cluster Qualification Suite (`mercury_sim/`, `scripts/run_qualification.py`):**
  - `specs.py`: Locked `Mercury-1S` (PCIe 340W) and `Mercury-1D` (OCP OAM 2.0 680W) single-mask TSMC N4P hardware contracts.
  - `thermal_power.py`: 2-node thermal RC ODE simulator (GPU die $T_j$ + backside clamshell GDDR7 $T_{\text{mem}}$), 85 °C `2× tREFI`, closed-loop DVFS/PROCHOT fault protection, and ATE `84→76` SCC wafer binning (`7` tests in `tests/test_thermal_power.py`).
  - `compute_isa.py`: OCP OFP8 E4M3/E5M2 bit-faithful codec, `m.mma.fp8.blk` vs. Hopper 14-bit error stress cases, 2:4 structured sparsity parity, and L1/RF/Shared-SRAM tile scheduling (`17` tests in `tests/test_compute_isa.py`).
  - `cluster_bench.py`: 10-scenario server/cluster suite covering `TP=2+FSDP` vs. `TP=8`, MoE `EP=8`, batch=1 decode bottleneck honesty, and 64-GPU 1T-token TCO (`5` tests in `tests/test_cluster_bench.py`).
  - `docs/verification/qualification-report.md`: Automated qualification report.

## [0.1.0] - 2026-11-25

### Added
- **Phase 1 Research & Architecture Specifications (`docs/`):**
  - `docs/research/riscv-open-gpu-landscape.md`: Evidence-backed claims audit across `OpenRVGPUCourse`, Vortex 3.0, Ventus, Tenstorrent Blackhole, and NVIDIA H100/B200 BOM economics.
  - `docs/architecture/mercury-1-spec.md`: Complete single-mask 453.8 mm² TSMC N4P floorplan, organic UCIe-S + GDDR7 memory hierarchy, Roofline, and BOM model.
  - `docs/architecture/isa-and-software-stack.md`: `RV64IMAFDC_Zve64f_Zvfbfmin` + `Xmer*` ISA encoding, `STME` Block-FP8 accumulator dataflow, and PyTorch/Triton/MLIR/RoCEv2 software stack.
