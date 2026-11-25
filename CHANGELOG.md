# Changelog

All notable changes to the **Mercury** open-source RISC-V AI Training GPU project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
