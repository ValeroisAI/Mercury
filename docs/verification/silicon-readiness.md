# Mercury-1 Silicon Readiness & Pre-Silicon Evidence Gates

> **Pre-silicon evidence.** Mercury-1 currently has runnable architectural models and verified standalone RTL blocks. The gates below track the path to a single-cluster FPGA prototype, integrated N4P implementation, and Extended OAM-L board validation.

---

## 1. Current Verified Baseline vs. Target Destinations

| Subsystem | Verified in Repository Today | Target Destination (FPGA / ASIC / Board) | Primary Blocker Today |
| :--- | :--- | :--- | :--- |
| **Compute Datapath (`STME` + `RVV`)** | [FP8 products and ordered group sum](./stme-group-rtl.md), [four-stage FP32 adder and tagged interleaving](./stme-interleaved-rtl.md), [scalar and six-stage FMA](./stme-boundary-rtl.md), [two scalar dot interfaces](./stme-fp8-dot-rtl.md), [Python ISA VM](../../mercury_isa/vm.py) | `84` fabricated / `76` active SCCs/die @ `1.85 GHz` (`540.5 ps` period), `229.4 mm²` logic budget (`8×16×16` STME + `128` RVV lanes) | Integrate the independent arithmetic and scheduling prototypes into a tiled STME with RVV/SIMT control, then verify timing. |
| **Shared Tensor SRAM & DMA Control** | [Parallel arbiter and equivalence evidence](./shared-sram-rtl.md), [`512`-bit banked payload prototype](./sram-payload-rtl.md), [DMA token checks](./dma-token-rtl.md), Python crossbar (`10,949` vs `312,517` cyc) | `192 MB/die` (`64 × 3 MB`, `58.2 mm²`) @ `1.85 GHz`, multi-stream L1/Shared-SRAM DMA and write-visibility barriers | Add SRAM macros, ECC, multi-tag pipelining, DMA write-commit wiring, and clock-domain crossings. |
| **FP8 Training Numerics** | `compare_gemm_paths` (`K=1024/4096`: `18.0×` / `42.0×` lower L2 err vs Hopper-14b); `run_training_validation()` `12`-run multi-seed sweep (`12/12` lower step-0 accum sensitivity, `12/12` quantization-dominated E2E, `7/12` E2E win vs Hopper-14b) | Convergent multi-node FP8 Transformer training at `100B+` parameters | Only 7 Linear layers emulated in FP8 (`SDPA/RMSNorm/SiLU/AdamW` run in FP64); fast FP32 BLAS order during training (`exact_order=False`); no delayed/online `amax` calibration or full LM loss-curve proof |
| **GDDR7 PAM3 & `Extended OAM-L` PCB** | The [courtyard model](../../mercury_sim/physical_bom.py) rejects 48 packages in `102×165 mm` OAM v1.5 (`81.5% / 95.3%`) and `102×170 mm` OAM r2.0 (`79.1% / 92.5%`) under its routing allowance. `140×185 mm` Extended OAM-L passes the screen (`53.0% / 61.9%`, `32.0 mm` maximum estimated trace). | Dual `453.8 mm²` N4P dies, `1.2 TB/s` UCIe-S, `48` clamshell `32 Gbps PAM3` GDDR7 packages and an estimated `872 A` rail current (`680 W / 0.78 V`). | Verify eye margins, PDN impedance, interposer and package integrity, and routing on a physical layout. |
| **Power, Thermal & N4P Yield** | 2-node lumped RC ODE (`thermal_power.py`, `dt=0.5 s`): liquid cold plate holds `54.6°C` $T_j$ / `68.0°C` $T_{\text{mem}}$ (`639W/680W`), air `35°C` trips `2× tREFI` (`>85°C`); Murphy + 30-wafer MC yield (`70.9%` / `80.9%` w/ `84→76` salvage, `$2,456` HW BOM + `$85M` NRE curve) | Signoff-verified `680W` TDP (`35°C` liquid), N4P silicon yield `>=70%` | Lumped thermal model vs `exp(0.022*(Tj-85))` proxy (no PDK FF/SS/TT corner leakage or 2D hotspot CFD); foundry wafer/BOM/OSAT numbers are model inputs pending vendor quotes |
| **Scale-Out `400GbE` / RoCEv2** | Analytical ring/leaf FSDP2 sensitivity (`fabric.py`): `64`-card `2:1` leaf-major `0.58%` exposed vs `2.65%` striped at fixed `55.0 s` compute; deferred grad sync `0.31%` adds `+137.8 GB` retained BF16 grads (no VRAM fit implied) | Production `4×400GbE` RoCEv2 training cluster (`ECMP`, `PFC`, `DCQCN`) | Zero packet queues, zero ECMP hash collision or PFC pause-storm simulation; `2 us`/`5 us` RTT assumes cut-through Direct RDMA (not PCIe host staging); degraded-port topology assumes symmetric traffic |

---

## 2. Gate P1: Pipeline Latency, Timing Closure, Area & FPGA Partitioning

Current standalone RTL blocks prove logical structure in Yosys `0.71` (`0` check errors), **not physical timing or chip-level area**:

1. **STME Datapath Partitioning (`mercury_fp8_product.sv`, `mercury_fp32_add_*.sv`, `mercury_stme_*.sv`):**
   - **Verified Distinction Today:**
     - [`rtl/mercury_fp8_product.sv`](../../rtl/mercury_fp8_product.sv) matches `262,144` exhaustive Python `(a, b, a_e5m2, b_e5m2)` cases ([`stme-group-rtl.md`](./stme-group-rtl.md); `102` LUTs out-of-context in [`fpga-mapping.md`](./fpga-mapping.md)).
     - [`rtl/mercury_fp32_add_rne.sv`](../../rtl/mercury_fp32_add_rne.sv) (`572` LUTs) and four-stage [`rtl/mercury_fp32_add_pipe.sv`](../../rtl/mercury_fp32_add_pipe.sv) (`569` LUTs / `186` FFs) pass algebraic SAT + exact-rational vector checks ([`stme-interleaved-rtl.md`](./stme-interleaved-rtl.md)).
     - The [ordered group sum](./stme-group-rtl.md) checks the full 128-term boundary. The [interleaved controller](./stme-interleaved-rtl.md) multiplexes eight independent contexts. The [scalar dot](./stme-fp8-dot-rtl.md) checks the two scale interfaces through stalls and verifies reset recovery for the pre-rounded scale interface. The [six-stage FMA](./stme-boundary-rtl.md) has separate numerical and latency checks.
   - **Exit Gate G1a (Pipelined datapath integration):** Connect the interleaved group adder, scale generation, and six-stage FMA to a tagged tiled pipeline. Verify per-context order through stalls and resets against `compute_isa.py`, then close **`540.5 ps` (`1.85 GHz`)** N4P worst-corner STA.
   - **Exit Gate G1b (`256×128` Multicast Area Closure):** Architectural modeling (`microarch_cycle.py`) lifts FP8 throughput from `1,723.5` to `2,022.0 TFLOPS` (`+17.3%`) via `256×128 + A-multicast`, but doubles active RF accumulators (`64→128 KiB`) and bumps L1 (`64→96 KiB`) plus adds an `8×19` multicast tree (`+10` cycles). Gate requires **post-P&R SCC macro floorplan <= `2.73 mm²/SCC`** (`84 × 2.73 = 229.4 mm²` total) before committing `multicast_256x128` as default over `128×128`.
2. **Shared SRAM & DMA Integration (`mercury_sram_arbiter.sv`, `mercury_dma_tokens.sv`):**
   - **Verified Today & Handshake Semantics:** Parallel selector refactoring reduced `8×64` default arbitration from `214,673 → 25,712` generic cells and `1,561 → 26` topological levels (`7,567` UltraScale+ LUTs / `192` FFs in [`fpga-mapping.md`](./fpga-mapping.md)); `sram_arbiter_tb.sv` proves `4×4` circular priority (`7` steps) and `22`-bit bijection; `dma_tokens_tb.sv` proves single-stream `4`-slot/`3`-bit generation lifecycle (`8` steps, `63` LUTs / `20` FFs). Note the Mealy interface contract on `mercury_sram_arbiter.sv`: `bank_ready[b]` gates `req_grant[i]` and `bank_issue[b]` combinationally, so requesters must register `req_valid`/`req_line` without combinational dependence on `req_grant` and banks must not gate `bank_ready` on `bank_issue`; `REQUESTERS` and `BANKS` also lack an elaboration-time power-of-2 `$error` guard (`next_owner + 1'b1` assumes power-of-2 wrap).
   - **Payload prototype:** The [banked payload interface](./sram-payload-rtl.md) with a word-wide return tree passes structural checks at `8×64×512`. A requester has one read slot and can accept its next command after the response is consumed. The macro adaptor must return each read at least one cycle after acceptance and discard or drain old work on reset.
   - **Exit Gate G1c (SRAM payload and write visibility):** Register SRAM commands and widen read tracking to multiple tagged requests. Attach SRAM macros and ECC, then check that final write visibility precedes DMA completion under bank stalls, resets, address hazards, and clock-domain crossings.
3. **FPGA Emulation Feasibility Gate (`fpga-mapping.md` vs. Target FPGA Silicon):**
   - **Target device resources:** AMD lists `4,085,760` CLB LUTs for the VU19P and `8,460,288` LUTs for the VP1902. Raw BRAM and URAM total about `174` and `900` decimal Mbit respectively, calculated from the documented block counts and sizes. [AMD DS890, table 15](https://docs.amd.com/api/khub/documents/dGU6Y~1b8XPqDFk5ulti6g/content) · [AMD DS950, Versal Premium summary](https://docs.amd.com/api/khub/documents/BY0FMkLH05y85Nwu4GmWsw/content)
   - **Full-die memory budget:** The [out-of-context mapping](./fpga-mapping.md) counts primitives for individual RTL blocks. Even the active-only per-die capacity of `192 MB` Shared Tensor SRAM plus `38 MiB` L1/RF requires about `1,855` decimal Mbit. That is about `2.06×` the raw VP1902 BRAM and URAM capacity, before allocation overhead and routing. A reduced FPGA configuration must scale the memory and compute dimensions.
   - **Exit Gate G1d (Implementable FPGA Target):** Map a **scaled single-cluster (`1–4` SCCs + `1–4 MB` BRAM/URAM + PCIe DMA + 1 GDDR/DDR channel)** bitstream onto VU19P/VP1902 at `100–250 MHz` (matching Georgia Tech `Vortex` / Tsinghua `Ventus` FPGA precedents in [the landscape audit](../research/riscv-open-gpu-landscape.md)), running compiled `Xmer*` ELF binaries end-to-end before multi-FPGA or ASIC tape-out freeze.

---

## 3. Gate P2: FP8 Training Numerics & Compiler Validation

*Primary references:* OCP OFP8 Revision 1.0 (`E4M3FN`: bias `7`, max `448`, no Inf; `E5M2`: bias `15`, max `57,344`, Inf/NaN), IEEE 754 `binary32`, and DeepSeek-V3 Technical Report §3.5.2 ([arXiv:2412.19437](https://arxiv.org/html/2412.19437#S3.SS5.SSS2)).

- **What Present Evidence Proves (Measured CPU Emulation):**
  - On bit-paired single GEMMs (`compare_gemm_paths`), `m.mma.fp8.blk` cuts `K=4096` outlier relative L2 error **42.0×** (`2.46e-5` vs `1.03e-3`).
  - Across `12` multi-seed Transformer training runs ([`docs/verification/fp8-training-validation.md`](./fp8-training-validation.md)), step-0 gradient accumulator-path sensitivity is lower for Mercury than Hopper-14bit in **12/12 runs**, yet **FP8 input/weight quantization causes `10.7%–17.5%` weight update drift vs FP64 and dominates end-to-end convergence in 12/12 runs** (`Mercury` beats `Hopper-14b` end-to-end in only `7/12` runs).
- **Open Exit Gates:**
  1. **Gate G2a (Bit-Exact RTL Parity):** Match full `16×16` pipelined STME RTL (merging scalar [`mercury_stme_scaled_fp8_dot.sv`](../../rtl/mercury_stme_scaled_fp8_dot.sv) two-scale rounding + pipelined `mercury_fp32_fma_rne` + interleaved adder) bit-for-bit against `mma_fp8_blk_mercury` across `K=128..4096` outlier vectors and strict IEEE-754 special values (`-0`, subnormal, overflow, NaN).
  2. **Gate G2b (Quantization Policy & Optimizer Scope):** Validate online/delayed `amax` block scaling and sensitivity when moving `RMSNorm`, softmax (`QK^T/PV`), `SiLU`, and `AdamW` states out of FP64 (`BF16`/`FP32` mix) on **multi-layer LM pretraining (`>= 100M`–`1B` params, `>= 5k` steps)**—closing the `>100×` dominant quantization error source identified in `run_training_validation()`.
  3. **Gate G2c (Compiler Lowering):** Freeze binary subopcodes/descriptor ABI ([`isa-and-software-stack.md`](../architecture/isa-and-software-stack.md)) and build a Triton/MLIR -> LLVM `Xmer*` backend emitting `m.mma.fp8.blk` + `m.dma.*` ping-pong schedules without dropping MFU below the modeled `~43%–54%` window.

---

## 4. Gate P3: GDDR7 `32 Gbps PAM3` SI/PI, UCIe-S & `Extended OAM-L` Board

*Primary references:* JEDEC **JESD239** GDDR7 (`PAM3` `+1/0/-1` 3-bits/2-UI signaling, `2× tREFI` refresh halving at `>=85°C`); Micron GDDR7 brief (`12.0 × 14.0 × 1.1 mm` 266-ball FBGA, [PDF](https://assets.micron.com/adobe/assets/urn%3Aaaid%3Aaem%3A087330f6-6d71-4575-b622-cb10f20cdaf0/original/as/gddr7-product-brief.pdf)); OCP OAM v1.5 (`102×165 mm`) & r2.0 (`102×170 mm`, [PDF](https://www.opencompute.org/documents/oai-oam-base-specification-r2-0-v1-0-20230919-pdf)) using Molex Mirror Mezz 48V connectors; **UCIe 1.0 Standard Organic Package** (`100–130 µm` bump pitch, `<=25 mm` reach, `28–224 GB/s/mm` shoreline bandwidth, `~0.5 pJ/bit`).

- **What Present Modeling Proves vs. Assumes:**
  - `physical_bom.py` proves geometrically why `48` clamshell GDDR7 chips + dual `62×42 mm` organic packages + `872 A` VRM banks + Mirror Mezzanine keep-outs overflow `102×165 mm` (`95.3%` bottom fill) and `102×170 mm` (`92.5%` bottom fill), motivating **`Extended OAM-L (140×185 mm)`** (`53.0% / 61.9%` fill, `32.0 mm` Manhattan escape estimate) or a `96 GB / 32-chip` `102×165 mm` fallback (`69.4% / 68.2%`).
  - `UCIe-S` assumes `100 GB/s/mm` bidirectional across `12 mm` (`1.2 TB/s`), which sits inside UCIe 1.0 Standard Package limits (`28–224 GB/s/mm`), **pending package substrate validation**.
- **Open Exit Gates:**
  1. **Gate G3a (`32 Gbps PAM3` SI Signoff):** Extract 3D S-parameters across the target PCB stackup for worst-case `32 mm` front/back **clamshell blind/buried/through vias**, verify PAM3 outer/inner eye height/width after TX FFE + RX CTLE/DFE at JESD239 BER limits (`<= 1e-12`–`1e-15` pre-FEC/retry).
  2. **Gate G3b (`872 A` PDN / PI Signoff):** Simulate DC IR drop (`<= 2%`–`3%` of `0.78 V` nominal `Vcore`) and AC target impedance (`Z_target < 0.45 mOhm`) across full `48V -> 0.78V` VRM + backside decoupling placement without colliding with `24` backside GDDR7 packages or Mirror Mezzanine zones.
  3. **Gate G3c (Mechanical Chassis & UCIe-S Substrate):** Lock custom `140×185 mm` **Extended OAM-L** carrier pitch/UBB mechanical envelope (note: incompatible with `102 mm` standard OCP trays unless using the `96 GB / 32-chip` SKU) and pass warpage/electromigration simulation for dual `453.8 mm²` organic laminate dies.

---

## 5. Gate P4: Power, Thermal Torture & N4P Yield / BOM Economics

- **Modeled Today vs. Real Targets:** `thermal_power.py` (`dt=0.5 s` lumped RC ODE) demonstrates cold-plate stability (`54.6°C` $T_j$, `68.0°C` $T_{\text{mem}}$ @ `35°C` liquid), air-cooled `2× tREFI` throttling (`85°C`), and `PROCHOT` clamp recovery (`96.1°C` peak -> `117W`). `physical_bom.py` bounds unit cost from `$19,920` @ `10k` (`$2,456` BOM + `$85M` NRE @ 45% GM) down to `$6,011` @ `100k`.
- **Open Exit Gates:**
  1. **Gate G4a (PDK Leakage & Vector Power):** Replace `~0.31 W/BF16-TFLOP` and `exp(0.022*(Tj-85))` with foundry N4P synthesis + PrimeTime-PX activity-factor (`VCD`/`SAIF`) dynamic/leakage curves across TT/FF/SS corners.
  2. **Gate G4b (Conjugate 3D Thermal CFD):** Model localized SCC STME heat flux vs Shared SRAM density and cold-plate-to-backside-GDDR7 board heat conduction (`Tmem < 85°C` continuous margin, cold-plate flow/Delta-P).
  3. **Gate G4c (Foundry Yield & Supply Quotes):** Calibrate `D0` and critical-area sensitivity for `192 MB` HD/HP 6T SRAM repairs (separate from `84→76` SCC logic salvage) and obtain actual N4P wafer/OSAT/GDDR7 volume quotes.

---

## 6. Gate P5: `400GbE` / RoCEv2 Congestion, Failover & DMA Staging

*Primary references:* **IBTA Architecture Release 1.2.1 Annex A17** (RoCEv2 UDP port `4791`, L3 ECMP), **IEEE 802.1Qbb** (Priority-based Flow Control), and **RFC 3168 ECN / DCQCN**.

- **What Present Modeling Proves vs. Assumes:** `mercury_sim/fabric.py` (`docs/verification/fabric-sensitivity.md`) holds compute fixed at `55.0 s`, uses `90%` payload efficiency and `2 us` local / `5 us` inter-leaf round latencies, showing `0.58%` exposed (`leaf_major`) vs `2.65%` (`striped`) on a `64`-card `2:1` leaf/spine fabric, and `0.31%` when deferring gradient sync (`+137.8 GB` BF16 gradient increment exceeds available activation budget if kept naive).
- **Open Exit Gates:**
  1. **Gate G5a (Packet-Level Congestion & Link Asymmetry):** Validate `ECMP` hash polarization, `IEEE 802.1Qbb` PFC headroom, and `DCQCN` ECN marking thresholds on a packet simulator (ns-3 / SST) for simultaneous dual-ring FSDP + `4×400GbE` bursts, especially under `1`-port loss (`64_cards_striped_one_port_lost`, where fluid spine-cap math currently masks NIC asymmetry).
  2. **Gate G5b (Direct-RDMA vs. Host Bounce):** Validate GPU-memory peer-to-peer DMA visibility (`isa-and-software-stack.md:224`) so collectives hit `2–5 us` RDMA round latency instead of paying PCIe Gen5 x16 (`~55.5 GB/s` modeled) host-staging hops.

---

## 7. Phased Go / No-Go Signoff Ladder

| Phase Gate | Target Platform | Mandatory Pass Evidence |
| :--- | :--- | :--- |
| **Gate A — Module Formal & Bit-Exact** | Local RTL + Yosys/Verilator | Pipelined `mercury_stme_scaled_fp8_dot` + `16×16` array match `compute_isa.py` bit-exact; `8×64` arbiter + payload mux/return crossbar + multi-stream DMA formal pass |
| **Gate B — Single-Cluster FPGA** | AMD VU19P / VP1902 | `1–4` SCCs + `Xmer*` control + DMA + PCIe host driver boot running compiled `.masm`/ELF attention + GEMM kernels end-to-end on real hardware |
| **Gate C — Physical & System Pre-Tapeout** | TSMC N4P + `Extended OAM-L` Board Twin | `1.85 GHz` SS-corner STA + IR-drop signoff (`<=2.73 mm²/SCC`); `32 Gbps PAM3` IBIS eye closure (`<=32 mm`); `872 A` PDN Z-target; packet-level RoCEv2 ECN/PFC goodput signoff; multi-layer LM FP8 convergence proof |
| **Gate D — Silicon Bring-Up** | N4P Shuttle / `Extended OAM-L` & `Mercury-1S` | Wafer ATE `84→76` SCC salvage & SRAM BIST; `680W` liquid cold-plate soak (`Tmem < 85°C`); `8×` / `64×` RoCEv2 cluster MFU & 70B training run vs model |
