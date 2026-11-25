# Mercury-1 ISA and AI training software stack

Mercury-1 targets FP8 and BF16 transformer training with a RISC-V vector
execution model, asynchronous tensor instructions, and explicit on-chip
tiling. The hardware figures below are locked design inputs from the Mercury-1
architecture brief. Instruction semantics, dataflow, and software interfaces
in the following sections are proposed. The BOM, TDP, clock, and bandwidth
figures are targets or estimates pending implementation and measurement.

## Hardware contract

Both products use one TSMC N4P mask set for the **Mercury-1 Compute Die**
(~453.8 mm²). Each die has 84 fabricated SCCs and enables 76 for defect
tolerance. The sustained AI training clock target is **1.85 GHz**.

| | Mercury-1S | Mercury-1D (recommended for AI training) |
| --- | --- | --- |
| Module | PCIe Gen5 ×16 card | Extended OAM-L module (`140×185 mm`) |
| Dies and active SCCs | 1 die, 76 SCCs | 2 dies, 152 SCCs |
| Die-to-die interconnect | Single die | 1.2 TB/s bidirectional organic UCIe-S bridge |
| GDDR7 ECC | 72 GB (67.1 GiB), 384-bit at 32 Gbps, 1.54 TB/s | 144 GB (134.1 GiB), 2 × 384-bit (768-bit) at 32 Gbps, 3.07 TB/s |
| Shared Tensor SRAM | 192 MB, 64 × 3 MB banks, 7.58 TB/s | 384 MB, 64 × 3 MB banks per die, 7.58 TB/s per die and 15.16 TB/s total |
| Network | Host attachment via PCIe | 4 × 400GbE / RoCEv2, 1.6 Tbps (200 GB/s in each direction at full-duplex line rate) |
| TDP and BOM | ~340 W, ~$1,238 | ~680 W, ~$2,456 |

Each SCC has two RV64 warp/control schedulers, 32 threads per warp and
16 resident warps (512 threads per SCC), 128 FP32/INT32 `Zve64f` RVV lanes,
a dedicated **256 KiB Register File** (scalar, SIMT vector, and STME
accumulator fragments) plus a separate **256 KiB L1 Tensor Scratchpad**
(512 KiB total local SRAM per SCC; 38.0 MiB active per die / 76.0 MiB on 1D),
an asynchronous tensor DMA engine, and a collective reduction ALU. Eight
16 × 16 Systolic Tensor Matrix Engines (STMEs) supply 2,048 BF16, 4,096 FP8,
8,192 MXFP4, or 1,024 TF32 MACs per SCC per cycle, with hardware 2:4
structured sparsity.

| Format | Mercury-1D dense peak (TFLOPS) | Mercury-1D 2:4 effective peak (TFLOPS) |
| --- | ---: | ---: |
| BF16 | 1,151.8 | 2,303.6 |
| FP8 | 2,303.6 | 4,607.2 |
| MXFP4 | 4,607.2 | 9,214.4 |
| TF32 | 575.9 | |
| FP32 vector | 72.0 | |

Mercury-1S peaks are exactly half before rounding. Count two FLOPs per
MAC or vector FMA, so the BF16 peak is
`152 × 1.85 × 2,048 × 2 / 1,000 = 1,151.7952 TFLOPS` and the FP32 vector
peak is `152 × 1.85 × 128 × 2 / 1,000 = 71.9872 TFLOPS`.
The sparse column counts dense-equivalent work for a supported operand
with valid 2:4 compression. TF32 and FP32 vector figures are dense targets.

## Mercury Open RISC-V ISA

The design profile is **RV64IMAFDC_Zve64f_Zvfbfmin + Zmer_simt + Zmer_stme +
Zmer_dma + Zmer_coll**. Using `Zve64f` + `Zvfbfmin` instead of full `V`
intentionally provides 64-bit integer, 32-bit IEEE float, and BF16 vector
storage/conversion without `Zve64d` FP64 vector pipeline bloat, while
scalar `D` remains available in the control core. The four `Zmer_*` labels
name the design groups; use the non-standard names `Xmersimt`, `Xmerstme`,
`Xmerdma`, and `Xmercoll` for LLVM and ELF. The proposed lowercase toolchain
string is
`rv64imafdc_zve64f_zvfbfmin_xmercoll_xmerdma_xmersimt_xmerstme`, with
versions to be assigned before the ABI freeze. RISC-V specifies,
“Non-standard extensions are named by using a single "X" followed by the
alphanumeric name.” [RISC-V extension naming](https://docs.riscv.org/reference/isa/unpriv/naming.html)

### Proposed 32-bit major opcode map

| Major opcode | `inst[6:0]` | Extension | Operations |
| --- | --- | --- | --- |
| custom-0 | `0001011` / `0x0b` | `Zmer_simt` | `m.split`, `m.join`, `m.tmc`, `m.wspawn`, `m.bar.*` |
| custom-1 | `0101011` / `0x2b` | `Zmer_stme` | `m.mma.bf16`, `m.mma.fp8.blk`, `m.mma.wait`, reserved matrix subfunctions |
| custom-2 | `1011011` / `0x5b` | `Zmer_dma` | `m.dma.copy`, `m.dma.wait` |
| custom-3 | `1111011` / `0x7b` | `Zmer_coll` | `m.allreduce.sum`, `m.allreduce.max` |

The major bits come from the RISC-V base opcode map, which describes
custom-0 through custom-3 as “recommended for use by custom instruction-set
extensions within the base 32-bit instruction format.” Mercury assigns
the groups above. Use register operands for descriptor or fragment handles
and function fields for operation selection. Subopcodes, illegal encodings,
and the binary descriptor layout remain open. [RISC-V opcode map](https://docs.riscv.org/reference/isa/v20260120/unpriv/rv-32-64g.html)

### Warp control and RVV

- `m.split pred, else_pc, reconv_pc` partitions the current active mask,
  saves the entry mask and reconvergence PC, and starts the true path
  at fallthrough when it has active lanes. Otherwise it jumps to
  `else_pc`. With both paths populated, it records the pending false
  mask and `else_pc` in the reconvergence frame.
- `m.join` at `reconv_pc` redirects to the pending false path if one
  exists. On final arrival it restores the entry mask and pops the frame.
  The compiler emits structured nesting and forbids convergent barriers
  and MMA issue inside a divergent branch.
- `m.tmc mask` sets the 32-bit warp participation mask within the
  runtime-assigned live lanes and the current control-flow mask. A zero
  mask at the outermost scope retires the warp.
- `m.wspawn count, entry_pc` launches child warps in slots reserved
  for the workgroup, with runtime-initialized arguments and lane IDs.
  The workgroup admission calculation includes those slots, avoiding
  a parent waiting for slots occupied by other admitted workgroups.

Vortex's ISA manual lists “wspawn, tmc, split, join, bar, and tex,” and
Vortex 3.0 adds “warpgroup-level matrix multiplication (WGMMA).” Ventus
documents its divergence sequence as `setrpc`, `vbranch`, and `join`.
Mercury uses an explicit split/join pair for warp control and RVV for
lane arithmetic. In the proposed SIMT lowering, a vector operation's
participants are the intersection of the active warp mask, RVV's
`vl` and operand mask, with inactive hardware warp lanes preserved.
The standalone RVV path retains standard tail, mask, and restart
behavior under `Zve64f` (`ELEN=64` integer, `SEW<=32` FP) and `Zvfbfmin`
(BF16↔FP32 vector conversion), intentionally omitting `Zve64d` vector FP64
hardware to maximize BF16/FP8 systolic density per mm². Scalar `D` executes
on the SCC control scalar unit for host/runtime compatibility.
[Vortex ISA manual](https://github.com/vortexgpgpu/vortex-riscv-isa-manual/blob/main/main.tex),
[Vortex 3.0 changelog](https://github.com/vortexgpgpu/vortex/blob/master/CHANGELOG.md#30--2026-10-07),
[Ventus branching trace](https://github.com/THU-DSP-LAB/ventus-gpgpu#branching-output),
[RISC-V vector specification](https://docs.riscv.org/reference/isa/v20260120/unpriv/v-st-ext.html)

### Matrix, DMA, and collective semantics

**`m.mma.bf16 acc, a, b`** submits a converged warp-group matrix step.
The tile handles identify BF16 fragments and a read/write FP32 accumulator.
Accumulate `A × B` into `acc` with FP32 fused multiply-add. Tile layouts
map onto the 16 × 16 engines. The ABI will specify the warp-group size and
supported instruction tile shapes. Operations on one accumulator are
ordered, and `m.mma.wait` exposes the result to vector code and releases
the input fragments for reuse.

**`m.mma.fp8.blk acc, a, b`** uses E4M3 or E5M2 payloads and scale
descriptors. For `A[M,K] × B[K,N]`, A's 1 × 128 scale varies by output
row and K group. B's 128 × 128 scale varies by K group and group of 128
output columns. The descriptor indexes logical coordinates across
physical swizzles and transposition. The proposed FP32 scale-factor
reference for a finite-input K group `g` is:

```text
# Per K_group = 128 (8 systolic K=16 passes):
group_sum = 0.0_f32
for k in group g, in increasing logical K order:
    p = decode_fp8(A[i,k]) * decode_fp8(B[k,j])  # exact in FP32 for finite FP8
    group_sum = add_F32_RNE(group_sum, p)          # unscaled exact FP32 sum within group g
s = RN32(A.scale[i,g] * B.scale[g,j//128])         # or 1-cycle exponent add for UE8M0
acc[i,j] = fma_F32_RNE(group_sum, s, acc[i,j])     # 1 FP32 FMA per K=128 boundary
```

Unscaled partial sums accumulate inside each `K=128` group in IEEE-754
binary32 (round-to-nearest-even) and apply `s` once per `K=128` group
boundary (or via a single-cycle hardware exponent bias shift when `UE8M0`
power-of-two scales are selected). Factoring `s` to the `K=128` boundary
avoids placing a full FP32 scale multiplier inside every systolic PE while
maintaining full IEEE-754 binary32 precision across K groups. Binary scale
encoding, nonfinite inputs, exception flag reporting, and exact bitwise
equivalence against the scalar reference require RTL verification. This
design aims to remove compiler-emitted per-group descale/add loops and the
FP32 vector promotion used to work around Hopper's narrower accumulation.
DeepSeek reports that Hopper “only uses the highest 14 bits of each
mantissa product for addition.”
[DeepSeek-V3 Technical Report, §§3.3.2 and 3.5.2](https://arxiv.org/html/2412.19437#S3.SS5.SSS2)

**`m.dma.copy`** queues an asynchronous multidimensional copy and returns
a token. The descriptor specifies source and destination address
spaces, strides, bounds, and zero-fill for eligible tile tails. Payload
and scale planes share a buffer lifetime, and the consumer waits on
both transfers. **`m.dma.wait`** acquires completion and visibility in
the destination scope. A workgroup barrier publishes readiness to other
warps and returns a slot to the producer only after its consumers and
any MMA reads finish. DMA and MMA use separate completion tokens.

**`m.allreduce.*`** invokes the SCC reduction ALU on a converged
participant group for sum or max and returns the local group result.
MercuryCCL composes these primitives with transfers and fabric
completion events for cross-SCC, cross-die, and cross-node reductions.

## SCC issue and tensor dataflow

Each scheduler may issue one ready warp instruction per cycle when
operands, issue ports, and downstream credits are available. A
scoreboard tracks registers, accumulator groups, DMA tokens, and
barriers. A producer warp can issue tensor DMA while a compute warp
issues RVV work or dispatches an STME operation. STME execution and DMA
progress independently after issue. Two schedulers share execution
resources, so dual issue depends on resource availability.

```text
die-local GDDR7
    -> m.dma.copy -> die-local Shared Tensor SRAM [bank-striped stage 0 / 1]
    -> m.dma.copy -> SCC operand tiles [stage 0 / 1] -> STME / RVV
    -> local FP32 accumulators or online softmax state -> epilogue -> output
```

The tensor memory accelerator (TMA) schedule prefetches stage 1 while
stage 0 is consumed, then swaps their roles. Shared Tensor SRAM holds reusable operand tiles across SCCs on a die.
The compiler stages operand tiles and ping-pong buffers inside the
dedicated **256 KiB L1 Tensor Scratchpad** while keeping scalar, `Zve64f`
vector registers, and STME accumulator fragments inside the separate
**256 KiB Register File**, spilling larger reuse buffers to Shared Tensor
SRAM. Place workgroups and their buffers on the same die where possible.
The 15.16 TB/s figure sums both dies' local SRAM bandwidth targets.
Remote SRAM traffic consumes UCIe-S capacity and has its own latency.

For attention, keep the query tile, score tile, and online softmax state
on-chip while streaming key/value tiles. For an MLP, retain a tile of
the up/gate projection through its activation and elementwise product,
then consume it in the down projection when the schedule fits. These
choices reduce external reads and intermediate writes, addressing the
GDDR7 bandwidth trade-off relative to HBM3e. Using raw bandwidth, the
Mercury-1D dense FP8 DRAM roofline crosses over at about
`2,303.6 / 3.07 ≈ 750 FLOP/byte`. The FlashAttention-3 paper describes
“interleave block-wise matmul and softmax operations” as one of its
overlap techniques. That schedule motivates the Mercury port.
[FlashAttention-3](https://arxiv.org/abs/2407.08608),
[Mercury GPU landscape](../research/riscv-open-gpu-landscape.md)

## Open AI training software stack

| Layer | Mercury integration |
| --- | --- |
| PyTorch 2.x | `torch.compile` uses TorchDynamo for capture and AOTAutograd for training graphs. Add Mercury device dispatch and TorchInductor scheduling, including a BF16 device path for graphs or shapes outside the FP8 fast path. PyTorch documents that “AOT Autograd captures not only the user-level code, but also backpropagation.” [PyTorch compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html) |
| Triton / MLIR | Add a Mercury backend with layout selection, block-scale dot lowering, warp specialization, and explicit async dependences for tensor DMA and matrix issue. Triton documents how to dump IR “before every MLIR pass Triton runs.” [Triton compiler](https://github.com/triton-lang/triton#tips-for-hacking) |
| LLVM RVV + Mercury | Add Mercury LLVM intrinsics, target features, instruction selection, register/fragment allocation, assembler/disassembler support, and object attributes. Emit versioned RV64 device objects with feature, launch, and scratch metadata. LLVM describes vendor extensions as “defined by a hardware vendor.” [LLVM RISC-V target](https://llvm.org/docs/RISCVUsage.html#vendor-extensions) |
| Mercury Open Runtime and chipStar HIP | The native runtime owns device memory, code loading, queues, launches, events, and topology. Add a Mercury OpenCL/SPIR-V path that lowers chipStar kernels through the Mercury device compiler and shares the native runtime. chipStar runs HIP/CUDA “on SPIR-V via OpenCL or Level Zero APIs.” [chipStar](https://github.com/CHIP-SPV/chipStar) |
| MercuryCCL | Register a `torch.distributed` backend. Chunk gradient buckets, reduce inside the die, use UCIe-S across the 1D module, then use RoCEv2 reduce-scatter/all-gather over its four 400GbE ports. Signal stream completion only after data is visible to consumers. PyTorch supports “backends registered at run time” and RoCEv2 “encapsulates RDMA in UDP/IP packets.” [PyTorch distributed backends](https://docs.pytorch.org/docs/main/distributed/custom_backends.html), [RoCEv2 documentation](https://networking-docs.nvidia.com/doca/archive/3-6-0/doca-rdma-over-converged-ethernet) |

For RoCEv2, negotiate memory registration and device-buffer visibility
with the transport. Direct device RDMA is an integration target.
Start with registered host staging where direct RDMA is unavailable.
Measure application goodput separately from the 200 GB/s line-rate
target. ChipStar compatibility and the native compiler are separate
entry points to the same proposed runtime.

### Fused block-FP8 query projection and FlashAttention training

This forward-pass sketch assigns one 64-query tile and one 128-wide
head to a workgroup. It fuses the query projection with a causal
FlashAttention-3-style loop. K and V have already been projected.
`K8[j]` is packed for the transposed RHS and carries scales in logical
`[head_dim, key_positions]` coordinates. V is BF16. K-tail elements
along the reduction dimension are zero-filled. Token padding and
causal masking happen before softmax. This example uses no dropout.

```text
# Conceptual Triton / Mercury IR. Async stage iterators queue tile i+1
# while i is in use and recycle slots only after the last read.
q32 = zeros_f32([64, 128])
for x8, wq8 in dma_pingpong(X8[query_rows, :], Wq8[:, head], step_K=128):
    await_copy(x8, wq8)                       # m.dma.wait, data + scales
    q32 = mma_fp8_blk(q32, x8, wq8)            # m.mma.fp8.blk
await_mma(q32)
q8 = quantize_fp8_1x128(q32)                  # RVV, retain data and scales

row_max = -inf
row_sum = 0.0
numerator = zeros_f32([64, 128])
for k8t, v in dma_pingpong(K8_transposed, Vbf16, step_keys=128):
    await_copy(k8t, v)
    score = mma_fp8_blk(zeros_f32([64, 128]), q8, k8t)
    await_mma(score)
    score = causal_and_padding_mask(score / sqrt(128), fill=-inf)
    next_max = max(row_max, rowwise_max(score))
    alpha = 0 if row_max == -inf else exp(row_max - next_max)
    p = masked_exp(score - next_max, valid_tokens, all_masked_value=0)
    row_sum = alpha * row_sum + rowwise_sum(p)
    numerator = mma_bf16(alpha * numerator, cast_bf16(p), v)
    await_mma(numerator)
    row_max = next_max

store O = safe_div(numerator, row_sum, zero_value=0)
store LSE = row_max + log(row_sum)            # -inf for fully masked rows
```

`masked_exp` evaluates the subtraction and exponential only on valid
positions, and `safe_div` skips division on empty rows. Prefetching is
the first overlap to implement. Interleaving the next QK matrix step
with current softmax or PV is a later schedule using independent
accumulator groups. A 64 × 128 score tile and output numerator cost 32 KiB of FP32 state each
in the dedicated Register File / STME accumulator budget. In the separate
**256 KiB L1 Tensor Scratchpad**, two K/V stages use 96 KiB of payload
(`2 × 128 × 128 × (1 + 2)` bytes), leaving ~160 KiB for Q8, block scales,
and layout padding (~84 KiB spare headroom after ~12 KiB Q/scale staging),
while spilling larger reuse buffers to Shared Tensor SRAM if tile sizes
increase.

For backward, save O and LSE and retain or deterministically recompute
the quantized Q/K operands and their scales. Reconstruct softmax tiles
from scores and LSE in FP32 and accumulate dQ/dK/dV in tiles. For dV,
replay the forward tile order and BF16 probability casts to recover
the effective PV weights, including the later online rescalings.
Use straight-through gradients for quantize/cast operations under an
explicit mixed-precision policy. Propagate dQ through the query
projection to dX and dWq, using a wider-precision path where required,
and reduce dK/dV contributions across query workgroups. Fuse only
validated patterns and compare forward and backward against a
reference using the same rounding and masking policy.
[FlashAttention-3](https://arxiv.org/abs/2407.08608),
[DeepSeek-V3 FP8 training](https://arxiv.org/html/2412.19437#S3.SS3.SSS2)

## Known, inferred, and unknown

| Status | Claim | Confidence | Evidence needed to revise it |
| --- | --- | --- | --- |
| Known | The Mercury-1 brief locks the SKUs, clock, SCC resources, and peak inputs above. The peak table follows from the stated MAC rates. | High for the inputs and arithmetic | A change to the architecture brief or peak-counting convention |
| Known | RISC-V defines four custom major opcodes. Vortex and Ventus provide distinct open SIMT precedents. [RISC-V opcode map](https://docs.riscv.org/reference/isa/v20260120/unpriv/rv-32-64g.html), [Vortex](https://github.com/vortexgpgpu/vortex), [Ventus](https://github.com/THU-DSP-LAB/ventus-gpgpu) | High | A standards revision or a change in the upstream projects |
| Inferred | The proposed warp/DMA/matrix contracts can express the tiled forward schedule with bank-local reuse. | Medium for expressibility, low for achieved overlap | Executable ISA model, compiler lowering, and a cycle trace with bank-conflict, occupancy, and spill counters |
| Inferred | Fusing FP8 scales in the MAC can remove separate descale/add loops from the generated hot path. | Medium | Disassembly, binary32 reference comparison, and post-layout timing for the actual scaled datapath |
| Unknown | Full RVV conformance, FP8 corner cases, and scale format, subopcodes, and fragment ABI. | Low | Architectural tests, bitwise RTL differential tests, and a frozen assembler/ABI specification |
| Unknown | Clock, area, TDP, BOM, memory efficiency, and MercuryCCL multi-node goodput under training load. | Low for achieved results | N4P physical design, board and memory validation, component quotes, and instrumented multi-node training |
