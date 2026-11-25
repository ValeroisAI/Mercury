"""PyTorch Transformer-block parity check for Mercury-1 FP8 block GEMMs.

A real ``torch.nn`` pre-norm Transformer block (RMSNorm, Q/K/V/O
projections with causal scaled dot-product attention, then RMSNorm and a
SwiGLU MLP with gate/up/down) runs forward and backward in ``torch.float64``.
That gives golden activations and autograd gradients. The training GEMMs
of the SwiGLU ``gate`` projection are then re-executed with identical
FP8 block-quantized bytes on two accumulator models:

* Mercury-1 ``m.mma.fp8.blk``: ``block_quantize_1x128`` activations,
  ``block_quantize_128x128`` weights, K-ordered FP32 partials and one
  RNE scaled FMA per K=128 group (``mma_fp8_blk_mercury``).
* Hopper-style 14-bit aligned truncation (``mma_fp8_hopper_truncated``).

The three GEMMs are forward ``Y = X Wg^T`` (K = hidden), input-gradient
``dX = dY Wg`` (K = ffn) and weight-gradient ``dW = dY^T X`` (K = tokens).
Each is measured two ways. *End-to-end* error compares against the FP64
GEMM on the unquantized tensors, so it includes FP8 quantization. *Accumulator-only*
error compares against FP64 on the dequantized FP8 operands, which
isolates how the accumulator behaves.

``compute_isa`` is pure Python, so each GEMM covers a column slice of
the real tensors. This keeps the whole benchmark well under a second.

``simulate_fp8_training_loop`` runs multi-step AdamW training. For that it
uses vectorized torch versions of the same two GEMMs
(``mercury_fp8_gemm`` / ``hopper_fp8_gemm``). They are bit-identical to
``compute_isa`` on the tested shapes; ``tests/test_microarch_and_torch.py``
checks this. Every ``nn.Linear`` in the block (Q/K/V/O, gate/up/down) runs
its forward ``Y``, ``dX`` and ``dW`` GEMMs through the chosen path. The
non-GEMM ops (RMSNorm, softmax, SiLU, and SDPA's internal QK^T and PV
products) and the AdamW update stay in FP64.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .compute_isa import (
    _DECODE,
    BLOCK,
    block_quantize_128x128,
    block_quantize_1x128,
    mma_fp8_blk_mercury,
    mma_fp8_hopper_truncated,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MercuryTransformerBlock(nn.Module):
    """Pre-norm decoder block: RMSNorm -> causal MHA -> RMSNorm -> SwiGLU MLP."""

    def __init__(self, hidden: int = 512, n_heads: int = 8, ffn: int = 1024) -> None:
        super().__init__()
        if hidden % n_heads:
            raise ValueError("hidden must be divisible by n_heads")
        self.hidden, self.n_heads, self.ffn = hidden, n_heads, ffn
        self.attn_norm = RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_norm = RMSNorm(hidden)
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)
        self.gemm_mode = "fp64"  # or one of FP8_GEMM_MODES

    def _proj(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self.gemm_mode == "fp64":
            return layer(x)
        y = FP8Linear.apply(x.reshape(-1, x.shape[-1]), layer.weight, self.gemm_mode)
        return y.view(*x.shape[:-1], layer.weight.shape[0])

    def forward(self, x: torch.Tensor, capture: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        b, s, d = x.shape
        h = self.attn_norm(x)
        heads = lambda t: t.view(b, s, self.n_heads, d // self.n_heads).transpose(1, 2)
        q, k, v = (heads(self._proj(p, h)) for p in (self.q_proj, self.k_proj, self.v_proj))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self._proj(self.o_proj, attn.transpose(1, 2).reshape(b, s, d))
        m = self.mlp_norm(x)
        gate = self._proj(self.gate_proj, m)
        if capture is not None:
            m.retain_grad()
            gate.retain_grad()
            capture["mlp_in"], capture["gate_out"] = m, gate
        return x + self._proj(self.down_proj, F.silu(gate) * self._proj(self.up_proj, m))


def correlated_outlier_activations(
    batch: int, seq: int, hidden: int, *, outlier_magnitude: float = 24.0,
    outlier_stride: int = 32, seed: int = 0,
) -> torch.Tensor:
    """LLM-like activations: fixed per-channel signs, lognormal magnitudes,
    and one massive-activation channel per ``outlier_stride`` channels."""
    gen = torch.Generator().manual_seed(seed)
    sign = torch.where(torch.rand(hidden, generator=gen) < 0.5, -1.0, 1.0).double()
    mag = torch.exp(torch.randn(batch, seq, hidden, generator=gen, dtype=torch.float64) * 0.7 - 0.5)
    x = sign * mag
    chans = [start + int(torch.randint(0, outlier_stride, (1,), generator=gen))
             for start in range(0, hidden, outlier_stride)]
    x[..., chans] = sign[chans] * outlier_magnitude
    return x


def _flat(t: torch.Tensor) -> list[float]:
    return t.detach().to(torch.float64).contiguous().view(-1).tolist()


def _dequant(q: bytes, scales: list[float], rows: int, cols: int, row_block: int) -> torch.Tensor:
    vals = torch.tensor([_DECODE["e4m3"][c] for c in q], dtype=torch.float64).view(rows, cols)
    kg = math.ceil(cols / BLOCK)
    sc = torch.tensor(scales, dtype=torch.float64).view(-1, kg)
    sc = sc.repeat_interleave(row_block, 0)[:rows].repeat_interleave(BLOCK, 1)[:, :cols]
    return vals * sc


def _metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (out - ref).flatten()
    ref_norm = float(ref.norm())
    return {
        "rel_l2": float(diff.norm()) / ref_norm if ref_norm else float(diff.norm()),
        "cos_sim": float(F.cosine_similarity(out.flatten(), ref.flatten(), dim=0)),
        "max_abs": float(diff.abs().max()),
    }


def compare_gemm_paths(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """C = A[M,K] @ B[K,N] on FP64 golden, Mercury m.mma.fp8.blk and Hopper-14bit."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dimensions differ")
    golden = a.double() @ b.double()
    aq, sa = block_quantize_1x128(_flat(a), m, k)
    bq, sb = block_quantize_128x128(_flat(b), k, n)
    deq_ref = _dequant(aq, sa, m, k, 1) @ _dequant(bq, sb, k, n, BLOCK)
    mercury = torch.tensor(mma_fp8_blk_mercury(aq, sa, bq, sb, m, n, k), dtype=torch.float64).view(m, n)
    hopper = torch.tensor(mma_fp8_hopper_truncated(aq, sa, bq, sb, m, n, k), dtype=torch.float64).view(m, n)
    return {
        "shape_mnk": [m, n, k],
        "mercury": _metrics(mercury, golden),
        "hopper_14bit": _metrics(hopper, golden),
        "mercury_accumulator_only": _metrics(mercury, deq_ref),
        "hopper_accumulator_only": _metrics(hopper, deq_ref),
    }


@lru_cache(maxsize=4)
def _bridge(hidden: int, ffn: int, n_heads: int, batch: int, seq: int, n_slice: int,
            outlier_magnitude: float, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    block = MercuryTransformerBlock(hidden, n_heads, ffn).double()
    x = correlated_outlier_activations(batch, seq, hidden, outlier_magnitude=outlier_magnitude, seed=seed)
    x.requires_grad_(True)
    cap: dict[str, torch.Tensor] = {}
    out = block(x, capture=cap)
    target = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64)
    loss = F.mse_loss(out, target)
    loss.backward()

    tokens = batch * seq
    m_in = cap["mlp_in"].detach().reshape(tokens, hidden)
    dy = cap["gate_out"].grad.detach().reshape(tokens, ffn)
    w = block.gate_proj.weight.detach()            # [ffn, hidden]
    dw_autograd = block.gate_proj.weight.grad.detach()

    # Autograd alignment: the manual GEMMs reproduce what torch computed.
    y_manual = m_in @ w.T
    dw_manual = dy.T @ m_in
    alignment = {
        "forward_max_abs": float((y_manual - cap["gate_out"].detach().reshape(tokens, ffn)).abs().max()),
        "dW_max_abs": float((dw_manual - dw_autograd).abs().max()),
    }

    cols = slice(0, n_slice)
    gemms = {
        "forward_Y": compare_gemm_paths(m_in, w.T[:, cols]),        # K = hidden
        "grad_dX": compare_gemm_paths(dy, w[:, cols]),               # K = ffn
        "grad_dW": compare_gemm_paths(dy.T[: 2 * n_slice], m_in[:, : 2 * n_slice]),  # K = tokens
    }
    worst_cos = min(g["mercury"]["cos_sim"] for g in gemms.values())
    return {
        "model": {"hidden": hidden, "ffn": ffn, "n_heads": n_heads, "batch": batch, "seq": seq,
                  "dtype_golden": "torch.float64", "loss": float(loss.detach())},
        "outliers": {"magnitude": outlier_magnitude, "stride": 32,
                     "max_abs_activation": float(x.detach().abs().max())},
        "autograd_alignment": alignment,
        "gemms": gemms,
        "summary": {
            "mercury_min_cos_sim": worst_cos,
            "mercury_preserves_0p999": worst_cos > 0.999,
            # FP8 quantization dominates end-to-end error and is identical on both
            # paths (same bytes), so the end-to-end margin is tiny and can flip sign.
            # The accumulator-only comparison is what isolates the 14-bit drift.
            "mercury_beats_hopper_end_to_end_rel_l2": {
                name: g["mercury"]["rel_l2"] < g["hopper_14bit"]["rel_l2"] for name, g in gemms.items()},
            "mercury_beats_hopper_accumulator_rel_l2": all(
                g["mercury_accumulator_only"]["rel_l2"] < g["hopper_accumulator_only"]["rel_l2"]
                for g in gemms.values()),
            "accumulator_error_ratio_hopper_over_mercury": {
                name: g["hopper_accumulator_only"]["rel_l2"] / max(g["mercury_accumulator_only"]["rel_l2"], 1e-300)
                for name, g in gemms.items()},
        },
    }


# --------------------------------------------------------------------------
# Vectorized, bit-exact FP8 GEMM paths for multi-step training
# --------------------------------------------------------------------------

E4M3_MAX = 448.0
FP8_GEMM_MODES = ("fp8_quant_fp64_accum", "mercury_fp8_blk", "hopper_14bit")


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32).to(torch.float64)


def _block_scale(amax: torch.Tensor) -> torch.Tensor:
    s = _f32(amax / E4M3_MAX).clamp_min(2.0**-149)
    return torch.where(amax > 0, s, torch.ones_like(s))


def _encode_e4m3(v: torch.Tensor) -> torch.Tensor:
    """RNE to OCP E4M3 with saturation, returned as exact FP64 values."""
    return v.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float64)


def _pad(x: torch.Tensor, dim: int, mult: int = BLOCK) -> torch.Tensor:
    extra = -x.shape[dim] % mult
    if not extra:
        return x
    shape = list(x.shape)
    shape[dim] = extra
    return torch.cat([x, x.new_zeros(shape)], dim)


def quantize_rows_1x128(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch twin of ``block_quantize_1x128``: FP8 values [M, Kp], scales [M, G]."""
    a = _pad(a.double(), 1)
    m, kp = a.shape
    blocks = a.view(m, kp // BLOCK, BLOCK)
    s = _block_scale(blocks.abs().amax(-1))
    return _encode_e4m3(blocks / s[..., None]).view(m, kp), s


def quantize_blocks_128x128(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch twin of ``block_quantize_128x128``: FP8 values [Kp, N], scales [G, N]."""
    b = _pad(b.double(), 0)
    kp, n = b.shape
    bp = _pad(b, 1)
    s = _block_scale(bp.view(kp // BLOCK, BLOCK, -1, BLOCK).abs().amax((1, 3)))
    per_col = s.repeat_interleave(BLOCK, 1)[:, :n]
    return _encode_e4m3(b / per_col.repeat_interleave(BLOCK, 0)), per_col


def _fp8_operands(a: torch.Tensor, b: torch.Tensor):
    va, sa = quantize_rows_1x128(a)
    vb, sb = quantize_blocks_128x128(b)
    groups = va.shape[1] // BLOCK
    scale = _f32(sa.T[:, :, None] * sb[:, None, :])  # [G, M, N]: one FP32 scale product per group
    return va, vb, groups, scale


def mercury_fp8_gemm(a: torch.Tensor, b: torch.Tensor, *, exact_order: bool = True) -> torch.Tensor:
    """m.mma.fp8.blk: FP32 partial per K=128 group, then one FP32 FMA by the block scale.

    ``exact_order=True`` adds the products in strict K order (bit-exact with
    ``compute_isa``). ``False`` uses an FP32 BLAS batched GEMM for the same
    partials. The precision class is the same but the summation order is not
    specified; the deviation is measured in the tests. The training loop
    uses this to run about 6x faster.
    """
    va, vb, groups, scale = _fp8_operands(a, b)
    m, n = va.shape[0], vb.shape[1]
    av = va.view(m, groups, BLOCK).permute(1, 0, 2).float()
    bv = vb.view(groups, BLOCK, n).float()
    if exact_order:
        part = torch.zeros(groups, m, n, dtype=torch.float32)
        for k in range(BLOCK):  # FP8 x FP8 products are exact in FP32; each add rounds RNE
            part = part + av[:, :, k:k + 1] * bv[:, k:k + 1, :]
    else:
        part = torch.bmm(av, bv)
    acc = torch.zeros(m, n, dtype=torch.float64)
    for g in range(groups):  # FP64 holds the product exactly, then one FP32 rounding
        acc = _f32(part[g].double() * scale[g] + acc)
    return acc


def hopper_fp8_gemm(a: torch.Tensor, b: torch.Tensor, alignment_width: int = 32) -> torch.Tensor:
    """14-bit aligned truncation per 32 products, FP32 partial, separately rounded descale/add."""
    va, vb, groups, scale = _fp8_operands(a, b)
    m, n = va.shape[0], vb.shape[1]
    # K-outer layout [G, K, M] x [G, K, N]: the reduction over each 32-product window
    # then runs across contiguous M x N planes, which is about 2.7x faster than a
    # last-dimension reduction.
    av = va.view(m, groups, BLOCK).permute(1, 2, 0).float()
    bv = vb.view(groups, BLOCK, n).float()
    width = BLOCK
    if groups == 1:  # short K (e.g. dW over 16 tokens): skip all-zero padded windows
        k_real = a.shape[1]
        if k_real < alignment_width:  # a single short alignment window, as in compute_isa
            alignment_width = width = k_real
        else:
            width = -(-k_real // alignment_width) * alignment_width
        av, bv = av[:, :width], bv[:, :width]
    chunks = width // alignment_width
    prod = (av[:, :, :, None] * bv[:, :, None, :]).view(groups, chunks, alignment_width, m, n)
    peak = prod.abs().amax(2)
    # Power-of-two quantum: scaling by its reciprocal is exact, then truncate toward zero.
    exponent = torch.frexp(peak).exponent
    inv_quantum = torch.ldexp(torch.ones_like(peak), 14 - exponent)
    aligned = prod.mul_(inv_quantum[:, :, None]).trunc_().sum(2)  # integer sums: exact
    aligned = torch.where(peak > 0, aligned * torch.ldexp(torch.ones_like(peak), exponent - 14),
                          torch.zeros_like(aligned))
    part = torch.zeros(groups, m, n, dtype=torch.float32)
    for c in range(chunks):
        part = part + aligned[:, c]
    acc = torch.zeros(m, n, dtype=torch.float32)
    for g in range(groups):
        acc = acc + (part[g].double() * scale[g]).float()
    return acc.double()


def fp8_quant_fp64_accum_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Same FP8 bytes and scales, exact FP64 accumulation: isolates quantization error."""
    va, vb, groups, scale = _fp8_operands(a, b)
    m, n = va.shape[0], vb.shape[1]
    part = torch.bmm(va.view(m, groups, BLOCK).permute(1, 0, 2), vb.view(groups, BLOCK, n))
    return (part * scale).sum(0)


_GEMMS = {
    "fp8_quant_fp64_accum": fp8_quant_fp64_accum_gemm,
    "mercury_fp8_blk": lambda a, b: mercury_fp8_gemm(a, b, exact_order=False),
    "hopper_14bit": hopper_fp8_gemm,
}


class FP8Linear(torch.autograd.Function):
    """Y = X W^T; dX = dY W; dW = dY^T X, all three through one FP8 GEMM path."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, mode: str) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.mode = mode
        return _GEMMS[mode](x, weight.T)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight = ctx.saved_tensors
        gemm = _GEMMS[ctx.mode]
        return gemm(dy, weight), gemm(dy.T, x), None


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


@lru_cache(maxsize=4)
def _training(steps: int, hidden: int, ffn: int, n_heads: int, batch: int, seq: int,
              outlier_magnitude: float, lr: float, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    init = MercuryTransformerBlock(hidden, n_heads, ffn).double().state_dict()
    x = correlated_outlier_activations(batch, seq, hidden, outlier_magnitude=outlier_magnitude, seed=seed)
    target = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64)
    w0 = None
    finals: dict[str, torch.Tensor] = {}
    curves: dict[str, list[float]] = {}
    grads0: dict[str, torch.Tensor] = {}
    for mode in ("fp64",) + FP8_GEMM_MODES:
        model = MercuryTransformerBlock(hidden, n_heads, ffn).double()
        model.load_state_dict(init)
        model.gemm_mode = mode
        if w0 is None:
            w0 = _flat_params(model)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        losses = []
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), target)
            loss.backward()
            if step == 0:  # same weights on every path: a clean one-step gradient comparison
                grads0[mode] = torch.cat([q.grad.flatten() for q in model.parameters()])
            opt.step()
            losses.append(float(loss.detach()))
        curves[mode] = losses
        finals[mode] = _flat_params(model)
    gold = finals["fp64"]
    update_norm = float((gold - w0).norm())
    quant_w, quant_g = finals["fp8_quant_fp64_accum"], grads0["fp8_quant_fp64_accum"]
    rel = lambda u, v: float((u - v).norm() / v.norm())
    paths = {}
    for mode, w in finals.items():
        diff = float((w - gold).norm())
        curve = curves[mode]
        paths["fp64_golden" if mode == "fp64" else mode] = {
            "loss_curve": curve,
            "initial_loss": curve[0],
            "final_loss": curve[-1],
            "loss_reduction": 1.0 - curve[-1] / curve[0],
            "max_loss_rel_dev_vs_fp64": max(abs(a - g) / g for a, g in zip(curve, curves["fp64"])),
            "final_weight_rel_l2_drift": diff / float(gold.norm()),
            "update_rel_l2_drift": diff / update_norm,
            "update_rel_l2_drift_vs_fp8_exact_accum": float((w - quant_w).norm()) / update_norm,
            "first_step_grad_rel_l2_vs_fp64": rel(grads0[mode], grads0["fp64"]),
            "first_step_grad_rel_l2_vs_fp8_exact_accum": rel(grads0[mode], quant_g),
            "first_step_grad_cos_sim_vs_fp64": float(F.cosine_similarity(grads0[mode], grads0["fp64"], dim=0)),
        }
    quant = paths["fp8_quant_fp64_accum"]["update_rel_l2_drift"]
    merc = paths["mercury_fp8_blk"]["update_rel_l2_drift"]
    hop = paths["hopper_14bit"]["update_rel_l2_drift"]
    return {
        "config": {"steps": steps, "hidden": hidden, "ffn": ffn, "n_heads": n_heads, "batch": batch,
                   "seq": seq, "outlier_magnitude": outlier_magnitude, "optimizer": "torch.optim.AdamW",
                   "lr": lr, "weight_decay": 0.01, "fp8_gemms_per_step": 7 * 3,
                   "fp64_ops": "RMSNorm, softmax/SDPA internals, SiLU, AdamW update"},
        "paths": paths,
        "attribution": {
            # Weight-update drift vs FP64, relative to the FP64 update's norm.
            "fp8_quantization_update_drift": quant,
            "mercury_update_drift": merc,
            "hopper_update_drift": hop,
            "mercury_extra_over_quantization": merc - quant,
            "hopper_extra_over_quantization": hop - quant,
            "dominant_error_source": "fp8_quantization" if abs(merc - quant) < 0.1 * quant else "accumulation",
            # Accumulator-only effect: distance from the FP8 path with exact FP64 accumulation.
            "mercury_first_step_grad_err_vs_fp8_exact_accum":
                paths["mercury_fp8_blk"]["first_step_grad_rel_l2_vs_fp8_exact_accum"],
            "hopper_first_step_grad_err_vs_fp8_exact_accum":
                paths["hopper_14bit"]["first_step_grad_rel_l2_vs_fp8_exact_accum"],
            "note": ("Same FP8 bytes on all FP8 paths. End-to-end drift vs FP64 is set by FP8 "
                     "quantization (~27% of the 20-step update norm on this outlier stress case), "
                     "and multi-step AdamW trajectories decorrelate, so end-to-end Hopper vs Mercury "
                     "differences are noise. The accumulator signal is the *_vs_fp8_exact_accum metrics."),
        },
    }


def simulate_fp8_training_loop(
    steps: int = 20, *, hidden: int = 256, ffn: int = 512, n_heads: int = 4, batch: int = 2,
    seq: int = 8, outlier_magnitude: float = 24.0, lr: float = 3e-3, seed: int = 0,
) -> dict[str, Any]:
    """Train identical blocks with AdamW on FP64, FP8+FP64-accum, Mercury and Hopper GEMMs."""
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    return deepcopy(_training(steps, hidden, ffn, n_heads, batch, seq,
                              float(outlier_magnitude), float(lr), seed))


def run_pytorch_bridge_benchmark(
    hidden: int = 512, ffn: int = 1024, n_heads: int = 8, batch: int = 2, seq: int = 8,
    n_slice: int = 32, outlier_magnitude: float = 24.0, seed: int = 0,
) -> dict[str, Any]:
    """JSON-serializable forward/gradient parity metrics for one Transformer block."""
    report = deepcopy(_bridge(hidden, ffn, n_heads, batch, seq, n_slice, float(outlier_magnitude), seed))
    report["training_convergence"] = simulate_fp8_training_loop(20)
    return report


# --------------------------------------------------------------------------
# Multi-seed, multi-distribution FP8 training validation
# --------------------------------------------------------------------------

TRAINING_DISTRIBUTIONS = ("gaussian", "lognormal_outlier_24", "student_t3", "outlier_100")
VALIDATION_PATHS = ("fp64_golden",) + FP8_GEMM_MODES
_MODE_OF_PATH = {"fp64_golden": "fp64", **{m: m for m in FP8_GEMM_MODES}}
_TRAINING_ERRORS = (RuntimeError, ValueError, FloatingPointError, OverflowError)
_TARGET_SEED_OFFSET = 10_007
# torch.manual_seed accepts at most 2**64 - 1; the target generator uses seed + offset.
MAX_VALIDATION_SEED = 2**64 - 1 - _TARGET_SEED_OFFSET


def seeded_activations(distribution: str, batch: int, seq: int, hidden: int, seed: int) -> torch.Tensor:
    """Reproducible FP64 activations [batch, seq, hidden] from a named distribution.

    * ``gaussian``: i.i.d. N(0, 1), so no outliers (a control for the outlier cases).
    * ``lognormal_outlier_24``: per-channel signs and lognormal magnitudes,
      plus one magnitude-24 channel per 32 channels (correlated outliers).
    * ``student_t3``: i.i.d. Student-t with 3 degrees of freedom (heavy tails,
      outliers scattered rather than per channel).
    * ``outlier_100``: like ``lognormal_outlier_24`` but at magnitude 100.
    """
    if distribution == "lognormal_outlier_24":
        return correlated_outlier_activations(batch, seq, hidden, outlier_magnitude=24.0, seed=seed)
    if distribution == "outlier_100":
        return correlated_outlier_activations(batch, seq, hidden, outlier_magnitude=100.0, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, seq, hidden)
    if distribution == "gaussian":
        return torch.randn(shape, generator=gen, dtype=torch.float64)
    if distribution == "student_t3":
        z = torch.randn(shape, generator=gen, dtype=torch.float64)
        chi2 = torch.randn((3, *shape), generator=gen, dtype=torch.float64).pow(2).sum(0)
        return z / torch.sqrt(chi2 / 3.0)
    raise ValueError(f"Unknown distribution {distribution!r}; choose from {TRAINING_DISTRIBUTIONS}")


def _train_one_path(mode: str, init: dict[str, torch.Tensor], x: torch.Tensor, target: torch.Tensor,
                    steps: int, hidden: int, ffn: int, n_heads: int, lr: float) -> dict[str, Any]:
    with torch.random.fork_rng():  # construction draws init RNG; weights are reloaded below
        model = MercuryTransformerBlock(hidden, n_heads, ffn).double()
    model.load_state_dict(init)
    model.gemm_mode = mode
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses: list[float] = []
    grad0 = None
    try:
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), target)
            value = float(loss.detach())
            if not math.isfinite(value):
                return {"status": "non_finite_loss", "losses": losses, "step": step}
            loss.backward()
            grads = torch.cat([p.grad.flatten() for p in model.parameters()])
            if not bool(torch.isfinite(grads).all()):
                return {"status": "non_finite_grad", "losses": losses, "step": step}
            if step == 0:
                grad0 = grads.clone()
            opt.step()
            losses.append(value)
    except _TRAINING_ERRORS as exc:  # report, never hide, a failing numeric path
        return {"status": f"error: {type(exc).__name__}: {exc}", "losses": losses, "step": len(losses)}
    weights = _flat_params(model)
    if not bool(torch.isfinite(weights).all()):
        return {"status": "non_finite_weights", "losses": losses, "step": steps}
    return {"status": "ok", "losses": losses, "grad0": grad0, "weights": weights}


def _rel(u: torch.Tensor, v: torch.Tensor, denom: torch.Tensor | None = None) -> float:
    d = float((v if denom is None else denom).norm())
    return float((u - v).norm()) / d if d else float((u - v).norm())


@lru_cache(maxsize=64)
def _validation_run(seed: int, distribution: str, steps: int, hidden: int, ffn: int, n_heads: int,
                    batch: int, seq: int, lr: float) -> dict[str, Any]:
    with torch.random.fork_rng():  # do not disturb the caller's global RNG
        torch.manual_seed(seed)
        init = MercuryTransformerBlock(hidden, n_heads, ffn).double().state_dict()
    w0 = torch.cat([t.flatten() for t in init.values()])
    x = seeded_activations(distribution, batch, seq, hidden, seed)
    target = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed + _TARGET_SEED_OFFSET), dtype=torch.float64)
    raw = {path: _train_one_path(_MODE_OF_PATH[path], init, x, target, steps, hidden, ffn, n_heads, lr)
           for path in VALIDATION_PATHS}
    gold, quant = raw["fp64_golden"], raw["fp8_quant_fp64_accum"]
    paths: dict[str, Any] = {}
    for path, r in raw.items():
        entry: dict[str, Any] = {"status": r["status"], "loss_curve": r["losses"]}
        if r["status"] == "ok":
            curve = r["losses"]
            entry.update({"final_loss": curve[-1],
                          "loss_reduction": 1.0 - curve[-1] / curve[0] if curve[0] else None})
            if gold["status"] == "ok":
                update = gold["weights"] - w0
                entry["update_drift_vs_fp64"] = _rel(r["weights"], gold["weights"], update)
                entry["first_step_grad_err_vs_fp64"] = _rel(r["grad0"], gold["grad0"])
            if quant["status"] == "ok" and path != "fp64_golden":
                entry["update_drift_vs_fp8_exact_accum"] = _rel(r["weights"], quant["weights"],
                                                                quant["weights"] - w0)
                entry["first_step_grad_err_vs_fp8_exact_accum"] = _rel(r["grad0"], quant["grad0"])
        else:
            entry["failed_at_step"] = r["step"]
        paths[path] = entry
    return {"seed": seed, "distribution": distribution,
            "max_abs_activation": float(x.abs().max()), "paths": paths}


def _reduces_loss(entry: dict[str, Any]) -> bool:
    reduction = entry.get("loss_reduction")
    return reduction is not None and reduction > 0


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": math.fsum(values) / len(values), "min": min(values), "max": max(values), "n": len(values)}


def run_training_validation(
    seeds: tuple[int, ...] = (0, 1, 2),
    distributions: tuple[str, ...] = TRAINING_DISTRIBUTIONS,
    steps: int = 8,
    *, hidden: int = 256, ffn: int = 512, n_heads: int = 4, batch: int = 2, seq: int = 8,
    lr: float = 3e-3,
) -> dict[str, Any]:
    """Multi-seed FP8 training sensitivity: FP64 vs FP8-exact-accum vs Mercury vs Hopper.

    Within one (seed, distribution) run, every path gets the same initial
    weights, data, target, optimizer and step count. All FP8 paths use one
    quantizer (``_fp8_operands``). They see identical FP8 bytes only where
    their GEMM operands are identical: the step-0 forward GEMMs that read
    the shared input and initial weights. After the first FP8 GEMM, the
    intermediate activations, gradients and, later, the weights differ per
    path, so each path re-quantizes slightly different tensors. The
    ``*_vs_fp8_exact_accum`` metrics therefore measure accumulator-path
    sensitivity, which FP8 re-quantization can amplify. They are not a
    bit-paired accumulator-only measurement; the bit-paired one is
    ``compare_gemm_paths``.

    Inputs are validated. The caller's global torch RNG state is preserved.
    The result is strict JSON: ``json.dumps(..., allow_nan=False)`` succeeds,
    and any non-finite metric becomes ``None``.
    """
    seeds = tuple(seeds)
    distributions = tuple(distributions)
    if not seeds or not distributions:
        raise ValueError("need at least one seed and one distribution")
    if any(type(s) is not int or not 0 <= s <= MAX_VALIDATION_SEED for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct integers in [0, {MAX_VALIDATION_SEED}]")
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be an integer >= 2")
    for d in distributions:
        if d not in TRAINING_DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution {d!r}; choose from {TRAINING_DISTRIBUTIONS}")
    if len(set(distributions)) != len(distributions):
        raise ValueError("distributions must be distinct")
    dims = {"hidden": hidden, "ffn": ffn, "n_heads": n_heads, "batch": batch, "seq": seq}
    for name, value in dims.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if hidden % n_heads:
        raise ValueError("hidden must be divisible by n_heads")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 0 < lr <= 1:
        raise ValueError("lr must be a finite number in (0, 1]")
    cfg = (steps, hidden, ffn, n_heads, batch, seq, float(lr))
    with torch.random.fork_rng():
        return _json_strict(_training_validation(seeds, distributions, cfg))


def _json_strict(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so allow_nan=False always works."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_strict(v) for v in obj]
    return obj


def _training_validation(seeds: tuple[int, ...], distributions: tuple[str, ...],
                         cfg: tuple) -> dict[str, Any]:
    steps, hidden, ffn, n_heads, batch, seq, lr = cfg
    runs = [deepcopy(_validation_run(s, d, *cfg)) for d in distributions for s in seeds]

    metrics = ("final_loss", "loss_reduction", "update_drift_vs_fp64", "first_step_grad_err_vs_fp64",
               "update_drift_vs_fp8_exact_accum", "first_step_grad_err_vs_fp8_exact_accum")
    aggregate: dict[str, Any] = {}
    for path in VALIDATION_PATHS:
        ok = [r["paths"][path] for r in runs if r["paths"][path]["status"] == "ok"]
        aggregate[path] = {"ok_runs": len(ok), "failed_runs": len(runs) - len(ok)}
        for m in metrics:
            vals = [p[m] for p in ok if p.get(m) is not None]  # skip undefined metrics
            if vals:
                aggregate[path][m] = _stats(vals)

    def accum_ratio(r: dict[str, Any]) -> float | None:
        h = r["paths"]["hopper_14bit"].get("first_step_grad_err_vs_fp8_exact_accum")
        mm = r["paths"]["mercury_fp8_blk"].get("first_step_grad_err_vs_fp8_exact_accum")
        return None if h is None or mm is None else h / max(mm, 1e-300)

    per_distribution = {}
    for d in distributions:
        rs = [r for r in runs if r["distribution"] == d]
        ratios = [x for x in map(accum_ratio, rs) if x is not None]
        quant = [r["paths"]["fp8_quant_fp64_accum"].get("update_drift_vs_fp64") for r in rs]
        per_distribution[d] = {
            "hopper_over_mercury_accum_grad_err": _stats(ratios) if ratios else None,
            "fp8_quantization_update_drift_vs_fp64": _stats([q for q in quant if q is not None]) if any(
                q is not None for q in quant) else None,
        }

    # Controls: determinism of the FP64 reference, and exact-vs-fast Mercury partial order.
    # Both report a status instead of raising, so failures stay visible in the JSON.
    probe = runs[0]
    rerun = _train_one_path("fp64", *_probe_inputs(probe["seed"], probe["distribution"], hidden, ffn,
                                                   n_heads, batch, seq), steps, hidden, ffn, n_heads, lr)
    first_gold = _validation_gold_weights(probe["seed"], probe["distribution"], *cfg)
    if rerun["status"] != "ok" or first_gold is None:
        rerun_equal, rerun_status = None, "reference failed: " + rerun["status"]
    else:
        rerun_equal, rerun_status = bool(torch.equal(rerun["weights"], first_gold)), "ok"
    try:
        a_probe = seeded_activations(probe["distribution"], batch, seq, hidden, probe["seed"]).reshape(-1, hidden)
        with torch.random.fork_rng():
            torch.manual_seed(probe["seed"])
            w_probe = MercuryTransformerBlock(hidden, n_heads, ffn).double().gate_proj.weight.detach()
        exact = mercury_fp8_gemm(a_probe, w_probe.T, exact_order=True)
        fast = mercury_fp8_gemm(a_probe, w_probe.T, exact_order=False)
        order_rel, order_status = _rel(fast, exact), "ok"
    except _TRAINING_ERRORS as exc:
        order_rel, order_status = None, f"error: {type(exc).__name__}: {exc}"
    controls = {
        "fp64_rerun_bitwise_equal": rerun_equal,
        "fp64_rerun_status": rerun_status,
        "same_quantizer_across_fp8_paths": True,  # one _fp8_operands implementation for all FP8 GEMMs
        "identical_fp8_bytes_scope": ("only GEMMs with identical operands (step-0 forward GEMMs that read "
                                      "the shared input and initial weights); later operands diverge per path"),
        "mercury_fast_vs_exact_order_rel_l2": order_rel,
        "mercury_order_probe_status": order_status,
        "quantization_only_path": "fp8_quant_fp64_accum",
        "accumulator_path_sensitivity_metric": (
            "*_vs_fp8_exact_accum: each FP8 path vs fp8_quant_fp64_accum through the full block. "
            "Includes FP8 re-quantization amplification; not bit-paired accumulator-only error "
            "(see compare_gemm_paths for that)."),
    }

    ok_pairs = [r for r in runs if all(r["paths"][p]["status"] == "ok" for p in VALIDATION_PATHS)]
    merc_acc = [r["paths"]["mercury_fp8_blk"]["first_step_grad_err_vs_fp8_exact_accum"] for r in ok_pairs]
    hop_acc = [r["paths"]["hopper_14bit"]["first_step_grad_err_vs_fp8_exact_accum"] for r in ok_pairs]
    quant_dom = [abs(r["paths"]["mercury_fp8_blk"]["update_drift_vs_fp64"]
                     - r["paths"]["fp8_quant_fp64_accum"]["update_drift_vs_fp64"])
                 < 0.1 * r["paths"]["fp8_quant_fp64_accum"]["update_drift_vs_fp64"] for r in ok_pairs]
    e2e_merc_wins = [r["paths"]["mercury_fp8_blk"]["update_drift_vs_fp64"]
                     < r["paths"]["hopper_14bit"]["update_drift_vs_fp64"] for r in ok_pairs]
    traj_merc_wins = [r["paths"]["mercury_fp8_blk"]["update_drift_vs_fp8_exact_accum"]
                      < r["paths"]["hopper_14bit"]["update_drift_vs_fp8_exact_accum"] for r in ok_pairs]
    verdict = {
        "runs": len(runs),
        "all_paths_finite": all(r["paths"][p]["status"] == "ok" for r in runs for p in VALIDATION_PATHS),
        # An undefined reduction (zero initial loss) counts as not converging, never as a crash.
        "all_paths_reduce_loss": all(_reduces_loss(r["paths"][p]) for r in runs for p in VALIDATION_PATHS),
        "undefined_loss_reduction_paths": sum(
            r["paths"][p]["status"] == "ok" and r["paths"][p].get("loss_reduction") is None
            for r in runs for p in VALIDATION_PATHS),
        "mercury_accum_grad_err_max": max(merc_acc) if merc_acc else None,
        "hopper_accum_grad_err_min": min(hop_acc) if hop_acc else None,
        "mercury_accum_below_hopper_runs": sum(m < h for m, h in zip(merc_acc, hop_acc)),
        "quantization_dominates_end_to_end_runs": sum(quant_dom),
        "mercury_beats_hopper_end_to_end_runs": sum(e2e_merc_wins),
        "mercury_closer_to_exact_accum_trajectory_runs": sum(traj_merc_wins),
        "comparable_runs": len(ok_pairs),
        "metric_semantics": ("mercury_accum_* / hopper_accum_* are accumulator-path sensitivities of "
                             "the step-0 gradient vs fp8_quant_fp64_accum, not bit-paired "
                             "accumulator-only errors."),
        "note": ("A Mercury accumulator-path gradient error far above FP32 epsilon (seen at ~7e-4 for "
                 "student_t3 seed 1 with the default config, concentrated in Q/K/V/attn_norm and "
                 "unchanged with strict K order) is consistent with FP8 re-quantization flips: a ~1e-7 "
                 "FP32 difference upstream moves a value across an E4M3 rounding midpoint in a "
                 "downstream GEMM."),
    }
    return {
        "config": {"seeds": list(seeds), "distributions": list(distributions), "steps": steps,
                   "hidden": hidden, "ffn": ffn, "n_heads": n_heads, "batch": batch, "seq": seq,
                   "optimizer": "torch.optim.AdamW", "lr": float(lr), "weight_decay": 0.01,
                   "paths": list(VALIDATION_PATHS)},
        "runs": runs,
        "aggregate": aggregate,
        "per_distribution": per_distribution,
        "controls": controls,
        "verdict": verdict,
        "limitations": [
            "Synthetic memorization task: one fixed batch of 16 tokens and a random target, "
            "not language-model training. Short horizon (default 8 steps).",
            "One pre-norm block (hidden 256, ffn 512). Only the 7 nn.Linear layers run FP8 "
            "(forward Y, dX, dW). SDPA QK^T/PV, RMSNorm, SiLU and AdamW run in FP64.",
            "The training path computes Mercury partials with FP32 BLAS order "
            "(controls.mercury_fast_vs_exact_order_rel_l2). Strict-K-order bit-exactness vs "
            "compute_isa is tested on sample GEMMs, not on every training GEMM.",
            "The Hopper model is the DeepSeek-V3-reported 14-bit/32-product alignment approximation, "
            "not measured H100 hardware.",
            "AdamW normalizes update magnitudes and trajectories decorrelate over steps, so "
            "end-to-end drift mixes accumulator error with chaotic divergence. The step-0 "
            "gradient error vs fp8_quant_fp64_accum is the cleanest training-level accumulator "
            "signal, but it still includes FP8 re-quantization amplification.",
            "FP8 bytes are bit-identical across FP8 paths only for GEMMs with identical operands; "
            "after the first FP8 GEMM each path re-quantizes its own (slightly different) tensors.",
            "CPU-only emulation. It says nothing about speed or about convergence at scale.",
        ],
    }


def _probe_inputs(seed: int, distribution: str, hidden: int, ffn: int, n_heads: int,
                  batch: int, seq: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        init = MercuryTransformerBlock(hidden, n_heads, ffn).double().state_dict()
    x = seeded_activations(distribution, batch, seq, hidden, seed)
    target = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed + _TARGET_SEED_OFFSET), dtype=torch.float64)
    return init, x, target


@lru_cache(maxsize=16)
def _validation_gold_weights(seed: int, distribution: str, steps: int, hidden: int, ffn: int,
                             n_heads: int, batch: int, seq: int, lr: float) -> torch.Tensor | None:
    """FP64 reference weights for the determinism control, or None if that run failed."""
    init, x, target = _probe_inputs(seed, distribution, hidden, ffn, n_heads, batch, seq)
    out = _train_one_path("fp64", init, x, target, steps, hidden, ffn, n_heads, lr)
    return out["weights"] if out["status"] == "ok" else None
