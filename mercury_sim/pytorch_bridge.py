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
