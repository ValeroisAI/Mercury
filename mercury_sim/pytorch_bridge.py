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

    def forward(self, x: torch.Tensor, capture: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        b, s, d = x.shape
        h = self.attn_norm(x)
        heads = lambda t: t.view(b, s, self.n_heads, d // self.n_heads).transpose(1, 2)
        q, k, v = heads(self.q_proj(h)), heads(self.k_proj(h)), heads(self.v_proj(h))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(b, s, d))
        m = self.mlp_norm(x)
        gate = self.gate_proj(m)
        if capture is not None:
            m.retain_grad()
            gate.retain_grad()
            capture["mlp_in"], capture["gate_out"] = m, gate
        return x + self.down_proj(F.silu(gate) * self.up_proj(m))


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


def run_pytorch_bridge_benchmark(
    hidden: int = 512, ffn: int = 1024, n_heads: int = 8, batch: int = 2, seq: int = 8,
    n_slice: int = 32, outlier_magnitude: float = 24.0, seed: int = 0,
) -> dict[str, Any]:
    """JSON-serializable forward/gradient parity metrics for one Transformer block."""
    return deepcopy(_bridge(hidden, ffn, n_heads, batch, seq, n_slice, float(outlier_magnitude), seed))
