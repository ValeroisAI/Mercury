"""Functional Mercury-1 STME and SCC scheduling models.

FP8 encoding follows the OCP OFP8 specification:
https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
import random
import struct

from .specs import MERCURY_1D, MercurySKU


BLOCK = 128
_FP8_LAYOUTS = {"e4m3": (3, 7, 0x7E), "e5m2": (2, 15, 0x7B)}


def _decode_byte(code: int, fmt: str) -> float:
    fraction_bits, bias, max_code = _FP8_LAYOUTS[fmt]
    magnitude = code & 0x7F
    sign = -1.0 if code & 0x80 else 1.0
    if magnitude > max_code:
        special = math.inf if fmt == "e5m2" and magnitude == 0x7C else math.nan
        return math.copysign(special, sign)
    exponent = magnitude >> fraction_bits
    fraction = magnitude & ((1 << fraction_bits) - 1)
    if exponent:
        fraction += 1 << fraction_bits
    return sign * math.ldexp(fraction, max(exponent, 1) - bias - fraction_bits)


_DECODE = {fmt: tuple(_decode_byte(i, fmt) for i in range(256)) for fmt in _FP8_LAYOUTS}
_POSITIVE = {fmt: _DECODE[fmt][:layout[2] + 1] for fmt, layout in _FP8_LAYOUTS.items()}


def _format(fmt: str) -> str:
    fmt = fmt.lower()
    if fmt not in _FP8_LAYOUTS:
        raise ValueError(f"Unsupported FP8 format: {fmt}")
    return fmt


def decode_fp8(code: int, fmt: str = "e4m3") -> float:
    """Decode an OCP OFP8 byte, including signed zeros and specials."""
    if not isinstance(code, int) or not 0 <= code <= 255:
        raise ValueError("FP8 code must be an integer byte")
    return _DECODE[_format(fmt)][code]


def _encode(value: float, fmt: str, saturate: bool) -> int:
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    nan_code = 0x7F if fmt == "e4m3" else 0x7E
    if math.isnan(value):
        return sign | nan_code
    levels = _POSITIVE[fmt]
    magnitude = abs(value)
    hi = bisect_left(levels, magnitude)
    if hi == len(levels):
        max_value = levels[-1]
        midpoint = max_value + (max_value - levels[-2]) / 2
        if saturate or magnitude < midpoint or (magnitude == midpoint and fmt == "e4m3"):
            return sign | (hi - 1)
        return sign | (0x7C if fmt == "e5m2" else nan_code)
    if hi:
        low_delta = magnitude - levels[hi - 1]
        high_delta = levels[hi] - magnitude
        if low_delta < high_delta or (low_delta == high_delta and hi % 2):
            hi -= 1
    return sign | hi


def encode_fp8(value: float, fmt: str = "e4m3", *, saturate: bool = True) -> int:
    """Encode with nearest-even rounding and selectable OCP OFP8 saturation."""
    return _encode(value, _format(fmt), saturate)


def _f32(value: float) -> float:
    try:
        return struct.unpack(">f", struct.pack(">f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _fma_f32(a: float, b: float, c: float) -> float:
    """Round an exact product-plus-add once to binary32, including subnormals."""
    if not all(math.isfinite(x) for x in (a, b, c)):
        return _f32(a * b + c)
    na, da = a.as_integer_ratio()
    nb, db = b.as_integer_ratio()
    nc, dc = c.as_integer_ratio()
    product_denominator = da * db
    denominator = max(product_denominator, dc)
    exact = na * nb * (denominator // product_denominator) + nc * (denominator // dc)
    if exact == 0:
        # An exact zero under RNE is negative only for two negative zeros.
        product = a * b
        negative_zeros = product == c == 0 and math.copysign(1, product) == math.copysign(1, c) == -1
        return -0.0 if negative_zeros else 0.0
    sign = -1.0 if exact < 0 else 1.0
    significand = abs(exact)
    exponent = 1 - denominator.bit_length()
    shift = max(0, significand.bit_length() - 24, -149 - exponent)
    if shift:
        significand, remainder = divmod(significand, 1 << shift)
        half = 1 << (shift - 1)
        significand += remainder > half or (remainder == half and significand % 2)
    return _f32(math.ldexp(sign * significand, exponent + shift))


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _check_dimensions(*dimensions: int) -> None:
    if any(type(d) is not int or d <= 0 for d in dimensions):
        raise ValueError("Matrix and tile dimensions must be positive integers")


def _check_length(values: Sequence[object], expected: int, name: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} elements, got {len(values)}")


def _quantize(
    matrix: Sequence[float], rows: int, cols: int, row_block: int, fmt: str
) -> tuple[bytes, list[float]]:
    _check_dimensions(rows, cols)
    _check_length(matrix, rows * cols, "matrix")
    if any(not math.isfinite(x) for x in matrix):
        raise ValueError("Block quantization requires finite inputs")
    fmt = _format(fmt)
    output = bytearray(rows * cols)
    scales = []
    for row in range(0, rows, row_block):
        for col in range(0, cols, BLOCK):
            end_col = min(col + BLOCK, cols)
            spans = [(i * cols + col, i * cols + end_col) for i in range(row, min(row + row_block, rows))]
            maximum = max(abs(matrix[j]) for start, end in spans for j in range(start, end))
            scale = max(_f32(maximum / _POSITIVE[fmt][-1]), 2.0**-149) if maximum else 1.0
            if not math.isfinite(scale):
                raise ValueError("Block scale exceeds FP32 range")
            scales.append(scale)
            for start, end in spans:
                for j in range(start, end):
                    output[j] = _encode(matrix[j] / scale, fmt, True)
    return bytes(output), scales


def block_quantize_1x128(
    matrix: Sequence[float], rows: int, cols: int, fmt: str = "e4m3"
) -> tuple[bytes, list[float]]:
    """Quantize a flat row-major activation matrix. Scales index [row, K group]."""
    return _quantize(matrix, rows, cols, 1, fmt)


def block_quantize_128x128(
    matrix: Sequence[float], rows: int, cols: int, fmt: str = "e4m3"
) -> tuple[bytes, list[float]]:
    """Quantize a flat row-major weight matrix. Scales index [K group, N group]."""
    return _quantize(matrix, rows, cols, BLOCK, fmt)


def _hopper_sum(products: list[float], alignment_width: int) -> float:
    if not all(math.isfinite(p) for p in products):
        return _f32(sum(products))
    result = 0.0
    for start in range(0, len(products), alignment_width):
        chunk = products[start:start + alignment_width]
        peak = max(map(abs, chunk))
        if peak == 0:
            continue
        quantum = math.ldexp(1.0, math.frexp(peak)[1] - 14)
        # Align to the largest exponent and drop bits toward zero.
        aligned = sum(math.trunc(p / quantum) for p in chunk) * quantum
        result = _f32(result + aligned)
    return result


def _mma(
    A_q: bytes, A_scale: Sequence[float], B_q: bytes, B_scale: Sequence[float],
    M: int, N: int, K: int, a_format: str, b_format: str, alignment_width: int = 0,
    rescale_full_k: bool = False,
) -> list[float]:
    _check_dimensions(M, N, K)
    k_groups = _ceil_div(K, BLOCK)
    n_groups = _ceil_div(N, BLOCK)
    _check_length(A_q, M * K, "A_q")
    _check_length(B_q, K * N, "B_q")
    _check_length(A_scale, M * k_groups, "A_scale")
    _check_length(B_scale, k_groups * n_groups, "B_scale")
    a_scales = [_f32(x) for x in A_scale]
    b_scales = [_f32(x) for x in B_scale]
    if any(not math.isfinite(x) or x <= 0 for x in (*a_scales, *b_scales)):
        raise ValueError("MMA scales must be finite positive FP32 values")
    a_table, b_table = _DECODE[_format(a_format)], _DECODE[_format(b_format)]
    a = [a_table[x] for x in A_q]
    b = [b_table[x] for x in B_q]
    output = []
    for i in range(M):
        for j in range(N):
            acc = 0.0
            scales = [
                _f32(a_scales[i * k_groups + group] * b_scales[group * n_groups + j // BLOCK])
                for group in range(k_groups)
            ]
            if rescale_full_k:
                if any(scale != scales[0] for scale in scales):
                    raise ValueError("Full-K outer rescaling requires a uniform scale product")
                products = [a[i * K + k] * b[k * N + j] for k in range(K)]
                output.append(_f32(_hopper_sum(products, alignment_width) * scales[0]))
                continue
            for group, scale in enumerate(scales):
                start = group * BLOCK
                stop = min(start + BLOCK, K)
                if alignment_width:
                    products = [a[i * K + k] * b[k * N + j] for k in range(start, stop)]
                    partial = _hopper_sum(products, alignment_width)
                    acc = _f32(acc + _f32(partial * scale))
                else:
                    partial = 0.0
                    for k in range(start, stop):
                        partial = _f32(partial + a[i * K + k] * b[k * N + j])
                    acc = _fma_f32(partial, scale, acc)
            output.append(acc)
    return output


def mma_fp8_blk_mercury(
    A_q: bytes, A_scale: Sequence[float], B_q: bytes, B_scale: Sequence[float],
    M: int, N: int, K: int, *, a_format: str = "e4m3", b_format: str = "e4m3",
) -> list[float]:
    """Compute row-major C using K-ordered FP32 partials and one scaled FMA per group."""
    return _mma(A_q, A_scale, B_q, B_scale, M, N, K, a_format, b_format)


def mma_fp8_hopper_truncated(
    A_q: bytes, A_scale: Sequence[float], B_q: bytes, B_scale: Sequence[float],
    M: int, N: int, K: int, *, a_format: str = "e4m3", b_format: str = "e4m3",
    accumulation: str = "group", alignment_width: int = 32,
) -> list[float]:
    """Approximate aligned 14-bit products and separately rounded FP32 descale/add.

    The default alignment width follows DeepSeek-V3 reporting 32 aligned
    products (https://arxiv.org/html/2412.19437#S3.SS5.SSS2). This model
    omits undocumented Hopper accumulator internals. Full-K mode requires a
    uniform scale product; alignment windows still use alignment_width.
    """
    _check_dimensions(alignment_width)
    if accumulation not in ("group", "full"):
        raise ValueError("accumulation must be 'group' or 'full'")
    return _mma(
        A_q, A_scale, B_q, B_scale, M, N, K, a_format, b_format,
        alignment_width, accumulation == "full",
    )


@dataclass(frozen=True)
class NumericalComparisonResult:
    mercury_rel_l2_err: float
    hopper_rel_l2_err: float
    mercury_max_abs_err: float
    hopper_max_abs_err: float
    mercury_beats_hopper: bool


def compare_fp8_training_accuracy(
    M: int, N: int, K: int, outlier_magnitude: float, seed: int
) -> NumericalComparisonResult:
    """Compare with FP64 on a correlated outlier stress case, using identical FP8 bytes.

    Co-located signed spikes model a few correlated high-magnitude channels.
    The ordinary channels have correlated signs and a spread of magnitudes.
    A large dot-product spike can hide aligned low-order products.
    """
    _check_dimensions(M, N, K)
    if not math.isfinite(outlier_magnitude) or outlier_magnitude < 0:
        raise ValueError("outlier_magnitude must be finite and nonnegative")
    rng = random.Random(seed)
    channel_sign = [rng.choice((-1, 1)) for _ in range(K)]
    a = [channel_sign[k % K] * rng.lognormvariate(-0.5, 0.7) for k in range(M * K)]
    b = [channel_sign[k] * rng.lognormvariate(-0.5, 0.7) for k in range(K) for _ in range(N)]
    if outlier_magnitude:
        for start in range(0, K, 32):
            k = rng.randrange(start, min(start + 32, K))
            for i in range(M):
                a[i * K + k] = channel_sign[k] * outlier_magnitude
            for j in range(N):
                b[k * N + j] = channel_sign[k] * outlier_magnitude
    aq, sa = block_quantize_1x128(a, M, K)
    bq, sb = block_quantize_128x128(b, K, N)
    mercury = mma_fp8_blk_mercury(aq, sa, bq, sb, M, N, K)
    hopper = mma_fp8_hopper_truncated(aq, sa, bq, sb, M, N, K)
    golden = [
        math.fsum(a[i * K + k] * b[k * N + j] for k in range(K))
        for i in range(M) for j in range(N)
    ]
    norm = math.hypot(*golden)
    errors = [[abs(value - ideal) for value, ideal in zip(path, golden)] for path in (mercury, hopper)]
    m_rel, h_rel = [math.hypot(*error) / norm if norm else math.hypot(*error) for error in errors]
    return NumericalComparisonResult(m_rel, h_rel, max(errors[0]), max(errors[1]), m_rel < h_rel)


def prune_to_2_4_mask(
    matrix: Sequence[float], rows: int, cols: int,
) -> tuple[list[float], list[int]]:
    """Compress B[K,N] along K. Return values and local indices encoding the mask.

    Layout is [group of four rows, output column, kept lane (two)].
    Magnitude ties keep the lower lane. Retained lanes are emitted in K order.
    """
    _check_dimensions(rows, cols)
    _check_length(matrix, rows * cols, "matrix")
    if rows % 4 or any(not math.isfinite(x) for x in matrix):
        raise ValueError("2:4 pruning requires finite values and K divisible by four")
    values, indices = [], []
    for k in range(0, rows, 4):
        for j in range(cols):
            group = [matrix[(k + lane) * cols + j] for lane in range(4)]
            kept = sorted(sorted(range(4), key=lambda lane: (-abs(group[lane]), lane))[:2])
            values.extend(group[lane] for lane in kept)
            indices.extend(kept)
    return values, indices


def execute_sparse_2_4_mma(
    A_dense: Sequence[float], B_24_values: Sequence[float], B_24_indices: Sequence[int],
    M: int, N: int, K: int,
) -> list[float]:
    """Multiply row-major A by 2:4-compressed B using FP64 reference arithmetic."""
    _check_dimensions(M, N, K)
    if K % 4:
        raise ValueError("2:4 sparse MMA requires K divisible by four")
    _check_length(A_dense, M * K, "A_dense")
    _check_length(B_24_values, K * N // 2, "B_24_values")
    _check_length(B_24_indices, K * N // 2, "B_24_indices")
    columns: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    for group in range(K // 4):
        for j in range(N):
            offset = (group * N + j) * 2
            first, second = B_24_indices[offset:offset + 2]
            if any(type(lane) is not int or not 0 <= lane < 4 for lane in (first, second)) or first == second:
                raise ValueError("Every 2:4 group must name two distinct lanes in [0, 3]")
            columns[j].extend((group * 4 + B_24_indices[p], B_24_values[p]) for p in range(offset, offset + 2))
    return [math.fsum(A_dense[i * K + k] * value for k, value in column) for i in range(M) for column in columns]


class TileCapacityError(OverflowError):
    """The schedule's simultaneous allocations exceed a hardware budget."""


@dataclass(frozen=True)
class SCCScheduleResult:
    kernel: str
    tile_m: int
    tile_n: int
    rf_used_bytes: int
    l1_used_bytes: int
    shared_sram_bytes: int
    gddr7_bytes_transferred: int
    naive_gddr7_bytes_transferred: int
    gddr7_traffic_reduction: float
    bank_conflict_penalty: float
    achieved_ai_gddr7: float
    effective_mfu: float


def simulate_scc_kernel_schedule(
    kernel: str, tile_m: int = 64, tile_n: int = 128, *,
    sku: MercurySKU = MERCURY_1D, reuse_tiles: int = 16, kv_tiles: int = 8,
    bank_stride: int = 1, clock_ghz: float | None = None,
) -> SCCScheduleResult:
    """Model a per-die wave of query or token tiles sharing stationary operands.

    Byte totals are logical payloads for the modeled wave. The baseline
    has the same weight/KV reuse but spills intermediate tensors to GDDR7.
    MFU is an optimistic tensor roofline for participating SCCs against
    their nominal FP8 peak, with perfect overlap and no launch, vector,
    or barrier time. It is not measured whole-model utilization.
    """
    _check_dimensions(tile_m, tile_n, reuse_tiles, kv_tiles, bank_stride)
    if reuse_tiles > sku.active_sccs_per_die:
        raise ValueError("One wave cannot use more SCCs than the active die")
    clock = sku.nominal_clock_ghz if clock_ghz is None else clock_ghz
    if not 0 < clock <= sku.nominal_clock_ghz:
        raise ValueError("clock_ghz must be positive and at most the SKU nominal clock")

    # Compiler allocation per resident warp: 12 GPRs and 4 FPRs per
    # lane, plus eight vector registers with 32 64-bit elements each.
    resident_warps, lanes = 16, 32
    registers = resident_warps * lanes * (12 + 4 + 8) * 8
    m, n = tile_m, tile_n
    activation = m * BLOCK + 4 * m  # FP8 input, one FP32 scale per row
    weights = BLOCK * n + 4 * _ceil_div(n, BLOCK)
    kernel = kernel.casefold().replace("-", "_").replace(" ", "_")
    if kernel in ("flashattention_3", "flashattention3"):
        kernel = "flashattention_3"
        # One projection group, then a fully visible shared KV prefix.
        # V has BF16 elements. Q8 and online softmax state stay on-chip.
        kv = n * BLOCK * (1 + 2) + 4 * _ceil_div(n, BLOCK)
        rf_used = registers + max(4 * m * BLOCK, 4 * m * (n + BLOCK) + 8 * m)
        l1_used = max(2 * (activation + BLOCK**2 + 4), activation + 2 * kv)
        shared_sram = BLOCK**2 + 4 + kv * kv_tiles
        per_tile = activation + 2 * m * BLOCK + 4 * m  # X, final O, LSE
        spills = 2 * (activation + m * n * kv_tiles * (4 + 2))  # Q8, scores, probabilities
        fp8_flops = 2 * m * BLOCK**2 + 2 * m * n * BLOCK * kv_tiles
        bf16_flops = 2 * m * n * BLOCK * kv_tiles
    elif kernel in ("swiglu_mlp", "swiglu"):
        kernel = "swiglu_mlp"
        # Two FP8 projections and a BF16 down step with width BLOCK.
        down_weights = 2 * n * BLOCK
        hidden = 2 * m * n
        rf_used = registers + max(2 * 4 * m * n, 4 * m * BLOCK)
        l1_used = max(2 * (activation + 2 * weights), hidden + 2 * down_weights)
        shared_sram = 2 * weights + down_weights
        per_tile = activation + 2 * m * BLOCK  # input and output
        spills = 2 * 3 * hidden  # materialized up, gate, and SwiGLU tensors
        fp8_flops = 4 * m * n * BLOCK
        bf16_flops = 2 * m * n * BLOCK
    else:
        raise ValueError(f"Unsupported SCC kernel: {kernel}")

    budgets = (
        ("RF", rf_used, sku.rf_kib_per_scc * 1024),
        ("L1", l1_used, sku.l1_kib_per_scc * 1024),
        ("Shared SRAM", shared_sram, int(sku.shared_sram_mb_per_die * 1_000_000)),
    )
    for name, used, limit in budgets:
        if used > limit:
            raise TileCapacityError(f"{kernel} {name} overflow: {used} bytes exceeds {limit} bytes")

    transfers = shared_sram + reuse_tiles * per_tile
    naive_transfers = transfers + reuse_tiles * spills
    banks = sku.shared_sram_banks_per_die
    bank_loads = [0] * banks
    for worker in range(reuse_tiles):
        bank_loads[worker * bank_stride % banks] += 1
    # Model one simultaneous cache-line request per worker, then rotate
    # these bank offsets uniformly across subsequent cache lines.
    penalty = max(bank_loads) / _ceil_div(reuse_tiles, banks)
    gddr7_bytes_per_ns = sku.vram_bandwidth_gbs / sku.num_dies
    shared_bytes_per_ns = banks * sku.shared_sram_bytes_per_cycle_per_bank * clock
    fp8_flops_per_cycle = sku.bf16_macs_per_scc * 4
    flops = reuse_tiles * (fp8_flops + bf16_flops)
    compute_ns = (fp8_flops + 2 * bf16_flops) / (fp8_flops_per_cycle * clock)
    shared_ns = shared_sram * (1 + reuse_tiles * penalty) / shared_bytes_per_ns
    wave_ns = max(compute_ns, transfers / gddr7_bytes_per_ns, shared_ns)
    nominal_peak = reuse_tiles * fp8_flops_per_cycle * sku.nominal_clock_ghz
    return SCCScheduleResult(
        kernel, m, n, rf_used, l1_used, shared_sram, transfers, naive_transfers,
        naive_transfers / transfers, penalty, flops / transfers, flops / (wave_ns * nominal_peak),
    )


def run_compute_isa_diagnostics(sku: MercurySKU = MERCURY_1D) -> dict[str, object]:
    """Return JSON-serializable numeric, sparse, and tile-model diagnostics."""
    numerical = {
        str(k): asdict(compare_fp8_training_accuracy(4, 8, k, 32.0, 7))
        for k in (1024, 4096)
    }
    values, indices = prune_to_2_4_mask([1.0, -4.0, 2.0, 8.0], 4, 1)
    sparse = execute_sparse_2_4_mma([2.0, 3.0, 4.0, 5.0], values, indices, 1, 1, 4)
    kernels = ("flashattention_3", "swiglu_mlp")
    return {
        "numerical": numerical,
        "sparsity": {"dense_parity": sparse == [28.0], "sparse_result": sparse},
        "schedules": {kernel: asdict(simulate_scc_kernel_schedule(kernel, sku=sku)) for kernel in kernels},
    }
