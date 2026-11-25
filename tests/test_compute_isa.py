"""Numerical and memory-budget checks for the Mercury-1 compute model."""

from dataclasses import replace
import json
import math
import struct
import unittest

from mercury_sim.compute_isa import (
    TileCapacityError,
    block_quantize_1x128,
    block_quantize_128x128,
    compare_fp8_training_accuracy,
    decode_fp8,
    encode_fp8,
    execute_sparse_2_4_mma,
    mma_fp8_blk_mercury,
    mma_fp8_hopper_truncated,
    prune_to_2_4_mask,
    run_compute_isa_diagnostics,
    simulate_scc_kernel_schedule,
)
from mercury_sim.specs import MERCURY_1D, MERCURY_1S


class TestFP8Codec(unittest.TestCase):
    def test_signed_zero_and_subnormals(self):
        for fmt, unit, min_normal in (("e4m3", 2.0**-9, 2.0**-6), ("e5m2", 2.0**-16, 2.0**-14)):
            with self.subTest(fmt=fmt):
                self.assertEqual(encode_fp8(-0.0, fmt), 0x80)
                self.assertEqual(math.copysign(1.0, decode_fp8(0x80, fmt)), -1.0)
                self.assertEqual(decode_fp8(0x01, fmt), unit)
                self.assertEqual(decode_fp8(0x81, fmt), -unit)
                self.assertEqual(encode_fp8(min_normal, fmt), 0x08 if fmt == "e4m3" else 0x04)
                self.assertEqual(encode_fp8(unit / 2, fmt), 0x00)
                self.assertEqual(encode_fp8(1.5 * unit, fmt), 0x02)
                self.assertEqual(encode_fp8(2.5 * unit, fmt), 0x02)

    def test_extremes_and_saturation_modes(self):
        self.assertEqual(decode_fp8(0x7E), 448.0)
        self.assertEqual(decode_fp8(0xFE), -448.0)
        self.assertEqual(encode_fp8(1e30), 0x7E)
        self.assertEqual(encode_fp8(-math.inf), 0xFE)
        self.assertEqual(encode_fp8(464.0, saturate=False), 0x7E)
        self.assertEqual(encode_fp8(math.nextafter(464, math.inf), saturate=False), 0x7F)
        self.assertEqual(decode_fp8(0x7B, "e5m2"), 57344.0)
        self.assertEqual(decode_fp8(0xFC, "e5m2"), -math.inf)
        self.assertEqual(encode_fp8(math.inf, "e5m2"), 0x7B)
        self.assertEqual(encode_fp8(61439.0, "e5m2", saturate=False), 0x7B)
        self.assertEqual(encode_fp8(61440.0, "e5m2", saturate=False), 0x7C)

    def test_nan_and_finite_byte_round_trip(self):
        for fmt in ("e4m3", "e5m2"):
            with self.subTest(fmt=fmt):
                for code in range(256):
                    value = decode_fp8(code, fmt)
                    if math.isfinite(value):
                        self.assertEqual(encode_fp8(value, fmt), code)
                    elif math.isnan(value):
                        self.assertTrue(math.isnan(decode_fp8(encode_fp8(value, fmt), fmt)))
        self.assertTrue(math.isnan(decode_fp8(0x7F)))
        self.assertTrue(math.isnan(decode_fp8(0xFF)))
        self.assertEqual(encode_fp8(-math.nan), 0xFF)
        with self.assertRaises(ValueError):
            decode_fp8(256)
        with self.assertRaises(ValueError):
            encode_fp8(1.0, "e3m4")

    def test_block_quantizers_scale_layout_and_tail(self):
        activation = [0.0] * (2 * 129)
        activation[0], activation[128], activation[129] = 448.0, -896.0, 112.0
        values, scales = block_quantize_1x128(activation, 2, 129)
        self.assertIsInstance(values, bytes)
        self.assertEqual(scales, [1.0, 2.0, 0.25, 1.0])
        self.assertEqual((values[0], values[128], values[129], values[-1]), (0x7E, 0xFE, 0x7E, 0))

        weights = [0.0] * (129**2)
        corners = (0, 128, 128 * 129, 129**2 - 1)
        for index, scale in zip(corners, (1.0, 2.0, 4.0, 8.0)):
            weights[index] = scale * 57344.0
        values, scales = block_quantize_128x128(weights, 129, 129, "e5m2")
        self.assertEqual(scales, [1.0, 2.0, 4.0, 8.0])
        self.assertEqual([values[index] for index in corners], [0x7B] * 4)

    def test_scale_is_fp32_and_nonfinite_input_is_rejected(self):
        _, scales = block_quantize_1x128([1.0], 1, 1)
        exact_fp32 = struct.unpack(">f", struct.pack(">f", 1.0 / 448))[0]
        self.assertEqual(scales, [exact_fp32])
        with self.assertRaisesRegex(ValueError, "finite inputs"):
            block_quantize_128x128([math.nan], 1, 1)
        with self.assertRaisesRegex(ValueError, "must contain"):
            block_quantize_1x128([0.0], 2, 1)


class TestBlockMMA(unittest.TestCase):
    def test_partial_groups_and_scale_indexing(self):
        one = encode_fp8(1.0)
        result = mma_fp8_blk_mercury(
            bytes([one]) * (2 * 129), [1.0, 2.0, 6.0, 7.0],
            bytes([one]) * 129**2, [1.0, 3.0, 4.0, 5.0], 2, 129, 129,
        )
        self.assertEqual(result, [136.0] * 128 + [394.0] + [796.0] * 128 + [2339.0])

    def test_k_order_rounding_and_fused_group_boundary(self):
        def fp8(data):
            return bytes(encode_fp8(value, "e5m2") for value in data)

        result = mma_fp8_blk_mercury(
            fp8([57344.0, 1.0, -57344.0]), [1.0], fp8([57344.0, 1.0, 57344.0]), [1.0],
            1, 1, 3, a_format="e5m2", b_format="e5m2",
        )
        self.assertEqual(result, [0.0])  # The middle product is rounded away.

        result = mma_fp8_blk_mercury(
            fp8([-1.0] + [0.0] * 127 + [1.0, 2.0**-12]), [1.0, 1.0 - 2.0**-23],
            fp8([1.0] * 129 + [2.0**-11]), [1.0, 1.0],
            1, 1, 130, a_format="e5m2", b_format="e5m2",
        )
        self.assertEqual(result, [-2.0**-46])  # Separately rounded multiplication yields zero.

    def test_binary32_gradual_underflow(self):
        subnormal = 2.0**-149
        values = bytes([encode_fp8(0.5), encode_fp8(0.75), encode_fp8(-0.75)])
        result = mma_fp8_blk_mercury(
            values, [subnormal] * 3, bytes([encode_fp8(1.0)]), [1.0], 3, 1, 1,
        )
        self.assertEqual(result, [0.0, subnormal, -subnormal])

    def test_hopper_alignment_discards_low_products(self):
        data = bytes([encode_fp8(448.0), encode_fp8(1.0)])
        args = (data, [1.0], data, [1.0], 1, 1, 2)
        self.assertEqual(mma_fp8_blk_mercury(*args), [200705.0])
        self.assertEqual(mma_fp8_hopper_truncated(*args), [200704.0])
        self.assertEqual(mma_fp8_hopper_truncated(*args, alignment_width=1), [200705.0])

    def test_full_k_requires_uniform_scale(self):
        x = bytes([encode_fp8(1.0)]) * 129
        self.assertEqual(
            mma_fp8_hopper_truncated(x, [1.0, 2.0], x, [2.0, 1.0], 1, 1, 129, accumulation="full"),
            [258.0],
        )
        with self.assertRaisesRegex(ValueError, "uniform scale"):
            mma_fp8_hopper_truncated(x, [1.0, 2.0], x, [1.0, 1.0], 1, 1, 129, accumulation="full")
        with self.assertRaisesRegex(ValueError, "finite positive"):
            mma_fp8_blk_mercury(x, [1.0, 0.0], x, [1.0, 1.0], 1, 1, 129)

    def test_long_k_correlated_outlier_stress_cases(self):
        for k in (1024, 4096):
            with self.subTest(k=k):
                comparison = compare_fp8_training_accuracy(4, 8, k, 32, 7)
                self.assertTrue(comparison.mercury_beats_hopper)
                self.assertLess(comparison.mercury_rel_l2_err, 1e-4)
                self.assertGreater(comparison.hopper_rel_l2_err, 8 * comparison.mercury_rel_l2_err)
                self.assertGreater(comparison.hopper_max_abs_err, comparison.mercury_max_abs_err)


class TestSparseMMA(unittest.TestCase):
    def test_sparse_matches_independent_dense_reference(self):
        weights = [1, 4, -5, 1, 2, -4, 3, 0, 0, 9, 2, -3, 2, 4, -1, 5]
        a = [0.125, 1, 1, 2, 4, 2, 3, 3, 0, -1, 0.5, 3, 2, 0, 1, -2]
        expected_pruned = [0, 4, -5, 0, 0, -4, 3, 0, 0, 9, 2, 0, 2, 0, 0, 5]
        expected = [
            math.fsum(a[8 * i + k] * expected_pruned[2 * k + j] for k in range(8))
            for i in range(2) for j in range(2)
        ]
        kept, mask = prune_to_2_4_mask(weights, 8, 2)
        self.assertEqual(mask, [1, 3, 0, 2, 1, 2, 0, 3])
        self.assertEqual(execute_sparse_2_4_mma(a, kept, mask, 2, 2, 8), expected)
        self.assertEqual(prune_to_2_4_mask([0.0] * 4, 4, 1), ([0.0, 0.0], [0, 1]))

    def test_rejects_non_2_4_metadata_and_k(self):
        for bad in ([0, 0], [0, 4], [0, -1], [0, 1, 2], [0, 1.0]):
            with self.subTest(indices=bad), self.assertRaises(ValueError):
                execute_sparse_2_4_mma([1.0] * 4, [2.0, 3.0], bad, 1, 1, 4)
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            prune_to_2_4_mask([1.0] * 3, 3, 1)


class TestSCCSchedule(unittest.TestCase):
    def test_default_tiles_match_byte_budgets_and_cut_gddr7_traffic(self):
        cases = (
            ("flashattention_3", (164352, 106760, 409636, 811044, 13664292), 570425344),
            ("swiglu_mlp", (163840, 82448, 65544, 462856, 2035720), 100663296),
        )
        for kernel, expected, flops in cases:
            with self.subTest(kernel=kernel):
                result = simulate_scc_kernel_schedule(kernel)
                actual = (
                    result.rf_used_bytes, result.l1_used_bytes, result.shared_sram_bytes,
                    result.gddr7_bytes_transferred, result.naive_gddr7_bytes_transferred,
                )
                self.assertEqual(actual, expected)
                self.assertLessEqual(result.rf_used_bytes, 262144)
                self.assertLessEqual(result.l1_used_bytes, 262144)
                self.assertGreater(result.gddr7_traffic_reduction, 2.5)
                self.assertAlmostEqual(result.achieved_ai_gddr7, flops / actual[3])
                self.assertGreater(result.effective_mfu, 0)
                self.assertLessEqual(result.effective_mfu, 1)

    def test_rf_boundary_and_l1_overflow(self):
        exact = simulate_scc_kernel_schedule("flashattention_3", 160, 126)
        self.assertEqual(exact.rf_used_bytes, 262144)
        with self.assertRaisesRegex(TileCapacityError, "RF overflow"):
            simulate_scc_kernel_schedule("flashattention_3", 161, 126)
        with self.assertRaisesRegex(TileCapacityError, "L1 overflow"):
            simulate_scc_kernel_schedule("flashattention_3", 1, 384)
        with self.assertRaisesRegex(TileCapacityError, "Shared SRAM overflow"):
            simulate_scc_kernel_schedule("swiglu_mlp", sku=replace(MERCURY_1D, shared_sram_mb_per_die=0.05))

    def test_bank_conflicts_and_lower_clock_reduce_mfu(self):
        base = simulate_scc_kernel_schedule("flashattention_3", sku=MERCURY_1S)
        conflict = simulate_scc_kernel_schedule("flashattention_3", bank_stride=64)
        slower = simulate_scc_kernel_schedule("flashattention_3", clock_ghz=1.25)
        self.assertEqual(base.bank_conflict_penalty, 1.0)
        self.assertEqual(conflict.bank_conflict_penalty, 16.0)
        self.assertEqual(base.gddr7_bytes_transferred, conflict.gddr7_bytes_transferred)
        self.assertLess(conflict.effective_mfu, base.effective_mfu)
        self.assertLess(slower.effective_mfu, base.effective_mfu)

    def test_diagnostics_contract_is_json_serializable(self):
        result = json.loads(json.dumps(run_compute_isa_diagnostics(), allow_nan=False))
        self.assertEqual(set(result["numerical"]), {"1024", "4096"})
        self.assertTrue(result["numerical"]["4096"]["mercury_beats_hopper"])
        self.assertTrue(result["sparsity"]["dense_parity"])
        self.assertEqual(set(result["schedules"]), {"flashattention_3", "swiglu_mlp"})
