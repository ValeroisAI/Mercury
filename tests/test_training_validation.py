"""Multi-seed FP8 training sensitivity: quantization vs accumulator controls."""

import json
import math

import pytest
import torch

import mercury_sim.pytorch_bridge as pb
from mercury_sim.pytorch_bridge import (
    TRAINING_DISTRIBUTIONS,
    VALIDATION_PATHS,
    _train_one_path,
    run_training_validation,
    seeded_activations,
)

CFG = dict(seeds=(0, 1), steps=4, ffn=256)  # reduced config: ~2 s of compute on CPU


@pytest.fixture(scope="module")
def report():
    return run_training_validation(**CFG)


def test_report_is_json_native_and_complete(report):
    assert json.loads(json.dumps(report)) == report
    assert report["config"]["distributions"] == list(TRAINING_DISTRIBUTIONS)
    assert len(report["runs"]) == 2 * len(TRAINING_DISTRIBUTIONS)
    for run in report["runs"]:
        assert set(run["paths"]) == set(VALIDATION_PATHS)
    assert len(report["limitations"]) >= 5


def test_distributions_are_distinct_and_reproducible():
    xs = {d: seeded_activations(d, 2, 8, 256, seed=3) for d in TRAINING_DISTRIBUTIONS}
    for d, x in xs.items():
        assert torch.equal(x, seeded_activations(d, 2, 8, 256, seed=3)), d
        assert not torch.equal(x, seeded_activations(d, 2, 8, 256, seed=4)), d
        assert bool(torch.isfinite(x).all()), d
    assert xs["outlier_100"].abs().max() == 100.0
    assert xs["lognormal_outlier_24"].abs().max() == 24.0
    assert xs["gaussian"].abs().max() < 6.0
    with pytest.raises(ValueError):
        seeded_activations("uniform", 1, 1, 8, 0)


def test_all_paths_finite_and_train(report):
    v = report["verdict"]
    assert v["all_paths_finite"] and v["all_paths_reduce_loss"]
    assert v["comparable_runs"] == v["runs"]
    for run in report["runs"]:
        for name, p in run["paths"].items():
            assert p["status"] == "ok", (run["seed"], run["distribution"], name)
            assert len(p["loss_curve"]) == CFG["steps"]
            assert all(math.isfinite(x) for x in p["loss_curve"])


def test_controls_are_apples_to_apples(report):
    c = report["controls"]
    assert c["fp64_rerun_bitwise_equal"] is True and c["fp64_rerun_status"] == "ok"
    assert c["same_quantizer_across_fp8_paths"]
    assert "identical operands" in c["identical_fp8_bytes_scope"]
    assert "not bit-paired" in c["accumulator_path_sensitivity_metric"]
    assert c["mercury_order_probe_status"] == "ok"
    assert c["mercury_fast_vs_exact_order_rel_l2"] < 1e-6
    for run in report["runs"]:
        gold = run["paths"]["fp64_golden"]
        quant = run["paths"]["fp8_quant_fp64_accum"]
        assert gold["update_drift_vs_fp64"] == 0.0 and gold["first_step_grad_err_vs_fp64"] == 0.0
        assert quant["update_drift_vs_fp8_exact_accum"] == 0.0
        # Same FP8 bytes: every FP8 path starts from the same quantized loss.
        assert run["paths"]["mercury_fp8_blk"]["loss_curve"][0] == pytest.approx(quant["loss_curve"][0], rel=1e-6)


def test_quantization_dominates_but_accumulator_separates_paths(report):
    v = report["verdict"]
    assert v["quantization_dominates_end_to_end_runs"] == v["runs"]
    assert v["mercury_accum_below_hopper_runs"] == v["runs"]
    assert v["mercury_closer_to_exact_accum_trajectory_runs"] == v["runs"]
    agg = report["aggregate"]
    quant_drift = agg["fp8_quant_fp64_accum"]["update_drift_vs_fp64"]["mean"]
    merc_acc = agg["mercury_fp8_blk"]["first_step_grad_err_vs_fp8_exact_accum"]["mean"]
    hop_acc = agg["hopper_14bit"]["first_step_grad_err_vs_fp8_exact_accum"]["mean"]
    assert merc_acc < 0.01 * quant_drift          # accumulator error << quantization error
    assert hop_acc > 10 * merc_acc
    for d, s in report["per_distribution"].items():
        assert s["hopper_over_mercury_accum_grad_err"]["min"] > 1.0, d


def test_non_finite_inputs_are_reported_not_hidden():
    from mercury_sim.pytorch_bridge import MercuryTransformerBlock
    torch.manual_seed(0)
    init = MercuryTransformerBlock(128, 4, 128).double().state_dict()
    x = seeded_activations("gaussian", 1, 4, 128, 0)
    x[0, 0, 0] = float("inf")
    target = torch.zeros_like(x)
    for mode in ("fp64", "mercury_fp8_blk", "hopper_14bit"):
        out = _train_one_path(mode, init, x, target, 2, 128, 128, 4, 1e-3)
        assert out["status"] != "ok", mode
        assert out["status"].startswith(("non_finite", "error")), mode


@pytest.mark.parametrize("kwargs", [
    dict(steps=1), dict(steps=2.0), dict(distributions=("nope",)), dict(seeds=()),
    dict(seeds=(0, 0)), dict(seeds=(-1,)), dict(seeds=(1.5,)), dict(seeds=(True,)),
    dict(distributions=("gaussian", "gaussian")), dict(hidden=0), dict(ffn=-4),
    dict(hidden=250, n_heads=4), dict(batch=1.0), dict(lr=0.0), dict(lr=float("nan")),
    dict(lr=float("inf")), dict(lr=True), dict(lr="0.01"),
])
def test_argument_validation(kwargs):
    with pytest.raises(ValueError):
        run_training_validation(**kwargs)


def _clear_caches():
    pb._validation_run.cache_clear()
    pb._validation_gold_weights.cache_clear()


def test_strict_json_without_nan(report):
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text) == report


def test_caller_rng_state_is_preserved():
    _clear_caches()  # force real training work, not cache hits
    try:
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        run_training_validation(seeds=(5,), distributions=("gaussian",), steps=2, hidden=128, ffn=128)
        assert torch.equal(torch.get_rng_state(), before)
        # Lower-level helper used by every path must not consume the caller's RNG either.
        init, x, target = pb._probe_inputs(5, "gaussian", 128, 128, 4, 2, 8)
        before = torch.get_rng_state()
        pb._train_one_path("fp64", init, x, target, 2, 128, 128, 4, 1e-3)
        assert torch.equal(torch.get_rng_state(), before)
    finally:
        _clear_caches()


def test_reference_failure_is_reported_as_strict_json(monkeypatch):
    real = pb._train_one_path

    def failing_fp64(mode, *args, **kwargs):
        if mode == "fp64":
            return {"status": "non_finite_loss", "losses": [], "step": 0}
        return real(mode, *args, **kwargs)

    _clear_caches()
    monkeypatch.setattr(pb, "_train_one_path", failing_fp64)
    try:
        r = run_training_validation(seeds=(7,), distributions=("gaussian",), steps=2, hidden=128, ffn=128)
    finally:
        _clear_caches()
    json.dumps(r, allow_nan=False)
    assert r["controls"]["fp64_rerun_bitwise_equal"] is None
    assert r["controls"]["fp64_rerun_status"].startswith("reference failed")
    assert r["verdict"]["all_paths_finite"] is False
    assert r["verdict"]["comparable_runs"] == 0
    gold = r["runs"][0]["paths"]["fp64_golden"]
    assert gold["status"] == "non_finite_loss" and gold["failed_at_step"] == 0
    merc = r["runs"][0]["paths"]["mercury_fp8_blk"]
    assert merc["status"] == "ok" and "update_drift_vs_fp64" not in merc
    assert r["aggregate"]["fp64_golden"]["failed_runs"] == 1


def test_zero_initial_loss_is_undefined_not_a_crash(monkeypatch):
    real = pb._train_one_path

    def zero_loss(mode, *args, **kwargs):
        out = real(mode, *args, **kwargs)
        out["losses"] = [0.0] * len(out["losses"])  # forced: loss already zero at step 0
        return out

    _clear_caches()
    monkeypatch.setattr(pb, "_train_one_path", zero_loss)
    try:
        r = run_training_validation(seeds=(3,), distributions=("gaussian",), steps=2, hidden=128, ffn=128)
    finally:
        _clear_caches()
    json.dumps(r, allow_nan=False)
    for p in r["runs"][0]["paths"].values():
        assert p["status"] == "ok" and p["loss_reduction"] is None
    assert r["verdict"]["all_paths_reduce_loss"] is False
    assert r["verdict"]["undefined_loss_reduction_paths"] == len(VALIDATION_PATHS)
    assert "loss_reduction" not in r["aggregate"]["mercury_fp8_blk"]


def test_seed_boundaries():
    assert pb.MAX_VALIDATION_SEED + 10_007 == 2**64 - 1
    for bad in (pb.MAX_VALIDATION_SEED + 1, 2**64, -1):
        with pytest.raises(ValueError):
            run_training_validation(seeds=(bad,))
    _clear_caches()
    try:  # the largest legal seed runs end to end, including the seed + offset target generator
        r = run_training_validation(seeds=(pb.MAX_VALIDATION_SEED,), distributions=("gaussian",),
                                    steps=2, hidden=128, ffn=128)
    finally:
        _clear_caches()
    assert r["runs"][0]["seed"] == pb.MAX_VALIDATION_SEED
    assert r["verdict"]["all_paths_finite"]
    json.dumps(r, allow_nan=False)


def test_json_strict_sanitizer():
    assert pb._json_strict({"a": [1.0, float("nan"), float("inf")], "b": (2, "x")}) == {
        "a": [1.0, None, None], "b": [2, "x"]}
