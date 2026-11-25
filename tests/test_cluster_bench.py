"""End-to-end qualification tests across cluster parallelism, MoE, batch=1 decode, and TCO."""

from mercury_sim.cluster_bench import (
    simulate_1t_token_pretraining_tco,
    simulate_batch1_decode_honesty,
    simulate_llm_training_cluster,
    simulate_moe_ep8_training,
)
from mercury_sim.specs import MERCURY_1D


def test_tp2_fsdp_beats_naive_tp8_over_ethernet():
    tp8 = simulate_llm_training_cluster("tp8", MERCURY_1D, num_gpus=8, tp_degree=8, bytes_per_param=11)
    tp2 = simulate_llm_training_cluster("tp2", MERCURY_1D, num_gpus=8, tp_degree=2, bytes_per_param=11)
    assert tp2.fits_in_vram
    assert tp2.exposed_comm_time_s < tp8.exposed_comm_time_s
    assert tp2.achieved_mfu > tp8.achieved_mfu
    assert tp2.activation_headroom_gib > 40.0


def test_16x_pod_fits_full_16b_adamw_for_70b():
    res16 = simulate_llm_training_cluster("16x", MERCURY_1D, num_gpus=16, tp_degree=2, bytes_per_param=16)
    assert res16.fits_in_vram
    assert res16.static_mem_per_gpu_gib < 70.0
    assert res16.activation_headroom_gib > 65.0


def test_hot_air_35c_degrades_cluster_throughput_vs_oam_liquid():
    air = simulate_llm_training_cluster(
        "air35", MERCURY_1D, num_gpus=8, tp_degree=2, bytes_per_param=11, cooling="server_4u_air_35c"
    )
    liq = simulate_llm_training_cluster(
        "liq35", MERCURY_1D, num_gpus=8, tp_degree=2, bytes_per_param=11, cooling="oam_liquid_35c"
    )
    assert liq.steady_clock_ghz == 1.85
    assert air.steady_clock_ghz < liq.steady_clock_ghz
    assert liq.cluster_tokens_per_sec > air.cluster_tokens_per_sec


def test_moe_and_batch1_decode_honesty():
    # 141B MoE cannot fit on 8x 1D (needs ~180.6 GiB/GPU > 134.1 GiB) - verify honest rejection!
    moe_8_fail = simulate_moe_ep8_training(num_gpus=8, total_params_b=141.0)
    assert not moe_8_fail.fits_in_vram

    # Across 16x 1D (2 nodes, 2,145.8 GiB total), 141B MoE EP=8+FSDP=2 fits cleanly with >40 GiB headroom
    moe_16 = simulate_moe_ep8_training(num_gpus=16, ep_degree=8, total_params_b=141.0)
    assert moe_16.fits_in_vram
    assert moe_16.activation_headroom_gib > 40.0
    assert moe_16.achieved_mfu > 0.30

    dec = simulate_batch1_decode_honesty()
    assert 30.0 < dec.tokens_per_sec < 45.0
    assert dec.tokens_per_sec < dec.h200_ref_tok_s < dec.b200_ref_tok_s


def test_1t_token_tco_shows_substantial_capex_and_token_savings():
    tco = simulate_1t_token_pretraining_tco(num_gpus=64)
    m1d, h100, b200 = tco
    assert m1d.capex_usd < h100.capex_usd * 0.25
    assert m1d.cost_per_million_tokens_usd < h100.cost_per_million_tokens_usd
    assert m1d.cost_per_million_tokens_usd < b200.cost_per_million_tokens_usd
