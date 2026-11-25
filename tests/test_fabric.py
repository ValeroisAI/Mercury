"""Test topology, explicit FSDP traffic and synchronization timing."""

from dataclasses import replace
import json

import pytest

from mercury_sim.fabric import FabricConfig, FSDPWorkload, run_fabric_sensitivity, simulate_fsdp_fabric


def test_single_card_requires_no_collective():
    config = FabricConfig(cards=1, live_ports_per_card=(0,))
    work = FSDPWorkload(compute_step_s=17)
    result = simulate_fsdp_fabric(config, work)
    assert result.reachable
    assert result.step_time_s == 17
    assert result.required_tx_bytes_per_card == result.collective_rounds == 0
    assert result.exposed_communication_s == 0


def test_explicit_two_card_one_layer_schedule_has_no_overlap():
    config = FabricConfig(cards=2, payload_efficiency=1, local_round_latency_us=0)
    work = FSDPWorkload(parameters=800_000_000, layers=1, microbatches=2, compute_step_s=1)
    result = simulate_fsdp_fabric(config, work)
    assert result.required_tx_bytes_per_card == 3.2e9
    assert result.required_tx_bytes_across_leaf_uplinks == 0
    assert result.collective_rounds == 6
    assert result.serialized_collective_time_s == pytest.approx(0.016)
    assert result.step_time_s == pytest.approx(1.016)


def test_topology_and_single_port_loss_have_different_bottlenecks():
    config = FabricConfig(cards=64, oversubscription=2)
    local = simulate_fsdp_fabric(config)
    striped_config = replace(config, ring_order="striped")
    striped = simulate_fsdp_fabric(striped_config)
    ports = (3,) + (4,) * 63
    degraded_local = simulate_fsdp_fabric(replace(config, live_ports_per_card=ports))
    degraded_striped = simulate_fsdp_fabric(replace(striped_config, live_ports_per_card=ports))

    assert local.leaf_count == striped.leaf_count == 8
    assert local.required_tx_bytes_per_card == striped.required_tx_bytes_per_card
    assert local.busiest_leaf_ring_edges == 1
    assert striped.busiest_leaf_ring_edges == 8
    assert striped.required_tx_bytes_across_leaf_uplinks == 8 * local.required_tx_bytes_across_leaf_uplinks
    assert striped.exposed_communication_s > local.exposed_communication_s
    assert degraded_local.exposed_communication_s > local.exposed_communication_s
    assert degraded_striped.exposed_communication_s == striped.exposed_communication_s


def test_backward_reductions_and_final_gather_have_exposed_tail():
    config = FabricConfig(cards=2, payload_efficiency=1, local_round_latency_us=0)
    work = FSDPWorkload(parameters=800_000_000, layers=2, microbatches=1, compute_step_s=1)
    result = simulate_fsdp_fabric(config, work)
    # Each gather is 1 ms and each reduction is 2 ms. The first
    # gather of each direction and the last reduction are exposed.
    assert result.exposed_communication_s == pytest.approx(0.004)
    assert result.exposed_communication_s < result.serialized_collective_time_s


def test_slow_compute_is_available_to_hide_traffic():
    config = FabricConfig(cards=64, oversubscription=2, ring_order="striped")
    fast = simulate_fsdp_fabric(config, FSDPWorkload(compute_step_s=15))
    slow = simulate_fsdp_fabric(config, FSDPWorkload(compute_step_s=110))
    assert slow.serialized_collective_time_s == fast.serialized_collective_time_s
    assert slow.exposed_communication_s < fast.exposed_communication_s
    assert fast.step_time_s >= fast.serialized_collective_time_s


def test_retaining_parameters_trades_backward_gather_for_memory():
    config = FabricConfig(cards=8)
    work = FSDPWorkload(param_bytes=2, gradient_bytes=2)
    resharded = simulate_fsdp_fabric(config, work)
    retained = simulate_fsdp_fabric(config, replace(work, reshard_after_forward=False))
    assert retained.required_tx_bytes_per_card == pytest.approx(resharded.required_tx_bytes_per_card * 2 / 3)
    assert retained.collective_rounds * 3 == resharded.collective_rounds * 2
    assert resharded.retained_param_bytes_per_card == 0
    assert retained.retained_param_bytes_per_card == 122.5e9
    assert retained.step_time_s < resharded.step_time_s


def test_deferring_gradient_sync_reduces_traffic_but_keeps_full_gradients():
    config = FabricConfig(cards=2, payload_efficiency=1, local_round_latency_us=0)
    work = FSDPWorkload(parameters=800_000_000, layers=1, microbatches=2, compute_step_s=1)
    deferred = simulate_fsdp_fabric(config, replace(work, reduce_each_microbatch=False))
    assert deferred.required_tx_bytes_per_card == 2.4e9
    assert deferred.collective_rounds == 5
    assert deferred.retained_gradient_bytes_per_card == 0.8e9
    assert deferred.step_time_s == pytest.approx(1.012)


def test_nondivisible_leaf_placement_and_failed_port():
    config = FabricConfig(cards=3, cards_per_leaf=2, live_ports_per_card=(4, 0, 4))
    result = simulate_fsdp_fabric(config)
    assert result.leaf_count == 2
    assert not result.reachable and result.step_time_s is None
    assert "no live Ethernet ports" in result.blocking_reason
    assert result.required_tx_bytes_per_card > 0


@pytest.mark.parametrize(
    "config",
    [
        FabricConfig(cards=0), FabricConfig(cards_per_leaf=0), FabricConfig(oversubscription=.5),
        FabricConfig(payload_efficiency=float("nan")), FabricConfig(interleaf_round_latency_us=1),
        FabricConfig(live_ports_per_card=(4,)), FabricConfig(live_ports_per_card=(5,) * 8),
        FabricConfig(ring_order="unknown"),
    ],
)
def test_invalid_configs_are_rejected(config):
    with pytest.raises(ValueError):
        simulate_fsdp_fabric(config)


def test_fabric_hook_returns_json_native_data():
    result = run_fabric_sensitivity()
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["workload"]["parameters"] == 70_000_000_000
    assert result["scenarios"]["8_cards_one_disconnected"]["step_time_s"] is None
    assert result["configurations"]["64_cards_striped"]["oversubscription"] == 2
