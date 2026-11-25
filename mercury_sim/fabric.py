"""Topology and scheduling sensitivity for dual-die FSDP training.

The model runs two equal TP shards per Mercury-1D, with one FSDP ring per
die. The two rings share each card's Ethernet injection bandwidth. This
is a sensitivity model for a leaf/spine fabric with nonblocking leaves,
not a packet simulation or a forecast of measured RoCEv2 efficiency.

The default schedule reflects PyTorch FSDP2 with reshard_after_forward:
"this reshards parameters after forward and re-all-gathers in backward."
https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

from .specs import MERCURY_1D


@dataclass(frozen=True)
class FabricConfig:
    """One leaf/spine network, with all four ports available unless overridden."""

    cards: int = 8
    cards_per_leaf: int = 8
    oversubscription: float = 1.0
    payload_efficiency: float = 0.90
    local_round_latency_us: float = 2.0
    interleaf_round_latency_us: float = 5.0
    ring_order: str = "leaf_major"
    live_ports_per_card: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FSDPWorkload:
    """Even layer sizes and a compute budget covering one full optimizer step."""

    parameters: int = 70_000_000_000
    layers: int = 80
    microbatches: int = 16
    param_bytes: int = 1
    gradient_bytes: int = 2
    compute_step_s: float = 55.0
    forward_compute_fraction: float = 1 / 3
    reshard_after_forward: bool = True
    reduce_each_microbatch: bool = True


@dataclass(frozen=True)
class FabricResult:
    reachable: bool
    blocking_reason: str | None
    compute_step_s: float
    step_time_s: float | None
    exposed_communication_s: float | None
    exposed_communication_pct: float | None
    serialized_collective_time_s: float | None
    required_tx_bytes_per_card: float
    required_tx_bytes_across_leaf_uplinks: float
    collective_rounds: int
    leaf_count: int
    busiest_leaf_ring_edges: int
    payload_nic_bandwidth_gbs: float
    leaf_uplink_bandwidth_gbs: float
    retained_param_bytes_per_card: float
    retained_gradient_bytes_per_card: float


def _validate(config: FabricConfig, work: FSDPWorkload) -> tuple[int, ...]:
    dimensions = (
        config.cards, config.cards_per_leaf, work.parameters, work.layers,
        work.microbatches, work.param_bytes, work.gradient_bytes,
    )
    if any(type(n) is not int or n <= 0 for n in dimensions):
        raise ValueError("Card, layer, parameter, microbatch and element counts must be positive integers")
    if not math.isfinite(config.oversubscription) or config.oversubscription < 1:
        raise ValueError("Oversubscription must be finite and at least one")
    if not 0 < config.payload_efficiency <= 1:
        raise ValueError("Payload efficiency must be in (0, 1]")
    latencies = (config.local_round_latency_us, config.interleaf_round_latency_us)
    if any(not math.isfinite(latency) or latency < 0 for latency in latencies) or latencies[1] < latencies[0]:
        raise ValueError("Expected finite nonnegative latency with interleaf >= local latency")
    if not math.isfinite(work.compute_step_s) or work.compute_step_s <= 0 or not 0 < work.forward_compute_fraction < 1:
        raise ValueError("Expected a finite positive compute budget and forward fraction in (0, 1)")
    if type(work.reshard_after_forward) is not bool:
        raise ValueError("Expected a boolean reshard_after_forward")
    if type(work.reduce_each_microbatch) is not bool:
        raise ValueError("Expected a boolean reduce_each_microbatch")
    if config.ring_order not in ("leaf_major", "striped"):
        raise ValueError("Ring order must be 'leaf_major' or 'striped'")
    ports = config.live_ports_per_card
    if ports is None:
        return (MERCURY_1D.eth_400g_ports,) * config.cards
    if (
        not isinstance(ports, tuple)
        or len(ports) != config.cards
        or any(type(n) is not int or not 0 <= n <= MERCURY_1D.eth_400g_ports for n in ports)
    ):
        raise ValueError("Provide one valid working-port count per card")
    return ports


def _ring_cut(config: FabricConfig) -> tuple[int, int, int]:
    leaf = tuple(rank // config.cards_per_leaf for rank in range(config.cards))
    leaves = leaf[-1] + 1
    ring = list(range(config.cards))
    if config.ring_order == "striped":
        ring.sort(key=lambda rank: (rank % config.cards_per_leaf, leaf[rank]))
    cross_edges = [(leaf[a], leaf[b]) for a, b in zip(ring, ring[1:] + ring[:1]) if leaf[a] != leaf[b]]
    tx = Counter(a for a, _ in cross_edges)
    rx = Counter(b for _, b in cross_edges)
    busiest = max((*tx.values(), *rx.values()), default=0)
    return leaves, len(cross_edges), busiest


def _step_time(work: FSDPWorkload, gather_s: float, reduce_s: float) -> float:
    """Run one FIFO network with one-layer prefetch within each phase."""
    network_free = compute_free = 0.0
    for microbatch in range(work.microbatches):
        for backward in (False, True):
            ratio = 1 - work.forward_compute_fraction if backward else work.forward_compute_fraction
            layer_compute = work.compute_step_s * ratio / (work.layers * work.microbatches)
            needs_gather = not backward or work.reshard_after_forward
            gather_ready = compute_free
            if needs_gather:
                network_free = max(network_free, compute_free) + gather_s
                gather_ready = network_free
            for layer in range(work.layers):
                start = max(compute_free, gather_ready)
                if needs_gather and layer + 1 < work.layers:
                    network_free = max(network_free, start) + gather_s
                    gather_ready = network_free
                compute_free = start + layer_compute
                if backward and (work.reduce_each_microbatch or microbatch + 1 == work.microbatches):
                    network_free = max(network_free, compute_free) + reduce_s
    return max(network_free, compute_free)


def simulate_fsdp_fabric(config: FabricConfig, work: FSDPWorkload = FSDPWorkload()) -> FabricResult:
    """Estimate an optimizer step with layer-sized gathers and gradient reductions.

    Each ring round sends one equal-sized shard chunk per rank. The
    two simultaneous rings consume the same aggregate bandwidth as
    one card-sized chunk. Leaf uplinks are full duplex, so timing uses
    the busiest direction. Each phase completes in lockstep at the
    slowest NIC or leaf. These assumptions leave PCIe, UCIe, switch
    buffers, ECMP, PFC and congestion recovery outside the model.
    """
    ports = _validate(config, work)
    leaves, cross_edges, busiest = _ring_cut(config)
    port_gbs = MERCURY_1D.eth_unidir_gbs / MERCURY_1D.eth_400g_ports
    nic_gbs = min(ports) * port_gbs * config.payload_efficiency
    leaf_gbs = config.cards_per_leaf * MERCURY_1D.eth_unidir_gbs * config.payload_efficiency / config.oversubscription
    rounds_per_collective = config.cards - 1
    gathers = 2 if work.reshard_after_forward else 1
    gather_microbatches = gathers * work.microbatches
    reduce_microbatches = work.microbatches if work.reduce_each_microbatch else 1
    total_rounds = (gather_microbatches + reduce_microbatches) * work.layers * rounds_per_collective
    volume = work.parameters * (gather_microbatches * work.param_bytes + reduce_microbatches * work.gradient_bytes)
    tx_per_card = volume * rounds_per_collective / config.cards
    # Persistent shard bytes are already allocated in both alternatives.
    # This excludes all optimizer, activation and temporary buffer memory.
    retained_param_bytes = (
        0.0 if work.reshard_after_forward
        else work.parameters * work.param_bytes * rounds_per_collective / config.cards
    )
    retained_gradient_bytes = (
        0.0 if work.reduce_each_microbatch or work.microbatches == 1
        else work.parameters * work.gradient_bytes * rounds_per_collective / config.cards
    )
    blocked = rounds_per_collective > 0 and nic_gbs == 0
    serialized = step = exposed = exposed_pct = None
    if not blocked:
        if rounds_per_collective:
            seconds_per_byte = max(1 / nic_gbs, busiest / leaf_gbs) / 1e9
            latency = config.interleaf_round_latency_us if cross_edges else config.local_round_latency_us
            chunk_parameters = work.parameters / (work.layers * config.cards)
            gather = rounds_per_collective * (latency / 1e6 + chunk_parameters * work.param_bytes * seconds_per_byte)
            reduce = rounds_per_collective * (latency / 1e6 + chunk_parameters * work.gradient_bytes * seconds_per_byte)
            serialized = work.layers * (gather_microbatches * gather + reduce_microbatches * reduce)
            step = max(work.compute_step_s, _step_time(work, gather, reduce))
        else:
            serialized, step = 0.0, work.compute_step_s
        exposed = step - work.compute_step_s
        exposed_pct = 100 * exposed / step
    return FabricResult(
        not blocked, "A participating card has no live Ethernet ports" if blocked else None,
        work.compute_step_s, step, exposed, exposed_pct, serialized, tx_per_card,
        tx_per_card * cross_edges, total_rounds, leaves, busiest, nic_gbs, leaf_gbs,
        retained_param_bytes, retained_gradient_bytes,
    )


def run_fabric_sensitivity() -> dict[str, object]:
    """Return JSON-native scenarios with matched compute and placement assumptions."""
    scenarios = {
        "8_cards_one_leaf": FabricConfig(),
        "64_cards_leaf_major": FabricConfig(cards=64, oversubscription=2),
        "64_cards_leaf_major_one_port_lost": FabricConfig(
            cards=64, oversubscription=2, live_ports_per_card=(3,) + (4,) * 63,
        ),
        "64_cards_striped": FabricConfig(cards=64, oversubscription=2, ring_order="striped"),
        "64_cards_striped_one_port_lost": FabricConfig(
            cards=64, oversubscription=2, ring_order="striped",
            live_ports_per_card=(3,) + (4,) * 63,
        ),
        "8_cards_one_disconnected": FabricConfig(live_ports_per_card=(0,) + (4,) * 7),
    }
    workload = FSDPWorkload()
    return {
        "workload": asdict(workload),
        "configurations": {
            name: {**asdict(config), "live_ports_per_card": list(config.live_ports_per_card)
                   if config.live_ports_per_card is not None else None}
            for name, config in scenarios.items()
        },
        "scenarios": {name: asdict(simulate_fsdp_fabric(config, workload)) for name, config in scenarios.items()},
    }
