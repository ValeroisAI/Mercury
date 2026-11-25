# FSDP fabric sensitivity

Run the topology and scheduling model from the repository root:

```sh
python -c "from mercury_sim.fabric import run_fabric_sensitivity; import json; print(json.dumps(run_fabric_sensitivity(), indent=2))"
python -m pytest tests/test_fabric.py -q
```

The default experiment holds a **55 s compute budget per optimizer step** fixed across placements. It uses a 70B model with 80 equally sized layers, 16 microbatches, FP8 parameter exchange and BF16 gradient exchange. Two modeled die-level rings share the same four 400GbE ports per Mercury-1D. Each layer gathers parameters before both passes and reduce-scatters gradients after backward. PyTorch describes the choice as “this reshards parameters after forward and re-all-gathers in backward.” [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)

| Modeled case | Required TX per card | Exposed communication | Modeled step |
| --- | ---: | ---: | ---: |
| 8 cards on one leaf | 3.92 TB | 0.49% | 55.27 s |
| 64 cards, leaf-major ring, 2:1 oversubscription | 4.41 TB | 0.58% | 55.32 s |
| 64 cards, leaf-major ring, one port lost on one card | 4.41 TB | 0.76% | 55.42 s |
| 64 cards, striped ring, 2:1 oversubscription | 4.41 TB | 2.65% | 56.50 s |
| 64 cards, striped ring, one port lost on one card | 4.41 TB | 2.65% | 56.50 s |
| 8 cards, one card disconnected | 3.92 TB required | Blocked | — |

The striped ring drives eight outgoing cross-leaf transfers per full leaf per round. The leaf-major ring drives one. The striped case already saturates its modeled leaf uplink, which masks the modeled loss of one port. Full disconnection leaves the original collective blocked. Rerouting and fault recovery require a separate model.

Communication uses 90% payload efficiency, 2 µs local round latency, and 5 µs interleaf round latency. These are input assumptions. The simulator grants one layer of prefetch within each phase and serializes the network stream, so it leaves the first gather of each phase exposed. It omits packet queues, PCIe and UCIe transfers, direct-RDMA validation, host staging, congestion control, timeouts, and repartitioning. Exposed communication uses total step time as its denominator.

In the 64-card leaf-major case, deferring gradient synchronization to the last microbatch reduces modeled exposure to **0.31%** but retains an additional **137.8 GB of BF16 gradients per card** during accumulation. Holding parameters until backward instead of gathering them twice reaches **0.43%** with an additional **68.9 GB of FP8 parameters per card**. Both are logical memory increments relative to persistent shards. Neither figure includes activations, optimizer state, temporary buffers, or extra memory to overlap kernels.

See [the fabric model](../../mercury_sim/fabric.py) and [tests](../../tests/test_fabric.py) for topology, traffic and scheduling inputs. A physical 400GbE system with measured collective traces and congested-link behavior would replace the latency and payload assumptions.
