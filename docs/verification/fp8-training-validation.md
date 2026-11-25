# FP8 training sensitivity

Run the multi-seed experiment and print the verdict from the repository root with PyTorch installed:

```sh
python -c "
import json
import mercury_sim.pytorch_bridge as bridge
result = bridge.run_training_validation()
print(json.dumps(result['verdict'], indent=2))
"
```

The default experiment sweeps three seeds and four activation distributions across eight AdamW steps. Each run uses one synthetic Transformer block, one fixed 16-token batch, and a random target. The four paths share initialization, inputs, targets, and optimizer settings. The three FP8 paths also share the quantizer. Identical FP8 payloads are guaranteed only for identical GEMM operands.

| Path | GEMM data | Accumulation |
| --- | --- | --- |
| FP64 reference | FP64 | FP64 |
| FP8 exact-accumulation control | Block-scaled E4M3 | FP64 |
| Mercury model | Block-scaled E4M3 | FP32 group partials and binary32 boundary accumulation |
| Hopper approximation | Block-scaled E4M3 | Aligned 14-bit product truncation approximation |

All four paths stayed finite and reduced loss in each of the 12 default runs. Mercury’s step-0 gradient was closer to the FP8 exact-accumulation control than the Hopper approximation in **12 of 12** runs. FP8 quantization dominated full update drift versus FP64 in **12 of 12**, and Mercury had lower full update drift than Hopper in **7 of 12**.

The first-step gradient tests the accumulator **path** across the block. Small differences can change the FP8 bits a later operation consumes. Use the paired single-GEMM tests to isolate accumulation on identical quantized operands. The training harness uses a fast FP32 BLAS reduction order, and its strict K-order probe compares a sample GEMM to the scalar reference. The workload is synthetic and the results provide no measured accelerator or full-language-model training behavior. [Training implementation](../../mercury_sim/pytorch_bridge.py) · [Regression tests](../../tests/test_training_validation.py) · [DeepSeek-V3 report §3.5.2](https://arxiv.org/html/2412.19437#S3.SS5.SSS2)
