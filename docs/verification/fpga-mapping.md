# FPGA primitive mapping

Run out-of-context mapping for the RTL prototypes from the repository root:

```sh
python scripts/map_rtl_fpga.py
```

Pass `--block mercury_sram_arbiter` to map only the default eight-requester,
64-bank arbiter. Pass `--yosys` to use a specific executable. See the
[Windows setup](./shared-sram-rtl.md).

Yosys 0.71 used `synth_xilinx -family xcup -flatten -noiopad -noclkbuf`,
then `check -assert` and `stat -json`. The rows are independent top-level
builds, some of which share RTL. LUT counts sum the reported LUT primitives.
[Yosys synthesis command reference](https://yosyshq.readthedocs.io/projects/yosys/en/latest/cmd/synth_xilinx.html)

| Top module | LUT primitives | Flip-flops | DSP primitives |
| --- | ---: | ---: | ---: |
| `mercury_fp8_product` | 102 | 0 | 0 |
| `mercury_fp32_add_rne` | 572 | 0 | 0 |
| `mercury_fp32_add_pipe` | 569 | 186 | 0 |
| `mercury_stme_interleaved_sum` | 912 | 776 | 0 |
| `mercury_stme_boundary_accumulator` | 1,696 | 65 | 2 |
| `mercury_stme_fp8_dot` | 2,297 | 172 | 2 |
| `mercury_stme_scaled_fp8_dot` | 3,095 | 172 | 4 |
| `mercury_dma_tokens` | 63 | 20 | 0 |
| `mercury_sram_arbiter` | 7,567 | 192 | 0 |

The original arbiter's default configuration hit the YoWASP call-stack
limit during FPGA mapping. The [parallel selector](./shared-sram-rtl.md)
completed on the same mapper. These counts describe the UltraScale+ family
mapping, independent of device choice, speed grade, placement, route,
timing, and power. The separate-scale scalar dot adds a binary32 rounded
product before the fused accumulation. The arbiter needs the SRAM macros
and payload path.
