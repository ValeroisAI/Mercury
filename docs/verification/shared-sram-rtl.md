# Shared Tensor SRAM controller prototype

The [SystemVerilog arbiter](../../rtl/mercury_sram_arbiter.sv) maps line addresses to SRAM banks and selects one request per ready bank per cycle. It gives an accepted-request strobe with the source index and physical row. Each bank keeps its own round-robin pointer and advances it only on acceptance. Holding a request valid with a stable line address until grant is the requester's responsibility.

Run the generic elaboration check, full-width mapping proof, and bounded arbitration proof with Yosys:

```sh
python scripts/check_sram_rtl.py
```

On Windows, a project-local YoWASP environment also works:

```powershell
python -m venv build/rtl-env
build/rtl-env/Scripts/python.exe -m pip install yowasp-yosys==0.71.0.0.post1285
python scripts/check_sram_rtl.py --yosys build/rtl-env/Scripts/yowasp-yosys.exe
```

The default RTL has eight requesters, 64 banks and 22-bit line addresses. The bank mapping follows the Python [cycle model](../../mercury_sim/microarch_cycle.py). For 64 banks, XOR the low six bits with the corresponding bits after shifts of six and thirteen. The physical row preserves the high address bits, so a specified row and bank uniquely identify a logical line. SWIZZLE is a build-time parameter.

The [formal harness](../../tests/rtl/sram_arbiter_tb.sv) proves the swizzled and linear mappings reversible for arbitrary 22-bit addresses with 64 banks. The separate arbitration harness checks four requesters and four banks across seven clock steps from reset. It checks destination selection, address preservation, issue and grant correspondence, and round-robin progression for a continuously contended bank with arbitrary backpressure.

This is an RTL prototype for one part of the SCC memory path. Request decode, arbitration, and row selection share a combinational path. SRAM macros, response and write-data networks, ECC, pipeline partitioning, physical timing closure, and accelerator-level integration remain open. Generic elaboration provides structural evidence and leaves PPA unmeasured.
