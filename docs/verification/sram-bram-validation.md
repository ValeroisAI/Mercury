# SRAM BRAM integration

The FPGA prototype connects four requesters to four single-port block RAM
banks. Each bank stores 512 lines of 64 bits for 16 KiB total. The bridge
shares the same handshake, swizzled mapping and same-edge turnover as the
[full-size payload RTL](./sram-payload-rtl.md). Each bank registers its read
reply one cycle after issue. Reset cancels pending reads and leaves RAM data
intact. Read locations only after initializing their bytes through writes.

With Icarus Verilog and Yosys on `PATH`, run from the repository root:

```sh
python scripts/check_sram_full_load.py --case bram
python scripts/map_rtl_fpga.py --block mercury_sram_bram
```

Icarus checked parallel access to the first and last rows of each bank,
masked and zero-mask writes, simultaneous response consumption and request
acceptance, consecutive same-bank reads, and reset while reads were live.
Yosys 0.71 mapped the wrapper to four `RAMB36E2`, 1,551 LUT primitives and
288 flip-flops in an out-of-context UltraScale+ build.

## Routed FPGA timing

Run the [Vivado flow](../../scripts/route_sram_bram.tcl) with the exact FPGA
part and speed grade, a target period in nanoseconds, and a fresh report
directory. For example, in PowerShell, with `$part` set to the FPGA part:

```powershell
vivado -mode batch -source scripts/route_sram_bram.tcl -tclargs $part 10.0 build/sram-route-100mhz
```

The script writes routed timing, utilization, route status, and a design
checkpoint. It reports the worst internal setup and hold slack, and exits
with an error for missing paths or negative slack. The 10 ns period is an
example FPGA target. Board I/O delays and pin constraints belong to the
board top. Vivado placement and routing have yet to be run for this block.
[AMD non-project implementation flow](https://docs.amd.com/r/en-US/ug892-vivado-design-flows-overview/Non-Project-Mode-Tcl-Script-Example) · [AMD timing path queries](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_timing_paths)

## Hardware acceptance evidence

| Target | Required input | Evidence to record |
| --- | --- | --- |
| FPGA block timing | FPGA part, speed grade, Vivado, clock period | Fully routed checkpoint, nonnegative internal setup and hold slack |
| FPGA board | Board top with clocks, reset, package pins and I/O standards, host stimulus or on-chip BIST, physical board | Read back masked writes, exercise one-bank contention and distributed load, verify reset while reads are live, record throughput and failures |
| N4P timing and macro integration | Target PDK and characterized SRAM macro timing and physical views, macro latency and mask contract, routed top with extracted parasitics | Setup and hold across required corners at the 1.85 GHz target, along with macro integration and physical verification reports |
| Mercury board | Fabricated board, measurement equipment and board qualification fixture | Bring-up, memory-link signal integrity, power integrity and sustained thermal load results |
