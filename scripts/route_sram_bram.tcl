# Out-of-context FPGA timing for the SRAM bridge and BRAM banks.
if {[llength $argv] != 3} {
    error "usage: vivado -mode batch -source scripts/route_sram_bram.tcl -tclargs PART PERIOD_NS NEW_OUTPUT_DIRECTORY"
}
lassign $argv part period output_dir
if {![regexp {^[0-9]+(\.[0-9]+)?$} $period] || $period <= 0} {
    error "PERIOD_NS must be a positive number such as 10.0"
}
set output_dir [file normalize $output_dir]
if {[file exists $output_dir]} {
    error "Output directory already exists: $output_dir"
}
if {[llength [get_parts -quiet $part]] != 1} {
    error "Specify exactly one installed FPGA part: $part"
}

file mkdir $output_dir
set root [file dirname [file dirname [file normalize [info script]]]]
foreach name {mercury_sram_arbiter mercury_sram_payload mercury_sram_bram} {
    read_verilog -sv [file join $root rtl $name.sv]
}
synth_design -top mercury_sram_bram -part $part -mode out_of_context
create_clock -name sram_clk -period $period [get_ports clk]
opt_design
place_design
route_design

report_utilization -file [file join $output_dir utilization.rpt]
report_route_status -file [file join $output_dir route-status.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output_dir timing.rpt]
write_checkpoint [file join $output_dir routed.dcp]

set clock [get_clocks sram_clk]
set failed 0
foreach delay_type {max min} label {setup hold} {
    set path [get_timing_paths -from $clock -to $clock -delay_type $delay_type -max_paths 1 -no_report_unconstrained]
    if {[llength $path] == 0} {
        error "No constrained internal $label path. See $output_dir/timing.rpt"
    }
    set slack [get_property SLACK $path]
    puts "SRAM internal $label slack: $slack ns"
    if {$slack < 0} { set failed 1 }
}
if {$failed} { error "SRAM internal timing failed. See $output_dir/timing.rpt" }
