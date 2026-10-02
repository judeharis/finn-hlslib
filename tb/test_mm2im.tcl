# Jude: Created
# csim + csynth + cosim of the MM2IM kernel for one golden configuration:
#   cd tb && python3 gen_mm2im_golden.py --cfg ... --out <dir>
#   MM2IM_CFG=<dir> [MM2IM_SKIP=0|1] [MM2IM_PROJ=name] FINN_HLS_ROOT=.. vitis_hls -f test_mm2im.tcl
set cfg $::env(MM2IM_CFG)
set skip [expr {[info exists ::env(MM2IM_SKIP)] ? $::env(MM2IM_SKIP) : 1}]
set flags "-std=c++14 -I$::env(FINN_HLS_ROOT) -I$::env(FINN_HLS_ROOT)/tb -I$cfg -DMM2IM_SKIP=$skip"
set proj [expr {[info exists ::env(MM2IM_PROJ)] ? $::env(MM2IM_PROJ) : "hls-syn-mm2im"}]
open_project -reset $proj
add_files mm2im_top.cpp -cflags $flags
add_files -tb mm2im_tb.cpp -cflags $flags
set_top mm2im_top
open_solution -reset sol1
set_part {xczu3eg-sbva484-1-i}
create_clock -period 5 -name default
csim_design
csynth_design
cosim_design
exit
