#!/usr/bin/env bash
# Jude: Created MM2IMv2
# Fast C simulation sweep of the MM2IM kernel with g++ (no Vitis HLS run needed):
# generates each configuration's golden, compiles tb + top, runs 2 frames, with ROM
# weights and SKIP on and off, and with streamed weights (mm2im_stream). Prints one
# PASS/FAIL line per run; exit code = number of failures.
#   ./run_mm2im_csim.sh                  # default sweep
#   XILINX_HLS=/path/to/Vitis_HLS/2024.1 ./run_mm2im_csim.sh
set -u
TB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$TB")"
: "${XILINX_HLS:=/mnt/Crucial/Xilinx2024/Vitis_HLS/2024.1}"
OUT="$TB/data/mm2im"
mkdir -p "$OUT"

# K S P H W CI CO PE SIMD IDT WDT
CONFIGS=(
  "2 1 0 5 5 1 1 1 1 UINT4 INT8"     # RMW hazard config: shortest same-address distance
  "1 1 0 4 3 4 2 2 2 INT8 INT8"      # pointwise
  "4 2 1 8 8 4 3 3 2 UINT4 INT8"     # rev2d's subbyte test geometry
  "4 2 1 6 8 4 4 2 2 INT4 INT8"      # non-square input
  "3 2 1 6 6 4 3 1 4 UINT4 INT8"     # K % S != 0 (rev2d cannot)
  "3 1 1 5 5 2 3 3 2 INT8 INT8"      # S = 1
  "2 3 0 5 4 2 2 1 1 UINT4 INT4"     # K < S (rev2d cannot)
  "5 3 2 5 5 4 2 2 4 UINT4 INT8"     # K % S != 0, P = K-S
  "5 2 4 6 6 2 2 1 2 UINT8 INT8"     # P = K-1
  "6 2 2 8 8 8 3 3 4 UINT4 INT8"     # ESPCN deconv shape, small image, PE3/SIMD4
  "6 2 2 8 8 8 3 1 1 UINT4 INT8"     # same, PE1/SIMD1
  "4 4 0 3 3 3 6 3 3 INT4 INT4"      # K = S (no overlap)
)

fails=0
run_one() {
  local cfg="$1" skip="$2" stream="$3"
  set -- $cfg
  local dir
  dir=$(python3 "$TB/gen_mm2im_golden.py" --cfg "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
        --idt "${10}" --wdt "${11}") || { echo "FAIL golden generation for: $cfg"; return 1; }
  local bin="$dir/csim_skip${skip}_stream$stream"
  g++ -std=c++14 -O2 -w -I"$XILINX_HLS/include" -I"$ROOT" -I"$TB" -I"$dir" -DMM2IM_SKIP="$skip" -DMM2IM_STREAM="$stream" \
      "$TB/mm2im_tb.cpp" "$TB/mm2im_top.cpp" -o "$bin" 2> "$bin.log" \
      || { echo "FAIL compile ($cfg, SKIP=$skip, STREAM=$stream): see $bin.log"; return 1; }
  "$bin"
}
for cfg in "${CONFIGS[@]}"; do
  for mode in "1 0" "0 0" "0 1"; do
    run_one "$cfg" $mode || fails=$((fails+1))
  done
done
echo "csim sweep: $(( ${#CONFIGS[@]} * 3 - fails )) / $(( ${#CONFIGS[@]} * 3 )) passed"
exit $fails
