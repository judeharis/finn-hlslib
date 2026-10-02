#!/usr/bin/env bash
# Jude: Created MM2IMv2
# Vitis HLS csim + csynth + C/RTL cosim of the MM2IM kernel over a set of configurations,
# in parallel, one project per run (tb/hls-syn-mm2im-<name>). Collects the cosim latency per
# frame, the closed-form iteration count and the csynth estimates into
# tb/data/mm2im/cosim_results.csv (the input of gen_mm2im_golden.py --fit-cycles).
#   ./run_mm2im_cosim.sh                 # default set, JOBS=4 in parallel
#   JOBS=8 ./run_mm2im_cosim.sh name...  # only the named runs
set -u
TB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$TB")"
: "${XILINX_HLS:=/mnt/Crucial/Xilinx2024/Vitis_HLS/2024.1}"
: "${JOBS:=4}"
OUT="$TB/data/mm2im"
CSV="$OUT/cosim_results.csv"
mkdir -p "$OUT"

# name SKIP STREAM ACC K S P H W CI CO PE SIMD IDT WDT   (STREAM=1: mm2im_stream, SKIP off)
RUNS=(
  "hazard     1 0 min  2 1 0 5 5 1 1 1 1 UINT4 INT8"     # shortest same-address RMW distance
  "pointwise  1 0 min  1 1 0 4 3 4 2 2 2 INT8 INT8"
  "kS         1 0 min  3 2 1 6 6 4 3 1 4 UINT4 INT8"     # K % S != 0
  "KltS       1 0 min  2 3 0 5 4 2 2 1 1 UINT4 INT4"     # K < S
  "KeqS       1 0 min  4 4 0 3 3 3 6 3 3 INT4 INT4"      # K = S
  "pKm1       1 0 min  5 2 4 6 6 2 2 1 2 UINT8 INT8"     # P = K-1, empty bands
  "pKm1-ns    0 0 min  5 2 4 6 6 2 2 1 2 UINT8 INT8"
  "espcn8     1 0 min  6 2 2 8 8 8 3 3 4 UINT4 INT8"     # ESPCN deconv shape, small image
  "espcn8-ns  0 0 min  6 2 2 8 8 8 3 3 4 UINT4 INT8"
  "espcn8-a32 1 0 32   6 2 2 8 8 8 3 3 4 UINT4 INT8"     # INT32 accumulator, for resources
  "espcn16    1 0 min  6 2 2 16 16 8 3 3 4 UINT4 INT8"
  "espcn8-p1  1 0 min  6 2 2 8 8 8 3 1 1 UINT4 INT8"
  # streamed weights
  "hazard-st  0 1 min  2 1 0 5 5 1 1 1 1 UINT4 INT8"
  "kS-st      0 1 min  3 2 1 6 6 4 3 1 4 UINT4 INT8"
  "KltS-st    0 1 min  2 3 0 5 4 2 2 1 1 UINT4 INT4"
  "espcn8-st  0 1 min  6 2 2 8 8 8 3 3 4 UINT4 INT8"
  "espcn8-p1-st 0 1 min 6 2 2 8 8 8 3 1 1 UINT4 INT8"
)

run_one() {
  local name="$1" skip="$2" stream="$3" acc="$4"
  shift 4
  local dir="$OUT/cosim_$name"
  python3 "$TB/gen_mm2im_golden.py" --cfg "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
      --idt "${10}" --wdt "${11}" --acc "$acc" --out "$dir" > /dev/null || { echo "FAIL golden: $name"; return 1; }
  echo "$skip" > "$dir/skip"
  echo "$stream" > "$dir/stream"
  (cd "$TB" && MM2IM_CFG="$dir" MM2IM_SKIP="$skip" MM2IM_STREAM="$stream" MM2IM_PROJ="hls-syn-mm2im-$name" FINN_HLS_ROOT="$ROOT" \
      vitis_hls -f test_mm2im.tcl > "$dir/vitis_hls.log" 2>&1)
  local rpt="$TB/hls-syn-mm2im-$name/sol1/sim/report/mm2im_top_cosim.rpt"
  if grep -q "Verilog|      Pass" "$rpt" 2> /dev/null; then echo "PASS $name"; else echo "FAIL $name: see $dir/vitis_hls.log"; return 1; fi
}
export -f run_one
export TB ROOT OUT

source "$XILINX_HLS/settings64.sh" > /dev/null 2>&1
sel=("${RUNS[@]}")
if [ $# -gt 0 ]; then
  sel=()
  for r in "${RUNS[@]}"; do for n in "$@"; do [ "${r%% *}" = "$n" ] && sel+=("$r"); done; done
fi
printf '%s\n' "${sel[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one {}'

# Collect every finished run (also ones from earlier invocations).
python3 - "$TB" "$OUT" "$CSV" <<'EOF'
import csv, glob, os, re, sys
import xml.etree.ElementTree as ET
tb, out, path = sys.argv[1:]
rows = []
for d in sorted(glob.glob(os.path.join(out, "cosim_*"))):
    name = os.path.basename(d)[len("cosim_"):]
    proj = os.path.join(tb, f"hls-syn-mm2im-{name}", "sol1")
    rpt = os.path.join(proj, "sim", "report", "mm2im_top_cosim.rpt")
    xml = os.path.join(proj, "syn", "report", "csynth.xml")
    if not (os.path.exists(rpt) and os.path.exists(xml)):
        continue
    m = re.search(r"Verilog\|\s*Pass\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", open(rpt).read())
    if not m:
        continue
    g = open(os.path.join(d, "golden.h")).read()
    val = lambda k: re.search(rf"constexpr unsigned(?: long long)?\s+{k} = (\d+)", g).group(1)
    skip = int(open(os.path.join(d, "skip")).read())
    sp = os.path.join(d, "stream")
    stream = int(open(sp).read()) if os.path.exists(sp) else 0
    ta = re.search(r"using  TA = (\S+);", g).group(1)
    r = ET.parse(xml).getroot()
    res = r.find("AreaEstimates/Resources")
    rows.append(dict(
        name=name, K=val("K"), S=val("S"), P=val("P"), H=val("H"), W=val("W"), CI=val("CI"), CO=val("CO"),
        PE=val("PE"), SIMD=val("SIMD"), SKIP=skip, STREAM=stream, TA=ta,
        iterations=val("ITER_SKIP") if skip else val("ITER_NOSKIP"), cosim_latency=m.group(2),
        est_clock_ns=r.findtext("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"),
        BRAM18=res.findtext("BRAM_18K"), DSP=res.findtext("DSP"), FF=res.findtext("FF"),
        LUT=res.findtext("LUT"), URAM=res.findtext("URAM")))
with open(path, "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print(f"{len(rows)} runs -> {path}")
EOF
