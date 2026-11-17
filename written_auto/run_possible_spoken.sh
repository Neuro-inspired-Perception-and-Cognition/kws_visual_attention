#!/bin/bash
# Spoken commands (GSCD v2 test utterances -> NAS -> GNN KWS on the FPGA), ALWAYS possible (VA_P_IMPOSSIBLE = 0),
# rule 2 v2, KWS errors = not compliant, KWS_UNKNOWN_SUPPRESS = 35 (as results_v2). Synthetic (10 scenes),
# DAVIS on a monitor (10 scenes), DAVIS on real objects (12 scenes); sequential (one board + jack).
# Usage: written_auto/run_possible_spoken.sh TRIALS NCMDS OUT_DIR
#   e.g. written_auto/run_possible_spoken.sh 5 10 written_auto/results_possible_t5c10
cd "$(dirname "$0")/.." || exit 1
TRIALS=${1:?trials}; export VA_NCMDS=${2:?commands per trial}; OUT=${3:?output dir}
mkdir -p $OUT
export VA_RULE=v2 VA_KWS_WRONG=all KWS_UNKNOWN_SUPPRESS=35
run() {   # name runner [extra args]
    echo "=== $1 start $(date '+%F %T')" | tee -a $OUT/run_all.log
    venv/bin/python written_auto/$2 --trials $TRIALS --p-impossible 0 --out $OUT/$1 "${@:3}" > $OUT/$1.log 2>&1
    echo "=== $1 end   $(date '+%F %T') rc=$?" | tee -a $OUT/run_all.log
}
run sim_gscd_hw    run_gscd_hw_random25.py
run davis_gscd_hw  run_davis_gscd_hw_random25.py
run rwo_gscd_hw    run_rwo_random25.py --spoken
