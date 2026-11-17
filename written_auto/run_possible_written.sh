#!/bin/bash
# Written commands, ALWAYS possible (VA_P_IMPOSSIBLE = 0), rule 2 v2, for synthetic (10 scenes), DAVIS on a monitor
# (10 scenes) and DAVIS on real objects (12 scenes).
# Usage: written_auto/run_possible_written.sh TRIALS NCMDS OUT_DIR
#   e.g. written_auto/run_possible_written.sh 5 10 written_auto/results_possible_t5c10   (= size of the manual baseline)
cd "$(dirname "$0")/.." || exit 1
TRIALS=${1:?trials}; export VA_NCMDS=${2:?commands per trial}; OUT=${3:?output dir}
mkdir -p $OUT
export VA_RULE=v2 VA_KWS_WRONG=all
run() {   # name runner
    echo "=== $1 start $(date '+%F %T')" | tee -a $OUT/run_all.log
    venv/bin/python written_auto/$2 --trials $TRIALS --p-impossible 0 --out $OUT/$1 > $OUT/$1.log 2>&1
    echo "=== $1 end   $(date '+%F %T') rc=$?" | tee -a $OUT/run_all.log
}
run sim_written    run_written_random25.py
run davis_written  run_davis_written_random25.py
run rwo_written    run_rwo_random25.py
