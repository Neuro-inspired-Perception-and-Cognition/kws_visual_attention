#!/bin/bash
# Written commands, ALWAYS possible (VA_P_IMPOSSIBLE = 0: every command has a target object), 5 commands per trial,
# 10 trials per scene, rule 2 v2. Synthetic (10 scenes), DAVIS on a monitor (10 scenes), DAVIS on real objects (12 scenes).
# Results: written_auto/results_possible5/<experiment>/ ; logs next to them.
cd "$(dirname "$0")/.." || exit 1
OUT=written_auto/results_possible5
mkdir -p $OUT
export VA_RULE=v2 VA_KWS_WRONG=all VA_NCMDS=5
run() {   # name runner [extra args]
    echo "=== $1 start $(date '+%F %T')" | tee -a $OUT/run_all.log
    venv/bin/python written_auto/$2 --trials 10 --p-impossible 0 --out $OUT/$1 "${@:3}" > $OUT/$1.log 2>&1
    echo "=== $1 end   $(date '+%F %T') rc=$?" | tee -a $OUT/run_all.log
}
run sim_written    run_written_random25.py
run davis_written  run_davis_written_random25.py
run rwo_written    run_rwo_random25.py
