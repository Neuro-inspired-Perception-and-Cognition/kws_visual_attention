#!/bin/bash
# Overnight run of all four random25 experiments with scoring rule 2 = v2 (see auto_experiment.py), sequentially:
# CPU-only written runs first, then the two spoken runs (board + jack, one at a time, nothing else running).
# Results: written_auto/results_v2/<experiment>/ ; logs next to them.
cd "$(dirname "$0")/.." || exit 1
OUT=written_auto/results_v2
mkdir -p $OUT
export VA_RULE=v2 VA_KWS_WRONG=all KWS_UNKNOWN_SUPPRESS=35   # raised from 25 (user + original agent sweep: 0.78 kw recall, 91 % non-kw unknown)
run() {   # name runner
    echo "=== $1 start $(date '+%F %T')" | tee -a $OUT/run_all.log
    venv/bin/python written_auto/$2 --trials 10 --out $OUT/$1 > $OUT/$1.log 2>&1
    echo "=== $1 end   $(date '+%F %T') rc=$?" | tee -a $OUT/run_all.log
}
run sim_written    run_written_random25.py
run davis_written  run_davis_written_random25.py
run sim_gscd_hw    run_gscd_hw_random25.py
run davis_gscd_hw  run_davis_gscd_hw_random25.py
