"""
Loads a scene mask (saved when a clip is rendered) and one or more run logs,
joins them, writes a results CSV per log.

Reads each run as a trajectory of object IDs: one line per typed command,
showing which object the fovea ended up on. A command that failed to move the
fovea off its current object is reported as "FAILED - did not move".

To run it:
    python -m evaluation.results_iterations                                  # defaults below
    python -m evaluation.results_iterations --log runs/run1.csv
    python -m evaluation.results_iterations --log runs/*.csv                 # several at once
    python -m evaluation.results_iterations --mask other.mask.npy --run-ds 4

Each log gets its own results file next to it (<log stem>.results.csv) unless
--out is given for a single log. A summary table is printed at the end.
"""

import argparse
import csv
import os

import numpy as np

from stimuli.ground_truth_helpers import centroids, score_trajectory

# identical circles, identified by position (matches render_stimuli_circles.py)
NAMES = ["top-left", "top-mid", "top-right", "bottom-left", "bottom-mid", "bottom-right"]

DEFAULT_MASK = "6_circles_tex_346x260.mask.npy"
DEFAULT_LOG = "run_log.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Score run logs against a scene mask.")
    p.add_argument("--mask", default=DEFAULT_MASK,
                   help="scene mask .npy saved by the generator")
    p.add_argument("--log", nargs="+", default=[DEFAULT_LOG],
                   help="one or more run_log CSVs")
    p.add_argument("--out", default=None,
                   help="output CSV (single log only; otherwise auto-named per log)")
    p.add_argument("--mask-ds", type=int, default=2,
                   help="PROC_DOWNSAMPLE used when the mask was saved")
    p.add_argument("--run-ds", type=int, default=2,
                   help="DOWNSAMPLE used in the controller for these runs")
    p.add_argument("--snap", type=float, default=None,
                   help="px tolerance for 'on an object' (default 1.5x object radius)")
    return p.parse_args()


def load_trials(path):
    with open(path) as f:
        return [{"direction": r["direction"], "k": int(r["k"]),
                 "ref":   (float(r["ref_x"]),   float(r["ref_y"])),
                 "fovea": (float(r["fovea_x"]), float(r["fovea_y"]))}
                for r in csv.DictReader(f)]


def main():
    a = parse_args()
    if a.run_ds % a.mask_ds:
        raise SystemExit(f"--run-ds ({a.run_ds}) must be a multiple of "
                         f"--mask-ds ({a.mask_ds})")
    step = a.run_ds // a.mask_ds

    mask = np.load(a.mask)[::step, ::step]
    truth = centroids(mask)
    print(f"mask : {a.mask}  ->  grid {mask.shape[1]}x{mask.shape[0]}  "
          f"({len(truth)} objects)\n")

    summary = []
    for i, log in enumerate(a.log):
        if not os.path.exists(log):
            print(f"!! missing: {log}")
            continue
        trials = load_trials(log)
        if not trials:
            print(f"!! empty: {log}")
            continue

        out = a.out if (a.out and len(a.log) == 1) else \
            os.path.splitext(log)[0] + ".results.csv"
        print(f"=== {log}  ({len(trials)} commands) " + "=" * max(0, 30 - len(log)))
        correct, n = score_trajectory(mask, truth, trials, names=NAMES,
                                      out_path=out, snap=a.snap)
        summary.append((log, correct, n))
        if i < len(a.log) - 1:
            print()

    if len(summary) > 1:
        print("\n=== summary ===")
        tot_c = tot_n = 0
        for log, c, n in summary:
            print(f"  {os.path.basename(log):<28} {c:>3}/{n:<3}  {100*c/n:5.1f}%")
            tot_c += c; tot_n += n
        print(f"  {'ALL':<28} {tot_c:>3}/{tot_n:<3}  {100*tot_c/tot_n:5.1f}%")


if __name__ == "__main__":
    main()