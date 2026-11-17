"""Re-run ONE trial of an automated experiment with the live window (VA_GUI=1), with the same settings as the runners,
so it reproduces the stored CSV. Results go to written_auto/_check/ (never to results/), then the steps of the new
CSV are printed next to the stored one (if given).

Usage (from the repo root):
    venv/bin/python written_auto/show_trial.py SET BATCH TRIAL [--ncmds 10] [--p-impossible 0] [--compare DIR]
SET: sim (synthetic IEBCS), monitor (DAVIS on a monitor), real (DAVIS on real objects). Written commands only.
Example: venv/bin/python written_auto/show_trial.py real 12 1 --compare written_auto/results_possible_t5c10/rwo_written
"""
import argparse, csv, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from batches import BATCHES
import run_rwo_random25 as RWO

DAVIS = ("/media/pwz/KINGSTON/NAS-KWS/Experiments/Event and Grayscale data - DAVIS-20261116T144619Z-1-001/"
         "Event and Grayscale data - DAVIS/raw/events")
STEMS = {1: "4_circles_mono_nobg", 2: "4_circles_mono_bg_jitter", 3: "6_circles_mono_nobg", 4: "6_circles_mono_bg_jitter",
         5: "6_objects_mono_nobg", 6: "6_objects_mono_bg_jitter", 7: "6_objects_color_nobg", 8: "6_objects_color_bg_jitter",
         9: "9_objects_color_nobg", 10: "9_objects_color_bg_jitter"}   # as run_davis_written_random25.py

ap = argparse.ArgumentParser()
ap.add_argument("set", choices=("sim", "monitor", "real"))
ap.add_argument("batch", type=int)
ap.add_argument("trial", type=int)
ap.add_argument("--ncmds", default="10")
ap.add_argument("--p-impossible", default="0")
ap.add_argument("--compare", default=None, help="folder with the stored CSV of this trial")
a = ap.parse_args()
b, t = a.batch, a.trial
out = os.path.join(HERE, "_check")
env = dict(os.environ, VA_MODE="random25", VA_LING="0", VA_TRIAL=str(t), VA_BATCH=str(b), VA_GUI="1",
           VA_RESULTS=out, VA_P_IMPOSSIBLE=a.p_impossible, VA_NCMDS=a.ncmds, VA_INIT=str(5 + 5 * ((t - 1) % 5 + 1)),
           VA_RULE="v2", VA_KWS_WRONG="all", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
if a.set == "sim":
    npy, mask, names = BATCHES[b]
    env.update(VA_VIS="0", VA_NPY=npy, VA_MASK=mask, VA_NAMES=json.dumps(names))
    pre = "00"
elif a.set == "monitor":
    env.update(VA_VIS="1", VA_NPY=os.path.join(DAVIS, STEMS[b] + "_346x260.npy"),
               VA_MASK=f"stimuli/ground_truth_masks/davis_{STEMS[b]}_346x260.mask.npy",
               VA_NAMES=json.dumps(BATCHES[b][2]))
    pre = "01"
else:
    env.update(VA_VIS="1", VA_NOISE_US="0", VA_NPY=os.path.join(RWO.RAW, RWO.SCENES[b][0] + ".npy"),
               VA_MASK=f"stimuli/ground_truth_masks/rwo_{RWO.SCENES[b][0]}.mask.npy", VA_NAMES=json.dumps(RWO.names(b)))
    pre = "01"
print(f"{a.set} batch {b} trial {t}: {env['VA_NPY']}")
subprocess.run([os.path.join(REPO, "venv", "bin", "python"), "kws_mod_live_words_new_exp.py"], cwd=REPO, env=env)

name = f"{pre}_t{t:02d}_b{b:02d}.csv"
for d in [out] + ([a.compare] if a.compare else []):
    f = os.path.join(d, name)
    if os.path.exists(f):
        print(f"\n== {f}")
        for r in csv.DictReader(open(f)):
            print(r["step"], r["command"], r["start"], "->", r["expected"], "| landed", r["landed_id"], "|",
                  r["verdict"], r["correct"])
