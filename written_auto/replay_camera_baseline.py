"""Replay the typed command sequences of the live-camera baseline sessions (results/01{trial}{batch}.csv)
on the pre-recorded DAVIS sessions of the same scene (kws_mod_live_words_new_exp.py, VA_MODE=list).
Separates 'different recording/setup' from 'different command protocol (random25)'.
Usage (repo root): python written_auto/replay_camera_baseline.py [--jobs 2]
"""
import argparse, csv, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from batches import BATCHES
DAVIS = "/media/pwz/KINGSTON/NAS-KWS/Experiments/Event and Grayscale data - DAVIS-20261116T144619Z-1-001/Event and Grayscale data - DAVIS/raw/events"
# live-camera mask (scene) -> (our Set-1 batch, DAVIS recording stem)
SCENE = {"camera_6_circles_tex": (3, "6_circles_mono_nobg"), "camera_6_circles_mono_jitter_bg": (4, "6_circles_mono_bg_jitter"),
         "camera_6_objects_mono_nobg": (5, "6_objects_mono_nobg"), "camera_6_objects_mono_bg": (6, "6_objects_mono_bg_jitter"),
         "camera_6_objects_color_bg": (8, "6_objects_color_bg_jitter"),
         "camera_9_objects_color_nobg": (9, "9_objects_color_nobg"), "camera_9_objects_color_bg": (10, "9_objects_color_bg_jitter")}
ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--out", default=os.path.join(HERE, "results_replay_camera_baseline")); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
jobs = []
for b in range(1, 11):
    for t in range(1, 6):
        R = list(csv.DictReader(open(os.path.join(REPO, "results", f"01{t}{b}.csv"))))
        key = R[0]["mask"].replace("_346x260.mask.npy", "")
        if key not in SCENE:
            print(f"skip 01{t}{b}: scene {key} has no matching recording"); continue
        jobs.append((t, b, key, [r["command"] for r in R]))

def run(j):
    t, b, key, cmds = j; sb, stem = SCENE[key]
    env = dict(os.environ, VA_TRIAL=str(t), VA_BATCH=str(b), VA_NPY=os.path.join(DAVIS, stem + "_346x260.npy"),
               VA_MASK=f"stimuli/ground_truth_masks/davis_{stem}_346x260.mask.npy", VA_NAMES=json.dumps(BATCHES[sb][2]),
               VA_RESULTS=a.out, VA_MODE="list", VA_LING="0", VA_VIS="1", VA_CMDS=",".join(cmds), VA_INIT=str(5 + 5 * t),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run(["nice", "-n", "19", os.path.join(REPO, "venv/bin/python"), os.path.join(REPO, "kws_mod_live_words_new_exp.py")],
                       cwd=REPO, env=env, capture_output=True, text=True)
    return t, b, key, p.returncode

with ThreadPoolExecutor(a.jobs) as ex:
    done = list(ex.map(run, jobs))
def ok(r):
    if r["verdict"] == "OK": return 1
    if r["verdict"].startswith("no object that way"): return int(r["landed_id"] == "0")
    return 0
G = {}
for t, b, key, rc in done:
    clut = "_bg" in key and "nobg" not in key
    for src, f in (("baseline (live)", os.path.join(REPO, "results", f"01{t}{b}.csv")), ("replay (DAVIS rec.)", os.path.join(a.out, f"01_t{t:02d}_b{b:02d}.csv"))):
        if not os.path.exists(f): continue
        R = list(csv.DictReader(open(f)))
        k = (("clutter" if clut else "no clutter"), src)
        G.setdefault(k, ([], [])); G[k][0].append(100 * np.mean([ok(r) for r in R])); G[k][1].append(100 * np.mean([r["landed_id"] != "0" for r in R]))
for k in sorted(G):
    C, L = G[k]; print(f"{k[0]:11s} {k[1]:20s} n={len(C):2d} compliance {np.mean(C):5.1f} ± {np.std(C):4.1f}   lock {np.mean(L):5.1f} ± {np.std(L):4.1f}")
