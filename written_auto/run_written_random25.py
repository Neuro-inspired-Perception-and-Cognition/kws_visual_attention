"""Written commands on all 10 Set-1 batches with randomly drawn commands, ~25 % of them impossible.

Each trial: 10 commands. At every command the current attended point decides which directions are
possible (have a target object, scoring rule 2) and which are impossible; with probability P_IMPOSSIBLE
an impossible direction is drawn, otherwise a possible one (seeded per batch+trial -> reproducible).
10 trials per batch -> 100 trials: 50 without clutter (batches 1,3,5,7,9), 50 with clutter (2,4,6,8,10).

Usage (from the repo root):
    python written_auto/run_written_random25.py      # run all 100 trials + print metrics
    python written_auto/run_written_random25.py --metrics-only
Results: written_auto/results_random25/00_tTT_bBB.csv
"""
import argparse, csv, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from batches import BATCHES, DESC

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--p-impossible", type=float, default=0.25)
ap.add_argument("--jobs", type=int, default=4)   # 8 parallel jobs x 2 threads froze the PC once - keep it light
ap.add_argument("--out", default=os.path.join(HERE, "results_random25"))
ap.add_argument("--metrics-only", action="store_true")
ap.add_argument("--batches", default="1-10", help="subset, e.g. 1,2 or 1-10 (default: all)")
a = ap.parse_args()
# batch subset: "1,2" or "1-10"
BATCH_LIST = sorted({b for part in a.batches.split(",") for b in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))})
os.makedirs(a.out, exist_ok=True)
# the ORIGINAL experiment script, switched to an automated run by VA_MODE (see auto_experiment.py)
CORE = os.path.join(REPO, "kws_mod_live_words_new_exp.py")
PY = os.path.join(REPO, "venv", "bin", "python")


def run(bt):
    b, t = bt
    npy, mask, names = BATCHES[b]
    env = dict(os.environ, VA_TRIAL=str(t), VA_BATCH=str(b), VA_NPY=npy, VA_MASK=mask,
               VA_NAMES=json.dumps(names), VA_RESULTS=a.out, VA_MODE="random25", VA_LING="0", VA_VIS="0",
               VA_P_IMPOSSIBLE=str(a.p_impossible), VA_INIT=str(5 + 5 * ((t - 1) % 5 + 1)),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")  # one thread per trial process
    p = subprocess.run(["nice", "-n", "19", PY, CORE], cwd=REPO, env=env, capture_output=True, text=True)
    open(os.path.join(a.out, f"log_t{t:02d}_b{b:02d}.txt"), "w").write(p.stdout + p.stderr)
    msg = [l for l in p.stdout.splitlines() if "commands correct" in l]
    return b, t, p.returncode, msg[0] if msg else "NO RESULT"


if not a.metrics_only:
    with ThreadPoolExecutor(a.jobs) as ex:
        for b, t, rc, msg in ex.map(run, [(b, t) for b in BATCH_LIST for t in range(1, a.trials + 1)]):
            print(f"batch {b:2d} trial {t:2d}: rc={rc} {msg}", flush=True)

# ------------------------------ metrics
def fmt(x): return f"{np.mean(x):6.1f} ± {np.std(x):5.1f}" if len(x) else " " * 14

G = {g: {"C": [], "L": [], "Cp": [], "Ci": [], "imp": []} for g in ("no clutter", "clutter (bg)", "all trials")}
print(f"\n{'batch':5s} {'setup':22s} {'n':>3s}  {'compliance %':>14s}  {'lock rate %':>14s}  "
      f"{'compl. possible':>15s}  {'compl. imposs.':>15s}  {'% imposs.':>9s}")
for b in range(1, 11):
    C, L, Cp, Ci, imp = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.out, f"00_t*_b{b:02d}.csv"))):
        R = list(csv.DictReader(open(f)))
        C.append(100 * np.mean([int(r["correct"]) for r in R]))
        L.append(100 * np.mean([int(r["landed_id"]) != 0 for r in R]))
        p = [int(r["correct"]) for r in R if r["impossible"] == "0"]
        i = [int(r["correct"]) for r in R if r["impossible"] == "1"]
        if p: Cp.append(100 * np.mean(p))
        if i: Ci.append(100 * np.mean(i))
        imp.append(100 * np.mean([int(r["impossible"]) for r in R]))
    if not C:
        continue
    for g in ("clutter (bg)" if "/bg/" in DESC[b] else "no clutter", "all trials"):
        for k, v in zip(("C", "L", "Cp", "Ci", "imp"), (C, L, Cp, Ci, imp)):
            G[g][k] += v
    print(f"{b:5d} {DESC[b]:22s} {len(C):3d}  {fmt(C)}  {fmt(L)}  {fmt(Cp):>15s}  {fmt(Ci):>15s}  {np.mean(imp):8.1f}")
for g, d in G.items():
    if d["C"]:
        print(f"{'ALL':5s} {g:22s} {len(d['C']):3d}  {fmt(d['C'])}  {fmt(d['L'])}  {fmt(d['Cp']):>15s}  "
              f"{fmt(d['Ci']):>15s}  {np.mean(d['imp']):8.1f}")
