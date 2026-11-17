"""Real World Objects: DAVIS346 looking at real objects on a table (no monitor), random25 protocol and scoring
as the other experiments (rules 1 / 2 v2 / 3 of auto_experiment.py), written or spoken (GSCD v2 -> NAS-KWS on FPGA).

12 scenarios = {2, 3, 4 (2 x 2)} objects x {jitter, circular} camera movement x {no clutter, clutter}, numbered as
"batches" 1-12 (see SCENES). Ground truth: the masks shipped with the recordings, copied to
stimuli/ground_truth_masks/rwo_*. The recordings were noise-filtered at recording time (record_camera.py,
BA filter 500 us), so the second filter pass used for the older DAVIS recordings is off (VA_NOISE_US=0).
Attention / controller parameters: unchanged (same as the DAVIS experiments).

Usage (from the repo root):
    venv/bin/python written_auto/run_rwo_random25.py                 # written, 10 trials per scenario
    venv/bin/python written_auto/run_rwo_random25.py --spoken        # GSCD-HW (board + jack, sequential)
    venv/bin/python written_auto/run_rwo_random25.py [--spoken] --metrics-only
Results: written_auto/results_rwo/{written,gscd_hw}/{01,11}_tTT_bBB.csv; tables: written_auto/tables_rwo.py
"""
import argparse, csv, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RAW = "/media/pwz/KINGSTON/NAS-KWS/Experiments/Real World Objects-20261116T144614Z-1-001/Real World Objects/raw/events"
PY = os.path.join(REPO, "venv", "bin", "python")
CORE = os.path.join(REPO, "kws_mod_live_words_new_exp.py")   # the ORIGINAL script, switched by VA_MODE

# batch -> (stem, n objects, movement, clutter)
SCENES = {}
for n in (2, 3, 4):
    for mv in ("jitter", "circular"):
        for cl in (False, True):
            SCENES[len(SCENES) + 1] = (f"{n}_objects_real_{mv}{'_clutter' if cl else ''}_346x260", n, mv, cl)


def desc(b):
    _, n, mv, cl = SCENES[b]
    return f"{n} obj/{mv}/{'clutter' if cl else 'no clutter'}"


def names(b):
    rows = csv.DictReader(open(os.path.join(REPO, "stimuli/ground_truth_masks", f"rwo_{SCENES[b][0]}.truth.csv")))
    return [r["name"] for r in sorted(rows, key=lambda r: int(r["id"]))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spoken", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--batches", default="1-12", help="subset, e.g. 1,2 or 1-12")
    ap.add_argument("--jobs", type=int, default=None, help="default: 2 written, 1 spoken (one board)")
    ap.add_argument("--p-impossible", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metrics-only", action="store_true")
    a = ap.parse_args()
    ling = "1" if a.spoken else "0"
    out = a.out or os.path.join(HERE, "results_rwo", "gscd_hw" if a.spoken else "written")
    jobs = a.jobs or (1 if a.spoken else 2)
    blist = sorted({b for p in a.batches.split(",") for b in range(int(p.split("-")[0]), int(p.split("-")[-1]) + 1)})
    os.makedirs(out, exist_ok=True)

    def run(bt):
        b, t = bt
        env = dict(os.environ, VA_TRIAL=str(t), VA_BATCH=str(b), VA_NPY=os.path.join(RAW, SCENES[b][0] + ".npy"),
                   VA_MASK=f"stimuli/ground_truth_masks/rwo_{SCENES[b][0]}.mask.npy", VA_NAMES=json.dumps(names(b)),
                   VA_RESULTS=out, VA_MODE="random25", VA_LING=ling, VA_VIS="1", VA_NOISE_US="0",
                   VA_P_IMPOSSIBLE=str(a.p_impossible), VA_INIT=str(5 + 5 * ((t - 1) % 5 + 1)),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        cmd = [PY, CORE] if a.spoken else ["nice", "-n", "19", PY, CORE]   # spoken: real time, not niced
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        open(os.path.join(out, f"log_t{t:02d}_b{b:02d}.txt"), "w").write(p.stdout + p.stderr)
        msg = [l for l in p.stdout.splitlines() if "commands correct" in l]
        return b, t, p.returncode, msg[0] if msg else "NO RESULT"

    if not a.metrics_only:
        with ThreadPoolExecutor(jobs) as ex:
            for b, t, rc, msg in ex.map(run, [(b, t) for b in blist for t in range(1, a.trials + 1)]):
                print(f"scene {b:2d} ({desc(b)}) trial {t:2d}: rc={rc} {msg}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "tables_rwo.py"), out])
