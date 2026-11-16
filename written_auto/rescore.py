"""Re-score saved random25 results (written or GSCD+HW) offline with scoring rule 2 + reachability fix.

object_near() / oracle() / mask_centroids() are taken verbatim from kws_mod_live_words_new_exp.py (via ast),
so only the rule-2 decision differs:
  old : after an unlocked command the start object is never excluded
  new : ... kept only if its centroid is >= MIN_TRAVEL - READOUT_R from the attended point (REJECTED:
        ref/fovea in the CSV are the attended point, not the fovea, so reachability cannot be judged)
  either (default): after an unlocked command, landing on the start object (if it lies in the commanded
        direction) OR on the nearest other object in that direction is OK; impossible = neither exists
Writes a copy of every CSV with re-computed expected/landed/verdict/correct/impossible
(previous values kept as *_old columns).

Usage (repo root):
    python written_auto/rescore.py IN_DIR OUT_DIR [--rule either|old|new]   (either = auto_experiment.score_step)
    python written_auto/run_written_random25.py --metrics-only --out OUT_DIR       (written, 00_*)
    python written_auto/run_gscd_hw_random25.py --metrics-only --out OUT_DIR       (spoken, 10_*)
"""
import argparse, ast, csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from batches import BATCHES

src = open(os.path.join(REPO, "kws_mod_live_words_new_exp.py")).read()
ns = {"np": np}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("mask_centroids", "object_near", "oracle"):
        exec(compile(ast.Module([node], []), "kws_mod_live_words_new_exp.py", "exec"), ns)
        continue
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in ("MIN_TRAVEL", "READOUT_R", "DOWNSAMPLE", "MASK_DS"):
        ns[node.targets[0].id] = ast.literal_eval(node.value)
mask_centroids, object_near, oracle = ns["mask_centroids"], ns["object_near"], ns["oracle"]
MIN_TRAVEL, READOUT_R, STEP = ns["MIN_TRAVEL"], ns["READOUT_R"], ns["DOWNSAMPLE"] // ns["MASK_DS"]

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--rule", default="either", choices=["old", "new", "either"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cache = {}
changed = n = 0
for f in sorted(glob.glob(os.path.join(a.inp, "[01]0_t*_b*.csv"))):
    if f.endswith("_kws.csv"):
        continue
    R = list(csv.DictReader(open(f)))
    b = int(R[0]["batch"])
    if b not in cache:
        mask = np.load(os.path.join(REPO, BATCHES[b][1]))[::STEP, ::STEP]
        cache[b] = (mask, mask_centroids(mask))
    mask, cents = cache[b]
    names = BATCHES[b][2]

    def label(oid):   # same text as write_results() in kws_mod_live_words_new_exp.py
        return "background" if not oid else f"object {oid} ({names[oid - 1]})"

    for r in R:
        snap = float(r["snap_px"])
        ref = (float(r["ref_x"]), float(r["ref_y"])); fov = (float(r["fovea_x"]), float(r["fovea_y"]))
        start = object_near(mask, cents, ref, snap)
        excl = start
        if r["prev_locked"] == "0" and start is not None:
            cx, cy = cents[start]
            if a.rule == "old" or ((cx - ref[0]) ** 2 + (cy - ref[1]) ** 2) ** 0.5 >= MIN_TRAVEL - READOUT_R:
                excl = None
        target = oracle(cents, ref, r["command"], exclude=excl)
        targets = {target} - {None}
        if a.rule == "either" and r["prev_locked"] == "0" and start is not None:
            # after an unlocked command the fovea position is ambiguous: returning onto the start object
            # (if it lies in the commanded direction) or the nearest OTHER object that way are both compliant
            targets = {oracle(cents, ref, r["command"], exclude=None), oracle(cents, ref, r["command"], exclude=start)} - {None}
            target = min(targets) if targets else None
        landed = object_near(mask, cents, fov, snap)
        if target is not None and landed in targets:
            verdict, ok = "OK", True
        elif target is None:
            ok = landed is None or landed == start
            verdict = "no object that way" if ok else "no object that way - jumped to another object"
        elif landed == start:
            verdict, ok = "FAILED - did not move", False
        else:
            verdict, ok = f"WRONG - expected {label(target)}", False
        for k in ("expected_id", "expected", "verdict", "correct", "impossible"):
            r[k + "_old"] = r[k]
        r["expected_id"] = target or 0; r["expected"] = label(target); r["verdict"] = verdict
        r["correct"] = int(ok); r["impossible"] = int(target is None)
        n += 1; changed += int(str(r["correct"]) != r["correct_old"])
    with open(os.path.join(a.out, os.path.basename(f)), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(R[0].keys())); w.writeheader(); w.writerows(R)
print(f"rule={a.rule}: {n} steps re-scored, {changed} changed 'correct' -> {a.out}")
