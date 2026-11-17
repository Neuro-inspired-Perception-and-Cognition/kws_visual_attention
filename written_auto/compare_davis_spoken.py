"""Side-by-side metrics of DAVIS + written (exp. 3) and DAVIS + GSCD-HW (exp. 4), random25 protocol.

Per batch and per group (no clutter / clutter / all): compliance, lock rate, compliance for possible / impossible
commands (mean ± population std over trials, as in the runners) and KWS accuracy. For the spoken run, compliance
is also split by what the KWS did with the step: heard correctly / not recognised (unknown or none) / wrong
direction. The split pools steps (not trials), because a trial may have no step of a given kind.

Usage (from the repo root):
    python3 written_auto/compare_davis_spoken.py [--written DIR] [--spoken DIR]
"""
import argparse, csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from batches import DESC

ap = argparse.ArgumentParser()
ap.add_argument("--written", default=os.path.join(HERE, "results_random25_davis"))
ap.add_argument("--spoken", default=os.path.join(HERE, "results_random25_davis_gscd_hw"))
a = ap.parse_args()
DIRS = {"up", "down", "left", "right"}
GROUPS = ("no clutter", "clutter", "all")


def load(d, pre):
    """{batch: [rows of one trial, ...]}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"{pre}_t*_b*.csv"))):
        if f.endswith("_kws.csv"):
            continue
        b = int(os.path.basename(f).split("_b")[1][:2])
        out.setdefault(b, []).append(list(csv.DictReader(open(f))))
    return out


def trial_metrics(R, spoken):
    m = {"C": 100 * np.mean([int(r["correct"]) for r in R]),
         "L": 100 * np.mean([int(r["landed_id"]) != 0 for r in R])}
    p = [int(r["correct"]) for r in R if r["impossible"] == "0"]
    i = [int(r["correct"]) for r in R if r["impossible"] == "1"]
    m["Cp"] = 100 * np.mean(p) if p else None
    m["Ci"] = 100 * np.mean(i) if i else None
    if spoken:
        m["K"] = 100 * np.mean([r["recognized"] == r["command"] for r in R])
    return m


def kws_outcome(r):
    if r["recognized"] == r["command"]:
        return "heard"
    return "wrong dir" if r["recognized"] in DIRS else "missed"


def fmt(x):
    x = [v for v in x if v is not None]
    return f"{np.mean(x):5.1f} ± {np.std(x):4.1f}" if x else "     –      "


def group_of(b):
    return "clutter" if "/bg/" in DESC[b] else "no clutter"


W, S = load(a.written, "01"), load(a.spoken, "11")
cols = ("C", "L", "Cp", "Ci")
names = {"C": "compliance", "L": "lock rate", "Cp": "compl. possible", "Ci": "compl. imposs.", "K": "KWS acc"}

for title, data, spoken in (("DAVIS + written (exp. 3)", W, False), ("DAVIS + GSCD-HW (exp. 4)", S, True)):
    keys = cols + (("K",) if spoken else ())
    print(f"\n### {title}\n")
    print("| batch | scene | n | " + " | ".join(names[k] for k in keys) + " |")
    print("|---|---|---|" + "---|" * len(keys))
    G = {g: {k: [] for k in keys} for g in GROUPS}
    for b in sorted(data):
        M = [trial_metrics(R, spoken) for R in data[b]]
        for g in (group_of(b), "all"):
            for k in keys:
                G[g][k] += [m[k] for m in M]
        print(f"| {b} | {DESC[b]} | {len(M)} | " + " | ".join(fmt([m[k] for m in M]) for k in keys) + " |")
    for g in GROUPS:
        if G[g]["C"]:
            print(f"| **{g}** | | {len(G[g]['C'])} | " + " | ".join(fmt(G[g][k]) for k in keys) + " |")

if S:
    print("\n### DAVIS + GSCD-HW: compliance by KWS outcome (steps pooled)\n")
    print("| group | command | heard correctly | not recognised | wrong direction |")
    print("|---|---|---|---|---|")
    for g in GROUPS:
        rows = [r for b in S if g in ("all", group_of(b)) for R in S[b] for r in R]
        for kind, sel in (("possible", "0"), ("impossible", "1"), ("all", None)):
            cells = []
            for o in ("heard", "missed", "wrong dir"):
                x = [int(r["correct"]) for r in rows if kws_outcome(r) == o and (sel is None or r["impossible"] == sel)]
                cells.append(f"{100 * np.mean(x):5.1f} % (n={len(x)})" if x else "– (n=0)")
            print(f"| {g} | {kind} | " + " | ".join(cells) + " |")
