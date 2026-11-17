"""Tables for the four random25 experiments in one results folder (default: written_auto/results_v2/).

Per experiment: per batch and per group (no clutter / clutter / all) compliance, lock rate, compliance for possible /
impossible commands (mean ± population std over trials, as in the runners), % impossible and, for spoken runs,
KWS accuracy. Then a summary table of the groups side by side, and for the spoken runs compliance split by what the
KWS did with the step (heard correctly / not recognised = unknown or none / wrong direction), steps pooled.

Usage (from the repo root):
    python3 written_auto/tables_v2.py [RESULTS_DIR]
"""
import csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from batches import DESC

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_v2")
EXPS = [("sim_written", "00", "1 synthetic + written", False), ("sim_gscd_hw", "10", "2 synthetic + GSCD-HW", True),
        ("davis_written", "01", "3 DAVIS + written", False), ("davis_gscd_hw", "11", "4 DAVIS + GSCD-HW", True)]
DIRS = {"up", "down", "left", "right"}
GROUPS = ("no clutter", "clutter", "all")
NAMES = {"C": "compliance", "L": "lock rate", "Cp": "compl. possible", "Ci": "compl. impossible",
         "I": "% impossible", "K": "KWS acc"}


def load(d, pre):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"{pre}_t*_b*.csv"))):
        if not f.endswith("_kws.csv"):
            out.setdefault(int(os.path.basename(f).split("_b")[1][:2]), []).append(list(csv.DictReader(open(f))))
    return out


def trial_metrics(R, spoken):
    p = [int(r["correct"]) for r in R if r["impossible"] == "0"]
    i = [int(r["correct"]) for r in R if r["impossible"] == "1"]
    m = {"C": 100 * np.mean([int(r["correct"]) for r in R]),
         "L": 100 * np.mean([int(r["landed_id"]) != 0 for r in R]),
         "Cp": 100 * np.mean(p) if p else None, "Ci": 100 * np.mean(i) if i else None,
         "I": 100 * np.mean([int(r["impossible"]) for r in R])}
    if spoken:
        m["K"] = 100 * np.mean([r["recognized"] == r["command"] for r in R])
    return m


def fmt(x):
    x = [v for v in x if v is not None]
    return f"{np.mean(x):5.1f} ± {np.std(x):4.1f}" if x else "–"


def group_of(b):
    return "clutter" if "/bg/" in DESC[b] else "no clutter"


def outcome(r):
    if r["recognized"] == r["command"]:
        return "heard"
    return "wrong dir" if r["recognized"] in DIRS else "missed"


summary = []
for sub, pre, title, spoken in EXPS:
    data = load(os.path.join(ROOT, sub), pre)
    if not data:
        print(f"\n### {title}: no results in {os.path.join(ROOT, sub)}")
        continue
    keys = ("C", "L", "Cp", "Ci", "I") + (("K",) if spoken else ())
    rules = {r.get("rule", "?") for b in data for R in data[b] for r in R}
    print(f"\n### {title} (rule {', '.join(sorted(rules))})\n")
    print("| batch | scene | n | " + " | ".join(NAMES[k] for k in keys) + " |")
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
            summary.append((title, g, len(G[g]["C"]), {k: fmt(G[g][k]) for k in keys}))
    if spoken:
        print(f"\nCompliance by KWS outcome (steps pooled):\n")
        print("| group | command | heard correctly | not recognised | wrong direction |")
        print("|---|---|---|---|---|")
        for g in GROUPS:
            rows = [r for b in data if g in ("all", group_of(b)) for R in data[b] for r in R]
            for kind, sel in (("possible", "0"), ("impossible", "1"), ("all", None)):
                cells = []
                for o in ("heard", "missed", "wrong dir"):
                    x = [int(r["correct"]) for r in rows if outcome(r) == o and (sel is None or r["impossible"] == sel)]
                    cells.append(f"{100 * np.mean(x):5.1f} % (n={len(x)})" if x else "– (n=0)")
                print(f"| {g} | {kind} | " + " | ".join(cells) + " |")
        # what the KWS output was, per intended word
        steps = [r for b in data for R in data[b] for r in R]
        print("\nKWS output per intended word (steps):\n")
        labels = ["up", "down", "left", "right", "unknown", "none"]
        print("| intended | " + " | ".join(labels) + " | acc |")
        print("|---|" + "---|" * (len(labels) + 1))
        for w in ["up", "down", "left", "right"]:
            S = [r for r in steps if r["command"] == w]
            if S:
                print(f"| {w} | " + " | ".join(str(sum(r["recognized"] == l for r in S)) for l in labels)
                      + f" | {100 * np.mean([r['recognized'] == w for r in S]):.1f} % |")

if summary:
    print("\n### Summary (mean ± std over trials)\n")
    print("| experiment | group | n | compliance | lock rate | compl. possible | compl. impossible | KWS acc |")
    print("|---|---|---|---|---|---|---|---|")
    for title, g, n, m in summary:
        print(f"| {title} | {g} | {n} | {m['C']} | {m['L']} | {m['Cp']} | {m['Ci']} | {m.get('K', '–')} |")
