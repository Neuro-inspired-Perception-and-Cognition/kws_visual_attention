"""Tables for a Real World Objects results folder (run_rwo_random25.py): per scenario, per factor (number of objects,
camera movement, clutter) and all, with compliance, lock rate, compliance for possible / impossible commands
(mean ± population std over trials), % impossible and, for spoken runs, KWS accuracy plus compliance split by
KWS outcome (heard correctly / not recognised / wrong direction, steps pooled).

Usage (from the repo root):
    python3 written_auto/tables_rwo.py RESULTS_DIR
"""
import csv, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_rwo_random25 import SCENES, desc

D = sys.argv[1]
DIRS = {"up", "down", "left", "right"}
NAMES = {"C": "compliance", "L": "lock rate", "Cp": "compl. possible", "Ci": "compl. impossible",
         "I": "% impossible", "K": "KWS acc"}

data = {}
for f in sorted(glob.glob(os.path.join(D, "*_t*_b*.csv"))):
    if not f.endswith("_kws.csv"):
        data.setdefault(int(os.path.basename(f).split("_b")[1][:2]), []).append(list(csv.DictReader(open(f))))
if not data:
    sys.exit(f"no results in {D}")
spoken = any("recognized" in R[0] for b in data for R in data[b] if R)
keys = ("C", "L", "Cp", "Ci", "I") + (("K",) if spoken else ())


def metrics(R):
    p = [int(r["correct"]) for r in R if r["impossible"] == "0"]
    i = [int(r["correct"]) for r in R if r["impossible"] == "1"]
    m = {"C": 100 * np.mean([int(r["correct"]) for r in R]), "L": 100 * np.mean([int(r["landed_id"]) != 0 for r in R]),
         "Cp": 100 * np.mean(p) if p else None, "Ci": 100 * np.mean(i) if i else None,
         "I": 100 * np.mean([int(r["impossible"]) for r in R])}
    if spoken:
        m["K"] = 100 * np.mean([r["recognized"] == r["command"] for r in R])
    return m


def fmt(x):
    x = [v for v in x if v is not None]
    return f"{np.mean(x):5.1f} ± {np.std(x):4.1f}" if x else "–"


def row(label, bs):
    M = [metrics(R) for b in bs for R in data.get(b, [])]
    if M:
        print(f"| {label} | {len(M)} | " + " | ".join(fmt([m[k] for m in M]) for k in keys) + " |")


GROUPS = [("2 objects", lambda s: s[1] == 2), ("3 objects", lambda s: s[1] == 3), ("4 objects", lambda s: s[1] == 4),
          ("jitter", lambda s: s[2] == "jitter"), ("circular", lambda s: s[2] == "circular"),
          ("no clutter", lambda s: not s[3]), ("clutter", lambda s: s[3]), ("all", lambda s: True)]

rules = {r.get("rule", "?") for b in data for R in data[b] for r in R}
print(f"### Real World Objects, {'GSCD-HW' if spoken else 'written'} (rule {', '.join(sorted(rules))}, {D})\n")
print("| scenario | n | " + " | ".join(NAMES[k] for k in keys) + " |")
print("|---|---|" + "---|" * len(keys))
for b in sorted(data):
    row(f"{b}: {desc(b)}", [b])
for g, sel in GROUPS:
    row(f"**{g}**", [b for b in data if sel(SCENES[b])])

if spoken:
    print("\nCompliance by KWS outcome (steps pooled):\n")
    print("| group | command | heard correctly | not recognised | wrong direction |")
    print("|---|---|---|---|---|")

    def outcome(r):
        return "heard" if r["recognized"] == r["command"] else ("wrong dir" if r["recognized"] in DIRS else "missed")
    for g, sel in GROUPS:
        rows = [r for b in data if sel(SCENES[b]) for R in data[b] for r in R]
        for kind, s in (("possible", "0"), ("impossible", "1"), ("all", None)):
            cells = []
            for o in ("heard", "missed", "wrong dir"):
                x = [int(r["correct"]) for r in rows if outcome(r) == o and (s is None or r["impossible"] == s)]
                cells.append(f"{100 * np.mean(x):5.1f} % (n={len(x)})" if x else "– (n=0)")
            print(f"| {g} | {kind} | " + " | ".join(cells) + " |")
