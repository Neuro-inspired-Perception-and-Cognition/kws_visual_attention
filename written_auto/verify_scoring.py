"""Independent re-implementation of the automated scoring (rules 1, 2 v2, 3), checked against every stored CSV row.
Geometry written from the description (footprint distance + snap, dominant-axis nearest object), not copied.
Usage: venv/bin/python written_auto/verify_scoring.py DIR [DIR ...]   (folders with *_tTT_bBB.csv)
"""
import csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WA = HERE
sys.path.insert(0, WA)
from batches import BATCHES
from run_rwo_random25 import SCENES

DAVIS = {1: "4_circles_mono_nobg", 2: "4_circles_mono_bg_jitter", 3: "6_circles_mono_nobg", 4: "6_circles_mono_bg_jitter",
         5: "6_objects_mono_nobg", 6: "6_objects_mono_bg_jitter", 7: "6_objects_color_nobg", 8: "6_objects_color_bg_jitter",
         9: "9_objects_color_nobg", 10: "9_objects_color_bg_jitter"}
DIRS = {"up", "down", "left", "right"}
REPO = os.path.dirname(WA)


def mask_for(d, pre, b):
    if "rwo" in d:
        return f"stimuli/ground_truth_masks/rwo_{SCENES[b][0]}.mask.npy"
    if pre[1] == "1":
        return f"stimuli/ground_truth_masks/davis_{DAVIS[b]}_346x260.mask.npy"
    return BATCHES[b][1]


class Geo:
    def __init__(self, path):
        self.m = np.load(os.path.join(REPO, path))
        self.ids = [i for i in range(1, int(self.m.max()) + 1) if (self.m == i).any()]
        self.pix = {i: np.argwhere(self.m == i) for i in self.ids}            # (y, x)
        self.c = {i: (p[:, 1].mean(), p[:, 0].mean()) for i, p in self.pix.items()}

    def near(self, p, snap):
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= y < self.m.shape[0] and 0 <= x < self.m.shape[1] and self.m[y, x]:
            return int(self.m[y, x])
        d = {i: np.min((q[:, 1] - p[0]) ** 2 + (q[:, 0] - p[1]) ** 2) for i, q in self.pix.items()}
        i = min(d, key=d.get)
        return i if d[i] <= snap ** 2 else None

    def nearest_in_dir(self, p, direction, exclude):
        best = None
        for i, (x, y) in self.c.items():
            dx, dy = x - p[0], y - p[1]
            ok = {"left": dx < 0 and abs(dx) > abs(dy), "right": dx > 0 and abs(dx) > abs(dy),
                  "up": dy < 0 and abs(dy) > abs(dx), "down": dy > 0 and abs(dy) > abs(dx)}[direction]
            if ok and i != exclude and (best is None or dx * dx + dy * dy < best[1]):
                best = (i, dx * dx + dy * dy)
        return best[0] if best else None


def check(d):
    bad, n, geo = [], 0, {}
    for f in sorted(glob.glob(os.path.join(d, "*_t*_b*.csv"))):
        if f.endswith("_kws.csv"):
            continue
        pre, b = os.path.basename(f)[:2], int(os.path.basename(f).split("_b")[1][:2])
        mp = mask_for(d, pre, b)
        g = geo.setdefault(mp, Geo(mp))
        R = list(csv.DictReader(open(f)))
        state = False                                  # rule 2 v2 "lost"
        for i, r in enumerate(R):
            n += 1
            snap = float(r["snap_px"])
            ref, fov = (float(r["ref_x"]), float(r["ref_y"])), (float(r["fovea_x"]), float(r["fovea_y"]))
            start, landed = g.near(ref, snap), g.near(fov, snap)
            spoken = "recognized" in r
            rec = r["recognized"] if spoken else r["command"]
            # rule 2 v2: state at this command (decided before it); first command: not lost
            if i == 0:
                lost = False
            else:
                prev = R[i - 1]
                prev_rec = prev["recognized"] if spoken else prev["command"]
                lost = (r["prev_locked"] == "0") if prev_rec in DIRS else state
            state = lost
            T = {g.nearest_in_dir(ref, r["command"], start)}
            if lost and start is not None:
                T.add(g.nearest_in_dir(ref, r["command"], None))
            T.discard(None)
            if spoken and rec != r["command"]:
                ok = False
            elif T:
                ok = landed in T
            else:
                ok = landed is None or landed == start
            mine = {"start_id": start or 0, "landed_id": landed or 0, "either": int(lost), "impossible": int(not T),
                    "correct": int(ok)}
            diff = {k: (r[k], v) for k, v in mine.items() if str(v) != r[k]}
            if diff:
                bad.append((os.path.basename(f), r["step"], diff))
    print(f"{d}: {n} rows, {len(bad)} mismatching")
    for x in bad[:8]:
        print("   ", x)


for d in sys.argv[1:]:
    check(d)
