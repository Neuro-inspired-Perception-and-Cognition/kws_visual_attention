"""
Ground truth + oracle from one label mask.

The mask is a second image the same size as the processing grid, but each pixel
holds an object ID (1..N, 0 = background). It's built from
the same sprites the video uses, so it lines up with the frames exactly. From
that one array we get object positions (centroids), the oracle's correct target
(geometry on those centroids), and fovea scoring (a single pixel lookup).
"""

import csv
import numpy as np
from PIL import Image


def build_mask(sprites, homes, width, height, sprite_size):
    """One integer per pixel: object ID (1..N), 0 = background. Objects at home."""
    mask = Image.new("L", (width, height), 0)
    for oid, (sprite, (hx, hy)) in enumerate(zip(sprites, homes), start=1):
        silhouette = sprite.split()[-1].point(lambda a: 255 if a > 127 else 0)  # alpha -> shape
        tile = Image.new("L", sprite.size, 0)
        tile.paste(oid, (0, 0), silhouette)          # ID where the shape is
        px = round(hx - sprite_size / 2)
        py = round(hy - sprite_size / 2)
        mask.paste(tile, (px, py), silhouette)       # stamp it onto the big mask
    return np.asarray(mask)                           # shape (height, width), values 0..N


def centroids(mask):
    """{id: (x, y)} -- mean pixel position of each object region."""
    out = {}
    for oid in range(1, int(mask.max()) + 1):
        ys, xs = np.where(mask == oid)
        if len(xs):
            out[oid] = (float(xs.mean()), float(ys.mean()))
    return out


def save_truth(centroids_dict, names, path="scene_truth.csv"):
    """Write id, name, x, y to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "x", "y"])
        for oid, (x, y) in centroids_dict.items():
            w.writerow([oid, names[oid - 1], round(x, 1), round(y, 1)])
    print(f"wrote {path}: {len(centroids_dict)} objects")


def oracle(centroids_dict, ref, direction, k, exclude=None):
    """k-th nearest object in `direction` from point `ref` (k is 1-indexed).

    Two rules keep this honest on a grid:
      * `exclude` drops the object the fovea is ALREADY sitting on. Without it,
        a fovea parked a pixel past that object's centroid sees it as the
        "nearest object to the left" and the oracle returns the start position.
      * the commanded axis must DOMINATE: "up" means more vertical than
        horizontal displacement, so a same-row neighbour whose centroid is a
        fraction of a pixel higher cannot masquerade as being above.

    Returns (id, (x, y)), or (None, None) if there's no k-th object.
    """
    rx, ry = ref
    keep = {
        "left":  lambda x, y: x < rx,
        "right": lambda x, y: x > rx,
        "up":    lambda x, y: y < ry,
        "down":  lambda x, y: y > ry,
    }[direction]
    vertical = direction in ("up", "down")

    def dominant(x, y):
        return abs(y - ry) > abs(x - rx) if vertical else abs(x - rx) > abs(y - ry)

    cands = [(oid, x, y) for oid, (x, y) in centroids_dict.items()
             if oid != exclude and keep(x, y) and dominant(x, y)]
    cands.sort(key=lambda t: (t[1] - rx) ** 2 + (t[2] - ry) ** 2)  # nearest first
    if k <= len(cands):
        oid, x, y = cands[k - 1]
        return oid, (x, y)
    return None, None


def landed_on(mask, fovea, target_id):
    """True if the fovea pixel belongs to target_id. One array read, any shape."""
    x, y = round(fovea[0]), round(fovea[1])
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return False
    return bool(mask[y, x] == target_id)


def object_at(mask, point):
    """Object ID under `point`, or None for background / out of bounds."""
    x, y = round(point[0]), round(point[1])
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return None
    return int(mask[y, x]) or None


def object_radius(mask):
    """Median equivalent radius of the objects in the mask, in grid px."""
    import math
    rs = []
    for oid in range(1, int(mask.max()) + 1):
        area = int((mask == oid).sum())
        if area:
            rs.append(math.sqrt(area / math.pi))
    return float(np.median(rs)) if rs else 0.0


def object_near(mask, truth, point, snap):
    """Object under `point`, or the nearest centroid within `snap` px.

    The exact pixel lookup is not enough for the STARTING fixation: the fovea is
    seeded at the saliency peak, and the saliency blob is wider than the object,
    so the peak often sits a few px outside the footprint. Read exactly as
    "background" that leaves the current object un-excluded, and the oracle
    returns the object the fovea is already sitting on.
    """
    exact = object_at(mask, point)
    if exact is not None:
        return exact
    px, py = point
    best, best_d = None, None
    for oid, (x, y) in truth.items():
        d = (x - px) ** 2 + (y - py) ** 2
        if best_d is None or d < best_d:
            best, best_d = oid, d
    if best is not None and best_d <= snap ** 2:
        return best
    return None


def score_run(mask, truth, trials, ref=None, names=None, out_path=None):
    """Score a run's fovea landings against the oracle. Each trial is a dict:
        {"direction": "left", "k": 1, "fovea": (x, y), "ref": (x, y)}
    k defaults to 1 (single-object scenes); ref defaults to the `ref` given here.
    Prints a line per trial and the overall % correct. If out_path is given,
    writes a results CSV pairing the model's landing with the expected target.
    Returns (n_correct, n_total)."""
    def label(oid):
        if oid is None:
            return "none"
        if oid == 0:
            return "bg"
        return names[oid - 1] if names else str(oid)

    rows, correct = [], 0
    print("trial  command      start        target       landed       result")
    for i, t in enumerate(trials, start=1):
        r = t.get("ref", ref)
        k = t.get("k", 1)
        start_id = object_at(mask, r)                 # what it was looking at
        target, _ = oracle(truth, r, t["direction"], k, exclude=start_id)
        fx, fy = t["fovea"]
        landed = object_at(mask, (fx, fy)) or 0
        hit = target is not None and landed == target
        correct += hit
        cmd = t["direction"] + (f" k{k}" if k != 1 else "")
        print(f"{i:>3}    {cmd:<11}  {label(start_id):<11}  {label(target):<11}  "
              f"{label(landed):<11}  {'HIT' if hit else 'miss'}")
        rows.append({"trial": i, "direction": t["direction"], "k": k,
                     "ref_x": round(r[0], 1), "ref_y": round(r[1], 1),
                     "fovea_x": round(fx, 1), "fovea_y": round(fy, 1),
                     "start_id": "" if start_id is None else start_id,
                     "start": label(start_id),
                     "target_id": "" if target is None else target,
                     "target": label(target),
                     "landed_id": landed, "landed": label(landed),
                     "correct": int(hit)})

    n = len(trials)
    acc = 100 * correct / n if n else 0.0
    print(f"\n{correct}/{n} correct  ({acc:.1f}%)")
    if out_path and rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_path}")
    return correct, n


def score_trajectory(mask, truth, trials, names=None, out_path=None, snap=None):
    """Score a run as a TRAJECTORY of object IDs.

    One row per command actually typed. A command that failed to move the fovea
    off its current object is a FAILURE ("did not move"), not a dropped trial --
    silently skipping those inflates the score.

    Each trial: {"direction": "left", "k": 1, "ref": (x, y), "fovea": (x, y)}
    """
    def label(oid):
        if oid is None or oid == 0:
            return "background"
        return f"object {oid}" + (f" ({names[oid-1]})" if names else "")

    if snap is None:
        snap = 1.5 * object_radius(mask)      # tolerate a fixation just outside a blob
    rows, correct = [], 0
    first = object_near(mask, truth, trials[0]["ref"], snap) if trials else None
    print("begin visual attention.")
    print(f"most salient point in {label(first)}.")

    for i, t in enumerate(trials, start=1):
        r, k = t["ref"], t.get("k", 1)
        start_id = object_near(mask, truth, r, snap)
        target, _ = oracle(truth, r, t["direction"], k, exclude=start_id)
        landed = object_near(mask, truth, t["fovea"], snap)

        if landed == start_id:
            verdict, ok = "FAILED - did not move", False
        elif target is None:
            verdict, ok = "no object that way", landed is None
        elif landed == target:
            verdict, ok = "OK", True
        else:
            verdict, ok = f"WRONG - expected {label(target)}", False
        correct += ok

        print(f"command: {t['direction']}")
        print(f"now most salient point in {label(landed)}.   [{verdict}]")

        rows.append({"step": i, "command": t["direction"], "k": k,
                     "start_id": start_id or 0, "start": label(start_id),
                     "expected_id": target or 0, "expected": label(target),
                     "landed_id": landed or 0, "landed": label(landed),
                     "verdict": verdict, "correct": int(ok)})

    n = len(trials)
    print(f"\n{correct}/{n} commands correct  ({100*correct/n if n else 0:.1f}%)")
    moved = sum(1 for r in rows if "did not move" not in r["verdict"])
    print(f"fovea moved on {moved}/{n} commands")
    if out_path and rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {out_path}")
    return correct, n