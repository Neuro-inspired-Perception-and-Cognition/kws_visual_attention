"""
Helpers for ground truth and oracle from one label mask.

Each pixel holds an integer as an object ID. We get an array with the objects' 
positions (center of mass or centroids), the correct target (here named oracle), and the attention score.
"""

import csv
import numpy as np
from PIL import Image


def build_mask(sprites, homes, width, height, sprite_size):
    """One integer per pixel: object ID (1...N), 0=background"""
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
    """{id: (x, y)} - mean pixel position of each object region"""
    out = {}
    for oid in range(1, int(mask.max()) + 1):
        ys, xs = np.where(mask == oid)
        if len(xs):
            out[oid] = (float(xs.mean()), float(ys.mean()))
    return out


def save_truth(centroids_dict, names, path="scene_truth.csv"):
    """Write id, name, x, y to CSV"""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "x", "y"])
        for oid, (x, y) in centroids_dict.items():
            w.writerow([oid, names[oid - 1], round(x, 1), round(y, 1)])
    print(f"wrote {path}: {len(centroids_dict)} objects")


def oracle(centroids_dict, ref, direction, k):
    """k-th nearest object in `direction` from point `ref` (k is 1-indexed).
    Returns (id, (x, y)), or (None, None) if there's no k-th object"""
    rx, ry = ref
    keep = {
        "left":  lambda x, y: x < rx,
        "right": lambda x, y: x > rx,
        "up":    lambda x, y: y < ry,
        "down":  lambda x, y: y > ry,
    }[direction]
    cands = [(oid, x, y) for oid, (x, y) in centroids_dict.items() if keep(x, y)]
    cands.sort(key=lambda t: (t[1] - rx) ** 2 + (t[2] - ry) ** 2)  # nearest first
    if k <= len(cands):
        oid, x, y = cands[k - 1]
        return oid, (x, y)
    return None, None


def landed_on(mask, fovea, target_id):
    """True if the attention pixel belongs to target_id. One array read, any shape."""
    x, y = round(fovea[0]), round(fovea[1])
    return bool(mask[y, x] == target_id)


def score_run(mask, truth, trials, ref=None, names=None, out_path=None):
    """Score a run's fovea landings against the oracle. Each trial is a dict:
        {"direction": "left", "k": 1, "fovea": (x, y), "ref": (x, y)}
    k defaults to 1 (single-object scenes); ref defaults to the `ref` given here
    (pass per-trial ref for coupled scenes). Prints a line per trial and the
    overall % correct. If out_path is given, writes a results CSV pairing the
    model's landing with the expected target. Returns (n_correct, n_total)."""
    def label(oid):
        if oid is None:
            return "none"
        if oid == 0:
            return "bg"
        return names[oid - 1] if names else str(oid)

    rows, correct = [], 0
    print("trial  command      target      landed      result")
    for i, t in enumerate(trials, start=1):
        r = t.get("ref", ref)
        k = t.get("k", 1)
        target, _ = oracle(truth, r, t["direction"], k)
        fx, fy = t["fovea"]
        landed = int(mask[round(fy), round(fx)])
        hit = target is not None and landed_on(mask, (fx, fy), target)
        correct += hit
        cmd = t["direction"] + (f" k{k}" if k != 1 else "")
        print(f"{i:>3}    {cmd:<11}  {label(target):<10}  {label(landed):<10}  {'HIT' if hit else 'miss'}")
        rows.append({"trial": i, "direction": t["direction"], "k": k,
                     "fovea_x": round(fx, 1), "fovea_y": round(fy, 1),
                     "target_id": "" if target is None else target, "target": label(target),
                     "landed_id": landed, "landed": label(landed), "correct": int(hit)})

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