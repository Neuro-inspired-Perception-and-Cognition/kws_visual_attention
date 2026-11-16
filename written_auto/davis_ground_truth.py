"""Ground-truth masks for the PRE-RECORDED DAVIS346 sessions (monitor showing the rendered stimuli).

The camera_*.mask.npy files in stimuli/ground_truth_masks/ come from earlier live sessions with a different
camera placement and do NOT line up with these recordings, so new masks are built from the recordings'
own APS (grayscale) video:
  1. average the APS frames of the whole 30 s recording (moving hatches blur out, the objects stay)
  2. threshold (Otsu on the blurred average) -> open (remove thin hatch residue) -> fill holes
  3. keep the N largest connected components (N = objects in the stimulus)
  4. ids in reading order (row by row, left to right) - the same convention as the synthetic masks/NAMES
  5. label image downsampled to the processing grid (DOWNSAMPLE = 2 -> 173 x 130)
Outputs (stimuli/ground_truth_masks/): davis_<stem>.mask.npy, davis_<stem>.truth.csv, davis_<stem>.preview.png
CHECK the previews (outlines over the APS average and over the events) before trusting any score.

Usage (repo root):  python written_auto/davis_ground_truth.py [--data DIR]
"""
import argparse, csv, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from batches import BATCHES

DATA = "/media/pwz/KINGSTON/NAS-KWS/Experiments/Event and Grayscale data - DAVIS-20261116T144619Z-1-001/Event and Grayscale data - DAVIS/raw"
STEMS = {1: "4_circles_mono_nobg", 2: "4_circles_mono_bg_jitter", 3: "6_circles_mono_nobg", 4: "6_circles_mono_bg_jitter",
         5: "6_objects_mono_nobg", 6: "6_objects_mono_bg_jitter", 7: "6_objects_color_nobg", 8: "6_objects_color_bg_jitter",
         9: "9_objects_color_nobg", 10: "9_objects_color_bg_jitter"}
DS = 2
REL_THR = {9: 0.25, 10: 0.25}   # colour 9-object scenes: dim tree / mug need the top-hat threshold


def aps_average(path, step=3):
    cap = cv2.VideoCapture(path); acc = None; n = 0; i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
            acc = g if acc is None else acc + g; n += 1
        i += 1
    return (acc / n).astype(np.float32)


def segment(avg, n_obj, rel_thr=None):
    """rel_thr=None: Otsu on the blurred average (bright objects). rel_thr=f: white top-hat (local background
    removed) thresholded at f x its 99.5th percentile - picks up the dim objects (tree, mug) of the colour scenes."""
    img = cv2.GaussianBlur(avg, (0, 0), 2.0)
    if rel_thr is None:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thr, bw = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        close = 7
    else:
        th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
        thr = rel_thr * np.percentile(th, 99.5)
        bw = ((th > thr) * 255).astype(np.uint8)
        close = 11
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    # fill holes
    ff = bw.copy(); h, w = bw.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    bw = bw | cv2.bitwise_not(ff)
    n, lab, stats, cent = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    comps = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:n_obj]
    if len(comps) < n_obj:
        raise RuntimeError(f"found {len(comps)} components, expected {n_obj}")
    # reading order: group into rows by y (gap > half the median object height), then sort by x
    cs = sorted(comps, key=lambda i: cent[i][1])
    hmed = np.median([stats[i, cv2.CC_STAT_HEIGHT] for i in cs])
    rows, cur = [], [cs[0]]
    for i in cs[1:]:
        if cent[i][1] - cent[cur[-1]][1] > 0.5 * hmed:
            rows.append(cur); cur = [i]
        else:
            cur.append(i)
    rows.append(cur)
    order = [i for r in rows for i in sorted(r, key=lambda i: cent[i][0])]
    label = np.zeros_like(lab, np.uint8)
    for oid, i in enumerate(order, start=1):
        label[lab == i] = oid
    return label, thr, [len(r) for r in rows]


def fix_with_synthetic(label, b, min_frac=0.5):
    """The monitor showed exactly the synthetic scene, so map the synthetic mask onto the camera with a
    homography fitted on the centroids of the well-segmented objects; objects whose APS footprint is smaller
    than min_frac x their mapped synthetic footprint (dim objects, e.g. the green tree) take the mapped shape.
    Returns the fixed label and per-object IoU(APS, mapped synthetic) as a quality check."""
    syn = np.load(os.path.join(REPO, BATCHES[b][1]))                     # processing grid (DS=2)
    syn = cv2.resize(syn.astype(np.uint8), (346, 260), interpolation=cv2.INTER_NEAREST)
    ids = range(1, int(syn.max()) + 1)
    c_syn = {i: np.argwhere(syn == i).mean(0)[::-1] for i in ids}
    c_cam = {i: np.argwhere(label == i).mean(0)[::-1] for i in ids if (label == i).any()}
    area_ratio = {i: (label == i).sum() / max((syn == i).sum(), 1) for i in ids}
    # scale-normalised area: compare against the median ratio (the camera image is smaller than the render)
    med = np.median(list(area_ratio.values()))
    good = [i for i in ids if i in c_cam and area_ratio[i] >= min_frac * med]
    H, _ = cv2.findHomography(np.float32([c_syn[i] for i in good]), np.float32([c_cam[i] for i in good]), 0)
    warped = cv2.warpPerspective(syn, H, (346, 260), flags=cv2.INTER_NEAREST)
    out = label.copy(); iou = {}; replaced = []
    for i in ids:
        a, w = label == i, warped == i
        iou[i] = (a & w).sum() / max((a | w).sum(), 1)
        if i not in good:
            out[out == i] = 0
            out[w & (out == 0)] = i
            replaced.append(i)
    return out, iou, replaced


def event_image(path, ms=1000):
    ev = np.load(path, mmap_mode="r")
    t = np.asarray(ev["t"][:2_000_000], np.int64)
    k = np.searchsorted(t, t[0] + ms * 1000)
    acc = np.zeros((260, 346), np.float32)
    np.add.at(acc, (np.asarray(ev["y"][:k]), np.asarray(ev["x"][:k])), 1)
    return np.clip(acc / max(np.percentile(acc[acc > 0], 95), 1) * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=DATA); a = ap.parse_args()
    out_dir = os.path.join(REPO, "stimuli", "ground_truth_masks")
    for b, stem in STEMS.items():
        names = BATCHES[b][2]
        avg = aps_average(os.path.join(a.data, "gs_videos", stem + "_346x260.mp4"))
        label, thr, rows = segment(avg, len(names), REL_THR.get(b))
        expected_rows = [3, 3, 3] if len(names) == 9 else ([2, 2] if len(names) == 4 else [3, 3])
        if rows != expected_rows:
            print(f"!!! batch {b}: rows {rows}, expected {expected_rows} - check the preview")
        label, iou, replaced = fix_with_synthetic(label, b)
        print(f"         IoU(APS, synthetic->camera): " + " ".join(f"{names[i-1]}={v:.2f}" for i, v in iou.items())
              + (f"  | from synthetic: {[names[i-1] for i in replaced]}" if replaced else ""))
        base = os.path.join(out_dir, f"davis_{stem}_346x260")
        grid = label[::DS, ::DS]
        np.save(base + ".mask.npy", grid)
        with open(base + ".truth.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "name", "x", "y", "area_px", "radius_px"])
            for oid in range(1, len(names) + 1):
                ys, xs = np.nonzero(grid == oid)
                w.writerow([oid, names[oid - 1], round(xs.mean(), 1), round(ys.mean(), 1), len(xs), round((len(xs) / np.pi) ** 0.5, 1)])
        # preview: APS average | events (first 1 s), outlines + ids
        panels = []
        for base_img in (cv2.normalize(avg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                         event_image(os.path.join(a.data, "events", stem + "_346x260.npy"))):
            im = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            for oid in range(1, len(names) + 1):
                cs, _ = cv2.findContours((label == oid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, (0, 0, 255), 1)
                ys, xs = np.nonzero(label == oid)
                cv2.putText(im, f"{oid}:{names[oid-1]}", (int(xs.mean()) - 20, int(ys.min()) - 3), 0, 0.33, (0, 255, 0), 1)
            panels.append(im)
        cv2.imwrite(base + ".preview.png", np.hstack(panels))
        print(f"batch {b:2d} {stem:26s} objects {len(names)} rows {rows} otsu {thr:.0f} -> {os.path.relpath(base, REPO)}.mask.npy")


if __name__ == "__main__":
    main()
