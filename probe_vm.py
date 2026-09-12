'''
Von Mises parameter sweep — saliency only, ONE PARAMETER AT A TIME.

Varies one parameter at a time around your baseline and reports, for each
setting, whether the arcs FILL the objects and whether the response BLOWS UP at
the frame border. One-at-a-time means the cost is the SUM of the axes, not the
product, and each result is directly attributable to one knob.

Metrics (from events + saliency only; no ground-truth mask):
  fill   mean saliency INSIDE the objects / mean saliency ON the event rings.
         < 1 -> ring-shaped, centre not filled.  ~1+ -> arcs vote to the centre.
  edge   mean saliency in the border band / mean elsewhere.
         > 1 -> border artefact competing with the objects.  < 1 -> clean.
  secs   wall-clock cost of that setting.

NOTE on cost: the filter is resized by fltr_resize_perc before the conv, so the
EFFECTIVE kernel is size_krn * fltr_resize_perc[0]. It must fit inside the
image or torch raises "Kernel size can't be greater than actual input size".
Combinations that would exceed it are skipped, not crashed.
'''

import csv
import math
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes, label

from visual_attention.helpers_visual_att import initialise_attention, run_attention

# ---------------- config ----------------
EVENTS_PATH = "data/6_circles_346x260.npy"
DOWNSAMPLE  = 1           
WINDOW_MS   = 100
WINDOW_N    = 3
EDGE_PX     = 3
DEVICE      = torch.device("cpu")

BASE = {
    'size_krn': 16, 'r0': 8, 'rho': 0.015, 'theta': np.pi * 3 / 2,
    'thetas': np.arange(0, 2 * np.pi, np.pi / 4), 'thick': 12,
    'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
    'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
}

# One axis at a time. Comment out any you don't want to spend time on.
# 'n_thetas' is special-cased into the `thetas` array below.
SWEEP = {
    'num_pyr':          [1, 2, 3, 4, 5, 6],          # cheap: scale via the pyramid
    'rho':              [0.005, 0.015, 0.05, 0.15],  # cheap: arc concentration
    'thick':            [4, 8, 12, 20],              # cheap: arc thickness
    'n_thetas':         [4, 8, 16],                  # cheap: orientation count
    'fltr_resize_perc': [[1, 1], [2, 2], [3, 3]],    # scales the kernel w/o size_krn
    'r0':               [7, 12, 16, 22],             # expensive-ish
    'size_krn':         [16, 32, 48, 64],            # expensive
}


# ---------------- load events ----------------
def load_events(path):
    data = np.load(path, allow_pickle=True)
    if data.dtype.names:
        names = {n.lower(): n for n in data.dtype.names}
        x = data[names['x']].astype(int)
        y = data[names['y']].astype(int)
        t = data[names[next(k for k in ('t', 'ts', 'timestamp', 'time') if k in names)]].astype(float)
    else:
        diffs = np.diff(data, axis=0)
        t_col = np.argmax((diffs >= 0).mean(axis=0))
        rest = [c for c in range(data.shape[1]) if c != t_col]
        t = data[:, t_col].astype(float)
        x, y = data[:, rest[0]].astype(int), data[:, rest[1]].astype(int)
    return x, y, (t - t.min()) / 1e3          # us -> ms


x, y, t = load_events(EVENTS_PATH)
x, y = x // DOWNSAMPLE, y // DOWNSAMPLE
max_x, max_y = int(x.max()) + 1, int(y.max()) + 1
resolution = (max_y, max_x)

lo = WINDOW_N * WINDOW_MS
m = (t >= lo) & (t < lo + WINDOW_MS)
ev = np.zeros((max_y, max_x), dtype=np.float32)
ev[y[m], x[m]] = 255.0
window = torch.from_numpy(ev)[None]
print(f"grid: {max_x} x {max_y}  (DOWNSAMPLE={DOWNSAMPLE})")
print(f"window {WINDOW_N}: {int(m.sum())} events on {int((ev > 0).sum())} pixels")

# ---------------- measure the objects from the events ----------------
dens   = gaussian_filter((ev > 0).astype(float), sigma=1.2)
solid  = binary_fill_holes(binary_closing(dens > 0.18, np.ones((3, 3))))
lab, n = label(solid)
areas  = np.array([(lab == i).sum() for i in range(1, n + 1)])
keep   = areas >= max(30, 0.1 * areas.max()) if len(areas) else np.array([], bool)
radii  = [math.sqrt(a / math.pi) for a in areas[keep]]
obj_r  = float(np.median(radii)) if radii else float('nan')
print(f"objects detected: {int(keep.sum())}   median radius: {obj_r:.1f} px "
      f"(metrics are computed over these)")

ring     = ev > 0
interior = solid & ~binary_closing(ring, np.ones((3, 3)))
border = np.zeros((max_y, max_x), bool)
border[:EDGE_PX, :] = border[-EDGE_PX:, :] = True
border[:, :EDGE_PX] = border[:, -EDGE_PX:] = True

MAX_EFF_KRN = min(max_y, max_x)      # torch limit: effective kernel must fit


def metrics(sal):
    fill = (float(sal[interior].mean() / (sal[ring].mean() + 1e-9))
            if interior.any() and ring.any() else float('nan'))
    edge = float(sal[border].mean() / (sal[~border].mean() + 1e-9))
    ay, ax = np.unravel_index(int(np.nanargmax(sal)), sal.shape)
    return fill, edge, ax, ay, bool(border[ay, ax])


def build(name, val):
    """Baseline params with one axis overridden."""
    p = dict(BASE)
    p['thetas'] = np.array(BASE['thetas'], copy=True)
    if name == 'n_thetas':
        p['thetas'] = np.arange(0, 2 * np.pi, 2 * np.pi / val)
    else:
        p[name] = val
    return p


def run_one(p):
    eff = p['size_krn'] * p['fltr_resize_perc'][0]
    if eff > MAX_EFF_KRN:
        return None, f"skipped: effective kernel {eff} > image {MAX_EFF_KRN}"
    t0 = time.time()
    net = initialise_attention(DEVICE, p)
    with torch.no_grad():
        sal, _ = run_attention(window, net, DEVICE, resolution, p['num_pyr'])
    return (np.asarray(sal, dtype=float), time.time() - t0), None


# ---------------- sweep, one axis at a time ----------------
rows, panels = [], []
for name, values in SWEEP.items():
    print(f"\n--- {name} " + "-" * (56 - len(name)))
    for val in values:
        p = build(name, val)
        out, why = run_one(p)
        if out is None:
            print(f"  {name}={str(val):<12} {why}")
            continue
        sal, secs = out
        fill, edge, ax, ay, ob = metrics(sal)
        rows.append([name, str(val), round(fill, 3), round(edge, 3), ax, ay, ob, round(secs, 1)])
        panels.append((f"{name}={val}", sal, fill, edge, ax, ay))
        print(f"  {name}={str(val):<12} fill={fill:5.2f}  edge={edge:5.2f}  "
              f"argmax=({ax},{ay}){' BORDER' if ob else '':<7}  {secs:5.1f}s")

# ---------------- which knobs actually moved the metrics? ----------------
print("\n=== sensitivity: range each axis produced ===")
for name in SWEEP:
    sub = [r for r in rows if r[0] == name]
    if len(sub) < 2:
        continue
    fills = [r[2] for r in sub if not math.isnan(r[2])]
    edges = [r[3] for r in sub]
    print(f"  {name:<18} fill {min(fills):5.2f}..{max(fills):5.2f}   "
          f"edge {min(edges):5.2f}..{max(edges):5.2f}")

clean = [r for r in rows if r[3] < 1.0]
print("\nsettings where the border is DIMMER than the interior (edge < 1):")
for r in sorted(clean, key=lambda r: r[3]) or []:
    print(f"  {r[0]}={r[1]:<12} edge={r[3]:5.2f}  fill={r[2]:5.2f}")
if not clean:
    print("  none — the border response persists across every setting tried")

with open("sweep_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["param", "value", "fill", "edge", "argmax_x", "argmax_y",
                "argmax_on_border", "secs"])
    w.writerows(rows)
print("\nwrote sweep_results.csv")

# ---------------- montage ----------------
if panels:
    ncol = 4
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow))
    axes = np.array(axes).reshape(-1)
    for a, (title, sal, fill, edge, ax_, ay_) in zip(axes, panels):
        a.imshow(sal, cmap='jet')
        a.plot(ax_, ay_, '+', color='cyan', markersize=11, markeredgewidth=2)
        a.set_title(f"{title}\nfill={fill:.2f} edge={edge:.2f}", fontsize=8)
        a.axis('off')
    for a in axes[len(panels):]:
        a.axis('off')
    fig.suptitle(f"raw saliency, one param at a time   (object radius ~ {obj_r:.1f} px)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("sweep.png", dpi=110)
    print("wrote sweep.png")