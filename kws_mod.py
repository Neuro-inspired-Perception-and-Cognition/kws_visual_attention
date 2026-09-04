"""
Basic hard-coded KWS-modulated attention. Fovea panning over a decaying per-pixel membrane

events -> run_attention -> saliency 
per-pixel neurons decay + integrate saliency, with the fovea's target area BOOSTED:
    M = LEAK*M + (1-LEAK)*saliency*(1 + BOOST*foc)
the fovea starts on sal_max and the keyword pans it (or snaps it). attention is
read out near the fovea. one panel: the membrane, with fovea + attended markers.
"""

import numpy as np
import cv2
import sys
import torch
from datetime import datetime
from visual_attention.helpers_visual_att import initialise_attention, run_attention

# ---------------- config ----------------
NPY_PATH = ""
COL_X, COL_Y, COL_P, COL_T = 0, 1, 2, 3
TIME_SCALE = 1e-3
WINDOW_MS  = 50
DOWNSAMPLE = 2

ATTENTION_PARAMS = {
    'size_krn': 16, 'r0': 7, 'rho': 0.015, 'theta': np.pi*3/2,
    'thetas': np.arange(0, 2*np.pi, np.pi/4), 'thick': 12,
    'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
    'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
}

WORD       = "right"
CONF       = 0.6
THRESHOLD  = 0.6

LEAK        = 0.5          # membrane decay (higher = slower decay)
STEP        = 8.0          # px the fovea pans per window
SACCADE_JUMP = 60.0        # px the fovea jumps on a saccade command
READOUT_R   = 25.0         # radius: focus neighborhood AND boost width
BOOST       = 2.0          # how much the fovea's target area is amplified
CAP_RATIO   = 2.5          # lock when attended M is this x the local zone mean
MIN_TRAVEL  = 50.0         # px the fovea must travel before it is allowed to lock
MODE        = "pan"        # "pan" or "saccade"

DEBUG = True
DIRS = {"right": 0.0, "down": np.pi/2, "left": np.pi, "up": 3*np.pi/2} # this might have to be changed

# ---------------- load npy ----------------
device = torch.device("cpu")
print(f"Using device: {device}")

data = np.load(NPY_PATH)
if data.dtype.names is not None:
    names = data.dtype.names
    def _pick(*c):
        for n in c:
            if n in names: return data[n]
        raise KeyError(c)
    ev_x = _pick('x').astype(int); ev_y = _pick('y').astype(int)
    ev_t = _pick('timestamp', 't', 'ts').astype(float)
else:
    ev_x = data[:, COL_X].astype(int); ev_y = data[:, COL_Y].astype(int)
    ev_t = data[:, COL_T].astype(float)
ev_t = ev_t * TIME_SCALE
o = np.argsort(ev_t); ev_x, ev_y, ev_t = ev_x[o], ev_y[o], ev_t[o]

W_orig = int(ev_x.max())+1; H_orig = int(ev_y.max())+1
max_x = W_orig // DOWNSAMPLE; max_y = H_orig // DOWNSAMPLE
resolution = (max_y, max_x)
t0 = ev_t[0]
frame_idx = ((ev_t - t0) // WINDOW_MS).astype(int)
n_frames = int(frame_idx.max())+1

print(f"Loaded events        : {len(ev_t)}  span {ev_t[-1]-t0:.0f} ms")
print(f"Processing resolution: {max_x} x {max_y}")
print(f"Windows              : {n_frames} @ {WINDOW_MS} ms")
print(f"Mode                 : {MODE}  leak={LEAK} step={STEP} boost={BOOST} readout_r={READOUT_R}")
print(f"Command              : '{WORD}' (conf={CONF})")

net = initialise_attention(device, ATTENTION_PARAMS)

X, Y = np.meshgrid(np.arange(max_x), np.arange(max_y))
M = np.zeros((max_y, max_x))
fx = fy = None
active = None
locked = False
sx = sy = None      # where the current scan started

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_name = f"fovea_{MODE}_{WORD}_{timestamp}_{LEAK}.mp4"
vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W_orig, H_orig))  # single panel
if not vw.isOpened():
    print("ERROR: video writer"); sys.exit(1)

def to_bgr(m, cmap=cv2.COLORMAP_JET):
    m = m.astype(float); lo, hi = m.min(), m.max()
    if hi > lo: m = (m-lo)/(hi-lo)*255
    return cv2.applyColorMap(np.clip(m, 0, 255).astype(np.uint8), cmap)

# ---------------- loop ----------------
count = 0
for k in range(n_frames):
    m = frame_idx == k
    if not m.any(): continue
    xa = (ev_x[m] // DOWNSAMPLE).clip(0, max_x-1)
    ya = (ev_y[m] // DOWNSAMPLE).clip(0, max_y-1)
    window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
    window[0, ya, xa] = 255.0

    saliency, salmax = run_attention(window, net, device, resolution, ATTENTION_PARAMS['num_pyr'])
    saliency = np.asarray(saliency)
    if np.isnan(saliency).any() or saliency.max() == saliency.min(): continue

    if fx is None:
        fy, fx = float(salmax[0]), float(salmax[1])
        sx, sy = fx, fy

    # BOOST the fovea's target area, then decay + integrate
    foc = np.exp(-((X - fx)**2 + (Y - fy)**2) / (2 * READOUT_R**2))
    M = LEAK*M + (1.0 - LEAK)*saliency*(1.0 + BOOST*foc)

    if WORD != active:            # new command -> release the lock, scan again
        locked = False
        sx, sy = fx, fy           # scan starts here; must travel MIN_TRAVEL to lock

    if (not locked) and WORD in DIRS and CONF >= THRESHOLD:
        psi = DIRS[WORD]
        if MODE == "pan":
            fx = float(np.clip(fx + STEP*np.cos(psi), 0, max_x-1))
            fy = float(np.clip(fy + STEP*np.sin(psi), 0, max_y-1))
        elif WORD != active:
            fx = float(np.clip(fx + SACCADE_JUMP*np.cos(psi), 0, max_x-1))
            fy = float(np.clip(fy + SACCADE_JUMP*np.sin(psi), 0, max_y-1))
    active = WORD

    zone = (X - fx)**2 + (Y - fy)**2 <= READOUT_R**2
    if zone.any():
        ay, ax = np.unravel_index(int(np.argmax(np.where(zone, M, -np.inf))), M.shape)
    else:
        ay, ax = int(round(fy)), int(round(fx))

    # capture: attended point is a real peak -> fovea snaps onto it and stops panning
    travel = np.hypot(fx - sx, fy - sy) if sx is not None else 0.0
    if (not locked) and travel >= MIN_TRAVEL:
        zone_mean = float(M[zone].mean()) if zone.any() else 0.0
        if zone_mean > 0 and M[ay, ax] >= CAP_RATIO * zone_mean:
            fx, fy = float(ax), float(ay)
            locked = True

    if DEBUG:
        print(f"win {k:3d}/{n_frames} | {'LOCK' if locked else 'pan '} | "
              f"fovea=({int(fx)},{int(fy)}) | attended=({ax},{ay}) | "
              f"travel={travel:5.1f} | M@att={M[ay,ax]:.1f}")

    ds = DOWNSAMPLE
    p = cv2.resize(to_bgr(M), (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    # cv2.drawMarker(p, (int(fx*ds), int(fy*ds)), (0,0,255), cv2.MARKER_CROSS, 22, 2)   # fovea
    cv2.circle(p, (int(ax*ds), int(ay*ds)), 11, (255,255,255), 3)                    # attended
    cv2.putText(p, f"'{WORD}' [{MODE}] {'LOCK' if locked else 'pan'}", (8,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0) if locked else (255,255,255), 2)
    vw.write(p); count += 1

vw.release()
print(f"\nVideo saved : '{out_name}'\nTotal frames: {count}")