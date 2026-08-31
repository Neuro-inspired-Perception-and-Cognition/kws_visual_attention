"""
fovea_pan_interactive.py — live version of fovea_pan.py.

Type direction commands in the terminal WHILE the video is playing; the
active command updates and the fovea reacts, live, in an OpenCV window.
Same M/LEAK/BOOST membrane dynamics as the batch script — only the
input/display layer is new.

Commands (typed in the terminal, Enter to submit):
    right / left / up / down     set the active direction (conf defaults to 1.0)
    right 0.8                    same, with an explicit confidence
    stop  (or: none, clear)      release the active command, freeze in place
    reset                        zero the membrane and re-fixate on next salmax
    mode pan   / mode saccade    switch panning mode live
    quit / exit / q              end the session (also: 'q' in the video window)

Requires a local display for cv2.imshow — this won't work over a headless
SSH session without X forwarding. If that's your setup, say so and I'll
switch this to a Jupyter/matplotlib-based live loop instead.
"""

import queue
import sys
import threading
from datetime import datetime

import cv2
import numpy as np
import torch

from visual_attention.helpers_visual_att import initialise_attention, run_attention
from command_parser import parse_command

# ---------------- config ----------------
NPY_PATH = "/home/rocharay/kws_attention/data/6_weird_jitter_objects_346x260.npy"
COL_X, COL_Y, COL_P, COL_T = 0, 1, 2, 3
TIME_SCALE = 1e-3
WINDOW_MS = 100
DOWNSAMPLE = 2

ATTENTION_PARAMS = {
    'size_krn': 16, 'r0': 7, 'rho': 0.015, 'theta': np.pi * 3 / 2,
    'thetas': np.arange(0, 2 * np.pi, np.pi / 4), 'thick': 12,
    'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
    'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
}

CONF = 1.0
THRESHOLD = 0.6

LEAK = 0.5
STEP = 8.0
SACCADE_JUMP = 60.0
READOUT_R = 25.0
BOOST = 2.0
CAP_RATIO = 2.5
MIN_TRAVEL = 50.0
MODE = "pan"                # "pan" or "saccade" — changeable live via "mode <x>"

LOOP_PLAYBACK = True        # replay the clip forever so the session doesn't just end
RECORD = True                # also save an .mp4 of the session alongside the live view
PLAYBACK_MS = WINDOW_MS      # cv2.waitKey delay: paces playback AND pumps the GUI

DEBUG = True
DIRS = {"right": 0.0, "down": np.pi / 2, "left": np.pi, "up": 3 * np.pi / 2}

# ---------------- stdin reader thread ----------------
cmd_queue = queue.Queue()


def stdin_reader(q):
    """Runs in a background thread. input() blocks THIS thread, never the video loop."""
    print("Type a command and press Enter (right / left / up / down / stop / reset / "
          "mode pan|saccade / quit).")
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            q.put({"type": "quit"})
            return
        q.put(line)


threading.Thread(target=stdin_reader, args=(cmd_queue,), daemon=True).start()

# ---------------- load npy ----------------
device = torch.device("cpu")
print(f"Using device: {device}")

data = np.load(NPY_PATH)
if data.dtype.names is not None:
    names = data.dtype.names

    def _pick(*c):
        for n in c:
            if n in names:
                return data[n]
        raise KeyError(c)

    ev_x = _pick('x').astype(int)
    ev_y = _pick('y').astype(int)
    ev_t = _pick('timestamp', 't', 'ts').astype(float)
else:
    ev_x = data[:, COL_X].astype(int)
    ev_y = data[:, COL_Y].astype(int)
    ev_t = data[:, COL_T].astype(float)

ev_t = ev_t * TIME_SCALE
o = np.argsort(ev_t)
ev_x, ev_y, ev_t = ev_x[o], ev_y[o], ev_t[o]

W_orig, H_orig = int(ev_x.max()) + 1, int(ev_y.max()) + 1
max_x, max_y = W_orig // DOWNSAMPLE, H_orig // DOWNSAMPLE
resolution = (max_y, max_x)
t0 = ev_t[0]
frame_idx = ((ev_t - t0) // WINDOW_MS).astype(int)
n_frames = int(frame_idx.max()) + 1

print(f"Loaded events        : {len(ev_t)}  span {ev_t[-1] - t0:.0f} ms")
print(f"Processing resolution: {max_x} x {max_y}")
print(f"Windows              : {n_frames} @ {WINDOW_MS} ms  (loop={LOOP_PLAYBACK})")

net = initialise_attention(device, ATTENTION_PARAMS)

X, Y = np.meshgrid(np.arange(max_x), np.arange(max_y))
M = np.zeros((max_y, max_x))
fx = fy = None
active = None
locked = False
sx = sy = None
word, conf = None, CONF        # no active command until the user types one

win_name = "fovea (live)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

vw = None
if RECORD:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"fovea_interactive_{timestamp}.mp4"
    vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W_orig, H_orig))
    if not vw.isOpened():
        print("WARNING: couldn't open video writer, continuing without recording")
        vw = None
    else:
        print(f"Recording to          : {out_name}")


def to_bgr(m, cmap=cv2.COLORMAP_JET):
    m = m.astype(float)
    lo, hi = m.min(), m.max()
    if hi > lo:
        m = (m - lo) / (hi - lo) * 255
    return cv2.applyColorMap(np.clip(m, 0, 255).astype(np.uint8), cmap)


def drain_commands():
    """Apply every command that's arrived since the last frame. Returns False on quit."""
    global word, conf, locked, sx, sy, MODE, M, fx, fy
    while True:
        try:
            item = cmd_queue.get_nowait()
        except queue.Empty:
            return True
        cmd = item if isinstance(item, dict) else parse_command(item, DIRS)
        if cmd is None:
            continue
        if cmd["type"] == "quit":
            return False
        elif cmd["type"] == "word":
            word, conf = cmd["word"], cmd["conf"]
            print(f"  -> command set: '{word}' (conf={conf})")
        elif cmd["type"] == "stop":
            word = None
            print("  -> command cleared, holding position")
        elif cmd["type"] == "reset":
            M[:] = 0.0
            fx = fy = None
            locked = False
            print("  -> membrane reset; re-fixating on next frame's salmax")
        elif cmd["type"] == "mode":
            MODE = cmd["mode"]
            print(f"  -> mode set: {MODE}")
        elif cmd["type"] == "unknown":
            print(f"  ?? unrecognised: {cmd['raw']!r}")
    return True


# ---------------- main loop ----------------
print("\n--- live. type commands below. ---\n")
count = 0
k = 0
running = True
try:
    while running:
        m = frame_idx == k
        k = (k + 1) % n_frames if LOOP_PLAYBACK else k + 1
        if k >= n_frames and not LOOP_PLAYBACK:
            print("Clip finished (LOOP_PLAYBACK=False). Waiting for 'quit'...")
            running = drain_commands()
            key = cv2.waitKey(200) & 0xFF
            if key == ord('q'):
                running = False
            continue
        if not m.any():
            continue

        running = drain_commands()
        if not running:
            break

        xa = (ev_x[m] // DOWNSAMPLE).clip(0, max_x - 1)
        ya = (ev_y[m] // DOWNSAMPLE).clip(0, max_y - 1)
        window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
        window[0, ya, xa] = 255.0

        with torch.no_grad():
            saliency, salmax = run_attention(window, net, device, resolution,
                                             ATTENTION_PARAMS['num_pyr'])
        saliency = np.asarray(saliency)
        if np.isnan(saliency).any() or saliency.max() == saliency.min():
            continue

        if fx is None:
            fy, fx = float(salmax[0]), float(salmax[1])
            sx, sy = fx, fy

        foc = np.exp(-((X - fx) ** 2 + (Y - fy) ** 2) / (2 * READOUT_R ** 2))
        M = LEAK * M + (1.0 - LEAK) * saliency * (1.0 + BOOST * foc)

        if word != active:
            locked = False
            sx, sy = fx, fy
        if (not locked) and word in DIRS and conf >= THRESHOLD:
            psi = DIRS[word]
            if MODE == "pan":
                fx = float(np.clip(fx + STEP * np.cos(psi), 0, max_x - 1))
                fy = float(np.clip(fy + STEP * np.sin(psi), 0, max_y - 1))
            elif word != active:
                fx = float(np.clip(fx + SACCADE_JUMP * np.cos(psi), 0, max_x - 1))
                fy = float(np.clip(fy + SACCADE_JUMP * np.sin(psi), 0, max_y - 1))
        active = word

        zone = (X - fx) ** 2 + (Y - fy) ** 2 <= READOUT_R ** 2
        if zone.any():
            ay, ax = np.unravel_index(int(np.argmax(np.where(zone, M, -np.inf))), M.shape)
        else:
            ay, ax = int(round(fy)), int(round(fx))

        travel = np.hypot(fx - sx, fy - sy) if sx is not None else 0.0
        if (not locked) and travel >= MIN_TRAVEL:
            zone_mean = float(M[zone].mean()) if zone.any() else 0.0
            if zone_mean > 0 and M[ay, ax] >= CAP_RATIO * zone_mean:
                fx, fy = float(ax), float(ay)
                locked = True

        if DEBUG:
            print(f"win {count:5d} | {'LOCK' if locked else 'pan ':4} | "
                  f"cmd={str(word):6} | fovea=({int(fx)},{int(fy)}) | "
                  f"attended=({ax},{ay}) | travel={travel:5.1f} | M@att={M[ay, ax]:.1f}",
                  end="\r")

        ds = DOWNSAMPLE
        p = cv2.resize(to_bgr(M), (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        cv2.drawMarker(p, (int(fx * ds), int(fy * ds)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(p, (int(ax * ds), int(ay * ds)), 11, (255, 255, 255), 3)
        label = f"'{word}' [{MODE}] {'LOCK' if locked else 'pan'}" if word else f"(no command) [{MODE}]"
        cv2.putText(p, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if locked else (255, 255, 255), 2)
        cv2.imshow(win_name, p)
        if vw is not None:
            vw.write(p)
        count += 1

        key = cv2.waitKey(max(1, PLAYBACK_MS)) & 0xFF
        if key == ord('q'):
            running = False

finally:
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()
    print(f"\n\nSession ended. Frames shown: {count}" +
          (f"  |  saved to '{out_name}'" if vw is not None else ""))