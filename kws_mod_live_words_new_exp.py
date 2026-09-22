"""
Synthetic video + live typed commands.

Set the categories in the EXPERIMENT block below, run, give the commands, quit.
The script scores the run itself and writes ONE file: results/{trial_id}.csv

Commands (typed in the terminal, Enter to submit):
    right / left / up / down 
    stop  (or: none, clear)      release the active command, freeze in place
    reset                        zero the membrane and re-fixate on next salmax
    mode pan   / mode saccade    switch panning mode live
    quit / exit / q              end the session (also: 'q' in the video window)

Requires a local display for cv2.imshow
"""

import csv
import os
import queue
import sys
import threading
from datetime import datetime

import cv2
import numpy as np
import torch

from visual_attention.helpers_visual_att import initialise_attention, run_attention
from command_parser import parse_command

# ============================ EXPERIMENT ============================
# Set these four, plus the two paths and names of the synthetic objectsbefore running. 
LINGUISTIC = 0      # 0 = written (typed), 1 = spoken        [0 for this script]
VISUAL     = 0      # 0 = simulated events, 1 = camera       [0 for this script]
TRIAL      = 5      # 1-5
BATCH      = 6      # 1-6, the stimulus categories

NPY_PATH  = "data/6_objects_color_bg_346x260.npy"  # the event stream to run
MASK_PATH = "stimuli/ground_truth_masks/6_objects_color_bg_346x260.mask.npy"

# Names of the synthetic objects
# NAMES = ["top-left", "top-mid", "top-right", "bottom-left", "bottom-mid", "bottom-right"]
NAMES = ["apple", "bottle", "star", "heart", "diamond", "mushroom"]

COMMAND_LIMIT = 10  # commands per trial (the starting fixation is not a command)
RESULTS_DIR = "results"

# trial_id digits, most significant first: linguistic | visual | trial | batch
TRIAL_ID = f"{LINGUISTIC}{VISUAL}{TRIAL}{BATCH}"
# ====================================================================

#  config
COL_X, COL_Y, COL_P, COL_T = 0, 1, 2, 3
TIME_SCALE = 1e-3
WINDOW_MS = 100
DOWNSAMPLE = 2

# DOWNSAMPLE to reduce the event image. The mask is downsampled again to match the run.
MASK_DS = 2


# SNAP overrides the factor with a fixed px value, meaning the fovea can land outside the object and inside the gaussian blob and still be scored as correct
SNAP_FACTOR = 2.0
SNAP = None

ATTENTION_PARAMS = {
    'size_krn': 16, 'r0': 8, 'rho': 0.015, 'theta': np.pi * 3 / 2,
    'thetas': np.arange(0, 2 * np.pi, np.pi / 4), 'thick': 12,
    'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
    'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
}

CONF = 1.0 # confidence applied for the typed commands only
THRESHOLD = 0.6 # threshold applied to the confidence of the typed commands

LEAK = 0.5 # membrane leak factor (0-1) - 0 = no leak, 1 = full leak
STEP = 8.0
SACCADE_JUMP = 60.0
READOUT_R = 25.0
BOOST = 2.0
CAP_RATIO = 1.5
MIN_TRAVEL = 50.0
MODE = "pan"                # "pan" or "saccade" — changeable live via "mode <x>"

LOOP_PLAYBACK = True        # replay the clip forever so the session doesn't just end
RECORD = True                # also save an .mp4 of the session alongside the live view
PLAYBACK_MS = WINDOW_MS      # cv2.waitKey delay: paces playback AND pumps the GUI
SAVE_DEBUG_FRAME = False     # True -> dump one event window + membrane, then stop

DEBUG = True
DIRS = {"right": 0.0, "down": np.pi / 2, "left": np.pi, "up": 3 * np.pi / 2}

print(f"trial {TRIAL_ID}  (linguistic={LINGUISTIC} visual={VISUAL} "
      f"trial={TRIAL} batch={BATCH})")
print(f"events : {NPY_PATH}")
print(f"mask   : {MASK_PATH}")
print(f"result : {os.path.join(RESULTS_DIR, TRIAL_ID + '.csv')}   "
      f"(limit {COMMAND_LIMIT} commands)")

# stdin reader thread
cmd_queue = queue.Queue()
run_log = []                   # one row per command, scored at the end


def stdin_reader(q):
    """Runs in a background thread. input() blocks THIS thread, never the video loop."""
    print("Type a command and press Enter (right/left/up/down/stop/reset/ "
          "mode pan|saccade / quit).")
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            q.put({"type": "quit"})
            return
        q.put(line)


threading.Thread(target=stdin_reader, args=(cmd_queue,), daemon=True).start()

#  load npy
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

# ============================== SCORING ==============================
# Self-contained on purpose: no helper module is imported, so no stale copy of
# one can change the numbers.

def mask_centroids(mask):
    """{id: (x, y)} -- mean pixel position of each object region."""
    out = {}
    for oid in range(1, int(mask.max()) + 1):
        ys, xs = np.where(mask == oid)
        if len(xs):
            out[oid] = (float(xs.mean()), float(ys.mean()))
    return out


def mask_object_radius(mask):
    """Median equivalent radius of the objects, in grid px."""
    rs = []
    for oid in range(1, int(mask.max()) + 1):
        area = int((mask == oid).sum())
        if area:
            rs.append((area / np.pi) ** 0.5)
    return float(np.median(rs)) if rs else 0.0


def object_near(mask, cents, point, snap):
    """Object under `point`, else the nearest centroid within `snap` px.

    The exact pixel is not enough. The fovea locks onto the membrane peak, which
    routinely sits a pixel or two OUTSIDE the object's footprint -- e.g. 16.3 px
    from a centre whose radius is 15.2. Read exactly, that is "background", and
    it costs two rows: the landing is wrong, and the next command's oracle then
    stops excluding the object the fovea is really sitting on.
    """
    x, y = int(round(point[0])), int(round(point[1]))
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
        return int(mask[y, x])
    best, best_d = None, None
    for oid, (cx, cy) in cents.items():
        d = (cx - point[0]) ** 2 + (cy - point[1]) ** 2
        if best_d is None or d < best_d:
            best, best_d = oid, d
    return best if (best is not None and best_d <= snap ** 2) else None


def oracle(cents, ref, direction, exclude=None):
    """Nearest object in `direction` from `ref`.

    Two rules keep this honest on a grid:
      * `exclude` drops the object the fovea is ALREADY on, otherwise a fovea
        parked a pixel past its centroid sees it as the nearest object that way.
      * the commanded axis must DOMINATE, so a same-row neighbour a fraction of
        a pixel higher cannot masquerade as being "up".
    """
    rx, ry = ref
    keep = {"left":  lambda x, y: x < rx,
            "right": lambda x, y: x > rx,
            "up":    lambda x, y: y < ry,
            "down":  lambda x, y: y > ry}[direction]
    vertical = direction in ("up", "down")

    def dominant(x, y):
        return abs(y - ry) > abs(x - rx) if vertical else abs(x - rx) > abs(y - ry)

    cands = [(oid, x, y) for oid, (x, y) in cents.items()
             if oid != exclude and keep(x, y) and dominant(x, y)]
    cands.sort(key=lambda t: (t[1] - rx) ** 2 + (t[2] - ry) ** 2)
    return cands[0][0] if cands else None


# ground truth
MASK = CENTS = None
RADIUS = SNAP_PX = 0.0
if not os.path.exists(MASK_PATH):
    print(f"\n*** MASK NOT FOUND: {MASK_PATH}\n"
          f"*** the session will run and save raw commands, but not be scored.\n")
else:
    MASK = np.load(MASK_PATH)[::DOWNSAMPLE // MASK_DS, ::DOWNSAMPLE // MASK_DS]
    CENTS = mask_centroids(MASK)
    RADIUS = mask_object_radius(MASK)
    gaps = [((CENTS[a][0] - CENTS[b][0]) ** 2 + (CENTS[a][1] - CENTS[b][1]) ** 2) ** 0.5
            for a in CENTS for b in CENTS if a < b]
    min_gap = min(gaps) if gaps else float("inf")
    SNAP_PX = SNAP if SNAP is not None else SNAP_FACTOR * RADIUS
    if SNAP_PX > 0.45 * min_gap:          # never close enough to claim a neighbour
        SNAP_PX = 0.45 * min_gap
        print(f"note: snap clamped to {SNAP_PX:.1f} px (objects are only "
              f"{min_gap:.1f} px apart)")
    print(f"Ground truth         : {MASK.shape[1]}x{MASK.shape[0]}, {len(CENTS)} objects, "
          f"radius {RADIUS:.1f} px, snap {SNAP_PX:.1f} px, spacing {min_gap:.1f} px")
    if MASK.shape != (max_y, max_x):
        print(f"*** MASK/RUN GRID MISMATCH: mask is {MASK.shape[1]}x{MASK.shape[0]}, "
              f"the run is {max_x}x{max_y}. Check DOWNSAMPLE vs MASK_DS.")
    else:
        # do the events actually sit on the mask's objects? a stale or wrongly
        # scaled mask shows up here as most events landing outside it.
        acc = np.zeros((max_y, max_x), bool)
        for _k in range(min(5, n_frames)):
            _m = frame_idx == _k
            acc[(ev_y[_m] // DOWNSAMPLE).clip(0, max_y - 1),
                (ev_x[_m] // DOWNSAMPLE).clip(0, max_x - 1)] = True
        inside = int((acc & (MASK > 0)).sum())
        total = int(acc.sum())
        if total and inside / total < 0.4:
            print(f"*** only {100*inside/total:.0f}% of event pixels fall on the mask's "
                  f"objects - the mask may not match this clip")

net = initialise_attention(device, ATTENTION_PARAMS)

X, Y = np.meshgrid(np.arange(max_x), np.arange(max_y))
M = np.zeros((max_y, max_x))
fx = fy = None
active = None
locked = False
sx = sy = None
word, conf = None, CONF        # no active command until the user types one
cmd_start = None               # fovea position when the current command was issued
rearm = False                  # a direction was typed -> re-arm the pan once, even
                               # if it repeats the word already active

win_name = "fovea (live)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

vw = None
out_name = None
if RECORD:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"fovea_{TRIAL_ID}_{timestamp}.mp4"
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


def close_command():
    """Record the outcome of the command that is currently active, if any.

    Called when a new command arrives and once at quit, so EVERY typed command
    gets exactly one row -- including ones where the fovea never moved.
    """
    global word, cmd_start
    if word in DIRS and fx is not None and cmd_start is not None:
        run_log.append({
            "direction": word, "k": 1,
            "ref_x": round(cmd_start[0], 1), "ref_y": round(cmd_start[1], 1),
            "fovea_x": round(fx, 1), "fovea_y": round(fy, 1),
        })
        cmd_start = None
        return True
    cmd_start = None
    return False


def drain_commands():
    """Apply every command that's arrived since the last frame. Returns False on quit."""
    global word, conf, locked, sx, sy, MODE, M, fx, fy, cmd_start, rearm
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
            pending = 1 if cmd_start is not None else 0
            if len(run_log) + pending >= COMMAND_LIMIT:
                print(f"  -> command limit reached ({COMMAND_LIMIT}); "
                      f"'{cmd['word']}' ignored. Type 'quit' to end the trial.")
                continue
            close_command()                # log the previous command's result
            word, conf = cmd["word"], cmd["conf"]
            cmd_start = (fx, fy) if fx is not None else None
            rearm = True
            print(f"  -> command set: '{word}' (conf={conf})  "
                  f"[{len(run_log)} logged, {COMMAND_LIMIT - len(run_log) - 1} left]")
        elif cmd["type"] == "stop":
            close_command()                # a held command ends here too
            word = None
            print("  -> command cleared, holding position")
        elif cmd["type"] == "reset":
            close_command()
            M[:] = 0.0
            fx = fy = None
            locked = False
            word = None
            print("  -> membrane reset; re-fixating on next frame's salmax")
        elif cmd["type"] == "mode":
            MODE = cmd["mode"]
            print(f"  -> mode set: {MODE}")
        elif cmd["type"] == "unknown":
            print(f"  ?? unrecognised: {cmd['raw']!r}")
    return True


def write_results():
    """Score the run against the mask and write results/{TRIAL_ID}.csv."""
    if not run_log:
        print("no commands logged - nothing to score")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{TRIAL_ID}.csv")

    if MASK is None:
        # never lose a session: keep the raw commands so it can be scored later
        raw = os.path.join(RESULTS_DIR, f"{TRIAL_ID}_unscored.csv")
        with open(raw, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(run_log[0].keys()))
            w.writeheader(); w.writerows(run_log)
        print(f"MASK NOT FOUND: {MASK_PATH}\nwrote {raw} (raw commands, unscored)")
        return

    mask, cents, radius, snap = MASK, CENTS, RADIUS, SNAP_PX

    def label(oid):
        if oid is None or oid == 0:
            return "background"
        return f"object {oid}" + (f" ({NAMES[oid - 1]})" if NAMES else "")

    print(f"\nmask {mask.shape[1]}x{mask.shape[0]}, {len(cents)} objects, "
          f"object radius {radius:.1f} px, snap {snap:.1f} px")
    print("begin visual attention.")
    first = object_near(mask, cents, (run_log[0]["ref_x"], run_log[0]["ref_y"]), snap)
    print(f"most salient point in {label(first)}.")

    rows, correct = [], 0
    for i, r in enumerate(run_log, start=1):
        ref = (r["ref_x"], r["ref_y"])
        fov = (r["fovea_x"], r["fovea_y"])
        start_id = object_near(mask, cents, ref, snap)
        target = oracle(cents, ref, r["direction"], exclude=start_id)
        landed = object_near(mask, cents, fov, snap)

        if landed == start_id:
            verdict, ok = "FAILED - did not move", False
        elif target is None:
            verdict, ok = "no object that way", landed is None
        elif landed == target:
            verdict, ok = "OK", True
        else:
            verdict, ok = f"WRONG - expected {label(target)}", False
        correct += ok

        # distance to the nearest centroid: the number behind every verdict
        nid, npx = min(((k, ((x - fov[0]) ** 2 + (y - fov[1]) ** 2) ** 0.5)
                        for k, (x, y) in cents.items()), key=lambda z: z[1])

        print(f"command: {r['direction']}")
        print(f"now most salient point in {label(landed)}.   [{verdict}]")
        rows.append({"step": i, "command": r["direction"], "k": r["k"],
                     # "ref_x": ref[0], "ref_y": ref[1],
                     # "fovea_x": fov[0], "fovea_y": fov[1],
                     "start_id": start_id or 0, "start": label(start_id),
                     "expected_id": target or 0, "expected": label(target),
                     "landed_id": landed or 0, # "landed": label(landed),
                     "verdict": verdict, # "correct": int(ok),
                     # "nearest_id": nid, "nearest_px": round(npx, 1),
                     # "snap_px": round(snap, 1),
                     "trial_id": TRIAL_ID, "linguistic": LINGUISTIC,
                     "visual": VISUAL, "trial": TRIAL, "batch": BATCH})

    n = len(rows)
    acc = round(100 * correct / n, 1)
    moved = sum(1 for r in rows if "did not move" not in r["verdict"])
    for r in rows:
        r["accuracy"] = acc
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n{correct}/{n} commands correct  ({acc:.1f}%)")
    print(f"fovea moved on {moved}/{n} commands")
    print(f"wrote {out_path}")


# main loop
print("\n--- Type commands below ---\n")
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
        if SAVE_DEBUG_FRAME and count == 5:          # one-shot dump, not every window
            np.save("dbg_events.npy", window[0].numpy())   # the event image you're watching
            np.save("dbg_M.npy", M)                        # the membrane, same frame
            print("\nsaved dbg_events.npy + dbg_M.npy")
        if np.isnan(saliency).any() or saliency.max() == saliency.min():
            continue

        if fx is None:
            fy, fx = float(salmax[0]), float(salmax[1])
            sx, sy = fx, fy
            if word in DIRS and cmd_start is None:   # command typed before first fixation
                cmd_start = (fx, fy)

        foc = np.exp(-((X - fx) ** 2 + (Y - fy) ** 2) / (2 * READOUT_R ** 2))
        M = LEAK * M + (1.0 - LEAK) * saliency * (1.0 + BOOST * foc)

        # rearm is what lets a repeated word move again: without it this gate is
        # only entered when the word changes, so typing 'left' twice leaves `locked`
        # set and the fovea frozen on the object it already found.
        if word != active or rearm:
            locked = False
            sx, sy = fx, fy
        if (not locked) and word in DIRS and conf >= THRESHOLD:
            psi = DIRS[word]
            if MODE == "pan":
                fx = float(np.clip(fx + STEP * np.cos(psi), 0, max_x - 1))
                fy = float(np.clip(fy + STEP * np.sin(psi), 0, max_y - 1))
            elif word != active or rearm:     # saccade: one jump per typed command
                fx = float(np.clip(fx + SACCADE_JUMP * np.cos(psi), 0, max_x - 1))
                fy = float(np.clip(fy + SACCADE_JUMP * np.sin(psi), 0, max_y - 1))
        rearm = False
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
            line = (f"win {count:5d} | {'LOCK' if locked else 'pan ':4} | "
                    f"cmd={str(word):6} | fovea=({int(fx)},{int(fy)}) | "
                    f"attended=({ax},{ay}) | travel={travel:5.1f} | "
                    f"M@att={M[ay, ax]:.1f}")
            print(line.ljust(110), end="\r")   

        ds = DOWNSAMPLE
        p = cv2.resize(to_bgr(M), (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        # cv2.drawMarker(p, (int(fx * ds), int(fy * ds)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(p, (int(ax * ds), int(ay * ds)), 11, (255, 255, 255), 3)
        label = (f"{TRIAL_ID} '{word}' [{MODE}] {'LOCK' if locked else 'pan'} "
                 f"{len(run_log)}/{COMMAND_LIMIT}" if word
                 else f"{TRIAL_ID} (no command) [{MODE}] {len(run_log)}/{COMMAND_LIMIT}")
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
    close_command()                 # flush the command that was still active
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()
    write_results()
    print(f"\nSession ended. Frames shown: {count}" +
          (f"  |  saved to '{out_name}'" if vw is not None else ""))