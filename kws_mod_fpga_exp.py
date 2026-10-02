"""
Synthetic video + spoken commands from the FPGA keyword spotter, scored against
the clip's ground-truth mask.

Commands arrive from two sources at once, both pushing the same dicts into
`cmd_queue`, so the main loop does not care which one spoke:

  1. the keyboard (stdin_reader)  -- always on, for debugging
  2. the FPGA keyword spotter (KWSSource) -- when kws_backend is "frontpanel"
     (live board) or "replay" (recorded CSV)

Typed commands persist until cleared. Spoken commands expire after kws_ttl_s so
a word releases on its own; "stop" still clears explicitly.

Commands (typed in the terminal, Enter to submit):
    right / left / up / down
    stop  (or: none, clear)      release the active command, freeze in place
    reset                        zero the membrane and re-fixate on next salmax
    mode pan   / mode saccade    switch panning mode live
    quit / exit / q              end the session (also: 'q' in the video window)

Requires a local display for cv2.imshow

Every accepted command produces exactly one row, carrying its confidence and
source. A row is written when the NEXT command arrives (and the last one at
quit), so a command that failed to move the fovea still gets a row. At the end
the run is scored against the mask and written to results/{trial_id}.csv.
"""

import csv
import os
import queue
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import torch

from visual_attention.helpers_visual_att import initialise_attention, run_attention
from command_parser import parse_command
from kws.source import KWSSource, make_backend

# ============================ EXPERIMENT ============================
linguistic = 1      # 0 = written (typed), 1 = spoken (FPGA keyword spotter)
visual     = 0      # 0 = simulated events, 1 = camera       [0 for this script]
trial      = 5      # 1-5, one per person
batch      = 10     # 1-6, the stimulus categories

npy_path  = "data/9_objects_color_bg_jitter_346x260.npy"  # the event stream to run
mask_path = "stimuli/ground_truth_masks/9_objects_color_bg_jitter_346x260.mask.npy"   # the mask to score against

# Names of the synthetic objects, in mask id order (top row left to right, then down)

# names_list = ["top-left", "top-right", "bottom-left", "bottom-right"] # 4 circles
# names_list = ["top-left", "top-mid", "top-right", "bottom-left", "bottom-mid", "bottom-right"] # 6 circles
# names_list = ["apple", "bottle", "star", "heart", "diamond", "mushroom"] # 6 objects
names_list = ["apple", "bottle", "star", "heart", "diamond", "mushroom", "moon", "tree", "mug"] # 9 objects

command_limit = 10  # commands per trial (the starting fixation is not a command)
results_dir = "results"

# trial_id digits, most significant first: linguistic | visual | trial | batch
trial_id = f"{linguistic}{visual}{trial}{batch}"
# ====================================================================

# keyword spotter. Follows `linguistic` unless you override it here.
kws_backend = "frontpanel" if linguistic == 1 else "off"   # "frontpanel" | "replay" | "off"
kws_bitfile = "bitstreams/ok_top_wrapper_newest.bit"       # 32-channel parallel build
kws_serial = ""                  # "" = first board found
kws_replay_path = "results/replay_kws.csv"
kws_replay_speed = 1.0
kws_accept_conf = 200            # raw 0-255; Piotr's live value. Lower if words are missed
kws_unknown_penalty = 50         # subtracted from 'unknown' before ranking (Piotr)
kws_refractory_ms = 800          # same word inside this window counts once
kws_resync_every = 10            # disarm/reset/arm every N batches, between words only
kws_poll_ms = 5
kws_ttl_s = 3.0                  # spoken command releases itself after this long

# config
col_x, col_y, col_p, col_t = 0, 1, 2, 3
time_scale = 1e-3
window_ms = 100
downsample = 2
mask_ds = 2                      # the mask is saved at this factor by the generator

attention_params = {
    'size_krn': 16, 'r0': 8, 'rho': 0.015, 'theta': np.pi * 3 / 2,
    'thetas': np.arange(0, 2 * np.pi, np.pi / 4), 'thick': 12,
    'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
    'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
}

default_conf = 1.0          # confidence applied to typed commands only
threshold = 0.5             # loop-level gate; kws_accept_conf gates at the source

# membrane/fovea-pan controller
leak = 0.5
step = 8.0
saccade_jump = 60.0
readout_r = 25.0
boost = 2.0
cap_ratio = 1.5
min_travel = 50.0
default_mode = "pan"        # "pan" or "saccade"; changeable live via "mode <x>"

snap_factor = 2.0
snap = None

loop_playback = True        # replay the clip forever so the session doesn't just end
record = True               # also save an .mp4 of the session alongside the live view
playback_ms = window_ms     # cv2.waitKey delay: paces playback and pumps the GUI
save_debug_frame = False    # True -> dump one event window + membrane, then stop

debug = True
dirs = {"right": 0.0, "down": np.pi / 2, "left": np.pi, "up": 3 * np.pi / 2}
win_name = "fovea (live)"


# ======================= Ground truth and scoring =======================

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


def object_near(mask, cents, point, snap_px):
    """Object under `point`, else the nearest centroid within `snap_px`.

    The exact pixel is not enough: the fovea locks onto the membrane peak, which
    routinely sits a few px OUTSIDE the footprint. Read exactly, that is
    "background", and it costs two rows -- the landing is wrong, and the next
    command's oracle then stops excluding the object the fovea is really on.
    """
    x, y = int(round(point[0])), int(round(point[1]))
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
        return int(mask[y, x])
    best, best_d = None, None
    for oid, (cx, cy) in cents.items():
        d = (cx - point[0]) ** 2 + (cy - point[1]) ** 2
        if best_d is None or d < best_d:
            best, best_d = oid, d
    return best if (best is not None and best_d <= snap_px ** 2) else None


def oracle(cents, ref, direction, exclude=None):
    """Nearest object in `direction` from `ref`.

    `exclude` drops the object the fovea is ALREADY on, and the commanded axis
    must DOMINATE, so an object that is mostly off to the side does not count as
    "up".
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


print(f"trial {trial_id}  (linguistic={linguistic} visual={visual} "
      f"trial={trial} batch={batch})")
print(f"events : {npy_path}")
print(f"mask   : {mask_path}")
print(f"result : {os.path.join(results_dir, trial_id + '.csv')}   "
      f"(limit {command_limit} commands)")

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
        if line.strip().lower() in ("quit", "exit", "q"):
            return              # stop reading, so nothing holds stdin at shutdown


threading.Thread(target=stdin_reader, args=(cmd_queue,), daemon=True).start()

# keyword spotter thread
kws = None
if kws_backend != "off":
    try:
        kws_board = make_backend(kws_backend, bitfile=kws_bitfile, serial=kws_serial,
                                 path=kws_replay_path, speed=kws_replay_speed)
    except Exception as e:
        sys.exit(f"\nKEYWORD SPOTTER NOT AVAILABLE: {e}\n"
                 "Close the FrontPanel app, check the USB cable, or set "
                 "linguistic = 0 for a typed trial.\n")
    os.makedirs(results_dir, exist_ok=True)
    kws = KWSSource(kws_board, cmd_queue,
                    accept_conf=kws_accept_conf,
                    unknown_penalty=kws_unknown_penalty,
                    refractory_ms=kws_refractory_ms,
                    resync_every_batches=kws_resync_every,
                    poll_ms=kws_poll_ms,
                    log_path=(os.path.join(results_dir, f"{trial_id}_kws.csv")
                              if kws_backend == "frontpanel" else None))
    kws.start()
    print(f"Keyword spotter      : {kws_backend}  "
          f"(conf > {kws_accept_conf}, unknown -{kws_unknown_penalty}, "
          f"refractory {kws_refractory_ms} ms, TTL {kws_ttl_s}s)")
else:
    print("Keyword spotter      : off (typed commands only)")

#  load npy
device = torch.device("cpu")
print(f"Using device: {device}")

data = np.load(npy_path)
if data.dtype.names is not None:
    field_names = data.dtype.names

    def _pick(*c):
        for n in c:
            if n in field_names:
                return data[n]
        raise KeyError(c)

    ev_x = _pick('x').astype(int)
    ev_y = _pick('y').astype(int)
    ev_t = _pick('timestamp', 't', 'ts').astype(float)
else:
    ev_x = data[:, col_x].astype(int)
    ev_y = data[:, col_y].astype(int)
    ev_t = data[:, col_t].astype(float)

ev_t = ev_t * time_scale
o = np.argsort(ev_t)
ev_x, ev_y, ev_t = ev_x[o], ev_y[o], ev_t[o]

w_orig, h_orig = int(ev_x.max()) + 1, int(ev_y.max()) + 1
max_x, max_y = w_orig // downsample, h_orig // downsample
resolution = (max_y, max_x)
t0 = ev_t[0]
frame_idx = ((ev_t - t0) // window_ms).astype(int)
n_frames = int(frame_idx.max()) + 1

print(f"Loaded events        : {len(ev_t)}  span {ev_t[-1] - t0:.0f} ms")
print(f"Processing resolution: {max_x} x {max_y}")
print(f"Windows              : {n_frames} @ {window_ms} ms  (loop={loop_playback})")

# ground truth
if not os.path.exists(mask_path):
    sys.exit(f"\nMASK NOT FOUND: {mask_path}\n"
             "Generate the clip's mask first, then set mask_path.\n")
mask = np.load(mask_path)[::downsample // mask_ds, ::downsample // mask_ds]
if mask.shape != (max_y, max_x):
    sys.exit(f"\nMASK/RUN GRID MISMATCH: the mask is {mask.shape[1]}x{mask.shape[0]}, "
             f"this run is {max_x}x{max_y}.\n"
             f"Check downsample ({downsample}) against mask_ds ({mask_ds}).\n")
cents = mask_centroids(mask)
if len(cents) < 2:
    sys.exit(f"\nthe mask has {len(cents)} object(s) -- directions need at least 2.\n")
if len(names_list) != int(mask.max()):
    sys.exit(f"\n{len(names_list)} names given for {int(mask.max())} objects in the mask -- "
             "pick the matching names_list.\n")
names = names_list
radius = mask_object_radius(mask)
gaps = [((cents[a][0] - cents[b][0]) ** 2 + (cents[a][1] - cents[b][1]) ** 2) ** 0.5
        for a in cents for b in cents if a < b]
min_gap = min(gaps)
snap_px = snap if snap is not None else snap_factor * radius
if snap_px > 0.45 * min_gap:            # never close enough to claim a neighbour
    snap_px = 0.45 * min_gap
    print(f"note: snap clamped to {snap_px:.1f} px (objects are only {min_gap:.1f} px apart)")
print(f"Ground truth         : {len(cents)} objects, radius {radius:.1f} px, "
      f"snap {snap_px:.1f} px, spacing {min_gap:.1f} px")
for oid, (x, y) in cents.items():
    print(f"    id {oid} {names[oid-1]:<12} ({x:5.1f},{y:5.1f})")

# outlines of each object on the full-resolution display
mask_big = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
outlines = []
for oid in range(1, int(mask.max()) + 1):
    found = cv2.findContours((mask_big == oid).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines.extend(found[0] if len(found) == 2 else found[1])


def draw_outlines(img):
    cv2.drawContours(img, outlines, -1, (0, 255, 255), 1)
    return img


net = initialise_attention(device, attention_params)

grid_x, grid_y = np.meshgrid(np.arange(max_x), np.arange(max_y))
membrane = np.zeros((max_y, max_x))
fx = fy = None
active = None
locked = False
sx = sy = None
word, conf = None, default_conf  # no active command until one is typed or spoken
word_src = None                  # "typed" | "kws" -- only kws commands expire
word_expires = None              # monotonic deadline for the active kws command
cmd_start = None                 # fovea position when the current command was issued
rearm = False                    # a direction arrived -> re-arm the pan once, even
                                 # if it repeats the word already active
mode = default_mode

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

vw = None
out_name = None
if record:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"fovea_{trial_id}_{ts}.mp4"
    vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w_orig, h_orig))
    if not vw.isOpened():
        print("WARNING: couldn't open video writer, continuing without recording")
        vw = None
    else:
        print(f"Recording to         : {out_name}")


def to_bgr(m, cmap=cv2.COLORMAP_JET):
    m = m.astype(float)
    lo, hi = m.min(), m.max()
    if hi > lo:
        m = (m - lo) / (hi - lo) * 255
    return cv2.applyColorMap(np.clip(m, 0, 255).astype(np.uint8), cmap)


def close_command():
    """Record the outcome of the command that is currently active, if any.

    Called when a new command arrives, when a spoken command expires, and once at
    quit, so EVERY accepted command gets exactly one row -- including ones where
    the fovea never moved.
    """
    global cmd_start
    if word in dirs and fx is not None and cmd_start is not None:
        run_log.append({
            "direction": word, "k": 1,
            "ref_x": round(cmd_start[0], 1), "ref_y": round(cmd_start[1], 1),
            "fovea_x": round(fx, 1), "fovea_y": round(fy, 1),
            "conf": round(float(conf), 3), "src": word_src or "typed",
        })
        cmd_start = None
        return True
    cmd_start = None
    return False


def expire_command():
    """Release a spoken command once its TTL has elapsed. Typed commands persist."""
    global word, word_src, word_expires
    if word_src == "kws" and word_expires is not None and time.monotonic() >= word_expires:
        close_command()
        word = None
        word_src = None
        word_expires = None
        print("\n  -> spoken command expired, holding position")


def drain_commands():
    """Apply every command that's arrived since the last frame. Returns False on quit."""
    global word, conf, locked, sx, sy, mode, membrane, fx, fy, cmd_start, rearm
    global word_src, word_expires
    while True:
        try:
            item = cmd_queue.get_nowait()
        except queue.Empty:
            return True
        cmd = item if isinstance(item, dict) else parse_command(item, dirs)
        if cmd is None:
            continue
        src = cmd.get("src", "typed")
        if cmd["type"] == "quit":
            return False
        elif cmd["type"] == "word":
            pending = 1 if cmd_start is not None else 0
            if len(run_log) + pending >= command_limit:
                print(f"  -> command limit reached ({command_limit}); "
                      f"'{cmd['word']}' ignored. Type 'quit' to end the trial.")
                continue
            close_command()                # log the previous command's result
            word, conf = cmd["word"], cmd["conf"]
            word_src = src
            word_expires = (time.monotonic() + kws_ttl_s) if src == "kws" else None
            cmd_start = (fx, fy) if fx is not None else None
            rearm = True
            print(f"\n  -> command set: '{word}' ({src}, conf={conf:.2f})  "
                  f"[{len(run_log)} logged, {command_limit - len(run_log) - 1} left]")
        elif cmd["type"] == "stop":
            close_command()                # a held command ends here too
            word = None
            word_src = None
            word_expires = None
            print(f"\n  -> command cleared ({src}), holding position")
        elif cmd["type"] == "reset":
            close_command()
            membrane[:] = 0.0
            fx = fy = None
            locked = False
            word = None
            word_src = None
            word_expires = None
            print("  -> membrane reset; re-fixating on next frame's salmax")
        elif cmd["type"] == "mode":
            mode = cmd["mode"]
            print(f"  -> mode set: {mode}")
        elif cmd["type"] == "unknown":
            print(f"  ?? unrecognised: {cmd['raw']!r}")
    return True


def write_results():
    """Score the run against the mask and write results/{trial_id}.csv."""
    if not run_log:
        print("no commands logged - nothing to score")
        return
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{trial_id}.csv")

    def label(oid):
        if oid is None or oid == 0:
            return "background"
        return f"object {oid} ({names[oid - 1]})"

    print(f"\nground truth: {len(cents)} objects, radius {radius:.1f} px, snap {snap_px:.1f} px")
    print("begin visual attention.")
    first = object_near(mask, cents, (run_log[0]["ref_x"], run_log[0]["ref_y"]), snap_px)
    print(f"most salient point in {label(first)}.")

    rows, correct = [], 0
    for i, r in enumerate(run_log, start=1):
        ref = (r["ref_x"], r["ref_y"])
        fov = (r["fovea_x"], r["fovea_y"])
        start_id = object_near(mask, cents, ref, snap_px)
        target = oracle(cents, ref, r["direction"], exclude=start_id)
        landed = object_near(mask, cents, fov, snap_px)

        travel = ((fov[0] - ref[0]) ** 2 + (fov[1] - ref[1]) ** 2) ** 0.5
        # "did not move": stayed on the same object, or stayed on the same
        # patch of background (less than one snap radius of travel)
        if start_id is not None:
            stuck = landed == start_id
        else:
            stuck = landed is None and travel < snap_px

        if stuck:
            verdict, ok = "FAILED - did not move", False
        elif target is None:
            verdict, ok = "no object that way", landed is None
        elif landed == target:
            verdict, ok = "OK", True
        else:
            verdict, ok = f"WRONG - expected {label(target)}", False
        correct += ok

        nid, npx = min(((k, ((x - fov[0]) ** 2 + (y - fov[1]) ** 2) ** 0.5)
                        for k, (x, y) in cents.items()), key=lambda z: z[1])

        print(f"command: {r['direction']}  ({r.get('src', 'typed')})")
        print(f"now most salient point in {label(landed)}.   [{verdict}]")
        rows.append({"step": i, "command": r["direction"], "k": r["k"],
                     "ref_x": ref[0], "ref_y": ref[1],
                     "fovea_x": fov[0], "fovea_y": fov[1],
                     "start_id": start_id or 0, "start": label(start_id),
                     "expected_id": target or 0, "expected": label(target),
                     "landed_id": landed or 0, "landed": label(landed),
                     "verdict": verdict, "correct": int(ok),
                     "src": r.get("src", "typed"), "conf": r.get("conf", default_conf),
                     "nearest_id": nid, "nearest_px": round(npx, 1),
                     "travel_px": round(travel, 1),
                     "snap_px": round(snap_px, 1),
                     "trial_id": trial_id, "linguistic": linguistic,
                     "visual": visual, "trial": trial, "batch": batch,
                     "mask": os.path.basename(mask_path)})

    n = len(rows)
    acc_pct = round(100 * correct / n, 1)
    moved = sum(1 for r in rows if "did not move" not in r["verdict"])
    for r in rows:
        r["accuracy"] = acc_pct
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n{correct}/{n} commands correct  ({acc_pct:.1f}%)")
    print(f"fovea moved on {moved}/{n} commands")
    print(f"wrote {out_path}")


# main loop
print("\n--- speak or type commands below ---\n")
count = 0
k = 0
running = True
try:
    while running:
        m = frame_idx == k
        k = (k + 1) % n_frames if loop_playback else k + 1
        if k >= n_frames and not loop_playback:
            print("Clip finished (loop_playback=False). Waiting for 'quit'...")
            running = drain_commands()
            expire_command()
            key = cv2.waitKey(200) & 0xFF
            if key == ord('q'):
                running = False
            continue
        if not m.any():
            continue

        running = drain_commands()
        if not running:
            break
        expire_command()

        xa = (ev_x[m] // downsample).clip(0, max_x - 1)
        ya = (ev_y[m] // downsample).clip(0, max_y - 1)
        window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
        window[0, ya, xa] = 255.0

        with torch.no_grad():
            saliency, salmax = run_attention(window, net, device, resolution,
                                             attention_params['num_pyr'])
        saliency = np.asarray(saliency)
        if save_debug_frame and count == 5:          # one-shot dump, not every window
            np.save("dbg_events.npy", window[0].numpy())   # the event image you're watching
            np.save("dbg_M.npy", membrane)                 # the membrane, same frame
            print("\nsaved dbg_events.npy + dbg_M.npy")
        if np.isnan(saliency).any() or saliency.max() == saliency.min():
            continue

        if fx is None:
            fy, fx = float(salmax[0]), float(salmax[1])
            sx, sy = fx, fy
            if word in dirs and cmd_start is None:   # command arrived before first fixation
                cmd_start = (fx, fy)

        foc = np.exp(-((grid_x - fx) ** 2 + (grid_y - fy) ** 2) / (2 * readout_r ** 2))
        membrane = leak * membrane + (1.0 - leak) * saliency * (1.0 + boost * foc)

        # `rearm` is what lets a REPEATED word move again: without it this gate is
        # only entered when the word CHANGES, so saying 'left' twice leaves `locked`
        # set and the fovea frozen on the object it already found. It is set once
        # per ACCEPTED command -- the KWS source does the edge detection, so the
        # 100 Hz prediction stream never reaches here.
        if word != active or rearm:
            locked = False
            sx, sy = fx, fy
        if (not locked) and word in dirs and conf >= threshold:
            psi = dirs[word]
            if mode == "pan":
                fx = float(np.clip(fx + step * np.cos(psi), 0, max_x - 1))
                fy = float(np.clip(fy + step * np.sin(psi), 0, max_y - 1))
            elif word != active or rearm:     # saccade: one jump per accepted command
                fx = float(np.clip(fx + saccade_jump * np.cos(psi), 0, max_x - 1))
                fy = float(np.clip(fy + saccade_jump * np.sin(psi), 0, max_y - 1))
        rearm = False
        active = word

        zone = (grid_x - fx) ** 2 + (grid_y - fy) ** 2 <= readout_r ** 2
        if zone.any():
            ay, ax = np.unravel_index(int(np.argmax(np.where(zone, membrane, -np.inf))),
                                      membrane.shape)
        else:
            ay, ax = int(round(fy)), int(round(fx))

        travel = np.hypot(fx - sx, fy - sy) if sx is not None else 0.0
        if (not locked) and travel >= min_travel:
            zone_mean = float(membrane[zone].mean()) if zone.any() else 0.0
            if zone_mean > 0 and membrane[ay, ax] >= cap_ratio * zone_mean:
                fx, fy = float(ax), float(ay)
                locked = True

        if debug:
            line = (f"win {count:5d} | {'LOCK' if locked else 'pan ':4} | "
                    f"cmd={str(word):6} | fovea=({int(fx)},{int(fy)}) | "
                    f"attended=({ax},{ay}) | travel={travel:5.1f} | "
                    f"M@att={membrane[ay, ax]:.1f}")
            print(line.ljust(110), end="\r")   # pad: \r alone leaves the old tail behind

        ds = downsample
        p = cv2.resize(to_bgr(membrane), (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        draw_outlines(p)
        cv2.circle(p, (int(ax * ds), int(ay * ds)), 11, (255, 255, 255), 3)
        if kws is None:
            heard = ""
        elif kws.error is not None or not kws.is_alive():
            heard = " kws DOWN"
        else:
            heard = " kws"
        if word:
            tag = f" {conf:.2f}" if word_src == "kws" else ""
            label = (f"{trial_id} '{word}'{tag} ({word_src}) [{mode}] "
                     f"{'LOCK' if locked else 'pan'} {len(run_log)}/{command_limit}{heard}")
        else:
            label = (f"{trial_id} (no command) [{mode}] "
                     f"{len(run_log)}/{command_limit}{heard}")
        cv2.putText(p, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if locked else (255, 255, 255), 2)
        cv2.imshow(win_name, p)
        if vw is not None:
            vw.write(p)
        count += 1

        key = cv2.waitKey(max(1, playback_ms)) & 0xFF
        if key == ord('q'):
            running = False

finally:
    close_command()                 # flush the command that was still active
    if kws is not None:
        kws.close()                 # disarms the board, writes results/{trial_id}_kws.csv
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()
    write_results()
    if run_log:
        spoken = sum(1 for r in run_log if r.get("src") == "kws")
        print(f"commands: {spoken} spoken, {len(run_log) - spoken} typed")
    print(f"\nSession ended. Frames shown: {count}" +
          (f"  |  saved to '{out_name}'" if vw is not None else ""))