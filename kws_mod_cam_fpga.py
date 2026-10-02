"""
Live DVS camera + spoken commands from the FPGA keyword spotter (and typed
commands, scored against a SAM ground-truth mask.
"""

import csv
import os
import queue
import sys
import threading
import time
from datetime import datetime, timedelta
 
import cv2
import numpy as np
import torch
import dv_processing as dv
 
from visual_attention.helpers_visual_att import initialise_attention, run_attention
from command_parser import parse_command
from kws import KWSSource, make_backend
 
# ============================ EXPERIMENT ============================
linguistic = 1      # 0 = written (typed), 1 = spoken (FPGA keyword spotter)
visual     = 1      # 0 = simulated events, 1 = camera       [1 for this script]
trial      = 4      # 1-5, one per person
batch      = 9      # 1-10, the stimulus categories
 
mask_path = "stimuli/ground_truth_masks/camera_6_objects_color_nobg_346x260.mask.npy" # ground truth
 
command_limit = 10  # commands per trial (the starting fixation is not a command)
results_dir = "results"
 
# trial_id digits, most significant first: linguistic | visual | trial | batch
trial_id = f"{linguistic}{visual}{trial}{batch}"
# ====================================================================

# keyword spotter 
# Follows `linguistic` unless you override it here.
kws_backend = "frontpanel" if linguistic == 1 else "off"
kws_bitfile = "bitstreams/ok_top_wrapper_newest.bit"   # 32-channel parallel build
kws_serial = ""                  # "" = first board found
kws_replay_path = "results/replay_kws.csv"
kws_accept_conf = 200            # raw 0-255; Piotr's live value. Lower if words are missed
kws_unknown_penalty = 50         # subtracted from 'unknown' before ranking (Piotr)
kws_refractory_ms = 800          # same word inside this window counts once
kws_resync_every = 10            # disarm/reset/arm every N batches, between words only
 
# config 
downsample = 2
window_ms = 100                 # slicer window
 
attention_params = {
    'size_krn': 16, 'r0': 7, 'rho': 0.015, 'theta': np.pi * 3 / 2,
    'thetas': np.arange(0, 2 * np.pi, np.pi / 4), 'thick': 12,
    'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
    'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
}
 
default_conf = 1.0
threshold = 0.80
 
# membrane/fovea-pan controller
leak = 0.5           # membrane decay: leaky integrator of saliency over time
step = 8.0           # px the fovea pans per window
saccade_jump = 60.0  # px the fovea jumps on a saccade command
readout_r = 25.0     # radius: focus neighbourhood and boost width
boost = 2.0          # how much the fovea's area is amplified
cap_ratio = 2.5      # how much the peak must exceed the zone mean to lock
min_travel = 50.0    # px the fovea must travel before it may lock
default_mode = "pan" # "pan" or "saccade"; changeable live via "mode <x>"
 
# first fixation: only fixate on a window that has a real peak
min_events_fix = 300     # events a window needs before the first fixation
fix_peak_ratio = 2.0     # the saliency peak must be this many times the map mean
follow_before_command = True   # until the first command, the fovea follows the saliency peak every window instead of freezing
noise_filter_us = 500 # it was 2000

snap_factor = 2.0
snap = None
 
record = True                   # also save an .mp4 of the session
show_events_panel = True        # left panel = raw events, right panel = fovea membrane
debug = True
 
dirs = {"right": 0.0, "down": np.pi / 2, "left": np.pi, "up": 3 * np.pi / 2}
win_name = "fovea (live camera)"
 
 
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
    "up". With real objects on a table: place them clearly left/right or clearly
    above/below each other, or a diagonal pair gives "no object that way".
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
 
 
def load_names(path, n):
    """Names from the .truth.csv written next to the mask, in id order."""
    csv_path = path.replace(".mask.npy", ".truth.csv")
    names = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                names[int(row["id"])] = row["name"]
    return [names.get(i, f"object {i}") for i in range(1, n + 1)]
 
 
print(f"trial {trial_id}  (linguistic={linguistic} visual={visual} "
      f"trial={trial} batch={batch})")
print(f"mask   : {mask_path}")
print(f"result : {os.path.join(results_dir, trial_id + '.csv')}   "
      f"(limit {command_limit} commands)")
 
# keyword spotter
kws = None
if kws_backend != "off":
    try:
        kws_board = make_backend(kws_backend, bitfile=kws_bitfile, serial=kws_serial,
                                 path=kws_replay_path)
    except Exception as e:
        sys.exit(f"\nKEYWORD SPOTTER NOT AVAILABLE: {e}\n"
                 "Close the FrontPanel app, check the USB cable, or set "
                 "linguistic = 0 for a typed trial.\n")
    print(f"Keyword spotter      : {kws_backend} ready "
          f"(conf > {kws_accept_conf}, unknown -{kws_unknown_penalty})")
else:
    kws_board = None
    print("Keyword spotter      : off (typed commands only)")

# camera 
device = torch.device("cpu")
print(f"Using device: {device}")
 
capture = dv.io.camera.open()
if not capture.isEventStreamAvailable():
    raise RuntimeError("Camera does not provide an event stream.")
 
w_orig, h_orig = capture.getEventResolution()
max_x = w_orig // downsample
max_y = h_orig // downsample
resolution = (max_y, max_x)
 
print(f"Camera resolution    : {w_orig} x {h_orig}")
print(f"Processing resolution: {max_x} x {max_y}  (downsample {downsample}x)")
print(f"Slicer window        : {window_ms} ms")
print(f"Noise filter         : " +
      (f"dv background activity, {noise_filter_us} us" if noise_filter_us else "off"))

noise_filter = None
if noise_filter_us:
    noise_filter = dv.noise.BackgroundActivityNoiseFilter(
        (w_orig, h_orig), backgroundActivityDuration=timedelta(microseconds=noise_filter_us))
 
# ground truth
if not os.path.exists(mask_path):
    sys.exit(f"\nMASK NOT FOUND: {mask_path}\n"
             "Build it first with the APS + SAM script, then set mask_path.\n")
mask = np.load(mask_path)
if mask.shape != (max_y, max_x):
    sys.exit(f"\nMASK/RUN GRID MISMATCH: the mask is {mask.shape[1]}x{mask.shape[0]}, "
             f"this run is {max_x}x{max_y}.\n"
             f"Rebuild the mask with --downsample {downsample}.\n")
cents = mask_centroids(mask)
if len(cents) < 2:
    sys.exit(f"\nthe mask has {len(cents)} object(s) -- directions need at least 2.\n")
names = load_names(mask_path, int(mask.max()))
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
 
 
# ---------------- stdin reader thread ----------------
cmd_queue = queue.Queue()
run_log = []                   # one row per command, scored at the end
 
 
def stdin_reader(q):
    """Runs in a background thread. input() blocks THIS thread, never the camera loop."""
    print("Type a command and press Enter (right / left / up / down / stop / reset / "
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

if kws_board is not None:
    os.makedirs(results_dir, exist_ok=True)
    kws = KWSSource(kws_board, cmd_queue,
                    accept_conf=kws_accept_conf,
                    unknown_penalty=kws_unknown_penalty,
                    refractory_ms=kws_refractory_ms,
                    resync_every_batches=kws_resync_every,
                    log_path=(os.path.join(results_dir, f"{trial_id}_kws.csv")
                              if kws_backend == "frontpanel" else None))
    kws.start()
 
net = initialise_attention(device, attention_params)
 
grid_x, grid_y = np.meshgrid(np.arange(max_x), np.arange(max_y))
 
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
 
# ---------------- video writer ----------------
vw = None
out_name = None
if record:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"fovea_{trial_id}_{ts}.mp4"
    panel_w = w_orig * (2 if show_events_panel else 1)
    vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (panel_w, h_orig))
    if not vw.isOpened():
        print("WARNING: couldn't open video writer, continuing without recording")
        vw = None
    else:
        print(f"Recording to         : {out_name}")
 
 
# ---------------- persistent state ----------------
class State:
    membrane = np.zeros((max_y, max_x))
    fx = None
    fy = None
    active = None
    locked = False
    sx = None
    sy = None
    word = None
    conf = default_conf
    src = None                  # "kws" | "typed" -- where the active command came from
    mode = default_mode
    count = 0
    running = True
    cmd_start = None            # fovea position when the current command was issued
    rearm = False               # a direction was typed -> re-arm the pan once, even
                                # if it repeats the word already active
 
 
state = State()
 
 
# ---------------- helpers ----------------
def to_bgr(m, cmap=cv2.COLORMAP_JET):
    m = m.astype(float)
    lo, hi = m.min(), m.max()
    if hi > lo:
        m = (m - lo) / (hi - lo) * 255
    return cv2.applyColorMap(np.clip(m, 0, 255).astype(np.uint8), cmap)
 
 
def add_label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img
 
 
def show_frame(frame):
    try:
        cv2.imshow(win_name, frame)
    except cv2.error:
        pass
 
 
def close_command(st):
    """Record the outcome of the command that is currently active, if any.
 
    Called when a new command arrives and once at quit, so EVERY typed command
    gets exactly one row -- including ones where the fovea never moved.
    """
    if st.word in dirs and st.fx is not None and st.cmd_start is not None:
        run_log.append({
            "direction": st.word, "k": 1,
            "ref_x": round(st.cmd_start[0], 1), "ref_y": round(st.cmd_start[1], 1),
            "fovea_x": round(st.fx, 1), "fovea_y": round(st.fy, 1),
            "conf": round(float(st.conf), 3), "src": st.src or "typed",
        })
        st.cmd_start = None
        return True
    st.cmd_start = None
    return False
 
 
def drain_commands(st):
    """Apply every command queued since the last call. Sets st.running=False on quit."""
    while True:
        try:
            item = cmd_queue.get_nowait()
        except queue.Empty:
            return
        cmd = item if isinstance(item, dict) else parse_command(item, dirs)
        if cmd is None:
            continue
        t = cmd["type"]
        src = cmd.get("src", "typed")
        if t == "quit":
            st.running = False
            return
        elif t == "word":
            pending = 1 if st.cmd_start is not None else 0
            if len(run_log) + pending >= command_limit:
                print(f"  -> command limit reached ({command_limit}); "
                      f"'{cmd['word']}' ignored. Type 'quit' to end the trial.")
                continue
            close_command(st)
            st.word, st.conf, st.src = cmd["word"], cmd["conf"], src
            st.cmd_start = (st.fx, st.fy) if st.fx is not None else None
            st.rearm = True
            print(f"\n  -> command set: '{st.word}' ({src}, conf={st.conf:.2f})  "
                  f"[{len(run_log)} logged, {command_limit - len(run_log) - 1} left]")
        elif t == "stop":
            close_command(st)
            st.word = None
            st.src = None
            print(f"\n  -> command cleared ({src}), holding position")
        elif t == "reset":
            close_command(st)
            st.membrane[:] = 0.0
            st.fx = st.fy = None
            st.locked = False
            st.word = None
            print("  -> membrane reset; re-fixating on next window's salmax")
        elif t == "mode":
            st.mode = cmd["mode"]
            print(f"  -> mode set: {st.mode}")
        elif t == "unknown":
            print(f"  ?? unrecognised: {cmd['raw']!r}")
 
 
# ---------------- slicer callback (one event window) ----------------
def slicing_callback(events: dv.EventStore):
    st = state
    if not st.running or events is None or len(events) == 0:
        return
 
    n_raw = len(events)
    if noise_filter is not None:
        noise_filter.accept(events)
        events = noise_filter.generateEvents()
        if len(events) == 0:
            return
    n_kept = len(events)

    ev = events.numpy()
    xa = (ev['x'].astype(int) // downsample).clip(0, max_x - 1)
    ya = (ev['y'].astype(int) // downsample).clip(0, max_y - 1)
    window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
    window[0, ya, xa] = 255.0
 
    with torch.no_grad():
        saliency, salmax = run_attention(window, net, device, resolution,
                                         attention_params['num_pyr'])
    saliency = np.asarray(saliency)
 
    # keep the window alive even when saliency is flat/NaN so you can see the events
    if np.isnan(saliency).any() or saliency.max() == saliency.min():
        ev_panel = cv2.resize(to_bgr(window[0].numpy(), cv2.COLORMAP_BONE),
                              (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        add_label(draw_outlines(ev_panel), "events (saliency flat/NaN)")
        frame = np.hstack([ev_panel, np.zeros_like(ev_panel)]) if show_events_panel else ev_panel
        show_frame(frame)
        return
 
    # is there a real peak in this window?
    py, px = int(salmax[0]), int(salmax[1])
    peak = float(saliency[py, px])
    sal_mean = float(saliency.mean())
    clear_peak = (n_raw >= min_events_fix and sal_mean > 0
                  and peak >= fix_peak_ratio * sal_mean)
 
    if st.fx is None:
        if not clear_peak:
            # no fixation from a weak window: show the events and wait
            ev_panel = cv2.resize(to_bgr(window[0].numpy(), cv2.COLORMAP_BONE),
                                  (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            add_label(draw_outlines(ev_panel),
                      f"waiting for a clear peak ({n_raw} ev, "
                      f"peak/mean {peak / sal_mean if sal_mean > 0 else 0:.1f})")
            frame = (np.hstack([ev_panel, np.zeros_like(ev_panel)])
                     if show_events_panel else ev_panel)
            show_frame(frame)
            return
        st.fy, st.fx = float(py), float(px)
        st.sx, st.sy = st.fx, st.fy
        print(f"\nfirst fixation at ({px},{py})  [{n_raw} events, "
              f"peak/mean {peak / sal_mean:.1f}]")
        if st.word in dirs and st.cmd_start is None:   # typed before the first fixation
            st.cmd_start = (st.fx, st.fy)
    elif (follow_before_command and st.word is None and not run_log
          and st.cmd_start is None and clear_peak):
        # free viewing before the first command: track the saliency peak
        st.fy, st.fx = float(py), float(px)
        st.sx, st.sy = st.fx, st.fy
 
    # decaying per-pixel membrane with foveal Gaussian boost
    foc = np.exp(-((grid_x - st.fx) ** 2 + (grid_y - st.fy) ** 2) / (2 * readout_r ** 2))
    st.membrane = leak * st.membrane + (1.0 - leak) * saliency * (1.0 + boost * foc)
 
    # `rearm` is what lets a repeated word move again: without it this gate is only
    # entered when the word changes, so typing 'left' twice leaves `locked` set and
    # the fovea frozen on the object it already found.
    if st.word != st.active or st.rearm:
        st.locked = False
        st.sx, st.sy = st.fx, st.fy
    if (not st.locked) and st.word in dirs and st.conf >= threshold:
        psi = dirs[st.word]
        if st.mode == "pan":
            st.fx = float(np.clip(st.fx + step * np.cos(psi), 0, max_x - 1))
            st.fy = float(np.clip(st.fy + step * np.sin(psi), 0, max_y - 1))
        elif st.word != st.active or st.rearm:     # saccade: one jump per typed command
            st.fx = float(np.clip(st.fx + saccade_jump * np.cos(psi), 0, max_x - 1))
            st.fy = float(np.clip(st.fy + saccade_jump * np.sin(psi), 0, max_y - 1))
    st.rearm = False
    st.active = st.word
 
    # readout: argmax of the membrane inside the foveal zone
    zone = (grid_x - st.fx) ** 2 + (grid_y - st.fy) ** 2 <= readout_r ** 2
    if zone.any():
        ay, ax = np.unravel_index(int(np.argmax(np.where(zone, st.membrane, -np.inf))),
                                  st.membrane.shape)
    else:
        ay, ax = int(round(st.fy)), int(round(st.fx))
 
    # capture / lock once we've travelled far enough onto a strong peak
    travel = np.hypot(st.fx - st.sx, st.fy - st.sy) if st.sx is not None else 0.0
    if (not st.locked) and travel >= min_travel:
        zone_mean = float(st.membrane[zone].mean()) if zone.any() else 0.0
        if zone_mean > 0 and st.membrane[ay, ax] >= cap_ratio * zone_mean:
            st.fx, st.fy = float(ax), float(ay)
            st.locked = True
 
    if debug:
        line = (f"win {st.count:5d} | {'LOCK' if st.locked else 'pan ':4} | "
                f"cmd={str(st.word):6} | fovea=({int(st.fx)},{int(st.fy)}) | "
                f"attended=({ax},{ay}) | travel={travel:5.1f} | "
                f"M@att={st.membrane[ay, ax]:.1f} | ev {n_raw}->{n_kept}")
        print(line.ljust(110), end="\r")   # pad: \r alone leaves the old tail behind
 
    # ---- render ----
    ds = downsample
    fovea_panel = cv2.resize(to_bgr(st.membrane), (w_orig, h_orig),
                             interpolation=cv2.INTER_LINEAR)
    draw_outlines(fovea_panel)
    # cv2.drawMarker(fovea_panel, (int(st.fx * ds), int(st.fy * ds)),
                   #(0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(fovea_panel, (int(ax * ds), int(ay * ds)), 11, (255, 255, 255), 3)
    if kws is None:
        heard = ""
    elif kws.error is not None or not kws.is_alive():
        heard = " kws DOWN"
    else:
        heard = " kws"
    label = (f"{trial_id} '{st.word}' ({st.src}) [{st.mode}] "
             f"{'LOCK' if st.locked else 'pan'} {len(run_log)}/{command_limit}{heard}"
             if st.word
             else f"{trial_id} (no command) [{st.mode}] "
                  f"{len(run_log)}/{command_limit}{heard}")
    add_label(fovea_panel, label, (0, 255, 0) if st.locked else (255, 255, 255))
 
    if show_events_panel:
        ev_panel = cv2.resize(to_bgr(window[0].numpy(), cv2.COLORMAP_BONE),
                              (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        kept = f"{n_raw} -> {n_kept} ev" if noise_filter is not None else f"{n_raw} ev"
        add_label(draw_outlines(ev_panel), f"events: {kept}  (yellow = ground truth)")
        frame = np.hstack([ev_panel, fovea_panel])
    else:
        frame = fovea_panel
 
    show_frame(frame)
    if vw is not None:
        vw.write(frame)
    st.count += 1
 
 
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
 
 
# ---------------- main loop ----------------
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(timedelta(milliseconds=window_ms), slicing_callback)
 
print("\n--- live. speak or type commands. 'q' in the window or 'quit' here to stop. ---\n")
try:
    while capture.isRunning() and state.running:
        drain_commands(state)
        if not state.running:
            break
 
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)
 
        key = cv2.waitKey(1) & 0xFF   # pumps the GUI + catches 'q'
        if key == ord('q'):
            state.running = False
 
finally:
    close_command(state)              # flush the command that was still active
    if kws is not None:
        kws.close()                   # disarms the board, writes results/{trial_id}_kws.csv
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()
    write_results()
    if run_log:
        spoken = sum(1 for r in run_log if r.get("src") == "kws")
        print(f"commands: {spoken} spoken, {len(run_log) - spoken} typed")
    print(f"\nSession ended. Frames shown: {state.count}" +
          (f"  |  saved to '{out_name}'" if vw is not None else ""))