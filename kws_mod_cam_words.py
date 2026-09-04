"""
Live camera + live typed commands.


Type direction commands in the terminal while camera streams events. 

Commands (typed in the terminal, Enter to submit):
    right / left / up / down     set the active direction (conf defaults to 1.0)
    stop  (or: none, clear)      release the active command, freeze in place
    reset                        zero the membrane and re-fixate on next salmax
    mode pan   / mode saccade    switch panning mode live
    quit / exit / q              end the session (also: 'q' in the video window,to restart, kill the terminal and run again)

Requires a local display for cv2.imshow and a connected DVS camera. 

NOTE on tuning: the spatial constants (READOUT_R, STEP, SACCADE_JUMP,
MIN_TRAVEL) were tuned for a 173x130 processing grid (346x260 downsampled 2x).
If your camera reports a different resolution these will need re-scaling.
"""

import queue
import sys
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
import dv_processing as dv

from visual_attention.helpers_visual_att import initialise_attention, run_attention
from command_parser import parse_command

# ---------------- config ----------------
DOWNSAMPLE = 2
WINDOW_MS = 100                 # slicer window == old npy frame window

ATTENTION_PARAMS = {
    'size_krn': 16, 'r0': 7, 'rho': 0.015, 'theta': np.pi * 3 / 2,
    'thetas': np.arange(0, 2 * np.pi, np.pi / 4), 'thick': 12,
    'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
    'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
}

CONF = 1.0
THRESHOLD = 0.6

# membrane / fovea-pan controller (unchanged from the interactive script)
LEAK = 0.5
STEP = 8.0
SACCADE_JUMP = 60.0
READOUT_R = 25.0
BOOST = 2.0
CAP_RATIO = 2.5
MIN_TRAVEL = 50.0
MODE = "pan"                    # "pan" or "saccade"; changeable live via "mode <x>"

RECORD = True                   # also save an .mp4 of the session
SHOW_EVENTS_PANEL = True        # left panel = raw events, right panel = fovea membrane
DEBUG = True

DIRS = {"right": 0.0, "down": np.pi / 2, "left": np.pi, "up": 3 * np.pi / 2}

WIN = "fovea (live camera)"

# ---------------- stdin reader thread ----------------
cmd_queue = queue.Queue()


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


threading.Thread(target=stdin_reader, args=(cmd_queue,), daemon=True).start()

# ---------------- camera ----------------
device = torch.device("cpu")
print(f"Using device: {device}")

capture = dv.io.camera.open()
if not capture.isEventStreamAvailable():
    raise RuntimeError("Camera does not provide an event stream.")

W_orig, H_orig = capture.getEventResolution()
max_x = W_orig // DOWNSAMPLE
max_y = H_orig // DOWNSAMPLE
resolution = (max_y, max_x)

print(f"Camera resolution    : {W_orig} x {H_orig}")
print(f"Processing resolution: {max_x} x {max_y}  (downsample {DOWNSAMPLE}x)")
print(f"Slicer window        : {WINDOW_MS} ms")

net = initialise_attention(device, ATTENTION_PARAMS)

X, Y = np.meshgrid(np.arange(max_x), np.arange(max_y))

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# ---------------- video writer ----------------
vw = None
out_name = None
if RECORD:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"fovea_live_camera_{ts}.mp4"
    panel_w = W_orig * (2 if SHOW_EVENTS_PANEL else 1)
    vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (panel_w, H_orig))
    if not vw.isOpened():
        print("WARNING: couldn't open video writer, continuing without recording")
        vw = None
    else:
        print(f"Recording to         : {out_name}")


# ---------------- persistent state ----------------
class State:
    M = np.zeros((max_y, max_x))
    fx = None
    fy = None
    active = None
    locked = False
    sx = None
    sy = None
    word = None
    conf = CONF
    mode = MODE
    count = 0
    running = True


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
        cv2.imshow(WIN, frame)
    except cv2.error:
        pass


def drain_commands(st):
    """Apply every command queued since the last call. Sets st.running=False on quit."""
    while True:
        try:
            item = cmd_queue.get_nowait()
        except queue.Empty:
            return
        cmd = item if isinstance(item, dict) else parse_command(item, DIRS)
        if cmd is None:
            continue
        t = cmd["type"]
        if t == "quit":
            st.running = False
            return
        elif t == "word":
            st.word, st.conf = cmd["word"], cmd["conf"]
            print(f"  -> command set: '{st.word}' (conf={st.conf})")
        elif t == "stop":
            st.word = None
            print("  -> command cleared, holding position")
        elif t == "reset":
            st.M[:] = 0.0
            st.fx = st.fy = None
            st.locked = False
            print("  -> membrane reset; re-fixating on next window's salmax")
        elif t == "mode":
            st.mode = cmd["mode"]
            print(f"  -> mode set: {st.mode}")
        elif t == "unknown":
            print(f"  ?? unrecognised: {cmd['raw']!r}")


# ---------------- slicer callback (one 100 ms event window) ----------------
def slicing_callback(events: dv.EventStore):
    st = state
    if not st.running or events is None or len(events) == 0:
        return

    ev = events.numpy()
    xa = (ev['x'].astype(int) // DOWNSAMPLE).clip(0, max_x - 1)
    ya = (ev['y'].astype(int) // DOWNSAMPLE).clip(0, max_y - 1)
    window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
    window[0, ya, xa] = 255.0

    with torch.no_grad():
        saliency, salmax = run_attention(window, net, device, resolution,
                                         ATTENTION_PARAMS['num_pyr'])
    saliency = np.asarray(saliency)

    # keep the window alive even when saliency is flat/NaN so you can see the events
    if np.isnan(saliency).any() or saliency.max() == saliency.min():
        ev_panel = cv2.resize(to_bgr(window[0].numpy(), cv2.COLORMAP_BONE),
                              (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        add_label(ev_panel, "events (saliency flat/NaN)")
        frame = np.hstack([ev_panel, np.zeros_like(ev_panel)]) if SHOW_EVENTS_PANEL else ev_panel
        show_frame(frame)
        return

    if st.fx is None:
        st.fy, st.fx = float(salmax[0]), float(salmax[1])
        st.sx, st.sy = st.fx, st.fy

    # decaying per-pixel membrane with foveal Gaussian boost
    foc = np.exp(-((X - st.fx) ** 2 + (Y - st.fy) ** 2) / (2 * READOUT_R ** 2))
    st.M = LEAK * st.M + (1.0 - LEAK) * saliency * (1.0 + BOOST * foc)

    # command -> fovea motion
    if st.word != st.active:
        st.locked = False
        st.sx, st.sy = st.fx, st.fy
    if (not st.locked) and st.word in DIRS and st.conf >= THRESHOLD:
        psi = DIRS[st.word]
        if st.mode == "pan":
            st.fx = float(np.clip(st.fx + STEP * np.cos(psi), 0, max_x - 1))
            st.fy = float(np.clip(st.fy + STEP * np.sin(psi), 0, max_y - 1))
        elif st.word != st.active:                 # saccade: one jump per new command
            st.fx = float(np.clip(st.fx + SACCADE_JUMP * np.cos(psi), 0, max_x - 1))
            st.fy = float(np.clip(st.fy + SACCADE_JUMP * np.sin(psi), 0, max_y - 1))
    st.active = st.word

    # readout: argmax of the membrane inside the foveal zone
    zone = (X - st.fx) ** 2 + (Y - st.fy) ** 2 <= READOUT_R ** 2
    if zone.any():
        ay, ax = np.unravel_index(int(np.argmax(np.where(zone, st.M, -np.inf))), st.M.shape)
    else:
        ay, ax = int(round(st.fy)), int(round(st.fx))

    # capture / lock once we've travelled far enough onto a strong peak
    travel = np.hypot(st.fx - st.sx, st.fy - st.sy) if st.sx is not None else 0.0
    if (not st.locked) and travel >= MIN_TRAVEL:
        zone_mean = float(st.M[zone].mean()) if zone.any() else 0.0
        if zone_mean > 0 and st.M[ay, ax] >= CAP_RATIO * zone_mean:
            st.fx, st.fy = float(ax), float(ay)
            st.locked = True

    if DEBUG:
        print(f"win {st.count:5d} | {'LOCK' if st.locked else 'pan ':4} | "
              f"cmd={str(st.word):6} | fovea=({int(st.fx)},{int(st.fy)}) | "
              f"attended=({ax},{ay}) | travel={travel:5.1f} | M@att={st.M[ay, ax]:.1f}",
              end="\r")

    # ---- render ----
    ds = DOWNSAMPLE
    fovea_panel = cv2.resize(to_bgr(st.M), (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    cv2.drawMarker(fovea_panel, (int(st.fx * ds), int(st.fy * ds)),
                   (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(fovea_panel, (int(ax * ds), int(ay * ds)), 11, (255, 255, 255), 3)
    label = (f"'{st.word}' [{st.mode}] {'LOCK' if st.locked else 'pan'}"
             if st.word else f"(no command) [{st.mode}]")
    add_label(fovea_panel, label, (0, 255, 0) if st.locked else (255, 255, 255))

    if SHOW_EVENTS_PANEL:
        ev_panel = cv2.resize(to_bgr(window[0].numpy(), cv2.COLORMAP_BONE),
                              (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        add_label(ev_panel, "raw events")
        frame = np.hstack([ev_panel, fovea_panel])
    else:
        frame = fovea_panel

    show_frame(frame)
    if vw is not None:
        vw.write(frame)
    st.count += 1


# ---------------- main loop ----------------
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(timedelta(milliseconds=WINDOW_MS), slicing_callback)

print("\n--- live. type commands below. 'q' in the window or 'quit' here to stop. ---\n")
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
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()
    print(f"\n\nSession ended. Frames shown: {state.count}" +
          (f"  |  saved to '{out_name}'" if vw is not None else ""))