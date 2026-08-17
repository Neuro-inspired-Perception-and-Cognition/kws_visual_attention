import numpy as np
import dv_processing as dv
import cv2
import torch
import sys
from datetime import timedelta, datetime
from visual_attention.helpers_visual_att import initialise_attention, run_attention


# -----------------------------------------------------------
# Configurations
# -----------------------------------------------------------

DEBUG = True
SHOW_REALTIME = True

class Config:
    ATTENTION_PARAMS = {
        'size_krn': 16,
        'r0': 7,
        'rho': 0.015,
        'theta': np.pi * 3 / 2,
        'thetas': np.arange(0, 2 * np.pi, np.pi / 4),
        'thick': 12,
        'fltr_resize_perc': [2, 2],
        'offsetpxs': 0,
        'offset': (0, 0),
        'num_pyr': 6,
        'tau_mem': 0.3,
        'stride': 1,
        'out_ch': 1
    }

THRESHOLD = 0.6
BOOST_MAX = 3.0
N_BOOST_FRAMES = 20
HARDCODED_WORD = "right"
HARDCODED_CONFIDENCE = 0.6


# -----------------------------------------------------------
# KWS modulation helpers (unchanged)
# -----------------------------------------------------------
def get_quadrant(x, y, mid_x, mid_y):
    h = "left"  if x < mid_x else "right"
    v = "top"   if y < mid_y else "bottom"
    return v, h


def make_gradient(H, W, word, strength, salmax_coords=None):
    peak = 1.0 + BOOST_MAX * strength
    mid_x, mid_y = W // 2, H // 2

    if salmax_coords is not None:
        y, x = salmax_coords
    else:
        y, x = mid_y, mid_x

    cols = np.arange(W, dtype=float)
    rows = np.arange(H, dtype=float)

    if word == "right":
        t_primary   = np.clip((cols - x) / max(W - x, 1), 0, 1)
        t_secondary = np.clip((rows - mid_y) / max(H - mid_y, 1), 0, 1) if y >= mid_y \
                      else np.clip((mid_y - rows) / mid_y, 0, 1)
        boost_2d = t_primary[None, :] * t_secondary[:, None]

    elif word == "left":
        t_primary   = np.clip((x - cols) / max(x, 1), 0, 1)
        t_secondary = np.clip((rows - mid_y) / max(H - mid_y, 1), 0, 1) if y >= mid_y \
                      else np.clip((mid_y - rows) / mid_y, 0, 1)
        boost_2d = t_primary[None, :] * t_secondary[:, None]

    elif word == "up":
        t_primary   = np.clip((y - rows) / max(y, 1), 0, 1)
        t_secondary = np.clip((mid_x - cols) / mid_x, 0, 1) if x < mid_x \
                      else np.clip((cols - mid_x) / max(W - mid_x, 1), 0, 1)
        boost_2d = t_primary[:, None] * t_secondary[None, :]

    elif word == "down":
        t_primary   = np.clip((rows - y) / max(H - y, 1), 0, 1)
        t_secondary = np.clip((mid_x - cols) / mid_x, 0, 1) if x < mid_x \
                      else np.clip((cols - mid_x) / max(W - mid_x, 1), 0, 1)
        boost_2d = t_primary[:, None] * t_secondary[None, :]

    else:
        boost_2d = np.zeros((H, W), dtype=float)

    return 1.0 + (peak - 1.0) * boost_2d


def kws_modulate(saliency_map, salmax_coords, word, confidence, boost_frame):
    H, W  = saliency_map.shape
    mid_x = W // 2
    mid_y = H // 2

    y, x = salmax_coords
    v_quad, h_quad = get_quadrant(x, y, mid_x, mid_y)

    if boost_frame >= N_BOOST_FRAMES:
        gradient   = make_gradient(H, W, word, 1.0, salmax_coords)
        boosted    = np.clip(saliency_map.astype(float) * gradient, 0, 255)
        new_coords = np.unravel_index(np.argmax(boosted), boosted.shape)
        if DEBUG:
            new_y, new_x = new_coords
            new_v, new_h = get_quadrant(new_x, new_y, mid_x, mid_y)
            print(f"Frame FULL | strength=1.00 | "
                  f"peak: ({v_quad}-{h_quad}) → ({new_v}-{new_h}) | "
                  f"coords: ({x},{y}) → ({new_x},{new_y})")
        return boosted, new_coords, True, boost_frame

    if DEBUG and boost_frame == 0:
        print(f"\n--- KWS MODULATION ---")
        print(f"Command       : '{word}'  (confidence={confidence:.2f})")
        print(f"Current peak  : x={x}, y={y}  [{v_quad}-{h_quad}]")

    if confidence < THRESHOLD:
        if DEBUG:
            print(f"REJECTED — confidence {confidence:.2f} below threshold {THRESHOLD}")
        return saliency_map.copy(), salmax_coords, False, boost_frame

    already_there = (
        (word == "left"  and h_quad == "left")  or
        (word == "right" and h_quad == "right") or
        (word == "up"    and v_quad == "top")   or
        (word == "down"  and v_quad == "bottom")
    )
    if already_there:
        if DEBUG and boost_frame == 0:
            print(f"REJECTED — already in target region")
        return saliency_map.copy(), salmax_coords, False, boost_frame

    strength    = min(boost_frame / N_BOOST_FRAMES, 1.0)
    boost_frame = min(boost_frame + 1, N_BOOST_FRAMES)

    gradient   = make_gradient(H, W, word, strength, salmax_coords)
    boosted    = np.clip(saliency_map.astype(float) * gradient, 0, 255)
    new_coords = np.unravel_index(np.argmax(boosted), boosted.shape)
    new_y, new_x = new_coords
    new_v, new_h = get_quadrant(new_x, new_y, mid_x, mid_y)

    if DEBUG:
        print(f"Frame {boost_frame:2d}/{N_BOOST_FRAMES} | "
              f"strength={strength:.2f} | "
              f"peak: ({v_quad}-{h_quad}) → ({new_v}-{new_h}) | "
              f"coords: ({x},{y}) → ({new_x},{new_y})")

    return boosted, new_coords, True, boost_frame


# -----------------------------------------------------------
# Helper: normalize a 2D map to uint8 BGR for display
# -----------------------------------------------------------
def to_bgr(map_2d, colormap=cv2.COLORMAP_JET):
    m = map_2d.astype(float)
    lo, hi = m.min(), m.max()
    if hi > lo:
        m = (m - lo) / (hi - lo) * 255
    m = np.clip(m, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(m, colormap)


def add_label(img, text):
    cv2.putText(img, text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img


# -----------------------------------------------------------
# Camera and pipeline setup
# -----------------------------------------------------------
device = torch.device("cpu")
print(f"Using device: {device}")

config = Config()

capture = dv.io.camera.open()
if not capture.isEventStreamAvailable():
    raise RuntimeError("Camera does not provide an event stream.")

W_orig, H_orig = capture.getEventResolution()

DOWNSAMPLE_FACTOR = 2
max_x = W_orig // DOWNSAMPLE_FACTOR
max_y = H_orig // DOWNSAMPLE_FACTOR
resolution = (max_y, max_x)

print(f"Camera resolution    : {W_orig} x {H_orig}")
print(f"Processing resolution: {max_x} x {max_y} (downsampled {DOWNSAMPLE_FACTOR}x)")
print(f"Hardcoded command    : '{HARDCODED_WORD}'  (confidence={HARDCODED_CONFIDENCE})")

net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"kws_debug_{HARDCODED_WORD}_{timestamp}.mp4"
fourcc          = cv2.VideoWriter_fourcc(*'mp4v')
fps             = 10
# video is 3 panels wide
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (W_orig * 3, H_orig))
if not video_writer.isOpened():
    print("ERROR: Could not open video writer!")
    sys.exit(1)

if SHOW_REALTIME:
    cv2.namedWindow("KWS Debug", cv2.WINDOW_NORMAL)


# -----------------------------------------------------------
# Persistent state
# -----------------------------------------------------------
class State:
    boost_frame    = 0
    active_command = None
    locked_coords  = None
    frame_count    = 0
    running        = True

state = State()


# -----------------------------------------------------------
# Slicer callback - DVS events
# -----------------------------------------------------------
def slicing_callback(events: dv.EventStore):
    if not state.running or events is None or len(events) == 0:
        return

    # --- Panel 1: raw events ---
    events_np = events.numpy()
    x_arr = (events_np['x'].astype(int) // DOWNSAMPLE_FACTOR).clip(0, max_x - 1)
    y_arr = (events_np['y'].astype(int) // DOWNSAMPLE_FACTOR).clip(0, max_y - 1)

    window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
    window[0, y_arr, x_arr] = 255

    window_np  = window[0].numpy()
    panel_events = cv2.resize(to_bgr(window_np, cv2.COLORMAP_BONE),
                              (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    add_label(panel_events, "1. raw events")

    # --- Panel 2: raw saliency ---
    saliency_map, salmax_coords = run_attention(
        window, net_attention, device, resolution,
        config.ATTENTION_PARAMS['num_pyr']
    )

    if np.isnan(saliency_map).any() or saliency_map.max() == saliency_map.min():
        # still show the events panel so you can see what's coming in
        blank = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
        add_label(blank, "2. saliency: NaN/flat")
        add_label(blank.copy(), "3. modulated: N/A")
        combined = np.hstack([panel_events, blank, blank])
        if SHOW_REALTIME:
            try:
                cv2.imshow("KWS Debug", combined)
            except cv2.error:
                pass
        return

    panel_saliency = cv2.resize(to_bgr(saliency_map),
                                (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    # mark saliency peak
    sal_peak = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    cv2.circle(panel_saliency,
               (int(sal_peak[1] * DOWNSAMPLE_FACTOR),
                int(sal_peak[0] * DOWNSAMPLE_FACTOR)),
               10, (255, 255, 255), 4)
    add_label(panel_saliency, "2. raw saliency")

    # --- Panel 3: modulated saliency ---
    if HARDCODED_WORD != state.active_command:
        state.active_command = HARDCODED_WORD
        state.boost_frame    = 0
        state.locked_coords  = None

    if state.locked_coords is None:
        state.locked_coords = salmax_coords

    boosted_map, new_coords, accepted, state.boost_frame = kws_modulate(
        saliency_map, state.locked_coords,
        HARDCODED_WORD, HARDCODED_CONFIDENCE,
        state.boost_frame
    )

    panel_modulated = cv2.resize(to_bgr(boosted_map),
                                 (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    peak_x = int(new_coords[1] * DOWNSAMPLE_FACTOR)
    peak_y = int(new_coords[0] * DOWNSAMPLE_FACTOR)
    cv2.circle(panel_modulated, (peak_x, peak_y), 10, (255, 255, 255), 4)

    status = f"{'ACC' if accepted else 'REJ'}  cmd:'{HARDCODED_WORD}'"
    add_label(panel_modulated, f"3. modulated — {status}")

    # --- Combine and show ---
    combined = np.hstack([panel_events, panel_saliency, panel_modulated])

    video_writer.write(combined)
    state.frame_count += 1

    if SHOW_REALTIME:
        try:
            cv2.imshow("KWS Debug", combined)
        except cv2.error:
            pass


# -----------------------------------------------------------
# Main loop
# -----------------------------------------------------------
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(timedelta(milliseconds=100), slicing_callback)

print("\nProcessing live events... Press 'q' to stop.")
print("Panels: [raw events] | [raw saliency] | [modulated saliency]")

try:
    while capture.isRunning() and state.running:
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

        if SHOW_REALTIME:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user.")
                state.running = False

finally:
    video_writer.release()
    if SHOW_REALTIME:
        cv2.destroyAllWindows()
    print(f"\nVideo saved : '{output_filename}'")
    print(f"Total frames: {state.frame_count}")