"""
Ground truth for the camera: one APS frame from the DAVIS + SAM 2.

Usage:

    python stimuli/cam_ground_truth.py                          # live view to take the picture
    python stimuli/cam_ground_truth.py --no-live                # grab frames straight away (not recommended)
    python stimuli/cam_ground_truth.py --image frame.png        # use a saved frame (for testing)
    python stimuli/cam_ground_truth.py --names                  # names, in the order clicked
    python stimuli/cam_ground_truth.py --mode auto --objects 3  # let SAM find everything, then filter

Live view (default): a window shows the APS stream live. Arrange the objects and
the camera while watching it, then:
    SPACE      take the picture (averages the next --frames frames)
    Esc        cancel
    ENTER      use it          
    R   retake (back to the live view)


Outputs, in stimuli/ground_truth_masks/ (all named after --stem):
    <stem>.aps.png       the frame used (re-run later with --image on this file)
    <stem>.mask.npy      label mask on the processing grid: 0 = background, 1..N = objects
    <stem>.truth.csv     id, name, x, y (processing grid) for each object
    <stem>.preview.png   the frame with each object tinted and numbered -> Check this to make sure the mask is correct 
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

truth_dir = "stimuli/ground_truth_masks"


# frame
def open_aps_camera():
    import dv_processing as dv
    capture = dv.io.camera.open()
    if not capture.isFrameStreamAvailable():
        sys.exit("This camera does not provide APS frames (isFrameStreamAvailable is False).")
    print(f"camera frame resolution: {capture.getFrameResolution()}")
    return capture


def next_gray(capture):
    """The next APS frame as uint8 grayscale, or None if none is ready yet."""
    frame = capture.getNextFrame()
    if frame is None:
        return None
    img = np.asarray(frame.image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def average_frames(capture, n_frames, timeout_s=10.0):
    """Average the next n APS frames (smooths flicker and sensor noise)."""
    import time
    frames, t0 = [], time.time()
    while len(frames) < n_frames and time.time() - t0 < timeout_s:
        img = next_gray(capture)
        if img is None:
            time.sleep(0.005)
            continue
        frames.append(img.astype(np.float32))
    if not frames:
        sys.exit(f"No APS frames arrived within {timeout_s:.0f} s. Is the frame stream "
                 "enabled and the exposure sensible?")
    print(f"averaged {len(frames)} APS frames")
    return np.mean(frames, axis=0).clip(0, 255).astype(np.uint8)


def grab_aps_frame(n_frames=10, timeout_s=10.0):
    """No live view: take the picture straight away."""
    return average_frames(open_aps_camera(), n_frames, timeout_s)


def exposure_control(capture):
    """(get, set) for the APS exposure in microseconds, or (None, None) if this
    camera object does not expose it."""
    from datetime import timedelta
    setter = getattr(capture, "setDavisExposureDuration", None)
    if setter is None:
        return None, None
    state = {"us": 4000}          # dv's usual DAVIS default is a few ms
    getter = getattr(capture, "getDavisExposureDuration", None)
    if getter is not None:
        try:
            state["us"] = int(getter().total_seconds() * 1e6)
        except Exception:
            pass

    def set_us(us):
        us = int(np.clip(us, 100, 200_000))
        setter(timedelta(microseconds=us))
        state["us"] = us
        return us
    return (lambda: state["us"]), set_us


def draw_help(view, lines, color=(0, 255, 0)):
    for i, text in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(view, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(view, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return view


def live_capture(n_frames=10, scale=2):
    """Show the APS stream live; the user decides when the picture is taken."""
    capture = open_aps_camera()
    get_exp, set_exp = exposure_control(capture)
    win = "APS live -- SPACE take picture, +/- exposure, Esc cancel"
    cv2.namedWindow(win)
    last = None

    while True:
        # ---------- live view ----------
        while True:
            img = next_gray(capture)
            if img is not None:
                last = img
            if last is not None:
                view = cv2.cvtColor(cv2.resize(last, None, fx=scale, fy=scale,
                                               interpolation=cv2.INTER_NEAREST),
                                    cv2.COLOR_GRAY2BGR)
                dark = 100 * (last <= 5).mean()
                bright = 100 * (last >= 250).mean()
                info = f"mean {last.mean():5.1f}   too dark {dark:4.1f}%   too bright {bright:4.1f}%"
                if get_exp is not None:
                    info += f"   exposure {get_exp() / 1000:.1f} ms"
                warn = (0, 0, 255) if dark > 5 or bright > 5 else (0, 255, 0)
                draw_help(view, ["LIVE   SPACE = take picture   +/- = exposure   Esc = cancel",
                                 info], warn)
                cv2.imshow(win, view)
            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                sys.exit("cancelled")
            if key in (ord("+"), ord("=")) and set_exp is not None:
                print(f"  exposure {set_exp(get_exp() * 1.25) / 1000:.1f} ms")
            if key in (ord("-"), ord("_")) and set_exp is not None:
                print(f"  exposure {set_exp(get_exp() / 1.25) / 1000:.1f} ms")
            if key == 32:                                  # SPACE
                break

        # take picture, then show it frozen
        print("taking the picture - keep the camera still ...")
        gray = average_frames(capture, n_frames)
        view = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        draw_help(view, ["PICTURE TAKEN   ENTER = use it   R = retake   Esc = cancel"],
                  (0, 255, 255))
        cv2.imshow(win, view)
        while True:
            key = cv2.waitKey(30) & 0xFF
            # keep draining the stream so the next live view is not stale
            next_gray(capture)
            if key in (13, 10):                            # ENTER
                cv2.destroyWindow(win)
                return gray
            if key in (ord("r"), ord("R")):
                print("retake")
                break
            if key == 27:
                cv2.destroyAllWindows()
                sys.exit("cancelled")


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"could not read image: {path}")
    return img


# clicks
def collect_clicks(gray, scale=3):
    """Show the frame; left-click objects, right-click undo, Enter/Space done, Esc cancel."""
    points = []
    win = "click each object -- Enter when done, right-click undo, Esc cancel"

    def redraw():
        view = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        for i, (x, y) in enumerate(points, start=1):
            cv2.circle(view, (x * scale, y * scale), 7, (0, 0, 255), -1)
            cv2.putText(view, str(i), (x * scale + 9, y * scale - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, view)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x // scale, y // scale))
            print(f"  object {len(points)} at ({x // scale}, {y // scale})")
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            print("  undo")
            redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, 32):          # Enter / Space
            break
        if key == 27:                    # Esc
            cv2.destroyWindow(win)
            sys.exit("cancelled")
    cv2.destroyWindow(win)
    return points


def parse_points(text):
    """'90,145 226,170' -> [(90, 145), (226, 170)]"""
    pts = []
    for tok in text.split():
        x, y = tok.split(",")
        pts.append((int(x), int(y)))
    return pts


# SAM - object segmentation
def load_sam(weights):
    try:
        from ultralytics import SAM
    except ImportError:
        sys.exit("SAM 2 needs ultralytics:  pip install ultralytics")
    return SAM(weights)


def sam_from_clicks(model, gray, points, device="cpu"):
    """One mask per clicked point, in click order."""
    bgr = np.stack([gray] * 3, -1)
    res = model(bgr, points=[list(p) for p in points], labels=[1] * len(points),
                device=device, verbose=False)
    masks = res[0].masks.data.cpu().numpy().astype(bool)
    if len(masks) != len(points):
        sys.exit(f"SAM returned {len(masks)} masks for {len(points)} clicks")
    return list(masks)


def sam_everything(model, gray, device="cpu", min_area=0.002, max_area=0.25, contain=0.8,
                   rel_size=0.25):
    """Segment everything, then keep only what looks like a whole object.

    What SAM returns on a table, measured on a test frame: the wall, each object,
    parts of objects (the coffee inside the mug, the mug handle), and shadows.
    The filters below drop those:
      - touching the image border  -> wall, table edge, anything cut off
      - area outside [min_area, max_area] of the frame -> specks, the table
      - mostly inside a larger kept mask -> parts (coffee inside mug)
      - smaller than rel_size x the largest survivor -> handles, shadows
    The last rule is the fragile one: a genuinely small object next to a big one
    would be dropped too. That is why click mode is the default.
    """
    bgr = np.stack([gray] * 3, -1)
    res = model(bgr, device=device, verbose=False)
    masks = [m for m in res[0].masks.data.cpu().numpy().astype(bool)]
    h, w = gray.shape
    total = h * w

    kept = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
            continue
        frac = m.sum() / total
        if not (min_area <= frac <= max_area):
            continue
        kept.append(m)

    kept.sort(key=lambda m: -m.sum())            # largest first
    wholes = []
    for m in kept:
        if any((m & big).sum() >= contain * m.sum() for big in wholes):
            continue                              # a part of something already kept
        wholes.append(m)

    if wholes:
        biggest = wholes[0].sum()
        wholes = [m for m in wholes if m.sum() >= rel_size * biggest]

    # order left to right, then top to bottom
    def centre(m):
        ys, xs = np.nonzero(m)
        return xs.mean(), ys.mean()
    wholes.sort(key=lambda m: (round(centre(m)[0] / 20), centre(m)[1]))
    return wholes


# outputs
def build_label_mask(masks):
    """Stack binary masks into one label image. Overlaps go to the smaller mask
    (a small object usually sits in front of a bigger one)."""
    h, w = masks[0].shape
    label = np.zeros((h, w), np.uint8)
    area = np.full((h, w), np.inf)
    for oid, m in enumerate(masks, start=1):
        a = m.sum()
        take = m & (a < area)
        label[take] = oid
        area[take] = a
    return label


def save_preview(gray, label, names, points, path):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    pal = [(69, 69, 233), (217, 141, 64), (58, 189, 240),
           (143, 82, 226), (200, 197, 58), (240, 116, 158),
           (80, 200, 80), (200, 80, 200)]
    for oid in range(1, int(label.max()) + 1):
        s = label == oid
        rgb[s] = 0.45 * rgb[s] + 0.55 * np.array(pal[(oid - 1) % len(pal)])
    scale = 3
    big = cv2.resize(rgb.clip(0, 255).astype(np.uint8), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    for oid in range(1, int(label.max()) + 1):
        ys, xs = np.nonzero(label == oid)
        if not len(xs):
            continue
        cx, cy = int(xs.mean() * scale), int(ys.mean() * scale)
        text = f"{oid} {names[oid - 1]}"
        cv2.putText(big, text, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(big, text, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for x, y in points or []:
        cv2.drawMarker(big, (x * scale, y * scale), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.imwrite(path, big)
    return big


# main
def main():
    p = argparse.ArgumentParser(description="Camera ground truth from an APS frame + SAM 2.")
    p.add_argument("--image", default=None, help="use this saved frame instead of the camera")
    p.add_argument("--mode", choices=["click", "auto"], default="click")
    p.add_argument("--points", default=None,
                   help='clicks as "x,y x,y ..." in full-resolution pixels (skips the window)')
    p.add_argument("--names", nargs="+", default=None, help="object names, in click order")
    p.add_argument("--objects", type=int, default=None,
                   help="auto mode: how many objects you expect; stop if SAM disagrees")
    p.add_argument("--stem", default="camera_6_objects_color_nobg_346x260", help="basename for all outputs")
    p.add_argument("--downsample", type=int, default=2,
                   help="processing grid factor -- must match DOWNSAMPLE in the controller")
    p.add_argument("--weights", default="sam2.1_t.pt",
                   help="SAM 2 weights (downloaded automatically on first use)")
    p.add_argument("--frames", type=int, default=10, help="APS frames to average")
    p.add_argument("--no-live", action="store_true",
                   help="skip the live view and grab the frame straight away")
    p.add_argument("--device", default="cpu",
                   help="where SAM runs. cpu by default: the lab's RTX 5080 is newer than the "
                        "PyTorch in kws_env, so 'cuda' fails with 'no kernel image'")
    a = p.parse_args()

    os.makedirs(truth_dir, exist_ok=True)
    base = os.path.join(truth_dir, a.stem)

    # 1. the frame
    if a.image:
        gray = load_image(a.image)
    elif a.no_live:
        gray = grab_aps_frame(a.frames)
    else:
        gray = live_capture(a.frames)
    if not a.image:
        cv2.imwrite(f"{base}.aps.png", gray)
        print(f"saved {base}.aps.png  (re-run with --image {base}.aps.png)")
    print(f"frame {gray.shape[1]}x{gray.shape[0]}")

    # 2. which objects
    model = load_sam(a.weights)
    points = None
    if a.mode == "click":
        points = parse_points(a.points) if a.points else collect_clicks(gray)
        if not points:
            sys.exit("no objects clicked")
        print(f"segmenting {len(points)} clicked objects ...")
        masks = sam_from_clicks(model, gray, points, a.device)
    else:
        print("segmenting everything (slow on CPU) ...")
        masks = sam_everything(model, gray, a.device)
        print(f"kept {len(masks)} objects after filtering")

    names = a.names or [f"object {i}" for i in range(1, len(masks) + 1)]
    if len(names) != len(masks):
        sys.exit(f"{len(names)} names given for {len(masks)} objects")

    label_full = build_label_mask(masks)
    save_preview(gray, label_full, names, points, f"{base}.preview.png")
    print(f"wrote {base}.preview.png   <- CHECK it before trusting any score")

    if a.objects is not None and len(masks) != a.objects:
        sys.exit(f"found {len(masks)} objects, expected {a.objects} -- mask NOT saved. "
                 "Look at the preview, or use click mode.")

    # 3. the mask, on the processing grid
    ds = a.downsample
    label = label_full[::ds, ::ds]
    np.save(f"{base}.mask.npy", label)

    with open(f"{base}.truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "x", "y", "area_px", "radius_px"])
        print(f"\n{'id':>3}  {'name':<12} {'centre (grid)':>15}  {'area':>6}  {'radius':>6}")
        for oid in range(1, len(masks) + 1):
            ys, xs = np.nonzero(label == oid)
            if not len(xs):
                print(f"{oid:>3}  {names[oid-1]:<12} -- vanished at downsample {ds} (too small)")
                continue
            area_px = len(xs)
            radius = (area_px / np.pi) ** 0.5
            w.writerow([oid, names[oid - 1], round(xs.mean(), 1), round(ys.mean(), 1),
                        area_px, round(radius, 1)])
            print(f"{oid:>3}  {names[oid-1]:<12} ({xs.mean():6.1f},{ys.mean():6.1f})  "
                  f"{area_px:>6}  {radius:>6.1f}")
    print(f"\nwrote {base}.mask.npy  (grid {label.shape[1]}x{label.shape[0]}, "
          f"downsample {ds})")
    print(f"wrote {base}.truth.csv")


if __name__ == "__main__":
    main()