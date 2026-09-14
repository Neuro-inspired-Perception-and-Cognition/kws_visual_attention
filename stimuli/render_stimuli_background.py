"""
Smooth jittering circles video generator (textured interiors, pencil background)

Render a 346x260 video of 6 textured circles over a faded pencil-scratch
background. Each circle "jitters" one pixel around its home position, cycling
through the eight neighbouring pixels. The background jitters too, on its own
phase.

Returns .mp4 video plus <stem>.mask.npy and <stem>.truth.csv for ground truth.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio.v2 as imageio

# Ground truth
from ground_truth_helpers import build_mask, centroids, save_truth

# identical circles -> identified by position
NAMES = ["top-left", "top-mid", "top-right", "bottom-left", "bottom-mid", "bottom-right"]

# Parameters
WIDTH           = 346       # output width
HEIGHT          = 260       # output height
SPRITE          = 100       # each object is drawn on a SPRITE x SPRITE canvas
N_OBJECTS       = 6         # number of objects
CIRCLE_R        = 0.20      # circle radius as a fraction of SPRITE
FPS             = 20
FRAMES_PER_STEP = 1         # frames held at each of the 8 positions (>=1)
N_STEPS         = 200       # number of jitter steps -> N_STEPS*FRAMES_PER_STEP frames
OUT_DIR         = "stimuli/frame_videos"  # the .mp4 saves here
TRUTH_DIR       = "stimuli/ground_truth_masks"   # the .mask.npy + .truth.csv saves here
OUT_NAME        = f"{N_OBJECTS}_circles_bg_{WIDTH}x{HEIGHT}.mp4"
OUT_PATH        = os.path.join(OUT_DIR, OUT_NAME)
STEM            = os.path.splitext(OUT_NAME)[0]  # shared basename: video <-> its truth

# texture: coarse enough to survive downsampling, bright enough to stay visible
TEX_LO, TEX_HI  = 110, 256  # grey range of the speckle
TEX_CELL        = 3         # speckle cell size in px (1 = per-pixel noise)
PROC_DOWNSAMPLE = 2         # must equal DOWNSAMPLE in the controller

# pencil-scratch background
BG_ENABLED   = True
BG_JITTER    = False        # static hatching
BG_STROKES   = 1100         # how many scratch strokes
BG_LO, BG_HI = 100, 200     # grey range of a stroke
BG_LEN       = (12, 55)     # stroke length in px
BG_WIDTH     = 1            # stroke width in px
BG_ANGLES    = (-35, 20)    # degrees: hatching leans these two ways
BG_BLUR      = 0.4          # low = crisp crossed lines; high = soft haze
BG_SEED      = 7

# 8 unit displacements around a circle of radius 1 px
CIRCLE = [
    ( 1,  0),  # E
    ( 1, -1),  # NE
    ( 0, -1),  # N
    (-1, -1),  # NW
    (-1,  0),  # W
    (-1,  1),  # SW
    ( 0,  1),  # S
    ( 1,  1),  # SE
]


def make_background():
    """Faded pencil hatching, drawn once. Built 2px larger on each side so it can
    be jittered without exposing an uncovered edge."""
    rng = np.random.default_rng(BG_SEED)
    pad = 2
    img = Image.new("L", (WIDTH + 2 * pad, HEIGHT + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    for _ in range(BG_STROKES):
        x0 = rng.uniform(0, WIDTH + 2 * pad)
        y0 = rng.uniform(0, HEIGHT + 2 * pad)
        ang = np.radians(rng.choice(BG_ANGLES) + rng.uniform(-8, 8))
        ln = rng.uniform(*BG_LEN)
        grey = int(rng.integers(BG_LO, BG_HI))
        d.line([x0, y0, x0 + ln * np.cos(ang), y0 + ln * np.sin(ang)],
               fill=grey, width=BG_WIDTH)
    img = img.filter(ImageFilter.GaussianBlur(BG_BLUR))
    return img


def make_sprite(seed):
    """One textured white disk on a transparent SPRITE x SPRITE tile."""
    rng = np.random.default_rng(seed)

    # coarse speckle: build small, upscale with NEAREST so cells are TEX_CELL px
    small = max(1, SPRITE // TEX_CELL)
    cells = rng.integers(TEX_LO, TEX_HI, size=(small, small)).astype(np.uint8)
    tex = np.array(Image.fromarray(cells).resize((SPRITE, SPRITE), Image.NEAREST))
    tex_rgba = Image.fromarray(np.stack([tex] * 3, -1), "RGB").convert("RGBA")

    # alpha mask = the disk itself (this is what build_mask reads for the footprint)
    r = SPRITE * CIRCLE_R
    cx = cy = SPRITE / 2
    mask = Image.new("L", (SPRITE, SPRITE), 0)
    ImageDraw.Draw(mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    img = Image.new("RGBA", (SPRITE, SPRITE), (0, 0, 0, 0))
    img.paste(tex_rgba, (0, 0), mask)
    return img


# home layout: 3 columns x 2 rows, evenly spaced.
def home_positions(cols=3, rows=2):
    xs = [WIDTH * (c + 0.5) / cols for c in range(cols)]
    ys = [HEIGHT * (r + 0.5) / rows for r in range(rows)]
    return [(x, y) for y in ys for x in xs]


def save_scene_truth(sprites, homes, stem=STEM, truth_dir=TRUTH_DIR):
    """stem = the clip's base name (no extension, no folder), e.g.
    '6_circles_bg_346x260'. The video and its ground truth live in different
    folders but share this basename, so a clip is always paired with its own
    mask -- that pairing is what keeps scoring honest.

    Unaffected by the background: the mask comes from each sprite's alpha (the
    disk), so ground truth is identical with or without clutter.
    """
    os.makedirs(truth_dir, exist_ok=True)
    base = os.path.join(truth_dir, stem)
    mask = build_mask(sprites, homes, WIDTH, HEIGHT, SPRITE)[::PROC_DOWNSAMPLE, ::PROC_DOWNSAMPLE]
    np.save(f"{base}.mask.npy", mask)                        # per-pixel footprint (scoring)
    save_truth(centroids(mask), NAMES, f"{base}.truth.csv")  # readable centroids
    print(f"saved {base}.mask.npy + {base}.truth.csv  (grid {mask.shape[1]}x{mask.shape[0]})")
    return mask


# render
def render(out_path=OUT_PATH):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sprites = [make_sprite(seed=i) for i in range(N_OBJECTS)]   # different texture each
    homes   = home_positions()[:N_OBJECTS]
    bg      = make_background() if BG_ENABLED else None
    pad     = 2

    phases = [(i * 3) % 8 for i in range(N_OBJECTS)]
    signs  = [1 if i % 2 == 0 else -1 for i in range(N_OBJECTS)]
    bg_phase, bg_sign = 5, 1        # its own phase so it doesn't track any object

    frames = []
    for step in range(N_STEPS):
        canvas = Image.new("L", (WIDTH, HEIGHT), 0)
        if bg is not None:
            if BG_JITTER:
                bx, by = CIRCLE[(bg_phase + bg_sign * step) % 8]
            else:
                bx = by = 0
            canvas.paste(bg, (-pad + bx, -pad + by))
        canvas = canvas.convert("RGB")
        for (hx, hy), sprite, ph, sg in zip(homes, sprites, phases, signs):
            dx, dy = CIRCLE[(ph + sg * step) % 8]
            px = int(round(hx - SPRITE / 2 + dx))
            py = int(round(hy - SPRITE / 2 + dy))
            canvas.paste(sprite, (px, py), sprite)   # alpha channel = mask
        frames.extend([np.asarray(canvas)] * FRAMES_PER_STEP)

    writer = imageio.get_writer(
        out_path, format="FFMPEG", mode="I", fps=FPS,
        codec="libx264rgb",
        output_params=["-crf", "0"],
        macro_block_size=None,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()

    print(f"wrote {out_path}: {len(frames)} frames, {WIDTH}x{HEIGHT}, {FPS} fps"
          f"  (background {'on, jittering' if BG_ENABLED and BG_JITTER else 'on, static' if BG_ENABLED else 'off'})")
    return sprites, homes


if __name__ == "__main__":
    sprites, homes = render()
    save_scene_truth(sprites, homes)