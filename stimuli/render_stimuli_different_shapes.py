"""
Smooth jittering shapes video generator

Render a 346x260 video of 6 different objects on a black background. Each object
"jitters" one pixel around its home position, cycling through the eight
neighbouring pixels in a circular pattern (the 4 orthogonal directions +
the 4 diagonals). Objects start at different phases and spin in different
directions, so the whole field shimmers with 1-pixel motion

Returns .mp4 video plus <stem>.mask.npy and <stem>.truth.csv for ground truth.
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio.v2 as imageio

# Ground truth
from ground_truth_helpers import build_mask, centroids, save_truth

NAMES = ["apple", "bottle", "star", "heart", "diamond", "mushroom"]

# Parameters
WIDTH           = 346       # output width
HEIGHT          = 260       # output height
SPRITE          = 100       # each object is drawn on a SPRITE x SPRITE canvas
N_OBJECTS       = 6         # number of objects
FPS             = 20
FRAMES_PER_STEP = 1         # frames held at each of the 8 positions (>=1)
N_STEPS         = 200       # number of jitter steps -> N_STEPS*FRAMES_PER_STEP frames-
COLOR           = True      
BACKGROUND      = True     
OUT_DIR         = "stimuli/frame_videos"         # the .mp4 saves here
TRUTH_DIR       = "stimuli/ground_truth_masks"   # the .mask.npy + .truth.csv save here
OUT_NAME        = (f"{N_OBJECTS}_objects_"
                   f"{'color' if COLOR else 'mono'}_"
                   f"{'bg' if BACKGROUND else 'nobg'}_"
                   f"{WIDTH}x{HEIGHT}.mp4")
OUT_PATH        = os.path.join(OUT_DIR, OUT_NAME)
STEM            = os.path.splitext(OUT_NAME)[0]  # shared basename: video <-> its truth

SCALE           = 0.6 # adjust size
SCALES          = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # per object, multiplied by SCALE


# texture (interior events)
TEXTURED        = True      # False -> flat fills, hollow event rings
TEX_CELL        = 3         # speckle cell size in px
TEX_MIN         = 0.55      # darkest the speckle drives a pixel (1.0 = unchanged)
PROC_DOWNSAMPLE = 2         # must equal DOWNSAMPLE in the controller

# pencil-scratch background (only used when BACKGROUND = True). It is STATIC, so it
# costs nothing in events -- it is visible in the .mp4 and picked up by a real
# camera (screen flicker/sensor noise), but silent in an IEBCS conversion.
BG_STROKES      = 1100      # how many scratch strokes
BG_LO, BG_HI    = 100, 200  # grey range of a stroke
BG_LEN          = (12, 55)  # stroke length in px
BG_WIDTH        = 1         # stroke width in px
BG_ANGLES       = (-35, 20) # degrees: hatching leans these two ways
BG_BLUR         = 0.4       # low = crisp crossed lines; high = soft haze
BG_SEED         = 7

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

# sprites
# Each drawer paints one object centred on a SPRITE x SPRITE transparent tile.
def _apple(d, s):
    cx, cy, r = s/2, s*0.56, s*0.32
    d.line([cx, cy-r, cx+s*0.06, cy-r-s*0.16], fill=(120,72,34), width=max(1,int(s*0.06)))
    d.ellipse([cx+s*0.03, cy-r-s*0.13, cx+s*0.25, cy-r+s*0.03], fill=(70,170,80))
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(220,45,45))


def _bottle(d, s):
    cx = s/2
    d.rectangle([cx-s*0.10, s*0.10, cx+s*0.10, s*0.18], fill=(235,225,70))
    d.rectangle([cx-s*0.07, s*0.18, cx+s*0.07, s*0.34], fill=(70,155,205))
    d.rounded_rectangle([cx-s*0.22, s*0.34, cx+s*0.22, s*0.88],
                        radius=s*0.08, fill=(70,155,205))


def _star(d, s):
    cx, cy, R, r = s/2, s/2, s*0.36, s*0.15
    pts = []
    for k in range(10):
        ang = -math.pi/2 + k*math.pi/5
        rad = R if k % 2 == 0 else r
        pts.append((cx + rad*math.cos(ang), cy + rad*math.sin(ang)))
    d.polygon(pts, fill=(245,200,45))


def _heart(d, s):
    cx, cy, r = s/2, s/2, s*0.17
    d.ellipse([cx-2*r, cy-r-s*0.06, cx, cy+r-s*0.06], fill=(225,55,95))
    d.ellipse([cx, cy-r-s*0.06, cx+2*r, cy+r-s*0.06], fill=(225,55,95))
    d.polygon([(cx-2*r+s*0.03, cy), (cx+2*r-s*0.03, cy), (cx, cy+2*r+s*0.03)],
              fill=(225,55,95))


def _diamond(d, s):
    cx, cy, r = s/2, s/2, s*0.34
    d.polygon([(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)],
              fill=(80,205,215))


def _mushroom(d, s):
    cx, cy = s/2, s/2
    d.pieslice([cx-s*0.34, cy-s*0.30, cx+s*0.34, cy+s*0.20],
               180, 360, fill=(210,60,60))
    d.rounded_rectangle([cx-s*0.16, cy-s*0.06, cx+s*0.16, cy+s*0.32],
                        radius=s*0.06, fill=(235,225,200))
DRAWERS = [_apple, _bottle, _star, _heart, _diamond, _mushroom]


def make_background():
    """Static pencil hatching, drawn once."""
    rng = np.random.default_rng(BG_SEED)
    img = Image.new("L", (WIDTH, HEIGHT), 0)
    d = ImageDraw.Draw(img)
    for _ in range(BG_STROKES):
        x0, y0 = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)
        ang = np.radians(rng.choice(BG_ANGLES) + rng.uniform(-8, 8))
        ln = rng.uniform(*BG_LEN)
        d.line([x0, y0, x0 + ln * np.cos(ang), y0 + ln * np.sin(ang)],
               fill=int(rng.integers(BG_LO, BG_HI)), width=BG_WIDTH)
    return img.filter(ImageFilter.GaussianBlur(BG_BLUR)) if BG_BLUR else img


def make_sprite(drawer, seed=0, scale=1.0):
    """Draw the object at `scale`, then modulate its interior with a fixed speckle."""
    inner = max(8, int(round(SPRITE * scale)))
    tile = Image.new("RGBA", (inner, inner), (0, 0, 0, 0))
    drawer(ImageDraw.Draw(tile), inner)

    img = Image.new("RGBA", (SPRITE, SPRITE), (0, 0, 0, 0))
    off = (SPRITE - inner) // 2            # centre it; negative if scale > 1 (crops)
    img.paste(tile, (off, off))
    if inner > SPRITE:
        print(f"  warning: scale {scale} exceeds the {SPRITE}px tile - shape clipped; "
              f"raise SPRITE instead")

    if not COLOR:                              # monochrome: white silhouette
        arr = np.asarray(img).copy()
        arr[..., :3] = 255                     # alpha untouched -> shape unchanged
        img = Image.fromarray(arr, "RGBA")

    if not TEXTURED:
        return img

    rng = np.random.default_rng(seed)
    small = max(1, SPRITE // TEX_CELL)
    cells = rng.uniform(TEX_MIN, 1.0, size=(small, small))
    tex = np.array(Image.fromarray((cells * 255).astype(np.uint8))
                   .resize((SPRITE, SPRITE), Image.NEAREST)) / 255.0

    arr = np.asarray(img).astype(float)
    arr[..., :3] *= tex[..., None]            # alpha untouched -> mask unchanged
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")


#  home layout
# 3 columns x 2 rows, evenly spaced.
def home_positions(cols=3, rows=2):
    xs = [WIDTH * (c + 0.5) / cols for c in range(cols)]
    ys = [HEIGHT * (r + 0.5) / rows for r in range(rows)]
    return [(x, y) for y in ys for x in xs]


def save_scene_truth(sprites, homes, stem=STEM, truth_dir=TRUTH_DIR):
    """stem = the clip's base name (no extension, no folder), e.g.
    '6_objects_346x260'. The video and its ground truth live in different folders
    but share this basename, so a clip is always paired with its own mask.
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
    sprites = [make_sprite(fn, seed=i, scale=SCALE * SCALES[i])
               for i, fn in enumerate(DRAWERS)][:N_OBJECTS]
    homes   = home_positions()[:N_OBJECTS]

    # Stagger each object's starting direction and spin sense so they all
    # move differently and together cover orthogonal + diagonal directions.
    phases = [(i * 3) % 8 for i in range(N_OBJECTS)]
    signs  = [1 if i % 2 == 0 else -1 for i in range(N_OBJECTS)]

    bg = make_background() if BACKGROUND else None

    frames = []
    for step in range(N_STEPS):
        if bg is not None:
            canvas = bg.convert("RGB")         # static: same hatching every frame
        else:
            canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
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
          f"  ({'colour' if COLOR else 'mono'}, "
          f"background {'on' if BACKGROUND else 'off'}, "
          f"texture {'on' if TEXTURED else 'off'})")
    return sprites, homes


if __name__ == "__main__":
    sprites, homes = render()
    save_scene_truth(sprites, homes)