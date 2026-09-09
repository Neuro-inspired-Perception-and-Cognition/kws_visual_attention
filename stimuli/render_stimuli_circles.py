"""
Smooth jittering circles video generator

Render a 346x260 video of 6 white circles on a black background. Each circle
"jitters" one pixel around its home position, cycling through the eight
neighbouring pixels in a circular pattern.

Returns .mp4 video and scene_truth.csv (object ID, name, x, y) for ground truth.
"""

import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

# Ground truth 
from ground_truth_helpers import build_mask, centroids, save_truth, oracle

NAMES = ["top-left", "top-mid", "top-right", "bottom-left", "bottom-mid", "bottom-right"]

def save_scene_truth(sprites, homes, stem):
    """stem = the clip's base name, e.g. 'scene01' (same as scene01.mp4)."""
    mask = build_mask(sprites, homes, WIDTH, HEIGHT, SPRITE)[::2, ::2]   # 173x130 grid
    np.save(f"{stem}.mask.npy", mask)  # per-pixel footprint (for scoring)
    save_truth(centroids(mask), NAMES, f"{stem}.truth.csv")  # readable centroids
    print(f"saved {stem}.mask.npy + {stem}.truth.csv")
    return mask

# Parameters 
WIDTH           = 346       # output width
HEIGHT          = 260       # output height
SPRITE          = 100       # each object is drawn on a SPRITE x SPRITE canvas
N_OBJECTS       = 6         # number of objects
FPS             = 20
FRAMES_PER_STEP = 1         # frames held at each of the 8 positions (>=1)
N_STEPS         = 200       # number of jitter steps -> N_STEPS*FRAMES_PER_STEP frames
OUT_PATH        = f"{N_OBJECTS}_circles_{WIDTH}x{HEIGHT}.mp4"

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

# sprite: one white circle, centred on a SPRITE x SPRITE transparent tile.
def _circle(d, s):
    r = s * 0.30
    cx = cy = s / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))

DRAWERS = [_circle] * N_OBJECTS


def make_sprite(drawer):
    img = Image.new("RGBA", (SPRITE, SPRITE), (0, 0, 0, 0))
    drawer(ImageDraw.Draw(img), SPRITE)
    return img


#  home layout
# 3 columns x 2 rows, evenly spaced.
def home_positions(cols=3, rows=2):
    xs = [WIDTH * (c + 0.5) / cols for c in range(cols)]
    ys = [HEIGHT * (r + 0.5) / rows for r in range(rows)]
    return [(x, y) for y in ys for x in xs]


# render
def render(out_path=OUT_PATH):
    sprites = [make_sprite(fn) for fn in DRAWERS][:N_OBJECTS]
    homes   = home_positions()[:N_OBJECTS]

    phases = [(i * 3) % 8 for i in range(N_OBJECTS)]
    signs  = [1 if i % 2 == 0 else -1 for i in range(N_OBJECTS)]

    frames = []
    for step in range(N_STEPS):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        for (hx, hy), sprite, ph, sg in zip(homes, sprites, phases, signs):
            dx, dy = CIRCLE[(ph + sg * step) % 8]
            px = int(round(hx - SPRITE / 2 + dx))
            py = int(round(hy - SPRITE / 2 + dy))
            canvas.paste(sprite, (px, py), sprite)   # alpha channel = mask
        arr = np.asarray(canvas)
        frames.extend([arr] * FRAMES_PER_STEP)

    writer = imageio.get_writer(
        out_path, format="FFMPEG", mode="I", fps=FPS,
        codec="libx264rgb",
        output_params=["-crf", "0", "-pix_fmt", "rgb24"],
        macro_block_size=None,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()

    print(f"wrote {out_path}: {len(frames)} frames, {WIDTH}x{HEIGHT}, {FPS} fps")
    return sprites, homes


if __name__ == "__main__":
    sprites, homes = render()
    save_scene_truth(sprites, homes, OUT_PATH.rsplit(".", 1)[0])