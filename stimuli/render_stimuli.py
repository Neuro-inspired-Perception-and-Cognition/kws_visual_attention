"""
Smooth jittering shapes video generator

Render a 346x260 video of 6 objects on a black background. Each object
"jitters" one pixel around its home position, cycling through the eight
neighbouring pixels in a circular pattern (the 4 orthogonal directions +
the 4 diagonals). Objects start at different phases and spin in different
directions, so the whole field shimmers with 1-pixel motion

Returns .mp4 video and scene_truth.csv (object ID, name, x, y) for ground truth.
"""

import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

# Ground truth 
from ground_truth_helpers import build_mask, centroids, save_truth, oracle

NAMES = ["apple", "bottle", "star", "heart", "diamond", "mushroom"]

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
OUT_PATH        = f"{N_OBJECTS}_objects_{WIDTH}x{HEIGHT}.mp4"

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
    cx, cy, r = s/2, s/2, s*0.34# the join
    d.polygon([(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)],
              fill=(80,205,215))


def _mushroom(d, s):
    cx, cy = s/2, s/2
    d.pieslice([cx-s*0.34, cy-s*0.30, cx+s*0.34, cy+s*0.20],
               180, 360, fill=(210,60,60))
    d.rounded_rectangle([cx-s*0.16, cy-s*0.06, cx+s*0.16, cy+s*0.32],
                        radius=s*0.06, fill=(235,225,200))
DRAWERS = [_apple, _bottle, _star, _heart, _diamond, _mushroom]


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

    # Stagger each object's starting direction and spin sense so they all
    # move differently and together cover orthogonal + diagonal directions.
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