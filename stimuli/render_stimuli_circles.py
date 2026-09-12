"""
Smooth jittering circles video generator (textured interiors)

Render a 346x260 video of 6 textured circles on a black background. Each circle
"jitters" one pixel around its home position, cycling through the eight
neighbouring pixels in a circular pattern.

Returns .mp4 video plus <stem>.mask.npy and <stem>.truth.csv for ground truth.
"""

import numpy as np
from PIL import Image, ImageDraw
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
CIRCLE_R        = 0.30      # circle radius as a fraction of SPRITE
FPS             = 20
FRAMES_PER_STEP = 1         # frames held at each of the 8 positions (>=1)
N_STEPS         = 200       # number of jitter steps -> N_STEPS*FRAMES_PER_STEP frames
OUT_PATH        = f"{N_OBJECTS}_circles_tex_{WIDTH}x{HEIGHT}.mp4"

# texture: coarse enough to survive downsampling, bright enough to stay visible
TEX_LO, TEX_HI  = 110, 256  # grey range of the speckle
TEX_CELL        = 3         # speckle cell size in px (1 = per-pixel noise)
PROC_DOWNSAMPLE = 2         # must equal DOWNSAMPLE in the controller

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


def make_sprite(seed):
    """One textured white disk on a transparent SPRITE x SPRITE tile.

    The texture is generated ONCE per sprite (fixed seed) so it travels with the
    object as it jitters. Random-per-frame noise would not do this -- it would
    look like static and swamp the objects.
    """
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


def save_scene_truth(sprites, homes, stem):
    """stem = the clip's base name, e.g. 'scene01' (same as scene01.mp4)."""
    mask = build_mask(sprites, homes, WIDTH, HEIGHT, SPRITE)[::PROC_DOWNSAMPLE, ::PROC_DOWNSAMPLE]
    np.save(f"{stem}.mask.npy", mask)                        # per-pixel footprint (scoring)
    save_truth(centroids(mask), NAMES, f"{stem}.truth.csv")  # readable centroids
    print(f"saved {stem}.mask.npy + {stem}.truth.csv  (grid {mask.shape[1]}x{mask.shape[0]})")
    return mask


# render
def render(out_path=OUT_PATH):
    sprites = [make_sprite(seed=i) for i in range(N_OBJECTS)]   # different texture each
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

    print(f"wrote {out_path}: {len(frames)} frames, {WIDTH}x{HEIGHT}, {FPS} fps")
    return sprites, homes


if __name__ == "__main__":
    sprites, homes = render()
    save_scene_truth(sprites, homes, OUT_PATH.rsplit(".", 1)[0])