"""
Smooth jittering shapes video generator - variable object count (4, 6 or 9)

Render a 346x260 video of n_objects different objects over a pencil-scratch
background. Each object "jitters" one pixel around its home position, cycling
through the eight neighbouring pixels in a circular pattern (the 4 orthogonal
directions + the 4 diagonals). The background can jitter too, locked to the
motion of one object (bg_follow).

Layouts:  4 -> 2x2 grid,  6 -> 3x2 grid,  9 -> 3x3 grid

Returns .mp4 video plus <stem>.mask.npy and <stem>.truth.csv for ground truth.
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio.v2 as imageio

# Ground truth
from ground_truth_helpers import build_mask, centroids, save_truth

# choose 4, 6 or 9
n_objects = 9

# "objects" = the nine different shapes; "circles" = identical textured disks,
# named by their position in the grid
shape_kind = "objects"     # "objects" or "circles"
circle_r   = 0.36          # disk radius as a fraction of the (scaled) tile

object_names = ["apple", "bottle", "star", "heart", "diamond", "mushroom",
                "moon", "tree", "mug"]

grids = {4: (2, 2), 6: (3, 2), 9: (3, 3)}   # n_objects -> (cols, rows)


def position_names(n):
    """top-left, top-mid, ... for identical objects."""
    cols, rows = grids[n]
    row_words = {2: ["top", "bottom"], 3: ["top", "middle", "bottom"]}[rows]
    col_words = {2: ["left", "right"], 3: ["left", "mid", "right"]}[cols]
    return [f"{r}-{c}" for r in row_words for c in col_words][:n]


names = object_names if shape_kind == "objects" else position_names(n_objects)

# pencil-scratch background motion (only used when background = True)
bg_jitter = True    # False = static hatching, True = hatching jitters
bg_follow = 0       # which object the background copies (index into names); None = own phase

# Parameters
width           = 346       # output width
height          = 260       # output height
sprite_size     = 100       # each object is drawn on a sprite_size x sprite_size canvas
fps             = 20
frames_per_step = 1         # frames held at each of the 8 positions (>=1)
n_steps         = 200       # number of jitter steps -> n_steps*frames_per_step frames
color           = True      # True = colored objects, False = monochrome white silhouettes
background      = True
out_dir         = "stimuli/frame_videos"         # the .mp4 saves here
truth_dir       = "stimuli/ground_truth_masks"   # the .mask.npy + .truth.csv save here
out_name        = (f"{n_objects}_{'objects' if shape_kind == 'objects' else 'circles'}_"
                   f"{'color' if color else 'mono'}_"
                   f"{('bg_jitter' if bg_jitter else 'bg') if background else 'nobg'}_"
                   f"{width}x{height}.mp4")
out_path        = os.path.join(out_dir, out_name)
stem            = os.path.splitext(out_name)[0]  # shared basename: video <-> its truth

# size: 3 rows leave ~86 px per cell, so default a bit smaller for 9
scale_by_n      = {4: 0.6, 6: 0.6, 9: 0.55}
scale           = scale_by_n[n_objects]
scales          = [1.0] * 9                      # per object, multiplied by scale


# texture (interior events)
textured        = True      # False -> flat fills, hollow event rings
tex_cell        = 3         # speckle cell size in px
tex_min         = 0.55      # darkest the speckle drives a pixel (1.0 = unchanged)
proc_downsample = 2         # must equal DOWNSAMPLE in the controller

# pencil-scratch background (only used when background = True)
bg_strokes      = 250       # how many scratch strokes
bg_lo, bg_hi    = 200, 250  # grey range of a stroke
bg_len          = (12, 55)  # stroke length in px
bg_width        = 1         # stroke width in px
bg_angles       = (-35, 20) # degrees: hatching leans these two ways
bg_blur         = 1.2       # low = crisp crossed lines; high = soft haze
bg_seed         = 7

# 8 unit displacements around a circle of radius 1 px
circle = [
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
# Each drawer paints one object centred on an s x s transparent tile.
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
    cx, cy, big_r, small_r = s/2, s/2, s*0.36, s*0.15
    pts = []
    for k in range(10):
        ang = -math.pi/2 + k*math.pi/5
        rad = big_r if k % 2 == 0 else small_r
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


def _moon(d, s):
    # crescent: full disc, then punch out an offset disc with transparent fill
    cx, cy, r = s/2, s/2, s*0.34
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(240,230,150))
    ox = s*0.18
    d.ellipse([cx-r+ox, cy-r-s*0.06, cx+r+ox, cy+r-s*0.06], fill=(0,0,0,0))


def _tree(d, s):
    cx = s/2
    d.rectangle([cx-s*0.06, s*0.68, cx+s*0.06, s*0.88], fill=(120,72,34))
    d.polygon([(cx, s*0.10), (cx+s*0.30, s*0.72), (cx-s*0.30, s*0.72)],
              fill=(60,160,75))


def _mug(d, s):
    cx = s/2
    d.ellipse([cx+s*0.10, s*0.36, cx+s*0.38, s*0.66],
              outline=(200,120,60), width=max(1, int(s*0.07)))
    d.rounded_rectangle([cx-s*0.26, s*0.24, cx+s*0.18, s*0.80],
                        radius=s*0.05, fill=(200,120,60))


def _circle(d, s):
    r = s * circle_r
    cx = cy = s / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(245, 245, 245))


object_drawers = [_apple, _bottle, _star, _heart, _diamond, _mushroom,
                  _moon, _tree, _mug]
drawers = object_drawers if shape_kind == "objects" else [_circle] * 9


def make_background(pad=2):
    """Pencil hatching, drawn once. Built `pad` px larger on each side so it can
    be jittered without exposing an uncovered edge."""
    rng = np.random.default_rng(bg_seed)
    img = Image.new("L", (width + 2 * pad, height + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    for _ in range(bg_strokes):
        x0, y0 = rng.uniform(0, width + 2 * pad), rng.uniform(0, height + 2 * pad)
        ang = np.radians(rng.choice(bg_angles) + rng.uniform(-8, 8))
        ln = rng.uniform(*bg_len)
        d.line([x0, y0, x0 + ln * np.cos(ang), y0 + ln * np.sin(ang)],
               fill=int(rng.integers(bg_lo, bg_hi)), width=bg_width)
    return img.filter(ImageFilter.GaussianBlur(bg_blur)) if bg_blur else img


def make_sprite(drawer, seed=0, scale=1.0):
    """Draw the object at `scale`, then modulate its interior with a fixed speckle."""
    inner = max(8, int(round(sprite_size * scale)))
    tile = Image.new("RGBA", (inner, inner), (0, 0, 0, 0))
    drawer(ImageDraw.Draw(tile), inner)

    img = Image.new("RGBA", (sprite_size, sprite_size), (0, 0, 0, 0))
    off = (sprite_size - inner) // 2       # centre it; negative if scale > 1 (crops)
    img.paste(tile, (off, off))
    if inner > sprite_size:
        print(f"  warning: scale {scale} exceeds the {sprite_size}px tile - shape clipped; "
              f"raise sprite_size instead")

    if not color:                              # monochrome: white silhouette
        arr = np.asarray(img).copy()
        arr[..., :3] = 255                     # alpha untouched -> shape unchanged
        img = Image.fromarray(arr, "RGBA")

    if not textured:
        return img

    rng = np.random.default_rng(seed)
    small = max(1, sprite_size // tex_cell)
    cells = rng.uniform(tex_min, 1.0, size=(small, small))
    tex = np.array(Image.fromarray((cells * 255).astype(np.uint8))
                   .resize((sprite_size, sprite_size), Image.NEAREST)) / 255.0

    arr = np.asarray(img).astype(float)
    arr[..., :3] *= tex[..., None]            # alpha untouched -> mask unchanged
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")


# home layout: cols x rows, picked from n_objects. One spacing for both axes,
# so rows are as far apart as columns, and the grid is centred in the frame.
def home_positions(n=n_objects):
    cols, rows = grids[n]
    step = min(width / cols, height / rows)
    xs = [width / 2 + (c - (cols - 1) / 2) * step for c in range(cols)]
    ys = [height / 2 + (r - (rows - 1) / 2) * step for r in range(rows)]
    gx = xs[1] - xs[0] if len(xs) > 1 else 0.0
    gy = ys[1] - ys[0] if len(ys) > 1 else 0.0
    print(f"layout {cols}x{rows}: column spacing {gx:.1f} px, row spacing {gy:.1f} px "
          f"({gx / proc_downsample:.1f} and {gy / proc_downsample:.1f} on the processing grid)")
    print(f"  x = {[round(v, 1) for v in xs]}")
    print(f"  y = {[round(v, 1) for v in ys]}")
    print(f"  margins to the frame edge: left/right {xs[0]:.1f}/{width - xs[-1]:.1f}, "
          f"top/bottom {ys[0]:.1f}/{height - ys[-1]:.1f} px")
    return [(x, y) for y in ys for x in xs][:n]


def save_scene_truth(sprites, homes, stem=stem, truth_dir=truth_dir):
    """The video and its ground truth live in different folders but share this
    basename, so a clip is always paired with its own mask.
    Unaffected by the background: the mask comes from each sprite's alpha.
    """
    os.makedirs(truth_dir, exist_ok=True)
    base = os.path.join(truth_dir, stem)
    mask = build_mask(sprites, homes, width, height, sprite_size)[::proc_downsample, ::proc_downsample]
    np.save(f"{base}.mask.npy", mask)                                   # per-pixel footprint (scoring)
    save_truth(centroids(mask), names[:len(sprites)], f"{base}.truth.csv")  # readable centroids
    print(f"saved {base}.mask.npy + {base}.truth.csv  (grid {mask.shape[1]}x{mask.shape[0]})")
    return mask


# render
def render(out_path=out_path):
    if n_objects not in grids:
        raise ValueError(f"n_objects must be one of {sorted(grids)}, got {n_objects}")
    if bg_follow is not None and bg_follow >= n_objects:
        raise ValueError(f"bg_follow={bg_follow} but only {n_objects} objects")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sprites = [make_sprite(fn, seed=i, scale=scale * scales[i])
               for i, fn in enumerate(drawers[:n_objects])]
    homes   = home_positions()

    # move objs differently and together cover orthogonal + diagonal directions.
    # phases = [(i * 3) % 8 for i in range(n_objects)]
    # signs  = [1 if i % 2 == 0 else -1 for i in range(n_objects)]
    # move everything in the same direction
    phases = [0] * n_objects  # set everything moving globally in the same direction
    signs  = [1] * n_objects  # set everything moving globally in the same direction

    # background copies the phase + direction of the followed object
    if bg_follow is not None:
        bg_phase, bg_sign = phases[bg_follow], signs[bg_follow]
    else:
        bg_phase, bg_sign = 5, 1    # its own phase so it doesn't track any object

    pad = 2
    bg  = make_background(pad) if background else None

    frames = []
    for step in range(n_steps):
        if bg is not None:
            if bg_jitter:
                bx, by = circle[(bg_phase + bg_sign * step) % 8]
            else:
                bx = by = 0
            layer = Image.new("L", (width, height), 0)
            layer.paste(bg, (-pad + bx, -pad + by))
            canvas = layer.convert("RGB")
        else:
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
        for (hx, hy), sprite, ph, sg in zip(homes, sprites, phases, signs):
            dx, dy = circle[(ph + sg * step) % 8]
            px = int(round(hx - sprite_size / 2 + dx))
            py = int(round(hy - sprite_size / 2 + dy))
            canvas.paste(sprite, (px, py), sprite)   # alpha channel = mask
        frames.extend([np.asarray(canvas)] * frames_per_step)

    writer = imageio.get_writer(
        out_path, format="FFMPEG", mode="I", fps=fps,
        codec="libx264rgb",
        output_params=["-crf", "0"],
        macro_block_size=None,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()

    bg_desc = ("off" if not background else
               "on, static" if not bg_jitter else
               f"on, following {names[bg_follow]}" if bg_follow is not None else
               "on, own phase")
    print(f"wrote {out_path}: {len(frames)} frames, {width}x{height}, {fps} fps"
          f"  ({n_objects} {'objects' if shape_kind == 'objects' else 'circles'}, "
          f"{grids[n_objects][0]}x{grids[n_objects][1]} grid, "
          f"{'colour' if color else 'mono'}, "
          f"background {bg_desc}, "
          f"texture {'on' if textured else 'off'})")
    return sprites, homes


if __name__ == "__main__":
    sprites, homes = render()
    save_scene_truth(sprites, homes)