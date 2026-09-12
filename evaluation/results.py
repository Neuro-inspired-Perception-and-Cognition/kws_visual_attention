"""
Loads a scene mask (saved when a clip is rendered) and run_log.csv,
joins them, writes results.csv.

Reads the run as a TRAJECTORY of object IDs: one line per typed command,
showing which object the fovea ended up on. A command that failed to move the
fovea off its current object is reported as "FAILED - did not move" rather than
being skipped.
"""

import csv
import numpy as np

from stimuli.ground_truth_helpers import centroids, score_trajectory

# identical circles, identified by position (matches render_stimuli_circles.py)
NAMES = ["top-left", "top-mid", "top-right", "bottom-left", "bottom-mid", "bottom-right"]

# --- resolution reconcile --------------------------------------------------
# The mask is saved at PROC_DOWNSAMPLE in the generator; the run uses DOWNSAMPLE
# in the controller. Bring the mask to the SAME grid so mask[y,x] and the log
# agree. When both are 2, step = 1 and this is a no-op.
MASK_DS = 2     # PROC_DOWNSAMPLE used when the mask was saved
RUN_DS  = 2     # DOWNSAMPLE used in the controller for this run
step = RUN_DS // MASK_DS

mask  = np.load("6_circles_tex_346x260.mask.npy")[::step, ::step]
truth = centroids(mask)

trials = [{"direction": r["direction"], "k": int(r["k"]),
           "ref":   (float(r["ref_x"]),   float(r["ref_y"])),
           "fovea": (float(r["fovea_x"]), float(r["fovea_y"]))}
          for r in csv.DictReader(open("run_log.csv"))]

score_trajectory(mask, truth, trials, names=NAMES, out_path="results.csv")