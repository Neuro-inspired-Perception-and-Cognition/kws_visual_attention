"""
Loads a scene mask in .npy (saved when a clip is rendered) and run_log.csv,
joins them, writes results.csv. 
"""

import csv
import sys
import numpy as np

from stimuli.ground_truth_helpers import centroids, score_run

NAMES = ["apple", "bottle", "star", "heart", "diamond", "mushroom"]

# ground truth
mask  = np.load("/home/rocharay/kws_attention/6_objects_346x260.mask.npy")
truth = centroids(mask)

trials = [{"direction": r["direction"], "k": int(r["k"]),
           "ref":   (float(r["ref_x"]),   float(r["ref_y"])),
           "fovea": (float(r["fovea_x"]), float(r["fovea_y"]))}
          for r in csv.DictReader(open("run_log.csv"))]

score_run(mask, truth, trials, names=NAMES, out_path="results.csv")