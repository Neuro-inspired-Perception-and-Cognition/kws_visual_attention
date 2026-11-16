"""Batch definitions (Set 1 list from the PDF) and command lists ('LISTS OF COMMANDS TO FEED AND COMPARE AGAINST')."""
C4 = ["top-left", "top-right", "bottom-left", "bottom-right"]
C6 = ["top-left", "top-mid", "top-right", "bottom-left", "bottom-mid", "bottom-right"]
O6 = ["apple", "bottle", "star", "heart", "diamond", "mushroom"]
O9 = ["apple", "bottle", "star", "heart", "diamond", "mushroom", "moon", "tree", "mug"]
GT = "stimuli/ground_truth_masks/"
BATCHES = {
    1:  ("data/4_circles_mono_nobg_346x260.npy",       GT + "4_circles_mono_nobg_346x260.mask.npy", C4),
    2:  ("data/4_circles_mono_bg_jitter_346x260.npy",  GT + "4_circles_mono_bg_jitter_346x260.mask.npy", C4),
    3:  ("data/6_circles_tex_346x260.npy",             GT + "6_circles_tex_346x260.mask.npy", C6),
    4:  ("data/6_circles_bg_jitter_346x260.npy",       GT + "6_circles_mono_bg_jitter_346x260.mask.npy", C6),
    5:  ("data/6_objects_mono_nobg_346x260.npy",       GT + "6_objects_mono_nobg_346x260.mask.npy", O6),
    6:  ("data/6_objects_mono_bg_jitter_346x260.npy",  GT + "6_objects_mono_bg_jitter_346x260.mask.npy", O6),
    7:  ("data/6_objects_color_nobg_346x260.npy",      GT + "6_objects_color_nobg_346x260.mask.npy", O6),
    8:  ("data/6_objects_color_bg_jitter_346x260.npy", GT + "6_objects_color_bg_jitter_346x260.mask.npy", O6),
    9:  ("data/9_objects_color_nobg_346x260.npy",      GT + "9_objects_color_nobg_346x260.mask.npy", O9),
    10: ("data/9_objects_color_bg_jitter_346x260.npy", GT + "9_objects_color_bg_jitter_346x260.mask.npy", O9),
}
L, R, U, D = "left", "right", "up", "down"
CMDS = {
    1:  [L, D, U, R, D, U, D, L, R, D],
    2:  [D, L, D, R, L, D, U, U, L, R],
    3:  [U, L, D, R, R, U, D, L, U, D],
    4:  [R, D, L, U, D, R, U, L, R, D],
    5:  [D, U, R, L, L, D, U, R, D, R],
    6:  [L, U, R, D, U, L, D, U, R, U],
    7:  [R, R, D, L, U, D, L, U, D, R],
    8:  [U, D, L, L, R, U, D, R, U, L],
    9:  [D, R, U, L, D, D, R, U, L, R],
    10: [L, D, R, U, L, R, D, U, R, L],
}
DESC = {1: "4 circles/nobg/mono", 2: "4 circles/bg/mono", 3: "6 circles/nobg/mono", 4: "6 circles/bg/mono",
        5: "6 objects/nobg/mono", 6: "6 objects/bg/mono", 7: "6 objects/nobg/color", 8: "6 objects/bg/color",
        9: "9 objects/nobg/color", 10: "9 objects/bg/color"}
