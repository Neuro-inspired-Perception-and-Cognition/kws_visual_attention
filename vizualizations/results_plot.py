"""
Read the trial CSVs written by the experiment script and write paper-ready PNGs
to vizualizations/figures/. Looks for the CSVs in vizualizations/results/ first,
then in results/ at the project root, and reports which it used. Both are
resolved next to this file, so it runs from any working directory.

    python vizualizations/results_plot.py

The CSVs do not all have to share a schema. Every file is read for whatever
columns it has, the union is pooled, and each figure is drawn only if its
columns are present — otherwise it is skipped with a line saying what was
missing. Only `verdict` (or `correct`) is needed to get anything at all; the
rest widens what can be drawn. On startup the script prints which columns it
found, in how many files, and what each absent one costs.

Figures produced:
    accuracy_written.png      accuracy per batch, simulated events vs camera
    accuracy_spoken.png       the same for spoken commands via the NAS-KWS-FPGA
    outcomes.png              per stimulus, what happened to every command
    confusion_matrix.png      expected object vs landed object, row-normalised
    accuracy_by_command.png   accuracy per commanded direction
    trajectories_batch{n}.png fovea path per trial, one panel per trial
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

here = os.path.dirname(os.path.abspath(__file__))
# the experiment script writes to results/ beside itself, at the project root;
# whichever of these holds CSVs is used
results_dirs = [os.path.join(here, "results"),
                os.path.join(os.path.dirname(here), "results")]
figures_dir = os.path.join(here, "figures")
dpi = 300

# optional: batch -> ground-truth mask, drawn behind the trajectories
mask_paths = {
    # 6: "stimuli/ground_truth_masks/6_objects_color_bg_jitter_346x260.mask.npy",
}

# ---------------------------------------------------------------- EDIT THIS
# What each batch number means. This is the only place batches are named: the
# numbers themselves come from the `batch` column, so nothing else needs
# changing when a batch is added. Write the description the way it should read
# on a figure. A batch with no entry here still plots, labelled by its number,
# and the script prints which ones are unnamed when it runs.
batch_labels = {
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
    7: "",
    8: "",
    9: "",
    10: "",
}

# Which batches to plot. None means every batch found in the CSVs. Set a list
# to narrow it — e.g. [1, 2, 3, 4] plots only those four, in that order, in
# every figure. Nothing else needs changing: the camera and the converter are
# already the two bars in each accuracy figure.
batches_to_plot = None
# --------------------------------------------------------------------------

# the two columns that split the experiment: how the command arrived, and where
# the events came from
command_labels = {
    0: "Written commands",
    1: "Spoken commands (NAS-KWS-FPGA)",
}
visual_labels = {
    0: "IEBCS simulated events",
    1: "DAVIS346 camera",
}
visual_colors = {0: "#2a78d6", 1: "#eb6834"}

# the four outcomes a command can have, in stacking order. Named as the reader
# should understand them, not as the verdict strings spell them.
categories = ["reached the target", "went to the wrong object",
              "no target that way", "did not move"]
category_colors = {
    "reached the target": "#1baf7a",
    "went to the wrong object": "#eb6834",
    "no target that way": "#2a78d6",
    "did not move": "#eda100",
}

# what each optional column buys, printed when it is absent
column_value = {
    "batch": "every per-batch figure",
    "command": "accuracy_by_command.png",
    "expected_id": "confusion_matrix.png",
    "landed_id": "confusion_matrix.png, and refusals cannot be verified",
    "linguistic": "the written/spoken split",
    "visual": "the simulated/camera bars",
    "ref_x": "trajectories",
    "ref_y": "trajectories",
    "fovea_x": "trajectories",
    "fovea_y": "trajectories",
    "trial": "trajectory panels are keyed by file instead",
    "accuracy": "the stored-vs-recomputed accuracy check",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": dpi,
    "savefig.bbox": "tight",
    "font.family": "serif",
    # Liberation/Nimbus Roman are the metric-compatible Linux stand-ins, used
    # only if Times New Roman itself isn't installed
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
})

rng = np.random.default_rng(0)


def have(df, *cols):
    """True when every column exists and holds at least one real value."""
    return all(c in df.columns and df[c].notna().any() for c in cols)


def category_of(verdict):
    if not isinstance(verdict, str):
        return None
    if verdict == "OK":
        return "reached the target"
    if verdict.startswith("WRONG"):
        return "went to the wrong object"
    if "did not move" in verdict:
        return "did not move"
    return "no target that way"


def category_from_ids(row):
    """The same four outcomes, worked out from the ids when verdict is absent."""
    expected, landed, start = row.get("expected_id"), row.get("landed_id"), \
        row.get("start_id")
    if pd.isna(expected) or pd.isna(landed):
        return None
    if expected == 0:
        return "no target that way"
    if not pd.isna(start) and landed == start:
        return "did not move"
    if landed == expected:
        return "reached the target"
    return "went to the wrong object"


def split_trial_id(value):
    """trial_id is linguistic | visual | trial | batch, one digit each but the
    batch, which may be two. Returns {} for anything that doesn't parse."""
    text = str(value).strip()
    if not text.isdigit() or len(text) < 4:
        return {}
    return {"linguistic": int(text[0]), "visual": int(text[1]),
            "trial": int(text[2]), "batch": int(text[3:])}


def load(paths):
    files = []
    for path in paths:
        files = [f for f in sorted(glob.glob(os.path.join(path, "*.csv")))
                 if not f.endswith("_unscored.csv")]
        if files:
            print(f"reading {path}/")
            break
    if not files:
        raise SystemExit("no CSVs in " + " or ".join(p + "/" for p in paths))

    frames, unreadable = [], []
    for f in files:
        try:
            part = pd.read_csv(f, dtype={"trial_id": str})
        except Exception as exc:                    # malformed beyond rescue
            unreadable.append(f"{os.path.basename(f)}: {exc}")
            continue
        if part.empty:
            unreadable.append(f"{os.path.basename(f)}: no rows")
            continue
        # when the rows carry more fields than the header names, pandas turns
        # the surplus into a row index and every column silently shifts
        if not part.index.equals(pd.RangeIndex(len(part))):
            unreadable.append(
                f"{os.path.basename(f)}: rows have more fields than the header "
                "names, so the columns are shifted — re-run or restore it")
            continue
        part["source_file"] = os.path.basename(f)
        frames.append(part)

    if unreadable:
        print("skipped " + str(len(unreadable)) + " file(s):")
        for line in unreadable:
            print("   " + line)
    if not frames:
        raise SystemExit("nothing readable to plot")

    # differing schemas are fine: the union is taken and absent cells are blank
    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"loaded {len(frames)} trials, {len(df)} steps, "
          f"{len(df.columns) - 1} columns")
    return df


def derive(df):
    """Fill in what can be worked out, and drop rows that carry no outcome."""
    # the split digits, for files written before those columns existed
    if "trial_id" in df.columns:
        parsed = pd.DataFrame([split_trial_id(v) for v in df["trial_id"]],
                              index=df.index)
        for col in ("linguistic", "visual", "trial", "batch"):
            if col in parsed.columns:
                if col in df.columns:
                    df[col] = df[col].fillna(parsed[col])
                else:
                    df[col] = parsed[col]

    # outcome: the verdict string if it is there, else the ids
    if "verdict" in df.columns:
        df["category"] = df["verdict"].map(category_of)
    else:
        df["category"] = None
    if df["category"].isna().any() and have(df, "expected_id", "landed_id"):
        filled = df[df["category"].isna()].apply(category_from_ids, axis=1)
        df.loc[df["category"].isna(), "category"] = filled

    dropped = int(df["category"].isna().sum())
    if dropped:
        files = sorted(df.loc[df["category"].isna(), "source_file"].unique())
        print(f"dropped {dropped} row(s) with no readable outcome, from: "
              + ", ".join(files))
        df = df[df["category"].notna()].copy()
    if df.empty:
        raise SystemExit("no rows carry a verdict or an expected/landed pair")

    # correctness: the logged column wins, since it is what the run scored
    scored = df["correct"].astype("boolean") if "correct" in df.columns \
        else pd.Series(pd.NA, index=df.index, dtype="boolean")
    refused = df["category"] == "no target that way"
    if "landed_id" in df.columns:
        # where landed_id was logged, check the fovea really stayed off every
        # object; where it wasn't, the verdict is all there is to go on
        refused &= (df["landed_id"] == 0) | df["landed_id"].isna()
    df["correct"] = scored.fillna(
        (df["category"] == "reached the target") | refused).astype(bool)

    # one key per trial, so panels and means group correctly either way
    df["trial_key"] = df["trial_id"] if have(df, "trial_id") \
        else df["source_file"].str.replace(".csv", "", regex=False)
    return df


def report_columns(df):
    """Say what is available, and what each absent column costs."""
    absent = [c for c in column_value if not have(df, c)]
    if absent:
        print("columns not found (figures adapt):")
        for col in absent:
            print(f"   {col:<12} {column_value[col]}")
    thin = [c for c in column_value
            if c in df.columns and df[c].isna().any() and df[c].notna().any()]
    if thin:
        print("columns present in only some files: " + ", ".join(sorted(thin)))


def object_names(df):
    """id -> object name, but only where every batch agrees on the name.

    Batches can use different name lists for the same ids (batch 1's "object 2"
    may be a position, another batch's a fruit), so an id with more than one
    name is left to show as its number rather than one batch's label.
    """
    seen = {}
    pairs = [("start_id", "start"), ("expected_id", "expected"),
             ("landed_id", "landed")]
    for id_col, label_col in pairs:
        if not have(df, id_col, label_col):
            continue
        for oid, label in zip(df[id_col], df[label_col]):
            if oid and not pd.isna(oid) and isinstance(label, str) \
                    and "(" in label:
                seen.setdefault(int(oid), set()).add(
                    label.split("(", 1)[1].rstrip(")"))
    clashing = sorted(k for k, v in seen.items() if len(v) > 1)
    if clashing:
        print(f"object ids {clashing} are named differently in different "
              "batches — the confusion matrix shows their numbers instead")
    return {0: "bg", **{k: next(iter(v)) for k, v in seen.items() if len(v) == 1}}


def per_trial_metrics(df):
    keys = ["trial_key"] + [c for c in ("batch", "trial", "linguistic", "visual")
                            if have(df, c)]
    agg = {"accuracy": ("correct", lambda s: 100 * s.mean()),
           "steps": ("category", "size")}
    if have(df, "landed_id"):
        # NaN means the trial never logged it, so it is left out of the share
        # rather than counted as "on an object"
        agg["on_object"] = ("landed_id",
                            lambda s: 100 * (s.dropna() != 0).mean()
                            if s.notna().any() else np.nan)
    if have(df, "accuracy"):
        agg["stored"] = ("accuracy", "first")

    out = df.groupby(keys, as_index=False, dropna=False).agg(**agg)
    if "stored" in out.columns:
        drift = out[(out["accuracy"].round(1) - out["stored"].round(1)).abs() > 0.05]
        if len(drift):
            print("\nrecomputed accuracy disagrees with the stored column:")
            print(drift[["trial_key", "accuracy", "stored"]].to_string(index=False))
            print("the scoring rule in the experiment script has changed; "
                  "update category_of/correct in this file to match.\n")
        out = out.drop(columns="stored")
    return out


def selected_batches(values):
    """The batches to show, in the order batches_to_plot gives them."""
    found = {int(b) for b in pd.Series(list(values)).dropna().unique()}
    if batches_to_plot is None:
        return sorted(found)
    return [b for b in batches_to_plot if b in found]


def apply_batch_filter(df):
    if batches_to_plot is None or not have(df, "batch"):
        return df
    keep = df[df["batch"].isin(batches_to_plot)]
    if keep.empty:
        raise SystemExit(
            f"batches_to_plot is {batches_to_plot} but the CSVs hold "
            + ", ".join(str(b) for b in selected_batches(df["batch"]))
            + " — nothing to plot")
    asked = [b for b in batches_to_plot if b not in set(df["batch"].dropna())]
    print(f"plotting batches {batches_to_plot}"
          + (f"; no trials for {asked}" if asked else ""))
    return keep


def batch_name(batch):
    """The description for a batch, or "" when it has not been filled in."""
    return str(batch_labels.get(int(batch), "")).strip()


def tick_labels(batches):
    return [f"{b}.  {batch_name(b)}".rstrip() for b in batches]


def report_batches(df):
    found = selected_batches(df["batch"])
    unnamed = [b for b in found if not batch_name(b)]
    print("batches in the data: " + ", ".join(str(b) for b in found))
    if unnamed:
        print("   not named yet: " + ", ".join(str(b) for b in unnamed)
              + " — fill them into batch_labels at the top of this file")


def fig_accuracy_by_source(trials, linguistic, path):
    """Accuracy per stimulus, simulated events beside camera events.

    One figure per command type. Returns False when no trial used this command
    type, so the caller can say so instead of writing an empty figure.
    """
    sub = trials[trials["linguistic"] == linguistic]
    if sub.empty:
        return False

    batches = selected_batches(sub["batch"])
    x = np.arange(len(batches))
    bar = 0.38
    fig, ax = plt.subplots(figsize=(max(4.5, 0.72 * len(batches) + 2.2), 3.6))

    drawn = []
    for k, (visual, label) in enumerate(visual_labels.items()):
        means, positions = [], []
        for i, b in enumerate(batches):
            cell = sub[(sub["batch"] == b) & (sub["visual"] == visual)]
            if cell.empty:          # not run yet: leave the slot empty
                continue
            means.append(cell["accuracy"].mean())
            positions.append(i + (k - 0.5) * bar)
        if not positions:           # no trial used this source: no legend entry
            continue
        ax.bar(positions, means, width=bar * 0.92, color=visual_colors[visual],
               zorder=2)
        drawn.append(Patch(facecolor=visual_colors[visual], label=label))

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b)) for b in batches])
    ax.set_xlabel("stimulus batch")
    ax.set_ylabel("commands that reached\nthe target object (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(command_labels[linguistic], fontsize=10, loc="left", pad=30)
    ax.grid(axis="y", color="#ececec", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=drawn, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.005),
              handlelength=1.1, handleheight=1.1, columnspacing=1.6)
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_outcomes(df, path):
    """One bar per stimulus, split into what happened to each command."""
    batches = selected_batches(df["batch"])
    share = np.zeros((len(batches), len(categories)))
    for i, b in enumerate(batches):
        sub = df[df["batch"] == b]
        for j, c in enumerate(categories):
            share[i, j] = 100 * (sub["category"] == c).mean()

    fig, ax = plt.subplots(figsize=(7.6, 0.44 * len(batches) + 1.6))
    y = np.arange(len(batches))
    left = np.zeros(len(batches))
    for j, c in enumerate(categories):
        ax.barh(y, share[:, j], left=left, height=0.62, color=category_colors[c],
                edgecolor="white", linewidth=1.4, label=c, zorder=2)
        left += share[:, j]

    # only the headline number is direct-labelled; the rest is legend + table
    hit = categories.index("reached the target")
    for i, b in enumerate(batches):
        ax.text(101.5, i, f"{share[i, hit]:.0f}%", va="center", fontsize=8,
                color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels([int(b) for b in batches]), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xlabel("% of commands given")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#ececec", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01),
              handlelength=1.1, handleheight=1.1, columnspacing=1.6)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion_matrix(df, names, path):
    pairs = df.dropna(subset=["expected_id", "landed_id"])
    ids = sorted(set(pairs["expected_id"]) | set(pairs["landed_id"]))
    counts = np.zeros((len(ids), len(ids)))
    index = {v: i for i, v in enumerate(ids)}
    for e, l in zip(pairs["expected_id"], pairs["landed_id"]):
        counts[index[e], index[l]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    size = 0.52 * len(ids) + 1.8
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    labels = [names.get(int(i), str(int(i))) for i in ids]
    ax.set_xticks(range(len(ids)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(ids)), labels)
    ax.set_xlabel("landed")
    ax.set_ylabel("expected")
    for i in range(len(ids)):
        for j in range(len(ids)):
            if counts[i, j]:
                ax.text(j, i, int(counts[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if frac[i, j] > 0.55 else "#1a1a1a")
    ax.set_xticks(np.arange(len(ids) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ids) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def fig_accuracy_by_command(df, path):
    order = ["left", "right", "up", "down"]
    present = [d for d in order if d in set(df["command"].dropna())]
    if not present:
        return False
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(present))
    values, counts = [], []
    for d in present:
        sub = df[df["command"] == d]
        values.append(100 * sub["correct"].mean())
        counts.append(len(sub))
    ax.bar(x, values, width=0.56, color="#dce4ec", edgecolor="#39566f",
           linewidth=0.8, zorder=1)
    for i, (v, n) in enumerate(zip(values, counts)):
        ax.text(i, v + 2, f"n={n}", ha="center", fontsize=7, color="#555555")
    ax.set_xticks(x, present)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% correct")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_trajectories(df, batch, names, path):
    sub = df[df["batch"] == batch].dropna(
        subset=["ref_x", "ref_y", "fovea_x", "fovea_y"])
    if sub.empty:
        return False
    panel_key = "trial" if have(sub, "trial") else "trial_key"
    panels = sorted(sub[panel_key].dropna().unique())
    mask = None
    if batch in mask_paths and os.path.exists(mask_paths[batch]):
        mask = np.load(mask_paths[batch])

    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.7 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for ax, t in zip(axes.flat, panels):
        ax.set_visible(True)
        steps = sub[sub[panel_key] == t]
        steps = steps.sort_values("step") if have(steps, "step") else steps
        if mask is not None:
            ax.imshow(mask > 0, cmap="Greys", alpha=0.18, interpolation="nearest")
            for oid in range(1, int(mask.max()) + 1):
                ys, xs = np.where(mask == oid)
                if len(xs):
                    ax.text(xs.mean(), ys.mean(), names.get(oid, str(oid)),
                            ha="center", va="center", fontsize=6, color="#333333")
            ax.set_xlim(0, mask.shape[1])
            ax.set_ylim(mask.shape[0], 0)
        else:
            pad = 8
            xs = pd.concat([steps["ref_x"], steps["fovea_x"]])
            ys = pd.concat([steps["ref_y"], steps["fovea_y"]])
            ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_ylim(ys.max() + pad, ys.min() - pad)

        stalls = 0
        for _, r in steps.iterrows():
            colour = category_colors[r["category"]]
            # travel_px is logged by the run; fall back to the coordinates
            travel = r["travel_px"] if "travel_px" in steps.columns \
                and not pd.isna(r["travel_px"]) \
                else np.hypot(r["fovea_x"] - r["ref_x"], r["fovea_y"] - r["ref_y"])
            if travel < 1.0:
                stalls += 1
                ax.plot(r["ref_x"], r["ref_y"], marker="x", markersize=6,
                        color=colour, markeredgewidth=1.4)
                continue
            ax.annotate("", xy=(r["fovea_x"], r["fovea_y"]),
                        xytext=(r["ref_x"], r["ref_y"]),
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                        linewidth=1.3, shrinkA=0, shrinkB=0))
            step_label = f"{int(r['step'])} " if "step" in steps.columns \
                and not pd.isna(r["step"]) else ""
            command = r["command"] if "command" in steps.columns \
                and isinstance(r["command"], str) else ""
            if step_label or command:
                ax.text((r["ref_x"] + r["fovea_x"]) / 2,
                        (r["ref_y"] + r["fovea_y"]) / 2,
                        f"{step_label}{command}".strip(), fontsize=6,
                        color=colour, ha="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.75))

        first = steps.iloc[0]
        ax.plot(first["ref_x"], first["ref_y"], marker="o", markersize=5,
                markerfacecolor="none", markeredgecolor="#333333",
                markeredgewidth=1.0)
        acc = 100 * steps["correct"].mean()
        name = f"trial {int(t)}" if panel_key == "trial" else str(t)
        title = f"{name} — {acc:.0f}% correct"
        if stalls:
            title += f", {stalls} stalled"
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

    handles = [plt.Line2D([], [], color=category_colors[c], linewidth=1.6, label=c)
               for c in categories]
    fig.legend(handles=handles, ncol=len(categories), loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"batch {int(batch)}  {batch_name(batch)}".rstrip(), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def summary(trials, df):
    rows = []
    group = selected_batches(trials["batch"]) if have(trials, "batch") \
        else [None]
    for b in group:
        t = trials if b is None else trials[trials["batch"] == b]
        s = df if b is None else df[df["batch"] == b]
        row = {
            "batch": "all" if b is None else int(b),
            "stimulus": "" if b is None else batch_name(b),
            "trials": len(t),
            "steps": len(s),
            "accuracy": round(t["accuracy"].mean(), 1),
            "sd": round(t["accuracy"].std(ddof=0), 1),
        }
        if "on_object" in t.columns:
            row["on_object"] = round(t["on_object"].mean(), 1)
        row["did_not_move"] = round(
            100 * (s["category"] == "did not move").mean(), 1)
        row["wrong_object"] = round(
            100 * (s["category"] == "went to the wrong object").mean(), 1)
        rows.append(row)
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def main():
    os.makedirs(figures_dir, exist_ok=True)
    df = apply_batch_filter(derive(load(results_dirs)))
    report_columns(df)
    if have(df, "batch"):
        report_batches(df)
    names = object_names(df)
    trials = per_trial_metrics(df)

    def out(name):
        return os.path.join(figures_dir, name)

    if have(trials, "batch", "linguistic", "visual"):
        for linguistic, stem in ((0, "accuracy_written"), (1, "accuracy_spoken")):
            if not fig_accuracy_by_source(trials, linguistic, out(stem + ".png")):
                print(f"no trials with {command_labels[linguistic].lower()} — "
                      f"{stem}.png skipped")
        absent = [(command_labels[l], visual_labels[v])
                  for l in sorted(trials["linguistic"].dropna().unique())
                  for v in visual_labels
                  if trials[(trials["linguistic"] == l)
                            & (trials["visual"] == v)].empty]
        if absent:
            print("no trials yet for: "
                  + "; ".join(f"{c} + {s}" for c, s in absent))
    else:
        print("no batch/linguistic/visual columns — "
              "accuracy_written.png and accuracy_spoken.png skipped")

    if have(df, "batch"):
        fig_outcomes(df, out("outcomes.png"))
    else:
        print("no batch column — outcomes.png skipped")

    if have(df, "expected_id", "landed_id"):
        fig_confusion_matrix(df, names, out("confusion_matrix.png"))
    else:
        print("no expected_id/landed_id columns — confusion_matrix.png skipped")

    if have(df, "command"):
        fig_accuracy_by_command(df, out("accuracy_by_command.png"))
    else:
        print("no command column — accuracy_by_command.png skipped")

    if have(df, "ref_x", "ref_y", "fovea_x", "fovea_y") and have(df, "batch"):
        skipped = []
        for b in selected_batches(df["batch"]):
            name = f"trajectories_batch{int(b)}.png"
            if not fig_trajectories(df, b, names, out(name)):
                skipped.append(int(b))
        if skipped:
            print(f"no coordinates logged for batches {skipped} — "
                  "trajectories skipped")
    else:
        print("no ref_x/ref_y/fovea_x/fovea_y columns — trajectories skipped")

    summary(trials, df)
    print(f"\nfigures written to {figures_dir}/")


if __name__ == "__main__":
    main()