"""
Read the trial CSVs in vizualizations/results/ and write paper-ready PNGs to
vizualizations/figures/. Both are resolved next to this file, so it runs from
any working directory.

    python vizualizations/results_plot.py

Figures produced:
    accuracy_and_lock.png     accuracy and lock rate per batch, trials as dots
    verdict_composition.png   share of each verdict category per batch
    confusion_matrix.png      expected object vs landed object, row-normalised
    accuracy_by_command.png   accuracy per commanded direction
    trajectories_batch{n}.png fovea path per trial, one panel per trial

Trajectories need ref_x/ref_y/fovea_x/fovea_y in the CSV. Trials logged before
those columns were enabled are skipped, with a note.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
dpi = 300

# optional: batch -> ground-truth mask, drawn behind the trajectories
mask_paths = {
    # 6: "stimuli/ground_truth_masks/6_objects_color_bg_jitter_346x260.mask.npy",
}

# batches 1-6 as named in the experiment script. Add 7-10 here once the object
# count runs are numbered; a batch with no entry is labelled by its number alone.
batch_labels = {
    1: "circles / mono / no bg",
    2: "circles / mono / bg",
    3: "shapes / mono / no bg",
    4: "shapes / mono / bg",
    5: "shapes / colour / no bg",
    6: "shapes / colour / bg",
}

categories = ["ok", "wrong", "stalled", "no object"]
category_colors = {
    "ok": "#029e73",
    "wrong": "#d55e00",
    "stalled": "#949494",
    "no object": "#0173b2",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": dpi,
    "savefig.bbox": "tight",
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


def category_of(verdict):
    if verdict == "OK":
        return "ok"
    if verdict.startswith("WRONG"):
        return "wrong"
    if "did not move" in verdict:
        return "stalled"
    return "no object"


required = ["step", "command", "start_id", "expected_id", "landed_id",
            "verdict", "trial_id", "trial", "batch", "accuracy"]


def load(path):
    files = [f for f in sorted(glob.glob(os.path.join(path, "*.csv")))
             if not f.endswith("_unscored.csv")]
    if not files:
        raise SystemExit(f"no CSVs in {path}/")

    frames, bad = [], []
    for f in files:
        part = pd.read_csv(f, dtype={"trial_id": str})
        missing = [c for c in required if c not in part.columns]
        if missing:
            bad.append(f"{f}: missing columns {missing}")
            continue
        part["source_file"] = os.path.basename(f)
        frames.append(part)
    if bad:
        raise SystemExit("\n".join(["unreadable files:"] + bad))

    df = pd.concat(frames, ignore_index=True)

    # a shifted row lands a number or a blank where the verdict should be
    broken = df[~df["verdict"].apply(lambda v: isinstance(v, str))]
    if len(broken):
        print("rows with no verdict — the columns are shifted in these files:")
        cols = ["source_file", "step", "command", "start_id", "expected_id",
                "landed_id", "verdict"]
        print(broken[cols].to_string(index=False))
        raise SystemExit("fix or re-run those trials, then plot")

    df["category"] = df["verdict"].map(category_of)
    # mirrors the scoring in the experiment script: a refusal counts only when
    # the fovea genuinely stayed off every object
    df["correct"] = ((df["category"] == "ok")
                     | ((df["category"] == "no object") & (df["landed_id"] == 0)))
    print(f"loaded {len(files)} trials, {len(df)} steps")
    return df


def object_names(df):
    names = {0: "bg"}
    for id_col, label_col in (("start_id", "start"), ("expected_id", "expected")):
        for oid, label in zip(df[id_col], df[label_col]):
            if oid and isinstance(label, str) and "(" in label:
                names[int(oid)] = label.split("(", 1)[1].rstrip(")")
    return names


def per_trial_metrics(df):
    out = df.groupby(["trial_id", "batch", "trial"], as_index=False).agg(
        accuracy=("correct", lambda s: 100 * s.mean()),
        lock=("landed_id", lambda s: 100 * (s != 0).mean()),
        steps=("step", "size"),
        stored=("accuracy", "first"),
    )
    drift = out[(out["accuracy"].round(1) - out["stored"].round(1)).abs() > 0.05]
    if len(drift):
        print("\nrecomputed accuracy disagrees with the stored column:")
        print(drift[["trial_id", "accuracy", "stored"]].to_string(index=False))
        print("the scoring rule in the experiment script has changed; "
              "update category_of/correct in this file to match.\n")
    return out.drop(columns="stored")


def tick_labels(batches):
    return [f"{b}  {batch_labels[b]}" if b in batch_labels else str(b)
            for b in batches]


def fig_accuracy_and_lock(trials, path):
    batches = sorted(trials["batch"].unique())
    x = np.arange(len(batches))
    width = max(7.2, 0.9 * len(batches) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(width, 3.4), sharex=True)
    for ax, col, title in zip(axes, ("accuracy", "lock"),
                              ("command accuracy", "lock rate")):
        means = [trials.loc[trials["batch"] == b, col].mean() for b in batches]
        ax.bar(x, means, width=0.62, color="#dce4ec", edgecolor="#39566f",
               linewidth=0.8, zorder=1)
        for i, b in enumerate(batches):
            y = trials.loc[trials["batch"] == b, col].to_numpy()
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(y)), y, s=13,
                       color="#39566f", zorder=3, clip_on=False)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels(batches), fontsize=7, rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of commands")
    axes[1].set_ylabel("% of steps on an object")
    fig.savefig(path)
    plt.close(fig)


def fig_verdict_composition(df, path):
    batches = sorted(df["batch"].unique())
    share = np.zeros((len(batches), len(categories)))
    for i, b in enumerate(batches):
        sub = df[df["batch"] == b]
        for j, c in enumerate(categories):
            share[i, j] = 100 * (sub["category"] == c).mean()
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(batches) + 1.4))
    y = np.arange(len(batches))
    left = np.zeros(len(batches))
    for j, c in enumerate(categories):
        ax.barh(y, share[:, j], left=left, height=0.62, color=category_colors[c],
                edgecolor="white", linewidth=0.6, label=c)
        left += share[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels(batches), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of steps")
    ax.legend(ncol=len(categories), loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(path)
    plt.close(fig)


def fig_confusion_matrix(df, names, path):
    ids = sorted(set(df["expected_id"]) | set(df["landed_id"]))
    counts = np.zeros((len(ids), len(ids)))
    index = {v: i for i, v in enumerate(ids)}
    for e, l in zip(df["expected_id"], df["landed_id"]):
        counts[index[e], index[l]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    size = 0.52 * len(ids) + 1.8
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    labels = [names.get(i, str(i)) for i in ids]
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
    present = [d for d in order if d in set(df["command"])]
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


def fig_trajectories(df, batch, names, path):
    sub = df[df["batch"] == batch].dropna(subset=["ref_x", "fovea_x"])
    if sub.empty:
        return False
    trials = sorted(sub["trial"].unique())
    mask = None
    if batch in mask_paths and os.path.exists(mask_paths[batch]):
        mask = np.load(mask_paths[batch])

    cols = min(3, len(trials))
    rows = int(np.ceil(len(trials) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.7 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for ax, t in zip(axes.flat, trials):
        ax.set_visible(True)
        steps = sub[sub["trial"] == t].sort_values("step")
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
            travel = np.hypot(r["fovea_x"] - r["ref_x"], r["fovea_y"] - r["ref_y"])
            if travel < 1.0:
                stalls += 1
                ax.plot(r["ref_x"], r["ref_y"], marker="x", markersize=6,
                        color=colour, markeredgewidth=1.4)
                continue
            ax.annotate("", xy=(r["fovea_x"], r["fovea_y"]),
                        xytext=(r["ref_x"], r["ref_y"]),
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                        linewidth=1.3, shrinkA=0, shrinkB=0))
            ax.text((r["ref_x"] + r["fovea_x"]) / 2,
                    (r["ref_y"] + r["fovea_y"]) / 2,
                    f"{int(r['step'])} {r['command']}", fontsize=6,
                    color=colour, ha="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                              alpha=0.75))

        first = steps.iloc[0]
        ax.plot(first["ref_x"], first["ref_y"], marker="o", markersize=5,
                markerfacecolor="none", markeredgecolor="#333333",
                markeredgewidth=1.0)
        acc = 100 * steps["correct"].mean()
        title = f"trial {int(t)} — {acc:.0f}% correct"
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
    fig.suptitle(f"batch {batch}  {batch_labels.get(batch, '')}".rstrip(),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True

def summary(trials, df):
    rows = []
    for b in sorted(trials["batch"].unique()):
        t = trials[trials["batch"] == b]
        s = df[df["batch"] == b]
        rows.append({
            "batch": b,
            "stimulus": batch_labels.get(b, ""),
            "trials": len(t),
            "steps": len(s),
            "accuracy": round(t["accuracy"].mean(), 1),
            "sd": round(t["accuracy"].std(ddof=0), 1),
            "lock": round(t["lock"].mean(), 1),
            "stalled": round(100 * (s["category"] == "stalled").mean(), 1),
            "wrong": round(100 * (s["category"] == "wrong").mean(), 1),
        })
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def main():
    os.makedirs(figures_dir, exist_ok=True)
    df = load(results_dir)
    names = object_names(df)
    trials = per_trial_metrics(df)

    fig_accuracy_and_lock(trials, os.path.join(figures_dir, "accuracy_and_lock.png"))
    fig_verdict_composition(df, os.path.join(figures_dir, "verdict_composition.png"))
    fig_confusion_matrix(df, names, os.path.join(figures_dir, "confusion_matrix.png"))
    fig_accuracy_by_command(df, os.path.join(figures_dir, "accuracy_by_command.png"))

    if "ref_x" in df.columns:
        drawn, skipped = [], []
        for b in sorted(df["batch"].unique()):
            name = f"trajectories_batch{b}.png"
            if fig_trajectories(df, b, names, os.path.join(figures_dir, name)):
                drawn.append(int(b))
            else:
                skipped.append(int(b))
        if skipped:
            print(f"\nno coordinates logged for batches {skipped} — "
                  "trajectories skipped")
    else:
        print("\nno ref_x/fovea_x columns — trajectories skipped")

    summary(trials, df)
    print(f"\nfigures written to {figures_dir}/")


if __name__ == "__main__":
    main()