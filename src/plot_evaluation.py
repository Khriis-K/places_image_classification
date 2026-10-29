"""Task 6: report figures and a results summary from the predictions saved by src/evaluate.py.

Runs on a laptop (no GPU). The first run is the one plotted in detail; every run goes into the
summary table. Writes to --out:
  confusion_full.png       all 365x365 classes, sorted so look-alike classes sit together
  confusion_top.png        readable sub-matrix of the classes in the most-confused pairs
  top_confusions.png       the most-confused (true -> predicted) pairs
  per_class_accuracy.png   distribution of the 365 per-class accuracies
  best_worst_classes.png   the classes the model gets most and least right
  compare_runs.png         per-class accuracy of the first two runs against each other (2+ runs)
  summary.md               accuracy table for every run, worst/best classes, most-confused pairs

Usage: python src/plot_evaluation.py --out docs/figures/task6 \
           "lr 0.04=logs/task6/r50_lr0.04.npz" "lr 0.04, 10-crop=logs/task6/r50_lr0.04_10crop.npz" \
           "lr 0.01=logs/task6/r50_1gpu.npz" [--tensorboard runs/carc/eval]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

from metrics import cluster_order, confusion_matrix, per_class_accuracy, top_confusions, topk_accuracy

# Single-hue sequential ramp (light -> dark blue) for "share of images"; white means none
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BAR = "#2a78d6"
MUTED = "#6b6b6b"


def load(path):
    d = np.load(path)
    labels, top5 = d["labels"].astype(np.int64), d["top5"].astype(np.int64)
    names = [str(n) for n in d["class_names"]]
    cm = confusion_matrix(labels, top5[:, 0], len(names))
    return {"labels": labels, "top5": top5, "loss": d["loss"], "names": names, "cm": cm,
            "acc": per_class_accuracy(cm), "ten_crop": bool(d["ten_crop"]), "seconds": float(d["seconds"])}


def row_share(cm):
    rows = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)


def plot_confusion_full(run, label):
    order = cluster_order(run["cm"])
    share = row_share(run["cm"])[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(9, 8))
    masked = np.ma.masked_equal(share, 0)  # never-confused cells stay white
    im = ax.imshow(masked, cmap=BLUES, norm=LogNorm(vmin=0.01, vmax=1), interpolation="nearest")
    ax.set_facecolor("white")
    ax.set(xlabel="predicted class", ylabel="true class", xticks=[], yticks=[],
           title=f"Confusion matrix, all {len(order)} classes ({label})\n"
                 "rows sum to 1; classes ordered by clustering so look-alikes sit together")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("share of the true class's images (log scale)")
    fig.tight_layout()
    return fig


def plot_confusion_top(run, label, pairs=15, max_classes=24):
    classes = []
    for t, p, _, _ in top_confusions(run["cm"], 4 * pairs):
        for c in (t, p):
            if c not in classes and len(classes) < max_classes:
                classes.append(c)
    sub = run["cm"][np.ix_(classes, classes)]
    classes = [classes[i] for i in cluster_order(sub)]  # keep each confused group together
    # share of each true class's images over ALL predictions, not just the ones shown
    share = row_share(run["cm"])[np.ix_(classes, classes)]
    names = [run["names"][c] for c in classes]

    # color scale tops out at the largest confusion, so the (larger) diagonal doesn't wash the confusions out
    off_diagonal = share * (1 - np.eye(len(classes)))
    vmax = max(off_diagonal.max(), 1e-9)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=vmax, interpolation="nearest")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if share[i, j] >= 0.05:
                ax.text(j, i, f"{100 * share[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if share[i, j] > 0.5 * vmax else "#1a1a1a")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set(xlabel="predicted class", ylabel="true class",
           title=f"Confusion among the most-confused classes ({label})\n"
                 "% of each true class's test images; cells under 5% not labeled; "
                 "diagonal (correct) cells shown at full color")
    fig.tight_layout()
    return fig


def plot_top_confusions(run, label, k=20):
    pairs = top_confusions(run["cm"], k)[::-1]  # largest at the top
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(pairs) + 1.2))
    shares = [100 * s for _, _, _, s in pairs]
    ax.barh(range(len(pairs)), shares, color=BAR, height=0.7)
    ax.set_yticks(range(len(pairs)), [f"{run['names'][t]} → {run['names'][p]}" for t, p, _, _ in pairs],
                  fontsize=8)
    for i, s in enumerate(shares):
        ax.text(s + 0.3, i, f"{s:.0f}%", va="center", fontsize=7, color=MUTED)
    ax.set(xlabel="% of the true class's test images", title=f"Most-confused pairs, true → predicted ({label})")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_per_class(run, label):
    acc = 100 * run["acc"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(acc, bins=np.arange(0, 101, 5), color=BAR, edgecolor="white", linewidth=2)
    mean = np.nanmean(acc)
    ax.axvline(mean, color="#1a1a1a", linewidth=1.5, linestyle="--")
    ax.text(mean + 1, ax.get_ylim()[1] * 0.92, f"mean {mean:.1f}%", fontsize=8)
    ax.set(xlabel="top-1 accuracy of the class (%)", ylabel="number of classes",
           title=f"Per-class top-1 accuracy, {len(acc)} classes ({label})")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_best_worst(run, label, k=15):
    acc = 100 * run["acc"]
    order = np.argsort(acc, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.3 * k + 1.4), sharex=True)
    for ax, classes, title in ((axes[0], order[:k][::-1], f"{k} worst classes"),
                               (axes[1], order[-k:], f"{k} best classes")):
        ax.barh(range(k), acc[classes], color=BAR, height=0.7)
        ax.set_yticks(range(k), [run["names"][c] for c in classes], fontsize=8)
        for i, c in enumerate(classes):
            ax.text(acc[c] + 1, i, f"{acc[c]:.0f}%", va="center", fontsize=7, color=MUTED)
        ax.set(xlim=(0, 105), xlabel="top-1 accuracy (%)", title=title)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle(f"Best and worst classes ({label})")
    fig.tight_layout()
    return fig


def plot_compare(a, label_a, b, label_b):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle="--")
    ax.scatter(100 * b["acc"], 100 * a["acc"], s=12, color=BAR, alpha=0.6, edgecolors="none")
    better = int(np.sum(a["acc"] > b["acc"]))
    worse = int(np.sum(a["acc"] < b["acc"]))
    ax.set(xlabel=f"{label_b}: per-class top-1 (%)", ylabel=f"{label_a}: per-class top-1 (%)",
           xlim=(0, 100), ylim=(0, 100), aspect="equal",
           title=f"Per-class accuracy, {label_a} vs {label_b}\n"
                 f"above the line: {label_a} better ({better} classes; {worse} worse)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def summary(runs, k=15):
    lines = ["| run | crops | top-1 | top-5 | loss | per-class top-1 mean / median / min / max | time |",
             "|---|---|---|---|---|---|---|"]
    for label, r in runs:
        acc = 100 * r["acc"]
        lines.append(f"| {label} | {10 if r['ten_crop'] else 1} | {100 * topk_accuracy(r['labels'], r['top5'], 1):.2f}% "
                     f"| {100 * topk_accuracy(r['labels'], r['top5'], 5):.2f}% | {r['loss'].mean():.3f} "
                     f"| {np.nanmean(acc):.1f} / {np.nanmedian(acc):.1f} / {np.nanmin(acc):.0f} / {np.nanmax(acc):.0f}% "
                     f"| {r['seconds']:.0f} s |")
    label, r = runs[0]
    acc, names = 100 * r["acc"], r["names"]
    order = np.argsort(acc, kind="stable")
    lines += ["", f"Details for {label}.", "", f"{k} worst classes:", ""]
    lines += [f"- {acc[c]:.0f}% {names[c]}" for c in order[:k]]
    lines += ["", f"{k} best classes:", ""]
    lines += [f"- {acc[c]:.0f}% {names[c]}" for c in order[::-1][:k]]
    lines += ["", "Most-confused pairs (true -> predicted, % of the true class's images):", ""]
    lines += [f"- {100 * s:.0f}% ({n}) {names[t]} -> {names[p]}" for t, p, n, s in top_confusions(r["cm"], 2 * k)]
    lines += ["", f"Classes below 30%: {int(np.sum(acc < 30))}; above 80%: {int(np.sum(acc > 80))}."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="label=path/to/predictions.npz; the first is plotted in detail")
    p.add_argument("--out", required=True)
    p.add_argument("--tensorboard", default=None, help="also log the figures to this TensorBoard directory")
    args = p.parse_args()

    runs = []
    for spec in args.runs:
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        runs.append((label, load(path)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    label, main_run = runs[0]
    figures = {
        "confusion_full": plot_confusion_full(main_run, label),
        "confusion_top": plot_confusion_top(main_run, label),
        "top_confusions": plot_top_confusions(main_run, label),
        "per_class_accuracy": plot_per_class(main_run, label),
        "best_worst_classes": plot_best_worst(main_run, label),
    }
    # compare against the first other run that uses the same number of crops
    others = [(l, r) for l, r in runs[1:] if r["ten_crop"] == main_run["ten_crop"]]
    if others:
        other_label, other_run = others[0]
        figures["compare_runs"] = plot_compare(main_run, label, other_run, other_label)

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tensorboard)
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.png", dpi=150)
        if writer:
            writer.add_figure(f"eval/{name}", fig)
        plt.close(fig)
    if writer:
        writer.close()

    (out / "summary.md").write_bytes(summary(runs).encode())  # bytes: LF line endings on Windows too
    print(summary(runs))
    print(f"Wrote {', '.join(figures)} and summary.md to {out}")


if __name__ == "__main__":
    main()
