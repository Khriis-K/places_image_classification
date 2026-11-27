"""Task 8: strong-scaling figures and table from the 1/2/4/8-GPU runs.

Multi-GPU runs are read from <run>/timing.jsonl (written by train.py since Task 7). The 1-GPU baseline
predates that file, so it is read from its TensorBoard log: training time from epoch/minutes, and
validation time estimated as the median gap between consecutive epochs minus that epoch's training
time (the median skips the two epochs that were interrupted and restarted).

Writes to --out:
  scaling.png         speedup and parallel efficiency vs. number of GPUs
  step_time.png       time per training step, split into compute and the extra time added by scaling
  accuracy_lr.png     learning-rate schedule and validation top-1 per epoch for each configuration
  summary.md          the table used in docs/task8_scaling.md

Usage: python src/plot_scaling.py --out docs/figures/task8 --baseline runs/carc/r50_lr0.04 \
           runs/carc/r50_ddp_1n2g runs/carc/r50_ddp_2n4g runs/carc/r50_ddp_4n8g
"""
import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_training import load, series

TRAIN_IMAGES = 1_713_290  # our training split (Task 2)
BATCH_PER_GPU = 256
# Reference categorical palette, slots 1-4 in fixed order (1, 2, 4, 8 GPUs)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e1"


def baseline_run(run_dir):
    """1-GPU run from TensorBoard only: one dict per epoch, same keys as timing.jsonl."""
    ea = load(run_dir)
    minutes = {e.step: e for e in ea.Scalars("epoch/minutes")}
    epochs = sorted(minutes)
    gaps = [(minutes[b].wall_time - minutes[a].wall_time) - minutes[b].value * 60 for a, b in zip(epochs, epochs[1:])]
    val_s = median(gaps)
    rate = dict(zip(*series(ea, "epoch/images_per_s")))
    top1, top5 = dict(zip(*series(ea, "epoch/val_top1"))), dict(zip(*series(ea, "epoch/val_top5")))
    return [{"epoch": n, "gpus": 1, "nodes": 1, "global_batch": BATCH_PER_GPU, "peak_lr": 0.04,
             "train_s": minutes[n].value * 60, "val_s": val_s, "images_per_s": rate[n],
             "val_top1": top1[n], "val_top5": top5[n]} for n in epochs]


def timing_run(run_dir):
    lines = (Path(run_dir) / "timing.jsonl").read_text().splitlines()
    by_epoch = {r["epoch"]: r for r in map(json.loads, lines)}  # a resumed run may repeat an epoch: keep the last
    return [by_epoch[n] for n in sorted(by_epoch)]


def summarize(name, run_dir, epochs):
    first = epochs[0]
    gpus = first["gpus"]
    train_min = sum(e["train_s"] for e in epochs) / 60
    val_min = sum(e["val_s"] for e in epochs) / 60
    best = max(epochs, key=lambda e: e["val_top1"])
    return {
        "name": name, "dir": run_dir, "gpus": gpus, "nodes": first["nodes"], "global_batch": first["global_batch"],
        "peak_lr": first["peak_lr"], "steps_per_epoch": TRAIN_IMAGES // gpus // BATCH_PER_GPU,
        "train_min": train_min, "val_min": val_min, "total_min": train_min + val_min,
        "steady_ips": mean(e["images_per_s"] for e in epochs[1:]),  # epoch 1 includes warm-up costs
        "top1": best["val_top1"], "top5": best["val_top5"], "epochs": epochs,
    }


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def gpu_axis(ax, gpus):
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpus, [str(g) for g in gpus])
    ax.minorticks_off()
    ax.set_xlabel("GPUs (2 per node)")


def plot_scaling(runs, out):
    gpus = [r["gpus"] for r in runs]
    base = runs[0]["total_min"]
    speedup = [base / r["total_min"] for r in runs]
    efficiency = [100 * s / g for s, g in zip(speedup, gpus)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(gpus, gpus, "--", color=MUTED, linewidth=1.5)
    a.text(gpus[-1], gpus[-1], "  ideal (linear)", color=MUTED, va="center", fontsize=9)
    a.plot(gpus, speedup, "o-", color=COLORS[0], linewidth=2, markersize=8, markeredgecolor="white", markeredgewidth=2)
    for g, s in zip(gpus[1:], speedup[1:]):
        a.annotate(f"{s:.2f}×", (g, s), textcoords="offset points", xytext=(8, -12), ha="left", fontsize=9, color=INK)
    a.set_ylabel("speedup over 1 GPU")
    a.set_yscale("log", base=2)
    a.set_yticks(gpus, [str(g) for g in gpus])
    a.minorticks_off()
    a.set_xlim(0.8, 13)
    a.set_title("Speedup (10 epochs, training + validation)", fontsize=10, loc="left", color=INK)

    b.axhline(100, linestyle="--", color=MUTED, linewidth=1.5)
    b.plot(gpus, efficiency, "o-", color=COLORS[0], linewidth=2, markersize=8, markeredgecolor="white", markeredgewidth=2)
    for g, e in zip(gpus[1:], efficiency[1:]):
        b.annotate(f"{e:.0f}%", (g, e), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=INK)
    b.set_ylim(80, 102)
    b.set_ylabel("parallel efficiency (%)")
    b.set_title("Parallel efficiency = speedup ÷ GPUs", fontsize=10, loc="left", color=INK)
    for ax in (a, b):
        style(ax)
        gpu_axis(ax, gpus)
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=150)
    plt.close(fig)


def plot_step_time(runs, out):
    """Every GPU does the same work per step (256 images), so ideal step time stays at the 1-GPU value."""
    compute = 1000 * BATCH_PER_GPU / runs[0]["steady_ips"]
    step = [1000 * r["global_batch"] / r["steady_ips"] for r in runs]
    extra = [s - compute for s in step]
    labels = [f"{r['gpus']} GPU{'s' if r['gpus'] > 1 else ''}\n{r['nodes']} node{'s' if r['nodes'] > 1 else ''}" for r in runs]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = range(len(runs))
    ax.barh(y, [compute] * len(runs), color=COLORS[0], height=0.6, label="1-GPU step time (forward, backward, update)")
    ax.barh(y, extra, left=[compute + 1.5] * len(runs), color=COLORS[1], height=0.6,
            label="extra time per step with more GPUs")
    for i, (s, e) in enumerate(zip(step, extra)):
        ax.text(compute + e + 4, i, f"{s:.0f} ms" + (f"  (+{e:.1f})" if e > 0.05 else ""), va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, compute * 1.35)
    ax.set_xlabel("milliseconds per training step (steady state, epochs 2–10)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False, fontsize=9)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "step_time.png", dpi=150)
    plt.close(fig)


def plot_accuracy_lr(runs, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for r, c in zip(runs, COLORS):
        label = f"{r['gpus']} GPU{'s' if r['gpus'] > 1 else ''} (batch {r['global_batch']}, lr {r['peak_lr']:g})"
        steps, lrs = series(load(r["dir"]), "step/lr")
        a.plot([s / r["steps_per_epoch"] for s in steps], lrs, color=c, linewidth=2, label=label)
        ep = r["epochs"]
        b.plot([e["epoch"] for e in ep], [100 * e["val_top1"] for e in ep], "o-", color=c, linewidth=2,
               markersize=6, markeredgecolor="white", markeredgewidth=1.5, label=label)
    a.set_xlabel("epoch")
    a.set_ylabel("learning rate")
    a.set_xlim(0, 10)
    a.set_title("Learning rate: warmup, then cosine decay", fontsize=10, loc="left", color=INK)
    b.set_xlabel("epoch")
    b.set_ylabel("validation top-1 (%)")
    b.set_xticks(range(1, 11))
    b.set_title("Validation accuracy per epoch", fontsize=10, loc="left", color=INK)
    for ax in (a, b):
        style(ax)
    handles, labels = b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "accuracy_lr.png", dpi=150)
    plt.close(fig)


def write_summary(runs, out):
    base = runs[0]
    rows = ["| GPUs | Nodes | Global batch | Peak lr | Steps/epoch | Throughput (img/s, epochs 2–10) | Training (min) "
            "| Validation (min) | Total (min) | Speedup | Efficiency | Val top-1 | Val top-5 |",
            "|" + "---|" * 13]
    for r in runs:
        s = base["total_min"] / r["total_min"]
        rows.append(f"| {r['gpus']} | {r['nodes']} | {r['global_batch']:,} | {r['peak_lr']:g} | {r['steps_per_epoch']:,} "
                    f"| {r['steady_ips']:,.0f} | {r['train_min']:.1f} | {r['val_min']:.1f} | {r['total_min']:.1f} "
                    f"| {s:.2f}× | {100 * s / r['gpus']:.1f}% | {100 * r['top1']:.2f}% | {100 * r['top5']:.2f}% |")
    note = (f"\n1-GPU validation time is estimated ({base['epochs'][0]['val_s']:.0f} s per epoch; see "
            "plot_scaling.py). Speedup and efficiency use total minutes (training + validation).\n")
    (out / "summary.md").write_bytes(("\n".join(rows) + "\n" + note).encode("utf-8"))
    print("\n".join(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", required=True, help="1-GPU run directory (TensorBoard only)")
    p.add_argument("runs", nargs="+", help="multi-GPU run directories with timing.jsonl")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = [summarize("1 GPU", args.baseline, baseline_run(args.baseline))]
    runs += [summarize(Path(d).name, d, timing_run(d)) for d in args.runs]
    runs.sort(key=lambda r: r["gpus"])
    plot_scaling(runs, out)
    plot_step_time(runs, out)
    plot_accuracy_lr(runs, out)
    write_summary(runs, out)


if __name__ == "__main__":
    main()
