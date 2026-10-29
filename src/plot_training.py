"""Task 5: turn TensorBoard logs into report figures, and add the train-vs-val chart layout to old runs.

Reads <run>/tensorboard of each run and writes to --out:
  <name>_loss.png       train and validation loss vs epoch
  <name>_accuracy.png   top-1 (train, val) and val top-5 vs epoch
  <name>_steps.png      per-step training loss and learning rate (first run only)
  <label>_augmented_batch.png, <label>_dataset_*.png   images logged by train.py / explore.py

Usage: python src/plot_training.py --out docs/figures/task5 --name final \
           "lr 0.04=runs/r50_lr0.04" "lr 0.01=runs/r50_1gpu"
       python src/plot_training.py --add-layout runs/r50_1gpu ...   (runs trained before the layout existed)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from train import EPOCH_LAYOUT


def load(run_dir):
    tb = Path(run_dir) / "tensorboard"
    ea = EventAccumulator(str(tb if tb.is_dir() else run_dir), size_guidance={"scalars": 0, "images": 0})
    ea.Reload()
    return ea


def series(ea, tag):
    """(steps, values) for a scalar tag, keeping the last value logged at each step (a resumed run may repeat one)."""
    if tag not in ea.Tags()["scalars"]:
        return [], []
    points = {e.step: e.value for e in ea.Scalars(tag)}
    steps = sorted(points)
    return steps, [points[s] for s in steps]


def smooth(values, weight=0.9):
    """Exponential moving average, like TensorBoard's smoothing slider."""
    out, avg = [], None
    for v in values:
        avg = v if avg is None else weight * avg + (1 - weight) * v
        out.append(avg)
    return out


def plot_epochs(runs, out, name):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for (label, ea), c in zip(runs, colors):
        ax.plot(*series(ea, "epoch/val_loss"), "o-", color=c, label=f"{label} val")
        ax.plot(*series(ea, "epoch/train_loss"), "s--", color=c, alpha=0.6, label=f"{label} train")
    ax.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss vs epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}_loss.png", dpi=150)
    plt.close(fig)

    fig, (top1, top5) = plt.subplots(1, 2, figsize=(10, 4))
    for (label, ea), c in zip(runs, colors):
        steps, val = series(ea, "epoch/val_top1")
        top1.plot(steps, [100 * v for v in val], "o-", color=c, label=f"{label} val")
        steps, train = series(ea, "epoch/train_top1")
        top1.plot(steps, [100 * v for v in train], "s--", color=c, alpha=0.6, label=f"{label} train")
        steps, val5 = series(ea, "epoch/val_top5")
        top5.plot(steps, [100 * v for v in val5], "o-", color=c, label=f"{label} val")
    top1.set(xlabel="epoch", ylabel="accuracy (%)", title="Top-1 accuracy vs epoch")
    top5.set(xlabel="epoch", ylabel="accuracy (%)", title="Validation top-5 accuracy vs epoch")
    for ax in (top1, top5):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}_accuracy.png", dpi=150)
    plt.close(fig)


def plot_steps(label, ea, out, name):
    steps, loss = series(ea, "step/train_loss")
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, loss, color="C0", alpha=0.25)
    ax.plot(steps, smooth(loss), color="C0", label="train loss (smoothed)")
    ax.set(xlabel="training step", ylabel="train loss", title=f"Per-step training loss and learning rate ({label})")
    ax.grid(alpha=0.3)
    lr_ax = ax.twinx()
    lr_ax.plot(*series(ea, "step/lr"), color="C1", label="learning rate")
    lr_ax.set_ylabel("learning rate")
    lines = ax.get_lines()[1:] + lr_ax.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}_steps.png", dpi=150)
    plt.close(fig)


def save_images(label, ea, out):
    for tag in ea.Tags()["images"]:
        events = ea.Images(tag)
        if events:
            filename = f"{label}_{tag.replace('/', '_')}.png".replace(" ", "_")
            (out / filename).write_bytes(events[-1].encoded_image_string)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="label=run_dir (or just run_dir with --add-layout)")
    p.add_argument("--out", default="docs/figures/task5")
    p.add_argument("--name", default="training", help="prefix for the plot filenames")
    p.add_argument("--add-layout", action="store_true", help="only write the Custom Scalars layout into each run")
    args = p.parse_args()

    if args.add_layout:
        for run in args.runs:
            with SummaryWriter(Path(run) / "tensorboard") as writer:
                writer.add_custom_scalars(EPOCH_LAYOUT)
            print(f"layout added to {run}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for spec in args.runs:
        label, _, run_dir = spec.rpartition("=")
        runs.append((label or Path(run_dir).name, load(run_dir)))
    plot_epochs(runs, out, args.name)
    plot_steps(*runs[0], out, args.name)
    for label, ea in runs:
        save_images(label, ea, out)
    print(f"figures written to {out}: {sorted(f.name for f in out.glob('*.png'))}")


if __name__ == "__main__":
    main()
