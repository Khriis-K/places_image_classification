"""Task 2 data exploration: class balance, image sizes, channel statistics, sample grids, loader speed.

Writes PNG figures + summary.json to --out and the same figures to TensorBoard.
Usage: python src/explore.py --root /scratch1/$USER/places365 --out /scratch1/$USER/runs/explore
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from data import IMAGENET_MEAN, IMAGENET_STD, build_datasets, build_loaders, build_transforms, denormalize, read_categories


def count_stats(samples, n_classes):
    counts = Counter(label for _, label in samples)
    per_class = [counts.get(i, 0) for i in range(n_classes)]
    return per_class, {"images": len(samples), "min": min(per_class), "max": max(per_class),
                       "mean": round(sum(per_class) / n_classes, 1)}


def plot_class_counts(per_class, names):
    order = sorted(range(len(per_class)), key=lambda i: per_class[i])
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(order)), [per_class[i] for i in order], width=1.0, color="#4C72B0")
    ax.set(title="Training images per class (sorted)", xlabel="class rank", ylabel="images", xlim=(-1, len(order)))
    for i in order[:3]:  # label the smallest classes
        ax.annotate(names[i], (order.index(i), per_class[i]), textcoords="offset points", xytext=(4, -12), fontsize=8)
    fig.tight_layout()
    return fig


def image_properties(ds, n, rng):
    sizes, modes = Counter(), Counter()
    for rel, _ in rng.sample(ds.samples, min(n, len(ds.samples))):
        with Image.open(ds.image_dir / rel) as im:
            sizes[f"{im.width}x{im.height}"] += 1
            modes[im.mode] += 1
    return {"sizes": dict(sizes), "modes": dict(modes)}


def channel_stats(ds, n, rng):
    """Per-channel mean/std over n random un-augmented training images (pixels in [0, 1])."""
    to_tensor = transforms.ToTensor()
    total, total_sq, pixels = torch.zeros(3), torch.zeros(3), 0
    for rel, _ in rng.sample(ds.samples, min(n, len(ds.samples))):
        x = to_tensor(Image.open(ds.image_dir / rel).convert("RGB")).flatten(1)
        total += x.sum(1)
        total_sq += (x ** 2).sum(1)
        pixels += x.shape[1]
    mean = total / pixels
    std = (total_sq / pixels - mean ** 2).sqrt()
    return [round(v, 4) for v in mean.tolist()], [round(v, 4) for v in std.tolist()]


def plot_samples(ds, names, rng, n_classes=32):
    """One random image from each of n_classes random classes, titled with its label."""
    by_class = {}
    for rel, label in ds.samples:
        by_class.setdefault(label, []).append(rel)
    chosen = rng.sample(sorted(by_class), min(n_classes, len(by_class)))
    cols = 8
    rows = -(-len(chosen) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, label in zip(axes.flat, chosen):
        ax.imshow(Image.open(ds.image_dir / rng.choice(by_class[label])).convert("RGB"))
        ax.set_title(names[label], fontsize=8)
    fig.suptitle("Places365 samples (one per class)")
    fig.tight_layout()
    return fig


def plot_augmentations(ds, rng, n=7):
    rel, _ = rng.choice(ds.samples)
    image = Image.open(ds.image_dir / rel).convert("RGB")
    augment = build_transforms(train=True)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 2, 2.4))
    axes[0].imshow(image)
    axes[0].set_title("original", fontsize=8)
    for ax in axes[1:]:
        ax.imshow(denormalize(augment(image)).permute(1, 2, 0))
        ax.set_title("augmented", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def loader_throughput(loader, n_batches):
    """Images/second the DataLoader alone can deliver (no model), after a warm-up batch."""
    it = iter(loader)
    next(it)
    start, images = time.perf_counter(), 0
    for _ in range(n_batches):
        images += next(it)[0].shape[0]
    return round(images / (time.perf_counter() - start), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="directory produced by scripts/extract_data.sh")
    p.add_argument("--out", required=True, help="where to write figures, summary.json and TensorBoard logs")
    p.add_argument("--stats-images", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--loader-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    writer = SummaryWriter(out / "tensorboard")

    names = read_categories(Path(args.root) / "filelists")
    datasets = build_datasets(args.root, seed=args.seed)
    summary = {"classes": len(names)}
    train_counts = None
    for split, ds in datasets.items():
        per_class, summary[split] = count_stats(ds.samples, len(names))
        if split == "train":
            train_counts = per_class
        print(f"{split:5s}: {summary[split]}")

    summary["image_properties"] = image_properties(datasets["train"], 1000, rng)
    print("image properties (1000 sampled):", summary["image_properties"])

    mean, std = channel_stats(datasets["train"], args.stats_images, rng)
    summary["channel_mean"], summary["channel_std"] = mean, std
    print(f"channel mean {mean} std {std}  (ImageNet: {IMAGENET_MEAN} {IMAGENET_STD})")

    figures = {
        "class_counts": plot_class_counts(train_counts, names),
        "samples": plot_samples(datasets["train"], names, rng),
        "augmentations": plot_augmentations(datasets["train"], rng),
    }
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.png", dpi=120)
        writer.add_figure(f"dataset/{name}", fig)

    loaders = build_loaders(datasets, args.batch_size, args.num_workers)
    batch, _ = next(iter(loaders["train"]))
    writer.add_image("dataset/train_batch", make_grid(denormalize(batch[:64]), nrow=8))

    summary["loader_images_per_sec"] = loader_throughput(loaders["train"], args.loader_batches)
    summary["loader_config"] = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    print(f"DataLoader throughput: {summary['loader_images_per_sec']} images/s "
          f"(batch {args.batch_size}, {args.num_workers} workers)")

    writer.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote figures, summary.json and TensorBoard logs to {out}")


if __name__ == "__main__":
    main()
