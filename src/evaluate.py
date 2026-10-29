"""Task 6: run a trained checkpoint on the test split and save every image's predictions.

The test split is the official Places365 validation set (36,500 images, 100 per class; src/data.py).
Writes <out>/<name>.npz so the confusion matrix and other figures can be redrawn on a laptop without a
GPU (src/plot_evaluation.py), and prints top-1/top-5, the worst classes and the most-confused pairs.

Single-crop (default): resize to 256, one 224x224 center crop, the same as validation during training.
10-crop (--ten-crop): the four corners and the center, each also mirrored; the softmax probabilities of
the 10 crops are averaged. This is how the Places paper reports its numbers.

Usage: python src/evaluate.py --root $TMPDIR/places365 --checkpoint /scratch1/$USER/runs/r50_lr0.04/best.pt \
           --out logs/task6 --name r50_lr0.04 [--ten-crop]
"""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import IMAGENET_MEAN, IMAGENET_STD, available_cpus, build_datasets, read_categories
from metrics import confusion_matrix, per_class_accuracy, top_confusions, topk_accuracy
from train import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="extracted dataset (scripts/extract_data.sh)")
    p.add_argument("--checkpoint", required=True, help="best.pt written by src/train.py")
    p.add_argument("--out", required=True, help="directory for <name>.npz")
    p.add_argument("--name", required=True, help="output file name, e.g. r50_lr0.04 or r50_lr0.04_10crop")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--ten-crop", action="store_true", help="average the predictions of 10 crops per image")
    p.add_argument("--batch-size", type=int, default=None, help="images per batch (default 256, or 32 with --ten-crop)")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: all CPUs)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="fp16 mixed precision (CUDA only)")
    p.add_argument("--limit-batches", type=int, default=None, help="stop early, for quick tests")
    return p.parse_args()


class StackCrops:
    """Turn TenCrop's tuple of PIL crops into one normalized (10, 3, H, W) tensor.

    A class rather than a lambda so DataLoader workers can pickle it.
    """
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __call__(self, crops):
        return torch.stack([self.to_tensor(c) for c in crops])


def ten_crop_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.TenCrop(image_size),  # 4 corners + center, and their horizontal mirrors
        StackCrops(),
    ])


@torch.no_grad()
def predict(model, loader, device, amp, memory_format, limit=None):
    """Per image: true label, the 5 highest-probability classes and their probabilities,
    the probability given to the true class, and the cross-entropy loss.

    Batches of shape (B, crops, 3, H, W) are 10-crop: the crops' softmax probabilities are averaged.
    """
    model.eval()
    out = {k: [] for k in ("labels", "top5", "top5_prob", "true_prob", "loss")}
    for i, (images, labels) in enumerate(loader):
        if limit is not None and i >= limit:
            break
        crops = images.shape[1] if images.dim() == 5 else 1
        images = images.flatten(0, 1) if crops > 1 else images
        images = images.to(device, non_blocking=True, memory_format=memory_format)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        # log of the crops' mean probability, computed in log space so tiny probabilities don't round to 0
        log_probs = logits.float().log_softmax(dim=1).view(len(labels), crops, -1)
        log_probs = log_probs.logsumexp(dim=1) - math.log(crops)
        true_log_prob = log_probs.gather(1, labels[:, None]).squeeze(1)
        top5_log_prob, top5 = log_probs.topk(5, dim=1)
        out["labels"].append(labels)
        out["top5"].append(top5)
        out["top5_prob"].append(top5_log_prob.exp())
        out["true_prob"].append(true_log_prob.exp())
        out["loss"].append(-true_log_prob)  # cross-entropy of the (averaged) prediction
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def print_summary(pred, class_names, worst=10, pairs=10):
    labels, top5 = pred["labels"], pred["top5"]
    cm = confusion_matrix(labels, top5[:, 0], len(class_names))
    acc = per_class_accuracy(cm)
    print(f"images {len(labels)}  top-1 {100 * topk_accuracy(labels, top5, 1):.2f}%  "
          f"top-5 {100 * topk_accuracy(labels, top5, 5):.2f}%  loss {pred['loss'].mean():.4f}")
    print(f"per-class top-1: mean {100 * np.nanmean(acc):.2f}%  median {100 * np.nanmedian(acc):.2f}%  "
          f"min {100 * np.nanmin(acc):.0f}%  max {100 * np.nanmax(acc):.0f}%")
    print(f"{worst} worst classes:")
    for c in np.argsort(acc)[:worst]:
        print(f"  {100 * acc[c]:5.1f}%  {class_names[c]}")
    print(f"{pairs} most-confused pairs (true -> predicted, share of the true class):")
    for t, p, count, share in top_confusions(cm, pairs):
        print(f"  {100 * share:5.1f}%  ({count})  {class_names[t]} -> {class_names[p]}")


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    amp = args.amp and device.type == "cuda"
    memory_format = torch.channels_last
    workers = available_cpus() if args.workers is None else args.workers
    batch_size = args.batch_size or (32 if args.ten_crop else 256)

    dataset = build_datasets(args.root)[args.split]
    if args.ten_crop:
        dataset.transform = ten_crop_transform()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    class_names = read_categories(Path(args.root) / "filelists")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["model"])
    model.to(device, memory_format=memory_format)
    print(f"{args.checkpoint}: epoch {ckpt['epoch'] + 1}, val top-1 {ckpt['val_top1']:.4f} during training")
    print(f"{args.split} split, {len(dataset)} images, {'10-crop' if args.ten_crop else 'single-crop'}, "
          f"batch {batch_size}, {workers} workers, {device.type}, AMP {'on' if amp else 'off'}")

    start = time.perf_counter()
    pred = predict(model, loader, device, amp, memory_format, args.limit_batches)
    seconds = time.perf_counter() - start
    print(f"{len(pred['labels'])} images in {seconds:.0f} s ({len(pred['labels']) / seconds:.0f} img/s)")
    print_summary(pred, class_names)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.npz"
    np.savez_compressed(
        path,
        labels=pred["labels"].astype(np.int16), top5=pred["top5"].astype(np.int16),
        top5_prob=pred["top5_prob"].astype(np.float32), true_prob=pred["true_prob"].astype(np.float32),
        loss=pred["loss"].astype(np.float32), class_names=np.array(class_names),
        image_paths=np.array([rel for rel, _ in dataset.samples[:len(pred["labels"])]]),
        checkpoint=args.checkpoint, epoch=ckpt["epoch"], ten_crop=args.ten_crop, split=args.split,
        seconds=seconds,
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
