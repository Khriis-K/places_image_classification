"""Task 4: fine-tune an ImageNet-pretrained ResNet-50 on Places365.

One script for 1 GPU and for many. Run with plain `python` it trains on one device; run with `torchrun`
it uses distributed data parallel (one process per GPU, see Task 7).

Checkpoints: <out>/last.pt after every epoch, <out>/best.pt whenever validation top-1 improves.
Rerunning the same command with the same --out resumes from last.pt, so a job that dies or hits its
time limit loses at most the epoch in progress. TensorBoard logs go to <out>/tensorboard.

Usage: python src/train.py --root $TMPDIR/places365 --out /scratch1/$USER/runs/r50_1gpu
"""
import argparse
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.utils import make_grid

from data import available_cpus, build_datasets, build_loaders, denormalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="extracted dataset (scripts/extract_data.sh)")
    p.add_argument("--out", required=True, help="run directory for checkpoints and TensorBoard logs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256, help="per GPU")
    p.add_argument("--lr", type=float, default=0.01, help="per 256 images; scaled linearly with the global batch")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers per GPU (default: its share of the CPUs)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="fp16 mixed precision (CUDA only)")
    p.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                   help="NHWC memory layout; ~1.5x faster AMP on V100 Tensor Cores (src/train_bench.py)")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="start from ImageNet weights")
    p.add_argument("--limit-train-batches", type=int, default=None, help="cap batches per epoch, for quick tests")
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50, help="steps between progress logs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def setup_distributed():
    """(rank, world_size, local_rank). torchrun sets WORLD_SIZE; plain python means a single process."""
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])


def build_model(pretrained, num_classes=365):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new 365-class head, randomly initialized
    return model


def warmup_cosine(warmup_steps, total_steps):
    """Learning-rate multiplier: linear ramp from ~0 to 1, then cosine decay to 0 at the last step."""
    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return multiplier


def save_checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save leaves the previous checkpoint intact


def all_reduce_sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


@torch.no_grad()
def evaluate(model, loader, device, amp, memory_format, limit=None):
    """Mean cross-entropy (no label smoothing), top-1 and top-5 accuracy over this split."""
    model.eval()
    totals = torch.zeros(4, device=device)  # loss sum, top-1 hits, top-5 hits, images
    for i, (images, labels) in enumerate(loader):
        if limit is not None and i >= limit:
            break
        images = images.to(device, non_blocking=True, memory_format=memory_format)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        top5 = logits.topk(5, dim=1).indices
        totals += torch.stack([
            F.cross_entropy(logits.float(), labels, reduction="sum"),
            (top5[:, 0] == labels).sum(),
            (top5 == labels[:, None]).any(dim=1).sum(),
            torch.tensor(len(labels), device=device),
        ]).float()
    loss, top1, top5, n = all_reduce_sum(totals).tolist()
    return loss / n, top1 / n, top5 / n


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    log = lambda *a: print(*a, flush=True) if is_main else None

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True  # pick the fastest conv kernels for our fixed input size
    else:
        device = torch.device("cpu")
    amp = args.amp and device.type == "cuda"
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    torch.manual_seed(args.seed)

    gpus_on_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    cpus = max(1, available_cpus() // gpus_on_node)  # this process's share of the node's CPUs
    torch.set_num_threads(cpus)
    workers = cpus if args.workers is None else args.workers

    datasets = build_datasets(args.root)
    loaders = build_loaders({s: datasets[s] for s in ("train", "val")}, args.batch_size, workers,
                            pin_memory=device.type == "cuda", distributed=world_size > 1)
    steps_per_epoch = len(loaders["train"])
    if args.limit_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)
    total_steps = args.epochs * steps_per_epoch
    global_batch = args.batch_size * world_size
    lr = args.lr * global_batch / 256  # linear scaling rule

    model = build_model(args.pretrained).to(device, memory_format=memory_format)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    warmup = min(args.warmup_steps, total_steps // 5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(warmup, total_steps))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out / "last.pt", out / "best.pt"
    start_epoch, step, best_top1 = 0, 0, 0.0
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, step, best_top1 = ckpt["epoch"] + 1, ckpt["step"], ckpt["best_top1"]
        log(f"Resumed from {last_path}: epoch {start_epoch}/{args.epochs} done, best val top-1 {best_top1:.4f}")

    log(f"{world_size} process(es) on {device.type}, {workers} loader workers each, AMP {'on' if amp else 'off'}, "
        f"channels_last {'on' if args.channels_last else 'off'}")
    log(f"train {len(datasets['train'])} / val {len(datasets['val'])} images, {steps_per_epoch} steps/epoch, "
        f"global batch {global_batch}, peak lr {lr:g}, warmup {warmup} steps")

    # On resume, hide anything a killed run logged after the checkpoint so the curves stay continuous
    writer = SummaryWriter(out / "tensorboard", purge_step=step + 1 if step else None) if is_main else None
    if is_main and start_epoch == 0:
        model.eval()  # trace in eval mode so BatchNorm statistics aren't touched
        writer.add_graph(model, torch.zeros(1, 3, 224, 224, device=device))

    raw_model = model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            loaders["train"].sampler.set_epoch(epoch)  # a different shuffle every epoch, same on all ranks
        model.train()
        totals = torch.zeros(3, device=device)  # loss sum, top-1 hits, images
        epoch_start = window_start = time.perf_counter()
        for i, (images, labels) in enumerate(loaders["train"]):
            if i >= steps_per_epoch:
                break
            if is_main and epoch == 0 and i == 0:
                writer.add_image("train/augmented_batch", make_grid(denormalize(images[:32]), nrow=8), 0)
            images = images.to(device, non_blocking=True, memory_format=memory_format)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            totals += torch.stack([loss.detach() * len(labels), (logits.argmax(1) == labels).sum(),
                                   torch.tensor(len(labels), device=device)]).float()

            if (i + 1) % args.log_every == 0:
                loss_value = loss.item()  # waits for the GPU, so the window timing below is accurate
                now = time.perf_counter()
                images_per_s = args.log_every * global_batch / (now - window_start)
                window_start = now
                log(f"epoch {epoch + 1} step {i + 1}/{steps_per_epoch}  loss {loss_value:.3f}  "
                    f"lr {scheduler.get_last_lr()[0]:.4g}  {images_per_s:.0f} img/s")
                if is_main:
                    writer.add_scalar("step/train_loss", loss_value, step)
                    writer.add_scalar("step/lr", scheduler.get_last_lr()[0], step)
                    writer.add_scalar("step/images_per_s", images_per_s, step)

        train_loss, train_hits, train_n = all_reduce_sum(totals).tolist()
        epoch_time = time.perf_counter() - epoch_start
        val_loss, val_top1, val_top5 = evaluate(model, loaders["val"], device, amp, memory_format, args.limit_val_batches)
        is_best = val_top1 > best_top1
        best_top1 = max(best_top1, val_top1)

        if is_main:
            save_checkpoint(last_path, {
                "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "step": step, "best_top1": best_top1, "args": vars(args),
            })
            if is_best:
                save_checkpoint(best_path, {"model": raw_model.state_dict(), "epoch": epoch,
                                            "val_top1": val_top1, "val_top5": val_top5, "args": vars(args)})
            n = epoch + 1  # x-axis for the "vs. epochs" plots
            writer.add_scalar("epoch/train_loss", train_loss / train_n, n)
            writer.add_scalar("epoch/train_top1", train_hits / train_n, n)
            writer.add_scalar("epoch/val_loss", val_loss, n)
            writer.add_scalar("epoch/val_top1", val_top1, n)
            writer.add_scalar("epoch/val_top5", val_top5, n)
            writer.add_scalar("epoch/images_per_s", train_n / epoch_time, n)
            writer.add_scalar("epoch/minutes", epoch_time / 60, n)
            writer.flush()
        log(f"== epoch {epoch + 1}/{args.epochs}  {epoch_time / 60:.1f} min ({train_n / epoch_time:.0f} img/s)  "
            f"train loss {train_loss / train_n:.3f} top-1 {train_hits / train_n:.4f}  "
            f"val loss {val_loss:.3f} top-1 {val_top1:.4f} top-5 {val_top5:.4f}" + ("  [best]" if is_best else ""))

    if is_main:
        writer.close()
        log(f"Done. Best val top-1 {best_top1:.4f}; checkpoints in {out}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
