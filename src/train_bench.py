"""Find what limits training throughput: the GPU itself, the copy to the GPU, or the DataLoader.

Each stage is timed on its own, in images/s:
  1. GPU only, synthetic batch already on the GPU   -> ceiling of the model, per precision/memory layout
       fp32 | AMP (fp16) | AMP + channels_last
  2. host-to-device copy of a pinned batch           -> cost of moving data to the GPU
  3. DataLoader only (no model)                       -> how fast the CPUs can supply images in this job
  4. full pipeline: DataLoader + AMP training step    -> should reproduce train.py's number
Whichever of 1-3 is closest to 4 is the bottleneck.

Usage: python src/train_bench.py --root $TMPDIR/places365
"""
import argparse
import time

import torch
import torch.nn as nn

from data import available_cpus, build_datasets, build_loaders
from train import build_model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_step(model, optimizer, scaler, criterion, images, labels, amp):
    with torch.autocast(images.device.type, dtype=torch.float16, enabled=amp):
        loss = criterion(model(images), labels)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def make_trainer(device, amp, channels_last):
    model = build_model(pretrained=False).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    return model, optimizer, scaler


def gpu_only(device, batch_size, steps, warmup, amp, channels_last):
    model, optimizer, scaler = make_trainer(device, amp, channels_last)
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    if channels_last:
        images = images.to(memory_format=torch.channels_last)
    labels = torch.randint(0, 365, (batch_size,), device=device)
    for _ in range(warmup):  # cudnn.benchmark tries kernels during the first steps
        train_step(model, optimizer, scaler, criterion, images, labels, amp)
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        train_step(model, optimizer, scaler, criterion, images, labels, amp)
    sync(device)
    return steps * batch_size / (time.perf_counter() - start)


def host_to_device(device, batch_size, steps):
    batch = torch.randn(batch_size, 3, 224, 224).pin_memory() if device.type == "cuda" else torch.randn(batch_size, 3, 224, 224)
    batch.to(device)
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        batch.to(device, non_blocking=True)
    sync(device)
    return steps * batch_size / (time.perf_counter() - start)


def loader_only(loader, steps, warmup):
    it = iter(loader)
    for _ in range(warmup):  # worker start-up and first batches
        next(it)
    start, n = time.perf_counter(), 0
    for _ in range(steps):
        n += next(it)[0].shape[0]
    return n / (time.perf_counter() - start)


def full_pipeline(loader, device, steps, warmup, amp, channels_last):
    model, optimizer, scaler = make_trainer(device, amp, channels_last)
    criterion = nn.CrossEntropyLoss()
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    it = iter(loader)
    start, n = None, 0
    for i in range(warmup + steps):
        if i == warmup:
            sync(device)
            start = time.perf_counter()
        images, labels = next(it)
        images = images.to(device, non_blocking=True, memory_format=memory_format)
        labels = labels.to(device, non_blocking=True)
        train_step(model, optimizer, scaler, criterion, images, labels, amp)
        if i >= warmup:
            n += images.shape[0]
    sync(device)
    return n / (time.perf_counter() - start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=100, help="timed steps per stage")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # same as train.py
    cpus = available_cpus()
    torch.set_num_threads(cpus)
    workers = cpus if args.workers is None else args.workers
    amp_ok = device.type == "cuda"

    print(f"torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}, device "
          f"{torch.cuda.get_device_name() if amp_ok else 'cpu'}, {cpus} CPUs, batch {args.batch_size}", flush=True)

    print("\n1. GPU only (synthetic batch already on the GPU)")
    for name, amp, cl in [("fp32", False, False), ("AMP", True, False), ("AMP + channels_last", True, True)]:
        if amp and not amp_ok:
            continue
        rate = gpu_only(device, args.batch_size, args.steps, args.warmup, amp, cl)
        print(f"  {name:28s} {rate:7.0f} images/s", flush=True)

    print("\n2. Host-to-device copy (pinned memory)")
    print(f"  {'one fp32 batch':28s} {host_to_device(device, args.batch_size, args.steps):7.0f} images/s", flush=True)

    loader = build_loaders({"train": build_datasets(args.root)["train"]}, args.batch_size, workers,
                           pin_memory=amp_ok)["train"]
    print(f"\n3. DataLoader only ({workers} workers, train transforms)")
    print(f"  {'loader':28s} {loader_only(loader, args.steps, args.warmup):7.0f} images/s", flush=True)

    print("\n4. Full pipeline: DataLoader + training step")
    for name, amp, cl in [("AMP (= train.py)", True, False), ("AMP + channels_last", True, True)]:
        if amp and not amp_ok:
            continue
        rate = full_pipeline(loader, device, args.steps, args.warmup, amp, cl)
        print(f"  {name:28s} {rate:7.0f} images/s", flush=True)
    if not amp_ok:
        print(f"  {'fp32 (CPU run)':28s} {full_pipeline(loader, device, args.steps, args.warmup, False, False):7.0f} images/s")


if __name__ == "__main__":
    main()
