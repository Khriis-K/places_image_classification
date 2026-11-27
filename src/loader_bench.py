"""Find where DataLoader time goes: disk reads, JPEG decoding, augmentation, or worker count.

Each stage is timed on its own so the slowest one stands out:
  1. read files, cold (never read before), 1 thread      -> filesystem latency
  2. read files, cold, many threads                       -> does parallel I/O help?
  3. read the same files again (now in the OS page cache) -> cost without the filesystem
  4. decode JPEG from memory                              -> CPU cost per image
  5. decode + training augmentation (with / without ColorJitter)
  6. full DataLoader at several worker counts, each on fresh (cold) files

Usage: python src/loader_bench.py --root /scratch1/$USER/places365
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import random
import time

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from data import IMAGENET_MEAN, IMAGENET_STD, available_cpus, build_datasets, build_transforms


def rate(n, seconds):
    return n / seconds if seconds > 0 else float("inf")


def report(name, n, seconds):
    print(f"  {name:48s} {rate(n, seconds):9.0f} images/s   ({seconds / n * 1000:6.2f} ms/image)")


def time_it(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def loader_rate(dataset, indices, batch_size, workers):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=workers,
                        pin_memory=False, shuffle=False)
    it = iter(loader)
    next(it)  # skip worker start-up and the first batch
    start, n = time.perf_counter(), 0
    for images, _ in it:
        n += images.shape[0]
    return rate(n, time.perf_counter() - start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--n", type=int, default=2000, help="images per file-read/decode stage")
    p.add_argument("--threads", type=int, default=32, help="threads for the parallel read test")
    p.add_argument("--workers", type=int, nargs="+", default=None, help="DataLoader worker counts to try")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--loader-batches", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)  # differs from explore.py so files start cold
    args = p.parse_args()

    cpus = available_cpus()
    torch.set_num_threads(cpus)  # torch otherwise sizes its thread pool to the whole node
    workers = args.workers or sorted({1, 4, cpus // 2, cpus} - {0})
    print(f"CPUs available to this job: {cpus}   (os.cpu_count={os.cpu_count()}, "
          f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')}, "
          f"SLURM_CPUS_ON_NODE={os.environ.get('SLURM_CPUS_ON_NODE')})")
    print(f"Data root: {args.root}\n")

    train = build_datasets(args.root)["train"]
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(train.samples)), 2 * args.n
                       + len(workers) * args.batch_size * args.loader_batches)
    paths = [train.image_dir / train.samples[i][0] for i in picks[:2 * args.n]]
    half = args.n // 2

    print("File reads")
    _, t = time_it(lambda: [p.read_bytes() for p in paths[:half]])
    report("cold, 1 thread", half, t)
    with ThreadPoolExecutor(args.threads) as pool:
        _, t = time_it(lambda: list(pool.map(Path.read_bytes, paths[half:args.n])))
    report(f"cold, {args.threads} threads", args.n - half, t)
    blobs, t = time_it(lambda: [p.read_bytes() for p in paths[:args.n]])
    report("warm (page cache), 1 thread", args.n, t)
    print(f"  average file size: {sum(map(len, blobs)) / len(blobs) / 1024:.1f} KB")

    print("\nCPU work per image (1 core, files already in memory)")
    decode = lambda b: Image.open(io.BytesIO(b)).convert("RGB")
    images, t = time_it(lambda: [decode(b) for b in blobs])
    report("JPEG decode", args.n, t)
    train_tf = build_transforms(train=True)
    no_jitter = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    _, t = time_it(lambda: [train_tf(im) for im in images])
    report("train transforms (crop, flip, jitter, normalize)", args.n, t)
    _, t = time_it(lambda: [no_jitter(im) for im in images])
    report("train transforms without ColorJitter", args.n, t)
    _, t = time_it(lambda: [train_tf(decode(b)) for b in blobs])
    per_core = rate(args.n, t)
    report("decode + train transforms", args.n, t)
    print(f"  -> CPU-bound ceiling with {cpus} cores: ~{per_core * cpus:.0f} images/s")

    print(f"\nFull DataLoader (batch {args.batch_size}, fresh cold files per run)")
    offset = 2 * args.n
    for w in workers:
        chunk = picks[offset:offset + args.batch_size * args.loader_batches]
        offset += len(chunk)
        print(f"  {w:3d} workers: {loader_rate(train, chunk, args.batch_size, w):9.0f} images/s")


if __name__ == "__main__":
    main()
