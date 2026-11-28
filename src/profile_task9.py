"""Profile one Task 9 configuration on the project training pipeline."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler

from data import build_datasets, build_loaders
from train import build_model, evaluate, setup_distributed, warmup_cosine
from train_bench import train_step


def training_batch(iterator, model, optimizer, scaler, loss, device, layout, amp, annotate=False):
    region = record_function if annotate else nullcontext
    with region("task9/load_batch"):
        images, labels = next(iterator)
    with region("task9/host_to_device"):
        images = images.to(device, non_blocking=True, memory_format=layout)
        labels = labels.to(device, non_blocking=True)
    with region("task9/model_update"):
        train_step(model, optimizer, scaler, loss, images, labels, amp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=["fp32", "amp", "amp_channels_last"], required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Task 9 requires CUDA.")
    local_rank = int(os.environ["LOCAL_RANK"])
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if torch.cuda.device_count() != local_size:
        parser.error("The visible GPU count differs from the requested GPUs per node.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    rank, world, _ = setup_distributed()
    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        parser.error("CUDA profiling requires Kineto/CUPTI support in this PyTorch environment.")

    seed, batch, warmup, traced = 0, 256, 20, 5
    total = 1024 // world  # identical global image budget at all four scales
    timed = total - warmup - traced
    workers = int(os.environ["SLURM_CPUS_PER_TASK"]) // local_size
    amp = args.variant != "fp32"
    layout = torch.channels_last if args.variant == "amp_channels_last" else torch.contiguous_format
    torch.set_num_threads(workers)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    datasets = build_datasets(args.root, seed=seed)
    train_loader = build_loaders({"train": datasets["train"]}, batch, workers, distributed=True)["train"]
    # Uneven slices cover validation exactly once, including the final partial batch.
    shard = Subset(datasets["val"], range(rank, len(datasets["val"]), world))
    val_loader = build_loaders({"val": shard}, batch, workers)["val"]
    raw_model = build_model(True).to(device, memory_format=layout)
    model = DistributedDataParallel(raw_model, device_ids=[local_rank]) if world > 1 else raw_model
    global_batch = batch * world
    lr = 0.04 * global_batch / 256
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(min(500, total // 5), total))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    torch.manual_seed(seed + rank)  # independent, repeatable augmentation streams
    iterator = iter(train_loader)

    def step(annotate=False):
        training_batch(iterator, model, opt, scaler, loss, device, layout, amp, annotate)
        schedule.step()

    variant_dir = args.out / args.variant
    out = variant_dir / f"rank-{rank}"
    out.mkdir(parents=True, exist_ok=False)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    # NVML indices need not match CUDA's process-local ordinal under Slurm.
    props = torch.cuda.get_device_properties(device)
    gpu_id = str(props.uuid)
    if not gpu_id.startswith("GPU-"):
        gpu_id = "GPU-" + gpu_id
    query = "timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw"
    dist.barrier(device_ids=[local_rank])
    torch.cuda.synchronize()
    with (out / "gpu.csv").open("w") as log:
        log.write(query + "\n")
        log.flush()
        monitor = subprocess.Popen(["nvidia-smi", "-i", gpu_id, "--query-gpu=" + query,
                                    "--format=csv,noheader,nounits", "-l", "1"], stdout=log)
        try:
            start = time.perf_counter()
            for _ in range(timed):
                step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if monitor.poll() is not None:
                raise RuntimeError(f"nvidia-smi exited early ({monitor.returncode}); check GPU UUID {gpu_id}")
        finally:
            monitor.terminate()
            monitor.wait()

    elapsed = torch.tensor(seconds, device=device, dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    seconds = elapsed.item()

    # These last five updates are excluded from throughput and included in every accuracy run.
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities,
                 schedule=torch.profiler.schedule(wait=0, warmup=1, active=traced - 1, repeat=1),
                 on_trace_ready=tensorboard_trace_handler(str(variant_dir), worker_name=f"rank-{rank}", use_gzip=True)) as prof:
        for _ in range(traced):
            step(annotate=True)
            prof.step()
        torch.cuda.synchronize()
    ops = prof.key_averages()
    (out / "operators.txt").write_text(ops.table(sort_by="self_device_time_total", row_limit=30))
    del iterator, train_loader
    # Evaluate rank 0's BatchNorm running statistics on every validation shard.
    if world > 1:
        for buf in raw_model.buffers():
            dist.broadcast(buf, src=0)
    raw_model.to(memory_format=torch.contiguous_format)
    val_loss, top1, top5 = evaluate(raw_model, val_loader, device, False, torch.contiguous_format)
    result = dict(variant=args.variant, seed=seed, batch=batch, workers=workers, warmup=warmup,
                  rank=rank, world=world, nodes=world // local_size, host=socket.gethostname(),
                  global_batch=global_batch, peak_lr=lr, timed=timed, traced=traced,
                  train_images=total * global_batch, val_images=len(datasets["val"]),
                  train_seconds=seconds, images_per_second=timed * global_batch / seconds,
                  val_loss=val_loss, val_top1=top1, val_top5=top5, init="IMAGENET1K_V2",
                  gpu=props.name, gpu_uuid=gpu_id, torch=str(torch.__version__), cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), job_id=os.environ.get("SLURM_JOB_ID"))
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if rank == 0:
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
