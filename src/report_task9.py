"""Build a Task 9 comparison from completed training runs and their raw evidence."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
from statistics import mean


VARIANTS = {"fp32": "FP32", "amp": "FP16 AMP", "amp_channels_last": "FP16 AMP + channels-last"}


def read_evidence(run, metrics):
    trace, = run.parent.glob(f"{run.name}.*.pt.trace.json.gz")
    with gzip.open(trace, "rt") as f:
        events = json.load(f)["traceEvents"]
    loader_us = 0
    kernels = Counter()
    copies = 0
    steps = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        name, us = event["name"], event["dur"]
        if event.get("cat") == "kernel":
            kernels[name] += us
        elif event.get("cat") == "gpu_memcpy" and "htod" in name.lower():
            copies += us
        elif event.get("cat") in {"user_annotation", "cpu_op"} and name == "task9/load_batch":
            loader_us += us
            steps += 1
    if not kernels or not copies or not steps:
        raise ValueError(f"{trace}: missing CUDA kernels, HtoD copies or annotated training steps")
    with (run / "gpu.csv").open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        loads = [float(row["utilization.gpu"]) for row in reader
                 if row["uuid"].strip().lower() == metrics["gpu_uuid"].lower()]
    if not loads:
        raise ValueError(f"{run}: no utilization samples for the training GPU")
    nccl = sum(us for name, us in kernels.items() if "nccl" in name.lower())
    if metrics["world"] > 1 and nccl == 0:
        raise ValueError(f"{trace}: no NCCL kernels in distributed training trace")
    return dict(wait=loader_us / steps / 1000, copy=copies / steps / 1000,
                nccl=nccl / steps / 1000, util=mean(loads), samples=len(loads))


def read_configuration(folder, variant, world):
    records, profiles = [], []
    for rank in range(world):
        run = folder / f"rank-{rank}"
        m = json.loads((run / "metrics.json").read_text())
        if (m["variant"], m["world"], m["rank"], m["nodes"]) != (variant, world, rank, max(1, world // 2)):
            raise ValueError(f"Unexpected configuration in {run}")
        if (m["batch"], m["global_batch"], m["train_images"]) != (256, 256 * world, 262144):
            raise ValueError(f"Unexpected training budget in {run}")
        records.append(m)
        profiles.append(read_evidence(run, m))
    m = records[0]
    if len({r["gpu_uuid"] for r in records}) != world or len({r["host"] for r in records}) != m["nodes"]:
        raise ValueError(f"GPU or node placement mismatch in {folder}")
    for key in ("images_per_second", "val_top1", "val_top5", "val_images", "job_id"):
        if len({r[key] for r in records}) != 1:
            raise ValueError(f"Ranks disagree on {key} in {folder}")
    return m, profiles


def render(root):
    lines = ["# Task 9 · Places365 profiling on CARC", "",
             "| GPUs | Nodes | Configuration | Images/s | Speedup vs previous | Val top-1 | Change (pp) | Val top-5 | Change (pp) |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    evidence = ["| GPUs | Configuration | Step (ms) | GPU busy | Samples | Loader wait (ms) | HtoD (ms) | NCCL (ms) |",
                "|---:|---|---:|---:|---:|---:|---:|---:|"]
    scaling = ["| GPUs | FP32 | FP16 AMP | AMP + channels-last |", "|---:|---:|---:|---:|"]
    baseline = {}
    notes = []
    for world in (1, 2, 4, 8):
        previous = None
        speedups = []
        for variant, label in VARIANTS.items():
            folder = root / f"{world}gpu" / variant
            m, profiles = read_configuration(folder, variant, world)
            if previous:
                for key in ("job_id", "seed", "torch", "init", "workers", "warmup", "timed", "traced", "val_images"):
                    if m[key] != previous[key]:
                        raise ValueError(f"Inconsistent {key} in {folder}")
            speed = m["images_per_second"]
            if world == 1:
                baseline[variant] = speed
            speedups.append(f"{speed / baseline[variant]:.2f}×")
            top1, top5 = 100 * m["val_top1"], 100 * m["val_top5"]
            ratio = f'{speed / previous["images_per_second"]:.2f}×' if previous else "—"
            d1 = f'{top1 - 100 * previous["val_top1"]:+.2f}' if previous else "—"
            d5 = f'{top5 - 100 * previous["val_top5"]:+.2f}' if previous else "—"
            lines.append(f"| {world} | {m['nodes']} | {label} | {speed:,.1f} | {ratio} | {top1:.2f}% | {d1} | {top5:.2f}% | {d5} |")
            wait, copy, nccl, util = [mean(p[k] for p in profiles) for k in ("wait", "copy", "nccl", "util")]
            samples = sum(p["samples"] for p in profiles)
            step_ms = 1000 * m["global_batch"] / speed
            evidence.append(f"| {world} | {label} | {step_ms:.1f} | {util:.1f}% | {samples} | {wait:.2f} | {copy:.2f} | {nccl:.2f} |")
            if util >= 90 and wait + copy < 0.2 * step_ms:
                finding = "High GPU activity and short input times point to work on the GPU as the main limit."
            elif util < 85 and wait > 0.3 * step_ms:
                finding = "The training thread spent a substantial part of a step waiting for input while GPU activity was low."
            else:
                finding = "The samples and stage times leave the limiting stage uncertain."
            if world > 1:
                finding += f" NCCL kernels accounted for {nccl:.1f} ms per traced step per rank, including time overlapping computation."
            notes.append(f"- **{world} GPU{'s' if world > 1 else ''}, {label}.** {finding}")
            previous = m
        scaling.append(f"| {world} | " + " | ".join(speedups) + " |")

    lines.extend(["", "Each run processed 262,144 training examples and evaluated the full "
                  f"{m['val_images']:,}-image validation split in FP32. Deltas compare adjacent "
                  "configurations at the same scale. Accuracy is measured after a short fine-tune, once per configuration.",
                  "", "## Throughput scaling relative to one GPU", "", *scaling,
                  "", "## Bottleneck evidence", "", *evidence, "",
                  "Step time uses the synchronized, unprofiled training window. GPU busy is mean utilization "
                  "sampled once per second per GPU during that window. The remaining times average the four traced "
                  "steps across all ranks. Loader wait measures the host thread fetching a batch. HtoD and NCCL "
                  "are summed device durations and may overlap compute. NCCL duration alone cannot measure exposed communication overhead.",
                  "", *notes, ""])
    (root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
