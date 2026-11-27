# Task 7 — Distributed Training (multiple GPUs, multiple nodes)

## Result

We trained the same ResNet-50 on **4 V100 GPUs spread over 2 nodes** with PyTorch **distributed data parallel (DDP)**. The 10-epoch run took **71.6 minutes** of training and validation, against **~265 minutes** on one GPU: a **3.7× speedup**, or **93% parallel efficiency**. Final accuracy was the same: **56.73% top-1 / 86.73% top-5** on our validation set, against 56.70% / 86.65% on one GPU.

| | 1 GPU (Task 4) | **4 GPUs, 2 nodes** |
|---|---|---|
| Run | `r50_lr0.04` | **`r50_ddp_2n4g`** |
| Global batch / peak learning rate | 256 / 0.04 | 1,024 / 0.16 |
| Training throughput (epochs 2–10) | 1,103 img/s | **4,122 img/s** |
| Training time, 10 epochs | 259.3 min | **70.0 min** |
| Validation time, 10 epochs | ~6 min | 1.7 min |
| **Training + validation** | **~265 min (4 h 25 min)** | **71.6 min (1 h 12 min)** |
| Whole Slurm job (incl. unpacking the dataset, start-up) | ~4.5 h if uninterrupted | 75 min |
| Speedup (training + validation) | 1× | **3.71×** |
| Parallel efficiency (speedup ÷ GPUs) | — | **93%** |
| Val top-1 / top-5 (best epoch = last epoch) | 56.70% / 86.65% | 56.73% / 86.73% |
| GPU-hours used | 4.4 | 4.8 |

**Speedup** is how many times faster the run finishes. **Parallel efficiency** is speedup divided by the number of GPUs: 100% would mean every added GPU helps as much as the first.

**Where the 1-GPU time comes from.** Task 4 reports 5 h 10 min for the 1-GPU run, but its TensorBoard timestamps show that two epochs were interrupted (epoch 2 took 59 min of wall-clock time and epoch 6 took 35, instead of ~26.5), costing about 42 minutes of restarts. The fair number is the uninterrupted time: 259.3 min of training (the sum of its logged epoch times) plus about 36 s of validation per epoch. Validation wasn't timed separately in Task 4, so that part is estimated from the gaps between epochs. The multi-GPU runs log both exactly (`timing.jsonl`, below).

## How distributed data parallel works

- **One process per GPU, each with a full copy of the model.** With 4 GPUs there are 4 copies of ResNet-50.
- **Each process trains on its own quarter of the data.** A `DistributedSampler` splits every epoch's shuffled image list into 4 disjoint parts, so each GPU sees different images and together they cover the whole training set once per epoch.
- **Gradients are averaged after every step.** After the backward pass, the GPUs exchange their gradients and each one ends up with the average over all 4 (an **all-reduce**). Every copy then applies the same update, so the 4 models stay identical. DDP starts sending gradients for the last layers while the backward pass is still computing the earlier ones, which hides much of the communication time.
- **The communication library is NCCL** (NVIDIA's GPU communication library). Within a node, data moves between the two GPUs over PCIe. Between nodes it goes over **InfiniBand**, a fast cluster network: CARC's V100 nodes have FDR InfiniBand at 56 Gb/s (`NET/IB` in the job log). The log also says "GPU Direct RDMA Disabled": gradients are copied through each node's main memory on the way to the network card, rather than straight from the GPU.

The training code already supported this (Task 4 wrote `src/train.py` for 1 or many GPUs). It wraps the model in `DistributedDataParallel`, sums the loss and accuracy counts over all GPUs before reporting them, and lets only process 0 write checkpoints and TensorBoard logs.

### Batch size and learning rate

Each GPU keeps a batch of 256 images, the same as on one GPU. The **global batch** (all GPUs together) is therefore 1,024, and one step processes 4× as many images, so an epoch has a quarter as many steps (1,673 instead of 6,692). To compensate for the fewer, larger steps, `train.py` scales the learning rate with the global batch (the **linear scaling rule**, Goyal et al., 2017): 0.04 × 1,024 / 256 = 0.16. The 500-step warmup was left as it was. Everything else (10 epochs, schedule, augmentation, label smoothing, AMP) is unchanged, so the only difference from the 1-GPU run is the number of GPUs and these two values.

## The Slurm job

`slurm/train_ddp.sbatch` asks for 2 nodes with 2 V100s, 16 CPUs and 128 GB each. It runs `slurm/ddp_node.sh` once per node with `srun`. That script:
1. **Unpacks the dataset into the node's own memory** (`$TMPDIR`, a 94 GB RAM disk on these nodes). Each node needs its own copy, because reading from the shared `/scratch1` is too slow (Task 2). The two nodes finished unpacking within 4 seconds of each other.
2. **Starts one training process per GPU with `torchrun`.** The first node hosts the rendezvous (the address every process connects to at start-up). The port is derived from the job ID, so two jobs on the same node can't collide.

Like the 1-GPU job, it checkpoints every epoch and requeues itself near the time limit. We ran it in the course's reserved node block (`--reservation=tac450-tu`).

```bash
mkdir -p logs
RUN=r50_ddp_2n4g SBATCH_EXCLUDE=d14-09 sbatch --reservation=tac450-tu --time=02:00:00 slurm/train_ddp.sbatch
```

Each finished epoch adds a line to `/scratch1/$USER/runs/<run>/timing.jsonl`: training and validation seconds, throughput, number of GPUs and nodes, and accuracy.

## Why 3.7× and not 4×

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 GPU, min | 26.4 | 25.8 | 25.9 | 25.9 | 25.9 | 25.8 | 25.8 | 25.8 | 26.0 | 26.0 |
| 4 GPUs, min | 7.6 | 7.0 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 |

- **Communication.** Each step every GPU has to send and receive ResNet-50's gradients: 25.6 M values, about 100 MB in fp32. Part of this can't be hidden behind the backward pass, and every GPU has to wait for the slowest one before the next step.
- **The first epoch is slower** (7.6 min, against 6.9 later). The data loader workers start up, and cuDNN tries several algorithms for each layer before settling on the fastest. This happens on one GPU too, but costs relatively more when the whole run is short.
- **Start-up costs don't shrink.** Unpacking the dataset (1–2 min) and starting the processes (43 s) take as long on 4 GPUs as on 1, so they are a bigger share of a 75-minute job than of a 4.5-hour one.

At a steady rate (epochs 2–10), 4 GPUs process 4,122 img/s against 1,103: 3.74×, or 93% efficiency. The data loader isn't the limit: each GPU still has 8 CPU workers, as on one GPU, and Task 3 measured those at 1,763 img/s per GPU, well above the ~1,030 img/s each GPU consumes here.

## Accuracy is unchanged

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 GPU, val top-1 % | 48.8 | 50.6 | 51.1 | 52.3 | 52.3 | 53.7 | 55.2 | 56.0 | 56.7 | 56.7 |
| 4 GPUs, val top-1 % | 48.2 | 49.9 | 47.8 | 51.1 | 52.1 | 53.3 | 54.8 | 55.9 | 56.6 | 56.7 |

The final accuracy differs by 0.03 points, far inside the ±0.2-point measurement noise of our 90,000-image validation set (Task 4). The 4-GPU run is a little behind in the middle (it dips to 47.8% at epoch 3), when its learning rate of 0.16 is near its peak. It catches up as the cosine schedule lowers the rate, the same pattern Task 4 saw when comparing learning rates 0.01 and 0.04.

## Cost

Finishing 3.7× sooner isn't free: 4 GPUs for 71.6 min is 4.8 GPU-hours, against 4.4 for one GPU, about 8% more of the course allocation, which is the 93% efficiency seen from the other side.

## Toward Task 8

The same reservation window also ran the 2-GPU (one node) and 8-GPU (4 nodes) configurations, so all four points of the scaling study are done:

| GPUs | Nodes | Run | Throughput (epochs 2–10) | Train + val | Speedup | Efficiency | Val top-1 |
|---|---|---|---|---|---|---|---|
| 1 | 1 | `r50_lr0.04` | 1,103 img/s | ~265 min | 1× | 100% | 56.70% |
| 2 | 1 | `r50_ddp_1n2g` | 2,109 img/s | 139.1 min | 1.91× | 95.5% | 56.81% |
| 4 | 2 | `r50_ddp_2n4g` | 4,122 img/s | 71.6 min | 3.71× | 92.6% | 56.73% |
| 8 | 4 | `r50_ddp_4n8g` | 8,070 img/s | 37.0 min | 7.18× | 89.8% | 56.75% |

Task 8 (`docs/task8_scaling.md`) turns this into speedup and efficiency plots and looks at where the lost time goes. One early observation: 2 GPUs on a single node, with no network involved, already lose 4% of perfect scaling, so not all the overhead comes from InfiniBand.
