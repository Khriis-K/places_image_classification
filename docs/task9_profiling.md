# Task 9 · Profiling the Task 8 GPU configurations

## Submit on Discovery

Upload **places-task9.tar.gz** to your home folder in CARC OnDemand. Open **Clusters → Discovery Cluster Shell Access** and run:

```bash
cd ~
tar -xzf places-task9.tar.gz
cd places-task9
bash scripts/submit_task9.sh
```

From a checkout that already contains Task 9, run the same submission script from the repository root. It prints four GPU job IDs, a dependent collection job ID, and a unique results directory. Keep the extracted directory in place until the collector finishes. Each submission starts a fresh series. If the course provides an active GPU reservation, use `TASK9_RESERVATION=name bash scripts/submit_task9.sh` to apply it to the four GPU jobs.

The launcher checks the course's shared data archives and caches the torchvision ImageNet V2 weights on the login node. It reuses the project CUDA environment if compatible with V100, or creates a dedicated Python 3.11 environment. First-time setup needs network access. Override `PY` with a compatible Python executable or `PLACES_ARCHIVES` with the directory holding the three project archives if needed.

## Runs and resources

| V100 GPUs | Nodes | GPUs/node | CPUs/node | RAM/node | Global batch | Peak LR | Training batches per rank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 8 | 64 GB | 256 | 0.04 | 1,024 |
| 2 | 1 | 2 | 16 | 128 GB | 512 | 0.08 | 512 |
| 4 | 2 | 2 | 16 | 128 GB | 1,024 | 0.16 | 256 |
| 8 | 4 | 2 | 16 | 128 GB | 2,048 | 0.32 | 128 |

Each GPU job requests up to 90 minutes on the course account `irahbari_1147`. Each node stages Places365 once on its own local storage. The three variants run sequentially on the same allocation. Task 8 used these same topologies, 256 images per GPU and the same linear LR scaling.

1. **FP32** establishes the real-data baseline with contiguous NCHW tensors.
2. **FP16 AMP** adds autocast and gradient scaling with the original layout.
3. **FP16 AMP + channels-last** changes the model and image memory layout.

Every variant starts from the same pretrained ResNet-50 weights and seeded classifier, uses the project's SGD and augmentation settings, and sees the same 262,144 training examples globally. A seeded sampler partitions a common shuffled sequence across ranks. The learning-rate warm-up and cosine decay are compressed for the shorter budget. Top-1 and top-5 use a single center crop in FP32 with rank 0's final model buffers. Each rank evaluates a disjoint shard of the project validation split and reduces hit counts across ranks. The official test set stays reserved for Task 6.

The throughput window starts after 20 warm-up batches and ends before the final five profiler batches. The rate uses global images divided by the slowest rank's synchronized elapsed time. Each rank runs PyTorch Profiler (one warm-up and four active steps) and records one-second GPU utilization samples during timing. The traces include data loading, host-to-device transfer, model updates and distributed NCCL kernels. The summary compares adjacent optimizations within each GPU count and computes throughput scaling relative to the one-GPU run of the same variant.

The milestone's 56.7% top-1 and 86.7% top-5 came after ten epochs. Task 9 measures early accuracy at an equal training-image budget. Task 8's reported total-time speedup also includes validation, whereas the Task 9 scaling table uses steady-state training throughput.

## Retrieve results

Use `squeue -u "$USER"` to monitor the jobs. The launcher saves the job IDs in `runs/task9-XXXXXXXX/jobs.txt`, where the suffix is unique to your submission. Inspect the outcome of a particular job with `sacct -j JOBID --format=JobID,State,Elapsed,ExitCode`. Its log is in `logs/places-task9-JOBID.out`; the collector uses `logs/task9-collect-JOBID.out`.

After all four GPU jobs succeed, the CPU collector verifies the 12 runs and all 45 rank traces, produces **report.md**, and creates the single **runs/task9-XXXXXXXX.tar.gz** file printed by the launcher. Download that archive through the OnDemand Files browser. It contains the combined report, raw metrics for each rank, GPU CSVs, operator tables and compressed profiler traces. If a job fails, its log and partial evidence remain available, and the completed jobs' results stay in the series directory.

To view a topology in TensorBoard, extract the results locally, install the viewer in your Python environment, and substitute the real series name:

```bash
python -m pip install tensorboard torch-tb-profiler
tensorboard --logdir task9-XXXXXXXX/4gpu
```

Open the **Profile** tab, select an optimization variant and rank, then choose **Trace**. Compare gaps on the GPU timelines with loader waits, transfers and NCCL activity. Device communication can overlap backward computation.
