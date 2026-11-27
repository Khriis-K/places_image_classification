# Task 4 — Model Training (single GPU)

## Result

The final model is ResNet-50 fine-tuned from ImageNet weights for 10 epochs at learning rate 0.04, on one V100. It reaches **56.70% top-1 / 86.65% top-5** on our validation set.

| Run | Learning rate | Val top-1 | Val top-5 | Train top-1 | Time |
|---|---|---|---|---|---|
| `r50_1gpu` | 0.01 | 56.32% | 86.28% | 59.4% | 4 h 31 min |
| **`r50_lr0.04`** (final) | **0.04** | **56.70%** | **86.65%** | 60.3% | 5 h 10 min |

Checkpoints are in `/scratch1/$USER/runs/<run>/`. `best.pt` (93 MB) holds the weights with the highest validation top-1. `last.pt` (186 MB) also holds the optimizer state, for resuming.

**How to read these numbers.** Our validation set is the 5% of the training images we held out (Task 2), and we score one center crop per image. The paper's numbers in Task 3 (ResNet-152 fine-tuned: 54.74% / 85.08%) are on the official validation set, which is our *test* set, and average 10 crops. The two aren't directly comparable. Task 6 measures the final model on the test set.

**Overfitting check.** Training accuracy ends only 3.6 points above validation accuracy, and validation accuracy rose every epoch. The model is not overfitting. (Training accuracy is measured on augmented images with label smoothing, so a gap this small is expected.)

## Training setup

| Setting | Value | Why |
|---|---|---|
| Model | ResNet-50, `IMAGENET1K_V2` weights, new 365-class final layer, all layers trained | Task 3 |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 | Standard for ResNets |
| Learning rate | 0.04 per 256 images, scaled linearly with the global batch | Chosen by the sweep below |
| Schedule | 500 warmup steps, then cosine decay to zero | Warmup protects the pretrained weights from large early updates |
| Batch | 256 per GPU | Fits on a 32 GB V100 with AMP |
| Loss | Cross-entropy with label smoothing 0.1 | Places365 classes overlap (Task 1) |
| Precision | fp16 mixed precision (AMP) with a gradient scaler | 3× faster than fp32; the V100 has no bf16 |
| Memory layout | channels_last | 1.5× faster; see below |
| Epochs | 10 | ~26 min each |

The code is `src/train.py`, run as a Slurm batch job by `slurm/train_1gpu.sbatch` (8 CPUs, 64 GB, one V100). The job unpacks the dataset into node-local memory (`$TMPDIR`, about 60 s) so the data loader never reads from the shared file system.

## Not losing progress

A 10-epoch run takes about 4.5 hours, so the job is built to survive interruptions:
- **Checkpoint every epoch.** `last.pt` is written to a temporary file and then renamed, so a crash mid-write never leaves a corrupt checkpoint. Resubmitting with the same `RUN` name resumes from the last finished epoch.
- **Requeue near the time limit.** Slurm sends the job a signal 5 minutes before its time runs out. The job then puts itself back in the queue (`--signal=B:USR1@300`, `scontrol requeue`) and resumes when it restarts.
- **Survive storage outages.** During the sweep, `/scratch1` dropped its connection on three different nodes ("Cannot send after transport endpoint shutdown") and crashed four jobs. In response, `train.py` now retries checkpoint saves three times, 30 s apart. It also writes TensorBoard logs to node-local storage and copies them to `/scratch1` once per epoch. A failed save or copy now prints a warning and training continues.

## Throughput: why the first run was slow

The first test job ran at **741 images/s**, about two-thirds of NVIDIA's reference (1,156 images/s for ResNet-50 with AMP on a V100). `src/train_bench.py` timed each stage of training on its own to find the bottleneck:

| Stage | Images/s |
|---|---|
| GPU alone, fp32 | 354 |
| GPU alone, AMP | 772 |
| GPU alone, AMP + channels_last | 1,186 |
| Data loader alone (8 CPUs) | 1,763 |
| Full pipeline, AMP | 743 |
| Full pipeline, AMP + channels_last | **1,108** |

The data loader was fast enough, and the GPU on its own was just as slow as the full pipeline. So the GPU itself was the bottleneck. The fix was the **memory layout**: the V100's Tensor Cores need **channels-last** data (NHWC, each pixel's channel values stored together). PyTorch's default is NCHW, so cuDNN was converting the layout around every convolution. Converting the model and inputs to channels-last once removes that cost. `train.py` now does this by default (`--no-channels-last` turns it off, for Task 9's comparison). Epochs dropped from ~38 to ~26 minutes.

## Hyperparameter tuning: learning-rate sweep

The **learning rate** (how big a step the optimizer takes on each update) matters most when fine-tuning. If it's too small, the model learns slowly. If it's too large, it overwrites what the pretrained weights already know. We ran a **log-scale sweep**, doubling the rate each time, with 3-epoch runs (the cosine schedule decays fully over the 3 epochs). Everything else was held fixed. The script is `slurm/lr_sweep.sh`.

| Learning rate | Epoch 1 val top-1 | Epoch 3 val top-1 | Epoch 3 val top-5 | Gain over the previous rate |
|---|---|---|---|---|
| 0.005 | 51.19% | 53.28% | 84.39% | — |
| 0.01 | 52.20% | 54.51% | 85.22% | +1.23 |
| 0.02 | 52.21% | 55.21% | 85.68% | +0.70 |
| **0.04** | 51.03% | **55.67%** | **85.92%** | +0.46 |
| 0.08 | 47.08% | 55.58% | 85.80% | −0.09 |

**Stopping rule.** After 0.04 came out on top, we wrote down a rule before running 0.08:
- If 0.08 is worse, or better by less than ~0.3 points (about the run-to-run noise), choose 0.04 and stop.
- If 0.08 is better by more than ~0.5 points, try 0.16.

For scale: the validation set has ~90,000 images, so the measurement error on top-1 is about ±0.2 points. 0.08 was slightly worse, so we chose 0.04.

**Why not higher.** At 0.08, epoch 1 is 4 points behind every other rate: the large early steps overwrite the ImageNet features. It catches up only because the cosine schedule lowers the rate later. A lower rate is also safer for Task 8: under the linear scaling rule, 8 GPUs multiply the rate by 8, giving 0.32 at our setting.

**Did it carry over to 10 epochs?** Yes, but the gain shrank. At 3 epochs, 0.04 beat 0.01 by 1.16 points. At 10 epochs it beat it by 0.38 points (56.70% vs 56.32%). A longer run gives a lower rate enough steps to catch up, so short sweeps tend to overstate the benefit of a higher rate. The 0.04 run also starts behind: it trails the 0.01 run through epoch 8 and pulls ahead only in epoch 9, once the learning rate has decayed.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| lr 0.01, val top-1 | 52.0 | 53.1 | 54.1 | 54.6 | 55.0 | 55.3 | 55.8 | 56.1 | 56.3 | 56.3 |
| lr 0.04, val top-1 | 48.8 | 50.6 | 51.1 | 52.3 | 52.3 | 53.7 | 55.2 | 56.0 | 56.7 | 56.7 |

Other hyperparameters (weight decay, label smoothing, warmup length, batch size) were kept at the standard values in Task 3. Each 3-epoch sweep run costs about 1.3 GPU-hours, and the learning rate usually has by far the largest effect when fine-tuning.

## Compute used

| Runs | GPU-hours |
|---|---|
| Test jobs and throughput benchmark | ~1 |
| 10-epoch run, lr 0.01 | 4.5 |
| Sweep, 5 × 3 epochs (plus four jobs lost to `/scratch1` errors) | ~7.5 |
| 10-epoch run, lr 0.04 | 5.2 |
| **Total** | **~18** |

## For later tasks

- **Task 5 (visualization):** TensorBoard logs are in `/scratch1/$USER/runs/<run>/tensorboard`. They include per-step loss, learning rate and throughput; per-epoch loss and accuracy; the model graph; and a grid of augmented training images. Compare all sweep runs at once with `--logdir /scratch1/$USER/runs/lr_sweep`.
- **Task 6 (evaluation):** evaluate `/scratch1/$USER/runs/r50_lr0.04/best.pt` on the test set.
- **Tasks 7–8 (scaling):** `train.py`'s default learning rate is now 0.04. Watch for divergence (the loss jumping up instead of falling) at 8 GPUs, where the scaled rate is 0.32.
