#!/bin/bash
# Task 4 learning-rate sweep: one 3-epoch single-GPU job per learning rate, all queued at once.
# Run from the repo root:  bash slurm/lr_sweep.sh [lr ...]
# Each run lands in /scratch1/$USER/runs/lr_sweep/lr<value>; compare them in TensorBoard with
#   --logdir /scratch1/$USER/runs/lr_sweep
set -euo pipefail

LRS="${*:-0.005 0.01 0.02 0.04}"
mkdir -p logs
for lr in $LRS; do
    RUN="lr_sweep/lr$lr" sbatch --job-name="sweep-lr$lr" --time=02:00:00 \
        slurm/train_1gpu.sbatch --epochs 3 --lr "$lr"
done
