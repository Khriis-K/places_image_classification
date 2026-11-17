#!/bin/bash
# Multi-GPU, multi-node training with distributed data parallel (Tasks 7-8). Submit from the repo root:
#   mkdir -p logs && RUN=r50_ddp_2n4g SBATCH_EXCLUDE=d13-02,d14-09 sbatch slurm/train_ddp.sbatch [extra train.py args]
# Default: 2 nodes x 2 V100 = 4 GPUs. Other layouts override the node/GPU/CPU/memory lines, e.g. 8 GPUs:
#   RUN=r50_ddp_4n8g sbatch --nodes=4 slurm/train_ddp.sbatch
#   RUN=r50_ddp_1n2g sbatch --nodes=1 slurm/train_ddp.sbatch
# Each node runs slurm/ddp_node.sh once: it unpacks the dataset to that node's $TMPDIR, then starts one
# training process per GPU with torchrun. Every process keeps a batch of 256 (global batch 256 x GPUs)
# and train.py scales the learning rate with the global batch.
# Resubmitting with the same RUN resumes from its last checkpoint, as with train_1gpu.sbatch.
#SBATCH --job-name=places-ddp
#SBATCH --account=irahbari_1147
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
set -euo pipefail

export RUN="${RUN:-r50_ddp_${SLURM_NNODES}n}"
export OUT="/scratch1/$USER/runs/$RUN"
export PY="${PY:-$HOME/.conda/envs/torch-env/bin/python}"
export PYTHONUNBUFFERED=1

# Rank 0 runs on the first node. Pick the port from the job ID so two jobs sharing a node don't collide.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
# NCCL prints which network it uses between nodes: "NET/IB" (InfiniBand) or "NET/Socket" (Ethernet)
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID, run $RUN -> $OUT"
echo "Nodes ($SLURM_NNODES): $SLURM_JOB_NODELIST; rendezvous at $MASTER_ADDR:$MASTER_PORT"
echo "Started $(date -Is)"
JOB_START=$SECONDS

trap 'echo "Time limit near: requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1

# One task per node. srun doesn't inherit --cpus-per-task from sbatch in recent Slurm, so pass it on,
# or each node's task would get a single CPU for all its data loader workers.
srun --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 \
    bash slurm/ddp_node.sh "$@" &
wait $!

echo "Finished $(date -Is); job wall time $(( (SECONDS - JOB_START) / 60 )) min (includes dataset unpacking)"
