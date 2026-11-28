#!/bin/bash
#SBATCH --job-name=places-task9
#SBATCH --account=irahbari_1147
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Resources and output directory are passed by scripts/submit_task9.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
out="$1/${2}gpu"
mkdir "$out"
srun --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 \
    bash slurm/task9_node.sh "$out" "$(($2 / SLURM_NNODES))"
