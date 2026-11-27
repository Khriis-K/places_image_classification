#!/bin/bash
# Single-GPU training job (Tasks 4-6). Submit from the repo root:
#   mkdir -p logs && RUN=r50_1gpu sbatch slurm/train_1gpu.sbatch [extra train.py args]
# RUN names the run directory /scratch1/$USER/runs/$RUN (checkpoints + TensorBoard).
# Resubmitting with the same RUN resumes from its last checkpoint.
#SBATCH --job-name=places-1gpu
#SBATCH --account=irahbari_1147
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
set -euo pipefail

RUN="${RUN:-r50_1gpu}"
OUT="/scratch1/$USER/runs/$RUN"
PY="${PY:-$HOME/.conda/envs/torch-env/bin/python}"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), run $RUN -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 5 min before the time limit Slurm sends USR1: requeue so the job restarts and resumes from last.pt
trap 'echo "Time limit near: requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1

# Node-local RAM copy of the dataset (Task 2: 5.5x faster loading than /scratch1)
bash scripts/extract_data.sh /project2/irahbari_1147/places365 "$TMPDIR/places365"

# Run in the background and wait, so the trap above can fire while training is running
"$PY" src/train.py --root "$TMPDIR/places365" --out "$OUT" "$@" &
wait $!
