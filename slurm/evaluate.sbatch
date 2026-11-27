#!/bin/bash
# Task 6 evaluation job: test-set predictions for the final model (single-crop and 10-crop) and the
# lr 0.01 comparison run. Submit from the repo root:
#   mkdir -p logs && SBATCH_EXCLUDE=d13-02,d14-09 sbatch slurm/evaluate.sbatch
# Writes logs/task6/<name>.npz (in the home directory, not /scratch1) for src/plot_evaluation.py.
#SBATCH --job-name=places-eval
#SBATCH --account=irahbari_1147
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

PY="${PY:-$HOME/.conda/envs/torch-env/bin/python}"
RUNS="/scratch1/$USER/runs"
OUT="$SLURM_SUBMIT_DIR/logs/task6"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/extract_data.sh /project2/irahbari_1147/places365 "$TMPDIR/places365"

# Copy checkpoints to node-local storage first, retrying: /scratch1 sometimes drops a node (errno 108)
fetch() {
    for attempt in 1 2 3; do
        cp "$RUNS/$1/best.pt" "$TMPDIR/$1.pt" && return 0
        echo "WARNING: copying $RUNS/$1/best.pt failed (attempt $attempt/3)"; sleep 30
    done
    return 1
}

evaluate() {  # evaluate <run> <output name> [evaluate.py args]
    local run="$1" name="$2"; shift 2
    echo; echo "=== $name ==="
    "$PY" src/evaluate.py --root "$TMPDIR/places365" --checkpoint "$TMPDIR/$run.pt" --out "$OUT" --name "$name" "$@"
}

fetch r50_lr0.04
fetch r50_1gpu
evaluate r50_lr0.04 r50_lr0.04
evaluate r50_lr0.04 r50_lr0.04_10crop --ten-crop
evaluate r50_1gpu r50_1gpu
ls -l "$OUT"
