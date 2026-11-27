#!/bin/bash
# Throughput diagnostic: times the GPU, the host-to-device copy and the DataLoader separately.
# Submit from the repo root:  mkdir -p logs && sbatch slurm/train_bench.sbatch
#SBATCH --job-name=places-bench
#SBATCH --account=irahbari_1147
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

PY="${PY:-$HOME/.conda/envs/torch-env/bin/python}"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"

bash scripts/extract_data.sh /project2/irahbari_1147/places365 "$TMPDIR/places365"

# GPU utilization and clock once a second, to read alongside the benchmark stages
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu --format=csv -l 1 \
    > "logs/places-bench-$SLURM_JOB_ID-gpu.csv" &

"$PY" src/train_bench.py --root "$TMPDIR/places365"
