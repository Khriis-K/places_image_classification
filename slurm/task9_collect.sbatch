#!/bin/bash
#SBATCH --job-name=task9-collect
#SBATCH --account=irahbari_1147
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
series=$1
"$PY" src/report_task9.py "$series"
tar -czf "$series.tar.gz.partial" -C "$(dirname "$series")" "$(basename "$series")"
mv "$series.tar.gz.partial" "$series.tar.gz"
echo "Download $series.tar.gz"
