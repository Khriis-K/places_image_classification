#!/bin/bash
# Run on the Discovery login node from the unpacked bundle.
set -euo pipefail
cd "$(dirname "$0")/.."
src="${PLACES_ARCHIVES:-/project2/irahbari_1147/places365}"
for file in train_256_places365standard.tar val_256.tar filelist_places365-standard.tar; do
    if [[ ! -r "$src/$file" ]]; then
        echo "Cannot read $src/$file. Check course project access or set PLACES_ARCHIVES." >&2
        exit 1
    fi
done
# Keep library imports within the login node's thread limit.
login_python() {
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"
}
compatible() {
    [[ -x "$1" ]] && login_python "$1" -c 'import torch, torchvision; from torch.profiler import tensorboard_trace_handler; assert torch.version.cuda and hasattr(torch.amp, "GradScaler") and "sm_70" in torch._C._cuda_getArchFlags().split()'
}
if [[ -n "${PY:-}" ]]; then
    compatible "$PY" || { echo "PY needs a V100-compatible CUDA PyTorch (sm_70) and matching torchvision." >&2; exit 1; }
elif compatible "$HOME/.conda/envs/torch-env/bin/python"; then
    PY="$HOME/.conda/envs/torch-env/bin/python"
else
    env_dir="$HOME/.conda/envs/places-task9"
    PY="$env_dir/bin/python"
    if ! compatible "$PY"; then
        module load conda
        export CONDA_PKGS_DIRS="/scratch1/$USER/.conda/pkgs"
        if [[ ! -x "$PY" ]]; then
            conda create -y --prefix "$env_dir" python=3.11 pip
        fi
        "$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
    fi
fi
export PY
# Cache weights on the login node so compute nodes can train offline.
login_python "$PY" -c 'from torchvision.models import ResNet50_Weights; ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True)'
mkdir -p logs runs
series=$(mktemp -d "$PWD/runs/task9-XXXXXXXX")
echo "Results directory: $series"
jobs=()
reservation=()
if [[ -n "${TASK9_RESERVATION:-}" ]]; then reservation=(--reservation="$TASK9_RESERVATION"); fi
for gpus in 1 2 4 8; do
    per_node=2
    if [[ $gpus == 1 ]]; then per_node=1; fi
    nodes=$((gpus / per_node))
    job=$(sbatch ${reservation[@]+"${reservation[@]}"} --parsable --nodes="$nodes" --gres="gpu:v100:$per_node" \
        --cpus-per-task="$((8 * per_node))" --mem="$((64 * per_node))G" \
        slurm/task9.sbatch "$series" "$gpus")
    jobs+=("${job%%;*}")
    printf '%s GPU(s): Slurm job %s\n' "$gpus" "$job" | tee -a "$series/jobs.txt"
done
dependency=$(IFS=:; echo "${jobs[*]}")
sbatch --dependency="afterok:$dependency" --kill-on-invalid-dep=yes \
    slurm/task9_collect.sbatch "$series" | tee -a "$series/jobs.txt"
echo "Download after collection: $series.tar.gz"
