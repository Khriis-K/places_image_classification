#!/bin/bash
set -euo pipefail
out=$1
gpus=$2
root="$TMPDIR/places365"
bash scripts/extract_data.sh "${PLACES_ARCHIVES:-/project2/irahbari_1147/places365}" "$root"
for variant in fp32 amp amp_channels_last; do
    echo "Starting $variant on $(hostname)"
    "$PY" -m torch.distributed.run --nnodes="$SLURM_NNODES" --node-rank="$SLURM_NODEID" \
        --nproc-per-node="$gpus" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
        src/profile_task9.py --variant "$variant" --root "$root" --out "$out"
    # A fresh store address prevents the next torchrun from meeting the previous one.
    MASTER_PORT=$((MASTER_PORT + 1))
done
