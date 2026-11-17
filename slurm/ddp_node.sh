#!/bin/bash
# Per-node part of slurm/train_ddp.sbatch (srun runs it once on every node). Not meant to be run by hand.
set -euo pipefail

echo "[$(hostname) node $SLURM_NODEID] $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -), $(nproc) CPUs"
# Every node needs its own copy of the dataset in node-local memory
bash scripts/extract_data.sh /project2/irahbari_1147/places365 "$TMPDIR/places365" > "$TMPDIR/extract.log" 2>&1 \
    || { cat "$TMPDIR/extract.log"; exit 1; }
echo "[$(hostname) node $SLURM_NODEID] dataset ready at $(date +%T) in $TMPDIR ($(df -h --output=fstype,size "$TMPDIR" | tail -n 1 | xargs))"

GPUS=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
# Static rendezvous: node 0 hosts it. Nodes that finish unpacking early wait for the others.
exec "$PY" -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" --node-rank="$SLURM_NODEID" --nproc-per-node="$GPUS" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    src/train.py --root "$TMPDIR/places365" --out "$OUT" "$@"
