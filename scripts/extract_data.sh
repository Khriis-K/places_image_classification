#!/bin/bash
# Unpack the Places365 tars from shared project storage into a fast working directory.
# Usage: bash scripts/extract_data.sh [SRC] [DEST]
#   SRC  default /project2/irahbari_1147/places365   (tars from carc_download_data.sh)
#   DEST default /scratch1/$USER/places365           (or $TMPDIR inside a Slurm job)
# Result: DEST/data_256/<letter>/<class>/*.jpg, DEST/val_256/*.jpg, DEST/filelists/*.txt
set -euo pipefail

SRC="${1:-/project2/irahbari_1147/places365}"
DEST="${2:-/scratch1/$USER/places365}"
mkdir -p "$DEST/filelists"

# Skip anything already unpacked so re-running is cheap
if [ ! -d "$DEST/data_256" ]; then
    echo "Extracting train images to $DEST ..."
    time tar -xf "$SRC/train_256_places365standard.tar" -C "$DEST"
fi
if [ ! -d "$DEST/val_256" ]; then
    echo "Extracting val images to $DEST ..."
    tar -xf "$SRC/val_256.tar" -C "$DEST"
fi
if [ ! -f "$DEST/filelists/categories_places365.txt" ]; then
    tar -xf "$SRC/filelist_places365-standard.tar" -C "$DEST/filelists"
fi

echo "Train images: $(find "$DEST/data_256" -name '*.jpg' | wc -l) (expect 1803460)"
echo "Val images:   $(find "$DEST/val_256" -name '*.jpg' | wc -l) (expect 18250)"
ls "$DEST/filelists"
