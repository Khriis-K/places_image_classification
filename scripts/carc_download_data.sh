#!/bin/bash
# Download Places365-Standard (256x256) into a shared CARC directory.
# Run on the transfer node: ssh <netid>@hpc-transfer1.usc.edu
# Usage: bash scripts/carc_download_data.sh /project2/<account>/places365
set -euo pipefail

DEST="${1:?usage: $0 <destination dir>}"
BASE_URL="https://data.csail.mit.edu/places/places365"
mkdir -p "$DEST"
cd "$DEST"

# -c resumes a partial download if the connection drops
wget -c "$BASE_URL/train_256_places365standard.tar"   # ~26 GB, 1,803,460 images
wget -c "$BASE_URL/val_256.tar"                       # 36,500 images (100/class), our test set
wget -c "$BASE_URL/filelist_places365-standard.tar"   # class names + val labels

# Checksums from http://places2.csail.mit.edu/download-private.html
md5sum -c <<'MD5'
53ca1c756c3d1e7809517cc47c5561c5  train_256_places365standard.tar
e27b17d8d44f4af9a78502beb927f808  val_256.tar
MD5

tar -xf filelist_places365-standard.tar

# Keep the big tars packed; jobs extract them to node-local storage at start.
ls -lh "$DEST"
