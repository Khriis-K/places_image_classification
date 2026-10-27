# Task 2 — Data Preprocessing

All numbers below come from `src/explore.py`, run on the full extracted dataset on CARC Discovery. The pipeline itself is in `src/data.py`.

## 1. Storage layout

The three official archives (training images, validation images, file lists; ~26 GB total) are stored once in the course's shared project space (`/project2/irahbari_1147/places365`) and verified against the published MD5 checksums. At the start of each job, `scripts/extract_data.sh` unpacks them into node-local RAM storage (`$TMPDIR`), which is where training reads from (§5 explains why not `/scratch1`). Keeping the archives packed in project storage avoids putting 1.8 M small files on the shared filesystem, which has a file-count quota.

Labels come from the official file lists (`categories_places365.txt`, `places365_train_standard.txt`, `places365_val.txt`), not from scanning directories. This guarantees our class indices match the published Places365 labels (0–364).

## 2. Dataset exploration

| Check | Result |
|---|---|
| Image size | 1,000 of 1,000 sampled images are exactly **256×256** |
| Color mode | 1,000 of 1,000 are **RGB** (no grayscale or CMYK to convert) |
| Classes | 365 |
| Class balance (full training set) | 3,068–5,000 images per class; 339 classes have exactly 5,000 |
| Per-channel mean (RGB, 5,000 images) | **0.458, 0.439, 0.403** |
| Per-channel std (RGB, 5,000 images) | **0.269, 0.266, 0.284** |
| ImageNet mean / std, for comparison | 0.485, 0.456, 0.406 / 0.229, 0.224, 0.225 |

Every image is already the same size and color mode, so no cleaning or format conversion is needed. The class imbalance is mild: the largest class is 1.6× the smallest and 93% of classes are full. We therefore don't resample or weight the loss, but we will check per-class accuracy in Task 6.

Places365 images are slightly darker and have a little more contrast than ImageNet's (means about 0.02–0.03 lower, standard deviations about 0.04–0.06 higher).

Figures (in `runs/explore/`, also logged to TensorBoard under `dataset/`):
- `class_counts.png`: images per class, sorted
- `samples.png`: one random image from each of 32 random classes, labeled
- `augmentations.png`: one image and seven random augmentations of it

## 3. Train / validation / test split

The official test labels are not public, so we build our own three-way split:

| Split | Source | Images | Per class |
|---|---|---|---|
| **Train** | 95% of official training set | 1,713,290 | 2,915–4,750 |
| **Validation** | 5% of official training set | 90,170 | 153–250 (mean 247) |
| **Test** | Official validation set (`val_256`) | 36,500 | exactly 100 |

The train/validation split is **stratified**: we hold out 5% of each class separately, so both splits keep the original class proportions. It is also **reproducible**: it uses a fixed random seed (0), so every run and every teammate gets the identical split. We chose 95/5 instead of the common 80/20 because of the dataset's size: 5% still gives ~90 k validation images, which estimates accuracy to about ±0.3 percentage points, while keeping ~1.7 M images for training and keeping per-epoch validation fast. For comparison, ImageNet's validation set is ~4% of its training set. The test set is never used for training or tuning, only for final evaluation in Task 6.

(The Places paper describes the validation set as 50 images per class, but the official archive and its label file contain 100 per class, so our test set has 36,500 images.)

## 4. Transforms

**Training (with augmentation):**
1. `RandomResizedCrop(224, scale=(0.35, 1.0))`: crop a random region covering 35–100% of the image, then resize to 224×224. This teaches the model to recognize a scene from partial views and different framings. We use a larger minimum scale than ImageNet's default of 0.08, because scene categories depend on overall context; a 8% crop of a "kitchen" may show only a cabinet.
2. `RandomHorizontalFlip()`: a mirrored scene is still the same scene.
3. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: robustness to lighting and camera differences. We don't change hue, since color is informative for scenes (sky, grass, water).
4. `ToTensor()`: converts pixels to floats in [0, 1].
5. `Normalize(mean, std)`: standardizes each channel to roughly zero mean and unit variance.

**Validation / test (deterministic):** `Resize(256)` (a no-op here, since images are already 256×256) → `CenterCrop(224)` → `ToTensor()` → `Normalize(mean, std)`.

Augmentation is applied on the fly by DataLoader workers, so each epoch sees a different random version of every training image, and nothing extra is stored on disk.

**Normalization statistics:** we normalize with the **ImageNet** mean/std. Our model (Task 3) starts from ImageNet-pretrained weights, which expect inputs normalized exactly this way, and the measured Places365 statistics are close to ImageNet's anyway. If we train a model from scratch, we can switch to the measured Places365 values above (`mean`/`std` arguments in `build_datasets`).

## 5. Data loading

Batches are assembled by a PyTorch `DataLoader`:
- batch size 256
- 8 worker processes decoding JPEGs in parallel
- `pin_memory=True` for faster copies from CPU to GPU
- `persistent_workers=True`, so workers aren't restarted every epoch
- training set shuffled every epoch

**Baseline throughput:** reading from `/scratch1`, the DataLoader alone delivers only **~294–399 images/s** (8 workers, no model). At this rate, reading one epoch of 1.71 M images takes about 80–97 minutes, slower than a V100 can train.

**Bottleneck diagnosis** (`src/loader_bench.py`, 8 CPU cores):

| Measurement | `/scratch1` (network FS) | Node-local RAM (`$TMPDIR`) |
|---|---|---|
| Read one uncached file | 22 ms | 0.03 ms |
| Decode + augment, 1 core | 361 images/s | 361 images/s |
| CPU-bound ceiling, 8 cores | ~2,890 images/s | ~2,890 images/s |
| **DataLoader, 8 workers** | **399 images/s** | **2,190 images/s** |

The CPU work per image is the same in both cases. The difference is opening files: on the network filesystem every file access costs ~22 ms of latency, and each worker reads its files one after another, so 8 workers top out at ≈ 8 × 45 = 360–400 images/s no matter how fast the CPUs are.

**Fix:** each job copies the dataset into node-local RAM-backed storage (`$TMPDIR`, a tmpfs) when it starts, using `scripts/extract_data.sh`. Reading the 25 GB archive from project storage is one large sequential read, which a network filesystem handles well. After that, every image read is a memory access. Unpacking all 1,839,960 images takes **~69 seconds** per job (measured), which is paid back within the first minute of training. The copy counts against the job's memory request, so training jobs request 64 GB. Result: **5.5× higher loader throughput** (399 → 2,190 images/s), now limited by JPEG decoding and augmentation on the CPU rather than by I/O. We revisit this, and further CPU-side optimizations, in Task 9.

Other observation: `ColorJitter` is the most expensive transform. It accounts for ~1.1 ms of the 2.4 ms per-image augmentation cost (412 vs. 736 images/s per core with and without it).
