# Teammate Guide: CARC, Slurm, Git branches, and adding your own model

This is the practical "how do I work on this project" guide. It covers getting set up on USC CARC Discovery, running jobs with Slurm, working on your own Git branch, and training a model of your own next to the existing ResNet-50 baseline.

Replace `<netid>` with your USC NetID and `<you>` with your short name (e.g. `jdoe`) throughout.

---

## 1. How the project is organized

| Path | What it is |
|---|---|
| `src/data.py` | Dataset, train/val split (stratified 5% hold-out), augmentations, DataLoaders |
| `src/train.py` | Training script (ResNet-50 fine-tune). Checkpoints, resume, TensorBoard, AMP |
| `src/evaluate.py` | Test-set predictions from a checkpoint (single-crop or 10-crop) → `.npz` |
| `src/metrics.py`, `src/plot_*.py` | Confusion matrix, per-class accuracy, figures for the report |
| `slurm/*.sbatch`, `slurm/lr_sweep.sh` | Batch jobs for CARC |
| `scripts/extract_data.sh` | Unpacks the dataset tars to fast storage at the start of each job |
| `docs/taskN_*.md` | Written answers for each task, with figures in `docs/figures/` |
| `tests/` | `pytest` tests for metrics and evaluation |

**Not in Git:** the dataset, checkpoints (`*.pt`), TensorBoard runs, and Slurm logs. They stay on CARC.

**Current baseline:** ImageNet-pretrained ResNet-50, 10 epochs, lr 0.04 per 256 images, warmup + cosine schedule, label smoothing 0.1, fp16 AMP. Results are in `docs/task6_evaluation.md`.

---

## 2. One-time CARC setup

### 2.1 Access

- Off campus, connect to the **USC VPN** first.
- Easiest way in: **CARC OnDemand** at https://ondemand.carc.usc.edu → *Clusters → Discovery Shell* for a terminal, or *Interactive Apps → JupyterLab* for notebooks, terminals and TensorBoard.
- Or ssh: `ssh <netid>@discovery.usc.edu`.

### 2.2 Use the course account, every time

Our allocation is **`irahbari_1147`**. Your *default* Slurm account may be a different group (run `myaccount` to see yours). If you leave the account out, the job is charged to that other group or is rejected.

- In sbatch scripts: `#SBATCH --account=irahbari_1147` (already set in all scripts in `slurm/`).
- On the command line: `salloc --account=irahbari_1147 ...`
- In OnDemand forms: choose `irahbari_1147` in the account box.

### 2.3 Storage: where things go

| Location | Use it for | Notes |
|---|---|---|
| `/home1/<netid>` | Code clone, conda env | Small quota. Backed up |
| `/project2/irahbari_1147/places365` | **Shared dataset tars** (already downloaded) | Read only for us in practice. 6M-file limit, so **never extract the dataset here** |
| `/scratch1/<netid>` | Your checkpoints and TensorBoard runs | Large, not backed up |
| `$TMPDIR` (inside a job) | Node-local RAM copy of the dataset | Fast. Deleted when the job ends |

Check that you can read the dataset:

```bash
ls -lh /project2/irahbari_1147/places365
```

You should see `train_256_places365standard.tar`, `val_256.tar`, and `filelist_places365-standard.tar`. If you get "Permission denied", tell Chris; we may need to fix group permissions.

**Why the dataset is copied to `$TMPDIR`:** reading 1.8M small JPEGs from `/scratch1` costs ~22 ms per file open, which caps loading at ~400 img/s. Unpacking the tar into node-local RAM at job start takes ~70 s and gives ~2,200 img/s. The sbatch scripts do this for you. Because `$TMPDIR` is RAM, it counts against the job's memory, so jobs request `--mem=64G` (32G jobs hang).

### 2.4 Python environment

You need your own conda env (you can't use someone else's home directory env). `/apps/conda` is read-only, so point conda at your own directories first:

```bash
module purge
module load conda
conda config --add envs_dirs ~/.conda/envs
conda config --add pkgs_dirs /scratch1/$USER/.conda/pkgs

conda create -y -n torch-env python=3.11
conda activate torch-env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tensorboard matplotlib numpy pillow pytest
```

The sbatch scripts use `$HOME/.conda/envs/torch-env/bin/python` by default. If you name your env differently, pass it in: `PY=/path/to/python sbatch ...`.

Quick GPU check from an interactive session (section 3.3):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2.5 Clone the repo on CARC

```bash
cd ~
git clone https://github.com/Khriis-K/places_image_classification.git
cd places_image_classification
```

The repo is public, so cloning and pulling need no login. **Pushing** does: see section 4.

---

## 3. Slurm basics

Slurm is the job scheduler. You don't run training on the login node. You either submit a **batch job** (`sbatch`) that runs on its own, or ask for an **interactive session** (`salloc`) to poke around on a GPU node.

### 3.1 Commands you'll actually use

| Command | What it does |
|---|---|
| `sbatch slurm/train_1gpu.sbatch` | Submit a batch job. Prints `Submitted batch job <JOBID>` |
| `squeue --me` | Your pending/running jobs. `ST`: `PD` = pending, `R` = running |
| `scancel <JOBID>` | Cancel a job |
| `sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode` | What happened to a finished job |
| `tail -f logs/places-1gpu-<JOBID>.out` | Watch a job's output live |
| `myaccount` | Your accounts and remaining allocation |
| `sinfo -p gpu` | GPU partition node states |

### 3.2 Anatomy of our sbatch script

From `slurm/train_1gpu.sbatch`:

```bash
#SBATCH --account=irahbari_1147   # charge the course allocation
#SBATCH --partition=gpu           # GPU nodes
#SBATCH --gres=gpu:v100:1         # one V100 (32 GB). fp16 AMP works; bf16 does not on V100
#SBATCH --cpus-per-task=8         # DataLoader workers
#SBATCH --mem=64G                 # includes the RAM copy of the dataset
#SBATCH --time=06:00:00           # wall-clock limit
#SBATCH --output=logs/%x-%j.out   # logs/<job name>-<job id>.out
#SBATCH --signal=B:USR1@300       # warn the script 5 min before the time limit...
#SBATCH --requeue                 # ...so it can requeue itself and resume
```

The script then unpacks the dataset to `$TMPDIR`, and runs `src/train.py` with any extra arguments you put after the script name.

`--output=logs/...` means **the `logs/` directory must exist before you submit**, or the job fails instantly with no output. Always `mkdir -p logs` first.

### 3.3 Interactive GPU session (for debugging)

```bash
salloc --account=irahbari_1147 --partition=gpu --gres=gpu:v100:1 \
       --cpus-per-task=8 --mem=64G --time=01:00:00
# once you get the node:
conda activate torch-env
nvidia-smi
```

Interactive sessions only get the CPUs you asked for, but PyTorch sees all the node's CPUs. Our code already handles this via `available_cpus()` in `src/data.py`; do the same (`len(os.sched_getaffinity(0))`) in any new script.

Alternatively, use the OnDemand JupyterLab app with a GPU, account `irahbari_1147`.

---

## 4. Git: working on your own branch

### 4.1 The rules

- **Never commit straight to `main`.** Each person works on their own branch and merges through a pull request (PR) on GitHub.
- Branch name: `<you>/dev` for general work, or `<you>/<topic>` for one feature (e.g. `jdoe/efficientnet`).
- Edit code on your laptop, push to GitHub, then `git pull` on CARC. Editing on CARC directly works too, but then you need push credentials there.

### 4.2 Get push access

Ask Chris to add your GitHub account as a **collaborator** on `Khriis-K/places_image_classification`. Accept the email invite.

To push from your laptop, signing in once with GitHub Desktop, VS Code, or `gh auth login` is the easiest. To push from CARC, use a GitHub [personal access token](https://github.com/settings/tokens) as the password, or add an ssh key from CARC to your GitHub account.

### 4.3 Create your branch

> **Note:** right now most of the project code lives on `ckuizon/dev` and hasn't been merged into `main` yet. Until it is, branch off `ckuizon/dev`. Once it's merged, branch off `main` instead.

On your laptop:

```bash
git clone https://github.com/Khriis-K/places_image_classification.git
cd places_image_classification
git fetch origin
git switch -c <you>/dev origin/ckuizon/dev   # later: origin/main
git push -u origin <you>/dev
```

On CARC:

```bash
cd ~/places_image_classification
git fetch origin
git switch <you>/dev
```

### 4.4 Daily loop

```bash
# laptop
git add src/my_change.py
git commit -m "Add EfficientNet-B0 option to build_model"
git push

# CARC
cd ~/places_image_classification && git pull
mkdir -p logs && RUN=<you>_effb0 sbatch slurm/train_1gpu.sbatch --arch efficientnet_b0
```

Things that should **not** be committed (already in `.gitignore`): `*.pt` checkpoints, `logs/`, `runs/`, dataset files, course PDFs. Don't `git add -f` around that.

### 4.5 Staying up to date and merging

Pull in other people's work regularly so you don't drift:

```bash
git fetch origin
git merge origin/main
```

When your work is ready, open a PR on GitHub from `<you>/dev` into `main`, describe what changed and what results it produced, and ask a teammate to review before merging.

---

## 5. Training runs

### 5.1 Quick smoke test first

Before a multi-hour run, make sure everything works end to end (finishes in a few minutes):

```bash
mkdir -p logs
RUN=<you>_smoke sbatch --time=00:30:00 slurm/train_1gpu.sbatch \
    --epochs 1 --limit-train-batches 300 --limit-val-batches 20
```

### 5.2 Full run

```bash
RUN=<you>_r50 sbatch slurm/train_1gpu.sbatch
```

- `RUN` names the output directory: `/scratch1/$USER/runs/$RUN/` containing `last.pt`, `best.pt`, and `tensorboard/`.
- **Resume:** resubmitting with the same `RUN` picks up from `last.pt`. Near the 6 h limit the job requeues itself automatically.
- **Use a new `RUN` name for every experiment**, or it will resume the old one instead of starting fresh.
- Any `train.py` argument can go after the script name: `--epochs`, `--lr`, `--batch-size`, `--label-smoothing`, `--no-pretrained`, `--no-amp`, etc. See `python src/train.py --help`.

### 5.3 Learning-rate sweep

```bash
bash slurm/lr_sweep.sh 0.01 0.02 0.04
```

This queues one 3-epoch job per learning rate under `/scratch1/$USER/runs/lr_sweep/`.

### 5.4 Watching progress in TensorBoard

Logs are small (~27 MB for all our runs), so the reliable way is to copy them to your laptop:

```bash
# CARC: bundle every run's TensorBoard event files (no checkpoints)
cd /scratch1/$USER/runs && find . -path '*/tensorboard/*' -type f | tar czf ~/tb_logs.tgz -T -
```

Download `~/tb_logs.tgz` from the OnDemand file browser, unpack it locally into `runs/carc/`, run `tensorboard --logdir runs/carc`, and open http://localhost:6006. The *Custom Scalars* tab shows train vs. val on the same chart; set Smoothing to 0 for per-epoch charts. Details in `docs/task5_visualization.md`.

### 5.5 Evaluating a model on the test set

`src/evaluate.py` runs a checkpoint on the official Places365 validation set (36,500 images, which we use as the test set) and writes `<name>.npz` for the plotting scripts:

```bash
python src/evaluate.py --root $TMPDIR/places365 \
    --checkpoint /scratch1/$USER/runs/<you>_r50/best.pt \
    --out logs/eval --name <you>_r50 [--ten-crop]
```

`slurm/evaluate.sbatch` is the version we used for Task 6. Copy it and change the `fetch`/`evaluate` lines at the bottom for your own runs. Then make figures with `src/plot_evaluation.py`.

---

## 6. Writing your own model

Right now the model is hardcoded as ResNet-50 in `build_model()` in `src/train.py`, and `src/evaluate.py` imports that same function to rebuild the model from a checkpoint. The cleanest way to add your own architecture is to turn that into a choice instead of making a copy of `train.py`, so you keep the data pipeline, AMP, checkpoint/resume, TensorBoard and evaluation for free.

### 6.1 Recommended pattern

1. **Add an `--arch` option** in `parse_args()` in `src/train.py`:

   ```python
   p.add_argument("--arch", default="resnet50", help="model name, see build_model()")
   ```

2. **Make `build_model` dispatch on it.** Each architecture needs its classifier head swapped for a 365-class layer:

   ```python
   def build_model(pretrained, num_classes=365, arch="resnet50"):
       if arch == "resnet50":
           model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
           model.fc = nn.Linear(model.fc.in_features, num_classes)
       elif arch == "efficientnet_b0":
           model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
           model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
       elif arch == "my_cnn":
           from my_model import MyCNN          # src/my_model.py, your own nn.Module
           model = MyCNN(num_classes=num_classes)
       else:
           raise ValueError(f"unknown arch {arch!r}")
       return model
   ```

   And pass it through where the model is built: `build_model(args.pretrained, arch=args.arch)`.

3. **Make evaluation rebuild the right model.** `train.py` already saves `"args": vars(args)` inside every checkpoint, so in `src/evaluate.py` use:

   ```python
   arch = ckpt.get("args", {}).get("arch", "resnet50")   # old checkpoints default to ResNet-50
   model = build_model(pretrained=False, num_classes=len(class_names), arch=arch)
   ```

4. **Write a from-scratch model** (if that's your plan) in its own file, e.g. `src/my_model.py`:

   ```python
   import torch.nn as nn

   class MyCNN(nn.Module):
       def __init__(self, num_classes=365):
           super().__init__()
           self.features = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
               nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
               nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
               nn.AdaptiveAvgPool2d(1),
           )
           self.classifier = nn.Linear(256, num_classes)

       def forward(self, x):          # x: (B, 3, 224, 224), ImageNet-normalized
           return self.classifier(self.features(x).flatten(1))
   ```

   Inputs are 224×224 crops normalized with ImageNet mean/std (see `build_transforms` in `src/data.py`). The model must return raw logits of shape `(batch, 365)`; the loss applies softmax.

5. **Test on your laptop / CPU first** that the shapes work, before spending GPU time:

   ```python
   import torch, sys; sys.path.insert(0, "src")
   from train import build_model
   m = build_model(pretrained=False, arch="my_cnn")
   print(m(torch.randn(2, 3, 224, 224)).shape)   # torch.Size([2, 365])
   ```

6. **Smoke test on CARC** (section 5.1) with `--arch my_cnn`, then do a full run.

### 6.2 Things to keep in mind

- **Learning rate.** `--lr 0.04` was tuned for fine-tuning a *pretrained* ResNet-50. A from-scratch model or a different architecture usually needs its own short sweep. `slurm/lr_sweep.sh` only takes learning rates, so copy it and add `--arch <name>` to its `sbatch` line (and change the `lr_sweep/` run prefix so you don't collide with existing runs).
- **Memory.** A bigger model may not fit `--batch-size 256` on a 32 GB V100. Drop the batch size; the learning rate is scaled linearly with the batch automatically.
- **Time.** One ResNet-50 epoch on 1 V100 takes ~26 minutes; a 10-epoch run is ~4.5 h, and a 3-epoch sweep run ~1.3 GPU-hours. The job requeues itself at the 6 h limit, so longer runs are fine, but GPU hours come out of the shared course allocation, so smoke-test first.
- **Comparisons.** For a fair comparison with the baseline, keep the data split, augmentation, epochs and evaluation the same, and only change the model (and its learning rate).
- Run `pytest tests/` before opening a PR.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Job fails instantly, no log file | `logs/` doesn't exist. `mkdir -p logs` in the repo root |
| `Invalid account or account/partition combination` | Missing `--account=irahbari_1147` |
| Job stuck in `PD` for a long time | GPU queue is busy. `squeue --me --start` shows an estimated start time |
| Job hangs or is killed during data extraction | Not enough memory: the RAM dataset copy needs `--mem=64G` |
| Training very slow (~400 img/s) | Reading from `/scratch1` instead of `$TMPDIR`. Use the sbatch script, which extracts to `$TMPDIR` |
| `CUDA out of memory` | Lower `--batch-size` |
| A new run "resumed" from epoch 7 | You reused a `RUN` name. Pick a new one or delete the old run dir |
| `errno 108` / I/O error on `/scratch1` | Occasional node glitch. Resubmit, or exclude the bad node: `sbatch --exclude=<node> ...` |
| `ModuleNotFoundError: torch` in the job | Job used the wrong Python. Pass `PY=$HOME/.conda/envs/<env>/bin/python` |
| `git push` asks for a password on CARC | Use a GitHub personal access token, or push from your laptop instead |

## 8. Useful links

- CARC user guides (Slurm, storage, conda): https://www.carc.usc.edu/user-guides
- CARC OnDemand: https://ondemand.carc.usc.edu
- torchvision model zoo (pretrained weights): https://pytorch.org/vision/stable/models.html
- Places365 dataset: http://places2.csail.mit.edu
