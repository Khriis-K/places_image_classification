# Places365 Scene Classification — TAC 450 Capstone

Scene recognition on Places365-Standard (365 classes, 1.8M images) with PyTorch, trained on USC CARC Discovery.

## Layout

| Path | Contents |
|---|---|
| `src/` | Python: data loading, models, training, evaluation |
| `slurm/` | Slurm batch scripts for CARC |
| `scripts/` | One-off setup scripts (data download, env setup) |
| `docs/` | Assignment, Places paper, written answers per task |

The dataset, checkpoints, TensorBoard runs, and logs are **not** in git. They live on CARC (see below).

## Workflow

Code is edited locally and synced to CARC through git:

```bash
git push                                            # local
cd ~/places_image_classification && git pull        # CARC (OnDemand terminal or ssh)
```

Each teammate works on their own branch (e.g. `ckuizon/dev`) and merges into `main` through a pull request. Off campus, connect to the USC VPN before using CARC. OnDemand JupyterLab (https://ondemand.carc.usc.edu) is handy for terminals, TensorBoard and quick interactive checks.

Run all jobs under the course account: `--account=irahbari_1147`.

## Data on CARC

Run once from a JupyterLab terminal (or the transfer node, `hpc-transfer1.usc.edu`, if compute nodes can't reach the internet):

```bash
bash scripts/carc_download_data.sh /project2/<account>/places365
```

`myaccount` on CARC shows the project name to use for `<account>`.

## Training (single GPU)

From the repo root on CARC:

```bash
mkdir -p logs
RUN=r50_1gpu sbatch slurm/train_1gpu.sbatch
```

The job unpacks the dataset into node-local RAM, then runs `src/train.py`. Checkpoints and TensorBoard logs go to `/scratch1/$USER/runs/$RUN`. Resubmitting with the same `RUN` resumes from the last completed epoch; near the time limit the job requeues itself and resumes automatically. Extra arguments are passed through to `train.py` (e.g. `--epochs 1 --limit-train-batches 300` for a quick test).
