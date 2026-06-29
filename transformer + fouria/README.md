# HAR — Transformer + Fourier (SLURM full run)

Self-contained, headless version of `hw_fourier_trans/HAR_Transformer_Deep_Colab_v2.ipynb`.
One script (`har_full_run.py`) runs the **complete** modeling cycle and writes every
result (metrics + plots) to `outputs/`.

## What it runs
| # | Model | Input |
|---|-------|-------|
| 1 | Logistic Regression | **6 selected features** (top-6 of 561 by mutual information) |
| 2 | Random Forest | **same 6 selected features** |
| 3 | Main Transformer (CLS, sinusoidal PE) | raw 128×9 windows |
| 4 | Ablation study (7 configs, validation-only) | raw 128×9 |
| 5 | Patch Transformer (patch size 8) | raw 128×9 |
| 6 | CNN-Transformer hybrid | raw 128×9 |
| 7 | Sensor ablation (channel groups) | channel subsets |
| 8 | **Fourier+Transformer** (dual-branch: time + FFT self-attention) | raw 128×9 |

Plus analysis: confusion matrices, per-subject accuracy, calibration (ECE), and a
final comparison leaderboard.

All deep models use a **subject-wise train/validation split** (no window leakage),
and are evaluated on the official UCI subject-disjoint test set.

## Files
- `har_full_run.py` — the full pipeline (run this)
- `requirements.txt` — Python dependencies
- `run_slurm.sbatch` — SLURM batch script (edit partition/modules for your cluster)
- `outputs/` — created at run time (metrics.json, *.csv, *.png, slurm logs)

## Setup (login node — needs internet)
The BGU cluster uses Anaconda (see `runbaseline.sbatch`). Either reuse an existing
env that has PyTorch, or create a dedicated one:
```bash
module load anaconda
conda create -n har_env python=3.10 -y
conda activate har_env
cd "transformer + fouria"
pip install -r requirements.txt
# For a CUDA node, install the matching torch wheel, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Then set the env name in `run_slurm.sbatch` (`conda activate har_env`).

Local / non-cluster alternative (venv):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data
The script auto-detects the UCI HAR Dataset in this order:
1. `--data /path/to/'UCI HAR Dataset'`
2. `$HAR_DATA`
3. `./data/UCI HAR Dataset/`
4. `../Data/human+activity+recognition+using+smartphones/` (the repo copy)

Easiest on the cluster: copy the dataset into `./data/` so the path is
`./data/UCI HAR Dataset/{train,test,activity_labels.txt,...}`.

## Run
```bash
# quick sanity check (2 epochs/model, CPU ok)
python har_full_run.py --fast

# full run
python har_full_run.py

# on SLURM
sbatch run_slurm.sbatch
```
Useful flags: `--fast`, `--data`, `--out`, `--device cuda|cpu`, `--num-workers N`, `--seed N`.

## Outputs
- `outputs/metrics.json` — every number (reports, ECE, ablation, sensor ablation, leaderboard)
- `outputs/final_comparison.csv` — model leaderboard by test macro-F1
- `outputs/per_subject_main.csv` — per-subject accuracy (main transformer)
- `outputs/*.png` — training curves, confusion matrices, ablation/comparison bar charts

## ⚠️ Note on the Fourier+Transformer
The original notebook **referenced but did not contain** the `FourierTransformer`
definition (it was run on a stale kernel and the defining cell was later deleted).
The `FourierTransformer` here is a **faithful reconstruction** matching the notebook's
description — a dual-branch model with one Transformer encoder over the raw
time-domain signal and one over the per-channel FFT log-magnitude spectrum, fused
before the classifier. Its exact numbers will therefore differ from the old saved
plots. If the original definition is recovered, drop it in place of the
`FourierTransformer` class in `har_full_run.py`.
