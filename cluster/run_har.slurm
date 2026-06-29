#!/bin/bash
#SBATCH --job-name=har_fourier
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/har_%j.out
#SBATCH --error=logs/har_%j.err

# ---- run the HW3 Fourier+Transformer robustness ablation on one GPU ----
set -e
echo "Job $SLURM_JOB_ID on $(hostname) | started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/.bashrc
conda activate subpop
echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results logs

python har_experiment.py \
    --data_dir "data/UCI HAR Dataset" \
    --out_dir  "results" \
    --configs  baseline aug aug_reg \
    --seeds    0 1 2 \
    --epochs   25 \
    --patience 6 \
    --batch_size 128

echo "Finished $(date)"
