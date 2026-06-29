#!/bin/bash

################################################################################################
### HAR — Transformer + Fourier: full modeling run.
###
### GPU run. The deep models (Main/Patch/CNN/Fourier transformers + ablations)
### train much faster on a GPU; the two sklearn baselines run on CPU.
###
### Submit from inside the "transformer + fouria" folder:
###     cd "transformer + fouria" && sbatch run_slurm.sbatch
### (the SBATCH --output path below is relative to the submit directory).
################################################################################################

#SBATCH --partition main
#SBATCH --gpus=rtx_4090:1
#SBATCH --time 0-2:00:00
#SBATCH --job-name har-tf-fourier
#SBATCH --output outputs/har-tf-fourier-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=potashda@post.bgu.ac.il
#SBATCH --mail-type=ALL

################  Following lines will be executed by a compute node    #######################

set -uo pipefail

module load anaconda
# module load cuda          # uncomment if your env needs a system CUDA module

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
# Use an env that has torch + scikit-learn (see README "Setup"). Change the name
# if you created a dedicated one, e.g. `conda activate har_env`.
conda activate daniel_env

PY="$(which python)"

# Cap BLAS thread pools to the allocation (the sklearn baselines use BLAS/OpenMP).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs

echo "=========================================================="
echo " host       : $(hostname)"
echo " conda env  : ${CONDA_DEFAULT_ENV:-<unset>}"
echo " which py   : $PY"
$PY -c "import sys; print(' python ver :', sys.version.split()[0])"
$PY -c "import torch; print(' torch      :', torch.__version__, '| cuda:', torch.cuda.is_available())"
echo " cpus       : ${SLURM_CPUS_PER_TASK:-?}"
echo " pwd        : $(pwd)"
echo " started    : $(date)"
echo "=========================================================="

# Full run. Add --fast for a quick sanity check, or --data /path if the dataset
# is not under ./data or ../Data.
$PY -u har_full_run.py --num-workers "${SLURM_CPUS_PER_TASK:-2}"

echo
echo " finished : $(date)"
echo "=========================================================="
