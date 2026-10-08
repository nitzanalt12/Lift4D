#!/usr/bin/env bash
#SBATCH --job-name=lift4d-check
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -eo pipefail
cd "${LIFT4D_REPO:-${SLURM_SUBMIT_DIR}}"
source "${LIFT4D_CONDA_ROOT:-/home/nitzan.alt/miniforge3}/etc/profile.d/conda.sh"
conda activate lift4d
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
python scripts/check_environment.py "$@"
