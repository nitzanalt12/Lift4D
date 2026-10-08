#!/usr/bin/env bash
#SBATCH --job-name=lift4d-eval-export
#SBATCH --account=acct-ykasten
#SBATCH --partition=part-preempt-classB
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /home/nitzan.alt/miniforge3/etc/profile.d/conda.sh
conda activate lift4d
set -u
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python -u -m evaluation.export_input_views --run runs/baseline/animals-camel-20261008T120751Z --sequence camel
python -u -m evaluation.export_input_views --run runs/baseline/animals-rhino-20261008T120751Z --sequence rhino
