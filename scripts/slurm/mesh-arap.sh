#!/bin/bash
#SBATCH --job-name=lift4d-mesh-arap
#SBATCH --mail-user=nitzan.alt@campus.technion.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --account=acct-ykasten
#SBATCH --partition=part-preempt-classB
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/nitzan.alt/Lift4D}"
source /home/nitzan.alt/miniforge3/etc/profile.d/conda.sh
conda activate lift4d
set -u
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST='8.0;8.6'
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -c 'import torch; assert torch.cuda.device_count()==1, "No allocated CUDA GPU"; print(torch.cuda.get_device_name(0))'
output="$1"
shift
python -u -m experiment2.project_arap --seed-run runs/sam3d_mesh_transfer/animals-20261008-v1/camel-2.1 --output-root "$output" "$@"
