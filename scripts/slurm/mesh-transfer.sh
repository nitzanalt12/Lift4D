#!/bin/bash
#SBATCH --job-name=lift4d-mesh-transfer
#SBATCH --mail-user=nitzan.alt@campus.technion.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --account=acct-ykasten
#SBATCH --partition=part-preempt-classB
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/nitzan.alt/Lift4D}"
source /home/nitzan.alt/miniforge3/etc/profile.d/conda.sh
conda activate lift4d
set -u
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST='8.0;8.6'
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
campaign="$1"
shift
mkdir -p "$campaign/logs"
python -c 'import torch; assert torch.cuda.device_count()==1, "No allocated CUDA GPU"; print(torch.cuda.get_device_name(0))'
for sequence in "$@"; do
    case "$sequence" in
        camel|rhino) stamp=20261008T120751Z ;;
        flamingo|cows) stamp=20261008T120844Z ;;
        *) echo "Unknown sequence: $sequence" >&2; exit 2 ;;
    esac
    for variant in 2.1 2.2; do
        python -u -m experiment2.transfer \
            --run "runs/baseline/animals-${sequence}-${stamp}" \
            --sequence "$sequence" --variant "$variant" \
            --output "$campaign/${sequence}-${variant}" \
            > "$campaign/logs/${sequence}-${variant}.log" 2>&1
    done
done
