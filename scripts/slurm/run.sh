#!/usr/bin/env bash
#SBATCH --mail-user=nitzan.alt@campus.technion.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=lift4d-baseline
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
# Explicit future execution only; do not submit for environment validation.
set -eo pipefail
cd "${LIFT4D_REPO:-${SLURM_SUBMIT_DIR}}"
if [ "$#" -ne 1 ]; then
    echo 'Usage: sbatch [SLURM options] scripts/slurm/run.sh runs/<experiment>/<run-id>' >&2
    exit 2
fi
source "${LIFT4D_CONDA_ROOT:-/home/nitzan.alt/miniforge3}/etc/profile.d/conda.sh"
conda activate lift4d
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
RUN_DIR="$(realpath "$1")"
test -f "$RUN_DIR/run.sh"
# Save scheduler allocation with the run, rather than only in a global log.
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/slurm-job.txt"
python - "$RUN_DIR/gpu-allocation.json" <<'PY'
import json
import os
from pathlib import Path
import sys
import torch

count = torch.cuda.device_count()
report = {'job_id': os.environ.get('SLURM_JOB_ID'),
          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'devices': [torch.cuda.get_device_name(i) for i in range(count)]}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
if count != 1:
    raise SystemExit('Expected one visible CUDA GPU; refusing to start the baseline')
PY
bash "$RUN_DIR/run.sh"
