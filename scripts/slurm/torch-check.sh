#!/usr/bin/env bash
#SBATCH --mail-user=nitzan.alt@campus.technion.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=lift4d-torch-check
#SBATCH --partition=part-preempt-classC
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
# Tiny Torch kernel check on Turing; not the full Ampere extension check.
set -eo pipefail
cd "${LIFT4D_REPO:-${SLURM_SUBMIT_DIR}}"
source "${LIFT4D_CONDA_ROOT:-/home/nitzan.alt/miniforge3}/etc/profile.d/conda.sh"
conda activate lift4d
python - <<'PY'
import json, os
from pathlib import Path
import torch
assert os.environ.get('SLURM_JOB_ID')
assert torch.__version__.startswith('2.5.1') and torch.version.cuda == '12.1'
assert torch.cuda.is_available()
x = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8, 8)
y = x @ x.T
assert torch.isfinite(y).all()
assert torch.equal(y.cpu(), x.cpu() @ x.cpu().T)
torch.cuda.synchronize()
report = {'job_id': os.environ['SLURM_JOB_ID'], 'node': os.environ.get('SLURMD_NODENAME'),
          'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'check': '8x8 CUDA matrix multiplication matches CPU', 'status': 'passed',
          'scope': 'Torch only; Ampere-specific CUDA extensions are checked by check.sh separately.',
          'note': 'No model, inference or training.'}
Path('runs/setup/torch-gpu-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
