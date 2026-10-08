#!/usr/bin/env bash
#SBATCH --mail-user=nitzan.alt@campus.technion.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=lift4d-install
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
set -eo pipefail
REPO_DIR="${LIFT4D_REPO:-${SLURM_SUBMIT_DIR}}"
cd "$REPO_DIR"
source "${LIFT4D_CONDA_ROOT:-/home/nitzan.alt/miniforge3}/etc/profile.d/conda.sh"
conda activate lift4d
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export FORCE_CUDA=1
export GIT_TERMINAL_PROMPT=0
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
export PIP_CONSTRAINT="$REPO_DIR/scripts/requirements-constraints.txt"
export PIP_RETRIES=1
export PIP_TIMEOUT=30
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
mkdir -p runs/setup
exec > >(tee -a runs/setup/install.log) 2>&1
printf 'INSTALL job=%s node=%s\n' "$SLURM_JOB_ID" "$(hostname)"
if [ "${LIFT4D_INSTALL_CPU_ONLY:-0}" != 1 ]; then nvidia-smi; fi
python -m pip install 'torch==2.5.1' 'torchvision==0.20.1' 'torchaudio==2.5.1' --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
python -m pip install 'numpy==1.26.4' hatchling hatch-requirements-txt ninja editables setuptools wheel packaging
python -m pip install -e ./sam3d --no-build-isolation
python -m pip install -e './sam3d[p3d]' --no-build-isolation
python -m pip install -r sam3d/requirements.inference.txt --no-build-isolation
# Current utils3d main removes an API used by the original baseline.
# Restore the revision already pinned by the original MoGe dependency.
python -m pip install --no-deps --no-build-isolation "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
python scripts/install_utils3d_compat.py
python sam3d/patching/hydra
python -m pip install -r lift4d_scgs/requirements.txt
python -m pip install ./lift4d_scgs/submodules/diff-gaussian-rasterization ./lift4d_scgs/submodules/simple-knn --no-build-isolation
# Notebook inference imports these even in non-notebook usage.
python -m pip install 'gradio<6' ipywidgets 'huggingface-hub[cli]<1.0' 'numpy==1.26.4'
python scripts/fix_decord_wheel.py
python scripts/clean_pip_config.py
python -m pip freeze > runs/setup/environment.freeze.txt
python -m pip check
if [ "${LIFT4D_INSTALL_CPU_ONLY:-0}" = 1 ]; then
    python scripts/check_environment.py --imports-only
else
    python scripts/check_environment.py
fi
printf 'INSTALL_DONE\n'
