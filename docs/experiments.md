# Minimal experiment infrastructure

The fork is `nitzanalt12/Lift4D`, with `origin` pointing to the fork and `upstream`
to `yehonathanlitman/Lift4D`. The infrastructure branch is
`infra/experiment-foundation`. The original commit is pinned in
`experiments/upstream.json`: `aa295d66d1baabc6f9954b619f8b1bdb557bd862`.
No original model, dataset loader, optimizer or inference code is changed.

## Installation

Run the original [installation instructions](../README.md#1-installation):
create the `lift4d` conda environment from `sam3d/environments/default.yml`,
install SAM3D (including p3d and Hydra patch), the SC-GS requirements and both
CUDA extensions. The upstream environment pins Python 3.11 and CUDA 12.1;
upstream reports testing on A100 40 GB. The experiment preparer uses only the
Python standard library, so environment installation is unnecessary for checks
and preparation. The initial infrastructure setup did not install the ML environment. The
installation and SLURM checks added later are described below; a passing
infrastructure inventory check alone does not imply inference readiness.

Weights are separate prerequisites. Follow upstream's gated model access and
weight instructions when you later intend to run. For this DAVIS subset masks
already exist, so SAM3 segmentation and SAM3 weights are unnecessary. SAM3D's
`pipeline.yaml` and its weights must remain together in the configured directory.
The Zero123 directory must contain both `stable_zero123.ckpt` and the upstream
`sd-objaverse-finetune-c_concat-256.yaml` (copy the tracked YAML alongside external
weights if needed). DA3 remains enabled exactly as upstream: its model identifier
is `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, and the original code uses the standard
Hugging Face cache. Configure `HF_HOME` before execution if you need an external
cache; the preparer neither loads nor downloads this model. Upstream dependencies
may also load cached models when actual training begins.

## Configuration and fixed input

From the repository root:

```bash
cp experiments/paths.example.json experiments/paths.local.json
# Edit paths.local.json; absolute paths or paths relative to the repository work.
python scripts/experiment.py check
python scripts/experiment.py prepare --run-id baseline-001
```

The existing local path file points to DAVIS already present at
`/home/nitzan.alt/DAVIS-data/DAVIS`. It is ignored by Git. `python` can be an
absolute path to an environment's Python executable. `runs_root` can be outside
the checkout; inside the checkout it must be ignored by Git.

`experiments/subsets/davis-v1.json` fixes rhino, 480p, all 90 frame stems and
matching PNG masks. Rhino follows the original README's DAVIS example and is a
minimal starting subset, not a decision about the final evaluation benchmark.
Checks reject missing/extra frames and missing masks. There is no random selection,
subsampling, mask conversion or data copying. The versioned stem list is shared
by all three experiment definitions; retain v1 and add a new manifest for future
subset changes. The snapshot records its SHA256. Image content is not hashed.

`experiments/configs/baseline.json` uses the original DAVIS SAM3D example and the
README's >50-frame geometry/appearance recipe, including DA3 and occlusion
compositing. All unspecified model settings retain upstream defaults. The existing
`_node` / `_node_delta` checkpoint lookup connects geometry to appearance without
copying or modifying checkpoints. This is not a newly tuned recipe or a promise
of bitwise deterministic training; the original random behavior remains intact.

The two other configs are independent descriptions starting from original LIFT4D,
not an inheritance chain of experiments. They reference the same baseline training
settings and subset but have `not_implemented` status. Preparation can snapshot
them; their generated script refuses to execute a pipeline:

```bash
python scripts/experiment.py prepare --config experiments/configs/sam3d_mesh.json
python scripts/experiment.py prepare --config experiments/configs/pixal3d.json
```

## Run records and later execution

Each preparation creates `runs/<experiment>/<run-id>/` exclusively and refuses
an existing run ID. It saves config, resolved paths, subset, frame inventory,
command arguments, environment overrides, branch/commit, upstream commit,
working-tree status and tracked patch, the preparer source, and preparation log.
Missing weights are reported, not downloaded. `check` exits successfully when the
configuration and DAVIS inventory are valid; missing weights are informational.
It does not verify full model checkpoints, CUDA, or dependency availability.

The generated `run.sh` is reviewable; preparing it does not execute it. Once the
ML environment and weights are ready, explicitly run it in the same checkout
and commit used during preparation:

```bash
conda activate lift4d
bash runs/baseline/baseline-001/run.sh
```

That command **does perform inference and training**, and was not run during setup.
It records `pip freeze`, merged stdout/stderr per stage, a pipeline log and exit
code. Products remain under that run: `sam3d/davis_rhino/`,
`models/davis_rhino_node/` and `models/davis_rhino_node_delta/`. A `.started`
directory refuses accidental execution twice; prepare a new run rather than reuse
partial results. Commit source changes before preparing scientific runs; the
recorded patch covers tracked edits, not arbitrary untracked source files. Paths,
weights and source must not change between preparation and execution.

Data, weights, local paths and outputs are ignored by Git. Do not force-add them.
Do not place external caches or datasets in new unignored checkout directories.

## Open experiment decisions

- Mesh transfer: canonical frame, mesh coordinate/scale alignment and which saved
  deformation components act on vertices. No transfer or extra training is implemented.
- Pixal3D: Gaussian representation/interface and coordinate, pose and frame
  correspondence. No adapter or change to baseline training is implemented.
- Final evaluation subset/metrics, any broader protocol, and eventual dependency
  locking are future decisions. All experiments must retain the same versioned
  subset and baseline settings when they are compared.

## SLURM installation and dependency checks

The isolated environment is `lift4d`; existing environments are left unchanged.
Create it with the upstream conda YAML before submitting installation:

```bash
conda env create -f sam3d/environments/default.yml
mkdir -p runs/setup
sbatch --output=runs/setup/install-%j.log --error=runs/setup/install-%j.err scripts/slurm/install.sh
sbatch --output=runs/setup/check-%j.log --error=runs/setup/check-%j.err scripts/slurm/check.sh
```

These jobs request one A100 through `part-preempt` / `qos-preempt`. The installer
compiles CUDA extensions for compute capabilities 8.0 and 8.6; this Torch/CUDA
version is not intended for the RTX PRO 6000 Blackwell partition. The installer
uses original requirements and installs notebook-import prerequisites; it never
executes the reconstruction or optimization pipeline. Check scripts execute only
tiny CUDA operations and imports, and SAM3D CLI help. Training CLI help is
intentionally skipped because its top-level imports instantiate LPIPS models.
Logs, the resolved `pip freeze` and `gpu-check.json` are under `runs/setup/`.
`LIFT4D_CONDA_ROOT` overrides the default local Miniforge root, and
`LIFT4D_REPO` overrides the submission directory. Resource flags can be overridden
with normal sbatch options. Preemptible jobs may need resubmission.

For an explicitly requested future baseline run, first prepare a fresh run in
the activated environment and submit the wrapper:

```bash
conda activate lift4d
python scripts/experiment.py prepare --run-id baseline-001
sbatch --output=runs/baseline/baseline-001/slurm-%j.log \
  --error=runs/baseline/baseline-001/slurm-%j.err \
  scripts/slurm/run.sh runs/baseline/baseline-001
```

This last command does inference and training. Installation/check jobs do not.
SAM3 segmentation is not installed: this fixed DAVIS subset already has masks.
Full SAM3D checkpoint download requires approval for the gated Hugging Face
model; no license or access request is accepted automatically on your behalf.

The local installation can compile on the available CPU allocation while the A100
is queued. This does not execute CUDA kernels or use Blackwell for the baseline:

```bash
install_job=$(sbatch --parsable --account=acct-ykasten --qos=qos-ykasten \
  --partition=part-ykasten --gres=none --mem=64G \
  --export=ALL,LIFT4D_INSTALL_CPU_ONLY=1 \
  --output=runs/setup/install-%j.log --error=runs/setup/install-%j.err \
  scripts/slurm/install.sh)
sbatch --dependency=afterok:"$install_job" \
  --output=runs/setup/check-%j.log --error=runs/setup/check-%j.err \
  scripts/slurm/check.sh
```

CPU installation forces CUDA extension compilation for the configured Ampere
architectures and runs imports/static argument checks. A separate A100 job runs
the small GPU checks. The obsolete `pypi.ngc.nvidia.com` index is omitted because
its DNS no longer resolves in this environment; PyPI, the original Torch cu121
index and the original Kaolin wheel page provide the dependencies.
`scripts/requirements-constraints.txt` keeps upstream Torch, NumPy and stage-3
versions compatible across pip installation phases.

Weight revisions are recorded in `experiments/weights.json`. After installing the
environment, download the needed weights without loading them:

```bash
python scripts/download_weights.py --model zero123 --model da3
# After the gated SAM3D access request has been approved:
python scripts/download_weights.py --model sam3d
```

These are large downloads; the script records download status under `runs/setup/`.
DA3 uses the standard HF cache, Zero123 uses its original checkpoint location,
and SAM3D uses the original `sam3d/checkpoints/hf` layout.

The published `decord==0.6.0` Linux wheel is named `py3-none` but incorrectly
contains a `cp36-cp36m` tag in its WHEEL metadata. `fix_decord_wheel.py` repairs
that metadata and its RECORD entry to match the published filename, after
checking the installed version and import. It does not change package code or
binaries. This avoids a false unsupported-platform failure in `pip check`.

Additional model files referenced by the original SAM3D and Zero123 configs
are cached without instantiating models:

```bash
python scripts/download_weights.py --model moge
python scripts/download_weights.py --cache-backbones
```

The latter downloads DINOv2 ViT-L/14 with registers, the original CLIP ViT-L/14,
AlexNet/VGG16 feature weights and PIQ LPIPS coefficients. It streams files into the standard
Torch/CLIP cache and checks available SHA256 prefixes. Cache destinations remain
the original loader defaults; `TORCH_HOME` controls the Torch cache.

A tiny Torch-only GPU check is also available on a Turing GPU, independent of
Ampere extension compilation:

```bash
sbatch --output=runs/setup/torch-check-%j.log \
  --error=runs/setup/torch-check-%j.err scripts/slurm/torch-check.sh
```

This is only an 8x8 matrix product with a CPU comparison. A passing result does
not certify the Ampere extensions or sufficient VRAM for full LIFT4D.

The unpinned utils3d inference requirement currently resolves to 1.7, which
removes `intrinsics_from_fov_xy` used by the unchanged baseline. The installer
restores utils3d commit `3913c65d81e05e47b9f367250cf8c0f7462a0900`, already
pinned by the original MoGe dependency, after installing inference requirements.
The static API check verifies the original utils3d calls remain available.

The baseline mixes old Torch utils3d names with three newer NumPy visualization
names. No single upstream version exposes both sets. The isolated environment
therefore uses the original MoGe revision plus three direct name aliases:
`depth_map_edge = depth_edge`, `point_map_to_normal_map = points_to_normals`,
and `build_mesh_from_map = image_mesh`. `install_utils3d_compat.py` installs
this additive shim through an environment-only `.pth` file. It adds no math or
new defaults and changes no original function, model source, or other environment.
The dependency check verifies both dotted calls and explicit imported APIs.

## Installed workspace validation

The environment, CUDA extensions and all baseline weight files have been
installed locally. `experiments/paths.local.json` selects the isolated environment
Python executable. `pip check`, dependency imports, original CLI parsing and
all required utils3d API names passed. The SAM3D pipeline's twelve referenced
config/checkpoint files exist and are non-empty.

SLURM job `102213` passed the full lightweight GPU check on an RTX A6000,
including Torch, PyTorch3D KNN, simple-knn, xformers and FlashAttention kernels.
Job `102170` also passed a Torch-only check on RTX 2080 Ti. The queued A100
check was cancelled after the compatible Ampere A6000 check passed. To repeat
the same A6000 check on this cluster:

```bash
sbatch --account=acct-ykasten --partition=part-preempt-classB \
  --qos=qos-preempt --gres=gpu:A6000:1 --cpus-per-task=2 --mem=16G \
  --time=00:10:00 --output=runs/setup/gpu-check-%j.log \
  --error=runs/setup/gpu-check-%j.err scripts/slurm/check.sh
```

Reports and resolved package versions are under `runs/setup/` (ignored by Git).
No LIFT4D inference, training or full model instantiation was executed.
