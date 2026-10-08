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
and preparation. No ML environment was installed as part of this infrastructure
setup and no inference readiness is implied by a passing infrastructure check.

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
