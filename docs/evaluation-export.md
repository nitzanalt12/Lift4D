# Final baseline evaluation export

`evaluation/export_input_views.py` is a separate saved-checkpoint renderer. It
never calls training, optimizer setup, SAM3D reconstruction or Zero123 guidance.
Original training source is unchanged. It loads the complete final appearance
bundle: canonical Gaussian PLY/features, node-base state, trained delta state and
per-frame compose scale/rotation/translation. It verifies every restored tensor.

It invokes the original `_deform_at_frame`, `apply_compose_transform_app`,
`apply_transform_to_gaussian` and `render_gaussian` methods. The pure smoothing
block is extracted from the original initializer, preserving the baseline's
configured window without executing training initialization. The checkpoint
iteration is used directly; no fresh appearance delta is initialized.

RGB uses the original black background, square canvas and integer center crop.
There is no resizing or camera alignment correction. Effective pinhole intrinsics
include the original crop offsets. The camera is the **baseline's estimated input
camera**, not a DAVIS ground-truth calibrated camera; its source and crop are
explicit in the manifest. Per-frame pose/compose follows the exact comparison
render chain. Deformation rotations are handled exactly as the original comparison
helper handles them; no extra rotation application is introduced.

For every frame, RGB is compared against the original final comparison's deformed
panel. Export refuses significant discrepancies (mean absolute uint8 error >0.5
or maximum >16), with per-frame observed errors recorded. The actual camel and
rhino exports matched exactly: mean and maximum errors zero for all 90 frames of
each sequence. Alpha is exported from the same rasterization, rounded to uint8
`round(alpha*255)`; evaluation uses this declared quantized alpha, not an RGB-derived
silhouette. The alpha threshold remains a visible evaluation setting.

## Run

```bash
# On an available GPU, with the installed environment:
python -m evaluation.export_input_views \
  --run runs/baseline/animals-camel-20261008T120751Z --sequence camel --checkpoint 30000

# Or submit the two-sequence export using SLURM:
sbatch --output=/home/nitzan.alt/.cache/lift4d-dashboard/export-evaluation-%j.log \
  scripts/slurm/export-evaluation.sh
```

The export supports exactly one visible CUDA GPU and refuses a missing allocation.
The SLURM wrapper processes camel then rhino. In this cluster two trial jobs failed
before export (a Conda activation nounset issue, then a missing actual GPU despite
SLURM allocation); activation was fixed and, as requested, the successful exports
ran on the SSH host's RTX 3090 using available memory. Baseline allocations were
not touched. No checkpoint or original comparison file was modified.

Each invocation creates a unique ignored `run/evaluation/<label>/` with `rgb/` and
`alpha/`, plus an atomically published `run/dashboard_exports/<label>.json`.
Completed individual frames can be discovered on refresh; complete=true is added
only when all frames are saved. Source checkpoint hashes, baseline/export commits,
source trainer hash, GPU, smoothing, model configuration and comparison parity are
recorded. Output belongs to its source run for discovery; source files are read only.

Input inventory IDs, saved SAM3D directory IDs, loader frame IDs and output IDs
must agree exactly. The exporter supports the inspected contiguous DAVIS frame
IDs, rejecting arbitrary positional pairing. DAVIS frame directories do not
supply trustworthy physical timestamps, so it does **not invent fps or seconds**.
Instead the manifest declares `time_mapping.kind=input_frame_identity`, explicit
input/render frame indices, and the original normalized deformation coordinate
`frame_index/max_frame_index`. The adapter verifies all three. Physical time is
shown as unavailable; playback fps is a viewing speed. Manifests with measured
physical input/render timestamps continue to use the existing timestamp contract.

## Dashboard and cached metrics

Refresh and choose **Input-camera RGB + alpha · appearance 30000**. Complete
verified exports automatically load/compute the metrics. CPU LPIPS is selected
by default; all five cards show valid-frame coverage. Threshold or mask-label
changes require refreshing the evaluation via the button and produce new cache
keys. Existing native comparison views still have N/A metrics; those original
files lack alpha/provenance and are not upgraded by assumption.

Only metric caches are written by the dashboard, outside run artifacts. LPIPS
uses existing local weights and CPU. IoU, symmetric boundary mean/p95, target-mask
PSNR and masked spatial LPIPS use the definitions in `docs/dashboard.md`.

Checks:

```bash
python -m unittest discover -s evaluation/checks -v
python -m unittest discover -s dashboard/tests -v
```

The browser smoke check now covers a verified final rhino export, automatic
90-frame metric coverage, overlay and five synchronized charts, in addition to
synthetic delayed-response frame synchronization checks.
