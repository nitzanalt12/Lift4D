# Experiment 2.3: fine-tune mesh deformation

This is a separate pilot from frozen transfer variants 2.1 and 2.2. Start with
camel experiment 2.1's exact final appearance state (baseline 30000, node base
plus node delta) and its canonical SAM3D mesh. Before training, require exact
checkpoint hashes, identical canonical arrays/features/nearest-Gaussian indices,
and bitwise identical deformed vertices to the saved 2.1 first/middle/last frames.

Only `control_point_deltas` (per-frame control-node translations) is trainable.
Node base, delta node positions/radii/rotations, vertex positions of the canonical
mesh, topology/colors/features, cameras, smoothed SAM3D placement and appearance
compose transforms stay fixed. Gaussian rotation/scale heads have no effect on
mesh vertex positions and are frozen. No baseline training code is modified.

Use the original `_deform_at_frame` implementation with its no-gradient decorator
unwrapped; its frozen base helper remains under no-gradient. The existing
nvdiffrast renderer supplies antialiased silhouette gradients. A first-step check
requires finite nonzero **mask-only** gradients to translations. After optimization,
all frozen deformation tensors must exactly match initialization, and translations
must have changed.

## Pilot settings

- 1000 Adam steps, learning rate 0.0001, random seed 23, gradient norm clip 1.
- One frame per step, shuffled permutations of all 90 input frames.
- Mask loss: `1 - soft IoU` over all native-resolution pixels, DAVIS object label
  255. Empty target masks or mismatched resolutions abort.
- Spatial loss (weight 0.01): mean squared difference between endpoint vertex
  displacements, divided by canonical bounding-box diagonal, on 20000 fixed
  sampled unique mesh edges. This penalizes spatial variation; it is not ARAP
  and does not guarantee triangle validity or prevent self-intersections.
- Temporal loss (weight 0.01): squared second finite difference of control-point
  corrections relative to the initial checkpoint, normalized by the same diagonal;
  first/last frame samples use the closest interior three-frame stencil.
- Anchor loss (weight 0.001): squared correction magnitude, normalized by the
  diagonal, over all control translations.

These are recorded pilot choices, not the original Gaussian training loss or
hyperparameters. There is no RGB/LPIPS training loss, geometry reconstruction,
texture optimization, vertex-feature learning, registration or camera fitting.
All frames are used for fitting: dashboard metrics measure fit quality on the
training sequence and are not evidence of held-out temporal/generalization
performance. Existing PSNR/LPIPS evaluation remains available but is not optimized.

## Launch

Dependencies are already installed in the `lift4d` environment. From repo root:

```bash
sbatch --output=runs/mesh-finetune-%j.log scripts/slurm/mesh-finetune.sh \
  runs/sam3d_mesh_finetune/camel-2.3-new
```

A two-step gradient/frozen-parameter check without rendering the full sequence:

```bash
sbatch --job-name=lift4d-mesh-gradient-check \
  --output=runs/mesh-gradient-check-%j.log scripts/slurm/mesh-finetune.sh \
  runs/sam3d_mesh_finetune/camel-gradient-new --steps 2 --smoke-check
```

For another sequence, use `python -m experiment2.finetune` with explicit `--run`,
`--seed-run`, `--sequence` and new `--output` on one allocated GPU. Source runs
are read only and cannot be used as output directories. SLURM uses the account's
verified completion/failure email default, also explicit in the script.

Each output contains config, commit/provenance, source/mask/checkpoint hashes,
input inventory, canonical mesh and fixed feature/edge assignments, gradient
check, per-step loss JSONL, checkpoints every 250 steps plus final optimizer/model
state, and final native-resolution RGB/alpha/object-space vertices. Checkpoints
and dashboard manifests are written atomically. `training-status.json` records
optimization completion; the final dashboard manifest separately records render
completion. Training logs remain usable if the job fails.

Refresh the existing dashboard and select the `sam3d_mesh_finetune` run and
`Experiment 2.3` view after rendering begins. Completed renders trigger cached
metrics as before. A running pilot without exported frames has unavailable
renders; no intermediate results are invented.

## Checks

```bash
python -m unittest discover -s experiment2/checks -v
```

Checks include mask-gradient direction, perfect overlap, uniform-translation
invariance of spatial regularization, linear-time invariance of temporal
regularization, and existing artifact/feature/checkpoint checks. The GPU smoke
check additionally verifies exact 2.1 reproduction, real rasterizer-to-delta
gradients, translation updates and unchanged frozen tensors.
