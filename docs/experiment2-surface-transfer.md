# Experiment 2 motion diagnostics and surface-aware transfer

Branch: `experiment2/surface-motion-diagnostics`. Original training code and
all baseline / 2.1 / 2.3 / original 2.4 artifacts remain unchanged. GPU work uses
SLURM; the scripts email END / FAIL to the configured project address.

## Check ARAP convergence first

Repeat the original 2.4 objective independently from raw 2.1, with the same mesh,
uniform weights, soft targets and tolerance, increasing only iteration count.

```bash
sbatch --output=runs/arap-convergence-%j.log scripts/slurm/mesh-arap.sh \
  runs/sam3d_mesh_arap/camel-convergence-new \
  --strengths 10 100 --iterations 150 --frames 00030 00060 00080
```

This is a three-frame partial result, not a completed 90-frame sequence. The
dashboard distinguishes ARAP execution labels by lambda and iteration count.
Saved logs retain convergence flags; reaching the iteration limit is not
convergence. Compare only matching explicit frame IDs.

## Diagnose the frozen transfer

```bash
sbatch --output=runs/mesh-motion-diagnostics-%j.log \
  scripts/slurm/mesh-motion-diagnostics.sh \
  --seed-run runs/sam3d_mesh_transfer/animals-20261008-v1/camel-2.1 \
  --output runs/motion_diagnostics/camel-new \
  --frames 00030 00060 00080
```

Restores the exact saved original appearance checkpoint 30000. Raw mesh
coordinates must reproduce 2.1 within 1e-6 before diagnostics. Separates base
translation from appearance delta translation, and reports displacement,
adjacent displacement differences and distortion of base-only, delta-only and
full motion. These ablations are diagnostic, not new trained experiments.

For each saved control node, attach its XYZ to the nearest vertex on the original
canonical mesh. Build an undirected graph of all unique original triangle edges,
weighted by their Euclidean lengths. Distance to a node is shortest edge-path
length to its anchor plus its node-to-anchor offset. This approximates surface
distance; it is neither exact continuous geodesic distance nor limb labeling.
The graph has no added links, welding, remeshing or alignment.

Report the original node influences' edge-path / spatial distance ratios.
The explicit diagnostic shortcut rule is ratio > 3 and path > 0.1 canonical
bounding-box diagonal. It indicates potentially nonlocal influence; it does
**not prove an anatomically incorrect leg association**. In particular, the
original base uses XYZ plus learned hyper-features, so nonlocal XYZ influences
can be intentional. The appearance delta uses spatial XYZ alone.

The actual camel canonical mesh contains 17 components. One 56-vertex component
has no attached node; 934 vertices have fewer than K=3 available surface nodes.
Unavailable paths remain infinity, and unattached vertices get zero surface
weights in diagnostic files. No cross-component fallback is invented.

## Experiment 2.5: change only base blend weights

Starts independently from original checkpoint 30000 and the original SAM3D mesh,
as in 2.1. No fine-tuning or ARAP result is used as an initialization.

Replace only the base vertex-to-node blend weights with the K=3 nearest nodes
by the distance above. Keep the exact original Gaussian kernel, saved radii,
node-weight policy and 1e-7 floor, then normalize over available nodes. Weights
stay fixed over time. Original nearest-Gaussian features remain saved, but no
longer determine **base node selection**; base network inputs and motions are
unchanged. Appearance delta interpolation remains exactly original.

For the explicitly recorded 56 unattached vertices, retain their original base
mapping. Other components with fewer than three nodes normalize only finite
paths, never inventing a link. Store the affected vertex IDs in `base-weights.npz`.
This policy preserves every original vertex and face without silently bridging
components or treating unavailable surface weights as zero motion.

```bash
sbatch --output=runs/mesh-surface-transfer-%j.log \
  scripts/slurm/mesh-surface-transfer.sh \
  --seed-run runs/sam3d_mesh_transfer/animals-20261008-v1/camel-2.1 \
  --diagnostics runs/motion_diagnostics/camel-new \
  --output runs/sam3d_mesh_surface/camel-2.5-new
```

For a three-frame pilot, add `--frames 00030 00060 00080 --smoke-check`.
Pilots are hidden from the default dashboard catalog and can be shown with
auxiliary runs enabled. A complete run appears as camel → 2.5 → source 30000.
Use Compare A/B against 2.1 or 2.4 with synchronized frames and cached metrics.

Before replacing the base weight cache, verify exact original indices/weights,
canonical arrays and checkpoint hashes, raw mesh reproduction within 1e-6,
and original RGB reproduction within one uint8 level on first/middle/last
selected frames. After rendering, verify every saved learned tensor is bitwise
unchanged. Cameras, saved smoothed object transforms, appearance compose,
resolution, colors, topology, target object 255 and alpha threshold 0.5 remain
unchanged. Outputs retain provenance, commit, mapping/driver hashes, fixed
weights, per-frame distortion logs, native RGB/alpha and atomic manifests.

## Observations from the camel pilot

On frames 30 / 60 / 80, raw edge stretch p95 is approximately 11.96 / 8.52 /
10.42. Base-only motion gives almost the same distortion; delta-only gives
1.21 / 1.22 / 1.24. Original base weight discontinuities correlate strongly with
adjacent translation discontinuities; nearest-Gaussian feature jumps correlate
as well. These correlations support investigating the transfer interface,
without proving semantic leg identity.

Increasing ARAP from 15 to 150 iterations has very little visible effect on these
frames and none reaches the 1e-5 stopping threshold. This is not proof of a global
optimum. Surface-base weights reduce stretch p95 to 2.50 / 1.93 / 2.40 without
ARAP, but the pilot still has leg confusion. Its three-frame IoU is 0.708;
high-strength 150-iteration ARAP is 0.751 on the same three frames. Less stretch
does not guarantee correct motion or better silhouette alignment.

No semantic leg tracking, collision constraints, contact constraints, temporal
regularization or extra mask fitting are introduced. Masks and shape distortion
alone cannot certify which overlapping leg is which. These remain separate
possible protocol changes, not an implicit part of 2.5.

Checks: `python -m unittest experiment2.checks.test_surface_weights
dashboard.tests.test_catalog -v`. Folded-strip and disconnected-component tests
check surface locality, unavailable paths, explicit preservation policy, invalid
weight rejection, and partial-run checkpoint catalog identity. Real GPU pilot
checks reproduction, finite vertices and frozen learned state.
