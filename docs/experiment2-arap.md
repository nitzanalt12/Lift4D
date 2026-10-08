# Experiment 2.4: ARAP projection of frozen mesh transfer

Start independently from saved experiment 2.1's object-space SAM3D mesh vertices
(source baseline appearance 30000). No network training or learned deformation
forward pass is run. Frozen context is restored solely to reuse the exact original
camera placement and renderer; first/middle/last selected uncorrected renders must
reproduce saved 2.1 RGB within one uint8 level before correction. Existing frame
IDs, cameras, smoothed transforms and appearance compose remain unchanged.

For each frame, optimize corrected vertices q and proper local rotations R with
canonical original mesh p and saved raw transfer targets y:

```
E = mean_i ||q_i-y_i||^2 / D^2
    + lambda * mean_(i,j in unique edges)
      (||(q_i-q_j)-R_i(p_i-p_j)||^2
       +||(q_i-q_j)-R_j(p_i-p_j)||^2) / (2 D^2)
```

D is the canonical bounding-box diagonal. Use **all original triangle edges**,
positive uniform weights and per-vertex SO(3) rotations. This is vertex/spokes
ARAP, not the baseline control-node ARAP or displacement smoothing from 2.3.
Uniform weights avoid negative cotangent weights on the supplied irregular mesh;
changing weighting is a separate protocol variation. Global translation/rotation
is unconstrained by ARAP and retained by the soft target term.

Alternate proper local SVD rotation fits (reflection corrected) with a global
sparse solve. The fixed SPD system `I + lambda*N/E*L` is factored once per
strength using CPU sparse LU; batched local SVD uses the allocated GPU. The
initial iterate is the saved raw target independently for each frame. No temporal
regularizer is added in this first controlled comparison.

## Pilot

Camel, all 90 original frames, strengths 1 / 10 / 100, up to 15 local/global
iterations each; early stop at maximum vertex change below 1e-5 of D. Record
actual iteration count and whether tolerance was reached, without claiming that
15 iterations guarantee convergence. Network, topology, canonical mesh, colors,
feature assignment and camera mapping remain fixed; **output vertex positions
are optimized**. This is geometric postprocessing, not a zero-optimization transfer.
All three variants are separate outputs starting from 2.1, not cumulative.

```bash
sbatch --output=runs/mesh-arap-%j.log scripts/slurm/mesh-arap.sh \
  runs/sam3d_mesh_arap/camel-2.4-new \
  --strengths 1 10 100 --iterations 15
```

Check a single explicit frame, without presenting it as a completed sequence:

```bash
sbatch --job-name=lift4d-arap-check --output=runs/mesh-arap-check-%j.log \
  scripts/slurm/mesh-arap.sh runs/sam3d_mesh_arap/camel-check-new \
  --strengths 10 --iterations 3 --frames 00030 --smoke-check
```

Outputs include original-source hashes/provenance, config/commit, corrected
object-space vertices, unchanged canonical topology/colors, native RGB/alpha,
atomic partial/completed dashboard manifests and per-frame optimization JSONL.
The latter reports initial/final ARAP and soft target energies, iteration changes,
edge stretch p50/p95/p99, fractions above 2x/5x and below 0.5x, area ratio p95 and
near-collapsed face fraction (rest-degenerate faces are excluded and counted).
These are shape-distortion diagnostics, not a self-intersection detector or a
proof of orientation preservation. ARAP does not repair missing/bad canonical
connectivity and does not guarantee collision-free or non-inverted surfaces.

In the dashboard choose camel → 2.4 → ARAP source 30000, then use the Execution
selector's **ARAP lambda** labels to choose strength. Completed RGB/alpha use the
existing cached metrics. Refresh `python -m dashboard.index_outputs` when new
runs complete. Check mask alignment versus geometric distortion and temporal
stability; an improved ARAP objective alone is not evidence of better video fit.
All videos/frames belong to the same input sequence; no generalization claim.

## Checks

`python -m unittest experiment2.checks.test_arap -v` verifies rigid-transform
preservation, proper rotations, reduced stretch/energy on a toy deformed mesh,
and invalid-input rejection. The real GPU smoke check verifies camera reproduction
and reduced objective on the original 152498-vertex / 304936-face camel mesh.
No source baseline or frozen-transfer artifacts are modified.
