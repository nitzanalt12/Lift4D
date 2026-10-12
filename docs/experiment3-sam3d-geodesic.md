# Experiment 3: fresh SAM3D mesh deformation with geodesic tracking neighborhoods

User-approved initialization: **new deformation network from scratch**. This
updates the original experiment-3 plan: start with SAM3D meshes; Pixal3D remains
a future reconstruction-source variant. Do not initialize from experiment 2's
trained deformation, fine-tuned deltas or ARAP projections.

## Plan and execution gates

1. Read canonical frame-0 SAM3D vertices/faces/colors and all original DAVIS
   frames from the existing four-animal subset. Preserve topology and native
   input cameras, with explicit frame/time mapping. Reuse the saved camera
   transforms for comparability, **not pretrained deformation weights/features**.
2. Select visible fixed canonical mesh vertices as video queries. CoTracker3
   produces native-pixel trajectories. Save tracker checkpoint/source hashes,
   image/mask hashes, visibility, forward/backward cycle errors and rejections.
   Never interpolate hidden points or substitute synthetic observations.
3. Build the original triangle-edge graph with 3D edge lengths. Shortest paths
   determine control influence and track-neighbor identities. Nearby pixels
   alone do not create neighbors or permit rebinding to another limb. Retain
   fixed mesh vertex identities throughout the sequence. Apply a conservative
   projection bound derived from canonical geodesic-neighbor distances to flag
   inconsistent visible track pairs, and save its depth/stretch assumptions.
4. Instantiate the original LIFT4D `DeformNetwork` MLP with random initialization
   and newly sampled mesh control anchors. Use fixed K=3 **geodesic** control
   influence, no Gaussian hyperfeatures or saved learned node radii. Train the
   shared MLP and position head; Gaussian rotation/scale heads cannot supervise
   mesh geometry and remain inactive. Subtract the frame-0 prediction so that
   the canonical pose is fixed exactly. This is full fresh geometry-field
   training, not optimization of old per-frame delta translations.
5. Train the freshly initialized LIFT4D position MLP for 20,000 steps, matching
   the baseline geometry-stage count for a 90-frame sequence. One frame is
   sampled uniformly at each step. The tracked material vertices, camera, and
   mesh are fixed; the network predicts control-node displacement, and the
   fixed K=3 geodesic weights interpolate it to every mesh vertex. Subtracting
   the frame-0 network prediction fixes that pose exactly. Rotation and scale
   heads stay inactive. Losses are soft silhouette IoU, robust track reprojection
   normalized by image diagonal, canonical triangle-edge length strain,
   second-difference temporal acceleration, geodesic material path stretch,
   displacement magnitude and an observed-track camera-depth barrier. Initial
   weights are mask=1, track=1, edge=1, temporal=.01, geodesic=1,
   displacement=.001, depth=100. These are explicit mesh-study choices, not a
   claim that the original Gaussian geometry objective directly applies to
   triangular surfaces. Canonical shortest edge paths between geodesic neighbors
   are fixed; their deformed lengths are differentiable material-path measures,
   **not silently recomputed exact surface geodesics on the deformed mesh**.
6. Run a separate one-step CUDA smoke job, including a real nonzero position
   head gradient and changed MLP weights. Only then run full training on SLURM,
   with END/FAIL mail enabled. Save atomically at 1,000-step intervals and
   export aligned all-frame RGB/alpha/vertices every 5,000 steps.
7. Verify full training completion, checkpoint provenance, finite trajectories,
   exact fixed canonical pose, saved rendering coverage and cached mask/boundary
   evaluation. Do not call a successful job proof that limb identity is solved.

The first validation sequence is camel because existing limb diagnostics can
expose the known failure. The common implementation then runs the same protocol
on rhino, camel, flamingo and cows. No experiment-2 source is modified. Data,
tracks, checkpoints and renders stay under ignored `runs/` directories.

The camel preparation is SLURM job 104574. Its current scheduled start estimate
is **2026-10-14 16:09 Asia/Jerusalem** (the job requests an A6000 on
`part-preempt-classB`); the trainer resolves the single settled camel
`prepared.json` automatically, so the preparation output path does not need to
be guessed when submitting its dependent jobs. The GPU remains unavailable from
the login node, and SSH host alias `rhino` does not resolve from this session.

Once tracking preparation finishes, submit the isolated smoke, then the full
training as a dependent job:

```bash
sbatch --dependency=afterok:104574 --output=runs/experiment3-smoke-%j.log \
  scripts/slurm/mesh-train-geodesic.sh train \
  --prepared auto --sequence camel \
  --output runs/sam3d_mesh_geodesic/camel-smoke-20261012 \
  --smoke-check --steps 1

# Submit the full job only after the smoke exits successfully.
sbatch --dependency=afterok:SMOKE_JOB_ID --output=runs/experiment3-train-%j.log \
  scripts/slurm/mesh-train-geodesic.sh train \
  --prepared auto --sequence camel \
  --output runs/sam3d_mesh_geodesic/camel-geodesic-20261012 \
  --steps 20000 --checkpoint-every 1000 --export-every 5000
```

The second command intentionally depends on the successful smoke job. The
trainer stores each model checkpoint separately from each all-frame dashboard
export. Dashboard experiment 3 selections show only real saved renders and
their actual MLP step counts.

## Limits to record

Video tracking and single-view 3D attachment can still be ambiguous. Mesh
geodesics follow SAM3D topology, including any incorrect fused limbs. A geodesic
neighborhood rejects spatial shortcuts; it does not establish anatomical joint
labels or guarantee absence of self-intersection. Metrics on training frames
measure fit quality, not held-out generalization.
