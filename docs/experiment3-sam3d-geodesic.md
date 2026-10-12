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
   sampled uniformly from frames 1..N-1 at each step (frame 0 is the fixed
   canonical pose and has no trainable deformation graph). The tracked material vertices, camera, and
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
   Use the original `DeformNetwork` position schedule: log-linear decay from
   `1.6e-4` to `1.6e-6` over its native 40,000-step horizon. A 20,000-step
   mesh run therefore ends at the midpoint of that schedule, as the native
   model would; the optimizer starts at `1.6e-4`.
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

The fixed camel tracks are already prepared at
`runs/experiment3/sam3d-geodesic/camel-20261012/prepared.json` (SLURM job
104582): 90 frames, 256 canonical queries, and 3,186 retained observations.
The preparation uses the existing SAM3D 2.1 camel mesh, the frozen baseline
appearance cameras and the local CoTracker3 checkpoint. SSH alias `rhino` did
not resolve, so the work uses SLURM's A100 allocation. To run a fresh smoke and
training chain:

```bash
sbatch --dependency=afterok:104582 --partition=part-preempt --gres=gpu:A100:1 \
  --output=runs/experiment3-smoke-%j.log \
  scripts/slurm/mesh-train-geodesic.sh train \
  --prepared auto --sequence camel \
  --output runs/sam3d_mesh_geodesic/camel-smoke-lrschedule-20261012 \
  --smoke-check --steps 1 --lr 1.6e-4 --lr-final 1.6e-6 --lr-schedule-steps 40000

# Submit the full job only after the smoke exits successfully.
sbatch --dependency=afterok:SMOKE_JOB_ID --partition=part-preempt --gres=gpu:A100:1 \
  --time=06:00:00 --output=runs/experiment3-train-%j.log \
  scripts/slurm/mesh-train-geodesic.sh train \
  --prepared auto --sequence camel \
  --output runs/sam3d_mesh_geodesic/camel-geodesic-lrschedule-20261012 \
  --steps 20000 --checkpoint-every 1000 --export-every 5000 \
  --lr 1.6e-4 --lr-final 1.6e-6 --lr-schedule-steps 40000
```

The second command intentionally depends on the successful smoke job. The
trainer stores each model checkpoint separately from each all-frame dashboard
export. Dashboard experiment 3 selections show only real saved renders and
their actual MLP step counts.

Two interrupted attempts are retained and labeled in the dashboard: the first
hit the fixed frame-0 pose, which has no trainable graph; the second
(`camel-geodesic-referencefix-20261012`) was stopped at step 3,000 after an
audit found a rising learning-rate schedule. Neither is a final result. The
complete run is `camel-geodesic-lrschedule-local-20261012`, trained for 20,000
steps with the native schedule and exported at 5k-step intervals.

## Limits to record

Video tracking and single-view 3D attachment can still be ambiguous. Mesh
geodesics follow SAM3D topology, including any incorrect fused limbs. A geodesic
neighborhood rejects spatial shortcuts; it does not establish anatomical joint
labels or guarantee absence of self-intersection. Metrics on training frames
measure fit quality, not held-out generalization.
