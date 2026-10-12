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
5. Train on all frames with valid visible-track reprojection, DAVIS silhouette,
   ARAP, temporal acceleration and neighborhood **material path length** losses.
   Canonical shortest edge paths between geodesic neighbors are fixed; sums of
   their deformed edge lengths are differentiable. These are lengths of the
   transported canonical paths, **not silently recomputed exact surface
   geodesics on the deformed mesh**. Record losses, weights and units explicitly.
6. Verify synthetic limb ambiguity, disconnected components, invisible tracks,
   projection gradients and rigid-motion invariance. Run a separate CUDA smoke
   job, including real nonzero position gradients and changed MLP weights. Only
   then submit full training on SLURM, with END/FAIL mail enabled. Export native
   input-view RGB/alpha/vertices and expose actual checkpoints in the dashboard.
7. Verify full training completion, checkpoint provenance, finite trajectories,
   exact fixed canonical pose, saved rendering coverage and cached mask/boundary
   evaluation. Do not call a successful job proof that limb identity is solved.

The first validation sequence is camel because existing limb diagnostics can
expose the known failure. The common implementation then runs the same protocol
on rhino, camel, flamingo and cows. No experiment-2 source is modified. Data,
tracks, checkpoints and renders stay under ignored `runs/` directories.

## Limits to record

Video tracking and single-view 3D attachment can still be ambiguous. Mesh
geodesics follow SAM3D topology, including any incorrect fused limbs. A geodesic
neighborhood rejects spatial shortcuts; it does not establish anatomical joint
labels or guarantee absence of self-intersection. Metrics on training frames
measure fit quality, not held-out generalization.
