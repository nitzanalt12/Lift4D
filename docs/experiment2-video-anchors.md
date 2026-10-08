# Experiment 2.6: fixed-camera video handles + mesh ARAP

Branch `experiment2/video-anchors`. This pilot changes geometry optimization in
experiment 2 only. Start independently from original saved 2.1 object-space
meshes and checkpoint 30000; do not initialize from 2.3 / 2.4 / 2.5. No network
training, deformation-network forward, learned feature fitting, camera fitting,
RGB fitting, silhouette fitting or temporal target interpolation is performed.

## Annotation / review

Open `http://localhost:8765/video-anchors`, or the **Video anchors** link in the
dashboard. Prepared camel input frames are 00000, 00020, 00030 at 854 x 480.
Four fixed foot-patch proposals are the mean of the lowest 2% Z vertices in each
canonical lower-leg core from the attachment audit. These are geometric sole
center proposals, **not fitted skeletal joints**. Colors and canonical vertex IDs
stay fixed across time. First-frame projected proposals and the 3D attachment
audit let you check which foot each patch represents.

The initial labels are approximate assistant visual annotations, explicitly
marked **assistant-draft**. Identity was inspected through original frames
0 / 10 / 20 / 30 and first-frame canonical projections. There are 11 visible
observations; far hind sole in frame 0 is partly occluded and has no target.
Confidence is 0.7–0.9. These pixel coordinates are not automatic tracker output,
ground-truth joint locations or subpixel measurements. The first pilot has
feet only. A separate six-handle draft adds two visible **surface bends** on the
near front/hind legs. These are fixed surface vertices, not skeletal joint
centers. First-frame manual pixels choose the nearest projection in the explicit
canonical leg, using the frontmost surface within 1.5px of the minimum distance.
Bindings farther than 15px from the reference are rejected. Single-view depth
remains uncertain; all coordinates, vertex IDs and the selection rule are saved.
Only visibly identified bends are added; hidden far-leg joints are not guessed.
Inspect this draft at
`/video-anchors?definition=video_anchors/camel-bends-20261009`.

Choose frame / fixed handle, click the visible sole or named surface bend, set confidence and
explicitly confirm its identity. **Hidden / uncertain** removes the target.
Clicks are converted from display coordinates to original image pixels without
resizing input data. Identity confirmation is not inferred from a click. Saving
creates a new immutable file under `runs/video_anchors/<definition>/annotations/`;
it does not fit, train or submit jobs. Earlier drafts, labels and source files
are never overwritten. The displayed saved path can be passed to the fitting
command. Reviewed labels are a prerequisite to stronger anatomical conclusions.

## Objective and solver

For saved raw mesh y, optimized mesh q, canonical bbox diagonal D, fixed handle
weights a_ji, visible observed pixels u_j and confidence w_j:

```
E = sum_i c_i ||q_i-y_i||^2 / (N D^2)
  + lambda_ARAP * original symmetric vertex-edge ARAP / D^2
  + lambda_video * sum_j w_j ||pi_f(sum_i a_ji q_i)-u_j||^2
                   / (sum_j w_j * (image_width^2 + image_height^2))
```

Use all original triangle edges, positive uniform edge weights, proper local
SO(3) rotations, original topology/colors and a frozen perspective camera.
Handle weights are fixed normalized vertex-patch means, not learned weights.
Hidden/uncertain points are absent from the objective. Per-frame camera affines
come from the exact original appearance compose plus saved smoothed object
transform, verified against the original helper within 2e-5 camera units. Native
uncorrected RGB must reproduce saved 2.1 within one uint8 level before fitting.
Input frame IDs, image hashes, resolution and camera descriptions are checked.
No fitted image transform or camera adjustment is allowed.

Alternate original ARAP local rotations and a global Gauss-Newton update of
nonlinear handle reprojection. Reuse scalar sparse factorization and solve a
small Woodbury handle system. Backtrack against the exact full nonlinear energy,
including recomputed ARAP rotations; reject behind-camera handles. Record
iteration count, stopping flags, energy and fitted pixel errors. Max 150
iterations / relative vertex-change tolerance 1e-5 does not imply convergence.

## Matched controls and raw-motion confidence

Pilot uses ARAP lambda 100 and video weights 0 / 10 on the same annotated frames.
Weight 0 is a matched control, starting from the same raw meshes with identical
ARAP and solver settings. Unit per-vertex raw-motion prior (`c_i=1`) reproduces
the existing ARAP solver when video weight is zero; a toy check verifies this.

The first unit-prior pilot fits sole positions closely but leaves visible
distortions because unobserved leg vertices still follow incorrect raw motion.
A second, separately initialized protocol uses `c_i=0.001` in the explicitly
recorded four canonical lower-leg cores and `c_i=1` for body / unclassified
vertices. It keeps raw motion as a weak prior inside legs without inventing
targets for hidden feet. Its own weight-0 control uses exactly the same prior
partition and weights, isolating the addition of video observations. This is a
recorded regularization variation, not a cumulative experiment or new training.
ARAP, cameras and network remain unchanged; no skeleton or bone-length
constraint is added. It does not guarantee correct depth, articulation or
collision-free geometry.

## Commands

Use the existing `lift4d` environment. GPU work runs through SLURM with END/FAIL
mail to the configured project address.

```bash
sbatch --output=runs/video-anchors-prepare-%j.log \
  scripts/slurm/mesh-video-anchors.sh prepare \
  --seed-run runs/sam3d_mesh_transfer/animals-20261008-v1/camel-2.1 \
  --output runs/video_anchors/camel-new --frames 00000 00020 00030

sbatch --output=runs/video-anchors-fit-%j.log \
  scripts/slurm/mesh-video-anchors.sh fit \
  --definition runs/video_anchors/camel-new/definition.json \
  --annotations runs/video_anchors/camel-new/annotations/REVIEWED.json \
  --output-root runs/sam3d_mesh_video_anchors/camel-new \
  --arap-strength 100 --keypoint-weights 0 10 --iterations 150 \
  --leg-prior-weight 0.001
```

Prepare job 102393: `runs/video_anchors/camel-20261009`. Initial unit-prior job
102394: `runs/sam3d_mesh_video_anchors/camel-pilot-20261009`.
Weak-leg-prior job 102395:
`runs/sam3d_mesh_video_anchors/camel-soft-leg-prior-20261009`.
Each contains separate `camel-2.6-anchors-0` / `camel-2.6-anchors-10` outputs.
Six-handle surface-bend job 102396 completed:
`runs/sam3d_mesh_video_anchors/camel-bends-20261009`. It uses the weak leg prior
and 17 visible draft observations on the same three frames. Checkpoint states
remain bitwise unchanged. Its weight-0 geometry matches the earlier weak-prior
control within 2.1e-7 object units, confirming that adding unused handles did
not change that control.

To add explicitly annotated surface bends without replacing a definition:

```bash
python -m experiment2.extend_video_handles \
  --definition runs/video_anchors/camel-20261009/definition.json \
  --annotations runs/video_anchors/camel-20261009/annotations-assistant-draft.json \
  --landmarks runs/video_anchors/camel-20261009/manual-bend-landmarks-draft.json \
  --output runs/video_anchors/camel-bends-new
```

The manual landmark JSON contains annotator/source and handle records with ID,
label, color, explicit canonical leg region, first-frame native reference pixels
and per-frame visibility / identity / confidence / native pixels. The command
reads saved geometry only. It writes a new definition and draft annotations;
it does not infer landmark observations or train a detector.

Dashboard: camel → **2.6** → source 30000, then select executions distinguished
by video weight and leg prior. They are **partial 3-frame pilots**, not complete
90-frame results, and checkpoint labels flag draft anchors. Compare the same
prior's control and anchored result. Other frames stay unavailable. Cached
Mask IoU, boundary, PSNR and LPIPS use the original evaluation settings. Sole
reprojection error is explicitly a **fit metric on supplied observations**, not
held-out validation. Very small fit residuals do not establish true image or
3D accuracy, especially with approximate draft labels.

Partial manifests with settled, verified renders now load cached metrics
automatically. Sparse graph measurements appear as individual points; missing
frame intervals remain gaps rather than interpolated curves.

## Observed pilot results

All values below use exactly input frames 00000 / 00020 / 00030, DAVIS object
255, original input cameras/resolutions and render-alpha threshold 0.5. Boundary
p95 is the **mean of each frame's pooled bidirectional p95**, not a pooled
sequence percentile. Cached source settings and native per-frame rows are in
`runs/video_anchors/camel-bends-20261009/evaluation.json` and `assessment.json`.

| Protocol | Mask IoU | Boundary mean px | Boundary p95 px | Object PSNR dB | Object LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original raw 2.1, same three frames | 0.7515 | 7.224 | 22.641 | 10.686 | 0.4842 |
| Weak-leg-prior ARAP, video weight 0 | 0.7064 | 8.608 | 25.145 | 9.456 | 0.4719 |
| Weak-leg-prior ARAP + four sole handles | 0.7057 | 7.705 | 19.553 | 9.540 | 0.4727 |
| Weak-leg-prior ARAP + soles and two surface bends | 0.7089 | 7.650 | 19.430 | 9.563 | 0.4726 |

Visible fitted-target mean residuals for the six-handle result are 0.0016 /
0.0020 / 0.0040 px. These tiny errors reflect fitting approximate supplied
coordinates; they are **not landmark accuracy**. All three frames hit the
150-iteration cap without the convergence criterion being met. Canonical edge
stretch p95 is 1.156 / 1.273 / 1.248. There are still visible leg/articulation
artifacts. Video anchors improve boundary errors against their matched control,
but the overall Mask IoU remains below raw 2.1. This is an implemented diagnostic
pilot, **not a solved limb-motion problem or a complete animation**. No conclusion
about unobserved depth, hidden joints or temporal motion is established.

Outputs preserve definition / annotation versions and hashes, commit, solver
and driver hashes, checkpoints, camera provenance, native RGB/alpha, original
canonical arrays and per-frame logs / distortion / convergence flags. Actual
run data and annotations stay out of Git. No existing source artifact is edited.

## Checks

`python -m unittest experiment2.checks.test_video_anchors
dashboard.tests.test_video_annotations dashboard.tests.test_catalog -v` checks
camera Jacobian, unchanged zero-video-weight ARAP, energy and fitted-target
improvement, recorded positive raw-prior weights, hidden/unverified target
rejection, annotation versions without source mutation, definition digests and
partial/draft catalog labels. `dashboard/tests/video_anchors_browser.cjs` checks
real draft labels, native-pixel clicks under display scaling, explicit identity
confirmation, hidden points, frame switching and save payloads. Accepted browser
test saves are intercepted, so they do not create labels in real runs. The GPU
driver checks camera/render reproduction and rejects source provenance changes.
`dashboard/tests/video_results_browser.cjs` checks real six-handle pilots:
automatic single-run partial metrics, draft labels, matched 3/90 coverage,
visible isolated graph points, exact decoded frame-20 images in both panels,
missing frame-10 renders and navigation only to measured problematic frames.
