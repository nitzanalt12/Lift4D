# Experiment 2: preparation, without deformation transfer

This branch adds only saved-artifact readers and preparation records. It does not
change the baseline, run SAM3D/LIFT4D inference, apply a learned warp to a mesh,
start training, or choose the final experiment protocol. Dependencies numpy,
trimesh and torch are already available in the installed `lift4d` environment.
No new installation or GPU allocation is needed for this preparation.

## Inspect a saved mesh

From the repository root, explicitly select an **inspection frame** and a new
output directory separate from the baseline source run:

```bash
/home/nitzan.alt/miniforge3/envs/lift4d/bin/python -m experiment2.prepare \
  --baseline-run runs/baseline/animals-rhino-20261008T120751Z \
  --sequence rhino --mesh-frame 00000 \
  --output runs/sam3d_mesh_preparation/rhino-inspection
```

Frame `00000` here is an inspection example, not a decision about the canonical
mesh for the experiment. `preparation.json` captures source baseline metadata,
preparation commit, mesh SHA-256, mesh instances/scene transforms, bounds and
checkpoint availability. `mesh_arrays/*.npz` stores unchanged local vertices and
triangular faces separately per geometry. No centering, scale normalization,
axis rotation, vertex merging, scene transform application or mesh re-export is
performed. Textures and the original GLB remain in the untouched source file.

When an explicitly chosen checkpoint exists and its producer's completion log
confirms all writes are finished, optionally add `--stage geometry --checkpoint
20000` (or a separately decided appearance checkpoint). This loads tensor state
onto **CPU** using `torch.load(weights_only=True)` and records shapes/hashes.
There is no default stage/checkpoint and no automatic latest-checkpoint selection.
These example flags illustrate the interface; they do not select experiment 2's
checkpoint in the shared configuration.

## Loading interface

`experiment2.checkpoints.load_bundle(run, sequence, stage, iteration)` returns
complete saved state dictionaries plus their SHA-256 provenance. Geometry uses
`deform.pth` and its associated canonical `gaussians.ply`. Appearance additionally
requires `deform_node_base.pth` and, for the original appearance recipe,
`compose_transforms_app.pt`. The reader preserves the full saved appearance delta
rather than calling the training resume loader, which initializes a fresh zero
delta layer. No network forward pass is called.

`attach_frozen(module, state)` attaches a dictionary to an **already constructed**
caller-selected module, requires identical tensor keys, shapes and dtypes, loads
strictly, and freezes it. The caller must eventually construct the exact original
architecture and restore frame mappings from the chosen baseline; this utility
never guesses architecture arguments, resizes node arrays, invents feature
values, creates an optimizer or evaluates a network. The original constructors
allocate CUDA, so real module instantiation/verification is deferred to an
explicit SLURM GPU check after the checkpoint/protocol decisions. Current CPU
checks validate the attachment contract using a small fixture module.

Files younger than two seconds or changing during a read are rejected. GLB header
length must match file size. Checkpoint loading requires the exact native
`[ITER N] Checkpoint saved to <path>` marker, emitted after the saver finishes,
plus every required component. Missing/in-progress checkpoints are listed without
being loaded. Preparation refuses to overwrite outputs or write into its source
baseline run. Untrusted pickle execution is not enabled.

## Findings and decisions still open

1. **Checkpoint stage/components:** geometry `node` versus appearance
   `node_delta`. The latter includes both frozen node-base and trainable delta,
   with optional learned per-frame compose transforms on the appearance path.
   A lone `deform.pth` is not the full appearance deformation.
2. **Coordinates and canonical frame:** SAM3D's `to_glb` explicitly converts
   Z-up vertices to Y-up using row-vector multiplication by
   `[[1,0,0],[0,0,-1],[0,1,0]]`. Saved Gaussians are not passed through that GLB
   export. Scene transforms, the selected canonical frame and changes to trained
   canonical Gaussian positions must also be accounted for. The reader records
   original GLB coordinates and does not apply a proposed inverse transform.
3. **Vertex hyper-features:** the original node setup uses `hyper_dim=8` and `K=3` in the inspected saved
   `cfg_args` (the trainer has a fallback value of 4, which is not the effective
   configuration of these runs). `ControlNodeWarp.cal_nn_weight` uses learned Gaussian features and
   nodes' hyper-coordinates in its neighbor lookup. A mesh has no corresponding
   learned features. Zero features, spatial-only KNN or feature interpolation would
   each be a protocol choice; none is implemented or silently selected here.
4. **Time and placement:** the baseline normalizes frame ID by the maximum frame
   ID and uses `1/num_frames` as the node time interval. Keep the exact baseline
   frame map. Decide whether appearance compose transforms are part of the
   transferred deformation or later camera placement; do not silently apply them.

No deformation transfer or renders are produced by these preparation commands.
Once one sequence's baseline completes and is reviewed, it can be used for the
first actual experiment 2 run after resolving these choices. Completion of all
four baseline sequences is not a requirement for preparation.

## Checks

```bash
/home/nitzan.alt/miniforge3/envs/lift4d/bin/python -m unittest discover -s experiment2/checks -v
```

Checks cover unchanged local mesh vertices/topology, preserved instance transforms,
partial GLB rejection, checkpoint completion gating, appearance component presence,
CPU state loading and strict frozen attachment without remapping. The actual saved
SAM3D GLBs from the baseline are also inspected on CPU; no CUDA/inference is run.
