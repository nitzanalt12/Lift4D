# Experiment 2: frozen SAM3D mesh transfer

Two independent variants start from the original baseline, with no optimizer or
additional training:

- **2.1:** final appearance checkpoint 30000, restored node base plus node delta.
- **2.2:** geometry checkpoint 20000, restored node only.

Both use the original SAM3D canonical mesh from frame `00000`, unchanged triangle
connectivity and saved vertex colors. Apply recorded GLB instance transforms and
invert the exact Z-up to Y-up rotation used by SAM3D's exporter. There is no
registration, normalization, scale fitting or learned adaptation. Associate each
vertex with its nearest **saved canonical Gaussian position** in source XYZ and
copy that Gaussian's learned features; save indices/distances/features and keep
them fixed for every frame. These are the user-selected protocol choices.

Evaluate the original `_deform_at_frame` helper with a minimal vertex/feature
adapter, strict complete checkpoint restoration and frozen parameters. Mesh
positions use the helper's `d_xyz`, as in the original baseline geometry path;
Gaussian orientation/scale outputs do not rotate individual mesh vertices.
Time is the original frame index divided by the maximum frame index.

Object-space vertex arrays are saved before camera placement. Camera rendering
uses the original five-frame transform smoothing for **both** stages and original
SAM3D camera-space conversion. Variant 2.1 additionally applies its saved
appearance compose transforms for rendering, exactly as the baseline does;
variant 2.2 has no appearance compose. This placement is recorded separately from
the deformation. Input cameras are baseline-estimated cameras, not externally
calibrated ground truth.

The mesh renderer is nvdiffrast CUDA, with interpolated original vertex colors,
no added lighting, black background and antialiased silhouette. Its appearance
renderer differs from Gaussian splatting; PSNR/LPIPS therefore include that
rendering/material difference. No occlusion compositing is applied to mesh
renders. RGB/alpha retain input dimensions; the manifest records frame identity
and the effective baseline camera without resizing or aligning.

## Run

The installed `lift4d` environment already contains the required dependencies.
From the repository root, on one allocated GPU:

```bash
python -m experiment2.transfer \
  --run runs/baseline/animals-camel-20261008T120751Z \
  --sequence camel --variant 2.1 \
  --output runs/sam3d_mesh_transfer/camel-2.1-example

sbatch --output=runs/mesh-transfer-%j.log scripts/slurm/mesh-transfer.sh \
  runs/sam3d_mesh_transfer/new-campaign camel rhino flamingo cows
```

Select `2.2` for geometry-only transfer. Outputs must be new directories outside
the baseline source run. Runs include configuration, baseline/checkpoint/mesh
provenance, implementation hashes, inventory, canonical topology/colors/fixed
feature assignments, per-frame object-space vertices, RGB/alpha and an atomic
partial/completed dashboard manifest. SLURM outer/per-variant logs are under
`runs/`; partial runs remain inspectable. All outputs are ignored by Git.

Open the existing dashboard on port 8765, refresh runs, and select the experiment
run and mesh view. Existing alignment checks and cached evaluation apply, with
DAVIS target object 255 and default alpha threshold 0.5. A completed manifest
triggers metrics; a partial manifest stays readable without claiming completion.

## SLURM email defaults

All project SLURM scripts request `END,FAIL` mail to
`nitzan.alt@campus.technion.ac.il`. The local account also has a `~/.local/bin/sbatch` wrapper that supplies these
options to `/usr/bin/sbatch`. `.bashrc` puts the wrapper on PATH before its
interactive-shell guard. New SSH/shell sessions therefore use these defaults;
already-open shells need `source ~/.bashrc`. Explicit later `sbatch` flags can
override them. Direct invocation of `/usr/bin/sbatch` bypasses the wrapper, so
project scripts retain explicit directives. SLURM 23.11 here does not honor the
proposed `SBATCH_MAIL_USER/TYPE` environment variables; job-level mail fields are
verified on the wrapper submission as well as the experiment scripts.

## Checks

```bash
python -m unittest discover -s experiment2/checks -v
```

Checks cover export coordinate round-trip, nearest-feature copying, invalid
coordinates, saved artifact completion and strict checkpoint attachment. Actual
GPU runs verify checkpoint restoration and finite vertices. Mesh rendering and
frame mapping are also checked against saved outputs in the dashboard.
