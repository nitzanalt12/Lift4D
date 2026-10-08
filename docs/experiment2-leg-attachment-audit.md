# Leg attachment audit (camel)

Open `http://localhost:8765/attachment-audit` through the existing SSH forward,
or choose **Leg attachment audit** in the results dashboard header. This separate
diagnostic view reads completed saved exports. It never evaluates LIFT4D or
modifies meshes, checkpoints, attachments or experiment outputs.

The canonical mesh has four provisional leg cores: four largest connected
components below explicit source-space Z=-0.06 in its largest component.
Front/hind is separated by canonical Y; sides are labeled -X/+X rather than
anatomical left/right. Other components and small lower fragments remain gray
and unclassified. Body and upper attachment regions are not leg cores. Check
the colors visually before interpreting them anatomically. Original connectivity
and vertex identities are preserved; there is no welding or resizing.

## Inspect

- Rotate / zoom the canonical mesh and lower its opacity to see control points.
- Select **Frozen base** or **Appearance delta**; points are colored by their
  canonical nearest-vertex attachment. White highlights the selected node.
- Click a point or enter its exact node ID. A white line joins it to its fixed
  attachment vertex. The table shows the regions receiving its mesh influence
  and its original baseline Gaussian influence.
- **Nodes influencing another leg** ranks nodes by cross-leg mesh influence.
  Selecting one shows an influence heatmap; return to fixed colors to inspect
  the four identities together.
- Switch base weights between original 2.1 and surface 2.5. Delta surface weights
  are unavailable because they are not part of 2.5. Black-to-red shows the sum
  of weights whose attached leg differs from the receiving vertex's leg.
- Choose saved frame 00000 / 00030 / 00060 / 00080. Mesh coordinates are the
  exact saved 2.1 or 2.5 arrays; region colors stay fixed by canonical vertex ID.
  Node positions show canonical XYZ plus the selected layer's saved translation,
  not the full composed mesh motion. Camera compose is not applied to this
  source-space diagnostic. These four poses are not a full-sequence animation.

Original Gaussian regions use nearest canonical vertex labels, with points more
than 0.02 canonical bbox diagonals away left unclassified. Node affinity sums
original Gaussian-to-node weights times saved Gaussian opacity. A high-purity
leg owner requires >=80% of **total** influence in one leg and support >=1.
Mixed, body-dominated and low-support nodes stay unclassified. These diagnostic
thresholds are recorded; they are not ground-truth segmentation.

## Observations

There are 2048 controls per layer; 265 attach to the lower-leg cores. Original
base interpolation gives a mean other-leg influence of 2.17% across all core
vertices, concentrated in Hind +X (10.15%). Original delta interpolation gives
0.00% under these labels. Base surface interpolation gives 0.00% in all four
cores. Body/unclassified influence is excluded from the crossing count and is
shown separately in the selected-node table.

Example node **1024** attaches to Hind -X. Of its original Gaussian influence,
75.6% falls in Hind -X and 24.4% is unclassified; none falls in Hind +X. Yet 22.5%
of its original **mesh** influence falls in Hind +X. This shows a transfer-weight
leak across canonical leg cores, without proving a physically wrong node anchor.

87 base and 113 delta nodes have high-purity Gaussian leg owners. None that
attaches to another leg core has a contradictory high-purity owner. Six base
and two delta high-purity leg owners attach outside the leg cores; the labels
cannot diagnose their upper-leg/body attachment. Lack of contradictions does
not establish correct motion, particularly for mixed nodes and unclassified
fragments. No automatic anchor reassignment is justified by this export.
Surface 2.5 already removes the measured cross-leg leakage but still has bad
leg motion. Visual inspection of the colored saved poses can now distinguish
within-leg folding / motion from mixing of vertex identities.

## Export / checks

```bash
sbatch --output=runs/mesh-leg-audit-%j.log scripts/slurm/mesh-leg-audit.sh \
  --seed-run runs/sam3d_mesh_transfer/animals-20261008-v1/camel-2.1 \
  --surface-run runs/sam3d_mesh_surface/camel-2.5-20261008 \
  --diagnostics runs/motion_diagnostics/camel-20261008-v2 \
  --output runs/attachment_audit/camel-new
```

GPU export uses the exact original base/delta node motions and Gaussian blend
weights; no optimizer is created. Raw meshes must reproduce 2.1 within 1e-6,
and every learned tensor must remain bitwise unchanged. Save checkpoint hashes,
commit, exporter/partition hashes, region definitions, node affinities, reports
and typed coordinate/index/weight buffers. Publish atomic `audit.json` last.
The actual export is `runs/attachment_audit/camel-20261008`, job 102373.

CPU server exposes only completed, settled exports and declared assets of the
correct byte size. Plotly is already installed in `lift4d`; independent setup
uses `pip install -r dashboard/requirements.txt`. Plotly JS is served locally;
the diagnostic needs browser WebGL and makes no external CDN requests.

`python -m unittest experiment2.checks.test_leg_audit
dashboard.tests.test_attachment_audit -v` checks influence accumulation, ownership
ambiguity, cross-leg mass, incomplete exports, undeclared assets and truncated
buffers. `dashboard/tests/attachment_browser.cjs` checks the actual full mesh,
fixed colors across poses, original/surface selection, node inspection and delta
availability. Use the existing Playwright environment variables. The real GPU
export verifies coordinate reproduction and frozen learned state.
