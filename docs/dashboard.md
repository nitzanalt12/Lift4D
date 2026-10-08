# Local result dashboard

The dashboard reads saved artifacts. It never imports LIFT4D, runs reconstruction,
loads a Gaussian checkpoint, changes training code or uses a GPU. Implementation:
`dashboard/catalog.py` (animal/experiment/checkpoint catalog),
`dashboard/artifacts.py` (adapters), `dashboard/metrics.py` (CPU evaluation and
cache), `dashboard/server.py` (local HTTP), `dashboard/static/` (browser UI).
No frontend build or external CDN is needed.

## Start

The installed environment already contains the dependencies. From the checkout:

```bash
/home/nitzan.alt/miniforge3/envs/lift4d/bin/python -m dashboard.server --runs-root runs --port 8765
```

Open `http://127.0.0.1:8765`. If running on the cluster login host, forward the port
from your own computer: `ssh -L 8765:127.0.0.1:8765 <your-cluster-login>`.
The server binds only to localhost. The runs root is configurable at startup;
its existing inventory snapshots supply input and target-mask paths.
Optional separate installation: create a Python 3.11 virtual environment **outside
the checkout**, then `pip install -r dashboard/requirements.txt`.
That installation supports all metrics except LPIPS until torch, torchvision,
`lpips==0.1.4` and local pretrained AlexNet/calibration weights are available.
The dashboard does not download weights. In this environment those weights are
already cached. LPIPS runs explicitly on CPU with two threads.

Select an animal, experiment and checkpoint. An execution selector appears only
when multiple runs match the same selection. The shared frame slider,
play/pause, charts and worst-frame buttons all use the same explicit input frame
IDs. Playback fps is a viewing speed, not inferred original-video timing.
Verified export timestamps are displayed; when absent, time is unavailable.
Overlay opacity changes display only. Cyan outlines mark the target object;
pink outlines require a saved alpha mask. Object mask value and alpha threshold
are explicit controls; changing evaluation settings invalidates displayed scores.
Click **Load / compute cached metrics** to read cached scores or evaluate only new
complete files. Refresh rescans outputs and retains the selected run/view when possible.

## What current baseline artifacts support

The inspected running baseline currently has input/mask inventories, metadata,
configuration, logs and DA3 NPZ files, but has not yet saved comparison images.
DA3 is not a rendered result and is never evaluated as one.

The native training saver writes
`models/davis_<sequence>_{node,node_delta}/comparison_iter_<iteration>/frame_<index>.png`.
These are labeled **2 x 3** composites. The adapter extracts the deformed panel
(bottom middle), explicitly removing the 30-pixel title bar per row. It checks
that composite dimensions equal `3W x 2(H+30)`; there is no resizing. Native IDs
are accepted only if the inventory has exactly contiguous numeric IDs starting
at zero, as required by the inspected native loader. Arbitrary file order is
never treated as an alignment map. Input pixels come directly from the inventory.

The native comparison has no saved alpha and no explicit camera/timestamp export.
It can be viewed but alignment scores are unavailable. Orbit renders are novel
views and are deliberately excluded. Reconstruction's `comparison.mp4` contains
compressed RGB panels, no alpha, no per-video frame map, and can be padded by the
encoder. It is not used for alignment evaluation. The current viewer uses saved
frame images to make a synchronized player, rather than guessing MP4 frame maps.
Original training code is unchanged. A separate [evaluation exporter](evaluation-export.md)
now produces saved RGB/alpha and explicit input-camera/frame metadata from final
checkpoints. The dashboard itself never performs checkpoint rendering.

## Minimal export adapter (also suitable for future experiments 2 and 3)

Place an immutable, completed JSON manifest in the run's `dashboard_exports/`.
Write the images completely first, then publish the manifest atomically using a
temporary file and rename. Partial sets of frames are supported. Add new frames
by atomically replacing the manifest. Paths below are relative to the run and
cannot escape it, including through symlinks. Input/mask paths and IDs remain the
run's existing `inventory.json`. No new exporter/inference is implemented here.

```json
{
  "schema": 1,
  "sequence": "rhino",
  "label": "Appearance / checkpoint 30000",
  "stage": "appearance",
  "checkpoint": "30000",
  "camera_space": "input",
  "target_object_id": 255,
  "frames": [
    {
      "input_id": "00000",
      "render_id": "00000",
      "input_time_seconds": 0.0,
      "render_time_seconds": 0.0,
      "input_camera": {
        "model": "pinhole", "width": 854, "height": 480,
        "fx": 1000.0, "fy": 1000.0, "cx": 427.0, "cy": 240.0,
        "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
      },
      "render_camera": {
        "model": "pinhole", "width": 854, "height": 480,
        "fx": 1000.0, "fy": 1000.0, "cx": 427.0, "cy": 240.0,
        "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
      },
      "rgb": "saved_renders/00000.png",
      "alpha": "saved_alpha/00000.png"
    }
  ]
}
```

**The camera numbers above illustrate the schema; they are not measured baseline
camera values.** A producer must supply the actual input camera intrinsics,
world-to-camera transform and time mapping from its render operation. The viewer
checks equal explicit IDs, finite matching nonnegative times, increasing unique
input timestamps, identical camera dictionaries, valid pinhole parameters and
image dimensions. The baseline exporter alternatively declares verified input-frame
identity and the normalized deformation coordinate when physical timestamps are
unavailable; no physical fps is invented (see the exporter documentation). A manifest is a producer declaration; equality checks cannot
independently prove a producer used the declared camera. Do not hand-invent camera
parameters or timestamps to make a render eligible. Undistorted pinhole cameras
only in v1; nonmatching or missing declarations are not evaluated.

For these four DAVIS-2016 sequences, inspected masks are grayscale with values
0 and 255; select value **255**. Indexed DAVIS masks with multiple objects require
an explicit selected nonzero label. The UI suggests a value only when the first
mask has a single foreground value, or the manifest declares the object.
Evaluation refuses a selected value differing from the manifest object value.
Mask source path, label value, sequence and alpha threshold are saved in cache
provenance; foreground is equality to that exact mask value, not all nonzero labels.
If alpha is missing, verified RGB still supports PSNR/LPIPS; silhouette scores
remain unavailable. Alpha files must be grayscale uint8 PNG at exact input size, normalized by 255;
foreground is `alpha >= threshold`. No RGB-derived silhouette approximation.

## Metric definitions and cache

- **Mask IoU:** intersection / union of selected target mask and thresholded alpha.
  Both empty yields 1; one empty yields 0.
- **Symmetric boundary error:** each mask's inner boundary is foreground minus
  3 x 3 erosion (8-neighbor convention, image exterior is background). For every
  pixel on each boundary, take Euclidean pixel-center distance to the nearest
  pixel on the other boundary. Pool the two directional distance vectors and
  report their mean and 95th percentile (linear interpolation) per frame. An empty boundary on either
  side is unavailable, with a reason; it does not become a misleading zero.
- **Object PSNR:** uint8 RGB normalized to [0,1], mean squared error over the
  three channels of **target-mask pixels only**, then `-10 log10(MSE)`. Background
  does not dilute the error. Perfect equality is explicitly infinity; empty target
  is unavailable. Saved RGB is evaluated as supplied, including its existing
  compositing; there is no implicit alpha recompositing or color correction.
- **Object LPIPS:** pretrained AlexNet LPIPS v0.1, spatial output. Both full-size
  RGB images have pixels outside the target mask replaced by identical neutral
  gray 0.5, then are normalized to [-1,1]. Average the spatial LPIPS map only over
  target-mask pixels. This is a defined masked perceptual score, not vanilla
  whole-frame LPIPS. Receptive fields still cross the ROI boundary. No ROI crop,
  image resizing or GPU is used. Missing local weights/dependencies yields N/A.
- Summary cards average available finite per-frame values equally, show coverage,
  and do not fill missing frames. The boundary p95 card is explicitly the **mean
  of per-frame p95s**, not a pooled sequence percentile. Perfect PSNR frames are
  counted and displayed as infinity; a finite mean excludes them and is labeled.
  Worst-frame lists exclude missing/nonfinite scores. Graph gaps remain gaps.

Default cache: `~/.cache/lift4d-dashboard`, configurable with `--cache-root`.
Each JSON cache record includes evaluation version/settings, SHA-256 hashes and
paths of input/RGB/mask/alpha files, frame/time/camera map and metrics. LPIPS keys
also include both pretrained weight hashes and package versions. Threshold, object,
source content or camera-map changes produce different keys. Atomic cache writes
occur outside source runs. Cached evaluation hashes source files to detect changes
but does not repeat the metric kernels. Unavailable/transient records are not cached.

Files younger than two seconds or changing during a read are rejected. PNGs must
have a final IEND record and decode fully. Evaluation rechecks source hashes after
reading pixels. These checks support in-progress native image sets; a complete
PNG is considered a committed individual frame even while its stage continues.
For arbitrary external producers use immutable files and the atomic manifest
publication contract above; no settling heuristic can prove an unknown writer
will never rewrite a file later. No source artifacts are written by this viewer.

## Checks and demo

```bash
python -m unittest discover -s dashboard/tests -v
python -m dashboard.demo --runs-root runs/dashboard-demo
```

Enable **Show setup / smoke checks**, then select animal `synthetic` and experiment `DEMO` in the viewer. The orange DEMO banner,
checkpoint and commit identify synthetic results. The intentionally shifted
frame `00007` should be worst by IoU. Demo creation refuses to overwrite an
existing run. It does not copy demonstration scores into real runs.

Optional Chromium smoke test (server running with the demo and original run root):

```bash
# Use an installed Playwright package/browser; no need to install these for viewing.
PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_EXECUTABLE=/path/to/chromium node dashboard/tests/browser.cjs
```

The browser check covers a verified final rhino run with automatic metric coverage, a synthetic run, intentionally
out-of-order image responses, synchronized decoded pixels, play/pause, refresh,
overlay and worst-frame navigation. Python checks cover identity/empty masks,
known boundary displacement, target-only PSNR, alpha thresholds, missing alpha,
LPIPS ROI preprocessing, frame/camera/time validation, strict dimensions,
partial writes, native panel extraction and cache invalidation.


## Organized result selection and output index

The default viewer presents **animal → experiment → checkpoint**. Baseline is
experiment 1; experiments 2.1 and 2.2 are frozen final/geometry mesh transfer;
2.3 is mesh fine-tuning. Stage is explicit in checkpoint labels: geometry 20000
and appearance 30000 are baseline iterations, while fine-tuning 1000 means
1000 additional pilot steps starting from baseline 30000. Saved intermediate
fine-tuning checkpoints without renders are selectable and explicitly unavailable
for visualization; the dashboard does not render them. Default selection is
the latest available checkpoint; refresh preserves an explicit current selection.

Scientific identity comes from saved run configuration/metadata and view stage/
checkpoint, not folder order. A verified RGB/alpha export replaces the redundant
native comparison option for the same run/stage/checkpoint. If multiple executions
exist for the same selection, an **Execution** selector appears so they remain
accessible independently. Setup, unlaunched stubs, synthetic demos and gradient
checks are hidden by default, accessible through the auxiliary checkbox. Invalid
superseded runs without valid metadata remain excluded. Partial real runs are
supported; missing renders and alignment metrics retain their explanations.

Create or refresh the navigable output index:

```bash
python -m dashboard.index_outputs --runs-root runs
```

Browse `runs/results/<animal>/<experiment>/<stage>-<checkpoint>/<execution>/`.
Each directory contains relative links to the original run, renders, manifest and
checkpoint when available, plus `selection.json`. `catalog.json` contains the
complete indexed result selections. No data, weights or renders are copied, moved
or modified. Original paths remain valid for checkpoint provenance, evaluation
caches and scripts. The index is ignored by Git. Refresh the index command when
new outputs appear; the dashboard discovers new outputs independently on refresh.
The index refuses to replace ordinary existing files. Retired index entries are
historical aliases, not live discovery; the generated catalog is the current view.
