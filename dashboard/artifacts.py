"""Stable artifact snapshots and explicit alignment adapters; no ML imports."""
import io
import json
from pathlib import Path
import re
import time
import numpy as np
from PIL import Image

SCHEMA = 1


def valid_camera(camera):
    if not isinstance(camera, dict) or camera.get('model') != 'pinhole':
        return False
    try:
        vals = np.asarray([camera[k] for k in ['width', 'height', 'fx', 'fy', 'cx', 'cy']], dtype=float)
        pose = np.asarray(camera['world_to_camera'], dtype=float)
        return bool(np.isfinite(vals).all() and (vals[:4] > 0).all()
                    and pose.shape == (4, 4) and np.isfinite(pose).all()
                    and np.allclose(pose[3], [0, 0, 0, 1]))
    except (KeyError, TypeError, ValueError):
        return False


def stable_bytes(path):
    path = Path(path)
    a = path.stat()
    if time.time() - a.st_mtime < 2:
        raise ValueError('File is still settling; refresh shortly')
    data = path.read_bytes()
    b = path.stat()
    if (a.st_size, a.st_mtime_ns) != (b.st_size, b.st_mtime_ns):
        raise ValueError('File changed during read')
    if path.suffix.lower() == '.png' and not data.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'):
        raise ValueError('PNG has not finished writing')
    return data


def read_json(path):
    return json.loads(stable_bytes(path))


def image(path, mode=None):
    with Image.open(io.BytesIO(stable_bytes(path))) as im:
        im.load()
        return np.array(im.convert(mode) if mode else im)


def inside(root, value):
    p = (Path(root) / value).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact path escapes its run directory')
    return p


def runs(root):
    return sorted(str(p.parent.relative_to(root)) for p in Path(root).glob('**/metadata.json')
                  if (p.parent / 'inventory.json').is_file())


def describe(root, run_id):
    run = inside(root, run_id)
    inv = read_json(run / 'inventory.json')
    meta = read_json(run / 'metadata.json')
    selected = meta.get('selected_objects', list(inv))
    return {'run': run_id, 'metadata': meta, 'config': read_json(run / 'config.json'),
            'sequences': [s for s in selected if s in inv], 'demo': meta.get('demo', False)}


def views(run, sequence):
    if sequence not in read_json(run / 'inventory.json'):
        raise ValueError('Sequence is not in this run inventory')
    result = []
    video = run / 'sam3d' / f'davis_{sequence}' / 'comparison.mp4'
    log = run / 'logs' / f'{sequence}_reconstruction.log'
    try:
        # The producer prints this only after closing the encoder. Size/age alone
        # cannot distinguish an MP4 still being written from a completed one.
        marker = f'Comparison video saved: {video.resolve()}'
        if marker in stable_bytes(log).decode(errors='replace'):
            stable_bytes(video)
            result.append({'id': str(video.relative_to(run)),
                           'label': 'Reconstruction · saved comparison video (view only)',
                           'kind': 'video', 'stage': 'reconstruction', 'checkpoint': None,
                           'reason': 'Saved comparison: input left, SAM3D render right. Encoded/padded RGB; no alpha or verified video frame map. Alignment metrics and overlay unavailable.'})
    except (OSError, ValueError):
        pass
    for p in sorted(run.glob('dashboard_exports/*.json')):
        try:
            m = read_json(p)
            if m.get('sequence') == sequence:
                result.append({'id': str(p.relative_to(run)), 'label': m.get('label', p.stem),
                               'kind': 'manifest', 'stage': m.get('stage'), 'checkpoint': m.get('checkpoint')})
        except (OSError, ValueError):
            continue
    for stage in ['node', 'node_delta']:
        base = run / 'models' / f'davis_{sequence}_{stage}'
        for p in sorted(base.glob('comparison_iter_*')):
            result.append({'id': str(p.relative_to(run)), 'label': f'{stage} / {p.name}',
                           'kind': 'composite', 'stage': stage,
                           'checkpoint': p.name.removeprefix('comparison_iter_')})
    return result


def frame_index(run, sequence, view=None):
    inv = read_json(run / 'inventory.json')[sequence]
    ids = inv['frames']
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate input frame IDs')
    records = []
    viewinfo = next((v for v in views(run, sequence) if v['id'] == view), None)
    manifest = read_json(inside(run, view)) if viewinfo and viewinfo['kind'] == 'manifest' else None
    mapped = {}
    if manifest:
        if manifest.get('schema') != SCHEMA or manifest.get('camera_space') != 'input':
            raise ValueError('Manifest must declare schema=1 and camera_space=input')
        for row in manifest['frames']:
            if row['input_id'] not in ids or row['input_id'] in mapped:
                raise ValueError('Unknown or duplicate mapped input frame ID')
            mapped[row['input_id']] = row
    # The native loader enumerates inputs but derives reconstruction IDs from names.
    # Only accept its index mapping when numeric IDs are exactly contiguous from 0.
    native_ok = all(str(i).zfill(len(fid)) == fid for i, fid in enumerate(ids))
    for pos, fid in enumerate(ids):
        rec = {'id': fid, 'input': str(Path(inv['frames_dir']) / (fid + '.jpg')),
               'target': str(Path(inv['masks_dir']) / (fid + '.png')),
               'render': None, 'alpha': None, 'reason': 'No saved input-camera RGB/alpha render yet',
               'time': None, 'alignment': False, 'crop': None}
        if manifest and fid in mapped:
            row = mapped[fid]
            rec.update(render=str(inside(run, row['rgb'])), alpha=str(inside(run, row['alpha'])) if row.get('alpha') else None,
                       time=row.get('input_time_seconds'), target_object_id=manifest.get('target_object_id'))
            ti, tr = row.get('input_time_seconds'), row.get('render_time_seconds')
            ci, cr = row.get('input_camera'), row.get('render_camera')
            valid_time = isinstance(ti, (int, float)) and not isinstance(ti, bool) and np.isfinite(ti) and ti >= 0 and ti == tr
            rec['alignment'] = bool(valid_time and valid_camera(ci) and ci == cr and row.get('render_id') == fid)
            rec['reason'] = '' if rec['alignment'] else 'Frame ID, timestamps or explicit camera parameters do not match'
            rec['proof'] = row
        elif viewinfo and viewinfo['kind'] == 'composite' and native_ok:
            p = inside(run, view) / f'frame_{int(fid):04d}.png'
            if p.is_file():
                rec.update(render=str(p), crop='native-six-panel',
                           reason='Native labeled panel: explicit frame ID, but no exported camera proof or alpha; metrics unavailable')
        records.append(rec)
    times = [r['time'] for r in records if r['time'] is not None]
    if times and (len(set(times)) != len(times) or times != sorted(times)):
        for r in records:
            r['alignment'] = False
            r['reason'] = 'Timestamps must be unique and increasing'
    return records, viewinfo, manifest


def arrays(rec, object_id=1):
    target = image(rec['input'], 'RGB')
    render = image(rec['render']) if rec['render'] else None
    if rec['crop'] == 'native-six-panel' and render is not None:
        h, w = target.shape[:2]
        if render.shape[:2] != (2*(h+30), 3*w):
            raise ValueError('Native composite dimensions do not match documented 2x3 layout')
        # Explicit documented panel extraction, no resizing or alignment.
        render = render[h+60:2*(h+30), w:2*w, :3]
    if render is not None and render.shape[:2] != target.shape[:2]:
        raise ValueError('Input/render resolutions differ; resizing is forbidden')
    if rec['alignment']:
        camera = rec['proof']['input_camera']
        if (camera['height'], camera['width']) != target.shape[:2]:
            raise ValueError('Declared camera dimensions differ from input pixels')
    mask = image(rec['target'])
    if mask.ndim != 2 or mask.shape != target.shape[:2]:
        raise ValueError('Target mask must be an indexed image at input resolution')
    alpha = image(rec['alpha']) if rec['alpha'] else None
    if alpha is not None:
        if alpha.ndim != 2 or alpha.shape != mask.shape or alpha.dtype != np.uint8:
            raise ValueError('Alpha must be an 8-bit grayscale PNG at input resolution')
        alpha = alpha.astype(float)/255
    return target, render[:, :, :3] if render is not None else None, mask == object_id, alpha
