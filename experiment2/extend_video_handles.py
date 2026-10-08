"""Add explicit draft surface landmarks to a new, immutable handle definition.

These are visible surface bends, not inferred skeletal joints. Single-view
depth and the canonical surface attachment remain geometric proposals.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from .leg_audit import leg_cores
from .mesh import snapshot
from .transfer import ROOT
from .video_anchors import project, validate_annotations
from evaluation.export_input_views import atomic_json


def surface_binding(rest, labels, region, xy, frame):
    if region not in (2, 3, 4, 5):
        raise ValueError('Explicit canonical leg region required')
    xy = np.asarray(xy, dtype=float)
    camera = frame['camera']
    if xy.shape != (2,) or not np.isfinite(xy).all() or not (0 <= xy[0] < camera['width'] and 0 <= xy[1] < camera['height']):
        raise ValueError('Reference landmark outside native image')
    ids = np.flatnonzero(labels == region)
    if not len(ids):
        raise ValueError('Empty canonical leg region')
    pixels = project(rest[ids], np.asarray(frame['linear']), np.asarray(frame['translation']), camera)
    distances = np.linalg.norm(pixels - xy, axis=1)
    # A pixel only gives a ray. Among nearly equally close projections use the
    # front surface; record this choice rather than claiming measured depth.
    candidates = ids[distances <= distances.min() + 1.5]
    depth = (rest[candidates] @ np.asarray(frame['linear']).T + frame['translation'])[:, 2]
    vertex = int(candidates[np.argmin(depth)])
    projected = project(rest[[vertex]], np.asarray(frame['linear']), np.asarray(frame['translation']), camera)[0]
    if np.linalg.norm(projected - xy) > 15:
        raise ValueError('Reference too far from the selected leg surface; review binding')
    return vertex, projected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--definition', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--landmarks', required=True, help='Explicit manual surface landmark JSON; never automatic keypoints')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    definition = json.loads(snapshot(Path(args.definition)))
    payload = {k: v for k, v in definition.items() if k != 'sha256'}
    if hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest() != definition['sha256']:
        raise ValueError('Source definition digest differs')
    annotations = json.loads(snapshot(Path(args.annotations)))
    validate_annotations(annotations, definition)
    manual = json.loads(snapshot(Path(args.landmarks)))
    if not manual.get('annotator') or not manual.get('annotation_source') or not manual.get('handles'):
        raise ValueError('Explicit manual landmark provenance required')
    raw = snapshot(Path(definition['seed_run']) / 'mesh/canonical.npz')
    if hashlib.sha256(raw).hexdigest() != definition['canonical_sha256']:
        raise ValueError('Canonical mesh differs')
    with np.load(io.BytesIO(raw)) as saved:
        rest, faces = saved['vertices'], saved['faces']
    labels, _ = leg_cores(rest, faces, definition['partition']['z_cutoff'])
    result = copy.deepcopy(definition)
    result['parent_definition_sha256'] = definition['sha256']
    result['manual_surface_landmarks'] = manual
    result['commit'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    result['extension_driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['binding_status'] = 'Draft canonical foot patches plus visible surface bend proposals; no skeletal joints or measured depth'
    first = result['frames'][0]
    seen = {h['id'] for h in result['handles']}
    added_rows = []
    for h in manual['handles']:
        if h['id'] in seen:
            raise ValueError('Duplicate fixed handle ID')
        seen.add(h['id'])
        vertex, projected = surface_binding(rest, labels, h['region_id'], h['reference_xy'], first)
        result['handles'].append({'id': h['id'], 'label': h['label'], 'region_id': h['region_id'], 'color': h['color'],
            'vertex_ids': [vertex], 'weights': [1.], 'canonical_xyz': rest[vertex].tolist(),
            'reference_frame_id': first['id'], 'reference_xy': h['reference_xy'], 'reference_projected_xy': projected.tolist(),
            'binding_source': 'Manual visible surface bend; nearest projection in explicit leg, front surface within 1.5px; fixed vertex, uncertain depth, not a bone joint'})
        for row in h['observations']:
            frame = next(f for f in result['frames'] if f['id'] == row['frame_id'])
            added_rows.append(dict(row, handle_id=h['id'], image_sha256=frame['image_sha256']))
    old_count = len(definition['handles'])
    for frame in result['frames']:
        with np.load(Path(definition['seed_run']) / 'mesh' / (frame['id'] + '.npz')) as saved:
            vertices = saved['vertices']
        added = result['handles'][old_count:]
        frame['original_handle_pixels'].extend(project(np.asarray([vertices[h['vertex_ids'][0]] for h in added]),
            np.asarray(frame['linear']), np.asarray(frame['translation']), frame['camera']).tolist())
        frame['canonical_handle_pixels'].extend(project(np.asarray([h['canonical_xyz'] for h in added]),
            np.asarray(frame['linear']), np.asarray(frame['translation']), frame['camera']).tolist())
    result.pop('sha256')
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    annotations['definition_sha256'] = result['sha256']
    annotations['parent_annotation_source'] = annotations['annotation_source']
    annotations['annotation_source'] += '; ' + manual['annotation_source']
    annotations['review_status'] = manual.get('review_status', 'manual-surface-binding-draft; surface identity and depth attachment require review')
    annotations['observations'].extend(added_rows)
    validate_annotations(annotations, result)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for frame in result['frames']:
        image_bytes = snapshot(Path(frame['image_path']))
        if hashlib.sha256(image_bytes).hexdigest() != frame['image_sha256']:
            raise ValueError('Input image differs')
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        draw = ImageDraw.Draw(image)
        for h, (x, y) in zip(result['handles'], frame['canonical_handle_pixels']):
            draw.ellipse((x-4, y-4, x+4, y+4), fill=h['color'])
            draw.text((x+6, y-9), h['id'], fill=h['color'])
        image.save(out / f'canonical-proposals-{frame["id"]}.png')
    atomic_json(out / 'annotations-template.json', {'schema': 1, 'definition_sha256': result['sha256'], 'annotator': '',
        'annotation_source': 'Manual original-image observations of fixed mesh surface handles', 'observations': []})
    atomic_json(out / 'annotations-assistant-draft.json', annotations)
    atomic_json(out / 'definition.json', result)
    print('Prepared new surface landmark definition: ' + str(out))


if __name__ == '__main__':
    main()
