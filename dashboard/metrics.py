"""CPU-only evaluation with content-addressed, atomic cache writes."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from .artifacts import arrays, stable_bytes

VERSION = 1
LPIPS_MODEL = None


def boundary(mask):
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def evaluate_arrays(target, render, gt, alpha, threshold=0.5, lpips_fn=None):
    if target.shape != render.shape or (alpha is not None and gt.shape != alpha.shape) or gt.shape != target.shape[:2]:
        raise ValueError('Evaluation requires exact dimensions')
    pred = alpha >= threshold if alpha is not None else None
    intersection = int((gt & pred).sum()) if pred is not None else 0
    union = int((gt | pred).sum()) if pred is not None else 0
    out = {'iou': (intersection / union if union else 1.0) if pred is not None else None,
           'boundary_mean': None, 'boundary_p95': None, 'psnr': None, 'lpips': None,
           'reasons': {}}
    if pred is None:
        out['reasons']['silhouette'] = 'Saved alpha unavailable; RGB does not define a silhouette'
    elif gt.any() and pred.any():
        a, b = boundary(gt), boundary(pred)
        distances = np.concatenate([distance_transform_edt(~b)[a], distance_transform_edt(~a)[b]])
        out['boundary_mean'] = float(distances.mean())
        out['boundary_p95'] = float(np.percentile(distances, 95, method='linear'))
    else:
        out['reasons']['boundary'] = 'An empty silhouette has no measurable boundary'
    if gt.any():
        difference = (target.astype(float)-render.astype(float))/255
        mse = float(np.mean(difference[gt]**2))
        out['psnr'] = float(-10*math.log10(mse)) if mse else 'Infinity'
        if lpips_fn:
            # Full-size frames, identical neutral gray outside GT; no crop/rescale.
            t, r = target.astype(np.float32)/255, render.astype(np.float32)/255
            t[~gt] = r[~gt] = 0.5
            try:
                out['lpips'] = float(lpips_fn(t, r, gt))
            except (ValueError, RuntimeError) as e:
                out['reasons']['lpips'] = str(e)
        else:
            out['reasons']['lpips'] = 'LPIPS was not requested or local weights are unavailable'
    else:
        out['reasons']['appearance'] = 'Target object mask is empty'
    return out


def local_lpips():
    global LPIPS_MODEL
    import torch
    import lpips
    weights = Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    calibration = Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
    if not weights.is_file() or not calibration.is_file():
        raise ValueError('Local AlexNet/LPIPS weights missing; dashboard never downloads weights')
    torch.set_num_threads(2)
    if LPIPS_MODEL is None:
        # Disable torchvision download path entirely; load the cached state ourselves.
        LPIPS_MODEL = lpips.LPIPS(net='alex', spatial=True, pnet_rand=True, verbose=False).cpu().eval()
        state = torch.load(weights, map_location='cpu', weights_only=True)
        from torchvision.models import alexnet
        backbone = alexnet(weights=None)
        backbone.load_state_dict(state)
        features = backbone.features
        offset = 0
        for layer in [LPIPS_MODEL.net.slice1, LPIPS_MODEL.net.slice2, LPIPS_MODEL.net.slice3,
                      LPIPS_MODEL.net.slice4, LPIPS_MODEL.net.slice5]:
            for child in layer.children():
                child.load_state_dict(features[offset].state_dict())
                offset += 1
    def score(t, r, mask):
        x = torch.from_numpy(t).permute(2,0,1)[None]*2-1
        y = torch.from_numpy(r).permute(2,0,1)[None]*2-1
        with torch.inference_mode():
            spatial = LPIPS_MODEL(x,y).squeeze().numpy()
        return spatial[mask].mean()
    fingerprint = hashlib.sha256(stable_bytes(weights)+stable_bytes(calibration)).hexdigest()
    return score, {'model':'alex-v0.1-spatial', 'weights_sha256': fingerprint,
                   'lpips_version': importlib.metadata.version('lpips'), 'torch_version': torch.__version__}


def cached_evaluate(records, settings, cache_root, lpips_fn=None):
    results = []
    for rec in records:
        row = {'id':rec['id'], 'time':rec['time'], 'iou':None, 'boundary_mean':None,
               'boundary_p95':None, 'psnr':None, 'lpips':None, 'reasons':{}}
        if not rec['alignment']:
            row['reasons']['alignment'] = rec['reason']
            results.append(row)
            continue
        try:
            object_id = settings['object_id']
            if rec.get('target_object_id') != object_id:
                raise ValueError('Selected object differs from export target_object_id')
            files = {k: hashlib.sha256(stable_bytes(rec[k])).hexdigest()
                     for k in ['input','target','render','alpha'] if rec.get(k)}
            provenance = {'version':VERSION, 'settings':settings, 'files':files,
                          'source_paths':{k:rec[k] for k in files}, 'alignment':rec.get('proof')}
            key = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
            dest = Path(cache_root)/f'{key}.json'
            if dest.is_file():
                row.update(json.loads(dest.read_text())['metrics'])
            else:
                t,r,g,a = arrays(rec,object_id)
                row.update(evaluate_arrays(t,r,g,a,settings['alpha_threshold'],lpips_fn))
                # Reject a source changed between fingerprinting and decoding.
                if any(hashlib.sha256(stable_bytes(rec[k])).hexdigest()!=v for k,v in files.items()):
                    raise ValueError('Artifact changed during evaluation; refresh')
                Path(cache_root).mkdir(parents=True,exist_ok=True)
                fd,tmp = tempfile.mkstemp(dir=cache_root,suffix='.tmp')
                with os.fdopen(fd,'w') as f:
                    json.dump({'provenance':provenance,'metrics':row},f,allow_nan=False)
                os.replace(tmp,dest)
            row['cache_key'] = key
        except (ValueError,OSError) as e:
            row['reasons']['unavailable'] = str(e)
        results.append(row)
    summaries = {}
    for metric in ['iou','boundary_mean','boundary_p95','psnr','lpips']:
        values = [r[metric] for r in results if isinstance(r[metric],(int,float))]
        perfect = sum(r[metric]=='Infinity' for r in results)
        summaries[metric] = {'mean':float(np.mean(values)) if values else ('Infinity' if perfect else None),
                             'available_frames':len(values)+perfect, 'perfect_frames':perfect}
    return {'settings':settings,'frames':results,'summary':summaries}
