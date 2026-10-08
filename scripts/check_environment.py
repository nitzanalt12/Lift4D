#!/usr/bin/env python3
"""Small GPU dependency checks only: no model load, inference or training."""
import importlib
import argparse
import ast
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'sam3d/notebook'), str(ROOT / 'sam3d'), str(ROOT / 'lift4d_scgs'), str(ROOT)]
os.environ['LIDRA_SKIP_INIT'] = '1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--imports-only', action='store_true', help='CPU imports and static CLI checks only')
args = parser.parse_args()
import torch
assert os.environ.get('SLURM_JOB_ID'), 'Run this GPU check through SLURM'
if not args.imports_only:
    assert torch.cuda.is_available(), 'CUDA GPU unavailable'
assert torch.__version__.startswith('2.5.1'), torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
result = {'job_id': os.environ['SLURM_JOB_ID'], 'torch': torch.__version__,
          'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
          'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if not args.imports_only else None, 'imports': []}
for name in ['torchvision', 'xformers', 'flash_attn', 'spconv.pytorch', 'pytorch3d',
             'kaolin', 'gsplat', 'nvdiffrast.torch', 'utils3d', 'depth_anything_3.api',
             'diff_gaussian_rasterization', 'simple_knn._C', 'lpips', 'piq',
             'inference', 'utils.stable_zero123_guidance']:
    importlib.import_module(name)
    result['imports'].append(name)
# Attribute checks catch utils3d API drift without executing reconstruction.
utils3d = importlib.import_module('utils3d')
required_api = set()
for file in (ROOT / 'sam3d/sam3d_objects').rglob('*.py'):
    source = file.read_text()
    required_api.update(re.findall(r'utils3d\.(?:torch|numpy)\.[A-Za-z0-9_]+', source))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module in ['utils3d.numpy', 'utils3d.torch']:
            required_api.update(node.module + '.' + item.name for item in node.names if item.name != '*')
missing_api = []
for dotted in sorted(required_api):
    obj = utils3d
    try:
        for part in dotted.split('.')[1:]:
            obj = getattr(obj, part)
    except AttributeError:
        missing_api.append(dotted)
assert not missing_api, f'utils3d APIs missing: {missing_api}'
result['utils3d_api'] = sorted(required_api)
if not args.imports_only:
    # Tiny CUDA kernels verify Torch, PyTorch3D and simple-knn ABI compatibility.
    x = torch.rand(16, 3, device='cuda')
    assert torch.isfinite(x @ x.T).all()
    from pytorch3d.ops import knn_points
    assert knn_points(x[None], x[None], K=1).dists.shape == (1, 16, 1)
    from simple_knn._C import distCUDA2
    assert torch.isfinite(distCUDA2(x)).all()
    from xformers.ops import memory_efficient_attention
    q = torch.randn(1, 8, 2, 32, device='cuda', dtype=torch.float16)
    assert torch.isfinite(memory_efficient_attention(q, q, q)).all()
    from flash_attn import flash_attn_func
    assert torch.isfinite(flash_attn_func(q, q, q)).all()
    torch.cuda.synchronize()
    result['kernels'] = ['torch-matmul', 'pytorch3d-knn', 'simple-knn', 'xformers-attention', 'flash-attention']
for cwd, filename in ([] if args.imports_only else [('sam3d', 'run_inference.py')]):
    proc = subprocess.run([sys.executable, filename, '--help'], cwd=ROOT / cwd,
                          capture_output=True, text=True)
    (ROOT / 'runs/setup' / (cwd + '-help.log')).write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, f'{filename} --help failed; see runs/setup/{cwd}-help.log'
# Reconstruct the original training parser without importing top-level model code.
import arguments
import lift4d_datasets as ds_registry
tree = ast.parse((ROOT / 'lift4d_scgs/train_lift4d_scgs.py').read_text())
body = next(n.body for n in tree.body if isinstance(n, ast.If) and '__main__' in ast.unparse(n.test))
selected = []
for stmt in body:
    if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in stmt.targets):
        break
    if isinstance(stmt, (ast.Assign, ast.Expr)):
        selected.append(stmt)
namespace = {'ArgumentParser': argparse.ArgumentParser, 'ModelParams': arguments.ModelParams,
             'OptimizationParams': arguments.OptimizationParams, 'ds_registry': ds_registry}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<original-training-parser>', 'exec'), namespace)
baseline = json.loads((ROOT / 'experiments/configs/baseline.json').read_text())
for stage in ['geometry_args', 'appearance_args']:
    namespace['parser'].parse_args(['--dataset', 'davis', '--object_name', 'rhino', *baseline[stage]])
result['training_cli'] = 'Original parser accepts both baseline phases without loading LPIPS'
result['note'] = 'No model instantiated, no inference, no training.'
(ROOT / 'runs/setup' / ('imports-check.json' if args.imports_only else 'gpu-check.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
