#!/usr/bin/env python3
"""Add three visualization API aliases to the original MoGe utils3d dependency."""
import importlib.metadata
import json
from pathlib import Path
import sysconfig

expected = '3913c65d81e05e47b9f367250cf8c0f7462a0900'
dist = importlib.metadata.distribution('utils3d')
source = json.loads(dist.read_text('direct_url.json'))
assert source.get('vcs_info', {}).get('commit_id') == expected, 'Install the original MoGe utils3d revision first'
site = Path(sysconfig.get_paths()['purelib'])
# Alias existing functions directly: no wrappers, changed defaults or new math.
module = '''import utils3d.numpy as _numpy
ALIASES = {
    "depth_map_edge": "depth_edge",
    "point_map_to_normal_map": "points_to_normals",
    "build_mesh_from_map": "image_mesh",
}
for _alias, _original in ALIASES.items():
    if not hasattr(_numpy, _alias) and hasattr(_numpy, _original):
        setattr(_numpy, _alias, getattr(_numpy, _original))
'''
(site / '_lift4d_utils3d_compat.py').write_text(module)
(site / 'lift4d_utils3d_compat.pth').write_text('import _lift4d_utils3d_compat\n')
print('Installed three NumPy visualization name aliases in the isolated environment; original functions unchanged.')
