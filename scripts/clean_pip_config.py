#!/usr/bin/env python3
"""Remove nvidia-pyindex's obsolete index from this environment's pip config."""
import configparser
from pathlib import Path
import sys

path = Path(sys.prefix) / 'pip.conf'
if path.is_file():
    config = configparser.ConfigParser()
    config.read(path)
    changed = False
    for key in ['extra-index-url', 'trusted-host', 'index-url']:
        if config.has_option('global', key):
            values = config.get('global', key).split()
            cleaned = [v for v in values if v.rstrip('/') not in ['https://pypi.ngc.nvidia.com', 'pypi.ngc.nvidia.com']]
            if cleaned != values:
                changed = True
                if cleaned:
                    config.set('global', key, ' '.join(cleaned))
                else:
                    config.remove_option('global', key)
    temporary_constraint = str(Path(__file__).resolve().parents[1] / 'scripts/requirements-constraints.txt')
    if config.get('global', 'constraint', fallback='') == temporary_constraint:
        config.remove_option('global', 'constraint')
        changed = True
    if changed:
        with path.open('w') as output:
            config.write(output)
        print('Removed obsolete NVIDIA index and temporary constraint from environment pip config.')
