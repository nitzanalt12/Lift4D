#!/usr/bin/env python3
"""Download baseline weights without loading models; requires HF gated access."""
import argparse
import json
import os
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', choices=['sam3d', 'zero123', 'da3', 'moge'], action='append')
parser.add_argument('--cache-backbones', action='store_true', help='Cache DINOv2, CLIP and LPIPS feature weights without loading models')
args = parser.parse_args()
manifest = json.loads((ROOT / 'experiments/weights.json').read_text())
summary_path = ROOT / 'runs/setup/weights-download.json'
summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
failed = False
for name in args.model or ([] if args.cache_backbones else ['sam3d', 'zero123', 'da3', 'moge']):
    spec = manifest[name]
    options = {'repo_id': spec['repo'], 'revision': spec['revision'], 'max_workers': 4}
    if name == 'sam3d':
        options.update(local_dir=str(ROOT / 'sam3d/checkpoints/hf-dl'), allow_patterns=['checkpoints/*'])
    elif name == 'zero123':
        options.update(local_dir=str(ROOT / 'lift4d_scgs/extern/ldm_zero123'), allow_patterns=['stable_zero123.ckpt'])
    elif name == 'moge':
        options.update(allow_patterns=['model.pt'])
    else:
        # Original DA3 loader uses its HF cache; do not modify inference code.
        options.update(allow_patterns=['config.json', 'model.safetensors'])
    try:
        location = Path(snapshot_download(**options))
        if name == 'sam3d':
            target = ROOT / 'sam3d/checkpoints/hf'
            if target.exists():
                # Place refreshed checkpoint files in the existing directory.
                import shutil
                shutil.copytree(location / 'checkpoints', target, dirs_exist_ok=True)
            else:
                (location / 'checkpoints').rename(target)
            location = target
        summary[name] = {'status': 'downloaded', **spec, 'path': str(location)}
    except HfHubHTTPError as error:
        failed = True
        summary[name] = {'status': 'blocked', **spec, 'http_status': error.response.status_code}
        print(f"{name}: access/download blocked ({error.response.status_code}); other models will continue.", flush=True)
if args.cache_backbones:
    torch_home = Path(os.environ.get('TORCH_HOME', str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'torch')))
    downloads = [(name, spec, torch_home / spec['cache']) for name, spec in manifest['torch_cache'].items()]
    downloads.append(('clip_vitl14', manifest['clip_cache'], Path.home() / '.cache/clip' / manifest['clip_cache']['cache']))
    def download(item):
        name, spec, target = item
        if target.is_file() and target.stat().st_size > 0:
            return name, {'status': 'cached_existing', 'url': spec['url'], 'path': str(target), 'size': target.stat().st_size}
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + '.partial')
        digest = hashlib.sha256()
        size = 0
        print('DOWNLOAD', name, flush=True)
        with urllib.request.urlopen(spec['url'], timeout=60) as response, temp.open('wb') as output:
            expected = int(response.headers.get('Content-Length', '0'))
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if expected and size != expected:
            raise ValueError(f'{name}: incomplete download')
        checksum = digest.hexdigest()
        prefix = spec.get('sha256_prefix')
        if prefix and not checksum.startswith(prefix):
            raise ValueError(f'{name}: SHA256 mismatch')
        temp.replace(target)
        return name, {'status': 'downloaded', 'url': spec['url'], 'path': str(target), 'size': size, 'sha256': checksum}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for name, result in executor.map(download, downloads):
            summary[name] = result
(ROOT / 'runs/setup').mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
raise SystemExit(1 if failed else 0)
