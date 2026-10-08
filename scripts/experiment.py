#!/usr/bin/env python3
"""Dependency-free run preparation; never imports or executes the ML pipeline."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    path = Path(os.path.expandvars(os.path.expanduser(path)))
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'prepare'])
    parser.add_argument('--config', default='experiments/configs/baseline.json')
    parser.add_argument('--paths', default='experiments/paths.local.json')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    config = read(resolve(args.config))
    paths = read(resolve(args.paths))
    if not re.fullmatch(r'[A-Za-z0-9_-]+', config['id']):
        raise ValueError('Invalid experiment id')
    subset = read(resolve(config['subset']))
    upstream = read(ROOT / 'experiments/upstream.json')
    git('merge-base', '--is-ancestor', upstream['commit'], 'HEAD')
    for key in ['data_root', 'davis_root', 'sam3d_weights', 'zero123_dir', 'runs_root']:
        paths[key] = str(resolve(paths[key]))
    # Explicit roots take precedence over inherited dataset environment settings.
    env = {'LIFT4D_DATA_ROOT': paths['data_root'], 'DAVIS_ROOT': paths['davis_root']}
    inventory = {}
    for name, stems in subset['objects'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not stems:
            raise ValueError('Invalid subset object')
        images = Path(paths['davis_root']) / 'JPEGImages' / subset['resolution'] / name
        masks = Path(paths['davis_root']) / 'Annotations' / subset['resolution'] / name
        actual = sorted(p.stem for p in images.glob('*.jpg'))
        if actual != stems:
            raise ValueError(f'{name}: DAVIS frame inventory differs from fixed subset')
        if any(not (masks / (stem + '.png')).is_file() for stem in stems):
            raise ValueError(f'{name}: missing masks')
        inventory[name] = {'frames': stems, 'frames_dir': str(images), 'masks_dir': str(masks)}
    required = [Path(paths['sam3d_weights']) / 'pipeline.yaml',
                Path(paths['zero123_dir']) / 'stable_zero123.ckpt',
                Path(paths['zero123_dir']) / 'sd-objaverse-finetune-c_concat-256.yaml']
    missing = [str(p) for p in required if not p.is_file()]
    report = {'experiment': config['id'], 'status': config['status'],
              'subset': subset['id'], 'frames': {k: len(v['frames']) for k, v in inventory.items()},
              'missing_weight_files': missing,
              'note': 'No ML imports, GPU checks, downloads, inference or training performed.'}
    if args.action == 'check':
        print(json.dumps(report, indent=2))
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = args.run_id or now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run id')
    run = Path(paths['runs_root']) / config['id'] / run_id
    # Keep runs in an ignored directory, or outside this checkout.
    if run.is_relative_to(ROOT):
        result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', str(run)])
        if result.returncode:
            raise ValueError('runs_root inside checkout must be ignored by Git')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'logs').mkdir()
    commands = []
    if config['status'] == 'ready':
        if config['id'] != 'baseline':
            raise ValueError('Only original baseline commands are implemented')
        for name, stems in subset['objects'].items():
            common = ['--dataset', 'davis', '--object_name', name]
            reconstruct = [paths['python'], '-u', 'run_inference.py', *common,
                           '--model_tag', paths['sam3d_weights'], '--sam3d_out', str(run / 'sam3d'),
                           '--subsampling', '1', '--num_frames', str(len(stems)),
                           *config['reconstruction_args']]
            commands.append({'stage': name + '_reconstruction', 'cwd': str(ROOT / 'sam3d'), 'argv': reconstruct})
            for stage, key in [('geometry', 'geometry_args'), ('appearance', 'appearance_args')]:
                argv = [paths['python'], '-u', 'train_lift4d_scgs.py', *common,
                        '--sam3d_input_dir', str(run / 'sam3d'), '--output_dir', str(run / 'models'),
                        '--zero123_dir', paths['zero123_dir'], *config[key]]
                commands.append({'stage': name + '_' + stage, 'cwd': str(ROOT / 'lift4d_scgs'), 'argv': argv})
    metadata = {'created_utc': now.isoformat(), 'commit': git('rev-parse', 'HEAD'),
                'branch': git('branch', '--show-current'), 'upstream': upstream,
                'git_status': git('status', '--porcelain'), 'python_preparer': sys.version,
                'subset_sha256': hashlib.sha256(resolve(config['subset']).read_bytes()).hexdigest()}
    for filename, value in [('config.json', config), ('paths.json', paths), ('subset.json', subset),
                            ('inventory.json', inventory), ('metadata.json', metadata),
                            ('commands.json', commands), ('environment.json', env), ('check.json', report)]:
        (run / filename).write_text(json.dumps(value, indent=2) + '\n')
    (run / 'logs/preparation.log').write_text(json.dumps(report, indent=2) + '\n')
    (run / 'preparer.py').write_text(Path(__file__).read_text())
    (run / 'working-tree.patch').write_text(git('diff', 'HEAD', '--binary'))
    script = ['#!/usr/bin/env bash', 'set -euo pipefail',
              'mkdir ' + shlex.quote(str(run / '.started')) + ' # refuse accidental rerun',
              'exec > >(tee -a ' + shlex.quote(str(run / 'logs/pipeline.log')) + ') 2>&1',
              "trap 'code=$?; printf \"%s\\n\" \"$code\" > " + shlex.quote(str(run / 'exit-code.txt')) + "' EXIT"]
    script += ['export ' + k + '=' + shlex.quote(v) for k, v in env.items()]
    script += ['python_command=' + shlex.quote(paths['python']),
               '"$python_command" -m pip freeze > ' + shlex.quote(str(run / 'environment.freeze.txt'))]
    for command in commands:
        script += ['cd ' + shlex.quote(command['cwd']),
                   shlex.join(command['argv']) + ' 2>&1 | tee ' + shlex.quote(str(run / 'logs' / (command['stage'] + '.log')))]
    if not commands:
        script += ["echo 'Experiment not implemented; no pipeline commands available.'", 'exit 2']
    (run / 'run.sh').write_text('\n'.join(script) + '\n')
    print(json.dumps({'run_dir': str(run), **report}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
