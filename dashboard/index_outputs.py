"""Create a navigable, non-destructive results index over original output paths."""
import argparse
import json
import os
from pathlib import Path
from . import artifacts as a
from .catalog import build


def atomic_json(path,data):
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)


def link(destination,source):
    if destination.is_symlink():
        if destination.resolve()==source.resolve():return
        destination.unlink()
    elif destination.exists():raise ValueError(f'Refusing to replace an existing file: {destination}')
    destination.symlink_to(os.path.relpath(source,destination.parent),target_is_directory=source.is_dir())


def create(root):
    root=Path(root).resolve();index=root/'results';index.mkdir(exist_ok=True)
    catalog=build(root)
    for entry in catalog['entries']:
        for key in ['animal','experiment','stage']:
            if not all(c.isalnum() or c in '._-' for c in entry[key]):raise ValueError('Invalid index component')
        checkpoint=f"{entry['stage']}-{entry['iteration'] if entry['iteration'] is not None else 'pending'}"
        run=a.inside(root,entry['run'])
        # Include the source path digest so identical run basenames cannot collide.
        import hashlib
        identity=run.name+'-'+hashlib.sha256(entry['run'].encode()).hexdigest()[:6]
        directory=index/entry['animal']/entry['experiment']/checkpoint/identity;directory.mkdir(parents=True,exist_ok=True)
        link(directory/'run',run)
        if entry['kind']=='manifest':
            manifest=a.read_json(a.inside(run,entry['view']))
            link(directory/'manifest.json',a.inside(run,entry['view']))
            if manifest.get('frames'):
                rgb=a.inside(run,manifest['frames'][0]['rgb'])
                link(directory/'renders',rgb.parent.parent)
        elif entry['kind']=='composite':link(directory/'renders',a.inside(run,entry['view']))
        config=a.read_json(run/'config.json')
        checkpoint_path=None
        if entry['experiment']=='2.3':checkpoint_path=run/'checkpoints'/f"delta-{entry['iteration']:06d}.pt" if entry['iteration'] is not None else None
        elif entry['iteration'] is not None and entry['stage'] in ['geometry','appearance']:
            baseline=Path(config.get('baseline_run',run)).resolve()
            prefix='node' if entry['stage']=='geometry' else 'node_delta'
            checkpoint_path=baseline/'models'/f"davis_{entry['animal']}_{prefix}"/'deform_gs'/f"iteration_{entry['iteration']}"
        if checkpoint_path is not None and checkpoint_path.exists():link(directory/'checkpoint',checkpoint_path)
        atomic_json(directory/'selection.json',entry)
    atomic_json(index/'catalog.json',catalog)
    (index/'README.md').write_text('# LIFT4D result index\n\nBrowse animal / experiment / stage-checkpoint / execution.\n'
                                  '`run`, `renders`, `manifest.json` and `checkpoint` are links to unchanged original artifacts.\n'
                                  'Checkpoint identifiers for experiment 2.3 are fine-tuning steps, starting at baseline 30000.\n'
                                  'Setup/smoke tests and superseded invalid runs are excluded.\n'
                                  'Refresh this index with `python -m dashboard.index_outputs --runs-root runs`.\n'
                                  'Live dashboard selectors discover current original artifacts independently.\n')
    return catalog


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs-root',default='runs');args=parser.parse_args()
    catalog=create(args.runs_root);print(f"Indexed {len(catalog['entries'])} selections in {Path(args.runs_root)/'results'}")

if __name__=='__main__':main()
