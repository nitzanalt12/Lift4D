"""CPU checkpoint bundles and strict attachment to already constructed modules.

No network is instantiated here: original constructors allocate CUDA. No forward
call, Gaussian-to-mesh feature mapping, frame remapping or transform is performed.
"""
import hashlib
import io
import json
from pathlib import Path
from .mesh import snapshot

STAGES={'geometry':'node','appearance':'node_delta'}


def checkpoint_path(run,sequence,stage,iteration):
    if stage not in STAGES or not isinstance(iteration,int) or iteration<=0:
        raise ValueError('An explicit geometry/appearance stage and positive iteration are required')
    inventory=json.loads(snapshot(Path(run)/'inventory.json'))
    if sequence not in inventory or not sequence.replace('_','').replace('-','').isalnum():
        raise ValueError('Sequence is not in the baseline inventory')
    return Path(run)/'models'/f'davis_{sequence}_{STAGES[stage]}'/'deform_gs'/f'iteration_{iteration}'


def components(run,stage):
    config=json.loads(snapshot(Path(run)/'config.json'))
    required=['gaussians.ply','deform.pth']
    if stage=='appearance':
        required.append('deform_node_base.pth')
        if '--optimize_per_frame_compose_transforms_app' in config['appearance_args']:
            required.append('compose_transforms_app.pt')
    return required


def inspect_checkpoint(run,sequence,stage,iteration):
    path=checkpoint_path(run,sequence,stage,iteration)
    required=components(run,stage)
    log=Path(run)/'logs'/f'{sequence}_{stage}.log'
    try:
        closed=f'[ITER {iteration}] Checkpoint saved to {path.resolve()}' in snapshot(log).decode(errors='replace')
    except (OSError,ValueError):
        closed=False
    missing=[name for name in required if not (path/name).is_file()]
    return {'stage':stage,'iteration':iteration,'path':str(path.resolve()),'required_files':required,
            'missing_files':missing,'producer_completed':closed,'ready':closed and not missing}


def load_bundle(run,sequence,stage,iteration):
    import torch
    report=inspect_checkpoint(run,sequence,stage,iteration)
    if not report['ready']:
        raise ValueError('Checkpoint is missing or has not been confirmed complete by its producer')
    states={};hashes={}
    for name in report['required_files']:
        data=snapshot(Path(report['path'])/name)
        hashes[name]=hashlib.sha256(data).hexdigest()
        if name.endswith(('.pth','.pt')):
            states[name]=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    for name in ['deform.pth','deform_node_base.pth']:
        if name in states and (not isinstance(states[name],dict) or
                             not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in states[name].items())):
            raise ValueError(f'{name} is not a tensor state dictionary')
    return {'checkpoint':report,'hashes':hashes,'states':states,
            'note':'Unapplied saved states; Gaussian features and compose transforms preserved as dependencies'}


def attach_frozen(module,state):
    """Attach to a caller-selected exact original architecture; never evaluate it.

    Reject every key/shape mismatch before modifying a module. Do not use the
    training resume loader, which initializes a fresh appearance delta layer.
    """
    current=module.state_dict()
    if set(current)!=set(state):
        raise ValueError('State keys differ; caller must construct the exact saved architecture')
    if any(current[k].shape!=state[k].shape or current[k].dtype!=state[k].dtype for k in current):
        raise ValueError('State tensor shapes/dtypes differ; implicit remapping is forbidden')
    # Original ControlNodeWarp mutates its input dictionary and directly assigns
    # node tensors; protect the bundle and match the caller module's devices.
    prepared={k:v.to(device=current[k].device).clone() for k,v in state.items()}
    module.load_state_dict(prepared,strict=True)
    import torch
    restored=module.state_dict()
    if set(restored)!=set(state) or any(not torch.equal(restored[k].detach().cpu(),state[k].detach().cpu()) for k in state):
        raise ValueError('Original loader did not exactly restore every saved tensor')
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module
