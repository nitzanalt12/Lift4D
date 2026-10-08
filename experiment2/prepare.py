"""Inspect a saved SAM3D mesh and baseline checkpoint inventory; no inference."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import subprocess
from .mesh import load_glb,save_vertices,snapshot
from .checkpoints import inspect_checkpoint,load_bundle

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',required=True)
    p.add_argument('--sequence',required=True)
    p.add_argument('--mesh-frame',required=True,help='Inspection frame only; does not choose the experiment canonical frame')
    p.add_argument('--stage',choices=['geometry','appearance'])
    p.add_argument('--checkpoint',type=int)
    p.add_argument('--output',required=True,help='New ignored directory or directory outside the checkout')
    args=p.parse_args()
    if bool(args.stage)!=(args.checkpoint is not None):
        p.error('--stage and --checkpoint must be specified together; there is no automatic latest selection')
    run=Path(args.baseline_run).resolve()
    metadata=json.loads(snapshot(run/'metadata.json'))
    inventory=json.loads(snapshot(run/'inventory.json'))
    if args.sequence not in inventory or args.mesh_frame not in inventory[args.sequence]['frames']:
        p.error('Sequence/frame is not in the saved baseline inventory')
    output=Path(args.output).resolve()
    if output.is_relative_to(run):
        p.error('Preparation output must be separate from its source baseline run')
    if output.is_relative_to(ROOT) and subprocess.run(['git','-C',str(ROOT),'check-ignore','-q',str(output)]).returncode:
        p.error('Preparation output inside checkout must be ignored')
    mesh=load_glb(run/'sam3d'/f'davis_{args.sequence}'/args.mesh_frame/'result.glb')
    plan=json.loads((ROOT/'experiments/configs/sam3d_mesh_preparation.json').read_text())
    checkpoints=[]
    model_configs={}
    for stage,suffix in [('geometry','node'),('appearance','node_delta')]:
        cfg_path=run/'models'/f'davis_{args.sequence}_{suffix}'/'cfg_args'
        if cfg_path.is_file():
            expression=ast.parse(snapshot(cfg_path).decode(),mode='eval').body
            if not isinstance(expression,ast.Call) or not isinstance(expression.func,ast.Name) or expression.func.id!='Namespace' or expression.args:
                raise ValueError('Unexpected cfg_args format; code execution is forbidden')
            model_configs[stage]={k.arg:ast.literal_eval(k.value) for k in expression.keywords if k.arg is not None}
        for path in sorted((run/'models'/f'davis_{args.sequence}_{suffix}'/'deform_gs').glob('iteration_*')):
            if path.name.removeprefix('iteration_').isdigit():
                checkpoints.append(inspect_checkpoint(run,args.sequence,stage,int(path.name.removeprefix('iteration_'))))
    bundle=load_bundle(run,args.sequence,args.stage,args.checkpoint) if args.stage else None
    output.mkdir(parents=True,exist_ok=False)
    geometry=save_vertices(mesh,output/'mesh_arrays')
    mesh_record={k:v for k,v in mesh.items() if k!='geometries'}
    mesh_record['geometries']=geometry
    report={'kind':'preparation_only','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'preparation_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'baseline_run':str(run),'baseline_metadata':metadata,'sequence':args.sequence,
            'inspection_mesh_frame':args.mesh_frame,'canonical_frame_selected':False,
            'mesh':mesh_record,'checkpoints':checkpoints,'plan':plan,
            'saved_model_configs':model_configs,
            'selected_checkpoint':bundle['checkpoint'] if bundle else None,
            'checkpoint_hashes':bundle['hashes'] if bundle else None,
            'state_shapes':{name:{k:list(v.shape) for k,v in state.items() if hasattr(v,'shape')}
                            for name,state in bundle['states'].items()} if bundle else None,
            'operations_performed':['read saved GLB','extract unchanged local vertices and faces','record scene transforms',
                                    'inspect checkpoint completion']+(['load explicit saved state on CPU'] if bundle else []),
            'deformation_applied':False,'training_performed':False}
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(output),'sequence':args.sequence,'mesh_geometry':geometry,
                      'checkpoint_count':len(checkpoints),'deformation_applied':False},indent=2))


if __name__=='__main__':
    main()
