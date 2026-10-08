"""Export final saved baseline RGB/alpha via its original input-view render path."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def namespace_file(path):
    expr=ast.parse(Path(path).read_text(),mode='eval').body
    if not isinstance(expr,ast.Call) or not isinstance(expr.func,ast.Name) or expr.func.id!='Namespace' or expr.args:
        raise ValueError('Invalid saved cfg_args')
    return {k.arg:ast.literal_eval(k.value) for k in expr.keywords if k.arg is not None}


def camera_description(width,height,focal):
    size=max(width,height)
    left=(size-width)//2;top=(size-height)//2
    return {'model':'pinhole','width':width,'height':height,'fx':float(focal),'fy':float(focal),
            'cx':size/2-left,'cy':size/2-top,'world_to_camera':np.eye(4).tolist(),
            'source':'Original baseline input-view estimated camera; points already transformed to camera space',
            'native_square_canvas':size,'native_crop_left':left,'native_crop_top':top}


def original_smoothing(gui,window,trainer_path):
    # Reuse the original initialization's pure smoothing block without running
    # initialization, guidance networks, loss precomputation or optimizer setup.
    tree=ast.parse(Path(trainer_path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GUI')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_init_lift4d')
    blocks=[n for n in method.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
            and isinstance(n.test.left,ast.Name) and n.test.left.id=='st_win']
    if len(blocks)!=1:
        raise ValueError('Original smoothing block changed; refusing to guess')
    import torch
    code=compile(ast.fix_missing_locations(ast.Module(body=blocks,type_ignores=[])),str(trainer_path),'exec')
    exec(code,{'self':gui,'st_win':window,'np':np,'torch':torch})


def restore(module,state):
    import torch
    state_gpu={k:v.cuda().clone() for k,v in state.items()}
    module.load_state_dict(dict(state_gpu),strict=True)
    restored=module.state_dict()
    if set(restored)!=set(state) or any(not torch.equal(restored[k].detach().cpu(),v) for k,v in state.items()):
        raise ValueError('Checkpoint restoration is incomplete; refusing to export')
    module.eval()
    for param in module.parameters():param.requires_grad_(False)


def atomic_json(path,data):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--sequence',required=True)
    p.add_argument('--checkpoint',type=int,default=30000)
    args=p.parse_args()
    run=Path(args.run).resolve();sequence=args.sequence
    from experiment2.checkpoints import load_bundle
    bundle=load_bundle(run,sequence,'appearance',args.checkpoint)
    inventory=json.loads((run/'inventory.json').read_text())[sequence]
    stems=inventory['frames']
    if not all(fid==str(i).zfill(len(fid)) for i,fid in enumerate(stems)):
        raise ValueError('Original baseline loader requires contiguous numeric input IDs from zero')
    os.environ.update(json.loads((run/'environment.json').read_text()))
    import torch
    if torch.cuda.device_count()!=1:
        raise ValueError('One allocated CUDA GPU is required')
    sys.path.insert(0,str(ROOT/'lift4d_scgs'))
    from train_lift4d_scgs import GUI,SAM3DDataLoader,GaussianModel,DeformModel
    source=ROOT/'lift4d_scgs/train_lift4d_scgs.py'
    model=Path(bundle['checkpoint']['path']).parents[1]
    cfg=namespace_file(model/'cfg_args')
    config=json.loads((run/'config.json').read_text())
    flags=config['appearance_args']
    window=int(flags[flags.index('--smooth_transforms_window')+1]) if '--smooth_transforms_window' in flags else 0
    loader=SAM3DDataLoader(selector=sequence,canonical_frame_idx=0,device='cuda',dataset='davis',
                           object_name=sequence,input_dir=str(run/'sam3d'/f'davis_{sequence}'))
    if loader.get_frame_indices()!=list(range(len(stems))):
        raise ValueError('SAM3D IDs and input inventory do not match exactly')
    gui=GUI.__new__(GUI)
    gui.sam3d_data=loader;gui.frame_indices=loader.get_frame_indices();gui.num_frames=loader.num_frames
    gui.iteration=args.checkpoint;gui.is_node_delta=True
    gui.gaussians=GaussianModel(sh_degree=0,fea_dim=cfg['hyper_dim'],with_motion_mask=False)
    gui.gaussians.load_ply(str(Path(bundle['checkpoint']['path'])/'gaussians.ply'))
    base=bundle['states']['deform_node_base.pth'];delta=bundle['states']['deform.pth']
    kwargs={k:cfg[k] for k in ['K','skinning','hyper_dim','use_hash','hash_time','local_frame','progressive_brand_time']}
    gui.deform_node_base=DeformModel(deform_type='node',is_blender=True,d_rot_as_res=True,
                                    node_num=base['nodes'].shape[0],with_arap_loss=True,
                                    pred_opacity=False,pred_color=False,max_d_scale=-1,
                                    enable_densify_prune=False,is_scene_static=False,**kwargs)
    restore(gui.deform_node_base.deform,base)
    gui.deform=DeformModel(deform_type='delta',is_blender=True,d_rot_as_res=True,K=cfg['K'],
                          node_num=delta['nodes'].shape[0],num_frames=delta['control_point_deltas'].shape[0],
                          with_arap_loss=True,opt_deform_rot='--opt_deform_rot' in flags)
    restore(gui.deform.deform,delta)
    gui.deform.deform.set_frame_mapping(gui.frame_indices)
    gui.optimize_per_frame_compose_transforms_app='--optimize_per_frame_compose_transforms_app' in flags
    if gui.optimize_per_frame_compose_transforms_app:
        compose=bundle['states']['compose_transforms_app.pt']
        gui.compose_app_scale=compose['compose_app_scale'].cuda()
        gui.compose_app_rotation=compose['compose_app_rotation'].cuda()
        gui.compose_app_translation=compose['compose_app_translation'].cuda()
        gui._compose_app_idx=compose['compose_app_idx']
        if set(gui._compose_app_idx)!=set(gui.frame_indices):raise ValueError('Compose frame map mismatch')
    original_smoothing(gui,window,source)
    label=f'{sequence}-appearance-{args.checkpoint}-{uuid.uuid4().hex[:8]}'
    out=run/'evaluation'/label;out.mkdir(parents=True,exist_ok=False)
    (out/'rgb').mkdir();(out/'alpha').mkdir();(run/'dashboard_exports').mkdir(exist_ok=True)
    manifest={'schema':1,'sequence':sequence,'target_object_id':255,'camera_space':'input',
              'label':f'Input-camera RGB + alpha · appearance {args.checkpoint}',
              'stage':'appearance','checkpoint':args.checkpoint,'frames':[],
              'time_mapping':{'kind':'input_frame_identity','physical_fps':None,
                              'deformation_time_rule':'frame_index/max_frame_index'},
              'provenance':{'export_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                            'baseline_metadata':json.loads((run/'metadata.json').read_text()),
                            'checkpoint_hashes':bundle['hashes'],'model_config':cfg,
                            'trainer_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                            'smoothing_window':window,'alpha_storage':'uint8, round(alpha*255)',
                            'background':'black, same as original comparison RGB',
                            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                            'gpu':torch.cuda.get_device_name(0),'optimizer_created':False}}
    destination=run/'dashboard_exports'/f'{label}.json'
    with torch.inference_mode():
        for i,fid in enumerate(stems):
            with Image.open(Path(inventory['frames_dir'])/(fid+'.jpg')) as im:width,height=im.size
            xyz=gui._deform_at_frame(i)
            scale=gui.gaussians.get_scaling
            if gui.optimize_per_frame_compose_transforms_app:
                xyz,scale=gui.apply_compose_transform_app(xyz,i,scale)
            transform=loader.get_transform(i)
            xyz,rotation,scale=loader.apply_transform_to_gaussian(xyz,gui.gaussians.get_rotation,scale,transform)
            rgb,alpha,_=loader.render_gaussian(xyz,rotation,scale,gui.gaussians.get_opacity,
                                              gui.gaussians._features_dc.permute(0,2,1),width,height,return_depth=True)
            rgb=rgb.cpu().numpy().clip(0,255).astype(np.uint8)
            alpha=np.round(alpha.cpu().numpy().squeeze(-1).clip(0,1)*255).astype(np.uint8)
            native=model/f'comparison_iter_{args.checkpoint:06d}'/f'frame_{i:04d}.png'
            with Image.open(native) as im:composite=np.array(im)
            if composite.shape[:2]!=(2*(height+30),3*width):raise ValueError('Native comparison size differs')
            reference=composite[height+60:2*(height+30),width:2*width,:3]
            error=np.abs(rgb.astype(float)-reference.astype(float))
            if error.mean()>0.5 or error.max()>16:
                raise ValueError(f'RGB differs from saved final render: {fid}, mean={error.mean()}, max={error.max()}')
            rgbpath=out/'rgb'/f'{fid}.png';alphapath=out/'alpha'/f'{fid}.png'
            Image.fromarray(rgb).save(rgbpath);Image.fromarray(alpha).save(alphapath)
            camera=camera_description(width,height,loader.focal_length)
            manifest['frames'].append({'input_id':fid,'render_id':fid,'input_frame_index':i,'render_frame_index':i,
                                       'deformation_time':i/max(1,len(stems)-1),'input_camera':camera,'render_camera':camera,
                                       'rgb':str(rgbpath.relative_to(run)),'alpha':str(alphapath.relative_to(run)),
                                       'native_rgb_mean_absolute_error':float(error.mean()),
                                       'native_rgb_max_absolute_error':float(error.max()),
                                       'object_transform':{k:v.cpu().numpy().tolist() for k,v in transform.items()}})
            atomic_json(destination,manifest)
            print(f'{sequence}: exported {i+1}/{len(stems)}, native RGB MAE={error.mean():.6f}',flush=True)
    manifest['complete']=True;atomic_json(destination,manifest)
    print(f'Export complete: {destination}',flush=True)


if __name__=='__main__':main()
