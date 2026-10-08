"""Frozen baseline deformation on canonical SAM3D mesh; no training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from evaluation.export_input_views import ROOT, atomic_json, camera_description, namespace_file, original_smoothing, restore
from experiment2.checkpoints import load_bundle
from experiment2.mesh import snapshot

GLB_FROM_SOURCE = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float32)


def nearest_features(vertices, gaussian_xyz, features):
    if not np.isfinite(vertices).all() or not np.isfinite(gaussian_xyz).all():
        raise ValueError('Nonfinite canonical coordinates')
    distances, indices = cKDTree(gaussian_xyz).query(vertices, k=1, workers=2)
    return indices, distances, features[indices]


def render_mesh(vertices, faces, colors, camera, context):
    import torch
    import nvdiffrast.torch as dr
    width,height=camera['width'],camera['height']
    z=vertices[:,2]; near,far=0.001,1000.0
    clip=torch.stack((2*camera['fx']/width*vertices[:,0]+(2*camera['cx']/width-1)*z,
                      -2*camera['fy']/height*vertices[:,1]-(2*camera['cy']/height-1)*z,
                      (far+near)/(far-near)*z-2*far*near/(far-near), z),dim=-1).contiguous()
    rast,_=dr.rasterize(context,clip[None],faces,resolution=[height,width])
    rgb,_=dr.interpolate(colors[None],rast,faces)
    alpha=(rast[...,3:4]>0).float()
    rgba=dr.antialias(torch.cat((rgb*alpha,alpha),dim=-1),rast,clip[None],faces)
    return rgba[0].flip(0).clamp(0,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--sequence',required=True)
    p.add_argument('--variant',choices=['2.1','2.2'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();run=Path(args.run).resolve();out=Path(args.output).resolve();sequence=args.sequence
    stage,iteration=('appearance',30000) if args.variant=='2.1' else ('geometry',20000)
    bundle=load_bundle(run,sequence,stage,iteration)
    inventory=json.loads(snapshot(run/'inventory.json'));stems=inventory[sequence]['frames']
    if stems!=[f'{i:05d}' for i in range(len(stems))]:raise ValueError('Input frame IDs must be contiguous from zero')
    os.environ.update(json.loads(snapshot(run/'environment.json')))
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    if torch.cuda.device_count()!=1:raise ValueError('Exactly one visible CUDA GPU is required')
    sys.path.insert(0,str(ROOT/'lift4d_scgs'))
    from train_lift4d_scgs import GUI,SAM3DDataLoader,GaussianModel,DeformModel
    cfg=namespace_file(Path(bundle['checkpoint']['path']).parents[1]/'cfg_args')
    config=json.loads(snapshot(run/'config.json'));flags=config['appearance_args'] if stage=='appearance' else config['geometry_args']
    loader=SAM3DDataLoader(selector=sequence,canonical_frame_idx=0,device='cuda',dataset='davis',object_name=sequence,input_dir=str(run/'sam3d'/f'davis_{sequence}'))
    if loader.get_frame_indices()!=list(range(len(stems))):raise ValueError('Loader frame IDs mismatch')
    gui=GUI.__new__(GUI);gui.sam3d_data=loader;gui.frame_indices=loader.get_frame_indices();gui.num_frames=loader.num_frames
    gui.iteration=iteration;gui.is_node_delta=stage=='appearance'
    gaussians=GaussianModel(sh_degree=0,fea_dim=cfg['hyper_dim'],with_motion_mask=False)
    gaussians.load_ply(str(Path(bundle['checkpoint']['path'])/'gaussians.ply'))
    kwargs={k:cfg[k] for k in ['K','skinning','hyper_dim','use_hash','hash_time','local_frame','progressive_brand_time']}
    base=bundle['states']['deform_node_base.pth' if stage=='appearance' else 'deform.pth']
    node=DeformModel(deform_type='node',is_blender=True,d_rot_as_res=True,node_num=base['nodes'].shape[0],with_arap_loss=True,pred_opacity=False,pred_color=False,max_d_scale=-1,enable_densify_prune=False,is_scene_static=False,**kwargs)
    restore(node.deform,base)
    if stage=='appearance':
        gui.deform_node_base=node;delta=bundle['states']['deform.pth']
        gui.deform=DeformModel(deform_type='delta',is_blender=True,d_rot_as_res=True,K=cfg['K'],node_num=delta['nodes'].shape[0],num_frames=delta['control_point_deltas'].shape[0],with_arap_loss=True,opt_deform_rot='--opt_deform_rot' in flags)
        restore(gui.deform.deform,delta);gui.deform.deform.set_frame_mapping(gui.frame_indices)
    else:gui.deform=node
    gui.optimize_per_frame_compose_transforms_app=stage=='appearance' and '--optimize_per_frame_compose_transforms_app' in flags
    if gui.optimize_per_frame_compose_transforms_app:
        compose=bundle['states']['compose_transforms_app.pt']
        for name in ['compose_app_scale','compose_app_rotation','compose_app_translation']:setattr(gui,name,compose[name].cuda())
        gui._compose_app_idx=compose['compose_app_idx']
        if set(gui._compose_app_idx)!=set(gui.frame_indices):raise ValueError('Compose frame IDs mismatch')
    window=int(flags[flags.index('--smooth_transforms_window')+1]) if '--smooth_transforms_window' in flags else 0
    original_smoothing(gui,window,ROOT/'lift4d_scgs/train_lift4d_scgs.py')
    meshpath=run/'sam3d'/f'davis_{sequence}'/'00000/result.glb';meshbytes=snapshot(meshpath)
    import io
    scene=trimesh.load(io.BytesIO(meshbytes),file_type='glb',force='scene',process=False)
    vertices=[];faces=[];colors=[];offset=0
    for name in sorted(scene.graph.nodes_geometry):
        transform,key=scene.graph[name];mesh=scene.geometry[key]
        if mesh.visual.kind!='vertex':raise ValueError('Only saved vertex colors supported; no substitute appearance')
        v=trimesh.transform_points(mesh.vertices,transform)@GLB_FROM_SOURCE.T
        vertices.append(v);faces.append(mesh.faces+offset);colors.append(mesh.visual.vertex_colors[:,:3]/255.0);offset+=len(v)
    vertices=np.concatenate(vertices).astype(np.float32);faces=np.concatenate(faces).astype(np.int32);colors=np.concatenate(colors).astype(np.float32)
    indices,distances,features=nearest_features(vertices,gaussians.get_xyz.detach().cpu().numpy(),gaussians.feature.detach().cpu().numpy())
    # Minimal adapter lets the exact original no-grad helper evaluate vertices.
    from types import SimpleNamespace
    gui.gaussians=SimpleNamespace(get_xyz=torch.from_numpy(vertices).cuda(),feature=torch.from_numpy(features).cuda())
    if out==run or out.is_relative_to(run):raise ValueError('Experiment outputs must be separate from source baseline')
    out.mkdir(parents=True,exist_ok=False)
    for directory in ['mesh','rgb','alpha','dashboard_exports']:(out/directory).mkdir()
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    settings={'experiment':args.variant,'sequence':sequence,'baseline_run':str(run),'stage':stage,'checkpoint':iteration,'canonical_frame':'00000','feature_mapping':'nearest saved canonical Gaussian in Euclidean source XYZ, fixed across time','coordinate_conversion':'apply GLB scene instance transform then invert SAM3D Z-up to Y-up export matrix; no alignment/normalization','smoothing_window':window,'compose_policy':'apply original appearance compose only to camera render; mesh arrays remain deformed object space','renderer':'nvdiffrast CUDA, saved vertex colors, no lighting, black background, antialiased alpha','alpha_threshold':0.5,'target_object_id':255,'target_mask_source':inventory[sequence]['masks_dir'],'optimizer_created':False,'transfer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'trainer_source_sha256':hashlib.sha256((ROOT/'lift4d_scgs/train_lift4d_scgs.py').read_bytes()).hexdigest(),'checkpoint_hashes':bundle['hashes'],'mesh_sha256':hashlib.sha256(meshbytes).hexdigest(),'model_config':cfg,'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    atomic_json(out/'config.json',settings);atomic_json(out/'inventory.json',{sequence:inventory[sequence]})
    atomic_json(out/'metadata.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'selected_objects':[sequence],'baseline_metadata':json.loads(snapshot(run/'metadata.json')),'experiment':args.variant})
    np.savez(out/'mesh/canonical.npz',vertices=vertices,faces=faces,colors=colors,gaussian_indices=indices,nearest_distances=distances,features=features)
    manifest={'schema':1,'sequence':sequence,'target_object_id':255,'camera_space':'input','label':f'Experiment {args.variant} · SAM3D mesh · {stage} {iteration}','stage':stage,'checkpoint':iteration,'frames':[],'time_mapping':{'kind':'input_frame_identity','physical_fps':None,'deformation_time_rule':'frame_index/max_frame_index'},'provenance':settings}
    context=dr.RasterizeCudaContext()
    faces_gpu=torch.from_numpy(faces).cuda().contiguous();colors_gpu=torch.from_numpy(colors).cuda()
    with torch.inference_mode():
        for i,fid in enumerate(stems):
            xyz=gui._deform_at_frame(i)
            if not torch.isfinite(xyz).all():raise ValueError('Nonfinite deformed vertices')
            np.savez(out/'mesh'/f'{fid}.npz',vertices=xyz.cpu().numpy())
            camera_xyz=xyz
            if gui.optimize_per_frame_compose_transforms_app:camera_xyz,_=gui.apply_compose_transform_app(camera_xyz,i,torch.ones_like(xyz))
            transform=loader.get_transform(i)
            rotations=torch.zeros((len(xyz),4),device='cuda');rotations[:,0]=1
            camera_xyz,_,_=loader.apply_transform_to_gaussian(camera_xyz,rotations,torch.ones_like(xyz),transform)
            with Image.open(Path(inventory[sequence]['frames_dir'])/(fid+'.jpg')) as im:width,height=im.size
            camera=camera_description(width,height,loader.focal_length)
            rgba=render_mesh(camera_xyz,faces_gpu,colors_gpu,camera,context).cpu().numpy()
            Image.fromarray(np.round(rgba[:,:,:3]*255).astype(np.uint8)).save(out/'rgb'/f'{fid}.png')
            Image.fromarray(np.round(rgba[:,:,3]*255).astype(np.uint8)).save(out/'alpha'/f'{fid}.png')
            manifest['frames'].append({'input_id':fid,'render_id':fid,'input_frame_index':i,'render_frame_index':i,'input_camera':camera,'render_camera':camera,'deformation_time':i/max(1,len(stems)-1),'rgb':f'rgb/{fid}.png','alpha':f'alpha/{fid}.png','object_transform':{k:v.cpu().numpy().tolist() for k,v in transform.items()}})
            atomic_json(out/'dashboard_exports/mesh.json',manifest)
            print(f'{args.variant} {sequence}: {i+1}/{len(stems)}',flush=True)
    manifest['complete']=True;atomic_json(out/'dashboard_exports/mesh.json',manifest)
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':main()
