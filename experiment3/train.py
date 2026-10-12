"""Train a fresh LIFT4D position MLP on a SAM3D mesh with geodesic controls."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from evaluation.export_input_views import atomic_json
from experiment2.mesh import snapshot
from experiment2.transfer import ROOT, render_mesh
from .geodesic import path_loss


def sha(path):
    return hashlib.sha256(snapshot(Path(path))).hexdigest()


def locate_prepared(root, sequence):
    choices=[]
    for path in Path(root).rglob('prepared.json'):
        try:
            value=json.loads(snapshot(path))
            if value.get('sequence')==sequence and path.with_name('geometry.npz').is_file() and path.with_name('tracks.npz').is_file():
                choices.append((value.get('created_utc',''),path.resolve()))
        except (OSError,ValueError,KeyError):
            continue
    if len(choices)!=1:
        raise ValueError(f'Expected one prepared {sequence} run under {root}; found {len(choices)}')
    return choices[0][1]


def load_position_mlp(torch, seed):
    sys.path.insert(0,str(ROOT/'lift4d_scgs'))
    from utils.time_utils import DeformNetwork
    torch.manual_seed(seed)
    model=DeformNetwork(is_blender=True,pred_opacity=False,pred_color=False,local_frame=False).cuda()
    # Reuse the original LIFT4D MLP trunk and position head. Rotation and scale
    # heads have no mesh supervision and are deliberately fixed.
    trainable=[]
    for name,parameter in model.named_parameters():
        active=name.startswith('linear.') or name.startswith('timenet.') or name.startswith('gaussian_warp.')
        parameter.requires_grad_(active)
        if active:trainable.append(name)
    expected=[name for name,_ in model.named_parameters() if name.startswith('linear.') or name.startswith('timenet.') or name.startswith('gaussian_warp.')]
    if trainable!=expected or not any(name.startswith('gaussian_warp.') for name in trainable):
        raise ValueError('Fresh LIFT4D position MLP parameter selection differs')
    return model,trainable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',default='auto');parser.add_argument('--runs-root',default='runs')
    parser.add_argument('--sequence',default='camel');parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=20000);parser.add_argument('--lr',type=float,default=1.6e-4)
    parser.add_argument('--checkpoint-every',type=int,default=1000);parser.add_argument('--export-every',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=31);parser.add_argument('--smoke-check',action='store_true')
    parser.add_argument('--lambda-mask',type=float,default=1.0);parser.add_argument('--lambda-track',type=float,default=1.0)
    parser.add_argument('--lambda-edge',type=float,default=1.0);parser.add_argument('--lambda-temporal',type=float,default=.01)
    parser.add_argument('--lambda-geodesic',type=float,default=1.0);parser.add_argument('--lambda-displacement',type=float,default=.001)
    args=parser.parse_args()
    if args.steps<1 or args.lr<=0 or args.checkpoint_every<1 or args.export_every<1:raise ValueError('Positive training settings required')
    for name in ['lambda_mask','lambda_track','lambda_edge','lambda_temporal','lambda_geodesic','lambda_displacement']:
        if getattr(args,name)<0:raise ValueError('Loss weights must be nonnegative')
    prepared_path=locate_prepared(args.runs_root,args.sequence) if args.prepared=='auto' else Path(args.prepared).resolve()
    prepared=json.loads(snapshot(prepared_path));sequence=prepared['sequence']
    if sequence!=args.sequence or prepared.get('schema')!=1:raise ValueError('Prepared run schema or sequence differs')
    geometry_path=prepared_path.with_name('geometry.npz');tracks_path=prepared_path.with_name('tracks.npz')
    if sha(geometry_path)!=prepared['geometry_sha256'] or sha(tracks_path)!=prepared['tracks_sha256']:
        raise ValueError('Prepared geometry or tracking observations changed')
    frames=prepared['frames'];nframes=len(frames)
    if nframes<3 or [frame['index'] for frame in frames]!=list(range(nframes)):
        raise ValueError('Contiguous explicit input frame order is required')
    times=np.asarray([f['time'] for f in frames],dtype=np.float64)
    expected_times=np.arange(nframes,dtype=np.float64)/max(1,nframes-1)
    if not np.allclose(times,expected_times,rtol=0,atol=1e-12):raise ValueError('Prepared deformation times must map exactly to input frame IDs')
    with np.load(geometry_path) as stored:geometry={key:stored[key].copy() for key in stored.files}
    with np.load(tracks_path) as stored:tracks={key:stored[key].copy() for key in stored.files}
    rest=geometry['vertices'].astype(np.float32);faces=geometry['faces'].astype(np.int32)
    edges=geometry['edges'].astype(np.int64);edge_lengths=geometry['edge_lengths'].astype(np.float32)
    node_vertices=geometry['node_vertices'].astype(np.int64);control_ids=geometry['control_indices'].astype(np.int64)
    control_weights=geometry['control_weights'].astype(np.float32);track_vertices=geometry['track_vertices'].astype(np.int64)
    query_xy=tracks['xy'].astype(np.float32);valid=tracks['valid'].astype(bool)
    if control_ids.shape!=(len(rest),3) or control_weights.shape!=control_ids.shape:raise ValueError('Expected fixed geodesic K=3 mesh influences')
    if query_xy.shape!=(nframes,len(track_vertices),2) or valid.shape!=query_xy.shape[:2]:raise ValueError('Track arrays do not match fixed canonical query identities')
    if not valid.any(axis=1).all():raise ValueError('Every training frame needs real cycle-checked, geodesic-gated track observations')
    if np.any(control_ids<0) or np.any(control_ids>=len(node_vertices)) or np.any(track_vertices<0) or np.any(track_vertices>=len(rest)):
        raise ValueError('Prepared canonical vertex index out of range')
    if not np.isfinite(control_weights).all() or np.any(control_weights<0) or not np.allclose(control_weights.sum(1),1,atol=1e-5):
        raise ValueError('Geodesic control weights must be finite and normalized')

    import torch
    import torch.nn.functional as F
    import nvdiffrast.torch as dr
    torch.set_num_threads(2);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    device='cuda';model,trainable=load_position_mlp(torch,args.seed);model.eval()
    node_xyz=torch.as_tensor(rest[node_vertices],device=device)
    canonical=torch.as_tensor(rest,device=device)
    faces_gpu=torch.as_tensor(faces,dtype=torch.int32,device=device).contiguous()
    colors=torch.as_tensor(geometry['colors'],dtype=torch.float32,device=device)
    edge_index=torch.as_tensor(edges,dtype=torch.long,device=device)
    edge_rest=torch.as_tensor(edge_lengths,dtype=torch.float32,device=device)
    control_index=torch.as_tensor(control_ids,dtype=torch.long,device=device)
    control_w=torch.as_tensor(control_weights,dtype=torch.float32,device=device)
    query_index=torch.as_tensor(track_vertices,dtype=torch.long,device=device)
    track_xy=torch.as_tensor(query_xy,dtype=torch.float32,device=device)
    track_valid=torch.as_tensor(valid,dtype=torch.bool,device=device)
    path_edges=torch.as_tensor(geometry['path_edge_ids'],dtype=torch.long,device=device)
    path_pairs=torch.as_tensor(geometry['path_pair_ids'],dtype=torch.long,device=device)
    path_rest=torch.as_tensor(geometry['rest_path_lengths'],dtype=torch.float32,device=device)
    scale=float(np.linalg.norm(np.ptp(rest,axis=0)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate SAM3D canonical mesh')
    renderer=dr.RasterizeCudaContext()
    targets=[];frame_cameras=[];inputs={}
    for frame in frames:
        image_path=Path(frame['image_path']);mask_path=Path(frame['mask_path'])
        if sha(image_path)!=frame['image_sha256'] or sha(mask_path)!=frame['mask_sha256']:raise ValueError('DAVIS frame or mask changed after tracking preparation')
        with Image.open(image_path) as image:shape=(image.height,image.width)
        camera=frame['camera']
        if shape!=(camera['height'],camera['width']):raise ValueError('Native input resolution differs from prepared camera')
        with Image.open(mask_path) as image:mask=np.asarray(image)
        if mask.shape!=shape or not np.any(mask==255):raise ValueError('Invalid native DAVIS mask')
        targets.append(torch.as_tensor(mask==255,dtype=torch.float32,device=device))
        frame_cameras.append(camera)
        inputs[frame['id']]={'rgb':frame['image_sha256'],'mask':frame['mask_sha256']}
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr,eps=1e-15)
    lr_final=args.lr*.01;delay=.01;steps=1 if args.smoke_check else args.steps
    def update_lr(step):
        ratio=min(1.,max(0.,step/max(1,steps)))
        lr=lr_final+(args.lr-lr_final)*np.exp(np.log(delay)*(1-ratio))
        for group in optimizer.param_groups:group['lr']=float(lr)
        return float(lr)
    def node_delta(t):
        current=model(node_xyz,torch.full((len(node_xyz),1),float(t),device=device))['d_xyz']
        rest_pose=model(node_xyz,torch.zeros((len(node_xyz),1),device=device))['d_xyz']
        return current-rest_pose
    def deform(frame_index):
        if frame_index==0:return canonical
        delta=node_delta(float(times[frame_index]))
        return canonical+(delta[control_index]*control_w[...,None]).sum(1)
    def camera_xyz(vertices,index):
        frame=frames[index]
        linear=torch.as_tensor(frame['linear'],dtype=vertices.dtype,device=device)
        translation=torch.as_tensor(frame['translation'],dtype=vertices.dtype,device=device)
        return vertices@linear.T+translation
    def pixels(vertices,index):
        q=camera_xyz(vertices,index);x,y,z=q.unbind(-1);cam=frame_cameras[index]
        safe_z=z.clamp_min(.001)
        return torch.stack((cam['fx']*x/safe_z+cam['cx'],cam['fy']*y/safe_z+cam['cy']),dim=-1),z
    def render(vertices,index):
        cam=frame_cameras[index];height,width=cam['height'],cam['width']
        q=camera_xyz(vertices,index);x,y,z=q.unbind(-1);near,far=.001,1000.
        clip=torch.stack((2*cam['fx']/width*x+(2*cam['cx']/width-1)*z,
            -2*cam['fy']/height*y-(2*cam['cy']/height-1)*z,
            (far+near)/(far-near)*z-2*far*near/(far-near),z),dim=-1).contiguous()
        rast,_=dr.rasterize(renderer,clip[None],faces_gpu,resolution=[height,width])
        rgb,_=dr.interpolate(colors[None],rast,faces_gpu);alpha=(rast[...,3:4]>0).float()
        return dr.antialias(torch.cat((rgb*alpha,alpha),dim=-1),rast,clip[None],faces_gpu)[0].flip(0).clamp(0,1)
    def losses(index,vertices):
        rgba=render(vertices,index);target=targets[index]
        alpha=rgba[...,3];intersection=(alpha*target).sum();union=(alpha+target-alpha*target).sum()
        silhouette=1-(intersection+1e-6)/(union+1e-6)
        active=track_valid[index];prediction,z=pixels(vertices[query_index[active]],index)
        observation=track_xy[index,active];camera=frame_cameras[index];diagonal=float(np.hypot(camera['height'],camera['width']))
        track=F.smooth_l1_loss(prediction/diagonal,observation/diagonal,beta=3./diagonal)
        depth=torch.relu(.001-z).square().mean()/(scale*scale)
        length=(vertices[edge_index[:,0]]-vertices[edge_index[:,1]]).norm(dim=-1)
        edge=((length-edge_rest)/scale).square().mean()
        geodesic=path_loss(vertices,edge_index,path_edges,path_pairs,path_rest)
        displacement=((vertices-canonical)/scale).square().sum(-1).mean()
        if 0<index<nframes-1:
            prev=node_delta(float(times[index-1]));current=node_delta(float(times[index]));next_=node_delta(float(times[index+1]))
            dt0=float(times[index]-times[index-1]);dt1=float(times[index+1]-times[index])
            acceleration=2*((next_-current)/dt1-(current-prev)/dt0)/(dt0+dt1)
            temporal=(acceleration*(1./max(1,nframes-1))**2).square().sum(-1).mean()/(scale*scale)
        else:temporal=vertices.sum()*0
        values={'mask':silhouette,'track':track,'edge':edge,'temporal':temporal,'geodesic':geodesic,'displacement':displacement,'depth':depth}
        total=(args.lambda_mask*silhouette+args.lambda_track*track+args.lambda_edge*edge+
               args.lambda_temporal*temporal+args.lambda_geodesic*geodesic+
               args.lambda_displacement*displacement+100.*depth)
        return total,values,rgba

    out=Path(args.output).resolve()
    if out.exists():raise ValueError('Training output must be a new run directory')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['logs','checkpoints','dashboard_exports','renders','mesh']:(out/name).mkdir()
    seed=Path(prepared['seed_run']);seed_config=json.loads(snapshot(seed/'config.json'))
    if seed_config.get('experiment')!='2.1' or seed_config.get('sequence')!=sequence:raise ValueError('Training must start from the matching frozen SAM3D 2.1 mesh')
    canonical_path=seed/'mesh/canonical.npz'
    if sha(canonical_path)!=prepared['source_canonical_sha256']:raise ValueError('Source SAM3D canonical mesh differs from tracking preparation')
    with np.load(canonical_path) as source_mesh:
        for name in ['vertices','faces','colors']:
            if not np.array_equal(source_mesh[name],geometry[name]):raise ValueError('Prepared geometry differs from source SAM3D '+name)
    metadata={'experiment':'3','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'selected_objects':[sequence],'display_label':f'Fresh LIFT4D MLP · SAM3D · geodesic tracks · seed {args.seed}',
        'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    settings={'experiment':'3','sequence':sequence,'source':'SAM3D canonical textured mesh',
        'seed_run':str(seed),'baseline_run':prepared['baseline_run'],'camera_source_checkpoint_hashes':seed_config['checkpoint_hashes'],
        'prepared_path':str(prepared_path),'prepared_sha256':sha(prepared_path),'geometry_sha256':prepared['geometry_sha256'],
        'tracks_sha256':prepared['tracks_sha256'],'source_sam3d_glb_sha256':prepared['sam3d_glb_sha256'],
        'network':'Original LIFT4D DeformNetwork; is_blender=True; random from scratch; position MLP trunk + d_xyz head trainable',
        'frozen_outputs':['Gaussian rotation head','Gaussian scale head','opacity/color heads absent'],
        'node_count':int(len(node_vertices)),'control_k':3,'control_policy':prepared['control_policy'],
        'neighbor_policy':prepared['neighbor_policy'],'geodesic_gate':{'stretch':prepared['maximum_neighbor_stretch'],'depth_assumption':prepared['neighbor_depth_assumption']},
        'iterations':steps,'requested_iterations':args.steps,'learning_rate':args.lr,'lr_final':lr_final,'lr_delay_multiplier':delay,
        'optimizer':'Adam eps=1e-15 over freshly initialized LIFT4D MLP position trunk/head only',
        'losses':{'mask':'1 minus soft IoU of differentiable nvdiffrast alpha and DAVIS label 255',
            'track':'Smooth-L1 of cycle-checked, DAVIS-contained, geodesic-gated fixed material vertices, normalized by image diagonal (3px beta)',
            'edge':'mean squared canonical triangle-edge length strain, divided by canonical diagonal squared',
            'temporal':'nonuniform-time second derivative of control-node displacement, normalized by canonical scale',
            'geodesic':'mean squared stretch of fixed canonical shortest edge paths between track-neighbor vertices',
            'displacement':'mean squared mesh displacement from SAM3D canonical, divided by canonical diagonal squared',
            'depth':'camera-space z barrier for observed track points'},
        'weights':{'mask':args.lambda_mask,'track':args.lambda_track,'edge':args.lambda_edge,'temporal':args.lambda_temporal,
            'geodesic':args.lambda_geodesic,'displacement':args.lambda_displacement,'depth':100.},
        'training_sampling':'uniform random permutation of all input frames, repeated without interpolation of hidden tracks',
        'frame_count':nframes,'track_query_count':len(track_vertices),'valid_track_observations':int(valid.sum()),
        'valid_tracks_per_frame':valid.sum(axis=1).tolist(),'target_object_id':255,'alpha_threshold':.5,
        'target_mask_source':prepared['frames'][0]['mask_path'],'camera_policy':'exact prepared frozen original input camera affine; no camera fitting',
        'time_mapping':{'kind':'input_frame_identity','physical_fps':None,'deformation_time_rule':'frame_index/max_frame_index'},
        'network_initialization_seed':args.seed,'network_initial_state_sha256':None,
        'trainable_parameters':trainable,'input_hashes':inputs,'smoke_check':args.smoke_check,
        'implementation_sha256':sha(__file__),'network_source_sha256':sha(ROOT/'lift4d_scgs/utils/time_utils.py'),
        'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    state0={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    state_digest=hashlib.sha256(b''.join(k.encode()+v.contiguous().numpy().tobytes() for k,v in sorted(state0.items()))).hexdigest()
    settings['network_initial_state_sha256']=state_digest
    atomic_json(out/'config.json',settings);atomic_json(out/'inventory.json',{sequence:prepared['inventory'][sequence]})
    atomic_json(out/'metadata.json',metadata)
    rng=np.random.default_rng(args.seed);frame_order=[];start=time.monotonic()
    def save_checkpoint(step):
        path=out/'checkpoints'/f'deformation-{step:06d}.pt';tmp=path.with_suffix('.pt.tmp')
        torch.save({'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'settings':settings,
            'seed':args.seed,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'numpy_rng':rng.bit_generator.state,'frame_order':frame_order},tmp)
        os.replace(tmp,path)
    def export(step):
        directory=out/'renders'/f'{step:06d}'
        for name in ['rgb','alpha','vertices']:(directory/name).mkdir(parents=True,exist_ok=False)
        manifest={'schema':1,'sequence':sequence,'target_object_id':255,'camera_space':'input',
            'label':f'Experiment 3 · fresh SAM3D mesh deformation · step {step:,}',
            'stage':'mesh_deformation','checkpoint':step,'frames':[],
            'time_mapping':settings['time_mapping'],'provenance':{'config':'config.json','checkpoint':str((out/'checkpoints'/f'deformation-{step:06d}.pt').relative_to(out)),
                'prepared':str(prepared_path)}}
        destination=out/'dashboard_exports'/f'mesh-{step:06d}.json'
        with torch.no_grad():
            for index,frame in enumerate(frames):
                q=deform(index);rgba=render(q,index).cpu().numpy()
                np.savez_compressed(directory/'vertices'/f'{frame["id"]}.npz',vertices=q.cpu().numpy())
                rgb=directory/'rgb'/f'{frame["id"]}.png';alpha=directory/'alpha'/f'{frame["id"]}.png'
                Image.fromarray(np.round(rgba[:,:,:3]*255).astype(np.uint8)).save(rgb)
                Image.fromarray(np.round(rgba[:,:,3]*255).astype(np.uint8)).save(alpha)
                frame_time=float(times[index])
                manifest['frames'].append({'input_id':frame['id'],'render_id':frame['id'],'input_frame_index':index,
                    'render_frame_index':index,'deformation_time':frame_time,'input_camera':frame['camera'],
                    'render_camera':frame['camera'],'input_time_seconds':None,'render_time_seconds':None,
                    'rgb':str(rgb.relative_to(out)),'alpha':str(alpha.relative_to(out))})
                atomic_json(destination,manifest)
                print(f'export checkpoint {step}: {index+1}/{nframes}',flush=True)
        manifest['complete']=True;atomic_json(destination,manifest)
        print(f'Exported dashboard checkpoint {step}',flush=True)

    steps_to_run=range(1,steps+1)
    previous=[]
    with (out/'logs/losses.jsonl').open('w') as log:
        for step in steps_to_run:
            if not frame_order:frame_order=rng.permutation(nframes).tolist()
            index=frame_order.pop();optimizer.zero_grad(set_to_none=True);lr=update_lr(step-1)
            q=deform(index);total,values,_=losses(index,q)
            if not torch.isfinite(total) or any(not torch.isfinite(x) for x in values.values()):raise ValueError('Nonfinite mesh deformation loss')
            if step==1:
                grads=torch.autograd.grad(total,[p for p in model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
                norm=torch.sqrt(sum((g.square().sum() for g in grads if g is not None),torch.zeros((),device=device)))
                if not torch.isfinite(norm) or norm<=0:raise ValueError('Fresh MLP received no finite mesh training gradient')
                head_grad=next((g for name,g in zip(trainable,grads) if name=='gaussian_warp.weight' and g is not None),None)
                if head_grad is None or not torch.isfinite(head_grad).all() or head_grad.abs().max()<=0:
                    raise ValueError('Mesh losses do not provide a real position-head gradient')
                atomic_json(out/'gradient-check.json',{'total_gradient_norm':float(norm),'position_head_gradient':float(head_grad.norm()),
                    'frame':frames[index]['id'],'valid_track_count':int(track_valid[index].sum()),'network_initial_state_sha256':state_digest})
            total.backward()
            params=[p for p in model.parameters() if p.requires_grad]
            grad_norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(grad_norm):raise ValueError('Nonfinite fresh-network gradients')
            optimizer.step()
            with torch.no_grad():
                current={k:v.detach().cpu() for k,v in model.state_dict().items()}
                if all(torch.equal(state0[k],current[k]) for k in trainable):raise ValueError('Fresh LIFT4D position MLP did not update')
            row={'step':step,'frame':frames[index]['id'],'lr':lr,'total':float(total.detach()),
                'gradient_norm':float(grad_norm),**{key:float(value.detach()) for key,value in values.items()}}
            log.write(json.dumps(row)+'\n');log.flush()
            if step==1 or step%25==0:print(json.dumps(row),flush=True)
            if step%args.checkpoint_every==0 or step==steps:
                saved=save_checkpoint(step)
                atomic_json(out/'training-status.json',{'complete':False,'step':step,'total_steps':steps,
                    'checkpoint':str(saved.relative_to(out)),'elapsed_seconds':time.monotonic()-start,
                    'network_initial_state_sha256':state_digest})
                if not args.smoke_check and (step%args.export_every==0 or step==steps):export(step)
    if args.smoke_check:
        final={k:v.detach().cpu() for k,v in model.state_dict().items()}
        changed=[name for name in trainable if not torch.equal(state0[name],final[name])]
        if not changed:raise ValueError('CUDA smoke step left fresh MLP unchanged')
        atomic_json(out/'training-status.json',{'complete':True,'smoke_check':True,'steps':steps,'changed_trainable_parameters':changed})
        print(f'CUDA smoke complete: {out}',flush=True);return
    if not np.isfinite(np.asarray([float(x) for x in values.values()])).all():raise ValueError('Final losses are nonfinite')
    atomic_json(out/'training-status.json',{'complete':True,'smoke_check':False,'steps':steps,
        'final_checkpoint':f'checkpoints/deformation-{steps:06d}.pt','final_manifest':f'dashboard_exports/mesh-{steps:06d}.json',
        'elapsed_seconds':time.monotonic()-start,'network_initial_state_sha256':state_digest,
        'camera_source_checkpoint_hashes':seed_config['checkpoint_hashes'],'prepared_sha256':sha(prepared_path)})
    print(f'Experiment 3 training complete: {out}',flush=True)


if __name__=='__main__':main()
