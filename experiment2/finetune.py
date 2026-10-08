"""Experiment 2.3: mesh-mask fine-tuning of saved node_delta translations only."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
from PIL import Image
from experiment2.transfer import ROOT, load_transfer, render_mesh
from evaluation.export_input_views import atomic_json, camera_description
from experiment2.mesh import snapshot


def mask_loss(alpha, target):
    intersection=(alpha*target).sum()
    union=(alpha+target-alpha*target).sum()
    return 1-(intersection+1e-6)/(union+1e-6)


def spatial_loss(xyz, canonical, edges, scale):
    displacement=xyz-canonical
    return ((displacement[edges[:,0]]-displacement[edges[:,1]])/scale).square().sum(-1).mean()


def temporal_loss(residual, frame, scale):
    if len(residual)<3:return residual.sum()*0
    center=min(max(frame,1),len(residual)-2)
    return ((residual[center-1]-2*residual[center]+residual[center+1])/scale).square().sum(-1).mean()


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--seed-run',required=True)
    p.add_argument('--sequence',default='camel');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=23);p.add_argument('--smoke-check',action='store_true')
    args=p.parse_args()
    if args.steps<1 or args.lr<=0:raise ValueError('Positive steps and learning rate required')
    run=Path(args.run).resolve();seedrun=Path(args.seed_run).resolve();out=Path(args.output).resolve()
    if out.exists() or out.is_relative_to(run) or out.is_relative_to(seedrun):raise ValueError('A separate new output directory is required')
    seedcfg=json.loads(snapshot(seedrun/'config.json'))
    if seedcfg['experiment']!='2.1' or seedcfg['sequence']!=args.sequence or Path(seedcfg['baseline_run'])!=run:raise ValueError('Seed run must be experiment 2.1 of this baseline and sequence')
    context=load_transfer(run,args.sequence,'appearance',30000)
    import torch
    import nvdiffrast.torch as dr
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    gui=context['gui'];vertices=context['vertices'];faces=context['faces'];colors=context['colors'];stems=context['stems']
    if context['bundle']['hashes']!=seedcfg['checkpoint_hashes']:raise ValueError('Seed checkpoint provenance mismatch')
    with np.load(seedrun/'mesh/canonical.npz') as seedmesh:
        for key in ['vertices','faces','colors','features','gaussian_indices']:
            value=context['indices'] if key=='gaussian_indices' else context[key]
            if not np.array_equal(seedmesh[key],value):raise ValueError('Seed mesh/feature assignment differs: '+key)
    # Unwrap the original helper's no-grad decorator; frozen base remains no-grad.
    def deform(frame):return gui._deform_at_frame.__wrapped__(gui,frame)
    for frame in [0,len(stems)//2,len(stems)-1]:
        with torch.no_grad():
            original=gui._deform_at_frame(frame);differentiable=deform(frame)
        if not torch.equal(original,differentiable):raise ValueError('Differentiable helper differs from baseline')
        with np.load(seedrun/'mesh'/f'{stems[frame]}.npz') as saved:
            if not np.array_equal(original.cpu().numpy(),saved['vertices']):raise ValueError('Seed deformation does not reproduce saved experiment 2.1')
    model=gui.deform.deform
    parameter=model.control_point_deltas
    parameter.requires_grad_(True)
    # Keep eval mode: use explicit mesh regularizers, not Gaussian training losses.
    model.eval()
    trainable=[name for name,value in model.named_parameters() if value.requires_grad]
    if trainable!=['control_point_deltas']:raise ValueError('Unexpected trainable tensors: '+repr(trainable))
    frozen={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name!='control_point_deltas'}
    frozen_base={name:value.detach().cpu().clone() for name,value in gui.deform_node_base.deform.state_dict().items()}
    initial=parameter.detach().clone()
    optimizer=torch.optim.Adam([parameter],lr=args.lr)
    all_edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    rng=np.random.default_rng(args.seed)
    edges=all_edges[rng.choice(len(all_edges),min(20000,len(all_edges)),replace=False)]
    edges_gpu=torch.tensor(edges,dtype=torch.long,device='cuda')
    scale=float(np.linalg.norm(vertices.max(0)-vertices.min(0)))
    if scale<=0:raise ValueError('Degenerate canonical mesh bounds')
    canonical=gui.gaussians.get_xyz
    faces_gpu=torch.tensor(faces,dtype=torch.int32,device='cuda');colors_gpu=torch.tensor(colors,device='cuda')
    renderer=dr.RasterizeCudaContext()
    inventory=context['inventory'][args.sequence]
    targets=[];cameras=[];input_hashes={}
    for fid in stems:
        imagepath=Path(inventory['frames_dir'])/(fid+'.jpg');maskpath=Path(inventory['masks_dir'])/(fid+'.png')
        with Image.open(imagepath) as im:width,height=im.size
        with Image.open(maskpath) as im:mask=np.array(im)
        if mask.ndim!=2 or mask.shape!=(height,width):raise ValueError('Target mask dimensions differ from input')
        target=(mask==255).astype(np.float32)
        if not target.any():raise ValueError('Empty target object mask: '+fid)
        targets.append(torch.from_numpy(target).cuda());cameras.append(camera_description(width,height,gui.sam3d_data.focal_length))
        input_hashes[fid]={'rgb':file_hash(imagepath),'mask':file_hash(maskpath)}
    rotations=torch.zeros((len(vertices),4),device='cuda');rotations[:,0]=1
    scaling=torch.ones_like(canonical)
    def render(xyz,frame):
        if gui.optimize_per_frame_compose_transforms_app:xyz,_=gui.apply_compose_transform_app(xyz,frame,scaling)
        xyz,_,_=gui.sam3d_data.apply_transform_to_gaussian(xyz,rotations,scaling,gui.sam3d_data.get_transform(frame))
        return render_mesh(xyz,faces_gpu,colors_gpu,cameras[frame],renderer)
    out.mkdir(parents=True)
    for folder in ['logs','checkpoints','mesh','dashboard_exports']:(out/folder).mkdir()
    np.savez(out/'mesh/canonical.npz',vertices=vertices,faces=faces,colors=colors,features=context['features'],gaussian_indices=context['indices'],nearest_distances=context['distances'],regularization_edges=edges)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    settings={'experiment':'2.3','sequence':args.sequence,'baseline_run':str(run),'seed_run':str(seedrun),'initial_checkpoint':30000,'steps':args.steps,'learning_rate':args.lr,'seed':args.seed,'optimizer':'Adam, translations only','trainable_parameters':trainable,'frozen':['node base','delta nodes/radii/rotations','canonical mesh/topology/colors/features','SAM3D smoothed transforms','appearance compose','camera intrinsics'],'losses':{'mask':'1-soft IoU, all input pixels, DAVIS label 255','spatial':'mean squared difference of neighboring vertex displacements divided by canonical bounding-box diagonal; 20000 fixed sampled mesh edges','temporal':'mean squared second difference of delta correction from initial checkpoint divided by same diagonal; neighboring frame rows','anchor':'mean squared delta correction divided by same diagonal'},'weights':{'mask':1.,'spatial':.01,'temporal':.01,'anchor':.001},'sampling':'shuffled full frame permutations, one frame per optimizer step','evaluation':'fit quality on training frames; no held-out generalization claim','target_object_id':255,'target_mask_source':inventory['masks_dir'],'alpha_threshold':.5,'seed_provenance':seedcfg,'checkpoint_hashes':context['bundle']['hashes'],'input_hashes':input_hashes,'implementation_sha256':file_hash(__file__),'transfer_sha256':file_hash(ROOT/'experiment2/transfer.py'),'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'smoke_check':args.smoke_check}
    atomic_json(out/'config.json',settings);atomic_json(out/'inventory.json',{args.sequence:inventory})
    atomic_json(out/'metadata.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'experiment':'2.3','selected_objects':[args.sequence],'baseline_metadata':json.loads(snapshot(run/'metadata.json'))})
    def checkpoint(step):
        path=out/'checkpoints'/f'delta-{step:06d}.pt'
        temporary=path.with_suffix('.pt.tmp')
        torch.save({'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'initial_control_point_deltas':initial,'settings':settings},temporary);os.replace(temporary,path)
        return path
    def export(step):
        directory=out/f'render-{step:06d}'
        for folder in ['rgb','alpha','vertices']:(directory/folder).mkdir(parents=True,exist_ok=False)
        manifest={'schema':1,'sequence':args.sequence,'target_object_id':255,'camera_space':'input','label':f'Experiment 2.3 · fine-tuned mesh · step {step}','stage':'mesh_finetune','checkpoint':step,'frames':[],'time_mapping':{'kind':'input_frame_identity','physical_fps':None,'deformation_time_rule':'frame_index/max_frame_index'},'provenance':{'settings':settings,'checkpoint_file':str(out/'checkpoints'/f'delta-{step:06d}.pt')}}
        destination=out/'dashboard_exports'/f'mesh-{step:06d}.json'
        with torch.no_grad():
            for i,fid in enumerate(stems):
                xyz=deform(i);rgba=render(xyz,i).cpu().numpy()
                np.savez(directory/'vertices'/f'{fid}.npz',vertices=xyz.cpu().numpy())
                rgb=directory/'rgb'/f'{fid}.png';alpha=directory/'alpha'/f'{fid}.png'
                Image.fromarray(np.round(rgba[:,:,:3]*255).astype(np.uint8)).save(rgb)
                Image.fromarray(np.round(rgba[:,:,3]*255).astype(np.uint8)).save(alpha)
                manifest['frames'].append({'input_id':fid,'render_id':fid,'input_frame_index':i,'render_frame_index':i,'input_camera':cameras[i],'render_camera':cameras[i],'deformation_time':i/max(1,len(stems)-1),'rgb':str(rgb.relative_to(out)),'alpha':str(alpha.relative_to(out))})
                atomic_json(destination,manifest)
        manifest['complete']=True;atomic_json(destination,manifest)
    queue=[]
    with (out/'logs/losses.jsonl').open('w') as log:
        for step in range(1,args.steps+1):
            if not queue:queue=rng.permutation(len(stems)).tolist()
            frame=queue.pop();optimizer.zero_grad(set_to_none=True)
            xyz=deform(frame);rgba=render(xyz,frame)
            residual=parameter-initial
            losses={'mask':mask_loss(rgba[:,:,3],targets[frame]),'spatial':spatial_loss(xyz,canonical,edges_gpu,scale),'temporal':temporal_loss(residual,frame,scale),'anchor':(residual/scale).square().sum(-1).mean()}
            total=sum(settings['weights'][name]*value for name,value in losses.items())
            if not torch.isfinite(total):raise ValueError('Nonfinite loss')
            # Verify silhouette alone provides a real differentiable training signal.
            if step==1:
                gradient=torch.autograd.grad(losses['mask'],parameter,retain_graph=True)[0]
                if not torch.isfinite(gradient).all() or gradient.abs().max()==0:raise ValueError('Silhouette gradient is missing, zero or nonfinite')
                atomic_json(out/'gradient-check.json',{'mask_gradient_max':float(gradient.abs().max()),'mask_gradient_norm':float(gradient.norm()),'trainable_parameters':trainable,'initial_seed_reproduction':'exact on first/middle/last frames'})
            total.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():raise ValueError('Invalid translation gradients')
            torch.nn.utils.clip_grad_norm_([parameter],1.)
            optimizer.step()
            row={'step':step,'frame':stems[frame],'total':float(total.detach()),**{key:float(value.detach()) for key,value in losses.items()}}
            log.write(json.dumps(row)+'\n');log.flush()
            if step==1 or step%25==0:print(json.dumps(row),flush=True)
            if step%250==0:checkpoint(step)
    for name,value in frozen.items():
        if not torch.equal(value,model.state_dict()[name].cpu()):raise ValueError('Frozen delta tensor changed: '+name)
    for name,value in frozen_base.items():
        if not torch.equal(value,gui.deform_node_base.deform.state_dict()[name].cpu()):raise ValueError('Frozen node base changed: '+name)
    if torch.equal(initial,parameter.detach()):raise ValueError('Training did not change translation parameters')
    final=checkpoint(args.steps)
    atomic_json(out/'training-status.json',{'complete':True,'steps':args.steps,'checkpoint':str(final),'frozen_tensors_unchanged':True,'translations_changed':True})
    if not args.smoke_check:export(args.steps)
    print(f'Fine-tuning complete: {out}',flush=True)


if __name__=='__main__':main()
