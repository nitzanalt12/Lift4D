"""Prepare real video tracks and canonical geodesic neighborhoods on SLURM."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from .geodesic import graph,influence,neighborhoods,neighbor_gate
from experiment2.transfer import ROOT,load_transfer,render_mesh
from experiment2.mesh import snapshot
from experiment2.video_anchors import project
from evaluation.export_input_views import atomic_json


def sha(path):return hashlib.sha256(snapshot(Path(path))).hexdigest()


def fps(vertices,count,device='cuda'):
    import torch
    from pytorch3d.ops import sample_farthest_points
    _,ids=sample_farthest_points(torch.tensor(vertices,dtype=torch.float32,device=device)[None],K=min(count,len(vertices)),random_start_point=False)
    return ids[0].cpu().numpy()


def visible_vertices(camera_xyz,faces,camera,mask,context,tolerance):
    import torch
    import nvdiffrast.torch as dr
    width,height=camera['width'],camera['height'];z=camera_xyz[:,2];near,far=.001,1000.
    clip=torch.stack((2*camera['fx']/width*camera_xyz[:,0]+(2*camera['cx']/width-1)*z,
        -2*camera['fy']/height*camera_xyz[:,1]-(2*camera['cy']/height-1)*z,
        (far+near)/(far-near)*z-2*far*near/(far-near),z),dim=-1)
    raster,_=dr.rasterize(context,clip[None].contiguous(),faces,resolution=[height,width])
    depth,_=dr.interpolate(z[None,:,None].contiguous(),raster,faces)
    depth=depth[0,:,:,0].flip(0).cpu().numpy();coverage=(raster[0,:,:,3]>0).flip(0).cpu().numpy()
    xyz=camera_xyz.cpu().numpy();pixels=project(xyz,np.eye(3),np.zeros(3),camera)
    rounded=np.rint(pixels).astype(int);x,y=rounded.T
    inside=(x>=0)&(x<width)&(y>=0)&(y<height)
    result=np.zeros(len(x),dtype=bool);ids=np.flatnonzero(inside)
    result[ids]=coverage[y[ids],x[ids]]&mask[y[ids],x[ids]]&(np.abs(xyz[ids,2]-depth[y[ids],x[ids]])<=tolerance)
    return np.flatnonzero(result),pixels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--nodes',type=int,default=2048);p.add_argument('--queries',type=int,default=256)
    p.add_argument('--cycle-threshold',type=float,default=3.)
    p.add_argument('--tracker-repo',default='/home/nitzan.alt/.cache/torch/hub/facebookresearch_co-tracker_main')
    p.add_argument('--tracker-checkpoint',default='/home/nitzan.alt/.cache/torch/hub/checkpoints/scaled_offline.pth')
    args=p.parse_args();seed=Path(args.seed_run).resolve();out=Path(args.output).resolve()
    if out.exists() or out.is_relative_to(seed):raise ValueError('Separate new output required')
    cfg=json.loads(snapshot(seed/'config.json'));sequence=cfg['sequence']
    if cfg['experiment']!='2.1':raise ValueError('Original canonical SAM3D seed required')
    context=load_transfer(Path(cfg['baseline_run']),sequence,'appearance',30000)
    if context['bundle']['hashes']!=cfg['checkpoint_hashes']:raise ValueError('Camera-source checkpoint changed')
    rest,faces,colors=context['vertices'],context['faces'],context['colors']
    with np.load(seed/'mesh/canonical.npz') as saved:
        for key in ['vertices','faces','colors']:
            if not np.array_equal(saved[key],context[key]):raise ValueError('Canonical SAM3D geometry changed')
    source=json.loads(snapshot(seed/'dashboard_exports/mesh.json'));stems=context['stems']
    rows={r['input_id']:r for r in source['frames']}
    if not source.get('complete') or set(rows)!=set(stems):raise ValueError('Complete explicit frame mapping required')
    import torch
    import nvdiffrast.torch as dr
    torch.set_num_threads(2)
    gui=context['gui'];probe=torch.tensor(np.r_[np.zeros((1,3)),10*np.eye(3)],dtype=torch.float32,device='cuda')
    def to_camera(xyz,index,row):
        if gui.optimize_per_frame_compose_transforms_app:xyz,_=gui.apply_compose_transform_app(xyz,index,torch.ones_like(xyz))
        transform={k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in row['object_transform'].items()}
        rotation=torch.zeros((len(xyz),4),device='cuda');rotation[:,0]=1
        return gui.sam3d_data.apply_transform_to_gaussian(xyz,rotation,torch.ones_like(xyz),transform)[0]
    frames=[];images=[];masks=[];inventory=context['inventory'][sequence]
    with torch.inference_mode():
        for index,fid in enumerate(stems):
            row=rows[fid]
            if row['render_id']!=fid or row['input_frame_index']!=index or row['render_frame_index']!=index or row['input_camera']!=row['render_camera']:
                raise ValueError('Camera/frame correspondence differs')
            values=to_camera(probe,index,row).cpu().numpy().astype(float);linear=(values[1:]-values[0]).T/10;translation=values[0]
            check=rest[::101];reference=to_camera(torch.tensor(check,device='cuda'),index,row).cpu().numpy()
            error=float(np.abs(check@linear.T+translation-reference).max())
            if error>2e-5:raise ValueError('Camera affine fails original helper reproduction')
            image_path=Path(inventory['frames_dir'])/(fid+'.jpg');mask_path=Path(inventory['masks_dir'])/(fid+'.png')
            image=np.asarray(Image.open(image_path).convert('RGB'));mask=np.asarray(Image.open(mask_path))
            camera=row['input_camera'];shape=(camera['height'],camera['width'])
            if image.shape[:2]!=shape or mask.shape!=shape or not np.any(mask==255):raise ValueError('Native input/mask mismatch')
            images.append(image);masks.append(mask==255)
            frames.append({'id':fid,'index':index,'time':row['deformation_time'],'camera':camera,'linear':linear.tolist(),'translation':translation.tolist(),
                'image_path':str(image_path),'mask_path':str(mask_path),'image_sha256':sha(image_path),'mask_sha256':sha(mask_path),'camera_reproduction_max_error':error})
    # Old saved networks were used only by the existing loader to recover camera
    # transforms; no forward pass, weights/features/nodes enter the new model.
    renderer=dr.RasterizeCudaContext();rest_gpu=torch.tensor(rest,device='cuda');faces_gpu=torch.tensor(faces,dtype=torch.int32,device='cuda')
    camera_xyz=torch.tensor(rest@np.asarray(frames[0]['linear']).T+frames[0]['translation'],dtype=torch.float32,device='cuda')
    diagonal=float(np.linalg.norm(np.ptp(rest,axis=0)));operator=np.linalg.norm(frames[0]['linear'],ord=2)
    candidates,pixels=visible_vertices(camera_xyz,faces_gpu,frames[0]['camera'],masks[0],renderer,.005*diagonal*operator)
    if len(candidates)<32:raise ValueError('Too few visible, masked canonical tracking vertices')
    anchors=candidates[fps(rest[candidates],args.queries)]
    adjacency,edges,edge_lengths,components,count=graph(rest,faces)
    representatives=np.asarray([np.flatnonzero(components==i)[0] for i in range(count)])
    if args.nodes<count:raise ValueError('Every component needs a fresh control anchor')
    selected=fps(rest,min(len(rest),args.nodes+count))
    node_vertices=np.r_[representatives,selected[~np.isin(selected,representatives)][:args.nodes-count]]
    print(f'{sequence}: {len(rest)} vertices, {count} components, {len(node_vertices)} fresh controls, {len(anchors)} visible tracking queries',flush=True)
    ids,weights,distances,sigma=influence(adjacency,node_vertices)
    neighbors=neighborhoods(adjacency,edges,anchors,maximum_distance=.2*diagonal)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'geometry.npz',vertices=rest,faces=faces,colors=colors,edges=edges,edge_lengths=edge_lengths,
        node_vertices=node_vertices,control_indices=ids,control_weights=weights,control_geodesic_distances=distances,
        track_vertices=anchors,**neighbors)
    del context,gui,renderer,rest_gpu,faces_gpu,camera_xyz
    torch.cuda.empty_cache()
    tracker_repo=Path(args.tracker_repo).resolve();sys.path.insert(0,str(tracker_repo))
    from cotracker.predictor import CoTrackerPredictor
    tracker=CoTrackerPredictor(checkpoint=args.tracker_checkpoint,offline=True,v2=False,window_len=60).cuda().eval()
    video=torch.tensor(np.stack(images).transpose(0,3,1,2),dtype=torch.float32,device='cuda')[None]
    queries=torch.tensor(np.c_[np.zeros(len(anchors)),pixels[anchors]],dtype=torch.float32,device='cuda')[None]
    with torch.inference_mode():
        tracks,visibility=tracker(video,queries=queries)
        tracks_np=tracks[0].cpu().numpy();visible=visibility[0].cpu().numpy().astype(bool)
        last=np.asarray([np.flatnonzero(visible[:,i])[-1] if visible[:,i].any() else 0 for i in range(len(anchors))])
        reverse_queries=np.c_[len(stems)-1-last,tracks_np[last,np.arange(len(anchors))]]
        back,back_visibility=tracker(video.flip(1),queries=torch.tensor(reverse_queries,dtype=torch.float32,device='cuda')[None],backward_tracking=True)
        cycle=np.linalg.norm(back[0,-1].cpu().numpy()-pixels[anchors],axis=1)
        cycle_good=(cycle<=args.cycle_threshold)&back_visibility[0,-1].cpu().numpy().astype(bool)
    valid=visible&cycle_good[None]&np.isfinite(tracks_np).all(axis=2)
    outside=np.zeros(valid.shape,dtype=bool)
    for i,mask in enumerate(masks):
        x,y=np.rint(tracks_np[i]).astype(int).T;h,w=mask.shape
        inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        keep=np.zeros(len(anchors),dtype=bool);keep[inside]=mask[y[inside],x[inside]]
        outside[i]=~keep;valid[i]&=keep
    valid,geo_rejected,bounds=neighbor_gate(tracks_np,valid,neighbors['pairs'],neighbors['rest_path_lengths'],frames,rest)
    if np.count_nonzero(valid)<len(stems)*8:raise ValueError('Insufficient real validated video observations for full training')
    np.savez_compressed(out/'tracks.npz',xy=tracks_np,tracker_visible=visible,valid=valid,cycle_error_px=cycle,
        cycle_good=cycle_good,outside_target_mask=outside,geodesic_rejected=geo_rejected,query_xy=pixels[anchors])
    tracker_source_hash=hashlib.sha256(b''.join(p.relative_to(tracker_repo).as_posix().encode()+p.read_bytes() for p in sorted(tracker_repo.rglob('*.py')) if '__pycache__' not in p.parts)).hexdigest()
    settings={'schema':1,'sequence':sequence,'seed_run':str(seed),'baseline_run':cfg['baseline_run'],'frames':frames,'source_camera_manifest_sha256':sha(seed/'dashboard_exports/mesh.json'),
        'source_canonical_sha256':sha(seed/'mesh/canonical.npz'),'sam3d_glb_sha256':cfg['mesh_sha256'],'node_count':len(node_vertices),'query_count':len(anchors),'control_k':3,
        'control_policy':'Fresh canonical vertex FPS with one anchor per component; fixed shortest-edge-path Gaussian softmax, no latent features, no spatial fallback',
        'control_sigma':sigma,'neighbor_policy':'6 nearest finite canonical edge-geodesic neighbors within 0.2 canonical diagonal; identities fixed across time',
        'maximum_neighbor_stretch':2.,'neighbor_depth_assumption':'At most one canonical diagonal of object-space motion beyond transformed canonical depth envelope',
        'tracker':'CoTracker3 scaled offline, explicit canonical visible vertex queries, native output coordinates, forward/reverse cycle',
        'tracker_checkpoint':str(Path(args.tracker_checkpoint).resolve()),'tracker_checkpoint_sha256':sha(args.tracker_checkpoint),'tracker_source_sha256':tracker_source_hash,
        'tracker_internal_resolution':list(tracker.interp_shape),'cycle_threshold_px':args.cycle_threshold,'target_object_id':255,'alpha_threshold':.5,
        'retained_observations':int(valid.sum()),'retained_by_frame':valid.sum(axis=1).tolist(),'cycle_accepted_tracks':int(cycle_good.sum()),
        'geodesic_rejected_observations':int(geo_rejected.sum()),'geodesic_gate_bounds':bounds,
        'geometry_sha256':hashlib.sha256((out/'geometry.npz').read_bytes()).hexdigest(),'tracks_sha256':hashlib.sha256((out/'tracks.npz').read_bytes()).hexdigest(),
        'network_initialization':'Random from scratch; old deformation weights and Gaussian features are never training inputs',
        'camera_policy':'Frozen original appearance camera placement, verified against original helper; no camera fitting',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'implementation_sha256':sha(__file__),'inventory':inventory}
    atomic_json(out/'prepared.json',settings)
    print(f'Prepared {sequence}: {valid.sum()} observations, cycle-accepted {cycle_good.sum()}/{len(anchors)}, geodesic rejected {geo_rejected.sum()}',flush=True)


if __name__=='__main__':main()
