"""Experiment 2.4: saved frozen-transfer vertices projected to mesh ARAP."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
from .arap import ARAP,distortion
from .transfer import ROOT,load_transfer,render_mesh
from .mesh import snapshot
from evaluation.export_input_views import atomic_json
from dashboard.artifacts import frame_index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-run',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--strengths',type=float,nargs='+',default=[1,10,100])
    p.add_argument('--iterations',type=int,default=15);p.add_argument('--frames',nargs='+')
    p.add_argument('--smoke-check',action='store_true')
    args=p.parse_args()
    if args.iterations<1:raise ValueError('Positive iteration count required')
    seed=Path(args.seed_run).resolve();root=Path(args.output_root).resolve()
    cfg=json.loads(snapshot(seed/'config.json'))
    if cfg.get('experiment')!='2.1':raise ValueError('ARAP must start from frozen experiment 2.1')
    if root.is_relative_to(seed):raise ValueError('Output must be separate from source artifacts')
    seq=cfg['sequence'];baseline=Path(cfg['baseline_run'])
    if root.is_relative_to(baseline):raise ValueError('Output must be separate from baseline')
    records,_,source_manifest=frame_index(seed,seq,'dashboard_exports/mesh.json')
    if not source_manifest.get('complete') or not all(r['alignment'] for r in records):raise ValueError('Verified complete seed frame map required')
    context=load_transfer(baseline,seq,'appearance',30000)
    if context['bundle']['hashes']!=cfg['checkpoint_hashes']:raise ValueError('Checkpoint provenance mismatch')
    import torch
    import nvdiffrast.torch as dr
    gui=context['gui'];renderer=dr.RasterizeCudaContext()
    with np.load(seed/'mesh/canonical.npz') as saved:
        rest=saved['vertices'].copy();faces=saved['faces'].copy();colors=saved['colors'].copy()
        for key,expected in [('vertices',rest),('faces',faces),('colors',colors),('features',saved['features']),('indices',saved['gaussian_indices'])]:
            if not np.array_equal(context[key],expected):raise ValueError('Canonical transfer context mismatch: '+key)
    indices={fid:i for i,fid in enumerate(context['stems'])}
    selected=args.frames or context['stems']
    if len(set(selected))!=len(selected) or not set(selected)<=set(indices):raise ValueError('Invalid explicit frame IDs')
    selected=sorted(selected,key=indices.get)
    rows={r['input_id']:r for r in source_manifest['frames']}
    faces_gpu=torch.tensor(faces,dtype=torch.int32,device='cuda');colors_gpu=torch.tensor(colors,dtype=torch.float32,device='cuda')
    rotations=torch.zeros((len(rest),4),device='cuda');rotations[:,0]=1;scaling=torch.ones((len(rest),3),device='cuda')
    def render(vertices,fid):
        xyz=torch.tensor(vertices,dtype=torch.float32,device='cuda');i=indices[fid]
        xyz,_=gui.apply_compose_transform_app(xyz,i,scaling)
        transform={key:torch.tensor(value,dtype=torch.float32,device='cuda') for key,value in rows[fid]['object_transform'].items()}
        xyz,_,_=gui.sam3d_data.apply_transform_to_gaussian(xyz,rotations,scaling,transform)
        return render_mesh(xyz,faces_gpu,colors_gpu,rows[fid]['input_camera'],renderer).cpu().numpy()
    # Prove that camera placement and rasterization match saved 2.1 before correction.
    with torch.inference_mode():
        for fid in [selected[0],selected[len(selected)//2],selected[-1]]:
            with np.load(seed/'mesh'/f'{fid}.npz') as data:vertices=data['vertices']
            rgba=render(vertices,fid)
            with Image.open(seed/rows[fid]['rgb']) as image:reference=np.array(image.convert('RGB'))
            error=np.abs(np.round(rgba[:,:,:3]*255).astype(np.int16)-reference.astype(np.int16))
            if error.max()>1:raise ValueError('Uncorrected camera/render differs from seed: '+fid)
    root.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    for strength in args.strengths:
        out=root/f'{seq}-2.4-arap-{strength:g}'
        out.mkdir(exist_ok=False)
        for folder in ['rgb','alpha','mesh','logs','dashboard_exports']:(out/folder).mkdir()
        np.savez(out/'mesh/canonical.npz',vertices=rest,faces=faces,colors=colors)
        settings={'experiment':'2.4','sequence':seq,'seed_run':str(seed),'baseline_run':str(baseline),'source_checkpoint':30000,'arap_strength':strength,'iterations':args.iterations,'relative_tolerance':1e-5,'objective':'mean squared soft target distance + strength * mean symmetric directed-edge ARAP energy; both normalized by canonical bbox diagonal squared','edge_weights':'positive uniform, all original unique triangle edges; proper per-vertex SO(3) rotations','canonical_reference':'unchanged original SAM3D canonical mesh in source XYZ','solver':'local proper rotations by batched SVD; global sparse CPU LU; fixed system factored once per strength','initialization':'saved raw 2.1 object-space vertices separately for every frame','temporal_regularization':False,'network_training':False,'network_forward':False,'vertex_optimization':True,'camera_alignment':False,'renderer':cfg['renderer'],'target_object_id':255,'alpha_threshold':.5,'checkpoint_hashes':cfg['checkpoint_hashes'],'seed_config':cfg,'solver_sha256':hashlib.sha256((ROOT/'experiment2/arap.py').read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'selected_frame_ids':selected,'smoke_check':args.smoke_check,'source_reproduction':'uncorrected RGB maximum error <=1 uint8 on first/middle/last selected frames'}
        atomic_json(out/'config.json',settings)
        atomic_json(out/'inventory.json',{seq:context['inventory'][seq]})
        atomic_json(out/'metadata.json',{'experiment':'2.4','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'selected_objects':[seq],'display_label':f'ARAP λ={strength:g}','baseline_metadata':json.loads(snapshot(baseline/'metadata.json'))})
        manifest={'schema':1,'sequence':seq,'target_object_id':255,'camera_space':'input','label':f'Experiment 2.4 · ARAP λ={strength:g} · source 30000','stage':'arap','checkpoint':30000,'frames':[],'time_mapping':source_manifest['time_mapping'],'provenance':settings}
        start=time.monotonic();solver=ARAP(rest,faces,strength,device='cuda')
        print(f'Factorization ready: λ={strength:g}, {time.monotonic()-start:.2f}s',flush=True)
        with (out/'logs/projection.jsonl').open('w') as log,torch.inference_mode():
            for fid in selected:
                start=time.monotonic()
                path=seed/'mesh'/f'{fid}.npz'
                source_bytes=snapshot(path)
                import io
                with np.load(io.BytesIO(source_bytes)) as saved:target=saved['vertices']
                vertices,report=solver.solve(target,args.iterations)
                before=distortion(rest,target,faces);after=distortion(rest,vertices,faces)
                rgba=render(vertices,fid)
                np.savez(out/'mesh'/f'{fid}.npz',vertices=vertices)
                Image.fromarray(np.round(rgba[:,:,:3]*255).astype(np.uint8)).save(out/'rgb'/f'{fid}.png')
                Image.fromarray(np.round(rgba[:,:,3]*255).astype(np.uint8)).save(out/'alpha'/f'{fid}.png')
                row=dict(rows[fid]);row.pop('native_rgb_mean_absolute_error',None);row.pop('native_rgb_max_absolute_error',None)
                row.update(rgb=f'rgb/{fid}.png',alpha=f'alpha/{fid}.png')
                manifest['frames'].append(row);atomic_json(out/'dashboard_exports/mesh.json',manifest)
                record={'frame':fid,'seconds':time.monotonic()-start,'source_vertices_sha256':hashlib.sha256(source_bytes).hexdigest(),'optimization':report,'before':before,'after':after}
                log.write(json.dumps(record)+'\n');log.flush()
                print(f'λ={strength:g} {fid} {len(manifest["frames"])}/{len(selected)}, {record["seconds"]:.2f}s, stretch p95 {before["edge_stretch_p95"]:.2f} -> {after["edge_stretch_p95"]:.2f}',flush=True)
        manifest['complete']=len(selected)==len(context['stems']);manifest['selected_frames_completed']=True
        atomic_json(out/'dashboard_exports/mesh.json',manifest)
        atomic_json(out/'projection-status.json',{'complete':True,'frames':len(selected),'all_input_frames':manifest['complete'],'network_training':False})
        del solver
        print('Completed '+str(out),flush=True)


if __name__=='__main__':main()
