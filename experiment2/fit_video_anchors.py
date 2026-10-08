"""Experiment 2.6: independent raw 2.1 + ARAP + visible video foot handles."""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
from .anchor_arap import VideoAnchorARAP
from .arap import distortion
from .video_anchors import validate_annotations,project
from .leg_audit import leg_cores
from .transfer import ROOT,load_transfer,render_mesh
from .mesh import snapshot
from evaluation.export_input_views import atomic_json
from dashboard.artifacts import frame_index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--definition',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--arap-strength',type=float,default=100)
    p.add_argument('--keypoint-weights',type=float,nargs='+',default=[0,10]);p.add_argument('--iterations',type=int,default=150)
    p.add_argument('--leg-prior-weight',type=float,default=1.,help='Raw-motion prior weight in canonical lower-leg cores; body/unclassified stay 1')
    args=p.parse_args()
    if not 0<args.leg_prior_weight<=1:raise ValueError('Leg prior weight must be in (0,1]')
    definition_bytes=snapshot(Path(args.definition));definition=json.loads(definition_bytes)
    payload={k:v for k,v in definition.items() if k!='sha256'}
    if hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()!=definition['sha256']:raise ValueError('Definition digest mismatch')
    annotation_bytes=snapshot(Path(args.annotations));annotations=json.loads(annotation_bytes)
    active=validate_annotations(annotations,definition)
    seed=Path(definition['seed_run']);baseline=Path(definition['baseline_run']);root=Path(args.output_root).resolve()
    if any(root.is_relative_to(path.resolve()) for path in [seed,baseline]):raise ValueError('Separate output required')
    cfg=json.loads(snapshot(seed/'config.json'))
    if cfg['experiment']!='2.1' or definition['checkpoint_hashes']!=cfg['checkpoint_hashes']:raise ValueError('Raw 2.1 seed required')
    records,_,source_manifest=frame_index(seed,definition['sequence'],'dashboard_exports/mesh.json')
    if not source_manifest.get('complete') or not all(r['alignment'] for r in records):raise ValueError('Verified source alignment required')
    canonical_bytes=snapshot(seed/'mesh/canonical.npz')
    if hashlib.sha256(canonical_bytes).hexdigest()!=definition['canonical_sha256']:raise ValueError('Canonical mesh changed')
    with np.load(io.BytesIO(canonical_bytes)) as saved:
        canonical={k:saved[k].copy() for k in saved.files}
    rest=canonical['vertices'];faces=canonical['faces'];colors=canonical['colors']
    context=load_transfer(baseline,definition['sequence'],'appearance',30000)
    if context['bundle']['hashes']!=definition['checkpoint_hashes']:raise ValueError('Checkpoint provenance changed')
    for key in ['vertices','faces','colors','features']:
        if not np.array_equal(context[key],canonical[key]):raise ValueError('Canonical transfer context differs')
    frames={f['id']:f for f in definition['frames']};rows={r['input_id']:r for r in source_manifest['frames']}
    selected=sorted(active,key=context['stems'].index)
    import torch
    import nvdiffrast.torch as dr
    gui=context['gui'];renderer=dr.RasterizeCudaContext()
    fg=torch.from_numpy(faces).cuda();cg=torch.from_numpy(colors).cuda()
    def camera_xyz(xyz,fid):
        i=context['stems'].index(fid)
        if gui.optimize_per_frame_compose_transforms_app:xyz,_=gui.apply_compose_transform_app(xyz,i,torch.ones_like(xyz))
        transform={k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in rows[fid]['object_transform'].items()}
        rot=torch.zeros((len(xyz),4),device='cuda');rot[:,0]=1
        return gui.sam3d_data.apply_transform_to_gaussian(xyz,rot,torch.ones_like(xyz),transform)[0]
    def render(vertices,fid):
        xyz=camera_xyz(torch.from_numpy(vertices).cuda(),fid)
        return render_mesh(xyz,fg,cg,rows[fid]['input_camera'],renderer).cpu().numpy()
    with torch.inference_mode():
        for fid in selected:
            f=frames[fid]
            if f['camera']!=rows[fid]['input_camera']:raise ValueError('Prepared camera differs from source')
            if hashlib.sha256(snapshot(Path(f['image_path']))).hexdigest()!=f['image_sha256']:raise ValueError('Input image changed')
            with np.load(seed/'mesh'/f'{fid}.npz') as saved:vertices=saved['vertices']
            xyz=camera_xyz(torch.from_numpy(vertices[::101]).cuda(),fid).cpu().numpy()
            affine=vertices[::101]@np.asarray(f['linear']).T+f['translation']
            if np.abs(xyz-affine).max()>2e-5:raise ValueError('Camera affine is not original camera placement')
            rgba=render(vertices,fid)
            with Image.open(seed/rows[fid]['rgb']) as image:reference=np.asarray(image.convert('RGB'),dtype=np.int16)
            if np.abs(np.round(rgba[:,:,:3]*255).astype(np.int16)-reference).max()>1:raise ValueError('Original render reproduction failed')
    root.mkdir(parents=True,exist_ok=True)
    leg_labels,_=leg_cores(rest,faces,definition['partition']['z_cutoff'])
    prior_weights=np.where(leg_labels>=2,args.leg_prior_weight,1.)
    solver=VideoAnchorARAP(rest,faces,args.arap_strength,definition['handles'],device='cuda',prior_weights=prior_weights)
    for weight in args.keypoint_weights:
        out=root/f'{definition["sequence"]}-2.6-anchors-{weight:g}';out.mkdir(exist_ok=False)
        for folder in ['rgb','alpha','mesh','logs','dashboard_exports']:(out/folder).mkdir()
        np.savez(out/'mesh/canonical.npz',**canonical)
        settings={'experiment':'2.6','sequence':definition['sequence'],'baseline_run':str(baseline),'seed_run':str(seed),
            'source_checkpoint':30000,'arap_strength':args.arap_strength,'keypoint_weight':weight,'iterations':args.iterations,
            'objective':'Mean per-vertex prior_weight * soft raw-target error + ARAP strength * mean symmetric edge ARAP / canonical D²; plus keypoint_weight * confidence-weighted mean 2D reprojection squared error / image diagonal²',
            'leg_prior_weight':args.leg_prior_weight,'body_prior_weight':1.,'prior_partition':definition['partition'],
            'solver':'Original local/global ARAP; fixed-camera Gauss-Newton handle reprojection; Woodbury small system; exact full-energy backtracking',
            'initialization':'Raw original 2.1 separately for every annotated frame, not 2.3/2.4/2.5',
            'network_training':False,'network_forward':False,'vertex_optimization':True,'camera_alignment':False,
            'temporal_regularization':False,'interpolated_targets':False,'annotation_metric':'Reprojection on fitted observations; not independent evaluation',
            'selected_frame_ids':selected,'annotation_source':annotations['annotation_source'],'annotator':annotations['annotator'],
            'annotation_review_status':annotations.get('review_status','unspecified'),
            'annotations_sha256':hashlib.sha256(annotation_bytes).hexdigest(),'definition_sha256':definition['sha256'],
            'checkpoint_hashes':definition['checkpoint_hashes'],'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'solver_sha256':hashlib.sha256((ROOT/'experiment2/anchor_arap.py').read_bytes()).hexdigest(),
            'target_object_id':255,'alpha_threshold':.5,'renderer':cfg['renderer'],'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
        atomic_json(out/'config.json',settings);atomic_json(out/'definition.json',definition);atomic_json(out/'annotations.json',annotations)
        atomic_json(out/'inventory.json',{definition['sequence']:context['inventory'][definition['sequence']]})
        atomic_json(out/'metadata.json',{'experiment':'2.6','selected_objects':[definition['sequence']],
            'display_label':f'ARAP {args.arap_strength:g} · '+('control, no video anchors' if weight==0 else f'video anchors {weight:g}')+f' · leg prior {args.leg_prior_weight:g}',
            'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        manifest={'schema':1,'sequence':definition['sequence'],'stage':'video_anchors','checkpoint':30000,'target_object_id':255,
            'camera_space':'input','label':f'2.6 · ARAP + video anchors weight {weight:g} · annotated-frame pilot',
            'time_mapping':source_manifest['time_mapping'],'provenance':settings,'frames':[]}
        with (out/'logs/projection.jsonl').open('w') as log,torch.inference_mode():
            for fid in selected:
                start=time.monotonic();f=frames[fid]
                with np.load(seed/'mesh'/f'{fid}.npz') as saved:target=saved['vertices']
                vertices,report=solver.solve_anchored(target,active[fid],np.asarray(f['linear']),np.asarray(f['translation']),f['camera'],weight,args.iterations)
                rgba=render(vertices,fid)
                np.savez(out/'mesh'/f'{fid}.npz',vertices=vertices)
                Image.fromarray(np.round(rgba[:,:,:3]*255).astype(np.uint8)).save(out/'rgb'/f'{fid}.png')
                Image.fromarray(np.round(rgba[:,:,3]*255).astype(np.uint8)).save(out/'alpha'/f'{fid}.png')
                row=dict(rows[fid]);row.update(rgb=f'rgb/{fid}.png',alpha=f'alpha/{fid}.png')
                manifest['frames'].append(row);atomic_json(out/'dashboard_exports/mesh.json',manifest)
                record={'frame':fid,'optimization':report,'before':distortion(rest,target,faces),'after':distortion(rest,vertices,faces),
                    'seconds':time.monotonic()-start,'observations':active[fid]}
                log.write(json.dumps(record)+'\n');log.flush()
                print(f'anchors={weight:g} {fid} {record["seconds"]:.1f}s: fit error {report["initial"]["reprojection_mean_px"]:.2f} -> {report["final"]["reprojection_mean_px"]:.2f}px',flush=True)
        manifest['complete']=len(selected)==len(context['stems']);manifest['selected_frames_completed']=True
        for name,module in [('deform_node_base.pth',gui.deform_node_base.deform),('deform.pth',gui.deform.deform)]:
            current=module.state_dict();saved=context['bundle']['states'][name]
            if set(current)!=set(saved) or any(not torch.equal(current[k].cpu(),saved[k]) for k in saved):raise ValueError('Learned state changed')
        atomic_json(out/'dashboard_exports/mesh.json',manifest)
        atomic_json(out/'projection-status.json',{'complete':True,'frames':len(selected),'all_input_frames':manifest['complete'],
            'network_training':False,'learned_state_bitwise_unchanged':True,'annotation_status':annotations.get('review_status','unspecified')})


if __name__=='__main__':main()
