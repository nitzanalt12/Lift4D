"""Experiment 2.5: frozen base motions with fixed surface-aware mesh weights.

Starts independently from the original baseline checkpoint used by 2.1. Only
base vertex-to-node blend weights change; delta weights stay exactly original.
No training, ARAP, camera fitting, feature fitting or mesh repair is performed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from .transfer import ROOT,load_transfer,render_mesh
from .mesh import snapshot
from .arap import distortion
from evaluation.export_input_views import atomic_json
from dashboard.artifacts import frame_index


def validated_weights(mapping,node_count,vertex_count):
    """Keep original influence only on explicitly reported unattached components."""
    indices=mapping['surface_indices'].copy();weights=mapping['surface_weights'].copy()
    missing=mapping['unattached_vertices']
    indices[missing]=mapping['original_indices'][missing]
    weights[missing]=mapping['original_weights'][missing]
    if indices.shape!=weights.shape or len(indices)!=vertex_count:raise ValueError('Weight shape mismatch')
    if not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=node_count):raise ValueError('Invalid control node IDs')
    if not np.isfinite(weights).all() or np.any(weights<0) or not np.allclose(weights.sum(1),1,atol=1e-6):raise ValueError('Invalid blend weights')
    return indices,weights.astype(np.float32),missing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-run',required=True);p.add_argument('--diagnostics',required=True)
    p.add_argument('--output',required=True);p.add_argument('--frames',nargs='+');p.add_argument('--smoke-check',action='store_true')
    args=p.parse_args();seed=Path(args.seed_run).resolve();diagnostics=Path(args.diagnostics).resolve();out=Path(args.output).resolve()
    cfg=json.loads(snapshot(seed/'config.json'));diagnostic_cfg=json.loads(snapshot(diagnostics/'config.json'))
    if cfg['experiment']!='2.1' or out.is_relative_to(seed) or out.is_relative_to(Path(cfg['baseline_run'])):
        raise ValueError('Separate output and frozen 2.1 seed required')
    if diagnostic_cfg['seed_run']!=str(seed) or diagnostic_cfg['checkpoint_hashes']!=cfg['checkpoint_hashes']:
        raise ValueError('Mapping was computed for a different source')
    records,_,source_manifest=frame_index(seed,cfg['sequence'],'dashboard_exports/mesh.json')
    if not source_manifest.get('complete') or source_manifest.get('camera_space')!='input' or not all(r['alignment'] for r in records):raise ValueError('Verified complete seed required')
    state=load_transfer(Path(cfg['baseline_run']),cfg['sequence'],'appearance',30000)
    if state['bundle']['hashes']!=cfg['checkpoint_hashes']:raise ValueError('Checkpoint provenance mismatch')
    import torch
    import nvdiffrast.torch as dr
    gui=state['gui'];base=gui.deform_node_base.deform
    with np.load(seed/'mesh/canonical.npz') as saved:
        canonical={key:saved[key].copy() for key in saved.files}
    for key,expected in [('vertices',state['vertices']),('faces',state['faces']),('colors',state['colors']),('features',state['features']),('gaussian_indices',state['indices'])]:
        if not np.array_equal(canonical[key],expected):raise ValueError('Canonical source changed: '+key)
    mapping_bytes=snapshot(diagnostics/'base-weights.npz')
    import io
    with np.load(io.BytesIO(mapping_bytes)) as mapping:
        original_weights,_,original_indices=base.cal_nn_weight(x=gui.gaussians.get_xyz,feature=gui.gaussians.feature)
        if not np.array_equal(original_indices.cpu().numpy(),mapping['original_indices']) or not np.array_equal(original_weights.cpu().numpy(),mapping['original_weights']):
            raise ValueError('Saved original mapping does not reproduce baseline')
        indices,weights,missing=validated_weights(mapping,len(base.nodes),len(state['vertices']))
        distances_squared=mapping['surface_distances']**2
        distances_squared[missing]=mapping['original_distances_squared'][missing]
    selected=args.frames or state['stems']
    if len(set(selected))!=len(selected) or not set(selected)<=set(state['stems']):raise ValueError('Invalid frame IDs')
    selected=sorted(selected,key=state['stems'].index)
    rows={row['input_id']:row for row in source_manifest['frames']}
    renderer=dr.RasterizeCudaContext();faces_gpu=torch.from_numpy(state['faces']).cuda();colors_gpu=torch.from_numpy(state['colors']).cuda()
    def render(xyz,fid):
        i=state['stems'].index(fid)
        camera_xyz=xyz
        if gui.optimize_per_frame_compose_transforms_app:
            camera_xyz,_=gui.apply_compose_transform_app(xyz,i,torch.ones_like(xyz))
        transform={key:torch.tensor(value,dtype=torch.float32,device='cuda') for key,value in rows[fid]['object_transform'].items()}
        rotations=torch.zeros((len(xyz),4),device='cuda');rotations[:,0]=1
        camera_xyz,_,_=gui.sam3d_data.apply_transform_to_gaussian(camera_xyz,rotations,torch.ones_like(xyz),transform)
        return render_mesh(camera_xyz,faces_gpu,colors_gpu,rows[fid]['input_camera'],renderer).cpu().numpy()
    # Original frozen helper must reproduce the seed before altering its cache.
    with torch.inference_mode():
        for fid in dict.fromkeys([selected[0],selected[len(selected)//2],selected[-1]]):
            raw=gui._deform_at_frame(state['stems'].index(fid)).cpu().numpy()
            with np.load(seed/'mesh'/f'{fid}.npz') as saved:
                if np.abs(raw-saved['vertices']).max()>1e-6:raise ValueError('Raw transfer reproduction failed')
            rgba=render(torch.from_numpy(raw).cuda(),fid)
            with Image.open(seed/rows[fid]['rgb']) as image:reference=np.asarray(image.convert('RGB'),dtype=np.int16)
            if np.abs(np.round(rgba[:,:,:3]*255).astype(np.int16)-reference).max()>1:
                raise ValueError('Original camera/render reproduction failed')
    base.cached_nn_weight=True
    base.nn_weight=torch.from_numpy(weights).cuda();base.nn_idxs=torch.from_numpy(indices).cuda()
    base.nn_dist=torch.tensor(distances_squared,dtype=torch.float32,device='cuda')
    # Original forward and saved motions are unchanged. Only the existing cache
    # of fixed blend weights is replaced on this restored evaluation instance.
    out.mkdir(parents=True,exist_ok=False)
    for folder in ['mesh','rgb','alpha','logs','dashboard_exports']:(out/folder).mkdir()
    np.savez(out/'mesh/canonical.npz',**canonical)
    np.savez_compressed(out/'base-weights.npz',indices=indices,weights=weights,original_mapping_vertices=missing)
    settings={'experiment':'2.5','sequence':cfg['sequence'],'baseline_run':cfg['baseline_run'],
        'seed_run':str(seed),'source_checkpoint':30000,'checkpoint_hashes':cfg['checkpoint_hashes'],
        'component_changed':'base vertex-to-control-node blend weights only',
        'base_weight_rule':'K=3 nearest by canonical mesh edge-path distance plus node-to-nearest-vertex offset; saved radius Gaussian kernel plus original 1e-7 floor; fixed across time',
        'hyper_feature_policy':'Original nearest Gaussian features retained; base node selection now uses surface paths instead of XYZ+hyper-feature distance. Node network inputs unchanged.',
        'unattached_component_policy':'Keep original base mapping on explicitly listed unattached vertices; never bridge components to invent surface paths',
        'unattached_vertices_count':len(missing),'unattached_vertices_file':'base-weights.npz:original_mapping_vertices',
        'delta_weights':'unchanged original spatial KNN','network_training':False,'vertex_optimization':False,
        'arap':False,'canonical_mesh':'unchanged','camera_alignment':False,'renderer':cfg['renderer'],
        'selected_frame_ids':selected,'smoke_check':args.smoke_check,'target_object_id':255,'alpha_threshold':.5,
        'diagnostics':str(diagnostics),'mapping_sha256':hashlib.sha256(mapping_bytes).hexdigest(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'raw_reproduction':'First/middle/last selected meshes within 1e-6; original RGB within one uint8 level',
        'seed_config':cfg,'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    atomic_json(out/'config.json',settings);atomic_json(out/'inventory.json',{cfg['sequence']:state['inventory'][cfg['sequence']]})
    atomic_json(out/'metadata.json',{'experiment':'2.5','selected_objects':[cfg['sequence']],
        'display_label':'Surface-aware base weights · frozen network',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()})
    manifest={'schema':1,'sequence':cfg['sequence'],'target_object_id':255,'camera_space':'input',
        'stage':'surface_transfer','checkpoint':30000,'label':'Experiment 2.5 · surface-aware base weights · source 30000',
        'time_mapping':source_manifest['time_mapping'],'provenance':settings,'frames':[]}
    with (out/'logs/transfer.jsonl').open('w') as log,torch.inference_mode():
        for fid in selected:
            i=state['stems'].index(fid);xyz=gui._deform_at_frame(i)
            if not torch.isfinite(xyz).all():raise ValueError('Nonfinite surface transfer')
            vertices=xyz.cpu().numpy();np.savez(out/'mesh'/f'{fid}.npz',vertices=vertices)
            rgba=render(xyz,fid)
            Image.fromarray(np.round(rgba[:,:,:3]*255).astype(np.uint8)).save(out/'rgb'/f'{fid}.png')
            Image.fromarray(np.round(rgba[:,:,3]*255).astype(np.uint8)).save(out/'alpha'/f'{fid}.png')
            row=dict(rows[fid]);row.update(rgb=f'rgb/{fid}.png',alpha=f'alpha/{fid}.png')
            row.pop('native_rgb_mean_absolute_error',None);row.pop('native_rgb_max_absolute_error',None)
            manifest['frames'].append(row);atomic_json(out/'dashboard_exports/mesh.json',manifest)
            record={'frame':fid,'distortion':distortion(state['vertices'],vertices,state['faces'])}
            log.write(json.dumps(record)+'\n');log.flush();print(fid,json.dumps(record),flush=True)
    # Cached interpolation tensors are not model parameters. Confirm saved
    # weights and every learned tensor remained bitwise frozen after evaluation.
    for label,module in [('deform_node_base.pth',base),('deform.pth',gui.deform.deform)]:
        current=module.state_dict();saved=state['bundle']['states'][label]
        if set(current)!=set(saved) or any(not torch.equal(current[key].cpu(),saved[key]) for key in saved):raise ValueError('Learned state changed')
    manifest['complete']=len(selected)==len(state['stems']);manifest['selected_frames_completed']=True
    atomic_json(out/'dashboard_exports/mesh.json',manifest)
    atomic_json(out/'transfer-status.json',{'complete':True,'frames':len(selected),'all_input_frames':manifest['complete'],
        'learned_state_bitwise_unchanged':True,'network_training':False})


if __name__=='__main__':main()
