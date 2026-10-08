"""Export saved-mesh leg colors, exact node motions and influence audits for UI."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from .transfer import ROOT,load_transfer
from .mesh import snapshot
from .leg_audit import leg_cores,node_affinity,ownership,cross_leg_summary,REGIONS,COLORS
from evaluation.export_input_views import atomic_json


def binary(out,name,array,dtype):
    array=np.asarray(array,dtype=dtype)
    array.tofile(out/name)
    return {'file':name,'dtype':np.dtype(dtype).name,'shape':list(array.shape)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-run',required=True);p.add_argument('--surface-run',required=True)
    p.add_argument('--diagnostics',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frames',nargs='+',default=['00000','00030','00060','00080'])
    p.add_argument('--z-cutoff',type=float,default=-.06)
    args=p.parse_args();seed=Path(args.seed_run).resolve();surface=Path(args.surface_run).resolve();diag=Path(args.diagnostics).resolve();out=Path(args.output).resolve()
    cfg=json.loads(snapshot(seed/'config.json'))
    if cfg['sequence']!='camel' or cfg['experiment']!='2.1':raise ValueError('This diagnostic leg partition is explicitly camel-specific')
    if any(out.is_relative_to(path) for path in [seed,surface,Path(cfg['baseline_run'])]):raise ValueError('Separate diagnostic output required')
    state=load_transfer(Path(cfg['baseline_run']),'camel','appearance',30000)
    if state['bundle']['hashes']!=cfg['checkpoint_hashes']:raise ValueError('Checkpoint provenance changed')
    with np.load(seed/'mesh/canonical.npz') as saved:
        for key,expected in [('vertices',state['vertices']),('faces',state['faces']),('features',state['features'])]:
            if not np.array_equal(saved[key],expected):raise ValueError('Canonical data changed')
    surface_cfg=json.loads(snapshot(surface/'config.json'))
    if surface_cfg['seed_run']!=str(seed) or surface_cfg['checkpoint_hashes']!=cfg['checkpoint_hashes']:raise ValueError('Surface run source differs')
    labels,partition=leg_cores(state['vertices'],state['faces'],args.z_cutoff)
    import torch
    gui=state['gui'];base=gui.deform_node_base.deform;delta=gui.deform.deform
    checkpoint=Path(state['bundle']['checkpoint']['path'])
    gaussians=PlyData.read(checkpoint/'gaussians.ply')['vertex'].data
    gaussian_xyz=np.stack([gaussians[key] for key in ['x','y','z']],1).copy()
    gaussian_features=np.stack([gaussians[f'fea_{i}'] for i in range(state['cfg']['hyper_dim'])],1).copy()
    offsets,nearest=cKDTree(state['vertices']).query(gaussian_xyz)
    gaussian_labels=labels[nearest].copy()
    distance_limit=.02*np.linalg.norm(np.ptp(state['vertices'],axis=0))
    gaussian_labels[offsets>distance_limit]=0
    opacity=1/(1+np.exp(-np.clip(gaussians['opacity'].astype(np.float64),-50,50)))
    out.mkdir(parents=True,exist_ok=False)
    manifest={'schema':1,'sequence':'camel','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'partition_sha256':hashlib.sha256((ROOT/'experiment2/leg_audit.py').read_bytes()).hexdigest(),
        'checkpoint_hashes':cfg['checkpoint_hashes'],'seed_run':str(seed),'surface_run':str(surface),
        'partition':partition,'regions':REGIONS,'colors':COLORS,
        'coordinate_space':'Canonical SAM3D source XYZ; no camera compose, alignment or scaling. −X/+X are coordinate sides, not anatomical left/right.',
        'semantic_status':'Provisional geometry-derived leg cores; visually inspect before treating ownership as anatomy',
        'gaussian_region_rule':f'Nearest canonical vertex region; distances > {distance_limit:g} are unclassified; opacity-weighted Gaussian influence',
        'ownership_rule':'Dominant Gaussian receiver region >=80% of total opacity-weighted influence and support >=1; otherwise unclassified',
        'geometry':{},'layers':{},'frames':[],'network_training':False,'mapping_modified':False,'complete':False}
    manifest['geometry']={'canonical':binary(out,'canonical.bin',state['vertices'],'<f4'),
        'faces':binary(out,'faces.bin',state['faces'],'<i4'),'labels':binary(out,'labels.bin',labels,'i1')}
    report={'partition':partition,'layers':{}}
    with torch.inference_mode():
        for name,module in [('base',base),('delta',delta)]:
            with np.load(diag/f'{name}-weights.npz') as saved:
                mapping={key:saved[key].copy() for key in saved.files}
            old_weights,_,old_indices=module.cal_nn_weight(x=gui.gaussians.get_xyz,feature=gui.gaussians.feature)
            if not np.array_equal(old_indices.cpu().numpy(),mapping['original_indices']) or not np.array_equal(old_weights.cpu().numpy(),mapping['original_weights']):raise ValueError('Original mapping does not reproduce')
            nodes=module.nodes[:,:3].cpu().numpy();anchors=mapping['anchors'];attached=labels[anchors]
            gw,_,gi=module.cal_nn_weight(x=torch.from_numpy(gaussian_xyz).cuda(),feature=torch.from_numpy(gaussian_features).cuda())
            affinity=node_affinity(gaussian_labels,gi.cpu().numpy(),gw.cpu().numpy(),len(nodes),opacity)
            owner,confidence,support=ownership(affinity)
            disputed=(attached>=2)&(owner>=2)&(attached!=owner)
            summary,cross=cross_leg_summary(labels,attached,mapping['original_indices'],mapping['original_weights'])
            layer={'nodes':binary(out,f'{name}-nodes.bin',nodes,'<f4'),'anchors':binary(out,f'{name}-anchors.bin',anchors,'<i4'),
                'attached_regions':attached.tolist(),'gaussian_owner_regions':owner.tolist(),
                'gaussian_owner_confidence':confidence.tolist(),'gaussian_support':support.tolist(),
                'gaussian_region_mass':affinity.tolist(),'disputed_nodes':np.flatnonzero(disputed).tolist(),
                'weights':{'original':{'indices':binary(out,f'{name}-original-indices.bin',mapping['original_indices'],'<i4'),
                    'values':binary(out,f'{name}-original-weights.bin',mapping['original_weights'],'<f4'),
                    'cross_mass':binary(out,f'{name}-original-cross.bin',cross,'<f4')}},
                'motion_description':'Saved base node translations only' if name=='base' else 'Saved appearance delta translations only'}
            layer_report={'node_count':len(nodes),'nodes_attached_to_leg_cores':int((attached>=2).sum()),
                'high_purity_gaussian_leg_owners':int((owner>=2).sum()),'attachment_owner_disagreements':int(disputed.sum()),
                'disputed_node_ids':layer['disputed_nodes'],'original':summary}
            if name=='base':
                idx=mapping['surface_indices'].copy();w=mapping['surface_weights'].copy();missing=mapping['unattached_vertices']
                idx[missing]=mapping['original_indices'][missing];w[missing]=mapping['original_weights'][missing]
                summary,cross=cross_leg_summary(labels,attached,idx,w)
                layer['weights']['surface']={'indices':binary(out,'base-surface-indices.bin',idx,'<i4'),
                    'values':binary(out,'base-surface-weights.bin',w,'<f4'),'cross_mass':binary(out,'base-surface-cross.bin',cross,'<f4')}
                layer_report['surface']=summary
            manifest['layers'][name]=layer;report['layers'][name]=layer_report
        for fid in args.frames:
            i=state['stems'].index(fid)
            raw=gui._deform_at_frame(i).cpu().numpy()
            with np.load(seed/'mesh'/f'{fid}.npz') as saved:
                if np.abs(raw-saved['vertices']).max()>1e-6:raise ValueError('Saved mesh reproduction failed')
            t=torch.tensor(i/max(1,len(state['stems'])-1),device='cuda')
            base_motion=base.node_deform(base.expand_time(t))['d_xyz'].cpu().numpy()
            delta_motion=delta.control_point_deltas[delta._frame_to_idx[i]].cpu().numpy()
            row={'id':fid,'original':binary(out,f'original-{fid}.bin',raw,'<f4'),
                 'base_motion':binary(out,f'base-motion-{fid}.bin',base_motion,'<f4'),
                 'delta_motion':binary(out,f'delta-motion-{fid}.bin',delta_motion,'<f4')}
            with np.load(surface/'mesh'/f'{fid}.npz') as saved:row['surface']=binary(out,f'surface-{fid}.bin',saved['vertices'],'<f4')
            manifest['frames'].append(row)
        for name,module in [('deform_node_base.pth',base),('deform.pth',delta)]:
            current=module.state_dict();saved=state['bundle']['states'][name]
            if any(not torch.equal(current[k].cpu(),saved[k]) for k in saved):raise ValueError('Learned state changed')
    manifest['summary']=report;manifest['complete']=True;manifest['learned_state_bitwise_unchanged']=True
    atomic_json(out/'audit.json',manifest);atomic_json(out/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
