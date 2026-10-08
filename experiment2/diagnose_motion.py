"""Read-only decomposition and surface-shortcut diagnostics for saved 2.1."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .arap import distortion,mesh_edges
from .transfer import load_transfer,ROOT
from .surface_weights import surface_weights
from evaluation.export_input_views import atomic_json


def describe(values):
    values=np.asarray(values);finite=values[np.isfinite(values)]
    return {'p50':float(np.percentile(finite,50)),'p95':float(np.percentile(finite,95)),
            'max':float(finite.max()),'nonfinite':int(values.size-finite.size)} if finite.size else {'nonfinite':int(values.size)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frames',nargs='+',default=['00030','00060','00080'])
    args=p.parse_args();seed=Path(args.seed_run).resolve();out=Path(args.output).resolve()
    cfg=json.loads((seed/'config.json').read_text())
    if cfg['experiment']!='2.1' or out.is_relative_to(seed):raise ValueError('Separate output and frozen 2.1 required')
    state=load_transfer(Path(cfg['baseline_run']),cfg['sequence'],'appearance',30000)
    if cfg['checkpoint_hashes']!=state['bundle']['hashes']:raise ValueError('Checkpoint hashes changed')
    import torch
    gui=state['gui'];xyz=gui.gaussians.get_xyz;vertices=state['vertices'];faces=state['faces'];edges=mesh_edges(faces)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'config.json',{'seed_run':str(seed),'checkpoint_hashes':cfg['checkpoint_hashes'],
        'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'surface_weights_sha256':hashlib.sha256((ROOT/'experiment2/surface_weights.py').read_bytes()).hexdigest(),
        'frame_ids':args.frames,'semantic_labels':False,'shortcut_rule':'edge-path / Euclidean distance > 3 AND edge-path > 0.1 canonical bbox diagonal; diagnostic threshold, not leg labels'})
    report={'frames':{},'mapping':{},'canonical_bbox_diagonal':float(np.linalg.norm(np.ptp(vertices,axis=0)))}
    with torch.inference_mode():
        for fid in args.frames:
            i=state['stems'].index(fid);t=torch.tensor(i/max(1,len(state['stems'])-1),device='cuda')
            full=gui._deform_at_frame(i).cpu().numpy()
            with np.load(seed/'mesh'/f'{fid}.npz') as data:
                error=float(np.abs(full-data['vertices']).max())
                if error>1e-6:raise ValueError(f'Saved raw transfer reproduction failed: {fid}: {error}')
            base=gui._compute_node_delta_base(xyz,t).cpu().numpy();delta=full-vertices-base
            def edge_motion(d):return describe(np.linalg.norm(d[edges[:,0]]-d[edges[:,1]],axis=1))
            report['frames'][fid]={'raw_reproduction_max_error':error,
                'base_displacement':describe(np.linalg.norm(base,axis=1)),
                'delta_displacement':describe(np.linalg.norm(delta,axis=1)),
                'base_adjacent_displacement_jump':edge_motion(base),'delta_adjacent_displacement_jump':edge_motion(delta),
                'base_only_distortion':distortion(vertices,vertices+base,faces),
                'delta_only_distortion':distortion(vertices,vertices+delta,faces),
                'full_distortion':distortion(vertices,full,faces)}
            print(fid,json.dumps(report['frames'][fid]),flush=True)
            np.savez_compressed(out/f'decomposition-{fid}.npz',base=base,delta=delta)
        for label,module in [('base',gui.deform_node_base.deform),('delta',gui.deform.deform)]:
            weights,distances,indices=module.cal_nn_weight(x=xyz,feature=gui.gaussians.feature)
            old_idx=indices.cpu().numpy();old_weights=weights.cpu().numpy();nodes=module.nodes[:,:3].cpu().numpy()
            start=time.monotonic();print('Surface paths: '+label,flush=True)
            surface=surface_weights(vertices,faces,nodes,module.node_radius.cpu().numpy(),k=module.K,
                original_indices=old_idx,node_weights=module.node_weight.cpu().numpy() if module.with_node_weight else None)
            old_surface=surface.pop('original_surface_distances');euclidean=np.linalg.norm(vertices[:,None,:]-nodes[old_idx],axis=2)
            ratio=old_surface/np.maximum(euclidean,1e-12)
            shortcut=(ratio>3)&(old_surface>.1*report['canonical_bbox_diagonal'])
            mass=(shortcut*old_weights).sum(1)
            report['mapping'][label]={'seconds':time.monotonic()-start,'components':surface['components'],
                'component_sizes':surface['component_sizes'],'node_anchor_offsets':describe(surface['anchor_offsets']),
                'original_node_euclidean_distances':describe(euclidean),'original_node_surface_distances':describe(old_surface),
                'surface_to_spatial_ratio':describe(ratio),'vertices_with_any_shortcut_fraction':float(shortcut.any(1).mean()),
                'shortcut_weight_mass':describe(mass),'mean_shortcut_weight_mass':float(mass.mean()),
                'vertices_with_fewer_than_k_nodes':surface['vertices_with_fewer_than_k_nodes']}
            np.savez_compressed(out/f'{label}-weights.npz',original_indices=old_idx,original_weights=old_weights,
                original_distances_squared=distances.cpu().numpy(),original_surface_distances=old_surface,
                surface_indices=surface['indices'],surface_weights=surface['weights'],surface_distances=surface['distances'],
                anchors=surface['anchors'],anchor_offsets=surface['anchor_offsets'],shortcut_weight_mass=mass)
            print(label,json.dumps(report['mapping'][label]),flush=True)
            atomic_json(out/'report.json',report)
    atomic_json(out/'report.json',report)


if __name__=='__main__':main()
