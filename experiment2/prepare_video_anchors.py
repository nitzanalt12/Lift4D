"""Prepare geometric foot-handle proposals and exact original camera projections."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from .transfer import ROOT,load_transfer
from .leg_audit import leg_cores,REGIONS,COLORS
from .video_anchors import project
from .mesh import snapshot
from evaluation.export_input_views import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frames',nargs='+',default=['00000','00020','00030'])
    args=p.parse_args();seed=Path(args.seed_run).resolve();out=Path(args.output).resolve()
    cfg=json.loads(snapshot(seed/'config.json'))
    if cfg['sequence']!='camel' or cfg['experiment']!='2.1':raise ValueError('Explicit camel 2.1 seed required')
    if out.is_relative_to(seed) or out.is_relative_to(Path(cfg['baseline_run'])):raise ValueError('Separate output required')
    state=load_transfer(Path(cfg['baseline_run']),'camel','appearance',30000)
    if state['bundle']['hashes']!=cfg['checkpoint_hashes']:raise ValueError('Checkpoint hashes changed')
    source=json.loads(snapshot(seed/'dashboard_exports/mesh.json'));rows={r['input_id']:r for r in source['frames']}
    labels,partition=leg_cores(state['vertices'],state['faces'],-.06)
    canonical_bytes=snapshot(seed/'mesh/canonical.npz')
    out.mkdir(parents=True,exist_ok=False)
    handles=[]
    for label in range(2,6):
        ids=np.flatnonzero(labels==label);z=state['vertices'][ids,2]
        patch=ids[z<=np.percentile(z,2)]
        handles.append({'id':f'foot-{label}','region_id':label,'label':REGIONS[label]+' · sole center proposal',
            'vertex_ids':patch.tolist(),'weights':(np.ones(len(patch))/len(patch)).tolist(),
            'color':COLORS[label],'canonical_xyz':state['vertices'][patch].mean(0).tolist(),
            'binding_source':'Mean of lowest 2% Z vertices in this canonical leg core; geometric proposal, not a skeletal joint'})
    data={'schema':1,'sequence':'camel','seed_run':str(seed),'baseline_run':cfg['baseline_run'],
        'checkpoint_hashes':cfg['checkpoint_hashes'],'canonical_sha256':hashlib.sha256(canonical_bytes).hexdigest(),
        'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'handles':handles,'frames':[],'partition':partition,'network_training':False,
        'binding_status':'Geometric foot sole proposals; inspect camera-0 overlay and canonical audit before annotation',
        'camera_policy':'Affine extracted from the exact frozen original compose and saved smoothed object transform; checked against original helper, no camera fitting'}
    import torch
    gui=state['gui'];points=torch.from_numpy(np.r_[np.zeros((1,3)),np.eye(3)*10].astype(np.float32)).cuda()
    def to_camera(xyz,i,row):
        if gui.optimize_per_frame_compose_transforms_app:xyz,_=gui.apply_compose_transform_app(xyz,i,torch.ones_like(xyz))
        transform={k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in row['object_transform'].items()}
        rot=torch.zeros((len(xyz),4),device='cuda');rot[:,0]=1
        xyz,_,_=gui.sam3d_data.apply_transform_to_gaussian(xyz,rot,torch.ones_like(xyz),transform)
        return xyz
    with torch.inference_mode():
        for fid in args.frames:
            i=state['stems'].index(fid);row=rows[fid];probe=to_camera(points,i,row).cpu().numpy().astype(float)
            linear=(probe[1:]-probe[0]).T/10;translation=probe[0]
            with np.load(seed/'mesh'/f'{fid}.npz') as saved:vertices=saved['vertices']
            subset=vertices[::101];reference=to_camera(torch.from_numpy(subset).cuda(),i,row).cpu().numpy()
            error=float(np.abs(subset@linear.T+translation-reference).max())
            if error>2e-5:raise ValueError('Camera affine failed original-transform reproduction')
            path=Path(state['inventory']['camel']['frames_dir'])/(fid+'.jpg');image_bytes=snapshot(path)
            handle_xyz=np.stack([(vertices[h['vertex_ids']]*np.asarray(h['weights'])[:,None]).sum(0) for h in handles])
            pixels=project(handle_xyz,linear,translation,row['input_camera'])
            canonical=project(np.asarray([h['canonical_xyz'] for h in handles]),linear,translation,row['input_camera'])
            record={'id':fid,'camera':row['input_camera'],'linear':linear.tolist(),'translation':translation.tolist(),
                'camera_reproduction_max_error':error,'image_path':str(path),'image_sha256':hashlib.sha256(image_bytes).hexdigest(),
                'original_handle_pixels':pixels.tolist(),'canonical_handle_pixels':canonical.tolist()}
            data['frames'].append(record)
            image=Image.open(path).convert('RGB');draw=ImageDraw.Draw(image)
            for h,xy in zip(handles,canonical):
                x,y=xy;draw.ellipse((x-5,y-5,x+5,y+5),fill=h['color']);draw.text((x+7,y-9),h['id'],fill=h['color'])
            image.save(out/f'canonical-proposals-{fid}.png')
    raw=json.dumps(data,sort_keys=True,allow_nan=False).encode();data['sha256']=hashlib.sha256(raw).hexdigest()
    atomic_json(out/'definition.json',data)
    atomic_json(out/'annotations-template.json',{'schema':1,'definition_sha256':data['sha256'],
        'annotator':'','annotation_source':'Manual clicks on original input images; fixed canonical foot patch identities',
        'observations':[]})
    print('Prepared '+str(out),flush=True)


if __name__=='__main__':main()
