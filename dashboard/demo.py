"""Generate clearly labeled synthetic saved artifacts; never runs LIFT4D."""
import argparse
import json
from pathlib import Path
import os
import time
import numpy as np
from PIL import Image


def create(root):
    run=Path(root)/'DEMO-synthetic';run.mkdir(parents=True,exist_ok=False)
    for sub in ['input','masks','renders','alpha','dashboard_exports']:
        (run/sub).mkdir()
    ids=[f'{i:05d}' for i in range(12)]
    camera={'model':'pinhole','width':160,'height':96,'fx':130,'fy':130,'cx':80,'cy':48,
            'world_to_camera':np.eye(4).tolist()}
    frames=[]
    y,x=np.mgrid[:96,:160]
    for i,fid in enumerate(ids):
        gt=(x-(45+i*4))**2+(y-48)**2<22**2
        pred=(x-(45+i*4+(10 if i==7 else 2)))**2+(y-48)**2<22**2
        t=np.full((96,160,3),32,dtype=np.uint8);t[gt]=[90,190,220]
        r=np.full_like(t,255);r[pred]=[100,180,215]
        for sub,pixels in [('input',t),('masks',gt.astype(np.uint8)),('renders',r),('alpha',pred.astype(np.uint8)*255)]:
            Image.fromarray(pixels).save(run/sub/(fid+('.jpg' if sub=='input' else '.png')))
        frames.append({'input_id':fid,'render_id':fid,'input_time_seconds':i/8,'render_time_seconds':i/8,
                       'input_camera':camera,'render_camera':camera,'rgb':f'renders/{fid}.png','alpha':f'alpha/{fid}.png'})
    values={'inventory.json':{'synthetic':{'frames':ids,'frames_dir':str((run/'input').resolve()),'masks_dir':str((run/'masks').resolve())}},
            'metadata.json':{'demo':True,'commit':'DEMO — not a LIFT4D result','selected_objects':['synthetic']},
            'config.json':{'demo':True,'description':'Synthetic moving circle; intentionally bad frame 00007'},
            'dashboard_exports/aligned.json':{'schema':1,'sequence':'synthetic','target_object_id':1,'camera_space':'input',
                                             'stage':'DEMO','checkpoint':'DEMO','label':'DEMO · Synthetic moving circle','frames':frames}}
    for name,value in values.items():
        (run/name).write_text(json.dumps(value,indent=2)+'\n')
    # Demo files are complete; age them beyond the settling window.
    for p in run.rglob('*'):
        if p.is_file():
            os.utime(p,(time.time()-5,time.time()-5))
    return run


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',default='runs/dashboard-demo')
    args=p.parse_args();print(create(args.runs_root))
