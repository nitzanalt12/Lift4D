"""Fixed mesh handles, verified cameras and explicit visible video observations."""
import numpy as np


def project(points,linear,translation,camera,jacobian=False):
    q=np.asarray(points)@np.asarray(linear).T+translation
    if not np.isfinite(q).all() or np.any(q[:,2]<=.001):raise ValueError('Handle behind camera / invalid coordinates')
    x,y,z=q.T
    pixels=np.column_stack((camera['fx']*x/z+camera['cx'],camera['fy']*y/z+camera['cy']))
    if not jacobian:return pixels
    j=np.zeros((len(q),2,3));j[:,0,0]=camera['fx']/z;j[:,1,1]=camera['fy']/z
    j[:,0,2]=-camera['fx']*x/z**2;j[:,1,2]=-camera['fy']*y/z**2
    return pixels,j@linear


def validate_annotations(data,definition):
    """No guessed observations, interpolation, duplicate identities or silent resize."""
    if data.get('schema')!=1 or data.get('definition_sha256')!=definition['sha256']:
        raise ValueError('Annotations must identify the exact prepared definition')
    if not data.get('annotator') or not data.get('annotation_source'):raise ValueError('Explicit annotation provenance required')
    handles={h['id']:h for h in definition['handles']};frames={f['id']:f for f in definition['frames']}
    seen=set();active={}
    for row in data.get('observations',[]):
        fid=row['frame_id'];hid=row['handle_id'];key=(fid,hid)
        if key in seen:raise ValueError('Duplicate frame / handle observation')
        seen.add(key)
        if fid not in frames or hid not in handles:raise ValueError('Unknown explicit frame / handle ID')
        if row.get('visible') is not True:
            if row.get('xy') is not None:raise ValueError('Hidden/uncertain handles must not have a target position')
            continue
        if row.get('identity_verified') is not True:raise ValueError('Visible observation needs verified identity')
        xy=np.asarray(row['xy'],dtype=float);w=float(row.get('confidence',1))
        camera=frames[fid]['camera'];width,height=camera['width'],camera['height']
        if xy.shape!=(2,) or not np.isfinite(xy).all() or not (0<=xy[0]<width and 0<=xy[1]<height):raise ValueError('Target outside original image resolution')
        if not 0<w<=1:raise ValueError('Confidence must be in (0,1]')
        if row.get('image_sha256')!=frames[fid]['image_sha256']:raise ValueError('Input image provenance differs')
        active.setdefault(fid,[]).append(row)
    if not active:raise ValueError('No visible, identity-verified observations; refusing unanchored fitting')
    return active
