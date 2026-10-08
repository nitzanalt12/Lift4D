"""Preserve GLB mesh coordinates, topology and scene transforms without alignment."""
import hashlib
import io
from pathlib import Path
import struct
import time
import numpy as np


def snapshot(path):
    path=Path(path)
    before=path.stat()
    if time.time()-before.st_mtime<2:
        raise ValueError('Artifact is still settling')
    data=path.read_bytes()
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Artifact changed during inspection')
    return data


def load_glb(path):
    import trimesh
    data=snapshot(path)
    if len(data)<12 or data[:4]!=b'glTF' or struct.unpack('<II',data[4:12])!=(2,len(data)):
        raise ValueError('GLB is incomplete or is not version 2')
    scene=trimesh.load(io.BytesIO(data),file_type='glb',force='scene',process=False)
    geometries={}
    for name,geometry in scene.geometry.items():
        if not isinstance(geometry,trimesh.Trimesh):
            raise ValueError(f'Unsupported non-triangle geometry: {name}')
        vertices=np.asarray(geometry.vertices).copy()
        faces=np.asarray(geometry.faces).copy()
        if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
            raise ValueError('Invalid mesh vertices')
        if faces.ndim!=2 or faces.shape[1]!=3 or (faces.size and (faces.min()<0 or faces.max()>=len(vertices))):
            raise ValueError('Invalid mesh topology')
        geometries[name]={'vertices':vertices,'faces':faces}
    if not geometries:
        raise ValueError('GLB has no triangle mesh')
    instances=[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node]
        if not np.isfinite(transform).all():
            raise ValueError('Invalid scene transform')
        instances.append({'node':str(node),'geometry':str(name),'scene_from_local':transform.tolist()})
    return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(data).hexdigest(),
            'geometries':geometries,'instances':instances,
            'coordinate_policy':'GLB local vertices unchanged; scene transforms recorded, not applied'}


def save_vertices(mesh,directory):
    """Write inspection arrays separately; do not re-export or modify the source GLB."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    records=[]
    for index,(name,geometry) in enumerate(mesh['geometries'].items()):
        filename=f'geometry_{index:03d}.npz'
        np.savez(directory/filename,vertices=geometry['vertices'],faces=geometry['faces'])
        vertices=geometry['vertices']
        records.append({'name':str(name),'arrays':filename,'vertices':len(vertices),'faces':len(geometry['faces']),
                        'bounds': [vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()] if len(vertices) else None})
    return records
