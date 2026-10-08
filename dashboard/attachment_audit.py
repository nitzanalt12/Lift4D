"""Read only completed, allowlisted diagnostic exports; never run a model."""
import numpy as np
from . import artifacts as a


def exports(root):
    result=[]
    for path in (root/'attachment_audit').glob('*/audit.json'):
        try:
            data=a.read_json(path)
            if data.get('schema')==1 and data.get('complete'):
                result.append({'id':str(path.parent.relative_to(root)),'sequence':data['sequence'],'created_utc':data.get('created_utc')})
        except (OSError,ValueError,KeyError):continue
    return sorted(result,key=lambda row:row['created_utc'] or '')


def read(root,identity):
    path=a.inside(root,identity)
    if not path.is_relative_to((root/'attachment_audit').resolve()):raise ValueError('Not an attachment audit')
    manifest=a.read_json(path/'audit.json')
    if not manifest.get('complete') or manifest.get('schema')!=1:raise ValueError('Audit export is not complete')
    return path,manifest


def asset(root,identity,filename):
    path,manifest=read(root,identity);descriptors={}
    def visit(value):
        if isinstance(value,dict):
            if all(k in value for k in ['file','dtype','shape']):descriptors[value['file']]=value
            else:
                for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(manifest)
    if filename not in descriptors:raise ValueError('Asset not declared in completed audit')
    data=a.stable_bytes(a.inside(path,filename));desc=descriptors[filename]
    if len(data)!=int(np.prod(desc['shape']))*np.dtype(desc['dtype']).itemsize:raise ValueError('Incomplete diagnostic buffer')
    return data
