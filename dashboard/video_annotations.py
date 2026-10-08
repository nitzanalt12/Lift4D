"""Read prepared handles; save explicit manual labels in new annotation files only."""
import datetime
import hashlib
import json
import uuid
from . import artifacts as a
from experiment2.video_anchors import validate_annotations


def definition(root,identity):
    path=a.inside(root,identity)
    if not path.is_relative_to((root/'video_anchors').resolve()):raise ValueError('Not a prepared video-anchor definition')
    data=a.read_json(path/'definition.json')
    payload={k:v for k,v in data.items() if k!='sha256'}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if data.get('schema')!=1 or digest!=data.get('sha256'):raise ValueError('Unverified prepared definition')
    return path,data


def listing(root):
    result=[]
    for file in (root/'video_anchors').glob('*/definition.json'):
        identity=str(file.parent.relative_to(root))
        try:
            _,data=definition(root,identity)
            result.append({'id':identity,'sequence':data['sequence'],'frames':[f['id'] for f in data['frames']]})
        except (ValueError,OSError,KeyError):continue
    return result


def image(root,identity,fid,proposal=False):
    path,data=definition(root,identity)
    frame=next((f for f in data['frames'] if f['id']==fid),None)
    if frame is None:raise ValueError('Frame not in prepared definition')
    if proposal:return a.stable_bytes(a.inside(path,f'canonical-proposals-{fid}.png')),'image/png'
    raw=a.stable_bytes(frame['image_path'])
    if hashlib.sha256(raw).hexdigest()!=frame['image_sha256']:raise ValueError('Input image differs from prepared camera frame')
    return raw,'image/jpeg'


def initial(root,identity):
    path,data=definition(root,identity)
    candidates=list((path/'annotations').glob('*.json'))
    candidate=max(candidates,key=lambda p:p.stat().st_mtime) if candidates else path/'annotations-assistant-draft.json'
    if candidate.is_file():
        result=a.read_json(candidate)
        if result.get('definition_sha256')!=data['sha256']:raise ValueError('Annotation definition differs')
        return {'annotations':result,'path':str(candidate.relative_to(root))}
    return {'annotations':a.read_json(path/'annotations-template.json'),'path':None}


def save(root,identity,data):
    path,prepared=definition(root,identity);validate_annotations(data,prepared)
    # Every save creates a distinct immutable label version. Never overwrite a
    # source image, definition, draft, prior label version or experiment result.
    directory=path/'annotations';directory.mkdir(exist_ok=True)
    result=dict(data);result['saved_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    dest=directory/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:12]+'.json')
    temp=dest.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(dest)
    return {'path':str(dest.relative_to(root)),'absolute_path':str(dest),'visible_observations':sum(row.get('visible') is True for row in result['observations']),
            'note':'Saved annotation version only; no fitting or SLURM submission triggered'}
