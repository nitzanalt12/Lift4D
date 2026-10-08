"""Pair saved views by explicit input identity; never align or resample images."""
import hashlib
from pathlib import Path
from . import artifacts as a


def same_file(first,second):
    if Path(first).resolve()==Path(second).resolve():return True
    return hashlib.sha256(a.stable_bytes(first)).digest()==hashlib.sha256(a.stable_bytes(second)).digest()


def pair(run_a,run_b,sequence,view_a,view_b):
    records_a,_,manifest_a=a.frame_index(run_a,sequence,view_a)
    records_b,_,manifest_b=a.frame_index(run_b,sequence,view_b)
    second={record['id']:record for record in records_b}
    frames=[]
    for first in records_a:
        other=second.get(first['id']);reason=''
        if other is None:reason='Frame ID absent from B input inventory'
        else:
            # Same IDs alone are insufficient: compare actual input and mask sources.
            if not same_file(first['input'],other['input']):raise ValueError('Input image content differs between experiments: '+first['id'])
            if not same_file(first['target'],other['target']):raise ValueError('Target mask content differs between experiments: '+first['id'])
            if not first['alignment'] or not other['alignment']:reason='A or B lacks verified input-camera/time mapping'
            elif first.get('target_object_id')!=other.get('target_object_id'):reason='Selected target object differs between exports'
            elif first['proof']['input_camera']!=other['proof']['input_camera']:reason='Input camera definitions differ between experiments'
            elif (manifest_a.get('time_mapping',{}).get('kind')!=manifest_b.get('time_mapping',{}).get('kind')
                  or first['time']!=other['time']
                  or first['proof'].get('deformation_time')!=other['proof'].get('deformation_time')):reason='Input time mappings differ between experiments'
        frames.append({'id':first['id'],'paired_verified':not reason,'reason':reason,'b_present':other is not None})
    return {'frames':frames,'pairing':'explicit input IDs; input/mask content, target object, camera and time checks',
            'a_frame_count':len(records_a),'b_frame_count':len(records_b)}
