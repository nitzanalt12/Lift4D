import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from dashboard.video_annotations import definition,save


class AnnotationStorageChecks(unittest.TestCase):
    def test_save_new_versions_without_mutating_sources_and_reject_hidden_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'video_anchors/camel';directory.mkdir(parents=True)
            d={'schema':1,'sequence':'camel','handles':[{'id':'foot'}],
               'frames':[{'id':'00000','image_sha256':'image','camera':{'width':854,'height':480}}]}
            d['sha256']=hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest()
            file=directory/'definition.json';file.write_text(json.dumps(d));os.utime(file,(time.time()-5,time.time()-5));before=file.read_bytes()
            a={'schema':1,'definition_sha256':d['sha256'],'annotator':'manual tester','annotation_source':'explicit manual clicks',
               'observations':[{'frame_id':'00000','handle_id':'foot','visible':True,'identity_verified':True,'xy':[321,400],'confidence':.8,'image_sha256':'image'}]}
            first=save(root,'video_anchors/camel',a);second=save(root,'video_anchors/camel',a)
            self.assertNotEqual(first['path'],second['path']);self.assertEqual(file.read_bytes(),before)
            self.assertEqual(len(list((directory/'annotations').glob('*.json'))),2)
            a['observations'][0]['visible']=False
            with self.assertRaises(ValueError):save(root,'video_anchors/camel',a)
            d['frames'][0]['camera']['width']=1;file.write_text(json.dumps(d));os.utime(file,(time.time()-5,time.time()-5))
            with self.assertRaises(ValueError):definition(root,'video_anchors/camel')


if __name__=='__main__':unittest.main()
