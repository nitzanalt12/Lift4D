import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from dashboard.demo import create
from dashboard.comparison import pair

class ComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.a=create(self.root/'a');self.b=self.root/'b/DEMO-synthetic';self.b.parent.mkdir();shutil.copytree(self.a,self.b)
    def tearDown(self):self.tmp.cleanup()
    def change(self,fn):
        p=self.b/'dashboard_exports/aligned.json';data=json.loads(p.read_text());fn(data);p.write_text(json.dumps(data));os.utime(p,(time.time()-5,time.time()-5))
    def result(self):return pair(self.a,self.b,'synthetic','dashboard_exports/aligned.json','dashboard_exports/aligned.json')
    def test_pair_by_identity_not_manifest_order(self):
        self.change(lambda m:m['frames'].reverse())
        data=self.result();self.assertEqual(data['frames'][0]['id'],'00000');self.assertTrue(all(r['paired_verified'] for r in data['frames']))
    def test_partial_view_keeps_missing_pairs_unavailable(self):
        self.change(lambda m:m.update(frames=m['frames'][:3]))
        rows=self.result()['frames'];self.assertEqual(sum(r['paired_verified'] for r in rows),3)
        self.assertFalse(rows[4]['paired_verified']);self.assertTrue(rows[4]['reason'])
    def test_camera_and_selected_object_mismatch(self):
        def camera(m):
            m['frames'][0]['input_camera']['fx']=140;m['frames'][0]['render_camera']['fx']=140
        self.change(camera);self.assertIn('camera',self.result()['frames'][0]['reason'])
        self.change(lambda m:m.update(target_object_id=2));self.assertIn('object',self.result()['frames'][1]['reason'])
    def test_input_contents_must_match_even_with_same_ids(self):
        inventory=self.b/'inventory.json';data=json.loads(inventory.read_text())
        data['synthetic']['frames_dir']=str(self.b/'input');inventory.write_text(json.dumps(data));os.utime(inventory,(time.time()-5,time.time()-5))
        p=self.b/'input/00000.jpg';p.write_bytes(b'different completed content');os.utime(p,(time.time()-5,time.time()-5))
        with self.assertRaisesRegex(ValueError,'content differs'):self.result()

if __name__=='__main__':unittest.main()
