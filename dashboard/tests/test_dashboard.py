import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dashboard.artifacts import frame_index, arrays, stable_bytes, inside, views
from dashboard.demo import create
from dashboard.metrics import evaluate_arrays, cached_evaluate


class Metrics(unittest.TestCase):
    def test_identical_and_empty(self):
        rgb=np.zeros((8,8,3),np.uint8);mask=np.zeros((8,8),bool);mask[2:6,2:6]=True
        r=evaluate_arrays(rgb,rgb,mask,mask.astype(float))
        self.assertEqual(r['iou'],1);self.assertEqual(r['boundary_mean'],0)
        self.assertEqual(r['boundary_p95'],0);self.assertEqual(r['psnr'],'Infinity')
        r=evaluate_arrays(rgb,rgb,np.zeros_like(mask),np.zeros_like(mask))
        self.assertEqual(r['iou'],1);self.assertIsNone(r['boundary_mean']);self.assertIsNone(r['psnr'])
        r=evaluate_arrays(rgb,rgb,mask,np.zeros_like(mask))
        self.assertEqual(r['iou'],0);self.assertIsNone(r['boundary_mean'])

    def test_boundary_translation_and_roi(self):
        rgb=np.zeros((5,6,3),np.uint8);r=rgb.copy();r[:]=255
        gt=np.zeros((5,6),bool);gt[2,2]=True;pred=np.zeros_like(gt);pred[2,4]=True
        value=evaluate_arrays(rgb,r,gt,pred.astype(float))
        self.assertEqual(value['boundary_mean'],2);self.assertEqual(value['boundary_p95'],2)
        self.assertEqual(value['psnr'],0)
        r[2,2]=0;self.assertEqual(evaluate_arrays(rgb,r,gt,pred.astype(float))['psnr'],'Infinity')

    def test_threshold_lpips_background_and_dimensions(self):
        rgb=np.ones((5,5,3),np.uint8)*255;mask=np.zeros((5,5),bool);mask[2,2]=True
        alpha=np.zeros((5,5));alpha[2,2]=0.5
        def score(t,r,g):
            self.assertTrue(np.all(t[~g]==0.5));self.assertTrue(np.all(r[~g]==0.5));return 0.2
        value=evaluate_arrays(rgb,rgb,mask,alpha,0.5,score)
        self.assertEqual(value['iou'],1);self.assertEqual(value['lpips'],0.2)
        self.assertEqual(evaluate_arrays(rgb,rgb,mask,alpha,0.51)['iou'],0)
        with self.assertRaises(ValueError):evaluate_arrays(rgb,rgb[:3],mask,alpha)

    def test_missing_alpha_does_not_hide_appearance(self):
        rgb=np.zeros((8,8,3),np.uint8);mask=np.ones((8,8),bool)
        result=evaluate_arrays(rgb,rgb,mask,None)
        self.assertIsNone(result['iou']);self.assertIsNone(result['boundary_mean'])
        self.assertEqual(result['psnr'],'Infinity')
        def broken(*args):raise ValueError('LPIPS dimensions unsupported')
        result=evaluate_arrays(rgb,rgb,mask,mask,lpips_fn=broken)
        self.assertEqual(result['iou'],1);self.assertIsNone(result['lpips'])


class Artifacts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.run=create(self.root)
    def tearDown(self):self.tmp.cleanup()
    def index(self):return frame_index(self.run,'synthetic','dashboard_exports/aligned.json')[0]
    def mutate(self,change):
        p=self.run/'dashboard_exports/aligned.json';m=json.loads(p.read_text());change(m);p.write_text(json.dumps(m))
        import os,time
        os.utime(p,(time.time()-5,time.time()-5))
    def test_explicit_mapping_not_file_order(self):
        self.mutate(lambda m:m['frames'].reverse())
        records=self.index();self.assertEqual(records[0]['id'],'00000');self.assertTrue(all(r['alignment'] for r in records))
        self.assertEqual(Path(records[4]['render']).stem,'00004')
    def test_reject_camera_time_and_id(self):
        self.mutate(lambda m:m['frames'][0].update(render_camera={'wrong':True}))
        self.assertFalse(self.index()[0]['alignment'])
        self.mutate(lambda m:m['frames'][1].update(render_time_seconds=100))
        self.assertFalse(self.index()[1]['alignment'])
        self.mutate(lambda m:m['frames'][2].update(render_id='wrong'))
        self.assertFalse(self.index()[2]['alignment'])
    def test_unknown_and_duplicate(self):
        self.mutate(lambda m:m['frames'].append(m['frames'][0]))
        with self.assertRaises(ValueError):self.index()

    def test_time_order_and_camera_dimensions(self):
        self.mutate(lambda m:m['frames'][1].update(input_time_seconds=0,render_time_seconds=0))
        self.assertFalse(any(r['alignment'] for r in self.index()))
        self.mutate(lambda m:m['frames'][1].update(input_time_seconds=0.125,render_time_seconds=0.125))
        def wrong_size(m):
            m['frames'][0]['input_camera']['width']=100
            m['frames'][0]['render_camera']['width']=100
        self.mutate(wrong_size)
        with self.assertRaises(ValueError):arrays(self.index()[0])

    def test_resolutions_and_cache_source_changes(self):
        from PIL import Image
        import os,time
        settings={'object_id':1,'alpha_threshold':0.5,'lpips':False}
        before=cached_evaluate(self.index(),settings,self.root/'cache')
        p=self.run/'renders/00000.png';pixels=np.array(Image.open(p));pixels[:]=0
        Image.fromarray(pixels).save(p);os.utime(p,(time.time()-5,time.time()-5))
        after=cached_evaluate(self.index(),settings,self.root/'cache')
        self.assertNotEqual(before['frames'][0]['cache_key'],after['frames'][0]['cache_key'])
        Image.fromarray(pixels[:40]).save(p);os.utime(p,(time.time()-5,time.time()-5))
        with self.assertRaises(ValueError):arrays(self.index()[0])
    def test_incomplete_and_escape(self):
        import os,time
        p=self.run/'broken.png';p.write_bytes(b'incomplete')
        with self.assertRaises(ValueError):stable_bytes(p)
        os.utime(p,(time.time()-5,time.time()-5))
        with self.assertRaisesRegex(ValueError,'finished writing'):stable_bytes(p)
        with self.assertRaises(ValueError):inside(self.run,'../outside')

    def test_video_requires_encoder_completion(self):
        import os,time
        p=self.run/'sam3d/davis_synthetic/comparison.mp4';p.parent.mkdir(parents=True)
        p.write_bytes(b'synthetic-test-video-bytes')
        log=self.run/'logs/synthetic_reconstruction.log';log.parent.mkdir()
        log.write_text('Rendering comparison video...')
        for f in [p,log]:os.utime(f,(time.time()-5,time.time()-5))
        self.assertFalse(any(v['kind']=='video' for v in views(self.run,'synthetic')))
        log.write_text(f'Comparison video saved: {p.resolve()}\n')
        os.utime(log,(time.time()-5,time.time()-5))
        video=next(v for v in views(self.run,'synthetic') if v['kind']=='video')
        records,_,_=frame_index(self.run,'synthetic',video['id'])
        self.assertFalse(any(r['alignment'] for r in records))
    def test_cache_settings_and_missing(self):
        settings={'object_id':1,'alpha_threshold':0.5,'lpips':False}
        a=cached_evaluate(self.index(),settings,self.root/'cache')
        b=cached_evaluate(self.index(),settings,self.root/'cache')
        self.assertEqual(a,b);self.assertEqual(len(list((self.root/'cache').glob('*.json'))),12)
        c=cached_evaluate(self.index(),{**settings,'alpha_threshold':0.7},self.root/'cache')
        self.assertNotEqual(a['frames'][0]['cache_key'],c['frames'][0]['cache_key'])
        (self.run/'alpha/00000.png').unlink()
        d=cached_evaluate(self.index(),settings,self.root/'cache')
        self.assertIsNone(d['frames'][0]['iou']);self.assertIn('unavailable',d['frames'][0]['reasons'])
    def test_native_panel_and_non_contiguous_ids(self):
        from PIL import Image
        import os,time
        p=self.run/'models/davis_synthetic_node/comparison_iter_020000';p.mkdir(parents=True)
        pixels=np.zeros((252,480,3),np.uint8);pixels[156:252,160:320]=123
        f=p/'frame_0000.png';Image.fromarray(pixels).save(f);os.utime(f,(time.time()-5,time.time()-5))
        records,_,_=frame_index(self.run,'synthetic',str(p.relative_to(self.run)))
        self.assertFalse(records[0]['alignment']);self.assertEqual(arrays(records[0])[1].shape,(96,160,3))
        self.assertTrue(np.all(arrays(records[0])[1]==123))
        inv=self.run/'inventory.json';value=json.loads(inv.read_text());value['synthetic']['frames'][0]='00100'
        inv.write_text(json.dumps(value));os.utime(inv,(time.time()-5,time.time()-5))
        records,_,_=frame_index(self.run,'synthetic',str(p.relative_to(self.run)))
        self.assertIsNone(records[0]['render'])


if __name__=='__main__':unittest.main()
