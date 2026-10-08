import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from dashboard.catalog import build
from dashboard.index_outputs import create as index_outputs

class CatalogTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def write(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));os.utime(path,(time.time()-5,time.time()-5))
    def make_run(self,name='camel-final',experiment='2.1',smoke=False):
        path=self.root/'mesh'/name
        self.write(path/'metadata.json',{'experiment':experiment,'selected_objects':['camel'],'created_utc':'2026-10-08T12:00:00Z'})
        self.write(path/'config.json',{'experiment':experiment,'smoke_check':smoke})
        self.write(path/'inventory.json',{'camel':{'frames':['00000'],'frames_dir':'input','masks_dir':'masks'}})
        return path
    def manifest(self,run,stage='appearance',checkpoint=30000,complete=True):
        self.write(run/'dashboard_exports/aligned.json',{'schema':1,'sequence':'camel','stage':stage,'checkpoint':checkpoint,'frames':[],'complete':complete})
    def test_group_checkpoint_prefer_manifest_and_hide_smoke(self):
        run=self.make_run();self.manifest(run)
        (run/'models/davis_camel_node_delta/comparison_iter_030000').mkdir(parents=True)
        smoke=self.make_run('gradient-check','2.3',True);self.manifest(smoke,'mesh_finetune',2)
        result=build(self.root)
        self.assertEqual(len(result['entries']),1)
        selected=result['entries'][0]
        self.assertEqual(selected['animal'],'camel');self.assertEqual(selected['experiment'],'2.1')
        self.assertEqual(selected['checkpoint'],'appearance:30000');self.assertEqual(selected['kind'],'manifest')
        self.assertEqual(len(selected['alternative_views']),1)
        self.assertEqual(result['excluded_auxiliary_runs'],1)
        self.assertEqual(len(build(self.root,True)['entries']),2)
    def test_repeated_runs_stay_distinct_and_partial_is_visible(self):
        self.manifest(self.make_run('attempt-a'))
        self.manifest(self.make_run('attempt-b'),complete=False)
        entries=build(self.root)['entries']
        self.assertEqual(len(entries),2);self.assertEqual(entries[0]['checkpoint'],entries[1]['checkpoint'])
        self.assertIn('partial',entries[1]['checkpoint_label'])
    def test_index_keeps_sources_unchanged_and_is_idempotent(self):
        run=self.make_run();self.manifest(run)
        before={str(p):p.read_bytes() for p in run.rglob('*') if p.is_file()}
        first=index_outputs(self.root);second=index_outputs(self.root)
        self.assertEqual(first,second)
        self.assertEqual(before,{str(p):p.read_bytes() for p in run.rglob('*') if p.is_file()})
        links=list((self.root/'results').glob('camel/2.1/appearance-30000/*/run'))
        self.assertEqual(len(links),1);self.assertTrue(links[0].is_symlink());self.assertEqual(links[0].resolve(),run.resolve())
        self.assertEqual(len(build(self.root)['entries']),1)
    def test_arap_partial_run_keeps_source_checkpoint_identity(self):
        run=self.make_run('arap-active','2.4')
        self.write(run/'config.json',{'experiment':'2.4','source_checkpoint':30000})
        self.write(run/'metadata.json',{'experiment':'2.4','selected_objects':['camel'],'display_label':'ARAP λ=10'})
        (run/'logs').mkdir();(run/'logs/projection.jsonl').write_text('in progress')
        entry=build(self.root)['entries'][0]
        self.assertEqual(entry['checkpoint'],'arap:30000')
        self.assertEqual(entry['execution_label'],'ARAP λ=10')
        self.assertEqual(entry['kind'],'missing')

    def test_finetune_checkpoint_without_render_is_selectable(self):
        run=self.make_run('trained','2.3');self.manifest(run,'mesh_finetune',1000)
        checkpoint=run/'checkpoints/delta-000250.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'atomic fixture')
        os.utime(checkpoint,(time.time()-5,time.time()-5))
        entries=build(self.root)['entries']
        missing=next(e for e in entries if e['iteration']==250)
        self.assertEqual(missing['kind'],'missing');self.assertEqual(missing['view'],'')
        self.assertIn('renders unavailable',missing['checkpoint_label'])

    def test_unlaunched_baseline_stub_hidden_but_active_producer_visible(self):
        run=self.make_run('queued','baseline')
        self.assertEqual(build(self.root)['entries'],[])
        (run/'logs').mkdir();(run/'logs/camel_geometry.log').write_text('starting')
        entries=build(self.root)['entries'];self.assertEqual(len(entries),1);self.assertEqual(entries[0]['kind'],'missing')

if __name__=='__main__':unittest.main()
