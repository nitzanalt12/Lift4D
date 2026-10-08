import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import torch
import trimesh
from experiment2.mesh import load_glb,save_vertices
from experiment2.checkpoints import load_bundle,attach_frozen,inspect_checkpoint


class Preparation(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def age(self,p):os.utime(p,(time.time()-5,time.time()-5))
    def test_mesh_preserves_local_vertices_scene_and_topology(self):
        scene=trimesh.Scene();mesh=trimesh.creation.box();tf=np.eye(4);tf[:3,3]=[1,2,3]
        scene.add_geometry(mesh,geom_name='box',node_name='instance',transform=tf)
        p=self.root/'result.glb';p.write_bytes(scene.export(file_type='glb'));self.age(p)
        before=p.read_bytes();data=load_glb(p)
        geometry=next(iter(data['geometries'].values()))
        np.testing.assert_allclose(geometry['vertices'],mesh.vertices)
        np.testing.assert_array_equal(geometry['faces'],mesh.faces)
        np.testing.assert_allclose(data['instances'][0]['scene_from_local'],tf)
        records=save_vertices(data,self.root/'arrays')
        with np.load(self.root/'arrays'/records[0]['arrays']) as result:
            np.testing.assert_array_equal(result['vertices'],geometry['vertices'])
        self.assertEqual(before,p.read_bytes())
    def test_partial_glb_rejected(self):
        p=self.root/'partial.glb';p.write_bytes(b'glTF'+b'\0'*8);self.age(p)
        with self.assertRaises(ValueError):load_glb(p)
    def fixture(self,stage):
        for name,value in [('inventory.json',{'rhino':{'frames':['00000']}}),
                           ('config.json',{'appearance_args':['--optimize_per_frame_compose_transforms_app']})]:
            p=self.root/name;p.write_text(json.dumps(value));self.age(p)
        suffix='node' if stage=='geometry' else 'node_delta'
        checkpoint=self.root/'models'/f'davis_rhino_{suffix}'/'deform_gs/iteration_20000';checkpoint.mkdir(parents=True)
        model=torch.nn.Linear(3,3)
        files={'deform.pth':model.state_dict()}
        if stage=='appearance':
            files['deform_node_base.pth']=model.state_dict()
            files['compose_transforms_app.pt']={'compose_app_scale':torch.ones(1),'compose_app_idx':{0:0}}
        for name,value in files.items():
            p=checkpoint/name;torch.save(value,p);self.age(p)
        p=checkpoint/'gaussians.ply';p.write_bytes(b'fixture only');self.age(p)
        log=self.root/'logs'/f'rhino_{stage}.log';log.parent.mkdir(exist_ok=True)
        log.write_text('Still writing');self.age(log)
        return model,checkpoint,log
    def test_complete_checkpoint_only_and_cpu_states(self):
        model,path,log=self.fixture('geometry')
        with self.assertRaises(ValueError):load_bundle(self.root,'rhino','geometry',20000)
        log.write_text(f'[ITER 20000] Checkpoint saved to {path}\n');self.age(log)
        bundle=load_bundle(self.root,'rhino','geometry',20000)
        self.assertEqual(set(bundle['states']),{'deform.pth'})
        self.assertTrue(all(v.device.type=='cpu' for v in bundle['states']['deform.pth'].values()))
        other=torch.nn.Linear(3,3);attach_frozen(other,bundle['states']['deform.pth'])
        self.assertFalse(other.training);self.assertTrue(all(not p.requires_grad for p in other.parameters()))
        np.testing.assert_array_equal(other.weight.detach(),model.weight.detach())
    def test_appearance_requires_base_and_compose(self):
        _,path,log=self.fixture('appearance')
        log.write_text(f'[ITER 20000] Checkpoint saved to {path}\n');self.age(log)
        bundle=load_bundle(self.root,'rhino','appearance',20000)
        self.assertEqual(set(bundle['states']),{'deform.pth','deform_node_base.pth','compose_transforms_app.pt'})
        (path/'deform_node_base.pth').unlink()
        self.assertFalse(inspect_checkpoint(self.root,'rhino','appearance',20000)['ready'])
    def test_no_implicit_parameter_remapping(self):
        a=torch.nn.Linear(3,3);before=a.weight.detach().clone()
        with self.assertRaises(ValueError):attach_frozen(a,torch.nn.Linear(4,3).state_dict())
        self.assertTrue(torch.equal(before,a.weight))
        state=a.state_dict();state['unexpected']=torch.zeros(1)
        with self.assertRaises(ValueError):attach_frozen(a,state)


if __name__=='__main__':unittest.main()
