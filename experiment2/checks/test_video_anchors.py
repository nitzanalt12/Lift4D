import unittest
import numpy as np
from experiment2.anchor_arap import VideoAnchorARAP
from experiment2.arap import ARAP
from experiment2.video_anchors import project,validate_annotations


class VideoAnchorChecks(unittest.TestCase):
    def setUp(self):
        self.rest=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[0,0,.5]])
        self.faces=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        self.handles=[{'id':str(i),'vertex_ids':[i],'weights':[1.]} for i in range(2)]
        self.camera={'width':640,'height':480,'fx':500.,'fy':500.,'cx':320.,'cy':240.}
        self.linear=np.eye(3);self.translation=np.array([0.,0,8.])
    def test_jacobian_matches_numerical_projection(self):
        pixels,j=project(self.rest,self.linear,self.translation,self.camera,True)
        for axis in range(3):
            q=self.rest.copy();q[:,axis]+=1e-6
            numeric=(project(q,self.linear,self.translation,self.camera)-pixels)/1e-6
            np.testing.assert_allclose(j[:,:,axis],numeric,rtol=1e-6,atol=1e-5)
    def test_zero_weight_preserves_original_arap(self):
        target=self.rest.copy();target[0]+=[1,0,0]
        obs=[{'handle_id':'0','xy':[250.,150.],'confidence':1}]
        result,_=VideoAnchorARAP(self.rest,self.faces,100,self.handles).solve_anchored(target,obs,self.linear,self.translation,self.camera,keypoint_weight=0,iterations=10)
        original,_=ARAP(self.rest,self.faces,100).solve(target,iterations=10)
        np.testing.assert_allclose(result,original,atol=1e-5)
    def test_visible_handle_error_reduced_with_fixed_camera(self):
        xy=project(self.rest[:1],self.linear,self.translation,self.camera)[0]+[35.,0.]
        obs=[{'handle_id':'0','xy':xy.tolist(),'confidence':1}]
        _,report=VideoAnchorARAP(self.rest,self.faces,100,self.handles).solve_anchored(self.rest,obs,self.linear,self.translation,self.camera,keypoint_weight=100,iterations=30)
        self.assertLess(report['final']['reprojection_mean_px'],report['initial']['reprojection_mean_px']/2)
        self.assertLessEqual(report['final']['total'],report['initial']['total'])
    def test_hidden_and_unverified_observations_do_not_become_targets(self):
        definition={'sha256':'definition','handles':[{'id':'foot'}],'frames':[{'id':'00000','camera':self.camera,'image_sha256':'image'}]}
        data={'schema':1,'definition_sha256':'definition','annotator':'test','annotation_source':'manual',
              'observations':[{'handle_id':'foot','frame_id':'00000','visible':False,'xy':None}]}
        with self.assertRaises(ValueError):validate_annotations(data,definition)
        data['observations'][0].update(visible=True,identity_verified=False,xy=[200,200],image_sha256='image')
        with self.assertRaises(ValueError):validate_annotations(data,definition)
        data['observations'][0]['identity_verified']=True
        self.assertEqual(list(validate_annotations(data,definition)),['00000'])
    def test_weighted_prior_uses_recorded_confidence_and_rejects_nonpositive_weights(self):
        prior=np.array([.001,.001,1,1,1])
        solver=VideoAnchorARAP(self.rest,self.faces,100,self.handles,prior_weights=prior)
        displaced=self.rest.copy();displaced[0,0]+=1
        energy=solver.energies(displaced,self.rest)
        self.assertAlmostEqual(energy['anchor'],.001/(len(self.rest)*solver.scale**2))
        with self.assertRaises(ValueError):VideoAnchorARAP(self.rest,self.faces,100,self.handles,prior_weights=np.zeros(5))


if __name__=='__main__':unittest.main()
