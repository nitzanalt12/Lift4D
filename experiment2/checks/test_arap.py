import unittest
import numpy as np
from experiment2.arap import ARAP,distortion

class ARAPChecks(unittest.TestCase):
    def setUp(self):
        self.rest=np.array([[0.,0.,0.],[1,0,0],[1,1,0],[0,1,0],[.5,.5,.5]])
        self.faces=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,3,2],[0,2,1]])
    def test_rigid_transform_preserved(self):
        angle=.7;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        target=self.rest@r.T+[4,2,-3]
        result,report=ARAP(self.rest,self.faces,100).solve(target)
        np.testing.assert_allclose(result,target,atol=2e-5)
        self.assertLess(report['final']['arap'],1e-10)
    def test_stretched_vertex_repaired_and_energy_decreased(self):
        target=self.rest.copy();target[4,2]=4
        result,report=ARAP(self.rest,self.faces,100).solve(target,iterations=30)
        self.assertLess(report['final']['total'],report['initial']['total'])
        self.assertLess(distortion(self.rest,result,self.faces)['edge_stretch_p95'],distortion(self.rest,target,self.faces)['edge_stretch_p95'])
        self.assertGreater(np.linalg.norm(result-self.rest),0)
    def test_proper_rotations_and_invalid_input(self):
        solver=ARAP(self.rest,self.faces,10)
        r=solver.rotations(self.rest.copy()).numpy()
        np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-5)
        with self.assertRaises(ValueError):solver.solve(np.full_like(self.rest,np.nan))

if __name__=='__main__':unittest.main()
