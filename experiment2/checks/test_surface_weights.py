import unittest
import numpy as np
from experiment2.surface_weights import surface_weights


class SurfaceWeightChecks(unittest.TestCase):
    def test_folded_strip_uses_connectivity_instead_of_spatial_shortcut(self):
        # Two nearby ends connected by a long U-shaped triangulated strip.
        center=np.array([[0.,0,0],[1,0,0],[2,0,0],[2,.1,0],[1,.1,0],[0,.1,0]])
        vertices=np.repeat(center,2,axis=0)
        vertices[1::2,2]=.01
        faces=[]
        for a in range(0,len(vertices)-2,2):faces.extend([[a,a+1,a+2],[a+1,a+3,a+2]])
        nodes=vertices[[0,4]];query=10
        spatial=np.linalg.norm(vertices[query]-nodes,axis=1)
        self.assertEqual(int(spatial.argmin()),0)
        result=surface_weights(vertices,np.array(faces),nodes,np.ones(2),k=1)
        self.assertEqual(int(result['indices'][query,0]),1)
        self.assertAlmostEqual(float(result['weights'][query].sum()),1)
    def test_disconnected_surfaces_do_not_share_nodes(self):
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,.01],[1,0,.01],[0,1,.01]])
        faces=np.array([[0,1,2],[3,4,5]])
        result=surface_weights(vertices,faces,vertices[[0,3]],np.ones(2),k=2,batch_size=1)
        np.testing.assert_array_equal(result['weights'],[[1,0]]*6)
        np.testing.assert_array_equal(result['indices'][:,0],[0,0,0,1,1,1])
        self.assertEqual(result['vertices_with_fewer_than_k_nodes'],6)
        with self.assertRaises(ValueError):surface_weights(vertices,faces,vertices[[0]],np.ones(1),k=1)
        unavailable=surface_weights(vertices,faces,vertices[[0]],np.ones(1),k=1,allow_unattached=True)
        np.testing.assert_array_equal(unavailable['unattached_vertices'],[3,4,5])
        np.testing.assert_array_equal(unavailable['weights'][3:],0)


if __name__=='__main__':unittest.main()
