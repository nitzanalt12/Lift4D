import unittest
import numpy as np
from experiment2.transfer import GLB_FROM_SOURCE, nearest_features

class TransferTests(unittest.TestCase):
    def test_export_rotation_roundtrip(self):
        source=np.array([[1.,2.,3.],[-2.,1.,4.]])
        np.testing.assert_allclose((source@GLB_FROM_SOURCE)@GLB_FROM_SOURCE.T,source)
    def test_mapping_is_saved_canonical_spatial_nearest(self):
        vertices=np.array([[.1,0,0],[1.8,0,0]])
        xyz=np.array([[0,0,0],[2,0,0]])
        features=np.array([[7,8],[9,10]])
        indices,distance,mapped=nearest_features(vertices,xyz,features)
        np.testing.assert_array_equal(indices,[0,1]);np.testing.assert_allclose(distance,[.1,.2])
        np.testing.assert_array_equal(mapped,features)
    def test_bad_coordinates_rejected(self):
        with self.assertRaises(ValueError):nearest_features(np.array([[np.nan,0,0]]),np.zeros((1,3)),np.ones((1,8)))

if __name__=='__main__':unittest.main()
