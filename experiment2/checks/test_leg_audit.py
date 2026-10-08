import unittest
import numpy as np
from experiment2.leg_audit import node_affinity,ownership,cross_leg_summary


class LegAuditChecks(unittest.TestCase):
    def test_receiver_affinity_differs_from_spatial_attachment(self):
        labels=np.array([2,2,3]);indices=np.array([[0],[0],[1]]);weights=np.ones((3,1))
        mass=node_affinity(labels,indices,weights,2,np.array([.8,.8,.5]))
        owner,confidence,support=ownership(mass)
        np.testing.assert_array_equal(owner,[2,0])
        np.testing.assert_allclose(support,[1.6,.5]);np.testing.assert_allclose(confidence,[1,1])
        report,cross=cross_leg_summary(np.array([2,3]),np.array([3,2]),np.array([[0,1],[0,1]]),np.array([[.7,.3],[.8,.2]]))
        np.testing.assert_allclose(cross,[.7,.2]);self.assertAlmostEqual(report['mean_other_leg_mass'],.45)
    def test_ambiguous_or_body_ownership_is_not_a_leg_label(self):
        mass=np.zeros((2,6));mass[0,2:4]=[3,2];mass[1,1:3]=[4,1]
        owner,_,_=ownership(mass)
        np.testing.assert_array_equal(owner,[0,0])


if __name__=='__main__':unittest.main()
