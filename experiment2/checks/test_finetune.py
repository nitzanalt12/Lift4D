import unittest
import torch
from experiment2.finetune import mask_loss, spatial_loss, temporal_loss

class FineTuneLosses(unittest.TestCase):
    def test_mask_gradient_and_perfect_overlap(self):
        target=torch.tensor([[1.,0.],[1.,0.]])
        alpha=torch.tensor([[.8,.1],[.7,.2]],requires_grad=True)
        loss=mask_loss(alpha,target);loss.backward()
        self.assertTrue(torch.all(alpha.grad[target.bool()]<0))
        self.assertTrue(torch.all(alpha.grad[~target.bool()]>0))
        self.assertAlmostEqual(float(mask_loss(target,target)),0)
    def test_spatial_regularizer_preserves_uniform_translation(self):
        xyz=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        edges=torch.tensor([[0,1],[1,2]])
        self.assertEqual(float(spatial_loss(xyz+3,xyz,edges,2)),0)
        changed=xyz.clone();changed[1,1]=1
        self.assertGreater(float(spatial_loss(changed,xyz,edges,2)),0)
    def test_temporal_constant_and_linear_corrections(self):
        residual=torch.arange(5.).reshape(5,1,1).expand(5,2,3).clone().requires_grad_()
        self.assertEqual(float(temporal_loss(residual,0,2)),0)
        curved=residual.square();loss=temporal_loss(curved,4,2)
        self.assertGreater(float(loss),0);loss.backward()
        self.assertTrue(torch.isfinite(residual.grad).all())

if __name__=='__main__':unittest.main()
