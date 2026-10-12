import unittest
import numpy as np
import torch
from experiment3.geodesic import graph,influence,neighborhoods,neighbor_gate,path_loss


class GeodesicChecks(unittest.TestCase):
    def setUp(self):
        # Two image-close sheets; their only surface connection is a long detour.
        self.vertices=np.array([[0.,0,0],[0,1,0],[1,1,0],[.01,0,.01],[.01,1,.01],[1,1,.01]])
        self.faces=np.array([[0,1,2],[3,4,5],[2,5,4]])
        self.graph,self.edges,self.lengths,self.components,_=graph(self.vertices,self.faces)
    def test_geodesic_control_assignment_avoids_spatial_shortcut(self):
        ids,weights,distances,_=influence(self.graph,np.array([1,3]),k=1)
        self.assertEqual(int(ids[0,0]),0)  # spatial nearest is control 3 on other sheet
        self.assertGreater(distances[0,0],.9)
        np.testing.assert_allclose(weights.sum(1),1)
    def test_disconnected_component_has_no_spatial_fallback(self):
        g,_,_,_,_=graph(self.vertices,self.faces[:2])
        with self.assertRaises(ValueError):influence(g,np.array([0]),k=1)
        ids,weights,distances,_=influence(g,np.array([0,3]),k=3)
        self.assertEqual(weights[0,0],1);self.assertTrue(np.isinf(distances[0,1]))
    def test_paths_are_rigid_invariant_and_penalize_material_stretch_with_gradient(self):
        n=neighborhoods(self.graph,self.edges,np.array([0,1,2]),k=2)
        edges=torch.tensor(self.edges);paths=torch.tensor(n['path_edge_ids']);pairs=torch.tensor(n['path_pair_ids']);rest=torch.tensor(n['rest_path_lengths'],dtype=torch.float64)
        rotation=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]],dtype=torch.float64)
        q=torch.tensor(self.vertices,dtype=torch.float64)@rotation+3
        self.assertLess(float(path_loss(q,edges,paths,pairs,rest)),1e-12)
        q=(torch.tensor(self.vertices,dtype=torch.float64)*1.5).requires_grad_()
        loss=path_loss(q,edges,paths,pairs,rest);loss.backward()
        self.assertGreater(float(loss),.1);self.assertTrue(torch.isfinite(q.grad).all());self.assertGreater(float(q.grad.abs().max()),0)
    def test_geodesic_neighbors_gate_impossible_track_separation_without_filling_hidden_points(self):
        tracks=np.array([[[100.,100],[101,100],[100,101],[400,400]]])
        pairs=np.array([[0,1],[0,2],[0,3],[1,3],[2,3]])
        frame={'id':'0','linear':np.eye(3),'translation':[0,0,20.],'camera':{'width':500,'height':500,'fx':500.,'fy':500.}}
        visible=np.ones((1,4),dtype=bool)
        valid,rejected,_=neighbor_gate(tracks,visible,pairs,np.full(5,.1),[frame],self.vertices)
        self.assertFalse(valid[0,3]);self.assertTrue(rejected[0,3])
        visible[0,3]=False
        valid,_,_=neighbor_gate(tracks,visible,pairs,np.full(5,.1),[frame],self.vertices)
        self.assertFalse(valid[0,3]);self.assertTrue(valid[0,0])


if __name__=='__main__':unittest.main()
