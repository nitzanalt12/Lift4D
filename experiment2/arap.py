"""Uniform-weight vertex ARAP with soft targets; original mesh connectivity."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def mesh_edges(faces):
    return np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)


class ARAP:
    """Minimize mean target error + strength * mean symmetric edge ARAP error.

    Fixed positive uniform weights; proper SO(3) local rotations. Global solves
    are CPU sparse LU, factored once per strength; batched rotation fits use the
    selected torch device. No learned model or data alignment is used.
    """
    def __init__(self,rest,faces,strength,device='cpu'):
        import torch
        if strength<=0:raise ValueError('ARAP strength must be positive')
        self.rest=np.asarray(rest,dtype=np.float64);self.edges=mesh_edges(faces)
        if not np.isfinite(self.rest).all() or not len(self.edges):raise ValueError('Invalid canonical mesh')
        n=len(rest);i,j=self.edges.T;e=self.rest[i]-self.rest[j]
        if np.any(np.linalg.norm(e,axis=1)<=1e-12):raise ValueError('Degenerate canonical edge')
        self.coefficient=float(strength)*n/len(self.edges)
        self.scale=float(np.linalg.norm(np.ptp(self.rest,axis=0)))
        row=np.concatenate([i,j,i,j]);col=np.concatenate([i,j,j,i]);val=np.concatenate([np.ones(len(i)*2),-np.ones(len(i)*2)])
        laplacian=sparse.coo_matrix((val,(row,col)),shape=(n,n)).tocsc()
        self.factor=splu(sparse.eye(n,format='csc')+self.coefficient*laplacian)
        self.device=device;self.indices=torch.tensor(self.edges,dtype=torch.long,device=device)
        self.original_edges=torch.tensor(e,dtype=torch.float32,device=device)
    def rotations(self,vertices):
        import torch
        q=torch.tensor(vertices,dtype=torch.float32,device=self.device)
        i,j=self.indices.T
        current=q[i]-q[j]
        outer=current[:,:,None]*self.original_edges[:,None,:]
        covariance=torch.zeros((len(q),3,3),device=self.device)
        covariance.index_add_(0,i,outer);covariance.index_add_(0,j,outer)
        u,_,vh=torch.linalg.svd(covariance,full_matrices=False)
        # Correct reflections, including neighborhoods with rank-deficient covariance.
        signs=torch.ones((len(q),3),device=self.device)
        signs[:,2]=torch.where(torch.linalg.det(u@vh)<0,-1.,1.)
        return (u*signs[:,None,:])@vh
    def rhs(self,rotations):
        import torch
        i,j=self.indices.T
        transformed=torch.einsum('nij,nj->ni',.5*(rotations[i]+rotations[j]),self.original_edges)
        rhs=torch.zeros((len(self.rest),3),device=self.device)
        rhs.index_add_(0,i,transformed);rhs.index_add_(0,j,-transformed)
        return rhs.cpu().numpy().astype(np.float64)
    def energies(self,vertices,target,rotations=None):
        import torch
        rotations=self.rotations(vertices) if rotations is None else rotations
        i,j=self.indices.T
        q=torch.tensor(vertices,dtype=torch.float32,device=self.device)
        actual=q[i]-q[j]
        first=torch.einsum('nij,nj->ni',rotations[i],self.original_edges)
        second=torch.einsum('nij,nj->ni',rotations[j],self.original_edges)
        arap=float(((actual-first).square().sum(-1)+(actual-second).square().sum(-1)).mean().cpu())/(2*self.scale**2)
        anchor=float(np.square(np.asarray(vertices)-target).sum(-1).mean())/self.scale**2
        strength=self.coefficient*len(self.edges)/len(self.rest)
        return {'arap':arap,'anchor':anchor,'total':anchor+strength*arap}
    def solve(self,target,iterations=15,tolerance=1e-5):
        import torch
        target=np.asarray(target,dtype=np.float64)
        if target.shape!=self.rest.shape or not np.isfinite(target).all():raise ValueError('Invalid target vertices')
        vertices=target.copy();history=[]
        with torch.inference_mode():
            initial=self.energies(vertices,target)
            for step in range(iterations):
                rotations=self.rotations(vertices)
                updated=self.factor.solve(target+self.coefficient*self.rhs(rotations))
                if not np.isfinite(updated).all():raise ValueError('Nonfinite ARAP solution')
                change=float(np.linalg.norm(updated-vertices,axis=1).max()/self.scale)
                vertices=updated
                history.append({'iteration':step+1,'max_relative_vertex_change':change})
                if change<tolerance:break
            final=self.energies(vertices,target)
        if final['total']>initial['total']+max(1e-7,initial['total']*1e-4):raise ValueError('ARAP energy increased')
        return vertices.astype(np.float32),{'initial':initial,'final':final,'iterations':len(history),'converged':history[-1]['max_relative_vertex_change']<tolerance,'history':history}


def distortion(rest,vertices,faces):
    edges=mesh_edges(faces);i,j=edges.T
    rest_length=np.linalg.norm(rest[i]-rest[j],axis=1)
    ratio=np.linalg.norm(vertices[i]-vertices[j],axis=1)/rest_length
    a,b,c=faces.T
    original_area=np.linalg.norm(np.cross(rest[b]-rest[a],rest[c]-rest[a]),axis=1)
    area=np.linalg.norm(np.cross(vertices[b]-vertices[a],vertices[c]-vertices[a]),axis=1)
    valid=original_area>np.linalg.norm(np.ptp(rest,axis=0))**2*1e-12
    return {'edge_stretch_p50':float(np.percentile(ratio,50)),'edge_stretch_p95':float(np.percentile(ratio,95)),
            'edge_stretch_p99':float(np.percentile(ratio,99)),'fraction_edges_above_2':float(np.mean(ratio>2)),
            'fraction_edges_above_5':float(np.mean(ratio>5)),
            'fraction_edges_below_half':float(np.mean(ratio<.5)),
            'area_ratio_p95':float(np.percentile(area[valid]/original_area[valid],95)),
            'fraction_faces_area_below_1pct':float(np.mean(area[valid]/original_area[valid]<.01)),
            'degenerate_rest_faces':int((~valid).sum())}
