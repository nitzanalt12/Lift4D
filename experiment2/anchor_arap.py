"""Original vertex ARAP plus observed 2D handles, with fixed cameras.

Local/global ARAP with Gauss-Newton handle reprojection and exact nonlinear
energy backtracking. Small handle systems use Woodbury; reuse the original
scalar sparse factorization. No network gradients or optimizer are involved.
"""
import numpy as np
from scipy import sparse
from .arap import ARAP
from .video_anchors import project


class VideoAnchorARAP(ARAP):
    def __init__(self,rest,faces,strength,handles,device='cpu'):
        super().__init__(rest,faces,strength,device)
        self.handles=handles;rows=[];cols=[];values=[]
        if len({h['id'] for h in handles})!=len(handles):raise ValueError('Duplicate handle ID')
        for i,h in enumerate(handles):
            ids=np.asarray(h['vertex_ids']);w=np.asarray(h['weights'])
            if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)!=len(w) or len(ids)==0:
                raise ValueError('Invalid fixed handle support')
            if np.any(ids<0) or np.any(ids>=len(rest)) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1):raise ValueError('Invalid handle binding')
            rows.extend([i]*len(ids));cols.extend(ids);values.extend(w)
        self.H=sparse.csr_matrix((values,(rows,cols)),shape=(len(handles),len(rest)))
        self.basis=self.factor.solve(self.H.T.toarray())
        self.gram=np.asarray(self.H@self.basis)

    def solve_anchored(self,target,observations,linear,translation,camera,keypoint_weight=10,iterations=150,tolerance=1e-5):
        import torch
        target=np.asarray(target,dtype=float)
        if target.shape!=self.rest.shape or not np.isfinite(target).all():raise ValueError('Invalid raw target')
        if keypoint_weight<0 or not np.isfinite(keypoint_weight) or iterations<1:raise ValueError('Invalid optimization settings')
        by_id={h['id']:i for i,h in enumerate(self.handles)}
        active=[by_id[r['handle_id']] for r in observations]
        if not active or len(set(active))!=len(active):raise ValueError('Explicit unique visible observations required')
        observed=np.asarray([r['xy'] for r in observations],dtype=float);confidence=np.asarray([r.get('confidence',1) for r in observations])
        if observed.shape!=(len(active),2) or not np.isfinite(observed).all() or np.any(confidence<=0) or np.any(confidence>1):raise ValueError('Invalid observations')
        H=self.H[active];basis=self.basis[:,active];gram=self.gram[np.ix_(active,active)]
        diagonal2=camera['width']**2+camera['height']**2
        gamma=keypoint_weight*len(self.rest)*self.scale**2/(confidence.sum()*diagonal2)
        def evaluate(q):
            result=self.energies(q,target)
            pixels=project(H@q,linear,translation,camera);errors=np.linalg.norm(pixels-observed,axis=1)
            reprojection=float(np.sum(confidence*errors**2)/(confidence.sum()*diagonal2))
            result.update(reprojection=reprojection,total=result['total']+keypoint_weight*reprojection,
                          reprojection_mean_px=float(errors.mean()),reprojection_rms_px=float(np.sqrt((errors**2).mean())),
                          per_handle_error_px=errors.tolist())
            return result
        vertices=target.copy();history=[];stalled=False
        with torch.inference_mode():
            initial=evaluate(vertices);current=initial
            for step in range(iterations):
                rotations=self.rotations(vertices)
                candidate=self.factor.solve(target+self.coefficient*self.rhs(rotations))
                if keypoint_weight:
                    handle_positions=H@vertices
                    pixels,jacobian=project(handle_positions,linear,translation,camera,jacobian=True)
                    S=np.zeros((2*len(active),3*len(active)))
                    for i,j in enumerate(jacobian):S[2*i:2*i+2,3*i:3*i+3]=np.sqrt(gamma*confidence[i])*j
                    g=(np.sqrt(gamma*confidence)[:,None]*(observed-pixels+np.einsum('hij,hj->hi',jacobian,handle_positions))).ravel()
                    residual=g-S@(H@candidate).ravel()
                    multiplier=np.linalg.solve(np.eye(len(g))+S@np.kron(gram,np.eye(3))@S.T,residual)
                    candidate+=basis@(S.T@multiplier).reshape(-1,3)
                accepted=None
                for backtrack in range(16):
                    fraction=2.**(-backtrack);q=vertices+fraction*(candidate-vertices)
                    try:energies=evaluate(q)
                    except ValueError:continue
                    if np.isfinite(energies['total']) and energies['total']<=current['total']+1e-12:
                        accepted=(q,energies,fraction);break
                if accepted is None:stalled=True;break
                q,energies,fraction=accepted
                change=float(np.linalg.norm(q-vertices,axis=1).max()/self.scale)
                vertices=q;current=energies
                history.append({'iteration':step+1,'max_relative_vertex_change':change,'step_fraction':fraction,
                    'total':current['total'],'reprojection_mean_px':current['reprojection_mean_px']})
                if change<tolerance:break
        return vertices.astype(np.float32),{'initial':initial,'final':current,'iterations':len(history),
            'converged':bool(history and history[-1]['max_relative_vertex_change']<tolerance and not stalled),
            'stalled':stalled,'history':history,'visible_handle_ids':[r['handle_id'] for r in observations],
            'metric_status':'Reprojection on fitted observations, not held-out evaluation'}
