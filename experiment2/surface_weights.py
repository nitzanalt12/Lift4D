"""Fixed control-node weights using shortest paths on the canonical mesh.

Nodes attach to their nearest canonical vertex. Distance is edge-path length
plus the node-to-anchor Euclidean offset. This is a topology-aware *approximation*
to surface distance, not semantic limb segmentation or exact surface geodesics.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components, dijkstra
from .arap import mesh_edges


def gaussian_weights(squared_distances, indices, radii, node_weights=None):
    radii=np.asarray(radii).reshape(-1)
    if np.any(radii<=0) or not np.isfinite(radii).all():
        raise ValueError('Finite positive saved node radii required')
    valid=np.isfinite(squared_distances)
    weights=np.exp(-squared_distances/(2*radii[indices]**2))
    if node_weights is not None:weights*=np.asarray(node_weights).reshape(-1)[indices]
    weights=np.where(valid,weights+1e-7,0)
    if np.any(weights.sum(1)<=0):raise ValueError('A mesh component has no attached control node')
    return weights/weights.sum(1,keepdims=True)


def surface_weights(vertices,faces,nodes,radii,k=3,original_indices=None,batch_size=32,node_weights=None,allow_unattached=False):
    vertices=np.asarray(vertices,dtype=np.float64);nodes=np.asarray(nodes,dtype=np.float64)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or nodes.ndim!=2 or nodes.shape[1]!=3:
        raise ValueError('Explicit source-space XYZ arrays required')
    if not np.isfinite(vertices).all() or not np.isfinite(nodes).all():raise ValueError('Nonfinite coordinates')
    if not 1<=k<=len(nodes) or batch_size<1:raise ValueError('Invalid K or batch size')
    edges=mesh_edges(faces);i,j=edges.T
    lengths=np.linalg.norm(vertices[i]-vertices[j],axis=1)
    if np.any(lengths<=0):raise ValueError('Degenerate canonical edge')
    graph=sparse.csr_matrix((np.tile(lengths,2),(np.r_[i,j],np.r_[j,i])),shape=(len(vertices),len(vertices)))
    count,components=connected_components(graph,directed=False)
    offsets,anchors=cKDTree(vertices).query(nodes,k=1)
    missing=set(np.unique(components))-set(components[anchors])
    if missing and not allow_unattached:raise ValueError(f'{len(missing)} canonical components have no attached node; no spatial fallback')
    best=np.full((len(vertices),k),np.inf);indices=np.zeros((len(vertices),k),dtype=np.int64)
    old_distances=np.full(original_indices.shape,np.inf) if original_indices is not None else None
    for start in range(0,len(nodes),batch_size):
        stop=min(start+batch_size,len(nodes))
        distances=dijkstra(graph,directed=False,indices=anchors[start:stop])+offsets[start:stop,None]
        if original_indices is not None:
            row,col=np.where((original_indices>=start)&(original_indices<stop))
            old_distances[row,col]=distances[original_indices[row,col]-start,row]
        candidates=np.concatenate((best,distances.T),axis=1)
        candidate_ids=np.concatenate((indices,np.broadcast_to(np.arange(start,stop),(len(vertices),stop-start))),axis=1)
        # Stable node-index tie break, including coincident projected anchors.
        order=np.lexsort((candidate_ids,candidates),axis=1)[:,:k]
        best=np.take_along_axis(candidates,order,axis=1)
        indices=np.take_along_axis(candidate_ids,order,axis=1)
    available=np.isfinite(best).any(1)
    weights=np.zeros_like(best)
    weights[available]=gaussian_weights(best[available]**2,indices[available],radii,node_weights)
    return {'indices':indices,'distances':best,'weights':weights,'anchors':anchors,
            'anchor_offsets':offsets,'original_surface_distances':old_distances,
            'components':int(count),'component_sizes':np.bincount(components).tolist(),
            'unattached_vertices':np.flatnonzero(~available),
            'vertices_with_fewer_than_k_nodes':int(np.sum(np.isfinite(best).sum(1)<k))}
