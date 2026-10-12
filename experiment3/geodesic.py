"""Canonical mesh geodesics, fixed material neighborhoods and track gates."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from experiment2.arap import mesh_edges


def graph(vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    edges = mesh_edges(faces)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if not np.isfinite(vertices).all():
        raise ValueError('Nonfinite canonical mesh')
    positive = lengths > 1e-12
    edges, lengths = edges[positive], lengths[positive]
    i, j = edges.T
    adjacency = sparse.csr_matrix((np.r_[lengths, lengths], (np.r_[i,j], np.r_[j,i])), shape=(len(vertices),len(vertices)))
    count, components = connected_components(adjacency, directed=False)
    return adjacency, edges, lengths, components, count


def influence(adjacency, node_vertices, k=3, batch=16):
    n = adjacency.shape[0]
    best = np.full((n,k), np.inf)
    ids = np.zeros((n,k), dtype=np.int64)
    for start in range(0,len(node_vertices),batch):
        stop = min(start+batch,len(node_vertices))
        distances = dijkstra(adjacency, directed=False, indices=node_vertices[start:stop])
        candidate = np.c_[best, distances.T]
        candidate_ids = np.c_[ids, np.broadcast_to(np.arange(start,stop),(n,stop-start))]
        order = np.lexsort((candidate_ids,candidate),axis=1)[:,:k]
        best = np.take_along_axis(candidate,order,axis=1)
        ids = np.take_along_axis(candidate_ids,order,axis=1)
    if not np.isfinite(best[:,0]).all():
        raise ValueError('A component has no control anchor; spatial fallback forbidden')
    finite_positive = best[np.isfinite(best)&(best>1e-12)]
    sigma = float(np.median(finite_positive)) if len(finite_positive) else 1.
    logits = -np.square(best/sigma)/2
    logits -= logits.max(axis=1,keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1,keepdims=True)
    return ids, weights.astype(np.float32), best.astype(np.float32), sigma


def neighborhoods(adjacency, edges, anchors, k=6, maximum_distance=np.inf):
    anchors = np.asarray(anchors,dtype=np.int64)
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError('Duplicate material tracking anchors')
    lookup = {tuple(edge):i for i,edge in enumerate(edges.tolist())}
    pairs, lengths, paths, path_pairs = [], [], [], []
    seen = set()
    for i,source in enumerate(anchors):
        dist, pred = dijkstra(adjacency,directed=False,indices=int(source),return_predecessors=True)
        order = np.argsort(dist[anchors],kind='stable')
        neighbors = [int(j) for j in order if j!=i and np.isfinite(dist[anchors[j]]) and dist[anchors[j]]<=maximum_distance][:k]
        for j in neighbors:
            pair = tuple(sorted((i,j)))
            if pair in seen:continue
            seen.add(pair)
            material_path = []
            target = int(anchors[j])
            while target != source:
                previous = int(pred[target])
                if previous < 0:raise ValueError('Broken canonical shortest path')
                material_path.append(lookup[tuple(sorted((target,previous)))])
                target = previous
            if not material_path:raise ValueError('Empty neighbor path')
            path_pairs.extend([len(pairs)]*len(material_path))
            paths.extend(material_path)
            pairs.append(pair);lengths.append(float(dist[anchors[j]]))
    if not pairs:raise ValueError('No connected geodesic tracking neighborhood')
    return {'pairs':np.asarray(pairs,dtype=np.int64),'rest_path_lengths':np.asarray(lengths,dtype=np.float32),
            'path_edge_ids':np.asarray(paths,dtype=np.int64),'path_pair_ids':np.asarray(path_pairs,dtype=np.int64)}


def neighbor_gate(tracks, visible, pairs, rest_lengths, frames, vertices, maximum_stretch=2.):
    """Conservative projection bound; never equate 2D distance with a geodesic.

    Assume path stretch <= maximum_stretch and vertices remain within one
    canonical diagonal of the transformed canonical depth envelope. Explicit
    bounds and rejected observations are returned for audit, not ground truth.
    """
    result = np.asarray(visible,dtype=bool).copy()
    failures = np.zeros(result.shape,dtype=np.int32)
    degrees = np.zeros(result.shape,dtype=np.int32)
    bounds = []
    scale = float(np.linalg.norm(np.ptp(vertices,axis=0)))
    i,j = pairs.T
    for f,frame in enumerate(frames):
        linear=np.asarray(frame['linear']);translation=np.asarray(frame['translation'])
        operator=float(np.linalg.norm(linear,ord=2))
        z_lower=float((vertices@linear.T+translation)[:,2].min()-operator*scale)
        if z_lower<=.001:raise ValueError('No positive depth envelope for geodesic tracking gate')
        camera=frame['camera']
        # sqrt(2) bounds the normalized ray derivative inside the image frustum.
        focal=max(camera['fx'],camera['fy'])
        ray=np.sqrt(1+((camera['width']/focal)**2+(camera['height']/focal)**2))
        bound=ray*focal/z_lower*operator*rest_lengths*maximum_stretch
        active=result[f,i]&result[f,j]
        bad=active&(np.linalg.norm(tracks[f,i]-tracks[f,j],axis=1)>bound)
        np.add.at(degrees[f],i,active.astype(int));np.add.at(degrees[f],j,active.astype(int))
        np.add.at(failures[f],i,bad.astype(int));np.add.at(failures[f],j,bad.astype(int))
        bounds.append({'frame_id':frame['id'],'z_lower':z_lower,'operator_norm':operator,'maximum_stretch':maximum_stretch,
            'bound_px_min':float(bound.min()),'bound_px_max':float(bound.max()),'inconsistent_pairs':int(bad.sum())})
    # One occluded/mistracked neighbor should not remove an otherwise supported
    # point. Reject only a majority with at least two independently bad edges.
    rejected=(failures>=2)&(failures*2>=np.maximum(degrees,1))
    result[rejected]=False
    return result,rejected,bounds


def path_loss(vertices, edges, path_edge_ids, path_pair_ids, rest_lengths):
    import torch
    lengths=(vertices[edges[:,0]]-vertices[edges[:,1]]).norm(dim=-1)
    sums=torch.zeros_like(rest_lengths).index_add(0,path_pair_ids,lengths[path_edge_ids])
    return ((sums/rest_lengths-1)**2).mean()
