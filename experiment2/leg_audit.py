"""Diagnostic canonical leg cores and node influence ownership, without motion edits."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .arap import mesh_edges

REGIONS=['Unclassified','Body / outside leg cores','Front −X','Front +X','Hind −X','Hind +X']
COLORS=['#555b65','#b3bac6','#ff876c','#47c7ed','#c294ff','#e7d65e']


def leg_cores(vertices,faces,z_cutoff):
    """Camel-specific diagnostic partition, never anatomical left/right labels.

    Four largest components below the explicit Z cutoff within the largest
    canonical component. Everything outside those cores stays body/unclassified.
    This proposal needs visual inspection; no labels are spread across gaps.
    """
    edges=mesh_edges(faces);i,j=edges.T;n=len(vertices)
    graph=sparse.csr_matrix((np.ones(len(i)*2),(np.r_[i,j],np.r_[j,i])),shape=(n,n))
    _,component=connected_components(graph,directed=False)
    main=int(np.bincount(component).argmax())
    lower=np.flatnonzero((component==main)&(vertices[:,2]<z_cutoff))
    _,parts=connected_components(graph[lower][:,lower],directed=False)
    sizes=np.bincount(parts)
    if len(sizes)<4:raise ValueError('Cannot identify four separate lower-leg cores')
    selected=np.argsort(-sizes,kind='stable')[:4]
    if sizes[selected].min()<.05*sizes[selected].max():raise ValueError('Four major leg cores not established')
    centers={int(k):vertices[lower[parts==k]].mean(0) for k in selected}
    by_y=sorted(selected,key=lambda k:centers[int(k)][1])
    ordered=sorted(by_y[:2],key=lambda k:centers[int(k)][0])+sorted(by_y[2:],key=lambda k:centers[int(k)][0])
    labels=np.zeros(n,dtype=np.int8);labels[component==main]=1
    # Small lower fragments within the main component are unclassified, not body.
    labels[lower]=0
    regions=[]
    for label,part in enumerate(ordered,2):
        ids=lower[parts==part];labels[ids]=label
        regions.append({'id':label,'name':REGIONS[label],'vertices':len(ids),'centroid':centers[int(part)].tolist()})
    return labels,{'z_cutoff':float(z_cutoff),'canonical_components':int(component.max()+1),
                   'main_component_vertices':int((component==main).sum()),'regions':regions,
                   'method':'Four largest lower-Z connected components of main canonical component; split front/hind by Y and sides by X; provisional diagnostic labels',
                   'unclassified_vertices':int((labels==0).sum())}


def node_affinity(labels,indices,weights,node_count,sample_weights=None):
    """Accumulate receiver-region mass per node; no winner inferred from XYZ."""
    labels=np.asarray(labels);indices=np.asarray(indices);weights=np.asarray(weights)
    if indices.shape!=weights.shape or len(labels)!=len(indices):raise ValueError('Mapping shape mismatch')
    if np.any(indices<0) or np.any(indices>=node_count):raise ValueError('Invalid node IDs')
    if not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('Invalid influence weights')
    scale=np.ones(len(labels)) if sample_weights is None else np.asarray(sample_weights)
    mass=np.zeros((node_count,len(REGIONS)))
    for region in range(len(REGIONS)):
        active=labels==region
        mass[:,region]=np.bincount(indices[active].ravel(),weights=(weights[active]*scale[active,None]).ravel(),minlength=node_count)
    return mass


def ownership(mass,purity=.8,min_support=1.):
    total=mass.sum(1);dominant=mass.argmax(1)
    confidence=np.divide(mass[np.arange(len(mass)),dominant],total,out=np.zeros_like(total),where=total>0)
    reliable=(confidence>=purity)&(total>=min_support)&(dominant>=2)
    owner=np.where(reliable,dominant,0)
    return owner,confidence,total


def cross_leg_summary(labels,node_labels,indices,weights):
    """Other-leg influence measured against spatial attachment labels, not ground truth."""
    selected=node_labels[indices];receiver=labels[:,None]
    crossing=(selected>=2)&(receiver>=2)&(selected!=receiver)
    mass=(crossing*weights).sum(1);leg=labels>=2
    by_leg=[]
    for region in range(2,len(REGIONS)):
        active=labels==region
        by_leg.append({'region':REGIONS[region],'mean_other_leg_mass':float(mass[active].mean()) if active.any() else None,
            'fraction_vertices_above_20pct_other_leg_mass':float((mass[active]>.2).mean()) if active.any() else None})
    return {'mean_other_leg_mass':float(mass[leg].mean()) if leg.any() else None,'by_leg':by_leg,
            'definition':'Mean sum of weights attached to a different canonical leg core; body/unclassified node influence is not counted as crossing. Diagnostic, not anatomical ground truth.'},mass
