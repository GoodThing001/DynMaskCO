"""Compare two rankings of the same ruin moves before identical distance-only repair."""
import numpy as np

def moves(group, distances, rng, count=64):
    group=np.asarray(group,dtype=int)
    result=[]
    for k in range(count):
        anchor=int(rng.choice(group))
        candidates=group[group!=anchor]
        if k%2==0:
            order=candidates[np.argsort(distances[anchor,candidates],kind="stable")]
            candidates=order[:18]
        selected=np.r_[anchor,rng.choice(candidates,9,replace=False)]
        result.append(tuple(sorted(map(int,selected))))
    return result

def merit(routes, heat, group, masks):
    # Compare how well the observed incident arcs and their possible replacements agree.
    ids=np.asarray(group,dtype=int)
    ranks=np.argsort(np.argsort(-heat,axis=1,kind="stable"),axis=1,kind="stable")+1
    incident={}
    for r in routes:
        for a,i,b in zip((0,)+r,r,r[1:]+(0,)):
            incident[i]=np.log(np.sqrt(float(ranks[i,a])*ranks[i,b]))
    scores=[]
    for selected in masks:
        selected=np.asarray(selected,dtype=int)
        other=ids[~np.isin(ids,selected)]
        # A movable customer and an unremoved anchor must mutually support the new connection.
        mat=np.sqrt(ranks[np.ix_(selected,other)]*ranks[np.ix_(other,selected)].T)
        best=np.log(mat.min(axis=1))
        scores.append(float((np.asarray([incident[i] for i in selected])-best).mean()))
    return np.asarray(scores)

def order(criterion):
    return np.argsort(-criterion,kind="stable")[:8]
