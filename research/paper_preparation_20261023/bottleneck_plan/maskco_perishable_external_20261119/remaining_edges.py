"""Economic beam reconstruction with feasible adjacency potential for pending customers."""
from functools import lru_cache
import math
import numpy as np


def reciprocal_weights(scores):
    weights=np.zeros_like(scores,dtype=float)
    for i in range(1,len(scores)):
        ids=[j for j in range(1,len(scores)) if i!=j and np.isfinite(scores[i,j])]
        if not ids:
            continue
        values=scores[i,ids]
        ascending=np.sort(values)
        ranks=1+len(ids)-np.searchsorted(ascending,values,side="right")
        weights[i,ids]=1./ranks
        weights[i]/=weights[i].sum()
    return weights


def beam_repair(parent,order,value,weights):
    hidden=set(order)
    rest=[tuple(i for i in r if i not in hidden) for r in parent]
    initial=tuple(sorted(r for r in rest if r))
    original=sum(value(r) for r in parent)
    start=sum(value(r) for r in initial)
    if not math.isfinite(start):
        return dict(U=original,routes=parent,expansions=0,unusable_skeleton=True)
    beam={initial:start}
    expanded=0

    @lru_cache(maxsize=None)
    def accessible(route,customer):
        near=set()
        for p in range(len(route)+1):
            if math.isfinite(value(route[:p]+(customer,)+route[p:])):
                if p:near.add(route[p-1])
                if p<len(route):near.add(route[p])
        return sum(weights[customer,j] for j in near)

    @lru_cache(maxsize=None)
    def potential(route,depth):
        return sum(accessible(route,q) for q in order[depth:])

    for layer,customer in enumerate(order,1):
        children={}
        for fleet,U in beam.items():
            expanded+=1
            options=[]
            for ri,r in enumerate(fleet):
                before=value(r)
                for pos in range(len(r)+1):
                    new=r[:pos]+(customer,)+r[pos:]
                    gain=value(new)-before
                    if math.isfinite(gain):
                        options.append((gain,ri,new))
            options.sort(key=lambda item:(-item[0],item[1],item[2]))
            for gain,ri,r in options[:4]+[(value((customer,)),len(fleet),(customer,))]:
                next_fleet=tuple(sorted(fleet[:ri]+fleet[ri+1:]+(r,)))
                children[next_fleet]=max(children.get(next_fleet,-math.inf),U+gain)
        best=sorted(children,key=lambda f:(-children[f],f))
        if layer==len(order):
            win=best[0]
            if children[win] <= original+1.e-7:win=parent
            return dict(U=sum(value(r) for r in win),routes=win,expansions=expanded,unusable_skeleton=False)
        if weights is None or len(best)<=6:
            selected=best[:6]
        else:
            selected=best[:3]
            extra=sorted(best[3:],key=lambda f:(-sum(potential(r,layer) for r in f),-children[f],f))
            selected+=extra[:3]
        beam={f:children[f] for f in selected}
    raise ValueError("Expected a nonempty masked customer group")
