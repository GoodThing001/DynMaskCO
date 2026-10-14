"""Isolated production-style insertion options with full retained-edge guard.

No model, complete reconstruct loop, final C0 evaluation, or deployment.
"""
import numpy as np
from delivery_retained_edge_action_guard_v1 import permits
from delivery_baselines import matrices,route_ok
def options(data,routes,customer,partial,logits,contract):
    d=matrices(data);off=~np.eye(len(d),dtype=bool)
    if partial.shape!=d.shape or logits.shape!=d.shape or not np.all(np.isfinite(logits[off])):raise ValueError('Action input dimensions/finite logits')
    ds=max(1e-9,float(d[off].mean()));ls=max(1e-8,float(np.std(logits[off])))
    required={(a,b) for a in range(len(d)) for b in range(a+1,len(d)) if partial[a,b]>0}
    rows=[]
    for slot,r in enumerate(routes):
        for at in range(len(r)+1):
            a=r[at-1] if at else 0;b=r[at] if at<len(r) else 0
            retained=permits(routes,customer,slot,at,required)
            physical_route_filter=route_ok(r[:at]+[customer]+r[at:],data,contract,d,0.)
            delta=d[a,customer]+d[customer,b]-d[a,b]
            gain=(logits[a,customer]+logits[customer,b]-(0. if a==b else logits[a,b]))/ls
            score=float(delta/ds-gain)
            rows.append(dict(customer=customer,vehicle=slot,position=at,predecessor=a,successor=b,
                old_edge_skip=bool(partial[a,b]>0),retained_union_ok=retained,original_route_ok=bool(physical_route_filter),
                eligible=bool(retained and physical_route_filter),score=score))
    best=min((x for x in rows if x['eligible']),key=lambda x:(x['score'],x['vehicle'],x['position']),default=None)
    return dict(rows=rows,best=best,dscale=ds,lscale=ls)
