"""Nominate capacity-compatible route groups and solve their bounded-route cover."""
from itertools import combinations
from functools import lru_cache
import warnings
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,linprog,milp
from scipy.sparse import csc_matrix

def candidate_groups(parts,limit=21):
    sizes=[len(r) for r in parts]
    for count in range(min(6,len(parts)-1),0,-1):
        pool=[g for g in combinations(range(len(parts)),count) if sum(sizes[i] for i in g)<=limit]
        if pool:return pool
    return []

def alternatives(parts,edge,mass):
    info=[]
    for k,r in enumerate(parts):
        for p,i in enumerate(r):
            prev=r[p-1] if p else 0;nxt=r[p+1] if p+1<len(r) else 0
            scores={}
            for j,t in enumerate(parts):
                if j==k:continue
                compatible=[v for v in t if mass[i]+mass[v]<=20000.+1.e-7]
                if compatible:
                    f=max(edge[i,v]-edge[i,nxt] for v in compatible)
                    b=max(edge[v,i]-edge[prev,i] for v in compatible)
                    scores[j]=float((f+b)/2)
            info.append((k,scores))
    return info

def group_scores(groups,info):
    out=[]
    for g in groups:
        gains=[]
        for owner,values in info:
            if owner not in g:continue
            available=sorted((values[j] for j in g if j in values),reverse=True)[:2]
            gains.append(sum(available)/len(available) if available else 0.)
        out.append(np.mean(gains))
    return out

def select_two(groups,scores):
    order=sorted(range(len(groups)),key=lambda j:-scores[j])
    first=groups[order[0]]
    if len(order)==1:return [first]
    second=next((groups[j] for j in order[1:] if len(set(groups[j])-set(first))>=2),groups[order[1]])
    return [first,second]

def priced_cover(parts,columns,value,seconds=2.):
    customers=sorted(i for r in parts for i in r);ids={v:i for i,v in enumerate(customers)}
    cand={tuple(sorted(r)):(value(r),r) for r in parts}
    for key,(v,r) in columns.items():
        if all(i in ids for i in key) and (key not in cand or v>cand[key][0]):cand[key]=v,r
    pool=list(cand.values());row=[];col=[]
    for j,(_,r) in enumerate(pool):
        row.extend(ids[i] for i in r);col.extend([j]*len(r))
    mat=csc_matrix((np.ones(len(row)),(row,col)),shape=(len(ids),len(pool)))
    singles=sum(value((v,)) for v in customers);base=sum(map(value,parts))
    c=np.array([sum(value((i,)) for i in r)-v for v,r in pool])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Unrecognized options detected.*threads")
        lp=linprog(c,A_eq=mat,b_eq=np.ones(len(ids)),bounds=(0,None),method="highs",options={"threads":1})
        if not lp.success:raise RuntimeError(lp.message)
        upper=singles-lp.fun
        if upper<=base+1.e-7:
            return dict(routes=parts,gain=0.,gap_U=max(0.,upper-base),optimal=True,columns=len(c))
        rc=c-np.asarray(mat.T@lp.eqlin.marginals).ravel()
        keep=lp.fun+rc<=singles-base+1.e-6
        mat=mat[:,keep];c=c[keep];pool=[z for z,k in zip(pool,keep) if k]
        out=milp(c,integrality=np.ones(len(c)),bounds=Bounds(0.,1.),
            constraints=LinearConstraint(mat,1.,1.),options=dict(time_limit=seconds,mip_rel_gap=0.,threads=1))
    if out.status not in (0,1):raise RuntimeError(out.message)
    win=[r for x,(_,r) in zip(out.x,pool) if x>.5] if out.x is not None else parts
    if sorted(v for t in win for v in t)!=customers:raise ValueError("Incomplete route cover")
    u=sum(map(value,win))
    if u<=base+1.e-7:u=base;win=parts
    bound=getattr(out,"mip_dual_bound",None)
    # Removed columns cannot produce a better cover than the parent.
    if bound is not None:upper=min(upper,max(base,singles-bound))
    return dict(routes=win,gain=u-base,gap_U=max(0.,upper-u),optimal=bool(out.success or upper-u<=1.e-6),columns=len(cand))

def region_columns(clients,mass,value):
    pool={}
    def visit(r,load):
        for i in clients:
            w=load+mass[i]
            if i in r or w>20000.+1.e-7:continue
            seq=r+(i,);v=value(seq);key=tuple(sorted(seq))
            if np.isfinite(v) and (key not in pool or v>pool[key][0]):pool[key]=v,seq
            if len(seq)<4:visit(seq,w)
    visit((),0.)
    return pool
