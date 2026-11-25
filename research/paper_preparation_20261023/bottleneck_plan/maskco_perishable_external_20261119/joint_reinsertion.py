"""Choose a coordinated mask and exactly price its retained-skeleton route options."""
from itertools import combinations,permutations
from functools import lru_cache
import warnings
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import csc_matrix

def nominate(parts,score,noise,limit=12):
    owner={i:k for k,r in enumerate(parts) for i in r}
    priority=[]
    for k,r in enumerate(parts):
        for pos,i in enumerate(r):
            last=r[pos-1] if pos else 0
            nxt=r[pos+1] if pos+1<len(r) else 0
            old=min(score[last,i],score[i,nxt])
            other=[j for j in owner if owner[j]!=k]
            if not other:
                priority.append((noise[i],i,i))
                continue
            edge,j=max((max(score[i,j],score[j,i]),-j) for j in other)
            priority.append((float(edge-old)+noise[i],i,-j))
    chosen=[]
    for _,i,j in sorted(priority,key=lambda v:(-v[0],v[1])):
        for v in (i,j):
            if v not in chosen:chosen.append(v)
            if len(chosen)==limit:return tuple(sorted(chosen))
    return tuple(sorted(chosen))

@lru_cache(None)
def interleavings(n,k):
    return tuple(combinations(range(n+k),n))

def mix(skeleton,free):
    n=len(skeleton);k=len(free)
    if not k:
        yield skeleton
        return
    if not n:
        yield from permutations(free)
        return
    for fixed in interleavings(n,k):
        for order in permutations(free):
            si=iter(skeleton);hi=iter(order)
            yield tuple(next(si) if i in fixed else next(hi) for i in range(n+k))

def optimize(parts,hidden,mass,value,time_limit=2.):
    parts=list(map(tuple,parts));hidden=tuple(sorted(hidden));hidden_set=set(hidden)
    if not hidden_set<=set(i for r in parts for i in r):raise ValueError("Mask outside region")
    visible=[tuple(i for i in r if i not in hidden_set) for r in parts]
    skeletons=[r for r in visible if r];base=len(hidden)
    hid={i:j for j,i in enumerate(hidden)}
    sid={r:base+j for j,r in enumerate(skeletons)}
    columns={}
    def insert(key,route):
        u=value(route)
        if np.isfinite(u) and (key not in columns or u>columns[key][0]):columns[key]=(u,route)
    for r,sk in zip(parts,visible):
        key=tuple(sorted(hid[i] for i in r if i in hidden_set))+((sid[sk],) if sk else ())
        insert(key,r)
    for sk in [()]+skeletons:
        rem=20000.-sum(mass[i] for i in sk)
        for n in range(int(not sk),max(0,4-len(sk))+1):
            for sub in combinations(hidden,n):
                if sum(mass[i] for i in sub)>rem+1.e-7:continue
                key=tuple(hid[i] for i in sub)+((sid[sk],) if sk else ())
                for route in mix(sk,sub):insert(key,route)
    keys=list(columns);pool=list(columns.values());a=[];j=[]
    for t,key in enumerate(keys):
        a.extend(key);j.extend([t]*len(key))
    mat=csc_matrix((np.ones(len(a)),(a,j)),shape=(base+len(skeletons),len(keys)))
    cost=np.array([sum(value((i,)) for i in r)-u for u,r in pool])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Unrecognized options detected.*threads")
        sol=milp(cost,integrality=np.ones(len(keys)),bounds=Bounds(0.,1.),
            constraints=LinearConstraint(mat,1.,1.),options=dict(time_limit=time_limit,mip_rel_gap=0.,threads=1))
    if sol.status not in (0,1):raise ValueError(sol.message)
    candidate=[r for x,(_,r) in zip(sol.x,pool) if x>.5] if sol.x is not None else parts
    if sorted(i for r in candidate for i in r)!=sorted(i for r in parts for i in r):raise ValueError("Changed coverage")
    if sorted(t for r in candidate if (t:=tuple(i for i in r if i not in hidden_set)))!=sorted(skeletons):raise ValueError("Visible route changed")
    if sum(map(value,candidate))<=sum(map(value,parts))+1.e-8:candidate=parts
    gain=sum(map(value,candidate))-sum(map(value,parts))
    dual=getattr(sol,"mip_dual_bound",None)
    gap=None if dual is None else max(0.,sum(value((i,)) for r in parts for i in r)-dual-sum(map(value,candidate)))
    return dict(gain=gain,chosen=candidate,gap_U=gap,optimal=sol.status==0,columns=len(keys),hidden=hidden)
