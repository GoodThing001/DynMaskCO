"""Generate masked multiclient columns against every visible vehicle skeleton."""
import warnings
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix
from scipy.stats import rankdata


class MaskedMaster:
    def __init__(self, routes, mask, value):
        self.value = value
        self.hidden = tuple(sorted(mask))
        hidden_set = set(mask)
        self.item = {i: k for k, i in enumerate(self.hidden)}
        self.skeletons = [tuple(v for v in r if v not in hidden_set) for r in routes]
        self.skeletons = tuple(s for s in self.skeletons if s)
        self.owner = {v: len(mask)+k for k,s in enumerate(self.skeletons) for v in s}
        self.basis = np.array([value((v,)) for v in self.hidden] + [value(s) for s in self.skeletons])
        self.parent = tuple(map(tuple,routes))
        if not np.isfinite(self.basis).all():
            raise ValueError("A standalone skeleton is infeasible")

    def cover(self, route):
        fixed = tuple(v for v in route if v in self.owner)
        ids = [self.item[v] for v in route if v in self.item]
        if fixed:
            owner = self.owner[fixed[0]]
            if fixed != self.skeletons[owner - len(self.hidden)]:
                raise ValueError("Visible skeleton differs")
            ids.append(owner)
        return tuple(sorted(ids))

    def insert(self, pool, route):
        u = self.value(route)
        if not np.isfinite(u):
            return None
        key = self.cover(route)
        if key not in pool or pool[key][0] < u:
            pool[key] = (u, route)
        return key, u

    def common(self):
        pool={}
        for r in self.parent+self.skeletons:
            self.insert(pool,r)
        for i in self.hidden:
            self.insert(pool,(i,))
            for j in self.hidden:
                if j!=i:
                    self.insert(pool,(i,j))
            for s in self.skeletons:
                for k in range(len(s)+1):
                    self.insert(pool,s[:k]+(i,)+s[k:])
        return pool

    def problem(self, pool):
        keys=list(pool)
        rows=[j for key in keys for j in key]
        cols=np.repeat(np.arange(len(keys)),[len(k) for k in keys])
        matrix=csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(self.basis),len(keys)))
        costs=np.array([self.basis[list(k)].sum()-pool[k][0] for k in keys])
        return keys,matrix,costs

    def prices(self,pool):
        _,mat,c=self.problem(pool)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected")
            lp=linprog(c,A_eq=mat,b_eq=np.ones(len(self.basis)),bounds=(0,None),method="highs",options={"threads":1})
        if not lp.success:raise RuntimeError(lp.message)
        return -lp.eqlin.marginals

    def integer(self,pool,fallback):
        keys,mat,c=self.problem(pool)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="Unrecognized options detected")
            ans=milp(c,integrality=np.ones(len(c)),bounds=Bounds(0,1),
                     constraints=LinearConstraint(mat,1,1),
                     options=dict(threads=1,node_limit=256,time_limit=5.,mip_rel_gap=0.))
        if ans.status not in (0,1):raise RuntimeError(ans.message)
        fleet=fallback
        if ans.x is not None:
            r=tuple(pool[keys[k]][1] for k in np.flatnonzero(ans.x>.5))
            if sorted(v for t in r for v in t)!=sorted(v for t in self.parent for v in t):
                raise ValueError("Incomplete integer cover")
            if sum(map(self.value,r))>sum(map(self.value,fleet)):
                fleet=r
        u=sum(map(self.value,fleet))
        bound=getattr(ans,"mip_dual_bound",None)
        gap=None if bound is None else max(0.,self.basis.sum()-bound-u)
        return fleet,u,gap


def ranks(x):
    return rankdata(x, method="average") / len(x)


def propose(master,common,prices,distance,signal,mass,with_prices=True):
    pool=common.copy()
    attempts=0
    active=np.array(master.hidden,dtype=int)
    depot=distance[0]
    for anchor in list(master.skeletons)+[(v,) for v in master.hidden]:
        free=anchor[0] in master.item
        beam=[anchor]
        for _ in range(4):
            branch={}
            for r in beam:
                options=active[(mass[active] <= 20000.-sum(mass[j] for j in r)+1.e-7)
                               & ~np.isin(active,r)]
                if free:options=options[options>anchor[0]]
                if not len(options):continue
                a=np.array((0,)+r); b=np.array(r+(0,))
                dist=2*depot[options,None]-distance[options[:,None],a]-distance[options[:,None],b]+distance[a,b]
                utility=ranks(dist.ravel())
                if signal is not None:
                    edges=signal[a,options[:,None]]+signal[options[:,None],b]-signal[a,b]-signal[0,options,None]-signal[options,0,None]
                    utility=.5*(utility+ranks(edges.ravel()))
                if with_prices:
                    low=np.array([-prices[master.item[v]] for v in options])
                    utility+=.5*np.repeat(ranks(low),len(a))
                for q in np.argsort(-utility,kind="stable")[:3]:
                    j,pos=divmod(int(q),len(a)); v=int(options[j]); seq=r[:pos]+(v,)+r[pos:]
                    outcome=master.insert(pool,seq);attempts+=1
                    if outcome is None:continue
                    key,u=outcome
                    gain=u-master.basis[list(key)].sum()
                    if with_prices:gain-=prices[list(key)].sum()
                    if key not in branch or gain>branch[key][0]:
                        branch[key]=(gain,seq)
            beam=[p for _,p in sorted(branch.values(),key=lambda t:(-t[0],t[1]))[:2]]
            if not beam:break
    return pool,attempts
