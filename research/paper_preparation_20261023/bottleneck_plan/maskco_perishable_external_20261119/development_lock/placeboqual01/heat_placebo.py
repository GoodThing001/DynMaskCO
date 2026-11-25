"""Randomize customer-edge alignment within distance and mask strata, preserving symmetric logits."""
import numpy as np

def remap(log_heat,dist,routes,removed,rng):
    h=np.asarray(log_heat,dtype=float)
    z=h[0]-h[:,0]
    raw=h+z[:,None]
    a,b=np.triu_indices(len(h),1)
    sym=(raw[a,b]+raw[b,a])/2
    hidden=np.zeros(len(h),dtype=bool);hidden[np.asarray(removed,dtype=int)]=True
    observed=np.zeros(h.shape,dtype=bool)
    for r in routes:
        for i,j in zip((0,)+tuple(r),tuple(r)+(0,)):
            if not hidden[i] and not hidden[j]:observed[i,j]=observed[j,i]=True
    types=4*(a==0)+2*(hidden[a].astype(int)+hidden[b].astype(int))+observed[a,b]
    # Separate depot stratum from all nondepot mask-visibility states.
    types=10*(a==0)+types
    changed=sym.copy()
    for typ in np.unique(types):
        ids=np.flatnonzero(types==typ)
        sorted_ids=ids[np.argsort(dist[a[ids],b[ids]],kind="stable")]
        for bucket in np.array_split(sorted_ids,min(8,len(sorted_ids))):
            changed[bucket]=sym[rng.permutation(bucket)]
    def normalize(edges):
        m=np.full(h.shape,-1e9)
        m[a,b]=m[b,a]=edges
        m-=m.max(axis=1,keepdims=True)
        return m-np.log(np.exp(m).sum(axis=1,keepdims=True))
    return normalize(sym),normalize(changed)

class PairedRanking:
    def __init__(self,provider,seed,null):
        self.provider,self.seed,self.null=provider,seed,null
        self.changed_top8=[]
        self.reconstruction_error=0.

    def __getattr__(self,key):return getattr(self.provider,key)

    def heatmap(self,task,routes,removed,variant,step):
        original=self.provider.heatmap(task,routes,removed,variant,step)
        real,shuffled=remap(original,np.asarray(task.ins["distance"]),routes,removed,np.random.default_rng([self.seed,step,81061]))
        off=~np.eye(len(real),dtype=bool)
        self.reconstruction_error=max(self.reconstruction_error,float(np.max(np.abs(real[off]-original[off]))))
        ids=np.asarray(removed,dtype=int)
        sorted_real=np.argsort(-real[ids],axis=1,kind="stable")[:,:8]
        sorted_null=np.argsort(-shuffled[ids],axis=1,kind="stable")[:,:8]
        self.changed_top8.append(float(np.mean([1-len(set(x)&set(y))/8 for x,y in zip(sorted_real,sorted_null)])))
        return shuffled if self.null else original
