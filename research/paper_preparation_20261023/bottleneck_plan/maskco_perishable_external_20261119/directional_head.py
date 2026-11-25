"""Cold-chain directional features from a masked customer's predicted neighbors."""
import numpy as np

def edge_features(nodes,dist,affinity):
    h=np.array(affinity,copy=True,dtype=float)
    np.fill_diagonal(h,-np.inf)
    h[:,0]=-np.inf
    order=np.argsort(-h,axis=1,kind="stable")
    rank=np.argsort(order,axis=1,kind="stable")
    local=np.exp(-np.minimum((rank+rank.T)/2,80.)/8.)
    mid=(nodes[:,4]+nodes[:,5])/2880
    load=nodes[:,3]/20000
    depot=dist[0]/max(1.,np.median(dist[0,1:]))
    expected=mid[order[:,:8]].mean(axis=1)
    fields=[local,local*(mid[None,:]-mid[:,None]),local*(load[None,:]-load[:,None]),
            local*(depot[None,:]-depot[:,None]),local*(mid[None,:]-expected[:,None]),
            local*(expected[None,:]-mid[:,None])]
    return np.stack(fields,axis=-1)

def prior_scores(a,b,i,prior):
    values=prior[a,i]+prior[i,b]-prior[a,b]
    rank=1+np.argsort(np.argsort(-values,kind="stable"),kind="stable")
    return -np.log(rank)
