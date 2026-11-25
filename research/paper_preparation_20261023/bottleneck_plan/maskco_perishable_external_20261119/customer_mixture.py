"""Preserve the physical depot probability and mix conditional directed customer predictions."""
import numpy as np


def probabilities(logits):
    z=logits-logits.max(axis=1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(axis=1,keepdims=True)


def combine(physical,neural,distance=None,shuffle_seed=None):
    ph=np.array(physical,dtype=np.float64,copy=True)
    nh=np.array(neural,dtype=np.float64,copy=True)
    np.fill_diagonal(ph,-1.e4)
    np.fill_diagonal(nh,-1.e4)
    joint=probabilities(ph)
    base=probabilities(ph[:,1:])
    learned=probabilities(nh[:,1:])
    if shuffle_seed is not None:
        order=np.argsort(-distance[:,1:],axis=1,kind="stable")
        rng=np.random.default_rng(shuffle_seed)
        delta=np.log(np.maximum(learned,1.e-200))-np.log(np.maximum(base,1.e-200))
        for i,indices in enumerate(order):
            candidates=indices[indices!=i-1]
            for start in range(0,len(candidates),8):
                block=candidates[start:start+8]
                delta[i,block]=delta[i,rng.permutation(block)]
        learned=probabilities(np.log(np.maximum(base,1.e-200))+delta)
    p=np.column_stack((joint[:,0],(1-joint[:,0,None])*.5*(base+learned)))
    np.fill_diagonal(p,0.)
    return np.log(np.maximum(p,1.e-200))
