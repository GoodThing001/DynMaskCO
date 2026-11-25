"""Reassign intact customer segments across routes with one exact bipartite assignment."""
from functools import lru_cache
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment

def segment_patterns(parts,seed,limit=48):
    rng=np.random.default_rng(seed);patterns=[]
    for draw in range(limit*20):
        choices=[]
        for part in parts:
            n=len(part);size=int(rng.integers(0,n+1));start=int(rng.integers(n-size+1)) if size else 0
            choices.append((start,size))
        if draw%2:
            whole,empty=rng.choice(len(parts),2,replace=False)
            choices[whole]=(0,len(parts[whole]));choices[empty]=(0,0)
        key=tuple(choices)
        if any(size for _,size in key) and key not in patterns:patterns.append(key)
        if len(patterns)==limit:break
    return patterns

def insertions(base,seg):
    if not seg:return (base,)
    directions=(seg,) if len(seg)==1 else (seg,seg[::-1])
    return tuple(dict.fromkeys(base[:p]+s+base[p:] for s in directions for p in range(len(base)+1)))

def assignments(parts,pattern,mass,measure):
    released=[r[a:a+n] for r,(a,n) in zip(parts,pattern)]+[()]
    skeletons=[r[:a]+r[a+n:] for r,(a,n) in zip(parts,pattern)]+[()]
    source=parts+[()];size=len(source);metrics=len(measure(()))
    table=np.full((metrics,size,size),-np.inf);route={}
    loads=lambda groups:np.asarray([sum(mass[i] for i in r) for r in groups])
    fits=loads(skeletons)[:,None]+loads(released)[None,:]<=20000.+1.e-7
    for i,j in zip(*np.where(fits)):
        options=insertions(skeletons[i],released[j]);values=np.asarray([measure(r) for r in options])
        best=values.argmax(axis=0)
        table[:,i,j]=values[best,np.arange(metrics)]
        route[i,j]=options[best[0]]
    # Every old segment can be returned to its original slot with its original direction.
    solved=[];real=[]
    for t in range(metrics):
        row,col=linear_sum_assignment(table[t],maximize=True)
        solved.append(float(table[t,row,col].sum()))
        if t==0:
            real=[route[i,j] for i,j in zip(row,col) if route[i,j]]
    base=np.asarray([measure(r) for r in source]).sum(axis=0);delta=np.asarray(solved)-base
    if np.any(delta< -1.e-6):raise ValueError("Old assignment lost")
    if sorted(v for r in real for v in r)!=sorted(v for r in parts for v in r):raise ValueError("Segment customer coverage")
    return np.maximum(delta,0.),real if delta[0]>1.e-7 else parts
