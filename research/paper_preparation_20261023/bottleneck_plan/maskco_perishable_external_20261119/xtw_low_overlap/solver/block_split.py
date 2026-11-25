"""Reconstruct a masked bundle of nearby routes using edge-guided orders and exact economic splitting."""
import ctypes
from pathlib import Path
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
F64=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
I32=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
I64=np.ctypeslib.ndpointer(dtype=np.int64,flags="C_CONTIGUOUS")
def library():
    import fcntl
    so=HERE/"split_order.so"
    with (HERE/"split_build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not so.exists():
            subprocess.run(["g++","-O3","-std=c++17","-fPIC","-shared",str(HERE/"split_order.cpp"),"-o",str(so)],check=True)
    fn=ctypes.CDLL(str(so)).split_order
    fn.argtypes=[F64,F64,F64,I32,ctypes.c_int,I32,I64];fn.restype=ctypes.c_double
    return fn

def split(fn,inputs,order):
    seq=np.ascontiguousarray(order,dtype=np.int32)
    prev=np.full(len(seq)+1,-1,dtype=np.int32);calls=np.zeros(1,dtype=np.int64)
    score=fn(*inputs,seq,len(seq),prev,calls)
    if not np.isfinite(score):raise ValueError("No feasible partition despite singleton admission")
    routes=[];j=len(seq)
    while j:
        i=int(prev[j])
        if not 0<=i<j:raise ValueError("Incomplete route partition")
        routes.append(tuple(map(int,seq[i:j])))
        j=i
    return score,tuple(reversed(routes)),int(calls[0])

def bundle(routes,dist,index):
    origin=routes[index]
    if len(origin)>24:return ()
    close=sorted((float(dist[np.ix_(origin,r)].min()),k) for k,r in enumerate(routes) if k!=index)
    size=len(origin);selected=[index]
    for _,k in close:
        if size+len(routes[k])<=24:
            selected.append(k);size+=len(routes[k])
        if len(selected)==4:break
    return tuple(sorted(selected)) if len(selected)>=2 else ()

def proposals(parts,weights,seed,limit):
    rng=np.random.default_rng(seed)
    ids=sorted(i for route in parts for i in route)
    priority={i:sorted((j for j in ids if j!=i),key=lambda j:(-(weights[i,j]+weights[j,i]),j)) for i in ids}
    for n in range(limit):
        seq=[]
        for k in rng.permutation(len(parts)):
            r=parts[k]
            seq.extend(r[::-1] if rng.random()<.25 else r)
        for _ in range(1+n%4):
            i=ids[int(rng.integers(len(ids)))]
            u=seq.index(i)
            available=[j for j in priority[i] if abs(seq.index(j)-u)>1]
            draw=rng.random(3)
            if not available:continue
            j=available[int(draw[0]**2*min(8,len(available)))]
            v=seq.index(j)
            if draw[1]<.5:
                l,r=sorted((u,v))
                seq[l+1:r+1]=reversed(seq[l+1:r+1])
            else:
                seq.pop(u);v=seq.index(j)
                seq.insert(v+int(draw[2]>=.5),i)
        yield seq
