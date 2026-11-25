"""Mask-guided cross-vehicle edits with exact cold-chain splitting along the full visit order."""
import math
import numpy as np


def chain(routes,first,dist):
    available=set(range(len(routes)))
    fleet=[]
    index=first
    while available:
        r=routes[index];fleet.extend(r);available.remove(index)
        if available:
            index=min(available,key=lambda i:(dist[r[-1],routes[i][0]],i))
    return tuple(fleet)


def split_order(order,mass,value):
    n=len(order);dp=[-math.inf]*(n+1);dp[0]=0.
    previous=[-1]*(n+1);calls=0
    for i in range(n):
        if not math.isfinite(dp[i]):continue
        weight=0.
        for j in range(i+1,n+1):
            weight+=mass[order[j-1]]
            if weight>20000.+1.e-7:break
            calls+=1
            u=dp[i]+value(order[i:j])
            if u>dp[j]+1.e-7:
                dp[j]=u;previous[j]=i
    if previous[n]<0:
        raise ValueError("No full feasible partition")
    end=n;routes=[]
    while end:
        start=previous[end]
        routes.append(order[start:end]);end=start
    return dp[n],tuple(reversed(routes)),calls


def propose(initial,mask,routes,score,mass,seed,count):
    rng=np.random.default_rng(seed)
    customers=sorted(initial)
    hidden=set(mask)
    owner={v:j for j,r in enumerate(routes) for v in r}
    priority={i:sorted((j for j in customers if i!=j and mass[i]+mass[j]<=20000.+1.e-7),
                       key=lambda j:(-score[i,j],j)) for i in mask}
    def choose(eligible,quantile):
        return eligible[min(len(eligible)-1,int(quantile**2*min(8,len(eligible))))]
    for t in range(count):
        seq=list(initial)
        for step in range(1+t%4):
            i=mask[int(rng.integers(len(mask)))]
            mate_draw,target_draw,side=rng.random(3)
            block=[i]
            if (t+step)%2:
                partners=[j for j in priority[i] if j in hidden and owner[j]!=owner[i]]
                if partners:block.append(choose(partners,mate_draw))
            endpoint=block[-1] if side<.5 else block[0]
            targets=[j for j in priority[endpoint] if j not in block and owner[j]!=owner[endpoint]]
            if not targets:continue
            target=choose(targets,target_draw)
            for customer in block:seq.remove(customer)
            pos=seq.index(target)+int(side>=.5)
            seq[pos:pos]=block
        yield tuple(seq)
