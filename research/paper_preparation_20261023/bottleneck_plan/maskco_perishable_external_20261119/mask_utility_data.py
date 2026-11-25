"""Price a fixed pool of structured movable-customer masks and collect blind selection features."""
from functools import lru_cache
from itertools import combinations
from math import comb
from pathlib import Path
import sys,time,json
import numpy as np
from route_columns import HERE,read,core,independent_audit,ranked
from joint_reinsertion import optimize,nominate

OUT=HERE/"results/utilitycapture01"

def bank(parts,d,mass,seed):
    rng=np.random.default_rng(seed);ids=sorted(i for r in parts for i in r);k=min(12,len(ids))
    cap=min(24,comb(len(ids),k))
    if cap<=24 and comb(len(ids),k)<=24:return list(combinations(ids,k))
    owner={i:n for n,r in enumerate(parts) for i in r}
    score=ranked(-d)
    savings={v:d[r[j-1] if j else 0,v]+d[v,r[j+1] if j+1<len(r) else 0]-d[r[j-1] if j else 0,r[j+1] if j+1<len(r) else 0] for r in parts for j,v in enumerate(r)}
    spread=max(1.,float(np.median(d[np.ix_(ids,ids)])))
    masks=[]
    for family in range(4):
        accepted=0
        for it in range(80):
            if accepted==6:break
            eps=rng.gumbel(size=101);anchor=int(rng.choice(ids))
            if family==0:mask=rng.choice(ids,k,replace=False)
            elif family==1:
                order=sorted(range(len(parts)),key=lambda n:(n!=owner[anchor],min(d[anchor,v] for v in parts[n])/spread-.5*eps[parts[n][0]]))
                mask=[j for n in order for j in sorted(parts[n],key=lambda i:-eps[i])][:k]
            elif family==2:mask=nominate(parts,score,eps*.8,k)
            else:
                priority={i:savings[i]/spread+mass[i]/20000.+eps[i]*.8 for i in ids}
                out=[]
                for i in sorted(ids,key=lambda j:-priority[j]):
                    if i in out:continue
                    out.append(i)
                    other=[j for j in ids if owner[i]!=owner[j] and j not in out]
                    if other:
                        j=min(other,key=lambda j:d[i,j]/spread+2*abs(mass[i]-mass[j])/20000.-.5*eps[j])
                        out.append(j)
                    if len(out)>=k:break
                mask=out[:k]
            row=tuple(sorted(map(int,mask)))
            if len(row)==k and row not in masks:masks.append(row);accepted+=1
    while len(masks)<cap:
        row=tuple(sorted(rng.choice(ids,k,replace=False).tolist()))
        if row not in masks:masks.append(row)
    return masks[:cap]

def edge_features(parts,mask,score):
    sel=set(mask);old=[];cross=[];both=[];adv=[]
    for r in parts:
        for pos,i in enumerate(r):
            if i not in sel:continue
            a=r[pos-1] if pos else 0;b=r[pos+1] if pos+1<len(r) else 0
            v=min(score[a,i],score[i,b]);old.append(v)
            other=[j for s in parts if s!=r for j in s]
            f=lambda js:max([max(score[i,j],score[j,i]) for j in js],default=v)
            c=f(other);g=f([j for j in other if j in sel])
            cross.append(c);both.append(g);adv.append(g-v)
    return [x for arr in (old,cross,both,adv) for x in (np.mean(arr),np.std(arr),np.max(arr))]

def physical_features(parts,mask,nodes):
    sub=set(mask);counts=np.array([sum(i in sub for i in r) for r in parts])
    released=np.array([sum(nodes[i,3] for i in r if i in sub)/20000. for r in parts])
    residual=[1.-sum(nodes[i,3] for i in r if i not in sub)/20000. for r in parts]
    mass=nodes[list(mask),3]/20000.;tw=nodes[list(mask),5]-nodes[list(mask),4]
    return [sum(n==len(r) for n,r in zip(counts,parts)),(counts>0).sum(),max(counts),
            np.mean(released),max(released),np.std(released),np.std(residual),np.mean(mass),np.std(mass),max(mass),
            np.mean(tw),np.std(tw),np.min(tw)]

def shuffle_edges(raw,d,rng):
    i,j=np.triu_indices(len(d),1);idx=np.argsort(d[i,j],kind="stable");s=raw.copy()
    for g in np.array_split(idx,12):
        vals=raw[i[g],j[g]][rng.permutation(len(g))]
        s[i[g],j[g]]=vals;s[j[g],i[g]]=vals
    return s

def run(ci):
    meta=read(HERE/"heldout_sources.json");archive=read(HERE/"archived_states.json");found=[];cpu=time.process_time();wall=time.perf_counter()
    phases=("jointcapture01","utilitycapture01")
    c=read(HERE/(phases[0]+".json"))["blocks"][ci]["case"]
    nodes=np.asarray(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
    d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);mass=nodes[:,3];dr=ranked(-d)
    model=core.PublicationModel(dict(name=c,nodes=nodes.tolist(),distance=d.tolist()),core.load_config(),"main_Product1",.8)
    @lru_cache(maxsize=40000)
    def value(r):
        z=model.route(r);return z.net_benefit if z.feasible else -np.inf
    for pi,phase in enumerate(phases):
        b=read(HERE/(phase+".json"))["blocks"][ci];folder=HERE/"results"/phase
        if b["case"]!=c or read(folder/f"out/{ci}.json")["case"]!=c:raise ValueError("Forward case")
        with np.load(folder/f"out/{ci}.npz") as src:
            emb=src["embed"];masks=src["mask"]
        for ri,r in enumerate(b["regions"]):
            parts=list(map(tuple,r["parts"]));state=r["state"];seed=156881+ci*4096+pi*128+ri
            if masks[ri].tolist()!=r["mask"]:raise ValueError("Occluded region identity")
            routes=list(map(tuple,archive[c][state]));outside=[a for a in routes if a not in parts]
            initial=independent_audit(model,[model.route(a) for a in routes],True)
            h=emb[ri].astype(float);raw=h@h.T;neural=ranked(raw);placebo=ranked(shuffle_edges(raw,d,np.random.default_rng(seed)))
            candidates=bank(parts,d,mass,seed)
            if len(set(candidates))!=min(24,comb(sum(map(len,parts)),min(sum(map(len,parts)),12))):raise ValueError("Candidate supply")
            priced=[]
            for m in candidates:
                sol=optimize(parts,m,mass,value);u=independent_audit(model,[model.route(a) for a in outside+sol["chosen"]],True)
                if abs(u-initial-sol["gain"])>1.e-6:raise ValueError("Audit")
                priced.append(dict(mask=m,gain=u-initial,optimal=sol["optimal"],gap_U=sol["gap_U"],chosen=sol["chosen"],
                    d=edge_features(parts,m,dr),p=physical_features(parts,m,nodes),h=edge_features(parts,m,neural),s=edge_features(parts,m,placebo)))
            found.append(dict(case=c,state=state,rep=r["rep"],phase=phase,parts=parts,candidates=priced))
    path=OUT/f"utility_{ci}.json";path.write_text(json.dumps(dict(case=c,rows=found,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall),
        default=lambda x:x.item(),allow_nan=False),encoding="utf8")
    print(c,len(found),sum(len(r["candidates"]) for r in found),flush=True)

if __name__=="__main__":
    from concurrent.futures import ProcessPoolExecutor
    if len(sys.argv)>1:run(int(sys.argv[1]))
    else:
        with ProcessPoolExecutor(max_workers=2) as jobs:list(jobs.map(run,range(5)))
