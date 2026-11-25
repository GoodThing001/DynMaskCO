"""Fit an out-of-geometry directed MaskCO readout on audited route successors."""
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCES=("PDPCDPG-D100","XTW-p100","GH-p100")
ARMS=("distance","frozen","physical","neural")


def read(p):return json.loads(p.read_text(encoding="utf8"))


def teacher(routes):
    succ=np.full(101,-1,dtype=int)
    for route in routes:
        for a,b in zip(route,route[1:]+[0]):
            if succ[a]>=0:raise ValueError("Repeated teacher customer")
            succ[a]=b
    if (succ[1:]<0).any():raise ValueError("Incomplete teacher")
    return succ


def distance_score(dist):
    d=np.array(dist,copy=True);np.fill_diagonal(d,np.inf)
    rank=1+np.argsort(np.argsort(d,axis=1,kind="stable"),axis=1,kind="stable")
    result=-np.log(rank).astype(np.float32)
    np.fill_diagonal(result,-1.e4)
    return result


def assemble():
    cfg=read(HERE/"headcapture01.json");p=HERE/"results/headcapture01"
    status=read(p/"status.json");meta=read(HERE/"heldout_sources.json")
    if status["failed"] or status["complete"]!=27 or not status["end"]:raise ValueError("Incomplete embedding run")
    labels=defaultdict(list);catalog={}
    prior=read(HERE/"replicate01.json")
    for i,b in enumerate(prior["blocks"]):
        if b["seed"]==32001:continue
        r=read(HERE/f"results/replicate01/out/{i}.json")
        pool=r["arms"]
        if not all(v["status"]=="AUDITED" for v in pool.values()):raise ValueError("Unaudited training label")
        arm=max(sorted(pool),key=lambda a:pool[a]["U"])
        labels[b["case"]].append(teacher(pool[arm]["routes"]))
        catalog[b["case"],b["seed"]]=arm
    examples=[]
    for ix,b in enumerate(cfg["blocks"]):
        c=b["case"];v=read(p/f"out/{ix}.json")
        if v["case"]!=c or v["contexts"]!=12 or len(labels[c])!=2 or v["max_logit_error"]>.002:raise ValueError(c)
        geo=meta["cases"][c];nodes=np.asarray(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        dist=distance_score(np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2))
        with np.load(p/f"out/{ix}.npz") as z:
            for mask,h,ph in zip(z["mask"],z["embed"],z["phys"],strict=True):
                examples.append(dict(case=c,src=geo["dataset"],geo=geo["geometry"],mask=mask.astype(int),h=h,
                            phys=ph,dist=dist,targets=np.asarray(labels[c])[:,mask].T,scale=v["scale"]))
    if len(examples)!=324:raise ValueError("Missing contexts")
    return examples,catalog


def weighting(rows):
    src_geo={s:{e["geo"] for e in rows if e["src"]==s} for s in SOURCES}
    cases=defaultdict(set)
    for e in rows:cases[e["src"],e["geo"]].add(e["case"])
    counts=Counter(e["case"] for e in rows)
    w=[1./(len(SOURCES)*len(src_geo[e["src"]])*len(cases[e["src"],e["geo"]])*counts[e["case"]]) for e in rows]
    return np.asarray(w)


def scaling(rows,w):
    ph=np.array([e["phys"] for e in rows]);h=np.array([e["h"] for e in rows],dtype=np.float64)
    mu=np.average(ph.mean(axis=1),axis=0,weights=w)
    sd=np.sqrt(np.average(((ph-mu)**2).mean(axis=1),axis=0,weights=w))
    eh=np.average(h.mean(axis=1),axis=0,weights=w);h-=eh
    flat=(h*np.sqrt(w/101)[:,None,None]).reshape(-1,h.shape[-1])
    eig,V=np.linalg.eigh(flat.T@flat)
    return mu,np.maximum(sd,.01),eh,V[:,-16:]/np.sqrt(np.maximum(.0001,eig[-16:]))


def design(row,scales,neural):
    mu,sd,eh,pca=scales
    physical=np.clip((row["phys"]-mu)/sd,-6,6)
    values=[physical,np.ones((len(physical),1))]
    if neural:values.append(np.clip((row["h"]-eh)@pca,-6,6))
    return np.concatenate(values,axis=1).astype(np.float32)


def optimize(rows,xs,ws):
    rng=np.random.default_rng(48103)
    X=np.stack(xs);mask=np.stack([e["mask"] for e in rows]);yy=np.stack([e["targets"] for e in rows])
    base=np.stack([e["dist"][e["mask"]] for e in rows])
    weights=np.zeros((X.shape[-1],X.shape[-1]),dtype=np.float32);m=np.zeros_like(weights);v=np.zeros_like(weights)
    step=0
    for epoch in range(8):
        shuffled=rng.permutation(len(rows))
        for start in range(0,len(rows),4):
            ids=shuffled[start:start+4];i=np.arange(len(ids))[:,None];j=np.arange(40)[None,:]
            all_nodes=X[ids];query=all_nodes[i,mask[ids]]
            logits=(query@weights)@np.swapaxes(all_nodes,1,2)+base[ids]
            logits-=logits.max(axis=2,keepdims=True)
            p=np.exp(logits);p/=p.sum(axis=2,keepdims=True)
            for target_col in range(2):
                p[i,j,yy[ids,:,target_col]]-=.5
            per=np.swapaxes(query,1,2)@(p@all_nodes)/40.
            grad=np.mean(per*(ws[ids]*len(rows))[:,None,None],axis=0)+.002*weights
            step+=1
            m=.9*m+.1*grad;v=.999*v+.001*grad*grad
            weights-=.01*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1.e-8)
    return weights


def measure(score,mask,targets):
    h=np.array(score[mask],copy=True);h[np.arange(len(mask)),mask]=-1.e4
    h-=h.max(axis=1,keepdims=True)
    logp=h-np.log(np.exp(h).sum(axis=1,keepdims=True))
    top=np.argsort(-h,axis=1,kind="stable")[:,:8]
    return [float(-logp[np.arange(len(mask))[:,None],targets].mean()),
            float((top[:,:,None]==targets[:,None,:]).any(axis=1).mean())]


def main():
    data,labels=assemble()
    fold={}
    for s in SOURCES:
        for i,g in enumerate(sorted({e["geo"] for e in data if e["src"]==s})):
            fold[s,g]=i%5
    scores=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    dest=HERE/"results/headcapture01"
    for k in range(5):
        train=[e for e in data if fold[e["src"],e["geo"]]!=k]
        test=[e for e in data if fold[e["src"],e["geo"]]==k]
        ws=weighting(train);stats=scaling(train,ws)
        fitted={arm:optimize(train,[design(e,stats,arm=="neural") for e in train],ws) for arm in ("physical","neural")}
        for e in test:
            raw=e["h"]@e["h"].T*e["scale"]
            choices=dict(distance=e["dist"],frozen=raw)
            for arm,W in fitted.items():
                x=design(e,stats,arm=="neural")
                choices[arm]=e["dist"]+(x@W)@x.T
            for arm,h in choices.items():
                scores[e["src"],e["geo"]][e["case"]][arm].append(measure(h,e["mask"],e["targets"]))
        np.savez_compressed(dest/f"readout_fold{k}.npz",mu=stats[0],sd=stats[1],eh=stats[2],pca=stats[3],**fitted)
    by_src={};pass_count=0;passed=True
    lines=["# 用其他几何训练的有向掩码解码器","","每个数是被遮客户的两个独立求解种子教师后继的平均表现。交叉熵越低越好，前八命中越高越好。各实例整组留出，表中先合并时窗变体，再对几何等权。","",
           "| 来源 | 排序 | 后继交叉熵 | 前八命中 |","|---|---|---:|---:|"]
    for s in SOURCES:
        geometry={}
        for (src,g),cases in scores.items():
            if src!=s:continue
            geometry[g]={m:np.mean([np.mean(v[m],axis=0) for v in cases.values()],axis=0).tolist() for m in ARMS}
        perf={m:np.mean([v[m] for v in geometry.values()],axis=0).tolist() for m in ARMS}
        for m in ARMS:lines.append(f"| {s} | {m} | {perf[m][0]:.4f} | {perf[m][1]:.2%} |")
        ok=perf["neural"][0]<perf["physical"][0] and perf["neural"][1]>=perf["distance"][1]
        passed &= ok;pass_count+=int(perf["neural"][1]>perf["frozen"][1])
        by_src[s]=dict(scores=perf,by_geometry=geometry,beats_physical_loss_and_distance_recall=ok)
    passed &= pass_count>=2
    lines+=["",f"324 个掩码上下文，12,960 个客户查询、25,920 个教师后继，五折均按几何隔离。三来源晋级实际求解的门槛"+("通过。" if passed else "未通过。"),"",
            "[训练目标、参照与门槛](../../有向掩码解码器开发.md)"]
    (dest/"fit.json").write_text(json.dumps(dict(sources=by_src,advance=passed,folds={e["case"]:fold[e["src"],e["geo"]] for e in data},
        teachers={f"{c}/{n}":a for (c,n),a in labels.items()}),indent=2))
    (dest/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":main()
