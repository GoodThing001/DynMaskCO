"""Evaluate leave-geometry-out recognition of whole directed insertion positions."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from directed_decoder_fit import HERE,SOURCES,assemble,design,optimize,weighting,read
from customer_mixture import combine
from directed_decoder_features import normalized_logits

ARMS=("distance","oneway","physical","shuffled","incident")
DEST=HERE/"results/headcapture01"


def prec(routes):
    y=np.full(101,-1,dtype=int)
    for r in routes:
        for a,b in zip([0]+r,r):y[b]=a
    if np.any(y[1:]<0):raise ValueError("Incomplete predecessor labels")
    return y


def choices(routes,dist,mass,client):
    index=next(k for k,r in enumerate(routes) if client in r)
    path=routes[index];pos=path.index(client)
    if len(path)==1:return None
    a=path[pos-1] if pos else 0
    b=path[pos+1] if pos+1<len(path) else 0
    starts=[];ends=[]
    for k,route in enumerate(routes):
        r=route if k!=index else path[:pos]+path[pos+1:]
        if mass[r].sum()+mass[client]>20000.+1.e-7:continue
        starts.extend([0]+r);ends.extend(r+[0])
    left,right=np.array(starts),np.array(ends)
    truth=np.flatnonzero((left==a)&(right==b))
    if len(truth)!=1:raise ValueError("Teacher position not recovered")
    detour=dist[left,client]+dist[client,right]-dist[left,right]
    return left,right,truth[0],-detour,bool(a==0 or b==0)


def main():
    start=time.process_time()
    rows,selected=assemble()
    foldmap=read(DEST/"fit.json")["folds"]
    meta=read(HERE/"heldout_sources.json")
    cases=defaultdict(list);preds=defaultdict(list);old=read(HERE/"replicate01.json")
    for ix,b in enumerate(old["blocks"]):
        c,seed=b["case"],b["seed"]
        if (c,seed) in selected:
            rr=read(HERE/f"results/replicate01/out/{ix}.json")["arms"][selected[c,seed]]["routes"]
            cases[c].append(rr)
            preds[c].append(prec(rr))
    for row in rows:row["pred"]=np.array(preds[row["case"]])[:,row["mask"]].T
    choices_by_case={}; counts=defaultdict(int);by=defaultdict(lambda:defaultdict(list))
    for c,solutions in cases.items():
        nodes=np.asarray(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2)
        choices_by_case[c]=[[None]+[choices(routes,d,nodes[:,3],i) for i in range(1,101)] for routes in solutions]
    for k in range(5):
        forward=dict(np.load(DEST/f"readout_fold{k}.npz"))
        stats=[forward[n] for n in ("mu","sd","eh","pca")]
        train=[dict(e,targets=e["pred"]) for e in rows if foldmap[e["case"]]!=k]
        w=weighting(train)
        backward={m:optimize(train,[design(e,stats,m=="neural") for e in train],w) for m in ("physical","neural")}
        np.savez_compressed(DEST/f"predecessor_fold{k}.npz",**backward)
        for ix,e in enumerate(rows):
            c=e["case"]
            if foldmap[c]!=k:continue
            matrices={}
            for name,coef in (("next",forward),("prev",backward)):
                raw={}
                for m in ("physical","neural"):
                    x=design(e,stats,m=="neural")
                    raw[m]=e["dist"]+(x@coef[m])@x.T
                matrices[name,"physical"]=normalized_logits(raw["physical"])
                matrices[name,"mix"]=combine(raw["physical"],raw["neural"])
                matrices[name,"shuffled"]=combine(raw["physical"],raw["neural"],e["dist"],56329+ix+(1000 if name=="prev" else 0))
            group=e["src"],e["geo"],c
            for i in e["mask"]:
                for teacher in choices_by_case[c]:
                    context=teacher[i]
                    if context is None:
                        counts["singleton"]+=1
                        continue
                    left,right,actual,D,edge=context
                    F=matrices["next","mix"]
                    score=dict(distance=D,oneway=F[left,i]+F[i,right]-F[left,right])
                    for m,name in (("mix","incident"),("physical","physical"),("shuffled","shuffled")):
                        score[name]=matrices["prev",m][i,left]+matrices["next",m][i,right]
                    idx=np.arange(len(left))
                    for arm,v in score.items():
                        rank=1+int((v>v[actual]).sum())+int(((v==v[actual])&(idx<actual)).sum())
                        record=(int(rank==1),int(rank<=8),rank)
                        by[group][arm,"all"].append(record)
                        by[group][arm,"end" if edge else "inside"].append(record)
                    counts["scored"]+=1
    out={}
    if counts["scored"]+counts["singleton"]!=25920:raise ValueError("Wrong number of teacher queries")
    lines=["# 整个掩码插位的识别能力","","从独立教师路线删去一个客户，在容量允许的位置中找回其原前驱与后继。表中为五折留几何的前八命中率，原路线的首尾都计入。","",
           "| 来源 | 绕行距离 | 单向边差 | 双向纯物理 | 双向置乱 | 双向 MASKCO |","|---|---:|---:|---:|---:|---:|"]
    both=0;passed=True
    for s in SOURCES:
        gs=sorted({g for src,g,c in by if src==s})
        by_geo={g:{gr:{arm:np.mean([np.mean(d[arm,gr],axis=0) for (ss,gg,c),d in by.items() if ss==s and gg==g and len(d[arm,gr])],axis=0).tolist() for arm in ARMS}
                   for gr in ("all","end","inside")} for g in gs}
        avg={gr:{a:np.mean([g[gr][a] for g in by_geo.values()],axis=0).tolist() for a in ARMS} for gr in ("all","end","inside")}
        q={a:avg["all"][a][1] for a in ARMS}
        passed &= q["incident"]>q["physical"] and q["incident"]>=q["distance"]
        both+=int(q["incident"]>max(q["oneway"],q["shuffled"]))
        out[s]=dict(averages=avg,by_geometry=by_geo)
        lines.append("| "+s+" | "+" | ".join(f"{q[a]:.2%}" for a in ARMS)+" |")
    passed=bool(passed and both>=2)
    lines+=["",f"总计 {counts['scored']:,} 个非单客车查询，另有 {counts['singleton']:,} 个单客车查询。进入完整限时求解开发的门槛"+("通过。" if passed else "未通过。"),"","[实验方案](../../双向掩码插位.md)"]
    (DEST/"incident_slot.json").write_text(json.dumps(dict(sources=out,passed=passed,counts=dict(counts),cpu_s=time.process_time()-start),allow_nan=False,indent=2))
    (DEST/"双向插位.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":main()
