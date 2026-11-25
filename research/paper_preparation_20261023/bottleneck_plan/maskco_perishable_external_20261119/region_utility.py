"""Predict which distinct nearby route region has a useful economic reconstruction."""
from collections import defaultdict
from statistics import mean
import json
import numpy as np
from scipy.stats import rankdata
from mask_utility_data import OUT,HERE,read,ranked,shuffle_edges
from mask_utility_fit import ARMS,predict

def summarize(x):
    return [float(np.mean(x)),float(np.std(x)),float(np.min(x)),float(np.max(x))]

def link_stats(parts,matrix):
    old=[];cross=[];both=[];hub=[]
    for r in parts:
        for pos,i in enumerate(r):
            a=r[pos-1] if pos else 0;b=r[pos+1] if pos+1<len(r) else 0
            this=min(matrix[a,i],matrix[i,b])
            other=[j for q in parts if q!=r for j in q]
            alt=max([max(matrix[i,j],matrix[j,i]) for j in other],default=this)
            old.append(this);cross.append(alt);both.append(alt-this);hub.append(max(matrix[0,i],matrix[i,0]))
    return sum(map(summarize,(old,cross,both,hub)),[])

def groups():
    meta=read(HERE/"heldout_sources.json");scores=defaultdict(list)
    for ci in range(5):
        f=read(OUT/f"utility_{ci}.json");c=f["case"];n=np.array(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(n[:,None,1:3]-n[None,:,1:3],axis=2);dr=ranked(-d)
        indexed={(r["phase"],r["state"],r["rep"],tuple(map(tuple,r["parts"]))):r for r in f["rows"]}
        seen=set()
        for phase in ("jointcapture01","utilitycapture01"):
            b=read(HERE/(phase+".json"))["blocks"][ci]
            with np.load(HERE/"results"/phase/f"out/{ci}.npz") as fwd:emb=fwd["embed"]
            for ri,r in enumerate(b["regions"]):
                if r["group"]!="neighbour":continue
                parts=tuple(map(tuple,r["parts"]));ident=r["state"],tuple(sorted(parts))
                if ident in seen:continue
                seen.add(ident)
                table=indexed[phase,r["state"],r["rep"],parts]["candidates"]
                two=[table[0],table[min(6,len(table)-1)]]
                h=emb[ri].astype(float);raw=h@h.T
                null=shuffle_edges(raw,d,np.random.default_rng(1259881+128*ci+int(r["rep"])+16*int(r["state"]=="terminal")))
                ids=[i for path in parts for i in path]
                spare=[1.-sum(n[list(path),3])/20000. for path in parts]
                x=dict(rep=r["rep"],pool=len(table),gain=max(t["gain"] for t in two),all_gain=max(t["gain"] for t in table),
                       d=link_stats(parts,dr)+summarize(d[np.ix_(ids,ids)])+summarize(d[ids,0]),
                       p=[len(parts),len(ids),sum(spare)]+summarize(spare)+summarize(n[ids,3]/20000.)+
                       summarize(n[ids,5]-n[ids,4]),h=link_stats(parts,ranked(raw)),s=link_stats(parts,ranked(null)))
                scores[c,r["state"]].append(x)
    return [dict(case=c,state=s,candidates=options) for (c,s),options in scores.items()]

def main():
    g=groups()
    if len(g)!=10 or sum(len(r["candidates"]) for r in g)!=42:raise ValueError("Distinct region count changed")
    results=[]
    for c in sorted({r["case"] for r in g}):
        train=[r for r in g if r["case"]!=c];test=[r for r in g if r["case"]==c]
        pred={arm:predict(train,test,keys) for arm,keys in ARMS.items()}
        for j,r in enumerate(test):
            q=r["candidates"]
            sel={arm:q[int(np.argmax(v[j]))] for arm,v in pred.items()}
            results.append(dict(case=c,state=r["state"],available=len(q),gain={arm:t["gain"] for arm,t in sel.items()},
                    selected={arm:t["rep"] for arm,t in sel.items()},two_best=max(t["gain"] for t in q),full_best=max(t["all_gain"] for t in q)))
    metric=lambda arr:{a:mean(r["gain"][a] for r in arr) for a in ARMS}|{k:mean(r[k] for r in arr) for k in ("two_best","full_best")}
    levels={s:metric([r for r in results if r["state"]==s]) for s in ("initial","terminal")}
    total=metric(results);wins=sum(metric([r for r in results if r["case"]==c])["neural"]>metric([r for r in results if r["case"]==c])["distance"]+1.e-6 for c in {r["case"] for r in results})
    passed=wins>=3 and all(v["neural"]>v["distance"]+1.e-6 for v in levels.values()) and all(total["neural"]>total[a]+1.e-6 for a in ("physical","shuffled"))
    data=dict(levels=levels,overall=total,wins=wins,passed=passed,results=results)
    (OUT/"area_selection.json").write_text(json.dumps(data,indent=2),encoding="utf8")
    lines=["# 先选路线区域，再同时重新配车","","42 个互异区域按五个几何和两种起点分组。各方法选一区，固定试两个原有掩码，以独立复算的较优净增益计分。选区上界为预知这两个掩码结果时可选到的最大值。","",
           "| 起点 | 距离 U | 纯物理 U | MASKCO U | 置乱 U | 选区上界 U | 区域及掩码全试 U |","|---|---:|---:|---:|---:|---:|---:|"]
    for label,v in levels.items():
        lines.append("| "+label+" | "+" | ".join(f"{(0. if abs(v[a])<1.e-6 else v[a]):.3f}" for a in list(ARMS)+["two_best","full_best"])+" |")
    lines+=["",f"按几何先均初终态，热图相对距离 {total['neural']-total['distance']:+.3f} U，相对纯物理 {total['neural']-total['physical']:+.3f} U，优于距离 {wins}/5 个几何。三来源开发门槛"+("通过。" if passed else "未通过。"),
            "","[逐几何选中区域](area_selection.json) · [预定门槛](../../按潜在换车收益选择路线区域.md)"]
    (OUT/"区域选择.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":main()
