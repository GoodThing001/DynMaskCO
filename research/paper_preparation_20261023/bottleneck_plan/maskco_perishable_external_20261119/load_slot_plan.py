"""Fit the load-conditioned readout on other geometries and score the held-out slots."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import json
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
SOURCES=("PDPCDPG-D100","XTW-p100","GH-p100")
ARMS=("distance","physical","shuffled","real")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def prepare():
    for stage,parent,count in (("loadqual01","oppqual01",1),("loadprobe01","opprobe01",6)):
        plan=read(HERE/(parent+".json"))
        plan["worker"]="load_slot_probe.py"
        plan["purpose"]="retrospective cross-geometry load-conditioned insertion"
        plan["masks_per_state"]=count
        plan["block_timeout_s"]=240
        del plan["groups_per_state"]
        plan["files"]=[p for p in plan["files"] if p!="paired_operator_probe.py"]+["load_slot_probe.py","heat_placebo.py"]
        (HERE/(stage+".json")).write_text(json.dumps(plan,indent=2),encoding="utf8")


def summarize():
    cfg=read(HERE/"loadprobe01.json")
    base=HERE/"results/loadprobe01"
    status=read(base/"status.json")
    if not status["end"] or status["complete"]!=27 or status["failed"]:raise ValueError(status)
    cases=read(HERE/"heldout_sources.json")["cases"]
    geometry={c:(m["dataset"],m["geometry"]) for c,m in cases.items()}
    geos={s:sorted({g for src,g in geometry.values() if src==s}) for s in SOURCES}
    fold={(s,g):j%5 for s in SOURCES for j,g in enumerate(geos[s])}
    by_geo=defaultdict(list)
    for c,g in geometry.items():by_geo[g].append(c)
    total_cpu,total_requests=0.,0
    data={}
    for ix,b in enumerate(cfg["blocks"]):
        c=b["case"];meta=read(base/f"out/{ix}.json");total_cpu+=meta["cpu_s"];total_requests+=meta["route_requests"]
        x=dict(np.load(base/f"out/{ix}.npz"))
        if (meta["case"]!=c or meta["contexts"]!=120 or meta["forwards"]!=12 or
            sum(x["lengths"])!=meta["slots"] or not all(np.isfinite(z).all() for z in x.values())):
            raise ValueError(c)
        data[c]=x
    def features(c,arm):
        x=data[c]
        if arm=="physical":return x["physical"]
        return np.column_stack((x["physical"],x[arm]))
    scores={c:{} for c in cases}
    opportunities=defaultdict(lambda:dict(contexts=0,strict_improvement_possible=0))
    for c,x in data.items():
        tally=opportunities[geometry[c][0]]
        start=0
        for n in x["lengths"]:
            tally["contexts"]+=1
            tally["strict_improvement_possible"]+=int(np.any(x["gain"][start:start+n]>1.e-7))
            start+=n
    for k in range(5):
        train=[c for c,g in geometry.items() if fold[g]!=k]
        test=[c for c,g in geometry.items() if fold[g]==k]
        for arm in ARMS:
            if arm!="distance":
                X=np.concatenate([features(c,arm) for c in train]).astype(np.float64)
                y=np.concatenate([np.log1p(data[c]["gain"]) for c in train])
                # Equal weight per source and training geometry, then per case in a geometry.
                weights=[]
                for c in train:
                    src,g=geometry[c];n=len(data[c]["gain"])
                    n_geo=sum(fold[(src,j)]!=k for j in geos[src])
                    weights.append(np.full(n,1./(3*n_geo*len(by_geo[src,g])*n)))
                w=np.concatenate(weights);center=w@X
                scale=np.sqrt(w@((X-center)**2));scale[scale<1.e-8]=1.
                X=(X-center)/scale
                coefficients=np.linalg.solve(X.T@(w[:,None]*X)+.01*np.eye(X.shape[1]),X.T@(w*(y-w@y)))
            for c in test:
                z=data[c];gains=z["gain"]
                ranking=-z["physical"][:,0] if arm=="distance" else ((features(c,arm)-center)/scale)@coefficients
                cursor=0; selected=[];oracle=[]
                for n in z["lengths"]:
                    end=cursor+n
                    ids=np.argsort(-ranking[cursor:end],kind="stable")[:6]
                    selected.append(float(np.max(gains[cursor:end][ids],initial=0.)))
                    oracle.append(float(np.max(gains[cursor:end],initial=0.)))
                    cursor=end
                scores[c][arm]=mean(selected)
                scores[c]["oracle"]=mean(oracle)
    summary={}
    lines=["# 载重条件插位的留几何预测","","每次在六个最近插位与单客路线之外再保留六个插位。数值是这六个额外席位带来的平均净效益 U。拟合与评价的客户几何互斥。","",
           "| 来源 | 纯距离 | 物理特征 | 同维置乱 | 真实 MASKCO | 遍历剩余插位 | 真减物理 | 改善几何 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    passed=True
    for s in SOURCES:
        grouped={g:{m:mean(scores[c][m] for c in by_geo[s,g]) for m in (*ARMS,"oracle")} for g in geos[s]}
        avg={m:mean(v[m] for v in grouped.values()) for m in (*ARMS,"oracle")}
        wins=sum(v["real"]>v["physical"]+1.e-7 for v in grouped.values())
        ok=all(avg["real"]>avg[m]+1.e-7 for m in ARMS[:-1]) and 2*wins>len(grouped)
        summary[s]=dict(means=avg,wins=wins,geometries=len(grouped),passes=ok,by_geometry=grouped)
        passed &= ok
        lines.append("| "+s+" | "+" | ".join(f"{avg[m]:.3f}" for m in (*ARMS,"oracle"))+f" | {avg['real']-avg['physical']:+.3f} | {wins}/{len(grouped)} |")
    lines+=["","穷举额外插位后，仍能超过最近六席和单客路线的机会数按来源为 "+
            "、".join(f"{s} {opportunities[s]['strict_improvement_possible']}/{opportunities[s]['contexts']}" for s in SOURCES)+"。这里按实际单客上下文计数。","",
            f"完整 3,240 个插位短名单、324 次真实掩码前向，严格路线请求 {total_requests:,} 次，采集 CPU {total_cpu:.1f} 秒。三来源晋级门槛"+("通过。" if passed else "未通过。"),"",
            "[预设机制、留组规则与门槛](../../载重条件插位试验.md)"]
    (base/"analysis.json").write_text(json.dumps(dict(summary=summary,scores=scores,opportunities=opportunities,fold_of_case={c:fold[g] for c,g in geometry.items()},
        passed=passed,cpu_s=total_cpu,route_requests=total_requests),indent=2))
    (base/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":
    prepare() if sys.argv[1]=="plan" else summarize()
