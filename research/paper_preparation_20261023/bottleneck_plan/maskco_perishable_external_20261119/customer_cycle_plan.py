"""Register and report three-route customer exchanges on previously used problems."""
from collections import defaultdict
from itertools import permutations,product
from pathlib import Path
from statistics import mean
import json
import math
import sys

HERE=Path(__file__).resolve().parent
SOURCES=("PDPCDPG-D100","XTW-p100","GH-p100")
ARMS=("distance","real","shuffled","oracle")


def read(path):return json.loads(path.read_text(encoding="utf8"))


def prepare():
    for phase,parent in (("cyclequal01","tailqual01"),("cycleprobe01","tailprobe01")):
        cfg=read(HERE/(parent+".json"))
        cfg.update(worker="customer_cycle_probe.py",purpose="retrospective masked three-route cycle",block_timeout_s=300)
        cfg["files"]=[f if f!="tail_group_probe.py" else cfg["worker"] for f in cfg["files"]]
        (HERE/(phase+".json")).write_text(json.dumps(cfg,indent=2),encoding="utf8")


def check():
    """Check every rotation of three real archived routes against literal reinsertion."""
    import numpy as np
    from customer_cycle_probe import probe,replacement_tables
    from prepare_panel import core
    case="dk_data100_a"
    nodes=read(HERE/f"inputs/{case}.json")["nodes"]
    dist=np.array([[math.dist(a[1:3],b[1:3]) for b in nodes] for a in nodes])
    model=core.PublicationModel(dict(nodes=nodes,distance=dist),core.load_config(),"main_Product1",.8)
    routes=tuple(map(tuple,read(HERE/"archived_states.json")[case]["initial"]))
    from types import SimpleNamespace
    task=SimpleNamespace(route=model.route,c=model.c,ins=model.ins)
    task.evaluate=lambda rs,audit:{"net_benefit_U":sum(model.route(r).net_benefit for r in rs)}
    first=next(i for i,r in enumerate(routes) if len(r)==1)
    others=[i for i,r in enumerate(routes) if len(r)>1][:4]
    chosen=[first]+others
    clients=tuple(routes[i][0] for i in chosen)
    weights=np.stack((-dist,-np.sqrt(dist),-np.log1p(dist)))
    computed,strict,_,owners=replacement_tables(task,routes,clients,weights)
    baseline=task.evaluate(routes,True)["net_benefit_U"]
    best=0.;valid=0
    for a,b,c in permutations(range(len(clients)),3):
        if a>min(b,c):continue
        targets=[];cost=0.
        for u,v in ((a,b),(b,c),(c,a)):
            old=routes[owners[clients[u]]];idx=old.index(clients[u])
            partial=old[:idx]+old[idx+1:]
            options=[partial[:z]+(clients[v],)+partial[z:] for z in range(len(old))]
            if sum(nodes[n][3] for n in options[0])<=model.c["routing"]["capacity_kg"]+1.e-7:
                def edge_total(w,r):return sum(w[i,j] for i,j in zip((0,)+r,r+(0,)))
                ref=[max(edge_total(w,r)-edge_total(w,old) for r in options) for w in weights]
                if not np.allclose(computed[u,v],ref,atol=1.e-6):raise ValueError("Edge delta accounting")
            targets.append([model.route(r) for r in options if model.route(r).feasible])
            cost+=model.route(old).net_benefit
        for rs in product(*targets):
            valid+=1;best=max(best,sum(v.net_benefit for v in rs)-cost)
    found,x,y,ids=probe(task,routes,clients,weights,baseline)
    if not valid or not len(ids) or abs(best-found["oracle"])>1.e-6:raise ValueError("Literal enumeration differs")
    print("Checked singleton and longer receivers,",valid,"feasible three-route reconstructions")


def report():
    cfg=read(HERE/"cycleprobe01.json")
    base=HERE/"results/cycleprobe01"
    meta=read(HERE/"heldout_sources.json")["cases"]
    s=read(base/"status.json")
    if not s["end"] or s["complete"]!=27 or s["failed"]:raise ValueError("Incomplete cycle run")
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    totals=dict(cycles=0,feasible=0,positive=0,requests=0,cpu_s=0.,forwards=0)
    for ix,b in enumerate(cfg["blocks"]):
        r=read(base/f"out/{ix}.json");c=b["case"]
        rows=r["rows"]
        if (r["case"]!=c or r["forwards"]!=12 or len(rows)!=12 or
            {(d["state"],d["t"]) for d in rows}!={(k,j) for k in ("initial","terminal") for j in range(6)}):
            raise ValueError(c)
        for key in ("requests","cpu_s","forwards"):totals[key]+=r[key]
        for d in rows:
            for key in ("cycles","feasible","positive"):totals[key]+=d[key]
            for arm in ARMS:
                groups[meta[c]["dataset"],d["state"]][meta[c]["geometry"]][arm].append(d["gains"][arm])
    results={};learn=True;direct=True
    lines=["# 掩码三路线客户轮换","","同一组 40 客掩码中，同步改动三条路线，每条路线接收一名新客户，独立挑选其最佳合法插位。每臂从同一列表选 32 个循环。数值为每个掩码相对原解的可行净收益 U。","",
           "| 来源 | 起点 | 距离 | 真热图 | 置乱 | 全枚举 | 真减距离 | 优于距离几何 | 上界高于距离几何 |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source in SOURCES:
        results[source]={}
        for label in ("initial","terminal"):
            g={gid:{m:mean(v) for m,v in d.items()} for gid,d in groups[source,label].items()}
            avg={m:mean(t[m] for t in g.values()) for m in ARMS}
            n=len(g);wins=sum(t["real"]>t["distance"]+1.e-7 for t in g.values())
            room=sum(t["oracle"]>t["distance"]+1.e-7 for t in g.values())
            if label=="terminal":
                learn &= 2*room>n and avg["oracle"]>avg["distance"]+1.e-7
                direct &= 2*wins>n and avg["real"]>max(avg["distance"],avg["shuffled"])+1.e-7
            results[source][label]=dict(means=avg,wins=wins,room=room,geometries=g)
            lines.append("| "+source+" | "+label+" | "+" | ".join(f"{avg[m]:.3f}" for m in ARMS)+f" | {avg['real']-avg['distance']:+.3f} | {wins}/{n} | {room}/{n} |")
    lines+=["",f"合计 {totals['cycles']:,} 个合载重的轮换，其中 {totals['feasible']:,} 个在三条路线都可行，{totals['positive']:,} 个改善原解。掩码前向 {totals['forwards']} 次，路线请求 {totals['requests']:,} 次，采集 CPU {totals['cpu_s']:.1f} 秒。","",
            "直接进入计时开发的门槛"+("通过。" if direct else "未通过。")+"单独训练三车动作排序的空间门槛"+("通过。" if learn else "未通过。"),"","[预设规则与面板](../../三路线闭环重构.md)"]
    (base/"analysis.json").write_text(json.dumps(dict(sources=results,fit_gate=learn,direct_gate=direct,totals=totals),indent=2))
    (base/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))

if __name__=="__main__":{"plan":prepare,"check":check,"report":report}[sys.argv[1]]()
