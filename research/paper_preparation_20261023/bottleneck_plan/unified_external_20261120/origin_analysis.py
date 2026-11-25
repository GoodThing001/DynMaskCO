"""Reconstruct paired original-model effects and replay the four-arm contemporary endpoints."""
from collections import defaultdict
from datetime import datetime
import csv
import json
from pathlib import Path
import sys
import numpy as np
from origin_campaign import HERE, OLD, MOD, read
from origin_worker import ARMS, check

META = read(OLD/"matrix.json")["cases"]
SOURCES = ("VRPTWQ-C2","XTW-p100","GH-p100")
EPS = 1e-6

def stats(rows, contrasts):
    out={}
    for dataset in dict.fromkeys(r["source"] for r in rows):
        part=[r for r in rows if r["source"]==dataset]
        geometries=sorted({r["geometry"] for r in part})
        seeds=sorted({r["seed"] for r in part})
        rng=np.random.default_rng(65201)
        index=rng.integers(len(geometries),size=(20000,len(geometries)))
        out[dataset]={}
        for name,(a,b) in contrasts.items():
            block=np.array([[np.mean([r[a]-r[b] for r in part if r["seed"]==seed and r["geometry"]==geom]) for seed in seeds] for geom in geometries])
            geo=block.mean(axis=1)
            ci=np.quantile(geo[index].mean(axis=1),[.025,.975]).tolist() if len(geo)>1 else None
            out[dataset][name]=dict(mean=float(geo.mean()),ci95=ci,
                wins=int((geo>EPS).sum()),ties=int((abs(geo)<=EPS).sum()),geometries=len(geo),
                by_seed=dict(zip(map(str,seeds),block.mean(axis=0).tolist())))
    return out

def history():
    raw=OLD/"deliveries/crossmain01/cells"
    rows=[]
    for p in (MOD/"results/orderdev01/out").glob("*.json"):
        receipt=read(p)
        case,seed=receipt["case"],receipt["seed"]
        g=dict(case=case,seed=seed,source=META[case]["dataset"],geometry=META[case]["geometry"])
        new=receipt["arms"];first=(new["ORDER_SPLIT"]["initial_routes"],new["ORDER_SPLIT"]["initial_U"])
        for m in ("MASK","SLOT","DIST","DIST_SLOT"):
            r=read(raw/f"{case}_{seed}_{m}.json")
            r["role"]="validation"
            check(r,m,receipt,r["cpu"],first);g[m]=r["U"]
        for m,r in new.items():
            if m in ("DIST","DIST_SLOT"):
                r["role"]="validation"
            check(r,m,receipt,r["cpu"],first)
            g[m if m not in ("DIST","DIST_SLOT") else m+"_repeat"]=r["U"]
        rows.append(g)
    assert len(rows)==len({(r["case"],r["seed"]) for r in rows})==78
    pairs={"order_minus_mask":("ORDER_SPLIT","MASK"),"order_minus_slot":("ORDER_SPLIT","SLOT"),
           "slot_minus_mask":("SLOT","MASK"),"mask_minus_distance":("MASK","DIST"),
           "slot_minus_slot_distance":("SLOT","DIST_SLOT"),"order_minus_order_distance":("ORDER_SPLIT","ORDER_SPLIT_D"),
           "distance_batch_shift":("DIST_repeat","DIST"),"distance_slot_batch_shift":("DIST_SLOT_repeat","DIST_SLOT")}
    return stats(rows,pairs),rows

def hybrid():
    out=[]
    path=MOD/"reserved_validation/results/reserve02/paired_test.csv"
    with path.open(encoding="utf-8-sig",newline="") as stream:
        for r in csv.DictReader(stream):
            out.append(dict(source=r["source"],seed=int(r["seed"]),geometry=r["geometry"],**{k:float(r[k]) for k in ("ORDER_SPLIT","HYBRID","ORDER_SPLIT_D","HYBRID_D")}))
    assert len(out)==81
    return stats(out,dict(locked_over_hybrid=("ORDER_SPLIT","HYBRID"),distance_over_hybrid_distance=("ORDER_SPLIT_D","HYBRID_D")))

def direct():
    folder=HERE/"results/origin01"
    plan,status=read(folder/"phase.json"),read(folder/"status.json")
    assert status["end"] and status["cells"]==64 and status["complete"]==16 and not status["failed"]
    project=next(p for p in HERE.parents if (p/"CC_Compare").is_dir())
    sys.path.insert(0,str(project/"C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"))
    sys.path.insert(0,str(HERE.parent/"current_state_economic_plan_20261117/B"))
    import profit_core as core
    import xtw_adapter
    import experiment_scope as access
    xtw_adapter.install(dict(read(OLD/"matrix.json"),paths=dict(workspace=str(OLD))))
    rows=[];maxerr=0.;routes=0;nc=0
    counts=defaultdict(lambda:defaultdict(float))
    for i,b in enumerate(plan["blocks"]):
        case,seed=b["case"],b["seed"]
        g=dict(source=META[case]["dataset"],geometry=META[case]["geometry"],case=case,seed=seed)
        with access.restricted_scope(access.Scope("local-origin-replay",frozenset([(0,"validation",case)]),frozenset(["initialize"]))):
            model=core.ProfitTask(case,fold=0,role="validation")
            initial=[list(r) for r in model.initial()]
            common=(initial,model.evaluate(initial,audit=True)["net_benefit_U"])
            cores=set()
            for m in b["methods"]:
                r=read(folder/"out"/f"{i}_{m}.json")
                check(r,m,b,r["cpu"],common)
                cores.update(r["cpu"])
                q=model.evaluate(r["routes"],audit=True)
                assert q["independently_audited"] and 0<=r["selected_s"]<=60
                for k in ("net_benefit_U","distance_km","quality_loss_value_U","refrigeration_cost_U","minimum_quality","used_vehicles"):
                    if abs(q[k]-r["endpoint"][k])>EPS:raise ValueError((i,m,k))
                err=abs(q["net_benefit_U"]-r["U"]);maxerr=max(err,maxerr)
                if err>EPS:raise ValueError((i,m,err))
                routes+=len(r["routes"]);g[m]=r["U"];nc+=1
                counts[m]["cpu_s"]+=r["cpu_s"];counts[m]["wall_s"]+=r["timing"]["total_measured_s"]
                counts[m]["setup_s"]+=r["setup_s"]
            assert len(cores)==1
        rows.append(g)
    assert nc==64
    pairs=dict(order_minus_mask=("ORDER_SPLIT","MASK"),order_minus_slot=("ORDER_SPLIT","SLOT"),
               slot_minus_mask=("SLOT","MASK"),order_minus_order_distance=("ORDER_SPLIT","ORDER_SPLIT_D"))
    elapsed=(datetime.fromisoformat(status["end"])-datetime.fromisoformat(status["start"])).total_seconds()
    return dict(audit=dict(status="PASS",endpoints=nc,routes=routes,maximum_absolute_U_error=maxerr),
                contrasts=stats(rows,pairs),costs=counts,controller_seconds=elapsed,allocated_online_core_seconds=3840),rows

def direct_report(out):
    current=out["direct"]
    info=current["audit"]
    sources=list(SOURCES)
    fmt=lambda x:f"{x:+,.2f}"
    lines=["# 原版 MASKCO 与经济分车同批复核","",
           f"8 个已测开发案例，两个种子，四臂各独占同一物理核 60 秒。{info['endpoints']} 个终点、{info['routes']:,} 条路线独立复算，最大差 {info['maximum_absolute_U_error']:.2g} U。","",
           "| 来源 | 比较 | ΔU | 95% 几何簇描述区间 U | 改善几何 | 种子 1000 / 1001 |",
           "|---|---|---:|---:|---:|---:|"]
    names={"order_minus_mask":"ORDER_SPLIT − MASK","order_minus_slot":"ORDER_SPLIT − SLOT",
           "slot_minus_mask":"SLOT − MASK","order_minus_order_distance":"ORDER_SPLIT − ORDER_SPLIT_D"}
    for source in sources:
        for key,name in names.items():
            q=current["contrasts"][source][key]
            ci=q["ci95"]
            intr=f"[{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "单一几何"
            lines.append(f"| {source} | {name} | {fmt(q['mean'])} | {intr} | {q['wins']}/{q['geometries']} | " +
                         " / ".join(fmt(x) for x in q["by_seed"].values())+" |")
    costs=current["costs"]
    lines+=["",f"配额 3,840 在线核秒，实测在线墙钟及收尾复算合计 {sum(v['wall_s'] for v in costs.values()):,.1f} 秒，CPU {sum(v['cpu_s'] for v in costs.values()):,.1f} 秒。数据无关的准备累计 {sum(v['setup_s'] for v in costs.values()):,.1f} 秒。控制器用时 {current['controller_seconds']/60:.1f} 分钟。",
            "","[同批四臂逐案例与种子](paired.csv) · [原始配对与热图置乱](../../原始MASKCO与机制消融.md)"]
    (HERE/"results/origin01/RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")

if __name__=="__main__":
    hist,rows=history()
    out=dict(historical=hist,concurrent_reserved=hybrid(),historical_pairs=len(rows))
    if (HERE/"results/origin01/status.json").exists():
        out["direct"],current=direct()
        direct_report(out)
        with (HERE/"results/origin01/paired.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(current[0]));w.writeheader();w.writerows(current)
    p=HERE/"origin_effects.json"
    p.write_text(json.dumps(out,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf8")
    print(json.dumps({k:{n:round(w["mean"],2) for n,w in v.items()} for k,v in hist.items()},ensure_ascii=False))

