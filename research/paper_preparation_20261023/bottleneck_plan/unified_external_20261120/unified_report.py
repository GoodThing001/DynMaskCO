"""Rebuild independent schedules and report paired results for a single 16-arm delivery."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import json
import math
import sys

import numpy as np
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"recent5_20261120"))
import recent_independent as audit
from unified_plan import METHODS, EXTERNAL

OUR="ORDER_SPLIT"
EPS=1e-6
BOOT_SEED=20261120
DRAWS=20000
IDENTITY={"ALNS":"项目冷链 ALNS 适配，统一收益接受", "VNS":"项目冷链 VNS 适配，统一收益接受",
          "NDS-Econ":"作者 VRPTW100 权重选移除客户，按冷链经济收益重插",
          **{m:"作者网络的 11,400 步项目冷链适配权重" for m in ("CaR","DeepACO","POMO","AttentionModel")},
          **{m:"作者公开预训练权重直接迁移，使用冷链可行掩码与净效益选优" for m in ("RouteFinder","CaDA","MVMoE","RRNCO")},
          "PyVRP":"PyVRP 0.11.3 混合遗传搜索，内层经济代理成本，外层严格净效益选优"}

def read(p):
    return json.loads(p.read_text(encoding="utf8"))

def write(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")

def csv_rows(p,rows):
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect(root,phase,results_root=None):
    cfg=read(root/"matrix.json")
    meta=read(root/"metadata.json")
    target=(results_root or root/"results")/phase
    controller=read(target/"status.json")
    methods=cfg["methods"];blocks=cfg["blocks"]
    sources=meta["cases"];nc=len(sources);nb=len(blocks);expected=16*nb
    assert set(METHODS)==set(methods) and len(methods)==16 and len(EXTERNAL)==12
    assert cfg["seconds"]==60 and len(set((b['case'],b['seed']) for b in blocks))==nb
    seeds=sorted({b["seed"] for b in blocks})
    assert len(seeds)==3 and {(b["case"],b["seed"]) for b in blocks}=={(c,s) for c in sources for s in seeds}
    assert controller["end"] and not controller["active"] and not controller["failed"]
    assert (controller["cells"],controller["total_cells"],controller["complete"],controller["total"])==(expected,expected,nb,nb)
    files={f"{i}_{m}.json" for i,b in enumerate(blocks) for m in b["methods"]}
    assert len(files)==expected and files=={p.name for p in (target/"out").glob("*.json")}
    refs=None
    if phase=="benchmark01":
        inputs,refs=audit.original_panel()
        assert set(sources)==set(inputs)
    else:
        catalogue=read(root/"heldout_sources.json")
        assert catalogue["cases"]==sources
        inputs={c:root/e["file"] for c,e in catalogue["derived_instances"].items()}
        assert set(inputs)==set(sources)
    report=dict(status="PASS",cases=nc,geometries_per_source={},blocks=nb,arms=16,external_arms=12,endpoints=0,
                routes=0,maximum_absolute_U_error=0.,maximum_initial_U_error=0.)
    result={};long=[]
    for i,b in enumerate(blocks):
        key=(b["case"],b["seed"])
        assert set(b["methods"])==set(methods) and len(b["methods"])==len(methods)
        common=audit.original_model(inputs[b["case"]],b["case"])
        if refs is not None:audit.reference(key,refs[key],*common[1:])
        group={};cores=set()
        for m in methods:
            row=read(target/"out"/f"{i}_{m}.json")
            nr,err,initial_err=audit.replay(row,key,m,*common)
            report["routes"]+=nr;report["endpoints"]+=1
            report["maximum_absolute_U_error"]=max(report["maximum_absolute_U_error"],err)
            report["maximum_initial_U_error"]=max(report["maximum_initial_U_error"],initial_err)
            cores.add(row["cpu"][0])
            end=row["endpoint"]
            rec=dict(block=i,case=key[0],seed=key[1],dataset=sources[key[0]]["dataset"],
                     geometry=sources[key[0]]["geometry"],method=m,core=row["cpu"][0],
                     selected_s=row["selected_s"],U=row["U"],initial_U=row["initial_U"],
                     cpu_s=row["cpu_s"],wall_s=row.get("wall_s",row.get("timing",{}).get("total_measured_s")),
                     setup_s=row["setup_s"],vehicles=end["used_vehicles"],distance_km=end["distance_km"],
                     quality_loss_U=end["quality_loss_value_U"],refrigeration_U=end["refrigeration_cost_U"],
                     minimum_quality=end["minimum_quality"],initial_retained=abs(row["U"]-row["initial_U"])<=EPS,
                     initial_routes_retained=row["initial_routes"]==row["routes"],
                     native_route_requests=row.get("native_route_requests"),
                     python_route_requests=row.get("requests"),forwards=row.get("forwards"),
                     native_iterations=row.get("iterations"),source=row.get("source"))
            assert all(math.isfinite(rec[k]) for k in ["U","wall_s","cpu_s","setup_s"])
            long.append(rec);group[m]=rec
        assert len(cores)==1
        result[key]=group
    for name in {x["dataset"] for x in sources.values()}:
        report["geometries_per_source"][name]=len({v["geometry"] for v in sources.values() if v["dataset"]==name})
    assert report["endpoints"]==expected
    write(target/"independent_audit.json",report)
    return cfg,meta,controller,report,result,long,seeds

def ranks(values):
    return {k:1+sum(x>=v-EPS for m,x in values.items() if k!=m) for k,v in values.items()}

def matrix(cases,meta,seeds,fn):
    groups={}
    for case in cases:
        groups.setdefault(meta["cases"][case]["geometry"],[]).append([fn(case,s) for s in seeds])
    names=sorted(groups)
    return names,np.asarray([np.mean(groups[g],axis=0) for g in names])

def interval(a):
    return np.quantile(a,[.025,.975]).tolist()

def holm(ps):
    ordered=sorted(ps,key=ps.get);peak=0.;result={}
    for i,k in enumerate(ordered):
        peak=max(peak,min(1.,(len(ordered)-i)*ps[k]))
        result[k]=peak
    return result

def panel_stats(name,cases,meta,data,seeds,rng):
    matrices={}
    for m in METHODS:
        names,matrices[m]=matrix(cases,meta,seeds,lambda c,s:data[c,s][m]["U"])
    scores={m:float(a.mean()) for m,a in matrices.items()}
    index=rng.integers(len(names),size=(DRAWS,len(names)))
    diffs={};draws={}
    for m in METHODS[1:]:
        d=matrices[OUR]-matrices[m];gd=d.mean(axis=1)
        draws[m]=gd[index].mean(axis=1)
        wins,lost=int((gd>EPS).sum()),int((gd < -EPS).sum())
        exact_p=float(binomtest(wins,wins+lost).pvalue) if wins+lost else 1.
        worst_idx=int(np.argmin(gd))
        wu,wc,ws=min((data[c,s][OUR]["U"]-data[c,s][m]["U"],c,s) for c in cases for s in seeds)
        parts={}
        for key in ["quality_loss_U","distance_km","refrigeration_U"]:
            _,arr=matrix(cases,meta,seeds,lambda c,s:data[c,s][m][key]-data[c,s][OUR][key])
            parts[key]=float(arr.mean())
        mean=float(gd.mean())
        if abs(sum(parts.values())-mean)>EPS:raise ValueError(f"Economic parts do not reconcile: {name} {m}")
        diffs[m]=dict(mean=mean,pct_relative=100.*mean/scores[m],ci95=interval(draws[m]),
                      wins=wins,ties=len(gd)-wins-lost,losses=lost,sign_p=exact_p,
                      by_seed=dict(zip(map(str,seeds),d.mean(axis=0).tolist())),
                      by_geometry=dict(zip(names,gd.tolist())),worst_geometry=[names[worst_idx],float(gd[worst_idx])],
                      worst_case=[wc,ws,wu],economic_savings=parts)
    adj=holm({k:diffs[k]["sign_p"] for k in EXTERNAL})
    for k,p in adj.items():diffs[k]["sign_p_holm"]=p
    coverage={}
    for m in METHODS:
        cells=[data[c,s][m] for c in cases for s in seeds]
        coverage[m]=dict(feasible_endpoints=len(cells),strict_gain_over_start=sum(r["U"]-r["initial_U"]>EPS for r in cells),
                         equal_initial_U=sum(r["initial_retained"] for r in cells),same_initial_routes=sum(r["initial_routes_retained"] for r in cells))
    strongest=max(EXTERNAL,key=scores.get)
    return dict(cases=len(cases),geometries=len(names),scores=scores,coverage=coverage,
                all_ranks=ranks(scores),external_plus_candidate_ranks=ranks({k:scores[k] for k in [OUR,*EXTERNAL]}),
                strongest_external=strongest,strongest_external_identity=IDENTITY[strongest],contrasts=diffs),draws

def summary(meta,data,seeds,long,status):
    primary=meta.get("primary",list(meta["cases"]))
    sources=list(dict.fromkeys(meta["cases"][c]["dataset"] for c in sorted(primary)))
    panels={s:sorted(c for c in primary if meta["cases"][c]["dataset"]==s) for s in sources}
    if meta.get("low_overlap"):panels["XTW 低重叠"]=sorted(meta["low_overlap"])
    for name,cases in meta.get("additional_panels",{}).items():
        if name in panels or len(set(cases)) != len(cases) or not cases or not set(cases).issubset(meta["cases"]):
            raise ValueError(f"Invalid additional panel {name}")
        panels[name]=sorted(cases)
    rng=np.random.default_rng(BOOT_SEED);studies={};draws={}
    for name,cases in panels.items():
        studies[name],draws[name]=panel_stats(name,cases,meta,data,seeds,rng)
    equal={}
    for m in METHODS[1:]:
        equal[m]=dict(mean=float(np.mean([studies[s]["contrasts"][m]["mean"] for s in sources])),
            ci95=interval(np.mean([draws[s][m] for s in sources],axis=0)),
            by_seed={str(k):float(np.mean([studies[s]["contrasts"][m]["by_seed"][str(k)] for s in sources])) for k in seeds})
    costs={}
    for m in METHODS:
        r=[a for a in long if a["method"]==m]
        costs[m]=dict(n=len(r),initial_retained=sum(x["initial_retained"] for x in r),
                      cpu_s=sum(x["cpu_s"] for x in r),wall_s=sum(x["wall_s"] for x in r),setup_s=sum(x["setup_s"] for x in r))
    return dict(status="COMPLETE",endpoints=len(long),methods=METHODS,external_methods=EXTERNAL,seeds=seeds,
                primary_sources=sources,panels=studies,equal_source=equal,costs=costs,
                elapsed_s=(datetime.fromisoformat(status["end"])-datetime.fromisoformat(status["start"])).total_seconds(),
                allocated_core_s=len(long)*60,
                criteria=dict(top3_each_external=all(studies[s]["external_plus_candidate_ranks"][OUR]<=3 for s in sources),
                              top3_each_all=all(studies[s]["all_ranks"][OUR]<=3 for s in sources),
                              higher_than_all_external_each_source=all(studies[s]["contrasts"][studies[s]["strongest_external"]]["mean"]>EPS for s in sources)))

def generate(root,phase,results_root=None):
    cfg,meta,state,ac,data,long,seeds=collect(root,phase,results_root)
    target=(results_root or root/"results")/phase;z=summary(meta,data,seeds,long,state)
    write(target/"analysis.json",z);csv_rows(target/"endpoints.csv",long)
    pairs=[dict(case=c,seed=s,source=meta["cases"][c]["dataset"],geometry=meta["cases"][c]["geometry"],comparator=m,
                U=b[OUR]["U"],comparator_U=b[m]["U"],delta_U=b[OUR]["U"]-b[m]["U"]) for (c,s),b in sorted(data.items()) for m in METHODS[1:]]
    csv_rows(target/"paired.csv",pairs)
    fmt=lambda x:f"{x:+,.2f}"
    panels=z["panels"]
    lines=["# 锁版 MASKCO 与十二个外部方法的同期结果","",
         f"{ac['cases']} 个不同案例、{ac['blocks']} 个案例种子配对块、每块 16 个方法。{ac['endpoints']:,} 个终点和 {ac['routes']:,} 条路线通过独立复算，最大净效益偏差 {ac['maximum_absolute_U_error']:.3g} U。",
         "","| 来源 | 实例 / 几何 | 外部十二法加 MASKCO 名次 /13 | 全部名次 /16 | 最强外部方法 | 对它的平均差 U | 95% 描述区间 U | 胜/平/负几何 |",
         "|---|---:|---:|---:|---|---:|---|---|"]
    for s,p in panels.items():
        m=p["strongest_external"];v=p["contrasts"][m];lo,hi=v["ci95"]
        lines.append(f"| {s} | {p['cases']} / {p['geometries']} | {p['external_plus_candidate_ranks'][OUR]} | {p['all_ranks'][OUR]} | {m} | {fmt(v['mean'])} | [{fmt(lo)}, {fmt(hi)}] | {v['wins']}/{v['ties']}/{v['losses']} |")
    lines+=["","各方法均按同一可行车队的净效益 U 计分。当前来源内先均种子和同几何实例，再对每个几何等权。","",
            "| 方法 | 类型 | "+" | ".join(panels)+" |","|---|---|"+"---:|"*len(panels)]
    for m in METHODS:
        cells=[f"{p['scores'][m]:,.2f}（{p['all_ranks'][m]}）" for p in panels.values()]
        label="外部方法" if m in EXTERNAL else "完整方法" if m==OUR else "内部参照"
        lines.append(f"| {m} | {label} | "+" | ".join(cells)+" |")
    lines+=["","## 各外部方法严格改善共同初解的次数","","单元格为严格提高初解净效益的终点数 / 全部已审计可行终点数。停留在初解效益的终点也有可行路线。相同路线与相同效益分别计数，见统计 JSON。","",
            "| 外部方法 | "+" | ".join(panels)+" |","|---|"+"---:|"*len(panels)]
    for m in EXTERNAL:
        vals=[f"{p['coverage'][m]['strict_gain_over_start']}/{p['coverage'][m]['feasible_endpoints']}" for p in panels.values()]
        lines.append(f"| {m} | "+" | ".join(vals)+" |")
    lines+=["","| 来源 | 平均净效益最高的外部参照 | 实现身份 |","|---|---|---|"]
    for name,p in panels.items():
        lines.append(f"| {name} | {p['strongest_external']} | {p['strongest_external_identity']} |")
    lines+=["","## 对照与经济分解","","差值为 MASKCO 减对照。品质、里程和制冷三项为对照成本减 MASKCO 成本，里程单位系数为 1 U/km。","",
            "| 来源 | 对照 | ΔU | 相对 % | 95% 区间 | 胜/平/负 | 逐种子 U | 品质 / 路程 / 制冷节约 U |",
            "|---|---|---:|---:|---|---|---|---|"]
    for s,p in panels.items():
        for m,v in p["contrasts"].items():
            lo,hi=v["ci95"];parts=v["economic_savings"];ss=" / ".join(fmt(a) for a in v["by_seed"].values())
            part=" / ".join(fmt(x) for x in parts.values())
            lines.append(f"| {s} | {m} | {fmt(v['mean'])} | {v['pct_relative']:+.3f}% | [{fmt(lo)}, {fmt(hi)}] | {v['wins']}/{v['ties']}/{v['losses']} | {ss} | {part} |")
    lines+=["","## 资源与结果文件","",f"同核在线配额 {z['allocated_core_s']:,} 核秒。控制器墙钟 {z['elapsed_s']/60:.1f} 分钟。数据无关的模型准备时间单列。","",
            "| 方法 | 回退或保持初解效益 | CPU 累计秒 | 在线墙钟累计秒 | 准备累计秒 |","|---|---:|---:|---:|---:|"]
    for m,c in z["costs"].items():
        lines.append(f"| {m} | {c['initial_retained']}/{c['n']} | {c['cpu_s']:.1f} | {c['wall_s']:.1f} | {c['setup_s']:.1f} |")
    lines+=["","[独立复算](independent_audit.json) · [逐终点经济成本与计时](endpoints.csv) · [全部配对](paired.csv) · [按几何、逐种子、几何簇区间与符号检验](analysis.json)"]
    (target/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print(z["criteria"],ac["endpoints"],ac["routes"])

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("--root",type=Path,default=HERE);cli.add_argument("--phase",default="benchmark01")
    cli.add_argument("--results-root",type=Path,help="directory containing phase receipt directories")
    args=cli.parse_args();generate(args.root,args.phase,args.results_root)

