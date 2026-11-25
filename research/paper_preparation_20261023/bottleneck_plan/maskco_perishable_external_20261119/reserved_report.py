"""Recover and audit the complete complementary-panel comparison on fresh cases and seeds."""
from pathlib import Path
from datetime import datetime
from statistics import mean
import json
import sys
from collections import defaultdict
HERE=Path(__file__).resolve().parent
DEST=HERE/"reserved_validation"
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
import reserved_assess as assess
import audit_delivery
from phase_plan import read
from mechanism_summary import source_means

FAMILIES=["PDPCDPG-D100","XTW-p100","GH-p100"]
def f(u):return f"{u:+.2f}"
def main():
    assess.HERE=audit_delivery.HERE=DEST
    audit_delivery.audit("reserve02");assess.report("reserve02")
    dir=DEST/"results/reserve02";trial=read(dir/"analysis.json")
    verified=read(dir/"independent_audit.json");lock=read(DEST/"locked_method.json")
    method=lock["candidate"];control=lock["matched"];meta=read(DEST/"heldout_sources.json")["cases"]
    if (trial["complete"]!=1053 or verified["endpoints"]!=1053 or verified["status"]!="PASS" or len(meta)!=27):raise ValueError("Missing independent comparison")
    passed=trial["passes_numerical_criteria"][method]
    lines=["# MASKCO 掩码重构的预留面板验证","",f"锁定 {method}。27 个新客户组合、3 个新种子、13 法的 1,053 个终点通过独立复算，共 {verified['routes']:,} 条路线。原 11 法阵容外，加入上一锁版及其同规则距离。预登记的三来源热图增量与排名联合门槛{'通过' if passed else '未通过'}。","",
           "| 来源 | 原 11 法 / 全 13 法名次 | 相对同规则距离 U | 改善几何 | 正向种子 | 相对 DIST U | 相对 DIST_SLOT U | 相对 ALNS U | 最高分参照及差 U |","|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    summary=[];values=[]
    for src in FAMILIES:
        v=trial["sources"][src];diff=v["contrasts"][method];x=diff[control]
        values.append(x["mean"])
        best=max((a for a in v["scores"] if a!=method),key=v["scores"].get)
        row=f"| {src} | {v['canonical_rank_11']} / {v['ranks'][method]} | {f(x['mean'])} | {x['wins']}/{x['geometries']} | {x['positive_seeds']}/3 | {f(diff['DIST']['mean'])} | {f(diff['DIST_SLOT']['mean'])} | {f(v['external_contrasts']['ALNS']['mean'])} | {best}，{f(v['scores'][method]-v['scores'][best])} |"
        lines.append(row);summary.append(row)
    lo,hi=trial["three_source_equal_weight_ci95"]
    lines+=["","| 来源 | 新神经版减旧 HYBRID U | 新距离版减旧 HYBRID_D U | 两版热图增量之差 U |","|---|---:|---:|---:|"]
    for s in FAMILIES:
        z=trial["sources"][s]["scores"];h=z[method]-z["HYBRID"];d=z[control]-z["HYBRID_D"]
        lines.append(f"| {s} | {f(h)} | {f(d)} | {f(h-d)} |")
    lines+=["",f"三来源等权神经增量 {f(mean(values))} U，固定三次种子、按每个来源的几何整簇重采样 95% 描述区间 [{f(lo)}, {f(hi)}] U。","",
            "| 来源 | 三个种子各自增益 U | 各种子名次 | 最差几何 U | 最差单块 U |","|---|---|---|---|---|"]
    for s in FAMILIES:
        r=trial["sources"][s];x=r["contrasts"][method][control]
        min_geo=min(x["by_geometry"],key=x["by_geometry"].get);mapping=[c for c,v in meta.items() if v["geometry"]==min_geo and v["dataset"]==s]
        bad,c,seed=r["worst_paired_heat_delta"]
        lines.append(f"| {s} | {' / '.join(f(i) for i in x['by_seed'].values())} | {' / '.join(str(j) for j in r['seed_ranks'].values())} | {'、'.join(mapping)} {f(x['by_geometry'][min_geo])} | {c}/{seed} {f(bad)} |")
    lines+=["","| 方法 | 丹麦补集 净效益 U（名次） | XTW 净效益 U（名次） | GH800 净效益 U（名次） |","|---|---:|---:|---:|"]
    for m in sorted(trial["sources"][FAMILIES[0]]["scores"],key=lambda k:(k!=method,k)):
        lines.append("| "+m+" | "+" | ".join(f"{trial['sources'][s]['scores'][m]:,.2f}（{trial['sources'][s]['ranks'][m]}）" for s in FAMILIES)+" |")
    external=("ALNS","VNS","NDS-Econ","CaR","DeepACO","POMO","AttentionModel")
    lines+=["","| 来源 | 最佳独立方法 | 锁定方法相对它的 U（%） | 对同规则距离的相对增益 |","|---|---|---:|---:|"]
    for s in FAMILIES:
        scores=trial["sources"][s]["scores"]
        top=max(external,key=scores.get);delta=scores[method]-scores[top]
        lines.append(f"| {s} | {top} | {f(delta)}（{100*delta/scores[top]:+.3f}%） | {100*(scores[method]-scores[control])/scores[control]:+.3f}% |")
    lines+=["","| 终点仍为共同初解 | 丹麦 /15 | XTW /30 | GH800 /36 |","|---|---:|---:|---:|"]
    for a in (method,control,*external):
        lines.append("| "+a+" | "+" | ".join(str(trial["sources"][s]["endpoints_at_initial_profit"][a]) for s in FAMILIES)+" |")
    comps={}
    geometry_scores=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    import profit_core as core
    registry=read(DEST/"heldout_sources.json")
    dk_models={c:audit_delivery.model_for(core,registry,c,True) for c,z in meta.items() if z["dataset"]=="PDPCDPG-D100"}
    dk_fleet=defaultdict(list)
    cfg=read(DEST/"reserve02.json")
    for i,b in enumerate(cfg["blocks"]):
        r=read(dir/f"out/{i}.json")["arms"]
        for arm,w in r.items():
            geometry_scores[meta[b["case"]]["dataset"]][meta[b["case"]]["geometry"]][arm].append(w["U"])
            if b["case"] in dk_models:
                routes=dk_models[b["case"]].solution(w["routes"])
                dk_fleet[arm].append((len(routes),max(v.returned-v.departure for v in routes)))
        h,d=r[method]["endpoint"],r[control]["endpoint"]
        row=dict(quality=d["quality_loss_value_U"]-h["quality_loss_value_U"],distance=d["distance_km"]-h["distance_km"],
                 refrigeration=d["refrigeration_cost_U"]-h["refrigeration_cost_U"])
        if abs(sum(row.values())-r[method]["U"]+r[control]["U"])>1e-6:raise ValueError("Economics differ")
        comps.setdefault(b["case"],[]).append(row)
    cost_by_source=source_means({c:{k:mean(r[k] for r in rows) for k in rows[0]} for c,rows in comps.items()},meta)
    lines+=["","| 来源 | 品质价值改善 U | 里程成本节省 U | 制冷成本节省 U |","|---|---:|---:|---:|"]
    for s in FAMILIES:
        lines.append("| "+s+" | "+" | ".join(f(cost_by_source[s][k]) for k in ("quality","distance","refrigeration"))+" |")
    lines+=["","丹麦 200 请求母实例给定 67 辆车、单路线最长 550 分钟。此处使用其中 100 个配送请求，并按共同模型限制最多 100 辆车。[作者原件与字段](https://data.mendeley.com/datasets/93w8ynrk59/2)","",
            "| 丹麦方法 | 实际用车范围 | 15 个终点中全部单条路线均 ≤550 分钟 | 最长单路线（分钟） |","|---|---:|---:|---:|"]
    for arm in (method,control,"HYBRID","ALNS"):
        vals=dk_fleet[arm]
        lines.append(f"| {arm} | {min(n for n,t in vals)}–{max(n for n,t in vals)} | {sum(t<=550+1e-6 for n,t in vals)}/15 | {max(t for n,t in vals):.1f} |")
    source_audit=read(DEST/"source_audit.json")
    if source_audit["status"]!="PASS" or source_audit["checked_customer_records"]!=2700:raise ValueError("Incomplete projection audit")
    lines+=["","| 来源 | 与此前单例最大客户坐标重叠 /100 | 逐次留出一个几何的热图增量范围 U | 逐次留出后的 13 法名次范围 |","|---|---:|---:|---:|"]
    for s in FAMILIES:
        groups=geometry_scores[s];effects=[];ranks=[]
        for omitted in groups:
            keep=[rs for geo,rs in groups.items() if geo!=omitted]
            scores={a:mean(mean(rs[a]) for rs in keep) for a in keep[0]}
            effects.append(scores[method]-scores[control])
            ranks.append(1+sum(v>=scores[method]-1e-6 for a,v in scores.items() if a!=method))
        duplicate=source_audit["maximum_customer_overlap_with_previous_same_source_cases"][s]
        lines.append(f"| {s} | {duplicate} | [{f(min(effects))}, {f(max(effects))}] | {min(ranks)}–{max(ranks)} |")
    lines+=["","## 三轮独立证据","",
            "| 受检方法 | 第一来源和热图增量 U | XTW 热图增量 U | GH 热图增量 U | 三来源名次 |",
            "|---|---|---:|---:|---|"]
    first=read(PREV/"results/confirm01/analysis.json")
    second=read(HERE/"results/replicate01/analysis.json")
    for study,name,source,m in ((first,"首轮 HYBRID","VRPTWQ-C1","HYBRID"),(second,"补充 HYBRID","PDPCDPG-D100","HYBRID"),(trial,"预留 "+method,"PDPCDPG-D100",method)):
        ds=study["sources"];ss=(source,"XTW-p100","GH-p100")
        differences=[ds[s]["contrasts"][m][m+"_D"]["mean"] for s in ss]
        lines.append(f"| {name} | {source} {f(differences[0])} | {f(differences[1])} | {f(differences[2])} | {' / '.join(str(ds[s]['ranks'][m]) for s in ss)} |")
    cost=trial["costs"];times=read(dir/"status.json")
    duration=(datetime.fromisoformat(times["end"])-datetime.fromisoformat(times["start"])).total_seconds()
    lines+=["","丹麦补集取 200 请求原件中与首次 100 请求集合不重复的全部客户请求，保留作者原配送时窗。两批样本共享 a–e 五个母实例，该来源检验同一业务背景中新请求的迁移。XTW-n351 至 n429 选用预定连续十例，GH800 六类各前两例；两者均取原仓库及前 100 个客户。比较目标为统一冷链净效益，单物理核搜索 60 秒。","",
            f"求解配额 {cost['search_budget_s']:,} 单核秒，实测求解与审计 CPU {cost['cpu_s']:,.1f} 秒，阶段墙钟 {duration/60:.1f} 分钟。","",
            "[预设名单与门槛](../开发方案.md) · [上一锁版的外部复核](../RESULTS.md) · [逐例逐种子](results/reserve02/paired_test.csv) · [独立复算](results/reserve02/independent_audit.json)"]
    if method=="ORDER_SPLIT":
        from order_mechanism import stage
        stage("reserve02")
        lines+=["","[单次求解分车纪录与新旧两版热图增量之差](results/reserve02/分车机制.md)"]
    (DEST/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Joint gate",passed,"mean",mean(values),"CI",(lo,hi))
if __name__=="__main__":main()
