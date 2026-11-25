"""Audit and report the complete prospective three-source replication."""
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
import assess
import audit_delivery
from mechanism_summary import source_means

PHASE="replicate01"
OUT=HERE/"results"/PHASE
SOURCES=["PDPCDPG-D100","XTW-p100","GH-p100"]
METHOD="HYBRID"
CONTROL="HYBRID_D"
def read(path):return json.loads(path.read_text(encoding="utf8"))
def fmt(x):return f"{x:+.2f}"

def report():
    a=read(OUT/"analysis.json"); audit=read(OUT/"independent_audit.json")
    if audit["endpoints"]!=891 or a["complete"]!=891 or audit["status"]!="PASS":
        raise ValueError("Independent grid incomplete")
    data=read(HERE/"heldout_sources.json"); plan=read(HERE/(PHASE+".json"))
    groups={}
    import profit_core as core
    danish_models={c:audit_delivery.model_for(core,data,c,True) for c,m in data["cases"].items() if m["dataset"]=="PDPCDPG-D100"}
    fleet={}
    for i,block in enumerate(plan["blocks"]):
        rr=read(OUT/"out"/f"{i}.json")["arms"]
        if block["case"] in danish_models:
            model=danish_models[block["case"]]
            for method,record in rr.items():
                routes=model.solution(record["routes"])
                fleet.setdefault(method,[]).append((len(routes),max(r.returned-r.departure for r in routes)))
        h,d=rr[METHOD]["endpoint"],rr[CONTROL]["endpoint"]
        if any(r["role"]!="external_test" for r in rr.values()):raise ValueError("External role mismatch")
        q=d["quality_loss_value_U"]-h["quality_loss_value_U"]
        cold=d["refrigeration_cost_U"]-h["refrigeration_cost_U"]
        distance=d["distance_km"]-h["distance_km"]
        profit=h["net_benefit_U"]-d["net_benefit_U"]
        if abs(q+cold+distance-profit)>1e-6:raise ValueError("Profit components do not reconcile")
        groups.setdefault(block["case"],[]).append(dict(quality=q,mileage=distance,refrigeration=cold,profit=profit))
    economics=source_means({c:{k:mean(v[k] for v in arr) for k in arr[0]} for c,arr in groups.items()},data["cases"])
    old=read(PREV/"results/confirm01/analysis.json")
    ranks=[a["sources"][s]["ranks"][METHOD] for s in SOURCES]
    passed=a["passes_numerical_criteria"][METHOD]
    interval=a["three_source_equal_weight_ci95"]
    vals=[a["sources"][s]["contrasts"][METHOD][CONTROL]["mean"] for s in SOURCES]
    lines=["# MASKCO 掩码重构在三个新外部面板的复核","",f"预定的三来源联合门槛{'通过' if passed else '未通过'}。27 例、3 个新种子、11 法的全部 891 个终点完成，独立复算 {audit['routes']:,} 条路线。","",
           "| 外部来源 | 11 法名次 | 相对同规则距离 U | 改善几何 | 对原 DIST U | 对原 DIST_SLOT U | 对 ALNS U |","|---|---:|---:|---:|---:|---:|---:|"]
    for s in SOURCES:
        ds=a["sources"][s]; v=ds["contrasts"][METHOD]; r=v[CONTROL]
        lines.append(f"| {s} | {ds['ranks'][METHOD]} | {fmt(r['mean'])} | {r['wins']}/{r['geometries']} | {fmt(v['DIST']['mean'])} | {fmt(v['DIST_SLOT']['mean'])} | {fmt(ds['external_contrasts']['ALNS']['mean'])} |")
    lines+=["",f"三来源等权热图增量 {fmt(mean(vals))} U，几何簇重采样的 95% 描述区间为 [{fmt(interval[0])}, {fmt(interval[1])}] U，固定三次搜索种子。","","| 来源 | 32001 / 32002 / 32003 增量 U | 几何簇 95% 区间 U | 最差配对 U |","|---|---|---|---|"]
    for s in SOURCES:
        ds=a["sources"][s];r=ds["contrasts"][METHOD][CONTROL]; ci=r["ci95"]; low,c,seed=ds["worst_paired_heat_delta"]
        lines.append(f"| {s} | {' / '.join(fmt(x) for x in r['by_seed'].values())} | [{fmt(ci[0])}, {fmt(ci[1])}] | {c} / {seed}，{fmt(low)} |")
    from figure import plot
    plot()
    lines+=["","![两次确认的热图配对增量与完整排名](independent_effects.png)","","点为来源内几何等权的热图净增量，线为固定种子的几何簇 95% 描述区间。首轮 VRPTWQ-C1 只有一个几何。"]
    lines+=["","## 完整方法比较","","净效益越高越好。沿用预设的保守名次，同分方法均列该组末位。","","| 方法 | PDPCDPG-D100 净效益 U（名次） | XTW 净效益 U（名次） | GH600 净效益 U（名次） |","|---|---:|---:|---:|"]
    for m in sorted(a["sources"][SOURCES[0]]["scores"],key=lambda m:(m!=METHOD,m)):
        lines.append("| "+m+" | "+" | ".join(f"{a['sources'][s]['scores'][m]:,.2f}（{a['sources'][s]['ranks'][m]}）" for s in SOURCES)+" |")
    lines+=["","| 来源 | 其余十法中最优 | HYBRID 与其差 U（%） | 最强独立方法¹ | HYBRID 与其差 U（%） | 三个种子各自名次 |",
            "|---|---|---:|---|---:|---|"]
    for s in SOURCES:
        d=a["sources"][s]; scores=d["scores"]
        others=[m for m in scores if m!=METHOD]
        rival=max(others,key=scores.get)
        independent=max((m for m in others if m not in (CONTROL,"DIST","DIST_SLOT")),key=scores.get)
        def gap(m):
            delta=scores[METHOD]-scores[m]
            return f"{fmt(delta)}（{delta/abs(scores[m])*100:+.3f}%）"
        lines.append(f"| {s} | {rival} | {gap(rival)} | {independent} | {gap(independent)} | {' / '.join(str(n) for n in d['seed_ranks'].values())} |")
    lines+=["","¹ 独立方法包括 ALNS、VNS、NDS-Econ、CaR、DeepACO、POMO、AttentionModel。NDS-Econ 使用作者的 VRPTW100 权重指导破坏，重建采用冷链经济插位；CaR、DeepACO、POMO、AttentionModel 使用各自的 11,400 步冷链适配权重，MASKCO 使用固定的 1,440 步适配权重。全部求解器采用相同初解与单核 60 秒上限，训练与加载单列。"]
    lines+=["","| 终点净效益与共同初解相同 | 易腐品 / 15 | XTW / 30 | GH600 / 36 |","|---|---:|---:|---:|"]
    for m in (METHOD,CONTROL,"ALNS","NDS-Econ","CaR","DeepACO","POMO","AttentionModel"):
        lines.append("| "+m+" | "+" | ".join(str(a["sources"][s]["endpoints_at_initial_profit"][m]) for s in SOURCES)+" |")
    lines+=["","## 易腐品原始车队口径","","作者数据表头为 33 辆车、最大路线时长 550 分钟。共同冷链模型允许最多 100 辆，固定计算配送阶段的品质与费用。原始取货、交叉转运以及全链乘载时间属于另一项任务。下表给出配送终态的实际用车及在途时长，供对照来源条件。","",
            "| 方法 | 用车范围 | ≤33 辆的终态 / 15 | 最长单条配送路线（分钟） |","|---|---:|---:|---:|"]
    for m,v in sorted(fleet.items(),key=lambda x:(x[0]!=METHOD,x[0])):
        lines.append(f"| {m} | {min(x[0] for x in v)}–{max(x[0] for x in v)} | {sum(k<=33 for k,_ in v)} | {max(t for _,t in v):.1f} |")
    lines+=["","## 热图改变了哪些成本","","| 来源 | 少损失的品质价值 U | 里程成本节省 U | 制冷成本节省 U | 净效益差 U |","|---|---:|---:|---:|---:|"]
    for s in SOURCES:
        e=economics[s]
        lines.append(f"| {s} | {fmt(e['quality'])} | {fmt(e['mileage'])} | {fmt(e['refrigeration'])} | {fmt(e['profit'])} |")
    overlap=read(HERE/"source_audit.json")
    inputs=read(HERE/"domain_profile.json")
    train=inputs["VRPTWQ train"];dk=inputs["External replication / PDPCDPG-D100"]
    lines+=["",f"丹麦来源平均单客需求占车辆容量 {100*dk['mean_capacity_share']:.1f}%，训练来源为 {100*train['mean_capacity_share']:.1f}%；最近客户距离中位数分别为 {dk['median_nearest_customer_km']:.1f} 和 {train['median_nearest_customer_km']:.1f} km。客户更密、单客更重，且配送端的服务时长口径不同。当前数据可定位这些输入差异，分离其因果贡献需专门对照。[各来源输入分布](来源输入对照.md)"]
    shared=overlap["overlap_with_most_similar_previous_case"]["xtw_n336_p100"]
    xtw=a["sources"]["XTW-p100"]["contrasts"][METHOD][CONTROL]["by_geometry"]
    geo=data["cases"]["xtw_n336_p100"]["geometry"]
    without=mean(v for g,v in xtw.items() if g!=geo)
    rr=[mean(v for other,v in xtw.items() if other!=g) for g in xtw]
    lines+=["","## 空间覆盖","",f"共复核 {overlap['checked_customer_records']:,} 行客户源字段。易腐品的五个不同几何之间至多共享 5 个配送位置。GH600 与此前受检 GH 几何至多共享 2/100 个客户。XTW-n336 与此前受检 XTW 几何重合 {shared}/100，另九例各至多 3/100。主分析纳入全部案例；单独留出 n336 时 XTW 热图均值为 {fmt(without)} U，逐个留出十个几何的均值范围为 [{fmt(min(rr))}, {fmt(max(rr))}] U。[逐例核查](source_audit.json)"]
    lines+=["","## 与首轮独立验证的关系","","| 首轮来源 | HYBRID 名次 | 热图对同规则距离 U |","|---|---:|---:|"]
    for s in ("VRPTWQ-C1","XTW-p100","GH-p100"):
        v=old["sources"][s];lines.append(f"| {s} | {v['ranks'][METHOD]} | {fmt(v['contrasts'][METHOD][CONTROL]['mean'])} |")
    lines+=["","| 两次确认累计来源 | 学习热图减同规则距离 U | 改善几何 / 几何总数 | 固定种子、按批次分层的几何簇 95% 区间 U |","|---|---:|---:|---|"]
    import numpy as np
    rng=np.random.default_rng(20261119)
    for s in ("VRPTWQ-C1","PDPCDPG-D100","XTW-p100","GH-p100"):
        gs={};draws=[]
        for trial in (old,a):
            if s not in trial["sources"]:continue
            values=trial["sources"][s]["contrasts"][METHOD][CONTROL]["by_geometry"]
            if set(gs)&set(values):raise ValueError("Repeated geometry in cumulative comparison")
            gs.update(values)
            x=np.array(list(values.values()))
            draws.append(x[rng.integers(len(x),size=(20000,len(x)))].sum(axis=1))
        ci=np.quantile(sum(draws)/len(gs),[.025,.975])
        interval_text="单一几何" if len(gs)==1 else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(f"| {s} | {fmt(mean(gs.values()))} | {sum(u>1e-6 for u in gs.values())}/{len(gs)} | {interval_text} |")
    lines+=["","两次研究使用同一锁定权重、热图解码器、单核 60 秒和相同 11 法。第二次使用不同客户几何与种子，易腐品静态配送子问题属于新来源。两次结果分列，以免更换来源后掩盖 C1 的负差。","",
            "易腐品数据采用作者归档中的 5 个静态 100 请求原件，按原仓库、配送点和配送时窗评价。底层几何来自丹麦物流数据衍生的公开算例，托盘需求按车辆容量比例换算，本研究采用统一的冷链品质与货值模型。结果适用于该静态配送投影。","",
            "阶段校准的 4,320 个训练插位机会经 R 与 RC 交叉复核。从 R 转到 RC 的短名单遗憾增加 0.344 U，反向持平，因此独立比较采用此前锁定配方。"]
    t=read(OUT/"status.json");wall=(datetime.fromisoformat(t["end"])-datetime.fromisoformat(t["start"])).total_seconds()
    cost=a["costs"]
    tech=read(HERE/"results/encqual01/out/0.json")
    dev=read(HERE/"results/cachescreen01/screen_gate.json")
    e=read(HERE/"results/edgeprobe01/analysis.json")
    lines+=["","## 旧开发例上的机制消融","",f"同一掩码复用编码并按当前路线重新解码，训练例 40 次前向的 CPU 从 {tech['full_cpu_s']:.2f} 秒降到 {tech['reuse_cpu_s']:.2f} 秒，非对角对数概率最大差 {tech['max_abs_log_heat_error']:.2g}。随后 8 个开发案例、2 个种子、3 臂的 48 个整程终点全部独立复算。","",
            "| 开发来源 | 循环掩码，热图减同规则距离 U | 循环掩码，复用减完整前向 U | 热图推荐连接的机会增量，初解 / 距离终态 U |",
            "|---|---:|---:|---:|"]
    for s in ("VRPTWQ-C2","XTW-p100","GH-p100"):
        lines.append(f"| {s} | {fmt(dev['heat_minus_distance'][s])} | {fmt(dev['reuse_minus_full'][s])} | {fmt(e['initial_routes'][s]['delta_U'])} / {fmt(e['routes'][s]['delta_U'])} |")
    lines+=["","最后一列使用同一起点、同一掩码和相同路线编辑。在初解上热图推荐的连接可获得更多收益；在已优化的距离终态上，两种推荐的合计可实现增量相同。该列为单掩码诊断。循环掩码版未达到升级门槛，两次外测均使用原 HYBRID。"]
    static=read(HERE/"results/staticscreen01/gate.json")
    training=read(HERE/"results/staticprobe01/gate.json")
    lines+=["",f"保留每轮随机掩码、只做一次实例编码，在 1,440 个训练插位机会中节约 {100*training['saved_cpu_fraction']:.1f}% 前向 CPU。独立于训练的八例开发筛查中，其相对同规则纯距离的终点增量分别为 C2 {fmt(static['heat_minus_distance']['VRPTWQ-C2'])}、XTW {fmt(static['heat_minus_distance']['XTW-p100'])}、GH {fmt(static['heat_minus_distance']['GH-p100'])} U，此版本未晋级。"]
    lines+=["","## 论文结论","",
            f"同一 MASKCO 方法在首轮 C1、XTW、GH 的 11 法名次为 2、3、1，在新易腐品配送、XTW、GH 的名次为 {'、'.join(map(str,ranks))}。名单包含原 DIST、原 DIST_SLOT 与同规则纯距离。学习模型在每一轮从受掩码路线预测邻接关系，其后由冷链净效益在距离、原始热图和校准热图以及两种重插方式之间选取。"]
    positive=[s for s,v in zip(SOURCES,vals) if v>1e-6]
    lines+=["",f"新面板中 {len(positive)}/3 个来源的几何等权热图增量为正，预先规定的三来源稳定正收益及排名联合门槛{'通过' if passed else '未通过'}。原 VRPTWQ-C1 的 −4.50 U 单列保留。统计区间固定搜索种子，适用范围为所测 100 客户投影和单物理核 60 秒。"]
    lines+=["",f"本次独立比较占用 {cost['search_budget_s']:,} 单核搜索配额秒，求解与审计 CPU {cost['cpu_s']:,.1f} 秒，作业墙钟 {wall/60:.1f} 分钟。软件验收的 11 个归档案例端点另计。","",
            "[预定设计与来源](研究设计.md) · [逐例逐种子终值](results/replicate01/paired_test.csv) · [独立审计](results/replicate01/independent_audit.json) · [作者数据 DOI](https://doi.org/10.17632/93w8ynrk59.2)"]
    (HERE/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Ranks",ranks,"gate",passed,"pooled",mean(vals),interval)
if __name__=="__main__":
    audit_delivery.HERE=HERE
    assess.HERE=HERE
    audit_delivery.audit(PHASE)
    assess.report(PHASE)
    report()
