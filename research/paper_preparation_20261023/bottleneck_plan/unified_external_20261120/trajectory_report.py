"""Independent route replay and paired seed/time analysis for the 60-second histories."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"recent5_20261120"))
from recent_independent import original_panel, original_model, independent_audit, read, replay


def summarize():
    folder=HERE/"results/trajectory01"
    cfg=read(HERE/"trajectory_matrix.json")
    status=read(folder/"status.json")
    if (not status["end"] or status["failed"] or status["complete"]!=90 or status["cells"]!=360):
        raise ValueError("Incomplete ten-seed batch")
    inputs,_=original_panel()
    meta=read(HERE/"metadata.json")["cases"]
    methods=cfg["methods"]
    cuts=(5,15,30,60)
    outcomes={}
    audit_count=0
    missing=0
    max_error=0.
    cases=defaultdict(set)
    cold_rows=[]
    for i,b in enumerate(cfg["blocks"]):
        c,s=b["case"],b["seed"]
        cases[meta[c]["dataset"]].add(c)
        model,first,u0=original_model(inputs[c],c)
        cores=set()
        for m in methods:
            row=read(folder/"out"/f"{i}_{m}.json")
            replay(row,(c,s),m,model,first,u0)
            cores.update(row["cpu"])
            setup=row["entry_to_search_s"]
            if not math.isfinite(setup) or setup<0:raise ValueError("Invalid startup clock")
            point=row["cold_start_60"]
            cold_value=None
            if point is not None:
                routes=point["routes"]
                if sorted(v for route in routes for v in route)!=list(range(1,101)) or not all(routes):
                    raise ValueError("Invalid cold-start route coverage")
                cold_value=independent_audit(model,model.solution(routes),True)
                if (abs(cold_value-point["U"])>1e-6 or not 0<=point["accepted_s"]<=60-setup
                        or cold_value>row["U"]+1e-6):
                    raise ValueError("Invalid cold-start incumbent")
            cold_rows.append(dict(source=meta[c]["dataset"],case=c,seed=s,method=m,
                                  entry_to_search_s=setup,U_at_60=cold_value))
            previous=-math.inf
            for t in cuts:
                selected=row["trajectory"][str(t)]
                key=(c,s,m,t)
                if key in outcomes:raise ValueError("Duplicate time point")
                if selected is None:
                    if previous!=-math.inf or t==60:raise ValueError("Lost incumbent")
                    outcomes[key]=None
                    missing+=1
                    continue
                routes=selected["routes"]
                if sorted(v for r in routes for v in r)!=list(range(1,101)) or not all(routes):
                    raise ValueError("Incomplete snapshot coverage")
                value=independent_audit(model,model.solution(routes),True)
                err=abs(value-selected["U"])
                if (err>1e-6 or not 0<=selected["accepted_s"]<=t or value<previous-1e-6
                        or (t==60 and abs(value-row["U"])>1e-6)):
                    raise ValueError((key,err))
                max_error=max(max_error,err)
                previous=value
                outcomes[key]=value
                audit_count+=1
        if len(cores)!=1:raise ValueError("Block used different cores")
    if any(len(g)!=3 for g in cases.values()) or sum(map(len,cases.values()))!=9:
        raise ValueError("Unexpected source coverage")
    seeds=tuple(range(65101,65111))
    lines=["# 十个配对种子与限时搜索的收敛轨迹","",
           "每个来源三个代表几何，每个种子同机、同核、同一起点比较四法。表中为 MASKCO 减对照的净效益 U。"
           "早期读数取该次 60 秒搜索到对应时刻实际接受的最优路线。","",
           "| 来源 | 对照 | 5 秒 | 15 秒 | 30 秒 | 60 秒 | 60 秒胜 / 平 / 负种子 | 60 秒最差几何 ΔU |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    summaries={}
    with (folder/"paired_seeds.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(("source","case","geometry","seed","readout_s","control","paired_delta_U"))
        for src in sorted(cases):
            group=sorted(cases[src])
            summaries[src]={}
            for control in methods[1:]:
                displays=[]
                for t in cuts:
                    paired={}
                    for c in group:
                        for s in seeds:
                            a,b=outcomes[c,s,methods[0],t],outcomes[c,s,control,t]
                            value=None if a is None or b is None else a-b
                            if value is not None:paired[c,s]=value
                            w.writerow((src,c,meta[c]["geometry"],s,t,control,value))
                    if len(paired)==len(group)*len(seeds):
                        displays.append(f"{mean(paired.values()):+.2f}")
                    else:displays.append("缺测")
                by_seed={s:mean(paired[c,s] for c in group) for s in seeds}
                by_geo={meta[c]["geometry"]:mean(paired[c,s] for s in seeds) for c in group}
                wins=sum(x>1e-6 for x in by_seed.values())
                ties=sum(abs(x)<=1e-6 for x in by_seed.values())
                lines.append(f"| {src} | {control} | "+" | ".join(displays)+
                             f" | {wins} / {ties} / {len(seeds)-wins-ties} | {min(by_geo.values()):+.2f} |")
                summaries[src][control]=dict(by_seed_at_60=by_seed,by_geometry_at_60=by_geo)
    lines+=["",f"完成 360 个正式终点，独立复算 {audit_count:,} 个已接受时点车队，缺少可行读数 {missing} 个，"
            f"最大净效益重算误差 {max_error:.3g} U。胜种子数指每颗种子先在来源内三个几何取均值后的胜负。"
            "此面板为已测几何上的重复性补充。","",
            "[逐例、逐种子与各时间点](paired_seeds.csv) · [实验登记](../../重复种子与收敛轨迹.md)"]
    with (folder/"cold_start.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(cold_rows[0]))
        writer.writeheader()
        writer.writerows(cold_rows)
    lines+=["","## 首次请求的 60 秒","",
            "从求解脚本入口计时，包含导入、模型加载和编译。数值按来源内三个几何和十个种子等权。",
            "","| 来源 | 对照 | 配对可行数 / 30 | MASKCO 减对照 U | MASKCO / 对照准备均值秒 |",
            "|---|---|---:|---:|---:|"]
    cold_summary={}
    for src in sorted(cases):
        subset={(r["case"],r["seed"],r["method"]):r for r in cold_rows if r["source"]==src}
        cold_summary[src]={}
        for ctrl in methods[1:]:
            pairs=[(subset[c,s,methods[0]],subset[c,s,ctrl]) for c in cases[src] for s in seeds]
            values=[a["U_at_60"]-b["U_at_60"] for a,b in pairs
                    if a["U_at_60"] is not None and b["U_at_60"] is not None]
            value=mean(values) if len(values)==len(pairs) else None
            prep=[mean(group["entry_to_search_s"] for group in column) for column in zip(*pairs)]
            shown="缺测" if value is None else f"{value:+.2f}"
            lines.append(f"| {src} | {ctrl} | {len(values)} / {len(pairs)} | {shown} | {prep[0]:.2f} / {prep[1]:.2f} |")
            cold_summary[src][ctrl]=dict(delta_U=value,paired=len(values),setup_seconds=prep)
    lines+=["","纯距离消融沿用锁版的共同加载入口。它在搜索时做零次神经前向；首次请求用时包含这个实现实际执行的权重准备。独立实现的纯距离求解器有机会省去这部分时间。",
            "","[独立进程逐次读取](cold_start.csv)"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(9.1,3.0),constrained_layout=True)
    for ax,src in zip(axes,sorted(cases)):
        for control,label,color in [(methods[1],"Matched distance","#b45a17"),("ALNS","ALNS","#246b94")]:
            y=[]
            for t in cuts:
                pairs=[(outcomes[c,s,methods[0],t],outcomes[c,s,control,t]) for c in cases[src] for s in seeds]
                dif=[a-b for a,b in pairs if a is not None and b is not None]
                y.append(mean(dif) if len(dif)==len(pairs) else float("nan"))
            ax.plot(cuts,y,marker="o",lw=1.7,ms=4,color=color,label=label)
        ax.axhline(0,lw=.8,color="0.40",ls="--")
        ax.set(title=src,xlabel="Online search time (s)",xticks=cuts)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("MASKCO minus control (U)")
    axes[-1].legend(frameon=False,fontsize=8)
    fig.savefig(folder/"paired_time.png",dpi=240)
    plt.close(fig)
    lines+=["","![四个时点 MASKCO 相对同规则距离与 ALNS 的来源均值](paired_time.png)"]
    (folder/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (folder/"analysis.json").write_text(json.dumps(dict(status="COMPLETE",endpoints=360,audited_snapshots=audit_count,
                missing_snapshots=missing,max_replay_error=max_error,sources=summaries,cold_start=cold_summary),indent=2),encoding="utf-8")
    print("PASS",audit_count,"audited snapshots")


if __name__=="__main__":summarize()
