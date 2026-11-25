"""Test conditional neural benefit when the declared mask equals the 20 actual movers."""
from collections import defaultdict,Counter
import json
from statistics import mean
from directed_decoder_fit import HERE, SOURCES, read
from mask_context_size import ARMS

OUT=HERE/"results/contextprobe01"


def aggregate(rows,measure):
    groups=defaultdict(list)
    for r in rows:groups[r["geometry"]].append(measure(r))
    return {g:mean(v) for g,v in groups.items()}


def report():
    cfg=read(HERE/"contextprobe01.json");end=read(OUT/"status.json");meta=read(HERE/"heldout_sources.json")["cases"]
    if end["failed"] or end["complete"]!=27 or not end["end"]:raise ValueError("Context comparison incomplete")
    rows=[];cost=Counter()
    for i,b in enumerate(cfg["blocks"]):
        out=read(OUT/f"out/{i}.json")
        if (out["case"],out["forwards"],out["audited_candidates"])!=(b["case"],18,60):raise ValueError("Different paired work")
        if {(r["seed"],r["state"]) for r in out["rows"]}!={(s,t) for s in cfg["seeds"] for t in cfg["states"]} or len(out["rows"])!=6:
            raise ValueError("Nonmatching cases or seeds")
        for n in ("forwards","audited_candidates","refinement_route_requests","cpu_s","wall_s"):cost[n]+=out[n]
        for r in out["rows"]:
            hidden=set(r["mask"]);pa=set(r["padding_a"]);pb=set(r["padding_b"])
            if len(hidden|pa|pb)!=60 or not all(len(v)==20 for v in (hidden,pa,pb)) or set(r["arms"])!=set(ARMS):
                raise ValueError("Movers, bystanders or controls differ")
            for arm in r["arms"].values():
                if sorted(i for route in arm["winning_routes"] for i in route)!=list(range(1,101)) or abs(arm["best_U"]-max(r["original_U"],arm["first_U"],arm["refined_U"]))>1.e-6:
                    raise ValueError("Full-fleet identity or selection differs")
            rows.append(dict(**r,**meta[b["case"]],case=b["case"]))
    sources={}
    def diff(r,control="distance",field="best_U"):
        def value(a):return max(r["original_U"],r["arms"][a][field])
        return value("true20")-(mean(value(a) for a in ("pad40a","pad40b")) if control=="padding_mean" else value(control))
    for src in SOURCES:
        grp=[r for r in rows if r["dataset"]==src]
        geo=aggregate(grp,diff)
        def measure(sub,fun):return mean(aggregate(sub,fun).values())
        seed={s:measure([r for r in grp if r["seed"]==s],diff) for s in cfg["seeds"]}
        sources[src]=dict(mean=mean(geo.values()),geometries=geo,wins=sum(x>1.e-6 for x in geo.values()),by_seed=seed,
           states={s:measure([r for r in grp if r["state"]==s],diff) for s in cfg["states"]},
           first=measure(grp,lambda r:diff(r,field="first_U")),padding=measure(grp,lambda r:diff(r,"padding_mean")),
           pad_a=measure(grp,lambda r:diff(r,"pad40a")),pad_b=measure(grp,lambda r:diff(r,"pad40b")),
           placebo=measure(grp,lambda r:diff(r,"shuffle20")),overlap=measure(grp,lambda r:mean(r["top8_overlap"].values())),
           timestep20=measure(grp,lambda r:r["context"]["true20"]["timestep"]),
           timestep40=measure(grp,lambda r:mean(r["context"][a]["timestep"] for a in ("pad40a","pad40b"))))
    passed=all(s["mean"]>1.e-6 and 2*s["wins"]>len(s["geometries"]) and sum(v>1.e-6 for v in s["by_seed"].values())>=2 for s in sources.values())
    passed=passed and all(mean(s[a] for s in sources.values())>1.e-6 for a in ("padding","placebo"))
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=sources,passed=passed,costs=cost),indent=2),encoding="utf8")
    lines=["# 同一组 20 客实际缺口的模型可见范围","",
        "各效果均为几何等权净效益 U。实际移动客户、起点、32 次后续重构与接受规则完全配对。填充列以两组无交集的额外 20 客为平均参照。","",
        "| 来源 | 第一次重构减距离 | 32 次后减距离 | 改善几何 | 只遮 20 减额外填充 | 只遮 20 减置乱 | 前八重合 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for n,s in sources.items():
        lines.append(f"| {n} | {s['first']:+.2f} | {s['mean']:+.2f} | {s['wins']}/{len(s['geometries'])} | "
            +f"{s['padding']:+.2f} | {s['placebo']:+.2f} | {s['overlap']:.1%} |")
    lines+=["","| 来源 | 55231 | 55232 | 55233 | 共同初解 | 距离第 64 步 |","|---|---:|---:|---:|---:|---:|"]
    for n,s in sources.items():
        lines.append("| "+n+" | "+" | ".join(f"{a:+.2f}" for a in [*s["by_seed"].values(),*s["states"].values()])+" |")
    lines+=["",f"27 例、162 个上下文，{cost['forwards']} 次前向、{cost['audited_candidates']:,} 次独立车队复算。"
       f"后续固定步重构共 {int(cost['refinement_route_requests']):,} 次路线请求，累计 CPU {cost['cpu_s']:.1f} 秒。单核 60 秒开发门槛"
       +("通过。" if passed else "未通过。"),"", "[原方案](../../实际缺口与预测掩码一致性.md) · [逐几何及可见邻接](analysis.json)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))


if __name__=="__main__":
    report()
