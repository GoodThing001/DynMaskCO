"""Compare second-pass reconstruction at fixed work and identify its marginal response."""
from collections import Counter
import json
from statistics import mean
from directed_decoder_fit import HERE, SOURCES, read
from iterative_remask import ARMS

OUT=HERE/"results/remaskprobe01"


def report():
    plan=read(HERE/"remaskprobe01.json")
    end=read(OUT/"status.json")
    meta=read(HERE/"heldout_sources.json")["cases"]
    if end["failed"] or end["complete"]!=len(plan["blocks"]) or not end["end"]:
        raise ValueError("Incomplete paired experiment")
    rows=[]
    costs=Counter()
    for ix,b in enumerate(plan["blocks"]):
        r=read(OUT/f"out/{ix}.json")
        if b["case"]!=r["case"] or (r["forwards"],r["repairs"],r["audited_candidates"])!=(12,28,28):
            raise ValueError("Unexpected two-stage search work")
        for k in ("forwards","repairs","audited_candidates","cpu_s","wall_s"):
            costs[k]+=r[k]
        for z in r["rows"]:
            valid=set(z["mask"])
            if (set(z["arms"])!=set(ARMS) or len(valid)!=40 or valid & set(z["padding"]) or
                    any(len(set(z[k]))!=20 or set(z[k])-valid for k in ("distance_mask","neural_mask","random_mask"))):
                raise ValueError("Mask or controls differ")
            for a in z["arms"].values():
                if abs(a["best"]-max(z["original"],a["first"],a["second"]))>1.e-6:
                    raise ValueError("Best feasible fleet differs")
            rows.append(dict(**z,**meta[r["case"]],case=r["case"]))
    sources={}
    for src in SOURCES:
        group=[r for r in rows if r["dataset"]==src]
        def geo_effect(arm,other,subset,measure="best"):
            def value(r,name):
                score=r["arms"][name][measure]
                return max(r["original"],score) if measure=="first" else score
            return {g:mean(value(r,arm)-value(r,other) for r in subset if r["geometry"]==g) for g in sorted({r["geometry"] for r in subset})}
        def delta(other="distance",subset=group,measure="best"):
            return mean(geo_effect("updated",other,subset,measure).values())
        geo=geo_effect("updated","distance",group)
        src_result=dict(geometries=geo,wins=sum(x>1.e-6 for x in geo.values()),mean=mean(geo.values()),
            by_seed={s:delta(subset=[r for r in group if r["seed"]==s]) for s in plan["seeds"]},
            by_state={k:delta(subset=[r for r in group if r["state"]==k]) for k in plan["states"]},
            first=delta(measure="first"),controls={a:delta(a) for a in ("stale","random","shuffled")})
        sources[src]=src_result
    passed=all(s["mean"]>1.e-6 and min(s["by_seed"].values())>1.e-6 and 2*s["wins"]>len(s["geometries"]) for s in sources.values())
    passed=passed and all(mean(s["controls"][a] for s in sources.values())>1.e-6 for a in ("stale","random","shuffled"))
    if len(rows)!=108:raise ValueError("Expected two seeds and two starting states on 27 cases")
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=sources,costs=costs,passed=passed),indent=2),encoding="utf8")
    lines=["# 低置信客户的第二轮神经重构","","数值均为开发几何等权净效益差 U。第一轮和第二轮均保留原起点。旧热图与更新热图的第二轮操作完全配对。","",
        "| 来源 | 第一轮热图减距离 | 两轮热图减距离 | 改善几何 | 53101 | 53102 | 更新减旧图 | 低置信减随机 | 真图减置乱 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n,s in sources.items():
        lines.append(f"| {n} | {s['first']:+.2f} | {s['mean']:+.2f} | {s['wins']}/{len(s['geometries'])} | "
            +" | ".join(f"{x:+.2f}" for x in [*s["by_seed"].values(),*s["controls"].values()])+" |")
    lines+=["",f"共 {len(rows)} 个上下文，{costs['forwards']} 次前向，{costs['audited_candidates']} 个生成车队独立复算。"
        f"累计 CPU {costs['cpu_s']:.1f} 秒。单核 60 秒开发门槛"+("通过。" if passed else "未通过。"),"",
        "[逐起点及几何](analysis.json) · [实验方案](../../低置信客户的二次掩码.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))


if __name__=="__main__":
    report()
