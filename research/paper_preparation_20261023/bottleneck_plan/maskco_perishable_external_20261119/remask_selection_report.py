"""Separate customer nomination from slot ranking using fixed-distance second-pass repairs."""
from collections import Counter, defaultdict
from statistics import mean
import json
from directed_decoder_fit import HERE, SOURCES, read

OUT=HERE/"results/selectprobe01"


def contrasts(rows,first,other):
    grouped=defaultdict(list)
    for r in rows:
        v=r["arms"][first]["second"]
        grouped[r["geometry"]].append(v["neural"]["best"]-v[other]["best"])
    return {g:mean(v) for g,v in grouped.items()}


def report():
    cfg=read(HERE/"selectprobe01.json");end=read(OUT/"status.json")
    meta=read(HERE/"heldout_sources.json")["cases"]
    if end["complete"]!=27 or end["failed"] or not end["end"]:raise ValueError("Incomplete nomination study")
    rows=[];cost=Counter()
    for i,b in enumerate(cfg["blocks"]):
        out=read(OUT/f"out/{i}.json");case=b["case"]
        if (out["case"],out["forwards"],out["audited_candidates"])!=(case,6,60):
            raise ValueError("Wrong paired work")
        expected={(state,seed) for state in cfg["states"] for seed in cfg["seeds"]}
        if {(v["state"],v["seed"]) for v in out["rows"]}!=expected or len(out["rows"])!=len(expected):
            raise ValueError("Missing case, seed or start")
        for key in ("cpu_s","wall_s","forwards","audited_candidates"):cost[key]+=out[key]
        for z in out["rows"]:
            for f in z["arms"].values():
                for v in f["second"].values():
                    if (len(set(v["mask"]))!=20 or set(v["mask"])-set(z["mask"])
                        or abs(v["best"]-max(z["original_U"],f["first_U"],v["U"]))>1.e-6):
                        raise ValueError("Unmatched eligibility or fleet selection")
            rows.append(dict(**z,**meta[case],case=case))
    sources={}
    for source in SOURCES:
        group=[r for r in rows if r["dataset"]==source]
        def effect(q=group,first="distance",other="distance"):
            return mean(contrasts(q,first,other).values())
        geo=contrasts(group,"distance","distance")
        seed_effects={s:effect([r for r in group if r["seed"]==s]) for s in cfg["seeds"]}
        sources[source]=dict(primary=mean(geo.values()),wins=sum(x>1.e-6 for x in geo.values()),by_geometry=geo,
            by_seed=seed_effects,by_state={s:effect([r for r in group if r["state"]==s]) for s in cfg["states"]},
            leave_best_out=(sum(geo.values())-max(geo.values()))/(len(geo)-1),
            shuffled=effect(other="shuffled"),random=effect(other="random"),
            neural_first_nomination_effect=effect(first="neural"))
        stages={}
        for state in cfg["states"]:
            selected=[r for r in group if r["state"]==state]
            opportunity=defaultdict(list)
            flat=Counter()
            for r in selected:
                f=r["arms"]["distance"]
                arms=f["second"]
                ceiling=max(v["best"] for v in arms.values())
                opportunity[r["geometry"]].append(ceiling-arms["distance"]["best"])
                flat["no_second_improvement"]+=int(ceiling<=max(r["original_U"],f["first_U"])+1.e-6)
                flat["observations"]+=1
                flat["shared_customers"]+=len(set(arms["neural"]["mask"])&set(arms["distance"]["mask"]))
            stages[state]=dict(**flat,oracle_over_distance=mean(mean(v) for v in opportunity.values()))
        sources[source]["observed_opportunity"]=stages
    passed=all(s["primary"]>1.e-6 and sum(x>1.e-6 for x in s["by_seed"].values())>=2 and 2*s["wins"]>len(s["by_geometry"]) for s in sources.values())
    passed=passed and all(mean(s[a] for s in sources.values())>1.e-6 for a in ("shuffled","random"))
    (OUT/"analysis.json").write_text(json.dumps(dict(passed=passed,sources=sources,costs=cost),indent=2),encoding="utf8")
    lines=["# 插位全用距离时，MASKCO 单独选客的贡献","",
        "第一步为相同的距离修复。第二步的所有臂使用同一纯距离插位，只有重新遮蔽的客户不同。数值为候选减对照的几何等权净效益 U。","",
        "| 来源 | 热图选客减距离选客 | 改善几何 | 53491 | 53492 | 53493 | 减置乱 | 减随机 | 去掉最大贡献几何 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n,s in sources.items():
        vals=[*s["by_seed"].values(),s["shuffled"],s["random"],s["leave_best_out"]]
        lines.append(f"| {n} | {s['primary']:+.2f} | {s['wins']}/{len(s['by_geometry'])} | "+" | ".join(f"{v:+.2f}" for v in vals)+" |")
    lines+=["","| 来源 / 起点 | 四个实际名单事后择优减距离 U | 第二轮均未能超过原起点及首轮最好值 | 两种 20 客名单平均重合 |","|---|---:|---:|---:|"]
    for n,s in sources.items():
        for state,z in s["observed_opportunity"].items():
            stage="共同初解" if state=="initial" else "距离第 64 步"
            count=z["observations"]
            lines.append(f"| {n} / {stage} | {z['oracle_over_distance']:+.2f} | {z['no_second_improvement']}/{count} | {z['shared_customers']/count:.1f}/20 |")
    lines+=["","若首轮改用热图插位，第二轮热图选客相对距离选客的增量（丹麦、XTW、GH）为 "
            +" / ".join(f"{s['neural_first_nomination_effect']:+.2f}" for s in sources.values())+" U。",
            "",f"27 例、162 个配对上下文、{cost['forwards']} 次前向、{cost['audited_candidates']:,} 次完整车队独立复算，累计 CPU {cost['cpu_s']:.1f} 秒。"
            +"单核 60 秒门槛"+("通过。" if passed else "未通过。"),"",
            "[原方案](../../固定距离插位的复遮选择.md) · [各起点、几何和种子](analysis.json)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))


if __name__=="__main__":
    report()
