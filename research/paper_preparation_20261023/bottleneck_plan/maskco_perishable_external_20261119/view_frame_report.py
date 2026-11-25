"""Geometry-equal effects of distance-preserving MASKCO input orientations."""
from collections import defaultdict
from statistics import mean
import json
from directed_decoder_fit import HERE,SOURCES,read
OUT=HERE/"results/viewprobe01"
CAND=("cycle","two","eight")

def summarize():
    status=read(OUT/"status.json");cfg=read(HERE/"viewprobe01.json")
    if status["complete"]!=27 or status["failed"] or not status["end"]:raise ValueError("Incomplete orientation run")
    meta=read(HERE/"heldout_sources.json")["cases"];cases={};cost=defaultdict(float);overlap={};by_state=defaultdict(list)
    allarms=("distance","v0","cycle","two","eight","shuffled","uniform","oracle")
    for ix,b in enumerate(cfg["blocks"]):
        c=b["case"];res=read(OUT/f"out/{ix}.json");rows=res["rows"]
        if res["case"]!=c or len(rows)!=4 or res["forwards"]!=32 or res["identity_error"]>.002:raise ValueError("Missing matched view")
        values=defaultdict(list)
        for r in rows:
            a=r["arms"];u={n:a[n]["gain"] for n in ("v0","distance","two","eight","shuffled")}
            if min(v["gain"] for v in a.values())< -1.e-6 or len(a)!=12:raise ValueError("Invalid incumbent")
            actual=[a["v"+str(j)]["gain"] for j in range(8)];u.update(cycle=a[r["cycle"]]["gain"],uniform=mean(actual),oracle=max(actual))
            for n,v in u.items():values[n].append(v)
            by_state[c,r["state"]].append(u)
            cost["native_route_requests"]+=sum(v["requests"] for v in a.values())
        cases[c]={n:mean(v) for n,v in values.items()};overlap[c]=mean(r["overlap"] for r in rows)
        for key in ("forwards","identity_checks","forward_cpu_s","cpu_s"):cost[key]+=res[key]
    panels={}
    for src in SOURCES:
        geo=defaultdict(list)
        for c in cases:
            if meta[c]["dataset"]==src:geo[meta[c]["geometry"]].append(c)
        def agg(fn):return mean(mean(fn(c) for c in group) for group in geo.values())
        details={g:{n:mean(cases[c][n] for c in group) for n in allarms} for g,group in geo.items()}
        m={n:agg(lambda c:cases[c][n]) for n in allarms}
        states={s:{n:agg(lambda c:mean(row[n] for row in by_state[c,s])) for n in allarms} for s in ("initial","terminal")}
        panels[src]=dict(means=m,by_state=states,geometries=details,overlap=agg(overlap.get),
             wins={a:sum(v[a]>v["distance"]+1.e-6 for v in details.values()) for a in CAND})
    passed={}
    for a in CAND:
        passed[a]=all(v["means"][a]>v["means"]["distance"]+1.e-6 and 2*v["wins"][a]>len(v["geometries"])
                      and v["means"][a]>v["means"]["shuffled"]+1.e-6 for v in panels.values()) and mean(v["means"][a]-v["means"]["v0"] for v in panels.values())>1.e-6
    winner=next((a for a in CAND if passed[a]),None)
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=panels,passed=passed,selected=winner,cost=cost),indent=2),encoding="utf8")
    lines=["# 等距视图与实际冷链重构","","27 个开发案例，每例两个起点、每起点两个掩码。下表为相对同规则公里距离排序的净效益增量 U，以及变换视图后被遮客户前八邻居与原向的平均重合比例。单视图轮换的选择先于求解固定。","",
        "| 来源 | 原向 | 单视图轮换 | 两视图均值 | 八视图均值 | 均值置乱 | 事后八选一 | 前八重合 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for src in SOURCES:
        v=panels[src];m=v["means"];lines.append("| "+src+" | "+" | ".join(f"{m[a]-m['distance']:+.2f}" for a in ("v0","cycle","two","eight","shuffled","oracle"))+f" | {v['overlap']:.1%} |")
    lines+=["",("进入 60 秒计时开发的候选为 "+winner+"。" if winner else "三项预设候选均未达到 60 秒开发门槛。"),
        "",f"固定 {int(cost['forwards'])} 次视图前向、{int(cost['identity_checks'])} 次原模型一致性比对；候选路线请求 {int(cost['native_route_requests']):,} 次，诊断累计 CPU {cost['cpu_s']:.1f} 秒，其中视图前向 {cost['forward_cpu_s']:.1f} 秒。","",
        "[逐起点、几何和晋级判定](analysis.json) · [方法与门槛](../../等距视图的掩码重构.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":summarize()
