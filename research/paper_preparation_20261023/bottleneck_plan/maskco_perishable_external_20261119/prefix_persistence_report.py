"""Quantify whether masked early gains persist through identical pure-distance moves."""
from collections import defaultdict
from statistics import mean
import json
from directed_decoder_fit import HERE,SOURCES,read
OUT=HERE/"results/carryprobe01"
STEPS=(0,64,256)

def main():
    cfg=read(HERE/"carryprobe01.json");status=read(OUT/"status.json");meta=read(HERE/"heldout_sources.json")["cases"]
    if status["complete"]!=27 or status["failed"] or not status["end"]:raise ValueError("Incomplete common-tail experiment")
    records=[];cost=defaultdict(float);seeds=cfg["seeds"]
    for ix,b in enumerate(cfg["blocks"]):
        c=b["case"];r=read(OUT/f"out/{ix}.json")
        if c!=r["case"] or r["strict_suffix_calls"]!=2048 or r["forwards"]!=16 or [s["seed"] for s in r["seeds"]]!=seeds:raise ValueError("Wrong paired work")
        for n in ("forwards","strict_suffix_calls","prefix_route_requests","cpu_s"):cost[n]+=r[n]
        for z in r["seeds"]:
            for a,v in z["arms"].items():
                ck=v["checkpoints"]
                if sorted(map(int,ck))!=list(STEPS) or any(ck[str(s)]["U"]<ck[str(t)]["U"]-1.e-6 for s,t in zip(STEPS[1:],STEPS)):raise ValueError("Non-monotone accepted fleet")
                for j in STEPS:
                    t=ck[str(j)]
                    if sorted(i for route in t["routes"] for i in route)!=list(range(1,101)):raise ValueError("Customer identity")
                    records.append(dict(case=c,src=meta[c]["dataset"],geo=meta[c]["geometry"],seed=z["seed"],arm=a,step=j,gain=t["U"]-z["orig_U"]))
    arms=("distance","original","eight","shuffled");sources={}
    passed=True;lines=["# 神经起点经过共同距离重构后还能保留多少收益","","每个开发案例使用两个新种子。下表为八视图 MASKCO 起点减纯距离起点的净效益，单位 U；起点之后各臂严格使用相同的有序移除名单和接受规则。","",
        "| 来源 | 起点 | 第 64 步 | 第 256 步 | 终点改善几何 | 51291 终点 | 51292 终点 |","|---|---:|---:|---:|---:|---:|---:|"]
    for src in SOURCES:
        subset=[r for r in records if r["src"]==src]
        def metrics(rows):
            geo={}
            for g in {r["geo"] for r in rows}:
                grp=[r for r in rows if r["geo"]==g]
                geo[g]={a:mean(v["gain"] for v in grp if v["arm"]==a) for a in arms}
            return geo,{a:mean(v[a] for v in geo.values()) for a in arms}
        stages={step:metrics([r for r in subset if r["step"]==step]) for step in STEPS}
        fx=lambda v:v["eight"]-v["distance"]
        means=[fx(stages[step][1]) for step in STEPS];g,last=stages[256]
        seedvals=[fx(metrics([r for r in subset if r["step"]==256 and r["seed"]==seed])[1]) for seed in seeds]
        wins=sum(fx(v)>1.e-6 for v in g.values());passed&=means[-1]>1.e-6 and min(seedvals)>1.e-6 and 2*wins>len(g)
        sources[src]=dict(stages={k:v[1] for k,v in stages.items()},final_by_geometry=g,seed_differences=seedvals,wins=wins)
        lines.append("| "+src+" | "+" | ".join(f"{v:+.2f}" for v in means)+f" | {wins}/{len(g)} | "+" | ".join(f"{v:+.2f}" for v in seedvals)+" |")
    passed=bool(passed and all(mean(s["stages"][256]["eight"]-s["stages"][256][a] for s in sources.values())>1.e-6 for a in ("original","shuffled")))
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=sources,cost=cost,passed=passed,seeds=seeds),indent=2),encoding="utf8")
    lines+=["",("一次八视图热图起点通过 60 秒开发门槛。" if passed else "一次八视图热图起点未达到 60 秒开发门槛。"),
      f"累计 {int(cost['forwards'])} 次前向、{int(cost['strict_suffix_calls']):,} 次后续距离重构、{int(cost['prefix_route_requests']):,} 次起点候选路线请求；CPU {cost['cpu_s']:.1f} 秒。","",
      "[逐种子、几何和参照](analysis.json) · [固定后续轨迹的检验](../../掩码重构起点的收益保留.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":main()
