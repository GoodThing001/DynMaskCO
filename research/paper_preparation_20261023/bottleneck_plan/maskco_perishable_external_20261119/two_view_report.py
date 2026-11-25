"""Compare two actual repaired fleets with a second independent search seed."""
from collections import defaultdict
from statistics import mean
import json
from directed_decoder_fit import HERE,SOURCES,read
OUT=HERE/"results/viewpair01"
PAIRS={"DD":("distance","distance"),"HH":("v0","v0"),"DH":("distance","v0"),"paired":("v0","v3"),"placebo":("v0","shuffled")}

def main():
    status=read(OUT/"status.json");cfg=read(HERE/"viewpair01.json");meta=read(HERE/"heldout_sources.json")["cases"]
    if status["complete"]!=27 or status["failed"] or not status["end"]:raise ValueError("Incomplete second branch")
    cases={};out=[];cost=dict(forwards=0,requests=0,cpu_s=0.)
    for ix,b in enumerate(cfg["blocks"]):
        c=b["case"];prior=read(HERE/f"results/viewprobe01/out/{ix}.json");new=read(OUT/f"out/{ix}.json")
        if prior["case"]!=c or new["case"]!=c or len(prior["rows"])!=4 or new["forwards"]!=8:raise ValueError("Block identity")
        cells=defaultdict(list);cost["forwards"]+=new["forwards"];cost["cpu_s"]+=new["cpu_s"]
        for a,z in zip(prior["rows"],new["rows"],strict=True):
            if any(a[k]!=z[k] for k in ("state","rep","group","mask")) or abs(a["original_U"]-z["original_U"])>1.e-6:raise ValueError("Unpaired initial state")
            gains={p:max(a["arms"][u]["gain"],z["arms"][v]["gain"]) for p,(u,v) in PAIRS.items()}
            for k,v in gains.items():cells[k].append(v)
            cost["requests"]+=sum(v["requests"] for v in z["arms"].values())
            out.append(dict(case=c,state=z["state"],rep=z["rep"],gains=gains))
        cases[c]={p:mean(g) for p,g in cells.items()}
    sources={};passed=True;lines=["# 两次等预算完整重构","","两次独立搜索种子，每次均运行 32 次插回和 24 个序列分车。各组合取两个候选车队中收益较高者。下表为原向加反向热图减去对应参照的可行净效益，单位 U。","",
          "| 来源 | 减双距离 | 减双原向 | 减距离加原向 | 减原向加置乱 | 优于双距离几何 | 去掉最大正向几何后 |","|---|---:|---:|---:|---:|---:|---:|"]
    for src in SOURCES:
        gs=defaultdict(list)
        for c,m in cases.items():
            if meta[c]["dataset"]==src:gs[meta[c]["geometry"]].append(m)
        geos={g:{k:mean(v[k] for v in vs) for k in PAIRS} for g,vs in gs.items()}
        delta={k:mean(v["paired"]-v[k] for v in geos.values()) for k in PAIRS}
        d=sorted(v["paired"]-v["DD"] for v in geos.values());wins=sum(z>1.e-6 for z in d);omit=mean(d[:-1])
        sources[src]=dict(geometries=geos,delta=delta,wins=wins,leave_best_out=omit)
        passed &= delta["DD"]>1.e-6 and 2*wins>len(d) and delta["placebo"]>1.e-6 and omit>1.e-6
        lines.append(f"| {src} | "+" | ".join(f"{delta[k]:+.2f}" for k in ("DD","HH","DH","placebo"))+f" | {wins}/{len(d)} | {omit:+.2f} |")
    passed=bool(passed and all(mean(s["delta"][k] for s in sources.values())>1.e-6 for k in ("HH","DH")))
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=sources,passed=passed,rows=out,cost=cost),indent=2),encoding="utf8")
    lines+=["",("双视图车队候选通过 60 秒开发门槛。" if passed else "双视图车队候选未达到 60 秒开发门槛。"),
       "",f"第二次搜索新增 {cost['forwards']} 次前向、{cost['requests']:,} 次候选路线请求与 {cost['cpu_s']:.1f} CPU 秒。第一次搜索沿用已审计结果。","",
       "[逐几何与分叉结果](analysis.json) · [方案](../../等预算双车队候选.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":main()
