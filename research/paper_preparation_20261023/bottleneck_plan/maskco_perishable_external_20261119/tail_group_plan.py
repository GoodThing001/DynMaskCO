"""Plan and summarize the masked route-pair selection diagnostic."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding="utf8"))

def prepare():
    for ph,parent in (("tailqual01","loadqual01"),("tailprobe01","loadprobe01")):
        plan=read(HERE/(parent+".json"))
        plan.update(worker="tail_group_probe.py",purpose="retrospective masked route-pair selection")
        plan["files"]=[p if p!="load_slot_probe.py" else "tail_group_probe.py" for p in plan["files"]]
        (HERE/(ph+".json")).write_text(json.dumps(plan,indent=2))

def report():
    plan=read(HERE/"tailprobe01.json");base=HERE/"results/tailprobe01"
    status=read(base/"status.json");meta=read(HERE/"heldout_sources.json")["cases"]
    if status["failed"] or status["complete"]!=27 or not status["end"]:raise ValueError("Incomplete experiment")
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    cpu=0.;cuts=requests=0
    for i,b in enumerate(plan["blocks"]):
        r=read(base/f"out/{i}.json");c=b["case"]
        if r["case"]!=c or r["forwards"]!=12 or len(r["rows"])!=12:raise ValueError(c)
        cpu+=r["cpu_s"];requests+=r["requests"]
        for row in r["rows"]:
            k=meta[c]["dataset"],row["state"]
            cuts+=row["cuts"]
            for m,val in row["gains"].items():groups[k][meta[c]["geometry"]][m].append(val)
    lines=["# MASKCO 选择整条待重构路线","","表中是每个掩码获得的最佳可行净效益提升。距离、真热图和置乱热图各可选择四个路线对，遍历列给出该批次可达到的上界。","",
           "| 来源 | 起点 | 距离 U | 真实 U | 置乱 U | 遍历 U | 真减距离 U | 改善几何 |","|---|---|---:|---:|---:|---:|---:|---:|"]
    results={};passed=True
    for source in ("PDPCDPG-D100","XTW-p100","GH-p100"):
        results[source]={}
        for state in ("initial","terminal"):
            g={gid:{m:mean(v) for m,v in d.items()} for gid,d in groups[source,state].items()}
            avg={m:mean(v[m] for v in g.values()) for m in ("distance","real","shuffled","oracle")}
            wins=sum(v["real"]>v["distance"]+1e-7 for v in g.values())
            ok=avg["real"]>max(avg["distance"],avg["shuffled"])+1e-7 and 2*wins>len(g)
            if state=="terminal":passed &= ok
            results[source][state]=dict(means=avg,wins=wins,geometries=len(g),by_geometry=g,passes=ok)
            lines.append("| "+source+" | "+state+" | "+" | ".join(f"{w:.3f}" for w in avg.values())+f" | {avg['real']-avg['distance']:+.3f} | {wins}/{len(g)} |")
    lines+=["",f"324 次前向，{cuts:,} 个满足载重条件的切口，{requests:,} 次路线请求。采集 CPU {cpu:.1f} 秒。强距离终态的三来源门槛"+("通过。" if passed else "未通过。"),"","[预设试验](../../路线对选择诊断.md)"]
    (base/"analysis.json").write_text(json.dumps(dict(sources=results,passed=passed,cuts=cuts,requests=requests,cpu_s=cpu),indent=2))
    (base/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))

if __name__=="__main__":prepare() if sys.argv[1]=="plan" else report()
