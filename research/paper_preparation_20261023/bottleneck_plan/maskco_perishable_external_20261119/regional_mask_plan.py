"""Register and evaluate the cross-mask regional reconstruction experiment."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import json
import sys

HERE=Path(__file__).resolve().parent
SOURCES=("PDPCDPG-D100","XTW-p100","GH-p100")
ARMS=("distance","real","shuffled","single")


def read(path):return json.loads(path.read_text(encoding="utf8"))


def prepare():
    for phase,parent,regions,orders in (("regionqual01","tailqual01",1,12),("regionprobe01","tailprobe01",2,96)):
        cfg=read(HERE/(parent+".json"))
        cfg.update(purpose="retrospective cross-mask route-region reconstruction",worker="regional_mask_probe.py",
                   regions_per_state=regions,orders=orders,block_timeout_s=420)
        cfg["files"]=[n if n!="tail_group_probe.py" else cfg["worker"] for n in cfg["files"]]
        del cfg["masks_per_state"]
        (HERE/(phase+".json")).write_text(json.dumps(cfg,indent=2),encoding="utf8")


def report():
    cfg=read(HERE/"regionprobe01.json");base=HERE/"results/regionprobe01"
    status=read(base/"status.json");metadata=read(HERE/"heldout_sources.json")["cases"]
    if status["complete"]!=27 or status["failed"] or not status["end"]:raise ValueError("Incomplete run")
    grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    fw=orders=requests=0;cpu=0.;minsize=100;maxsize=0
    for ix,b in enumerate(cfg["blocks"]):
        r=read(base/f"out/{ix}.json");case=b["case"]
        if (r["case"]!=case or r["forwards"]!=12 or len(r["rows"])!=4 or
          {(x["state"],x["region"]) for x in r["rows"]}!={(s,i) for s in ("initial","terminal") for i in (0,1)}):
            raise ValueError(case)
        fw+=r["forwards"]; cpu+=r["cpu_s"]
        for row in r["rows"]:
            n=row["customers"];minsize=min(minsize,n);maxsize=max(maxsize,n)
            if not 40<n<=60 or row["covered"]!=n*n:raise ValueError("Insufficient region coverage")
            g=grouped[metadata[case]["dataset"],row["state"]][metadata[case]["geometry"]]
            for arm in ARMS:
                x=row["arms"][arm]
                if x["orders"]!=96:raise ValueError("Unequal search budget")
                orders+=x["orders"];requests+=x["requests"]
                g[arm].append(x["gain"])
                g[arm+"_fleet"].append(float(x["fleet_delta"]!=0));g[arm+"_offers"].append(x["offers"])
    results={}
    lines=["# 重叠掩码的跨路线区域重构","","下表为一次区域重构相对共同起点的净收益 U。每个区域内各排序均尝试 96 条访问序列，对每条序列重新求最优分车。","",
       "| 来源 | 起点 | 纯距离 | 真三图 | 置乱三图 | 单张真图 | 真减距离 | 优于距离几何 | 真图改变车数 | 真图平均刷新次数 |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    mixture_ok=True
    for s in SOURCES:
        results[s]={};mix=defaultdict(list)
        for state in ("initial","terminal"):
            grouped_geo={g:{a:mean(z) for a,z in v.items()} for g,v in grouped[s,state].items()}
            means={m:mean(v[m] for v in grouped_geo.values()) for m in next(iter(grouped_geo.values()))}
            wins=sum(v["real"]>v["distance"]+1.e-7 for v in grouped_geo.values())
            for g,v in grouped_geo.items():mix[g].append(v["real"]-v["distance"])
            results[s][state]=dict(means=means,geometries=grouped_geo,wins=wins)
            lines.append("| "+s+" | "+state+" | "+" | ".join(f"{means[a]:.3f}" for a in ARMS)+f" | {means['real']-means['distance']:+.3f} | {wins}/{len(grouped_geo)} | {means['real_fleet']:.1%} | {means['real_offers']:.2f} |")
        advantages=[mean(v) for v in mix.values()]
        ok=mean(advantages)>1.e-7 and 2*sum(v>1.e-7 for v in advantages)>len(mix)
        mixture_ok &= ok
        results[s]["balanced_advantage"]=mean(advantages)
        results[s]["balanced_wins"]=sum(v>1.e-7 for v in advantages)
    term=[results[s]["terminal"]["means"] for s in SOURCES]
    terminal_d=mean(v["real"]-v["distance"] for v in term)
    terminal_null=mean(v["real"]-v["shuffled"] for v in term)
    advances=bool(mixture_ok and min(terminal_d,terminal_null)>1.e-7 and sum(v["real"]>v["single"]+1.e-7 for v in term)>=2)
    lines+=["",f"强终态三来源等权，三图相对距离 {terminal_d:+.3f} U，相对置乱 {terminal_null:+.3f} U。晋级 60 秒同批开发的门槛"+("通过。" if advances else "未通过。"),"",
            f"完整 108 个区域，区域大小 {minsize}–{maxsize} 客。共 {fw} 次前向、{orders:,} 个定序分车、{requests:,} 次路线请求，采集 CPU {cpu:.1f} 秒。","",
            "[预设方案](../../重叠掩码区域重构.md) · [归档终态之间的改进规模](../../终态改进结构.md)"]
    (base/"analysis.json").write_text(json.dumps(dict(sources=results,advance=advances,terminal_vs_distance=terminal_d,terminal_vs_shuffled=terminal_null,
       orders=orders,requests=requests,forwards=fw,cpu_s=cpu),indent=2))
    (base/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))


if __name__=="__main__":prepare() if sys.argv[1]=="plan" else report()
