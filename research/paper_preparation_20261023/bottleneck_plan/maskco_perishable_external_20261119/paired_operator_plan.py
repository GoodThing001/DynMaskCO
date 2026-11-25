"""Prepare the fixed-state comparison and summarize equal-work opportunities by geometry."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"maskco_heatmap_calibration_20261119"))
from phase_plan import read,save
def prepare():
    study=read(HERE/"replicate01.json")
    old=HERE/"results/replicate01"
    if read(old/"independent_audit.json")["endpoints"]!=891:raise ValueError("Archived reference incomplete")
    states={}
    for i,b in enumerate(study["blocks"]):
        if b["seed"]==32001:
            d=read(old/f"out/{i}.json")["arms"]["HYBRID_D"]
            states[b["case"]]={"initial":d["initial_routes"],"terminal":d["routes"]}
    src=read(HERE/"heldout_sources.json");cases=sorted(states)
    if set(cases)!=set(src["cases"]):raise ValueError("Archived cases differ")
    save(HERE/"archived_states.json",states)
    names=[p.name for p in (HERE/"reserved_validation/solver").iterdir()]
    paths={n:"reserved_validation/solver/"+n for n in names}
    paths.update({"runner.py":"development_runner.py","external_common.py":"independent.py"})
    derived=[v["file"] for v in src["derived_instances"].values()]
    files=names+["paired_operator_probe.py","archived_states.json","heldout_sources.json","runner.py","external_common.py"]+derived
    for phase,chosen,groups in (("oppqual01",[cases[0]],2),("opprobe01",cases,8)):
        save(HERE/(phase+".json"),dict(purpose="retrospective archived mechanism",worker="paired_operator_probe.py",file_sources=paths,
             files=files,groups_per_state=groups,block_timeout_s=420,blocks=[dict(case=c) for c in chosen]))
def report():
    base=HERE/"results/opprobe01";q=read(HERE/"opprobe01.json");s=read(base/"status.json")
    if not s["end"] or s["complete"]!=27 or s["failed"]:raise ValueError("Incomplete probe")
    meta=read(HERE/"heldout_sources.json")["cases"]
    per=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));cpu=0.;calls=0;fw=0
    for i,b in enumerate(q["blocks"]):
        rec=read(base/f"out/{i}.json");case=b["case"];cpu+=rec["cpu_s"];fw+=rec["forwards"]
        rows=rec["rows"]
        expected={(t,k,o) for t in ("initial","early","terminal") for k in range(8) for o in ("insertion","order")}
        if set((r["state"],r["group"],r["operator"]) for r in rows)!=expected or len(rows)!=48 or rec["forwards"]!=24:raise ValueError(case)
        for row in rows:
            v=row["options"];calls+=sum(z["requests"] for z in v.values())
            a,d,h=(v[key]["gain"] for key in ("common","distance","heat"))
            out=dict(heat_distance=h-d,mixed=max(a,h),double_distance=max(a,d),mix_delta=max(a,h)-max(a,d))
            item=per[meta[case]["dataset"],row["operator"],row["state"]][meta[case]["geometry"]]
            for k,z in out.items():item[k].append(z)
    def values(key):
        return {geo:{k:mean(v) for k,v in cols.items()} for geo,cols in per[key].items()}
    result={}
    lines=["# 同状态、等候选数的两种掩码重构","","复用已公布的首批丹麦、XTW 和 GH 样本。每个来源分别比较普通重插与给定次序的经济分车。固定一条距离候选序列，第二条使用另一个随机种子，比较真实热图和同种子纯距离。每列收益都相对该次的共同起点。","",
           "| 来源 | 操作 | 起点 | 热图独用减距离独用 U | 距离加热图 U | 两条距离 U | 配对增量 U | 正向几何 |","|---|---|---|---:|---:|---:|---:|---:|"]
    states=("initial","early","terminal");sources=("PDPCDPG-D100","XTW-p100","GH-p100")
    qualified=[]
    for op in ("insertion","order"):
        result[op]={};passed=True
        for src in sources:
            bystate={};gains=defaultdict(list)
            for t in states:
                g=values((src,op,t));means={k:mean(v[k] for v in g.values()) for k in next(iter(g.values()))}
                for gid,z in g.items():gains[gid].append(z["mix_delta"])
                bystate[t]=means
                lines.append("| "+src+" | "+op+" | "+t+" | "+" | ".join(f"{means[k]:+.2f}" for k in ("heat_distance","mixed","double_distance","mix_delta"))+f" | {sum(v['mix_delta']>1e-6 for v in g.values())}/{len(g)} |")
            point=mean(mean(v) for v in gains.values());wins=sum(mean(v)>1e-6 for v in gains.values())
            ok=point>1e-6 and 2*wins>len(gains)
            result[op][src]=dict(mean=point,wins=wins,geometries=len(gains),by_state=bystate,passes=ok)
            passed &= ok
        if passed:qualified.append(op)
    lines+=["",f"满足三个来源均为正、各自严格多数几何获益的操作：{'、'.join(qualified) or '无'}。共计 {fw} 次掩码前向，记录 {calls:,} 次候选路线请求，实测 CPU {cpu:.1f} 秒。前向成本与完整算法运行时间另计，本表检验相同工作量下热图是否提供额外解。"]
    (base/"analysis.json").write_text(json.dumps(dict(operators=result,qualified=qualified,forward_calls=fw,route_requests=calls,cpu_s=cpu),indent=2))
    (base/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":prepare() if sys.argv[1]=="plan" else report()
