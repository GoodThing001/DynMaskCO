"""Measure multi-step masked reconstruction on archived, geometry-labelled development states."""
from collections import defaultdict
from functools import lru_cache
import json
import math
from statistics import mean
import time
import numpy as np

from directed_decoder_fit import HERE, SOURCES, read
from prepare_panel import core, independent_audit
from remaining_edges import beam_repair, reciprocal_weights
from view_frame_probe import shuffle_probabilities

OUT=HERE/"results/remaining_edges01"
ARMS=("distance","neural","shuffled","economic_pair","objective")


def development_model(case,entry):
    nodes=np.asarray(read(HERE/entry["file"])["nodes"])
    distance=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2)
    model=core.PublicationModel(dict(name=case,nodes=nodes.tolist(),distance=distance.tolist()),core.load_config(),"main_Product1",.8)

    @lru_cache(maxsize=60000)
    def value(route):
        # Publication-model route evaluations include independent departure optimisation.
        res=model.route(route)
        return res.net_benefit if res.feasible else -math.inf
    return nodes,distance,model,value


def run_case(index,case,entry,states):
    cpu=time.process_time();wall=time.perf_counter()
    capture=HERE/"results/headcapture01/out"
    receipt=read(capture/f"{index}.json")
    if receipt["case"]!=case or receipt["max_logit_error"]>.002 or receipt["contexts"]!=12:
        raise ValueError("Masked model capture differs")
    nodes,distance,model,value=development_model(case,entry)

    spatial=reciprocal_weights(-distance)
    base=[value((i,)) if i else 0. for i in range(101)]
    pair=np.full((101,101),-np.inf)
    for i in range(1,101):
        for j in range(1,101):
            if i!=j:pair[i,j]=value((i,j))-base[i]-base[j]
    physical=reciprocal_weights(pair);rows=[]
    with np.load(capture/f"{index}.npz") as z:
        for k in (0,1,6,7):
            state="initial" if k<6 else "terminal";rep=k%6
            parent=tuple(map(tuple,states[state]));orig=independent_audit(model,[model.route(r) for r in parent],True)
            h=z["embed"][k];scores=(h@h.T)*receipt["scale"];mask=z["mask"][k].astype(int).tolist()
            anchor=mask[0];group=[anchor]+sorted(mask[1:],key=lambda i:(distance[anchor,i],i))[:11]
            order=tuple(sorted(group,key=lambda i:(-nodes[i,3],i)))
            shuffled=shuffle_probabilities(scores,distance,parent,mask,np.random.default_rng([index,k,54113]))
            rankers=(spatial,reciprocal_weights(scores),reciprocal_weights(shuffled),physical,None)
            arms={}
            for name,weights in zip(ARMS,rankers,strict=True):
                sol=beam_repair(parent,order,value,weights)
                u=independent_audit(model,[model.route(r) for r in sol["routes"]],True)
                if abs(u-sol["U"])>1.e-6 or u<orig-1.e-6:raise ValueError("Partial-beam fleet audit differs")
                arms[name]=sol
            rows.append(dict(state=state,rep=rep,mask=mask,order=order,original=orig,arms=arms))
    out=dict(case=case,rows=rows,strict_route_evaluations=value.cache_info().misses,
             cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)
    (OUT/f"{index}.json").write_text(json.dumps(out,allow_nan=False),encoding="utf8")


def summarize(plan,meta):
    rows=[];cpu=wall=cost=0
    for i,b in enumerate(plan):
        rr=read(OUT/f"{i}.json");c=b["case"]
        if rr["case"]!=c or len(rr["rows"])!=4:raise ValueError("Missing beam comparison")
        cpu+=rr["cpu_s"];wall+=rr["wall_s"];cost+=rr["strict_route_evaluations"]
        for r in rr["rows"]:rows.append(dict(**r,**meta[c],case=c))
    stats={}
    for source in SOURCES:
        rr=[r for r in rows if r["dataset"]==source]
        def by_geo(subset,control):
            groups=defaultdict(list)
            for r in subset:groups[r["geometry"]].append(r["arms"]["neural"]["U"]-r["arms"][control]["U"])
            return {g:mean(s) for g,s in groups.items()}
        def delta(subset,control="distance"):return mean(by_geo(subset,control).values())
        geo=by_geo(rr,"distance")
        stats[source]=dict(mean=mean(geo.values()),geometries=geo,wins=sum(x>1.e-6 for x in geo.values()),
            states={s:delta([r for r in rr if r["state"]==s]) for s in ("initial","terminal")},
            reps=[delta([r for r in rr if r["rep"]==j]) for j in (0,1)],
            controls={a:delta(rr,a) for a in ("shuffled","economic_pair","objective")},
            improved={a:sum(r["arms"][a]["U"]>r["original"]+1.e-6 for r in rr) for a in ARMS})
    passed=all(s["mean"]>1.e-6 and 2*s["wins"]>len(s["geometries"]) for s in stats.values())
    passed=passed and all(mean(s["reps"][n] for s in stats.values())>1.e-6 for n in (0,1))
    passed=passed and all(mean(s["controls"][a] for s in stats.values())>1.e-6 for a in ("shuffled","economic_pair","objective"))
    result=dict(sources=stats,passed=passed,cpu_s=cpu,wall_s_sequential_cases=wall,strict_route_evaluations=cost,
        independently_audited=len(rows)*len(ARMS),archived_forward_contexts=len(rows))
    (OUT/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    lines=["# 为剩余客户保留可行邻边的束搜索","","各臂展开相同数目的候选席位。数值为 MASKCO 减参照的开发几何等权净效益 U。","",
        "| 来源 | 对距离 | 改善几何 | 初始车队 | 强距离终态 | 对置乱 | 对双客经济 | 对全经济束 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source,s in stats.items():
        nums=[*s["states"].values(),*s["controls"].values()]
        lines.append(f"| {source} | {s['mean']:+.2f} | {s['wins']}/{len(s['geometries'])} | "+" | ".join(f"{x:+.2f}" for x in nums)+" |")
    lines+=["",f"共复用 {len(rows)} 个真实掩码前向，独立复算 {result['independently_audited']} 个车队。"
        f"新增 {cost:,} 次路线严格定价，累计 CPU {cpu:.1f} 秒。单核 60 秒开发门槛"+("通过。" if passed else "未通过。"),"",
        "[几何、两组掩码与改善频数](analysis.json) · [方案](../../给尚未插回的客户保留可行邻边.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))


def main():
    OUT.mkdir(exist_ok=True)
    cfg=read(HERE/"headcapture01.json");catalog=read(HERE/"heldout_sources.json");states=read(HERE/"archived_states.json")
    if read(HERE/"results/selectprobe01/analysis.json")["passed"]:
        raise ValueError("Single-forward selection already qualified")
    for i,b in enumerate(cfg["blocks"]):
        if not (OUT/f"{i}.json").exists():
            run_case(i,b["case"],catalog["derived_instances"][b["case"]],states[b["case"]])
    summarize(cfg["blocks"],catalog["cases"])

if __name__=="__main__":
    main()
