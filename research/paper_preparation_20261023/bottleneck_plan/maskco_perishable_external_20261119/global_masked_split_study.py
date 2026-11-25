"""Measure heat-guided permutations on complete fleets using existing masked development forwards."""
from collections import defaultdict
from statistics import mean
from pathlib import Path
import json
import math
import time
import numpy as np
from directed_decoder_fit import HERE, SOURCES, read
from prepare_panel import independent_audit
from remaining_edges_study import development_model
from global_masked_split import chain, propose, split_order
from view_frame_probe import shuffle_probabilities

OUT=HERE/"results/global_split01"
ARMS=("distance","neural","shuffled","economic_pair")


def run_case(index,case,entry,states):
    wall=time.perf_counter();cpu=time.process_time()
    nodes,d,model,value=development_model(case,entry)
    src=HERE/"results/headcapture01/out"
    receipt=read(src/f"{index}.json")
    if receipt["case"]!=case or receipt["max_logit_error"]>.002:raise ValueError("Model capture differs")
    mass=nodes[:,3];base=[0.]+[value((i,)) for i in range(1,101)]
    physical=np.full_like(d,-np.inf)
    for i in range(1,101):
        for j in range(1,101):
            if i!=j:physical[i,j]=value((i,j))-base[i]-base[j]
    rows=[]
    with np.load(src/f"{index}.npz") as arrays:
        for k in (2,3,8,9):
            state="initial" if k<6 else "terminal";rep=k%6
            routes=tuple(map(tuple,states[state]))
            old=independent_audit(model,[model.route(r) for r in routes],True)
            mask=arrays["mask"][k].astype(int).tolist()
            h=arrays["embed"][k];scores=(h@h.T)*receipt["scale"]
            shuffled=shuffle_probabilities(scores,d,routes,mask,np.random.default_rng([index,k,55619]))
            seed=55031+index*101+k
            first=int(np.random.default_rng(seed).integers(len(routes)))
            order=chain(routes,first,d)
            u0,r0,common_calls=split_order(order,mass,value)
            fixed=set(order)-set(mask);visible=tuple(i for i in order if i in fixed)
            if u0<old-1.e-6 or abs(u0-independent_audit(model,[model.route(r) for r in r0],True))>1.e-6:raise ValueError("Base split differs")
            arms={}
            for a,ranking in zip(ARMS,(-d,scores,shuffled,physical),strict=True):
                best,win,winner=u0,r0,order
                calls=0;unique=set()
                for seq in propose(order,mask,routes,ranking,mass,seed,32):
                    if len(seq)!=100 or set(seq)!=set(order) or tuple(i for i in seq if i in fixed)!=visible:
                        raise ValueError("Changed visible order or customer set")
                    u,fleet,spent=split_order(seq,mass,value);calls+=spent;unique.add(seq)
                    if u>best+1.e-7:best,win,winner=u,fleet,seq
                audited=independent_audit(model,[model.route(r) for r in win],True)
                if abs(best-audited)>1.e-6:raise ValueError("Global split independently rescheduled value differs")
                arms[a]=dict(U=audited,routes=win,order=winner,segment_requests=calls,unique_orders=len(unique))
            rows.append(dict(state=state,rep=rep,mask=mask,original=old,common=u0,common_calls=common_calls,arms=arms))
    doc=dict(case=case,rows=rows,strict_pricings=value.cache_info().misses,
             cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)
    (OUT/f"{index}.json").write_text(json.dumps(doc,allow_nan=False),encoding="utf8")


def report(blocks,meta):
    rows=[];cpu=wall=0.;requests=prices=0
    for i,b in enumerate(blocks):
        r=read(OUT/f"{i}.json");c=b["case"]
        if r["case"]!=c or len(r["rows"])!=4:raise ValueError("Incomplete full order experiment")
        cpu+=r["cpu_s"];wall+=r["wall_s"];prices+=r["strict_pricings"]
        for z in r["rows"]:
            requests+=z["common_calls"]+sum(v["segment_requests"] for v in z["arms"].values())
            rows.append(dict(**z,case=c,**meta[c]))
    result={}
    for src in SOURCES:
        group=[r for r in rows if r["dataset"]==src]
        def diff(part,fn):
            g=defaultdict(list)
            for r in part:g[r["geometry"]].append(fn(r))
            return {k:mean(v) for k,v in g.items()}
        def delta(r,a="distance"):return r["arms"]["neural"]["U"]-r["arms"][a]["U"]
        geo=diff(group,delta)
        result[src]=dict(mean=mean(geo.values()),geometries=geo,wins=sum(v>1.e-6 for v in geo.values()),
            common=mean(diff(group,lambda r:r["common"]-r["original"]).values()),
            heat_over_common=mean(diff(group,lambda r:r["arms"]["neural"]["U"]-r["common"]).values()),
            states={s:mean(diff([r for r in group if r["state"]==s],delta).values()) for s in ("initial","terminal")},
            reps=[mean(diff([r for r in group if r["rep"]==j],delta).values()) for j in (2,3)],
            controls={a:mean(diff(group,lambda r:delta(r,a)).values()) for a in ("shuffled","economic_pair")},
            leave_best_out=(sum(geo.values())-max(geo.values()))/(len(geo)-1))
    passed=all(s["mean"]>1.e-6 and 2*s["wins"]>len(s["geometries"]) for s in result.values())
    passed=passed and all(mean(s["reps"][i] for s in result.values())>1.e-6 for i in (0,1))
    passed=passed and all(mean(s["controls"][a] for s in result.values())>1.e-6 for a in ("shuffled","economic_pair"))
    doc=dict(sources=result,passed=passed,contexts=len(rows),tested_orders=len(rows)*(1+32*len(ARMS)),
             audited=len(rows)*(1+len(ARMS)),segment_requests=requests,strict_pricings=prices,cpu_s=cpu,wall_s_sequential_cases=wall)
    (OUT/"analysis.json").write_text(json.dumps(doc,indent=2),encoding="utf8")
    lines=["# 完整访问序列的掩码重构与经济分车","","全序列动态规划包括所有客户。共同拼接列为单独重新分车相对原车队的收益；神经顺序列是 32 个新序列中择优后相对共同拼接的额外收益。单位 U。","",
        "| 来源 | 共同拼接 | 神经顺序额外收益 | 热图减同规则距离 | 改善几何 | 热图减置乱 | 热图减双客经济 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for n,s in result.items():
        lines.append(f"| {n} | {s['common']:+.2f} | {s['heat_over_common']:+.2f} | {s['mean']:+.2f} | "
          +f"{s['wins']}/{len(s['geometries'])} | {s['controls']['shuffled']:+.2f} | {s['controls']['economic_pair']:+.2f} |")
    lines+=["",f"共 {doc['contexts']} 个真实掩码上下文，{doc['tested_orders']:,} 次完整分车，独立复算 {doc['audited']} 个共同或入选车队。"
      f"累计 CPU {cpu:.1f} 秒。单核 60 秒开发门槛"+("通过。" if passed else "未通过。"),"",
      "[起点、重复掩码和逐几何](analysis.json) · [预定方案](../../全车队掩码顺序与经济分车.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))


def main():
    if read(HERE/"results/remaining_edges01/analysis.json")["passed"]:raise ValueError("Partial fleet priority already qualified")
    OUT.mkdir(exist_ok=True)
    blocks=read(HERE/"headcapture01.json")["blocks"];catalog=read(HERE/"heldout_sources.json");states=read(HERE/"archived_states.json")
    for i,b in enumerate(blocks):
        if not (OUT/f"{i}.json").exists():
            run_case(i,b["case"],catalog["derived_instances"][b["case"]],states[b["case"]])
    report(blocks,catalog["cases"])


if __name__=="__main__":
    main()
