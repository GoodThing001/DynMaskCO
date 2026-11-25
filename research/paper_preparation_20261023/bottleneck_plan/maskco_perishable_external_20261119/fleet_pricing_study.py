"""Measure global receiving-skeleton reconstruction on matched development masks."""
from concurrent.futures import ProcessPoolExecutor
import argparse
from collections import Counter, defaultdict
import json
from statistics import mean
from time import perf_counter, process_time
import numpy as np
from directed_decoder_fit import HERE,SOURCES,read
from fleet_pricing import MaskedMaster,propose
from remaining_edges_study import development_model
from prepare_panel import independent_audit
from view_frame_probe import shuffle_probabilities

OUT=HERE/"results/fleetpricing01"
ARMS=("distance","neural","shuffled","without_prices")


def run_case(index):
    doc=read(HERE/"heldout_sources.json")
    case=read(HERE/"headcapture01.json")["blocks"][index]["case"]
    nodes,d,model,value=development_model(case,doc["derived_instances"][case])
    states=read(HERE/"archived_states.json")[case]
    capture=HERE/"results/headcapture01/out"
    receipt=read(capture/f"{index}.json")
    if receipt["case"]!=case or receipt["max_logit_error"]>.002:raise ValueError("Unmatched capture")
    records=[]; wall,cpu=perf_counter(),process_time()
    def audit(fleet,u):
        strict=independent_audit(model,[model.route(r) for r in fleet],True)
        if abs(strict-u)>1.e-6:raise ValueError("Independent full-fleet result differs")
        return strict
    with np.load(capture/f"{index}.npz") as tensors:
        for k in (2,3,8,9):
            parent=tuple(map(tuple,states["initial" if k<6 else "terminal"]))
            mask=tensors["mask"][k].astype(int).tolist()
            h=tensors["embed"][k]
            if len(set(mask))!=40:raise ValueError("Different masked move size")
            raw=h@h.T*receipt["scale"]
            shuffled=shuffle_probabilities(raw,d,parent,mask,np.random.default_rng([56731,index,k]))
            master=MaskedMaster(parent,mask,value)
            orig=audit(parent,sum(map(value,parent)))
            common=master.common();duals=master.prices(common)
            base,u0,g0=master.integer(common,parent);audit(base,u0)
            arms={}
            for a in np.random.default_rng([56697,index,k]).permutation(ARMS):
                signal=None if a=="distance" else shuffled if a=="shuffled" else raw
                pool,attempts=propose(master,common,duals,d,signal,nodes[:,3],a!="without_prices")
                fleet,u,gap=master.integer(pool,base);actual=audit(fleet,u)
                new=[r for r in fleet if master.cover(r) not in common or common[master.cover(r)][0]<value(r)-1.e-7]
                arms[a]=dict(U=actual,fleet=fleet,receiver_skeletons=len(master.skeletons),gap=gap,
                             selected_new=len(new),distinct_columns=len(pool),attempts=attempts)
            records.append(dict(k=k,mask=mask,original=orig,common=u0,common_gap=g0,common_columns=len(common),arms=arms))
    row=dict(case=case,rows=records,prices=value.cache_info().misses,cpu_s=process_time()-cpu,wall_s=perf_counter()-wall)
    (OUT/f"{index}.json").write_text(json.dumps(row,allow_nan=False),encoding="utf8")
    return case


def analyze(indices):
    boxes=[]; costs=Counter()
    meta=read(HERE/"heldout_sources.json")["cases"]
    plan=read(HERE/"headcapture01.json")["blocks"]
    for i in indices:
        doc=read(OUT/f"{i}.json");case=doc["case"]
        if case!=plan[i]["case"] or tuple(r["k"] for r in doc["rows"])!=(2,3,8,9):raise ValueError("Partial development block")
        for key in ("cpu_s","wall_s","prices"):costs[key]+=doc[key]
        for r in doc["rows"]:
            if set(r["arms"])!=set(ARMS):raise ValueError("Unmatched arm")
            boxes.append(dict(**r,case=case,**meta[case]))
    def aggregate(rows,fn):
        g=defaultdict(list)
        for r in rows:g[r["geometry"]].append(fn(r))
        return {k:mean(v) for k,v in g.items()}
    def macro(rows,fn):return mean(aggregate(rows,fn).values())
    results={}
    for src in SOURCES:
        rs=[r for r in boxes if r["dataset"]==src]
        if not rs:continue
        def delta(r,control="distance"):return r["arms"]["neural"]["U"]-r["arms"][control]["U"]
        by=aggregate(rs,delta)
        gaps=[r["arms"]["distance"]["gap"] for r in rs]
        worst=None if any(g is None for g in gaps) else macro(rs,lambda r:r["arms"]["distance"]["gap"])
        repeats=[macro([r for r in rs if r["k"]%6==k],delta) for k in (2,3)]
        place=macro(rs,lambda r:delta(r,"shuffled"))
        mean_u=mean(by.values());wins=sum(u>1.e-6 for u in by.values())
        passed=worst is not None and mean_u>worst+1.e-6 and 2*wins>len(by) and min(repeats)>1.e-6 and place>1.e-6
        results[src]=dict(mean=mean_u,by_geometry=by,wins=wins,repeats=repeats,placebo=place,
             price_effect=macro(rs,lambda r:delta(r,"without_prices")),base_gain=macro(rs,lambda r:r["common"]-r["original"]),
             neural_extra=macro(rs,lambda r:r["arms"]["neural"]["U"]-r["common"]),distance_gap=worst,
             max_gap=max((a["gap"] for r in rs for a in r["arms"].values() if a["gap"] is not None),default=None),
             selected_new=sum(r["arms"]["neural"]["selected_new"] for r in rs),
             moves=macro(rs,lambda r:r["arms"]["neural"]["attempts"]),
             states={st:macro([r for r in rs if (r["k"]<6)==(st=="initial")],delta) for st in ("initial","terminal")},
             passed=passed)
    stage="all" if len(indices)==27 else "denmark"
    bill=dict(costs,contexts=len(boxes),audits=6*len(boxes),archived_forwards=len(boxes))
    doc=dict(sources=results,costs=bill,passed=all(v["passed"] for v in results.values()),stage=stage)
    (OUT/f"{stage}.json").write_text(json.dumps(doc,indent=2,allow_nan=False),encoding="utf8")
    lines=["# 全车队接收路线的掩码定价","","相同的 40 名客户可向全车队重分配。表中是按几何等权的净效益差 U。共同库含原解、双客车和全部单客插位。","",
            "| 来源 | 共同库对原解 | 热图扩展对共同库 | 热图减同规则距离 | 改善几何 | 热图减置乱 | 使用对偶价格的增量 |","|---|---:|---:|---:|---:|---:|---:|"]
    for src,v in results.items():
        lines.append(f"| {src} | {v['base_gain']:+.2f} | {v['neural_extra']:+.2f} | {v['mean']:+.2f} | {v['wins']}/{len(v['by_geometry'])} | {v['placebo']:+.2f} | {v['price_effect']:+.2f} |")
    lines+=["","| 来源 | 初解 | 距离终态 | 第一掩码 | 第二掩码 | 距离臂平均求解界差 |","|---|---:|---:|---:|---:|---:|"]
    for s,v in results.items():
        gap=v["distance_gap"]
        lines.append(f"| {s} | "+" | ".join(f"{u:+.2f}" for u in (*v["states"].values(),*v["repeats"]))+" | "+("未知" if gap is None else f"{gap:.3f}")+" |")
    lines+=["",f"{len(boxes)} 个配对上下文、{bill['audits']} 次全车队独立复算、{int(bill['prices']):,} 次严格路线定价，累计 CPU {bill['cpu_s']:.1f} 秒。"
            +("开发扩展门槛通过。" if doc["passed"] else "开发扩展门槛未通过。"),
            "","[方法和门槛](../../全车队接收路线与联合掩码定价.md) · [逐几何结果]("+stage+".json)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
    return doc


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--case",type=int);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.case is not None:
        print(run_case(args.case));return
    meta=read(HERE/"heldout_sources.json")["cases"]
    cases=read(HERE/"headcapture01.json")["blocks"]
    dk=[i for i,b in enumerate(cases) if meta[b["case"]]["dataset"]==SOURCES[0]]
    if len(dk)!=5:raise ValueError("Missing Danish geometries")
    for group in (dk,list(range(len(cases)))):
        with ProcessPoolExecutor(max_workers=2) as pool:
            for case in pool.map(run_case,[i for i in group if not (OUT/f"{i}.json").is_file()]):
                print(case,flush=True)
        if not analyze(group)["passed"]:return


if __name__=="__main__":main()
