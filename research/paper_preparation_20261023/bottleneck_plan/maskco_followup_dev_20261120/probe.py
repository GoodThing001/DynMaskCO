"""Recombine all initial pricing columns with the same archived terminal fleet."""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter, process_time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "maskco_perishable_external_20261119"
sys.path.insert(0, str(DEV))
from directed_decoder_fit import SOURCES, read
from fleet_pricing import MaskedMaster, propose
from prepare_panel import independent_audit
from remaining_edges_study import development_model

ARMS = ("common", "selected", "neural", "distance")
EPS = 1e-6


def audit(model, routes, expected):
    if sorted(v for route in routes for v in route) != list(range(1, 101)):
        raise ValueError("Fleet must cover each customer exactly once")
    u = independent_audit(model, [model.route(r) for r in routes], True)
    if abs(u - expected) > EPS:
        raise ValueError("Cold-chain net benefit mismatch")
    return u


def solve_case(case, index, case_info):
    inputs = read(DEV / "heldout_sources.json")["derived_instances"][case]
    nodes, dist, model, value = development_model(case, inputs)
    states = read(DEV / "archived_states.json")[case]
    start = tuple(tuple(r) for r in states["initial"])
    terminal = tuple(tuple(r) for r in states["terminal"])
    terminal_u = audit(model, terminal, sum(map(value, terminal)))
    capture = DEV / "results/headcapture01/out"
    identity = read(capture / f"{index}.json")
    prior = read(DEV / f"results/fleetpricing01/{index}.json")
    if case != identity["case"] or case != prior["case"] or identity["max_logit_error"] > .002:
        raise ValueError("Mismatched development capture")
    cover = MaskedMaster(terminal, list(range(1, 101)), value)
    rows = []
    with np.load(capture / f"{index}.npz") as tensors:
        for k in (2, 3):
            past = next(r for r in prior["rows"] if r["k"] == k)
            mask = tensors["mask"][k].astype(int).tolist()
            if mask != past["mask"] or len(set(mask)) != 40:
                raise ValueError("Unpaired mask")
            h = tensors["embed"][k]
            signal = (h @ h.T) * identity["scale"]
            master = MaskedMaster(start, mask, value)
            common = master.common()
            duals = master.prices(common)
            generated = {}
            for name, heat in (("distance", None), ("neural", signal)):
                generated[name], _ = propose(master, common, duals, dist, heat, nodes[:,3])
                if len(generated[name]) != past["arms"][name]["distinct_columns"]:
                    raise ValueError(f"Frozen pricing reconstruction differs: {case}/{k}/{name}")
            base = {}
            for route in terminal:
                cover.insert(base, route)
            for _, route in common.values():
                cover.insert(base, route)
            accepted = tuple(tuple(r) for r in past["arms"]["neural"]["fleet"])
            audit(model, accepted, past["arms"]["neural"]["U"])
            pools = {"common":base, "selected":base.copy(), "neural":base.copy(), "distance":base.copy()}
            for route in accepted:
                cover.insert(pools["selected"], route)
            for name in ("neural", "distance"):
                for _, route in generated[name].values():
                    cover.insert(pools[name], route)
            stats = {}
            order = np.random.default_rng([20261120,index,k]).permutation(ARMS)
            for name in order:
                t0=perf_counter()
                fleet,u,gap=cover.integer(pools[name],terminal)
                elapsed=perf_counter()-t0
                score=audit(model,fleet,u)
                if score+EPS<terminal_u:
                    raise ValueError("Terminal retention failed")
                exclusive=0
                if name=="neural":
                    for r in fleet:
                        key=cover.cover(r)
                        previous=pools["distance"].get(key)
                        if previous is None or value(r)>previous[0]+EPS:
                            exclusive+=1
                stats[name]=dict(U=score, gap_U=gap, fleet=fleet, elapsed_s=elapsed,
                                 columns=len(pools[name]), neural_exclusive=exclusive)
            rows.append(dict(case=case, dataset=case_info["dataset"], geometry=case_info["geometry"],
                             mask_index=k, terminal_U=terminal_u, arms=stats))
    return rows


def geometric(rows, value):
    per_case = defaultdict(list)
    for r in rows:
        per_case[r["geometry"], r["case"]].append(value(r))
    result = defaultdict(list)
    for (g, _), vals in per_case.items():
        result[g].append(mean(vals))
    return {g:mean(vals) for g,vals in result.items()}


def summarize(rows):
    stats = {}
    for source in SOURCES:
        sample=[r for r in rows if r["dataset"]==source]
        result={}
        for name in ("distance","selected","common","terminal"):
            def delta(r):
                control = r["terminal_U"] if name=="terminal" else r["arms"][name]["U"]
                return r["arms"]["neural"]["U"]-control
            gs=geometric(sample,delta)
            vs=mean(gs.values())
            repeats=[mean(geometric([r for r in sample if r["mask_index"]==k],delta).values()) for k in (2,3)]
            gap=0 if name=="terminal" else None
            if name!="terminal" and all(r["arms"][name]["gap_U"] is not None for r in sample):
                gap=mean(geometric(sample,lambda r:r["arms"][name]["gap_U"]).values())
            result[name]=dict(mean=vs,wins=sum(u>EPS for u in gs.values()), groups=len(gs),
                              by_geometry=gs,repeats=repeats,control_gap_U=gap)
        d,sc=result["distance"],result["selected"]
        passed=(all(x["control_gap_U"] is not None and x["mean"]>x["control_gap_U"]+EPS for x in (d,sc))
                and d["wins"]*2>d["groups"] and min(d["repeats"])>EPS)
        stats[source]=dict(contrasts=result,passed=passed)
    return stats


def main():
    cases=read(DEV/"heldout_sources.json")["cases"]
    subset=sorted({r["case"] for r in read(DEV/"fleetscreen01.json")["blocks"]})
    blocks=read(DEV/"headcapture01.json")["blocks"]
    if len(subset)!=11 or set(subset)-set(cases):
        raise ValueError("Unexpected development panel")
    out=HERE/"results"/"columnprobe01"
    out.mkdir(parents=True)
    rows=[]
    t0,c0=perf_counter(),process_time()
    for case in subset:
        index=next(i for i,b in enumerate(blocks) if b["case"]==case)
        case_rows=solve_case(case,index,cases[case])
        (out/f"{case}.json").write_text(json.dumps(case_rows,allow_nan=False),encoding="utf8")
        rows.extend(case_rows)
        print(case, len(rows),flush=True)
    if len(rows)!=22:
        raise ValueError("Incomplete diagnostic")
    stats=summarize(rows)
    passed=all(s["passed"] for s in stats.values())
    bill=dict(contexts=len(rows),final_audits=4*len(rows),
              solve_s=sum(a["elapsed_s"] for r in rows for a in r["arms"].values()),
              cpu_s=process_time()-c0,wall_s=perf_counter()-t0)
    data=dict(status="COMPLETE",passed=passed,decision="RUN_TIMED_SCREEN" if passed else "STOP_COLUMN_RETENTION",
              sources=stats,costs=bill)
    with (out/"paired.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["source","geometry","case","mask_index","terminal_U",*ARMS])
        for r in rows:
            w.writerow([r["dataset"],r["geometry"],r["case"],r["mask_index"],r["terminal_U"],
                        *[r["arms"][a]["U"] for a in ARMS]])
    (out/"analysis.json").write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf8")
    lines=["# 首次定价候选路线与强终态的重组","",
           "从共同初解实际取得的两个 MASKCO 掩码分别生成路线列，随后与同一存档强距离终态配对重组。差值为神经全列减列标题指定的参照，单位 U。","",
           "| 开发来源 | 对强终态 | 对共同列 | 对前缀入选列 | 对全部距离列 | 改善几何 | 两次掩码对距离的差值 | 距离臂剩余界差 |",
           "|---|---:|---:|---:|---:|---:|---|---:|"]
    for source, record in stats.items():
        c=record["contrasts"];d=c["distance"]
        gap=d["control_gap_U"]
        lines.append("| "+source+" | "+" | ".join(f"{c[a]['mean']:+.2f}" for a in ("terminal","common","selected","distance"))+
                     f" | {d['wins']}/{d['groups']} | "+" / ".join(f"{v:+.2f}" for v in d["repeats"])+
                     " | "+("未知" if gap is None else f"{gap:.2f}")+" |")
    sizes=[r["arms"]["neural"]["columns"] for r in rows]
    bounds=[a["gap_U"] for r in rows for a in r["arms"].values()]
    if all(b is not None for b in bounds):
        lines.extend(["",f"神经重组池每次保留 {min(sizes):,}–{max(sizes):,} 个不同覆盖，四臂全部组合问题的最大剩余界差为 {max(bounds):.2e} U。"])
    lines.extend(["",f"11 个既有开发例、22 个掩码上下文、88 个重组终点完成独立复算。组合求解累计 {bill['solve_s']:.2f} 秒，全程 CPU {bill['cpu_s']:.2f} 秒、墙钟 {bill['wall_s']:.2f} 秒。",
                  "",f"预设门槛{'通过' if passed else '未通过'}（{data['decision']}）。本表衡量固定列池中的后期机会。","",
                  "[开发预登记](../../开发预登记.md) · [逐例记录](paired.csv) · [几何、界差与重复](analysis.json)"])
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
