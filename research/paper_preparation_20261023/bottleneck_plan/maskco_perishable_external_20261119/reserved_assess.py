"""Paired development analysis for a single declared matrix."""
from collections import defaultdict
import csv
from itertools import product
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"maskco_cross_dataset_20261119"
EPS=1e-6
def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def describe(values,metadata):
    case_rows=defaultdict(list)
    for (case,seed),v in values.items():
        case_rows[case].append(v)
    by_geometry=defaultdict(list)
    for case,vs in case_rows.items():
        by_geometry[metadata[case]["geometry"]].append(mean(vs))
    gs={g:mean(v) for g,v in by_geometry.items()}
    seed_means={}
    for s in sorted({s for c,s in values}):
        seed_means[s]=mean(mean(v for (c,t),v in values.items() if t==s and metadata[c]["geometry"]==g) for g in gs)
    ci=None
    if len(gs)>1:
        x=np.asarray(list(gs.values()))
        ix=np.random.default_rng(211419).integers(len(x),size=(10000,len(x)))
        ci=np.quantile(x[ix].mean(axis=1),[.025,.975]).tolist()
    return dict(mean=mean(gs.values()),by_seed=seed_means,by_geometry=gs,wins=sum(v>EPS for v in gs.values()),
                geometries=len(gs),positive_seeds=sum(v>EPS for v in seed_means.values()),ci95=ci,
                worst_case=min(((c,mean(v)) for c,v in case_rows.items()),key=lambda item:item[1]))

def report(phase):
    plan=read(HERE/(phase+".json"))
    folder=HERE/"results"/phase
    run=read(folder/"status.json")
    if not run["end"] or run["failed"] or len(plan["blocks"])!=run["complete"]:
        raise ValueError("Incomplete experiment")
    confirmation=plan.get("purpose")=="locked independent comparison"
    metadata=read(HERE/"heldout_sources.json" if confirmation else OLD/"matrix.json")["cases"]
    endpoints={}
    for i,block in enumerate(plan["blocks"]):
        row=read(folder/"out"/f"{i}.json")
        if (row["case"],row["seed"])!=(block["case"],block["seed"]) or set(row["arms"])!=set(block["arms"]):
            raise ValueError("Unexpected block identity")
        for arm,r in row["arms"].items():
            key=(row["case"],row["seed"],arm)
            if (r["status"]!="AUDITED" or not r["endpoint"]["independently_audited"] or
                r["timing"]["budget_s"]!=plan["seconds"] or r["selected_s"]>plan["seconds"] or
                abs(r["U"]-r["endpoint"]["net_benefit_U"])>EPS or key in endpoints):
                raise ValueError(f"Invalid endpoint {key}")
            endpoints[key]=r
    sources={}
    cases={c for c,s,arm in endpoints}
    for dataset in sorted({metadata[c]["dataset"] for c in cases}):
        pairs={(c,s) for c,s,m in endpoints if metadata[c]["dataset"]==dataset}
        methods=sorted({arm for c,s,arm in endpoints if (c,s) in pairs})
        def measure(m,control=None):
            return describe({(c,s):endpoints[c,s,m]["U"]-(endpoints[c,s,control]["U"] if control else 0.) for c,s in sorted(pairs)},metadata)
        scores={m:measure(m)["mean"] for m in methods}
        contrasts={}
        for m in methods:
            if m+"_D" not in methods:continue
            contrasts[m]={k:measure(m,k) for k in (m+"_D","DIST","DIST_SLOT") if k in methods}
        ranks={m:1+sum(other!=m and v>=scores[m]-EPS for other,v in scores.items()) for m in scores}
        sources[dataset]=dict(pairs=len(pairs),scores=scores,ranks=ranks,contrasts=contrasts)
    rows=list(endpoints.values())
    decisions={}
    for d in sources.values():
        for m in d["contrasts"]:
            if m not in decisions:
                contrasts=[v["contrasts"][m] for v in sources.values()]
                decisions[m]=all(all(r["mean"]>EPS and (confirmation or r["positive_seeds"]>=2) for r in vs.values())
                                 and vs[m+"_D"]["positive_seeds"]>=2 and vs[m+"_D"]["wins"]*2>vs[m+"_D"]["geometries"] for vs in contrasts)
                if confirmation:
                    decisions[m] = decisions[m] and all(v["ranks"][m]<=3 for v in sources.values())
    costs=dict(cpu_s=sum(r["cpu_s"] for r in rows),search_budget_s=len(rows)*plan["seconds"],
               recorded_forwards=sum(r["forwards"] or 0 for r in rows),cells_without_forward_count=sum(r["forwards"] is None for r in rows),native_route_requests=sum(r.get("native_route_requests",0) for r in rows),
               python_route_requests=sum(r.get("requests",0) for r in rows))
    result=dict(phase=phase,complete=len(rows),sources=sources,passes_numerical_criteria=decisions,costs=costs)
    if confirmation:
        candidate=read(HERE/"locked_method.json")["candidate"]
        paired_arms=[candidate,candidate+"_D","DIST","DIST_SLOT","ALNS","VNS","NDS-Econ","CaR","DeepACO","POMO","AttentionModel","HYBRID","HYBRID_D"]
        registered=read(HERE/"locked_method.json")["seeds"]
        expected=set(product(metadata,registered,paired_arms))
        if set(endpoints)!=expected:raise ValueError("Independent sources, seeds or arms differ from the lock")
        with (folder/"paired_test.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["source","geometry","case","seed","initial_U"]+paired_arms+["heat_delta_U"])
            for case,seed in sorted(product(metadata,registered)):
                values=[endpoints[case,seed,arm]["U"] for arm in paired_arms]
                writer.writerow([metadata[case]["dataset"],metadata[case]["geometry"],case,seed,endpoints[case,seed,candidate]["initial_U"]]+values+[values[0]-values[1]])
        for name,source in sources.items():
            scores=source["scores"]
            source["canonical_rank_11"]=1+sum(scores[m]>=scores[candidate]-EPS for m in paired_arms[:11] if m!=candidate)
            pairs=sorted((c,s) for c,s in product(metadata,registered) if metadata[c]["dataset"]==name)
            diff=[(endpoints[c,s,candidate]["U"]-endpoints[c,s,candidate+"_D"]["U"],c,s) for c,s in pairs]
            source["worst_paired_heat_delta"]=min(diff)
            source["external_contrasts"]={
                arm:describe({(c,s):endpoints[c,s,candidate]["U"]-endpoints[c,s,arm]["U"] for c,s in pairs},metadata)
                for arm in ("ALNS","NDS-Econ","HYBRID","HYBRID_D")}
            source["endpoints_at_initial_profit"]={
                arm:sum(abs(endpoints[c,s,arm]["U"]-endpoints[c,s,arm]["initial_U"])<=EPS for c,s in pairs)
                for arm in paired_arms}
            source["seed_ranks"]={}
            for seed in registered:
                sub=[(c,s) for c,s in pairs if s==seed]
                avg={a:describe({(c,s):endpoints[c,s,a]["U"] for c,s in sub},metadata)["mean"] for a in paired_arms}
                source["seed_ranks"][seed]=1+sum(v>=avg[candidate]-EPS for a,v in avg.items() if a!=candidate)
        rng=np.random.default_rng(261119)
        draws=[]
        for source in sources.values():
            delta=source["contrasts"][candidate][candidate+"_D"]
            x=np.array(list(delta["by_geometry"].values()))
            draws.append(x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1))
        interval=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist()
        result["three_source_equal_weight_ci95"]=interval
        result["pooled_positive_supported"]=interval[0]>0
    (folder/"analysis.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    lines=["| 来源 | 候选 | 对同规则距离 U | 改善几何 | 对 DIST U | 对 DIST_SLOT U | 对同规则距离逐种子 U |",
           "|---|---|---:|---:|---:|---:|---|"]
    for d,ds in sources.items():
        for m,v in ds["contrasts"].items():
            r=v[m+"_D"]
            def avg(control):
                return f"{v[control]['mean']:+.2f}" if control in v else "—"
            seed_text=" / ".join(f"{x:+.2f}" for x in r["by_seed"].values())
            lines.append(f"| {d} | {m} | {r['mean']:+.2f} | {r['wins']}/{r['geometries']} | {avg('DIST')} | {avg('DIST_SLOT')} | {seed_text} |")
    (folder/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
    print(decisions,costs)
if __name__=="__main__":
    report(sys.argv[1])
