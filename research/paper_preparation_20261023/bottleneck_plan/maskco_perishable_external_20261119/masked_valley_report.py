"""Estimate the incremental value of neural edge information after common local search."""
from collections import Counter
from statistics import mean
import json
import numpy as np
from directed_decoder_fit import HERE, SOURCES, read
from masked_order_fit import DESIGNS, fit, group_effect
from masked_valley import OUT, BANK, MASKS


def main():
    blocks=read(HERE/"headcapture01.json")["blocks"]
    meta=read(HERE/"heldout_sources.json")["cases"]
    rows=[]; costs=Counter()
    for i,b in enumerate(blocks):
        bank=read(BANK/f"{i}.json")
        run=read(OUT/f"{i}.json")
        case=b["case"]
        if bank["case"]!=case or run["case"]!=case or tuple(r["k"] for r in run["rows"])!=MASKS:
            raise ValueError("Incomplete paired extension")
        for name in ("cpu_s","wall_s","prices","attempts","audits"):
            costs[name]+=run[name]
        for a,r in zip(bank["rows"],run["rows"],strict=True):
            idx=r["selected"]
            if a["mask_index"]!=r["k"] or len(r["U"])!=1+len(idx) or len(set(idx))!=len(idx):
                raise ValueError("Different action set")
            direct=np.asarray(a["U"][1:])[idx]
            if not np.all(direct<a["U"][0]-1.e-6):
                raise ValueError("Continuation did not begin below its parent")
            rows.append(dict(**meta[case],case=case,index=i,k=r["k"],features=np.asarray(a["features"])[idx],
                             labels=np.array(r["U"][1:]),direct=direct,base=r["U"][0],indices=idx))
    if costs["attempts"]!=512*costs["audits"]:
        raise ValueError("Unequal local budgets")
    folds={(s,g):i%5 for s in SOURCES for i,g in enumerate(sorted({r["geometry"] for r in rows if r["dataset"]==s}))}
    key=lambda r:folds[r["dataset"],r["geometry"]]
    results=[]
    for fold in range(5):
        train=[r for r in rows if key(r)!=fold and r["k"] in (0,6) and len(r["labels"])>0]
        test=[r for r in rows if key(r)==fold and r["k"] not in (0,6)]
        if {r["geometry"] for r in train}&{r["geometry"] for r in test}:raise ValueError("Geometry overlap")
        weights={m:fit(train,cols) for m,cols in DESIGNS.items()}
        for r in test:
            n=len(r["labels"])
            scores={m:r["base"] for m in ("base","neural","physical","distance","shuffled","immediate","random","oracle")}
            chosen={}
            if n:
                X=r["features"]
                priorities={"immediate":r["direct"],"distance":X[:,0],
                            "random":np.random.default_rng([56519,r["index"],r["k"]]).random(n)}
                for m,cols in DESIGNS.items():
                    sd,coef=weights[m]
                    a=X[:,cols]
                    priorities[m]=np.clip((a-a.mean(axis=0))/sd,-8.,8.)@coef
                for m,v in priorities.items():
                    ids=np.argsort(-v,kind="stable")[:2]
                    scores[m]=max(scores[m],float(r["labels"][ids].max()))
                    chosen[m]=[r["indices"][int(j)] for j in ids]
                scores["oracle"]=max(scores["base"],float(r["labels"].max()))
            results.append(dict(case=r["case"],k=r["k"],geometry=r["geometry"],dataset=r["dataset"],
                                scores=scores,selected=chosen,options=n))
    sources={}
    for src in SOURCES:
        part=[r for r in results if r["dataset"]==src]
        controls={}
        for m in ("distance","physical","shuffled","immediate","random","oracle","base"):
            g=group_effect(part,m)
            controls[m]=dict(mean=mean(g.values()),geometries=g,wins=sum(u>1.e-6 for u in g.values()))
        sources[src]=dict(controls=controls,groups=len(controls["base"]["geometries"]),contexts=len(part),
                          available=sum(r["scores"]["oracle"]-r["scores"]["base"]>1.e-6 for r in part),
                          repeats=[mean(group_effect([r for r in part if r["k"]%6==rep],"physical").values()) for rep in (4,5)])
    passed=all(v["controls"][m]["mean"]>1.e-6 and 2*v["controls"][m]["wins"]>v["groups"]
               for v in sources.values() for m in ("physical","distance"))
    passed=passed and all(mean(v["repeats"][q] for v in sources.values())>1.e-6 for q in (0,1))
    passed=passed and all(mean(v["controls"][m]["mean"] for v in sources.values())>1.e-6 for m in ("shuffled","immediate"))
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=sources,passed=passed,costs=costs),indent=2),encoding="utf8")
    (OUT/"selections.json").write_text(json.dumps(results,separators=(",",":")),encoding="utf8")
    lines=["# 暂时下降后的车队价值","","每臂从相同可行车队中选两条继续，各走 512 次客户操作，并保留共同车队走同样步数的结果。机会为给所有入选中间车队继续搜索后，相对共同车队的事后最好增量。单位为几何等权 U。","",
           "| 来源 | 可改善的上下文 | 事后机会 | 热图减距离 | 改善几何 | 热图减物理 | 改善几何 | 热图减即时收益 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s,v in sources.items():
        z=v["controls"]
        lines.append(f"| {s} | {v['available']}/{v['contexts']} | {z['base']['mean']-z['oracle']['mean']:+.2f} | "
                     +f"{z['distance']['mean']:+.2f} | {z['distance']['wins']}/{v['groups']} | {z['physical']['mean']:+.2f} | "
                     +f"{z['physical']['wins']}/{v['groups']} | {z['immediate']['mean']:+.2f} |")
    lines+=["",f"独立复算 {costs['audits']:,} 个终态，{costs['attempts']:,} 个预排操作，严格路线定价 {costs['prices']:,} 次，累计 CPU {costs['cpu_s']:.1f} 秒。"
            +("单核 60 秒开发门槛通过。" if passed else "单核 60 秒开发门槛未通过。"),
            "","[预定门槛](../../较差车队继续重构的经济价值.md) · [逐几何和掩码](analysis.json)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))


if __name__=="__main__":
    main()
