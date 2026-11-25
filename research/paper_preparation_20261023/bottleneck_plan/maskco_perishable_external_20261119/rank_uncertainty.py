"""Paired bootstrap of all method ranks using geometry as the resampling unit."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import csv
import json
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"maskco_heatmap_calibration_20261119"
def read(f):return json.loads(f.read_text(encoding="utf8"))
def rank_stats(geos,methods,method):
    scores=np.array([[mean(g[m]) for m in methods] for g in geos])
    if len(geos)<2:return None
    idx=np.random.default_rng(261119714).integers(0,len(geos),size=(20000,len(geos)))
    replicates=scores[idx].mean(axis=1)
    m=methods.index(method)
    ranks=(replicates>=replicates[:,[m]]-1.e-6).sum(axis=1)
    lo,hi=np.quantile(ranks,[.025,.975],method="nearest").astype(int)
    return dict(top3_fraction=float(np.mean(ranks<=3)),middle95_rank=[int(lo),int(hi)])
def data(root,stage):
    meta=read(root/"heldout_sources.json")["cases"]
    cfg=read(root/(stage+".json"));folder=root/"results"/stage
    if read(folder/"independent_audit.json")["status"]!="PASS":raise ValueError("Missing independent audit")
    out=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for i,b in enumerate(cfg["blocks"]):
        rec=read(folder/f"out/{i}.json")["arms"];x=meta[b["case"]]
        for m,w in rec.items():out[x["dataset"]][x["geometry"]][m].append(w["U"])
    return out
def main():
    studies=[("首轮",OLD,"confirm01","HYBRID"),("易腐品复核",HERE,"replicate01","HYBRID")]
    future=HERE/"reserved_validation"
    if (future/"results/reserve02/analysis.json").is_file():studies.append(("锁版预留",future,"reserve02",read(future/"locked_method.json")["candidate"]))
    lines=["# 同一几何样本变化时的名次稳定性","","每次对某一来源的几何有放回抽样，纳入该几何的全部实例、三个种子和全部方法，保留方法间配对。表中为两万次重采样中名列前三的比例及名次的中间 95% 范围。单一几何只给实测名次。","",
           "| 面板 | 来源 | 几何组 | 方法数 | 实测名次 | 重采样进入前三 | 95% 名次范围 |","|---|---|---:|---:|---:|---:|---|"]
    measurements=[(title,method,data(root,stage)) for title,root,stage,method in studies]
    supplement=HERE/"xtw_low_overlap"
    if (supplement/"analysis.json").is_file():
        selected=read(supplement/"selection.json")
        cases=set(selected["retained"]+selected["added"])
        meta={**read(future/"heldout_sources.json")["cases"],**read(supplement/"heldout_sources.json")["cases"]}
        groups=defaultdict(lambda:defaultdict(list))
        with (supplement/"paired_xtw.csv").open(encoding="utf-8-sig",newline="") as handle:
            for row in csv.DictReader(handle):
                if row["case"] in cases:
                    group=groups[meta[row["case"]]["geometry"]]
                    for arm,score in row.items():
                        if arm not in ("case","seed","panel"):group[arm].append(float(score))
        if len(groups)!=10 or any(len(g)!=13 or any(len(v)!=3 for v in g.values()) for g in groups.values()):
            raise ValueError("Incomplete low-overlap rank comparison")
        measurements.append(("客户低重叠复核",read(future/"locked_method.json")["candidate"],{"XTW-p100":groups}))
    for title,method,raw in measurements:
        for src,groups in sorted(raw.items()):
            geos=list(groups.values());methods=sorted(geos[0]);z=rank_stats(geos,methods,method)
            observed={m:mean(mean(v[m]) for v in geos) for m in methods}
            rank=1+sum(v>=observed[method]-1e-6 for m,v in observed.items() if m!=method)
            frac=f"{z['top3_fraction']:.1%}" if z else "—";interval="–".join(map(str,z["middle95_rank"])) if z else "—"
            lines.append(f"| {title} | {src} | {len(geos)} | {len(methods)} | {rank} | {frac} | {interval} |")
    (HERE/"名次稳定性.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines[4:]))
if __name__=="__main__":main()
