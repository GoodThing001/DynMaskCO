"""Analyze the preregistered customer-disjoint sensitivity without rerunning seven instances."""
import csv
from pathlib import Path
from itertools import product
from statistics import mean
import sys
import json
import numpy as np
HERE=Path(__file__).resolve().parent
PRIMARY=HERE/"reserved_validation";SUPP=HERE/"xtw_low_overlap"
sys.path.insert(0,str(HERE.parent/"maskco_heatmap_calibration_20261119"))
from phase_plan import read
from reserved_assess import describe,EPS
import audit_delivery

def additional_audit():
    audit_delivery.HERE=SUPP
    audit_delivery.audit("xtwsupp01")
    a=read(SUPP/"results/xtwsupp01/independent_audit.json")
    if a["status"]!="PASS" or a["endpoints"]!=117:raise ValueError("Supplementary endpoints incomplete")

def main():
    selection=read(SUPP/"selection.json");lock=read(PRIMARY/"locked_method.json");method=lock["candidate"]
    rows={};meta={};actual_cpu=0.
    for folder,phase,expected in ((PRIMARY,"reserve02",1053),(SUPP,"xtwsupp01",117)):
        cfg=read(folder/(phase+".json"));out=folder/"results"/phase
        audit=read(out/"independent_audit.json")
        if audit["status"]!="PASS" or audit["endpoints"]!=expected:raise ValueError("Independent route-level audit incomplete")
        meta.update(read(folder/"heldout_sources.json")["cases"])
        for i,b in enumerate(cfg["blocks"]):
            if not b["case"].startswith("xtw_"):continue
            rr=read(out/f"out/{i}.json")["arms"]
            pair=(b["case"],b["seed"])
            if len(rr)!=13 or pair in rows or any(r["selected_s"]>60 for r in rr.values()):raise ValueError("Unmatched XTW score")
            rows[pair]={m:r["U"] for m,r in rr.items()}
            if phase=="xtwsupp01":actual_cpu+=sum(z["cpu_s"] for z in rr.values())
    added=selection["added"];kept=selection["retained"]
    primary=sorted(c for c in meta if c.startswith("xtw_") and c not in added)
    groups={"原十例":primary,"原低重叠七例":kept,"新增三例":added,"低重叠十例":kept+added}
    if len(primary)!=10 or len(added)!=3 or len(kept)!=7 or set(rows)!=set(product(primary+added,lock["seeds"])):raise ValueError("Unregistered cases or seeds")
    scores_by={};panels={};all_methods=sorted(next(iter(rows.values())))
    with (SUPP/"paired_xtw.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["case","seed","panel"]+all_methods)
        for (case,seed),v in sorted(rows.items()):w.writerow([case,seed,"supplement" if case in added else "low_overlap_original" if case in kept else "high_overlap_original"]+[v[m] for m in all_methods])
    for label,cases in groups.items():
        bs={k:v for k,v in rows.items() if k[0] in cases}
        scores={m:describe({k:v[m] for k,v in bs.items()},meta)["mean"] for m in all_methods}
        diffs={a:describe({k:v[method]-v[a] for k,v in bs.items()},meta) for a in all_methods if a!=method}
        rank=1+sum(s>=scores[method]-EPS for a,s in scores.items() if a!=method)
        d=diffs[lock["matched"]]
        gate=(rank<=3 and d["mean"]>EPS and d["positive_seeds"]>=2 and d["wins"]*2>d["geometries"] and min(diffs[a]["mean"] for a in ("DIST","DIST_SLOT"))>EPS)
        panels[label]=dict(rank=rank,contrasts=diffs,passes=gate)
        scores_by[label]=scores
    primary_result=read(PRIMARY/"results/reserve02/analysis.json")
    if panels["原十例"]["rank"]!=primary_result["sources"]["XTW-p100"]["ranks"][method]:raise ValueError("Original rank not reproduced")
    if abs(panels["原十例"]["contrasts"][lock["matched"]]["mean"]-primary_result["sources"]["XTW-p100"]["contrasts"][method][lock["matched"]]["mean"])>EPS:raise ValueError("Primary heat contrast differs")
    primary_pass=primary_result["passes_numerical_criteria"][method]
    strict=panels["低重叠十例"]["passes"]
    output=dict(panels=panels,scores=scores_by,primary_pass=primary_pass,low_overlap_pass=strict,joint_pass=primary_pass and strict,additional_search_core_s=7020,additional_cpu_s=actual_cpu)
    same_control=lock["matched"]
    rng=np.random.default_rng(261119);draws=[];means=[]
    for src,entry in primary_result["sources"].items():
        effects=panels["低重叠十例"]["contrasts"][same_control] if src=="XTW-p100" else entry["contrasts"][method][same_control]
        geo=np.asarray(list(effects["by_geometry"].values()))
        means.append(geo.mean())
        draws.append(geo[rng.integers(len(geo),size=(20000,len(geo)))].mean(axis=1))
    output["low_overlap_three_source_mean"]=float(np.mean(means))
    output["low_overlap_three_source_ci95"]=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist()
    (SUPP/"analysis.json").write_text(json.dumps(output,indent=2,ensure_ascii=False))
    def fmt(n):return f"{n:+.2f}"
    text=["# XTW 客户坐标低重叠复核","","所有子表使用同一锁定程序和 13 法，原七个低重叠实例复用同批配对记录。表中均为几何等权净效益差。","",
          "| XTW 面板 | 名次 /13 | 对同规则距离 U | 几何簇 95% 区间 U | 改善几何 | 各种子 U | 对 DIST U | 对 DIST_SLOT U | 对 ALNS U |","|---|---:|---:|---|---:|---|---:|---:|---:|"]
    for label,g in panels.items():
        df=g["contrasts"];d=df[lock["matched"]];lo,hi=d["ci95"]
        text.append(f"| {label} | {g['rank']} | {fmt(d['mean'])} | [{fmt(lo)}, {fmt(hi)}] | {d['wins']}/{d['geometries']} | {' / '.join(fmt(v) for v in d['by_seed'].values())} | {fmt(df['DIST']['mean'])} | {fmt(df['DIST_SLOT']['mean'])} | {fmt(df['ALNS']['mean'])} |")
    text+=["","| 方法 | 低重叠十例 U | 名次 /13 |","|---|---:|---:|"]
    scores=scores_by["低重叠十例"]
    for m,u in sorted(scores.items(),key=lambda q:-q[1]):
        rank=1+sum(a!=m and v>=u-EPS for a,v in scores.items())
        text.append(f"| {m} | {u:,.2f} | {rank} |")
    attributes=read(SUPP/"attribute_overlap.json")
    lo,hi=output["low_overlap_three_source_ci95"]
    text+=["",f"按低重叠十例计算 XTW，其余来源仍用原配对，三来源等权增量为 {fmt(output['low_overlap_three_source_mean'])} U，几何重采样 95% 描述区间 [{fmt(lo)}, {fmt(hi)}] U。"]
    eligible=kept+added
    max_union=max(attributes["XTW-"+c.split("_")[1]]["union_overlap"] for c in eligible)
    text+=["",f"去掉仓库并把所有历史 XTW 的客户位置取并集后，低重叠十例中每例最多仍有 {max_union}/100 个位置出现过。n359 与 n115 的一百个客户位置完全相同，但仓库不同，按位置、需求、时窗与服务时间整行比较，两者相同的客户记录为零。","",
           f"原主面板联合门槛{'通过' if primary_pass else '未通过'}，低重叠 XTW 门槛{'通过' if strict else '未通过'}。新增 117 个终点全部独立复算，求解配额 7,020 核秒，实测求解与终点复算 CPU {actual_cpu:,.1f} 秒。","","[预先规定的坐标准入与门槛](../reserved_validation/低重叠验证.md) · [逐实例重叠](attribute_overlap.json) · [逐例逐种子得分](paired_xtw.csv)"]
    (SUPP/"RESULTS.md").write_text("\n".join(text)+"\n",encoding="utf8")
    print("\n".join(text[4:9]));print("Combined gate",output["joint_pass"])
if __name__=="__main__":additional_audit() if sys.argv[1:]==["audit"] else main()
