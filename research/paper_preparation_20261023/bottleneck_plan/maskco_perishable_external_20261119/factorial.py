"""Quantify economic recombination and disagreement masking on the original development screen."""
from pathlib import Path
from statistics import mean
import json
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read,save,paired
from assess import describe

def plan():
    old=read(HERE/"disscreen01.json")
    cases={b["case"] for b in old["blocks"]}
    ops=["HYBRID","MASK_ONLY","CROSS_ONLY","DISAGREE"]
    arms=[m+suffix for m in ops for suffix in ("","_D")]
    paths=dict(old["file_sources"])
    for file in old["files"]:
        if file!="runner.py":paths[file]="development_lock/disscreen01/"+file
    paths["runner.py"]="development_runner.py"
    cfg=dict(old,worker="ablation_search.py",files=[f for f in old["files"] if f!="cached_search.py"]+["ablation_search.py"],file_sources=paths)
    save(HERE/"facqual01.json",dict(cfg,seconds=8,purpose="training-only component ablation acceptance",block_timeout_s=350,
          blocks=[dict(case="r101",role="train",seed=1000,arms=arms)]))
    save(HERE/"factorial01.json",dict(cfg,seconds=60,purpose="development screening",block_timeout_s=780,blocks=paired(cases,[1000,1001],arms)))

def analyze():
    import cache_assess
    cache_assess.main("factorial01")
    p=read(HERE/"factorial01.json");folder=HERE/"results/factorial01"
    meta=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    observations={}
    for i,b in enumerate(p["blocks"]):
        r=read(folder/f"out/{i}.json")["arms"]
        obs={}
        for suffix in ("","_D"):
            def value(m):return r[m+suffix]["U"]
            a,h,c,both=map(value,("HYBRID","MASK_ONLY","CROSS_ONLY","DISAGREE"))
            obs["mask_effect"+suffix]=.5*((h-a)+(both-c))
            obs["cross_effect"+suffix]=.5*((c-a)+(both-h))
            obs["interaction"+suffix]=both-c-h+a
        for m in ("HYBRID","MASK_ONLY","CROSS_ONLY","DISAGREE"):
            obs[m+"_heat"]=r[m]["U"]-r[m+"_D"]["U"]
        observations[b["case"],b["seed"]]=obs
    results={}
    for src in sorted({meta[c]["dataset"] for c,s in observations}):
        results[src]={}
        for k in next(iter(observations.values())):
            x={(c,s):v[k] for (c,s),v in observations.items() if meta[c]["dataset"]==src}
            results[src][k]=describe(x,meta)
    (folder/"factorial_effects.json").write_text(json.dumps(results,indent=2))
    lines=["# 旧开发面板的掩码与路线组合消融","","8 例、2 个种子、8 个方法臂共 128 个同批单核 60 秒终点。下表为 U，先种子、后几何等权。","",
           "| 来源 | 原热图增量 | 仅分歧掩码的热图增量 | 仅组合路线的热图增量 | 完整方法的热图增量 |","|---|---:|---:|---:|---:|"]
    for s,d in results.items():lines.append("| "+s+" | "+" | ".join(f"{d[m+'_heat']['mean']:+.2f}" for m in ("HYBRID","MASK_ONLY","CROSS_ONLY","DISAGREE"))+" |")
    lines+=["","| 来源 | 神经侧掩码主效应 | 神经侧组合主效应 | 神经侧交互 | 距离侧掩码主效应 | 距离侧组合主效应 | 距离侧交互 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for s,d in results.items():lines.append("| "+s+" | "+" | ".join(f"{d[k+suffix]['mean']:+.2f}" for suffix in ("","_D") for k in ("mask_effect","cross_effect","interaction"))+" |")
    lines+=["","主效应分别取另一因素开、关两种条件的平均配对效应。交互为两个因素同时开启时超出两项单独效应之和的差额。"]
    (folder/"效应分解.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":(plan if sys.argv[1]=="plan" else analyze)()
