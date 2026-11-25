"""Prepare and assess time-matched, out-of-geometry decoder trials."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import json
import sys

HERE=Path(__file__).resolve().parent
ARMS=["HEAD","PHYSICAL","ORDER_SPLIT","ORDER_SPLIT_D","DIST","DIST_SLOT"]
SOURCES=["PDPCDPG-D100","XTW-p100","GH-p100"]


def read(p):return json.loads(p.read_text(encoding="utf8"))


def prepare(mixture):
    data=read(HERE/"results/headcapture01/fit.json")
    folders=HERE/"results/headcapture01"
    approved=read(folders/"customer_mixture.json")["passed"] if mixture else data["advance"]
    if not approved:raise ValueError("Decoder did not qualify")
    (folders/"readout_folds.json").write_text(json.dumps(data["folds"],indent=2))
    old=read(HERE/"opprobe01.json")
    cases=sorted(data["folds"])
    screen=sorted([c for c in cases if c.startswith("dk_")]+["xtw_n298_p100","xtw_n308_p100","xtw_n322_p100","gh_c1_6_1_p100","gh_r1_6_1_p100","gh_rc1_6_1_p100"])
    if len(screen)!=11 or not set(screen)<=set(cases):raise ValueError("Registered examples unavailable")
    prefix="mix" if mixture else "head"
    for suffix,used,seeds,arms,seconds in (("path01",[screen[0]],[9072],ARMS[:4],6),
            ("screen01",screen,[46001,46002],ARMS,60),("dev01",cases,[46101,46102,46103],ARMS,60)):
        stage=prefix+suffix
        files=[x for x in old["files"] if x not in ("paired_operator_probe.py","archived_states.json")]
        extra=["directed_decoder_features.py","directed_decoder_fit.py","directed_decoder_search.py",
               "heldout_admission.json","locked_method.json","readout_folds.json"]+[f"readout_fold{k}.npz" for k in range(5)]
        if mixture:extra.append("customer_mixture.py")
        paths=dict(old["file_sources"])
        for f in extra:
            if f.startswith("readout_fold"):paths[f]="results/headcapture01/"+f
        paths["locked_method.json"]="reserved_validation/locked_method.json"
        cfg=dict(worker="directed_decoder_search.py",training_only=False,customer_mix=mixture,purpose="cross-geometry full search",seconds=seconds,
                 block_timeout_s=900,files=files+extra,file_sources=paths,
                 blocks=[dict(case=c,seed=k,arms=arms) for c in used for k in seeds])
        if any(not (HERE/paths.get(n,n)).is_file() for n in cfg["files"]):raise ValueError("Worker dependency missing")
        (HERE/(stage+".json")).write_text(json.dumps(cfg,indent=2))


def report(stage):
    cfg=read(HERE/(stage+".json"));base=HERE/"results"/stage;status=read(base/"status.json")
    if status["failed"] or status["complete"]!=len(cfg["blocks"]) or not status["end"]:raise ValueError(status)
    meta=read(HERE/"heldout_sources.json")["cases"]
    rows={};cpu=0.;budget=0;fw=0
    for ix,b in enumerate(cfg["blocks"]):
        q=read(base/f"out/{ix}.json");c,k=b["case"],b["seed"]
        if c!=q["case"] or k!=q["seed"] or set(q["arms"])!=set(ARMS):raise ValueError("Incomplete arms")
        for m,r in q["arms"].items():
            if (r["status"]!="AUDITED" or r["selected_s"]>60 or not r["endpoint"]["independently_audited"] or
                abs(r["U"]-r["endpoint"]["net_benefit_U"])>1e-6 or (c,k,m) in rows):raise ValueError("Endpoint not verified")
            rows[c,k,m]=r["U"];cpu+=r["cpu_s"];budget+=60;fw+=int(r["forwards"] or 0)
    out={};passed=True
    lines=["# 有向掩码头的 60 秒留几何开发","","所有方法使用同一初解、种子和完整搜索时钟。候选的权重只从其他几何学习。各列为 HEAD 相对该参照的几何等权净效益差 U。","",
           "| 来源 | 同规则距离 | 改善几何 | 逐种子 U | 物理输出 | 原 ORDER_SPLIT | DIST | DIST_SLOT |","|---|---:|---:|---|---:|---:|---:|---:|"]
    for s in SOURCES:
        cases=sorted({c for c,k,m in rows if meta[c]["dataset"]==s})
        seeds=sorted({k for c,k,m in rows if c in cases})
        groups=defaultdict(list)
        for c in cases:groups[meta[c]["geometry"]].append(c)
        src={}
        for m in ARMS[1:]:
            g={geo:mean(rows[c,k,"HEAD"]-rows[c,k,m] for c in cs for k in seeds) for geo,cs in groups.items()}
            bs={k:mean(mean(rows[c,k,"HEAD"]-rows[c,k,m] for c in cs) for cs in groups.values()) for k in seeds}
            src[m]=dict(mean=mean(g.values()),geometries=g,by_seed=bs,wins=sum(v>1.e-6 for v in g.values()))
        out[s]=src
        matched=src["ORDER_SPLIT_D"]
        passed &= matched["mean"]>1.e-6 and matched["wins"]*2>len(groups)
        if stage.endswith("dev01"):
            passed &= all(src[m]["mean"]>1.e-6 and sum(v>1.e-6 for v in src[m]["by_seed"].values())>=2 for m in ("ORDER_SPLIT_D","DIST","DIST_SLOT"))
        lines.append(f"| {s} | {matched['mean']:+.2f} | {matched['wins']}/{len(groups)} | "+
                     " / ".join(f"{v:+.2f}" for v in matched["by_seed"].values())+" | "+
                     " | ".join(f"{src[m]['mean']:+.2f}" for m in ("PHYSICAL","ORDER_SPLIT","DIST","DIST_SLOT"))+" |")
    passed &= mean(out[s]["PHYSICAL"]["mean"] for s in SOURCES)>1.e-6
    if stage.endswith("screen01"):passed &= mean(out[s]["ORDER_SPLIT"]["mean"] for s in SOURCES)>1.e-6
    lines+=["",f"{len(rows)} 个独立终点复算，搜索配额 {budget:,} 核秒；实测含复算 CPU {cpu:,.1f} 秒。晋级门槛"+("通过。" if passed else "未通过。"),"","[开发门槛](../../有向掩码解码器开发.md)"]
    if cfg.get("customer_mix"):
        lines[0]="# 固定回仓概率的 MASKCO 掩码重构与 60 秒净效益"
        lines[-1]="[开发门槛](../../回仓与客户分开解码.md)"
    (base/"analysis.json").write_text(json.dumps(dict(sources=out,passed=passed,total=len(rows),cpu_s=cpu,search_budget_s=budget,forwards=fw),indent=2))
    (base/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":
    if sys.argv[1] in ("plan","mixplan"):prepare(sys.argv[1]=="mixplan")
    else:report(sys.argv[1])
