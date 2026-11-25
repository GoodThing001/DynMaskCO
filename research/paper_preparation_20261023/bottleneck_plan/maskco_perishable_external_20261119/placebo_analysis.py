"""Frozen attribution study with the same solver, training weights, computational options and decoder calls."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import json
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read,save,paired
from assess import describe
import audit_delivery

def plan():
    primary=HERE/"reserved_validation";lock=read(primary/"locked_method.json")
    if lock["candidate"]!="ORDER_SPLIT":raise ValueError("Order-and-split lock missing")
    fs=[p.name for p in sorted((primary/"solver").iterdir())]
    paths={n:"reserved_validation/solver/"+n for n in fs}
    paths["runner.py"]="development_runner.py"
    files=fs+["runner.py","heat_placebo.py","placebo_worker.py"]
    arms=["ALIGNED","SHUFFLED","ALIGNED_D"]
    cases=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    save(HERE/"placeboqual01.json",dict(worker="placebo_worker.py",purpose="training-only solver acceptance",seconds=8,files=files,file_sources=paths,
         blocks=[dict(case="r101",role="train",seed=4101,arms=arms)],block_timeout_s=300))
    save(HERE/"placebodev01.json",dict(worker="placebo_worker.py",purpose="full development",seconds=60,files=files,file_sources=paths,
         blocks=paired(cases,[4101,4102,4103],arms),block_timeout_s=480))
def report():
    phase="placebodev01";folder=HERE/"results"/phase
    audit_delivery.HERE=HERE;audit_delivery.audit(phase)
    checked=read(folder/"independent_audit.json")
    if checked["endpoints"]!=234:raise ValueError("Three-way paired matrix incomplete")
    meta=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"];q=read(HERE/(phase+".json"))
    by=defaultdict(dict);cpu=0.
    for i,b in enumerate(q["blocks"]):
        r=read(folder/f"out/{i}.json")["arms"]
        if set(r)!={"ALIGNED","SHUFFLED","ALIGNED_D"} or r["ALIGNED_D"]["forwards"]!=0:raise ValueError("Method assignment differs")
        src=meta[b["case"]]["dataset"];pair=b["case"],b["seed"]
        a,h,d=(r[m] for m in ("ALIGNED","SHUFFLED","ALIGNED_D"))
        by[src][pair]=dict(aligned_shuffled=a["U"]-h["U"],aligned_distance=a["U"]-d["U"],shuffled_distance=h["U"]-d["U"],changed=h["changed_top8"])
        cpu+=sum(z["cpu_s"] for z in r.values())
    out={s:{k:describe({pair:v[k] for pair,v in cases.items()},meta) for k in next(iter(cases.values()))} for s,cases in by.items()}
    passed=all(v["aligned_shuffled"]["mean"]>1e-6 and v["aligned_shuffled"]["positive_seeds"]>=2 and
               2*v["aligned_shuffled"]["wins"]>v["aligned_shuffled"]["geometries"] for v in out.values())
    (folder/"analysis.json").write_text(json.dumps(dict(sources=out,alignment_supported_on_all=passed,cpu_s=cpu),indent=2))
    f=lambda x:f"{x:+.2f}"
    lines=["# 距离分层后的掩码边分置乱对照","","冻结 ORDER_SPLIT，旧开发案例使用三个新种子。完整保留神经前向、热图分值的局部距离分布、可见邻接状态、掩码和两种插位算子。置乱仅改变同距离分层中分值与具体客户边的对应。三臂 234 个终点均通过独立复算。","",
           "| 来源 | 真热图 − 置乱 U | 几何重采样 95% 区间 U | 改善几何 | 三个种子 U | 真热图 − 距离 U | 置乱 − 距离 U | 置乱后前八邻居更换比例 |","|---|---:|---|---:|---|---:|---:|---:|"]
    for s in ("VRPTWQ-C2","XTW-p100","GH-p100"):
        v=out[s];x=v["aligned_shuffled"]
        ci=x["ci95"]
        interval="单一几何" if ci is None else f"[{f(ci[0])}, {f(ci[1])}]"
        lines.append(f"| {s} | {f(x['mean'])} | {interval} | {x['wins']}/{x['geometries']} | {' / '.join(f(z) for z in x['by_seed'].values())} | {f(v['aligned_distance']['mean'])} | {f(v['shuffled_distance']['mean'])} | {v['changed']['mean']:.1%} |")
    lines+=["",f"三个来源均达到边对应关系贡献门槛：{'是' if passed else '否'}。该试验属于开发机制归因。搜索配额 14,040 核秒，实测求解和终点复算 CPU {cpu:,.1f} 秒。"]
    (folder/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines[4:]))
if __name__=="__main__":plan() if sys.argv[1]=="plan" else report()
