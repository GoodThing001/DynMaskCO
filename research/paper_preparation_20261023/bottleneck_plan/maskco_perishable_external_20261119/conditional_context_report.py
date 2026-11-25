"""Compare precise-mask and whole-area rankings of the same feasible vehicle fleets."""
from collections import defaultdict,Counter
from statistics import mean
import json
import numpy as np
from mask_utility_data import HERE,read,ranked,shuffle_edges

ARMS=("distance","broad","narrow","shuffled")
OUT=HERE/"results/skeletoncontext01"

def fleet_score(routes,hidden,utility):
    vals=[]
    for r in routes:
        for a,b in zip([0]+list(r),list(r)+[0]):
            if a in hidden or b in hidden:vals.append(utility[a,b])
    if not vals:raise ValueError("No reconstructed edges")
    return mean(vals)

def main():
    cfg=read(HERE/"skeletoncontext01.json");libs={(r["case"],r["context"]):r for r in read(HERE/"conditional_choice_library.json")}
    meta=read(HERE/"heldout_sources.json");out=[];worst=0.;cpu=0.
    status=read(OUT/"status.json")
    if status["complete"]!=5 or status["failed"] or not status["end"]:raise ValueError("Incomplete cohort")
    for ci,b in enumerate(cfg["blocks"]):
        c=b["case"];rep=read(OUT/f"out/{ci}.json")
        if rep["case"]!=c or rep["contexts"]!=len(b["regions"]) or rep["forwards"]!=2*rep["contexts"] or rep["max_error"]>.002:raise ValueError("Case or output mismatch")
        worst=max(worst,rep["max_error"]);cpu+=rep["cpu_s"]
        nodes=np.array(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2)
        with np.load(OUT/f"out/{ci}.npz") as x:
            for ri,z in enumerate(b["regions"]):
                case=libs[c,ri];hidden=set(z["hidden"]);narrow=x[f"{ri}_narrow"]
                if hidden!=set(case["hidden"]) or case["state"]!=z["state"]:raise ValueError("Economic alternatives mismatched")
                raw=dict(distance=-d,broad=x[f"{ri}_broad"],narrow=narrow,shuffled=shuffle_edges(narrow,d,np.random.default_rng(7773001+ci*128+ri)))
                arms={};choices=case["choices"]
                for a,mat in raw.items():
                    ranks=ranked(mat);sym=(ranks+ranks.T)*.5
                    scores=[fleet_score(v["routes"],hidden,sym) for v in choices];chosen=int(np.argmax(scores))
                    arms[a]=dict(gain=choices[chosen]["gain"],selected=chosen)
                out.append(dict(case=c,state=z["state"],context=ri,masks=sorted(hidden),arms=arms,available=len(choices),max_U=max(t["gain"] for t in choices)))
    if len(out)!=len(libs):raise ValueError("Incomplete paired results")
    levels={}
    for state in ("initial","terminal"):
        rows=[r for r in out if r["state"]==state];bygeom={}
        for c in sorted({r["case"] for r in rows}):
            group=[r for r in rows if r["case"]==c]
            bygeom[c]={a:mean(r["arms"][a]["gain"] for r in group) for a in ARMS}|dict(max_U=mean(r["max_U"] for r in group))
        m={a:mean(v[a] for v in bygeom.values()) for a in list(ARMS)+["max_U"]}
        levels[state]=dict(means=m,counts=len(rows),geometries=len(bygeom),by_geometry=bygeom)
    passed=all(v["means"]["narrow"]>v["means"][a]+1.e-6 for v in levels.values() for a in ("distance","broad")) and mean(v["means"]["narrow"]-v["means"]["shuffled"] for v in levels.values())>1.e-6
    report=dict(levels=levels,rows=out,passed=passed,cpu_s=cpu,max_logit_error=worst,forwards=2*len(out))
    (OUT/"analysis.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    lines=["# 显示可见路线骨架后，热图能否选出更好的换车解","","从既有车队中取同一掩码下可实现的替代解。此表限于至少有两个不同正净收益的上下文，每种边分从同一批车队中选一个，按几何等权。数值是相对父解的净效益 U。","",
        "| 起点 | 上下文 | 可比较几何 | 距离 | 整区热图 | 保留骨架热图 | 置乱 | 已知最好 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for state,v in levels.items():
        m=v["means"];lines.append(f"| {state} | {v['counts']} | {v['geometries']}/5 | "+" | ".join(f"{m[a]:.3f}" for a in list(ARMS)+["max_U"])+" |")
    lines+=["",f"共 {len(out)*2} 次配对前向，累计 CPU {cpu:.1f} 秒。转入在线候选生成的门槛"+("通过。" if passed else "未通过。"),
            "","[可比较掩码的母集](../utilitycapture01/conditional_opportunity.json) · [逐掩码结果](analysis.json) · [固定方案](../../保留可见骨架的条件掩码.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":main()
