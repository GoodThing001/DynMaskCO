"""Leave-one-geometry-out benefit ranking with one common, pretrained masked representation."""
import json
from statistics import mean
import numpy as np
from scipy.stats import rankdata
from mask_utility_data import HERE,OUT,read
ARMS={"distance":"d","physical":"dp","neural":"dph","shuffled":"dps"}

def design(r,keys):
    x=np.array([sum([row[k] for k in keys],[]) for row in r["candidates"]],dtype=float)
    return x-x.mean(axis=0)

def predict(training,evaluation,keys):
    x=[];y=[];weights=[]
    for r in training:
        cand=r["candidates"];X=design(r,keys);g=np.array([t["gain"] for t in cand])
        z=(rankdata(np.round(g,6))-.5)/len(g)-.5
        x.append(X);y.append(z);weights.extend([1./(len(training)*len(g))]*len(g))
    X=np.concatenate(x);Y=np.concatenate(y);wt=np.asarray(weights)
    sd=np.maximum(np.sqrt(wt@(X**2)),1.e-5);X/=sd
    beta=np.linalg.solve((X.T*wt)@X+.1*np.eye(X.shape[1]),(Y*wt)@X)
    return [(design(r,keys)/sd)@beta for r in evaluation]

def main():
    studies=[read(OUT/f"utility_{i}.json") for i in range(5)]
    allrows=[s for c in studies for s in c["rows"]];test=[r for r in allrows if r["phase"]=="utilitycapture01"];old=[r for r in allrows if r["phase"]=="jointcapture01"]
    if len(test)!=20 or len(old)!=50 or len({r["case"] for r in old})!=5:raise ValueError("Incomplete matrix")
    result=[];dupes=0
    def identity(r):return (r["case"],r["state"],tuple(sorted(map(tuple,r["parts"]))))
    prior={identity(r) for r in old};seen=set(prior)
    for c in sorted({r["case"] for r in test}):
        train=[r for r in old if r["case"]!=c];assess=[r for r in test if r["case"]==c]
        if len(train)!=40 or len(assess)!=4:raise ValueError("Outer split identity")
        out=[{} for _ in assess]
        for a,keys in ARMS.items():
            for m,(r,scores) in enumerate(zip(assess,predict(train,assess,keys))):
                idx=np.argsort(-scores,kind="stable")[:2];cand=r["candidates"]
                out[m][a]=dict(chosen=idx.tolist(),gain=max(cand[i]["gain"] for i in idx))
        for r,arms in zip(assess,out):
            ident=identity(r);repeat=ident in seen;seen.add(ident);dupes+=int(repeat)
            result.append(dict(case=c,state=r["state"],rep=r["rep"],seen_region=repeat,arms=arms,
                    attainable=max(t["gain"] for t in r["candidates"]),pool=len(r["candidates"])))
    def metrics(rows):
        return {a:mean(r["arms"][a]["gain"] for r in rows) for a in ARMS}|dict(attainable=mean(r["attainable"] for r in rows))
    src={s:metrics([r for r in result if r["state"]==s]) for s in ("initial","terminal")}
    overall=metrics(result);bygeom={c:metrics([r for r in result if r["case"]==c]) for c in sorted({r["case"] for r in result})}
    wins=sum(m["neural"]>m["distance"]+1.e-6 for m in bygeom.values())
    passed=wins>=3 and all(v["neural"]>v["distance"]+1.e-6 for v in src.values()) and all(overall["neural"]>overall[a]+1.e-6 for a in ARMS if a!="neural")
    cand=[t for r in allrows for t in r["candidates"]];opt=sum(v["optimal"] for v in cand)
    cpu=sum(s["cpu_s"] for s in studies)
    stats=dict(by_state=src,overall=overall,geometries=bygeom,wins=wins,passed=passed,audits=len(cand),optimal=opt,cpu_s=cpu,duplicates=dupes,rows=result)
    (OUT/"analysis.json").write_text(json.dumps(stats,indent=2),encoding="utf8")
    lines=["# 用受审计的换车收益选择可移动客户","","候选掩码由同一组距离、载重与随机规则生成。每个几何用其他四个几何的 40 个原区域拟合相同岭系数 0.1，评价本几何两个新增锚点、两种起点。每行给出每区选两次后的可行净增益均值，单位 U。","",
        "| 开发评价 | 仅距离 | 距离加物理 | 再加 MASKCO | 再加置乱 | 试遍候选池 | 神经减物理 |","|---|---:|---:|---:|---:|---:|---:|"]
    for label,m in list(src.items())+[("合计",overall)]:
        lines.append("| "+label+" | "+" | ".join(f"{m[a]:.3f}" for a in list(ARMS)+["attainable"])+f" | {m['neural']-m['physical']:+.3f} |")
    lines+=["",f"热图在 {wins}/5 个开发几何优于仅距离特征。三来源开发门槛"+("通过。" if passed else "未通过。"),
            f"20 个开发评价锚点中，{dupes} 个对应的路线区域与本几何先前区域或本次前序锚点相同。训练与评价几何相互独立。",
            "",f"训练及评价共独立审计 {len(cand)} 个候选车队，整数求解证实候选池最优 {opt} 个。五个几何累计 CPU {cpu:.1f} 秒。","",
            "[固定方案](../../整组掩码经济排序.md) · [逐几何与选中候选](analysis.json)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":main()
