"""Evaluate the conditional-customer mixture on all archived geometry-held-out predictions."""
from collections import defaultdict
import json
import numpy as np
from directed_decoder_fit import HERE,SOURCES,assemble,design,measure
from customer_mixture import combine,probabilities


def main():
    rows,_=assemble()
    path=HERE/"results/headcapture01"
    old=json.loads((path/"fit.json").read_text())
    values=defaultdict(lambda:defaultdict(list))
    for fold in range(5):
        with np.load(path/f"readout_fold{fold}.npz") as z:
            stats=[z[n] for n in ("mu","sd","eh","pca")]
            for i,row in enumerate(rows):
                if old["folds"][row["case"]]!=fold:continue
                score={}
                for arm in ("physical","neural"):
                    x=design(row,stats,arm=="neural")
                    score[arm]=row["dist"]+(x@z[arm])@x.T
                p=score["physical"]
                score["mixture"]=combine(p,score["neural"])
                score["shuffled"]=combine(p,score["neural"],row["dist"],50201+i)
                truth=probabilities(p.astype(np.float64))
                for arm in ("mixture","shuffled"):
                    test=np.exp(score[arm])
                    if max(np.max(abs(test[:,0]-truth[:,0])),np.max(abs(test.sum(axis=1)-1)))>1.e-8:
                        raise ValueError("Depot probability or row normalization differs")
                if i==0 and np.max(abs(np.exp(combine(p,p))-truth))>1.e-8:raise ValueError("Physical identity")
                for arm,v in score.items():
                    values[row["src"],row["geo"],row["case"]][arm].append(measure(v,row["mask"],row["targets"]))
    results={};passed=True;placebo=0;recall=0
    lines=["# 保留回仓概率的客户后继混合","","固定同一物理模型的回仓概率。每格是五折留几何的后继交叉熵或前八命中，三种模型的客户预测使用同一批独立种子教师。","",
           "| 来源 | 纯物理损失 | 客户混合损失 | 置乱损失 | 纯物理前八 | 客户混合前八 | 置乱前八 |","|---|---:|---:|---:|---:|---:|---:|"]
    for src in SOURCES:
        geos=sorted({g for s,g,c in values if s==src})
        by_geo={g:{a:np.mean([np.mean(v[a],axis=0) for (s,gg,c),v in values.items() if s==src and gg==g],axis=0).tolist()
                   for a in ("physical","neural","mixture","shuffled")} for g in geos}
        avg={a:np.mean([r[a] for r in by_geo.values()],axis=0).tolist() for a in next(iter(by_geo.values()))}
        prior=old["sources"][src]["scores"]
        if np.max(abs(np.array(avg["physical"])-prior["physical"]))>1.e-6:raise ValueError("Reference mismatch")
        ph,m,sh=[avg[a] for a in ("physical","mixture","shuffled")]
        passed &= m[0]<ph[0] and m[1]>=prior["distance"][1]
        placebo+=int(m[0]<sh[0]);recall+=int(m[1]>prior["frozen"][1])
        results[src]=dict(scores=avg,by_geometry=by_geo)
        lines.append(f"| {src} | {ph[0]:.4f} | {m[0]:.4f} | {sh[0]:.4f} | {ph[1]:.2%} | {m[1]:.2%} | {sh[1]:.2%} |")
    passed=bool(passed and placebo>=2 and recall>=2)
    lines+=["","进入完整 60 秒闭环开发的门槛"+("通过。" if passed else "未通过。"),"","[预设方法和晋级条件](../../回仓与客户分开解码.md)"]
    (path/"customer_mixture.json").write_text(json.dumps(dict(sources=results,passed=passed),indent=2))
    (path/"客户混合.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))

if __name__=="__main__":main()
