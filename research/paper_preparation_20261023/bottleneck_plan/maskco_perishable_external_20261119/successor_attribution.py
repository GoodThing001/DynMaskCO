"""Separate calibrated depot decisions from conditional customer-successor ranking."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from directed_decoder_fit import HERE,SOURCES,assemble,design


def log_softmax(v):
    clean=v-v.max(axis=1,keepdims=True)
    return clean-np.log(np.exp(clean).sum(axis=1,keepdims=True))


def loss_parts(logits,mask,targets):
    h=logits[mask].astype(np.float64)
    h[np.arange(len(mask)),mask]=-1.e4
    lp=log_softmax(h)
    nondepot=log_softmax(h[:,1:])
    zero=targets==0
    ld=lp[:,0][:,None]
    mass=np.log(-np.expm1(np.minimum(ld,-1.e-15)))
    binary=-np.where(zero,ld,mass)
    cond=np.zeros_like(binary)
    for k in range(2):
        ids=np.flatnonzero(~zero[:,k])
        cond[ids,k]=-nondepot[ids,targets[ids,k]-1]
    return [binary.mean(),cond.mean(),zero.mean()]


def main():
    data,_=assemble();dir=HERE/"results/headcapture01"
    fits=json.loads((dir/"fit.json").read_text());by=defaultdict(lambda:defaultdict(list))
    for k in range(5):
        z=dict(np.load(dir/f"readout_fold{k}.npz"));stats=[z[n] for n in ("mu","sd","eh","pca")]
        for e in data:
            if fits["folds"][e["case"]]!=k:continue
            for m in ("physical","neural"):
                x=design(e,stats,m=="neural")
                l=e["dist"]+(x@z[m])@x.T
                by[e["src"],e["geo"],e["case"]][m].append(loss_parts(l,e["mask"],e["targets"]))
    lines=["# 回仓与客户后继的误差来源","","将出折后继交叉熵精确分解为是否回仓的二分类损失，以及继续配送时选择客户的损失。表内后项含继续配送的实际占比，两项之和等于总交叉熵。","",
           "| 来源 | 教师回仓比例 | 物理头回仓损失 | 神经头回仓损失 | 物理头客户损失 | 神经头客户损失 |","|---|---:|---:|---:|---:|---:|"]
    source={}
    for s in SOURCES:
        groups=sorted({g for src,g,c in by if src==s})
        score={m:np.mean([np.mean([np.mean(d[m],axis=0) for (ss,gg,c),d in by.items() if ss==s and gg==g],axis=0)
                       for g in groups],axis=0) for m in ("physical","neural")}
        for m in score:
            if abs(sum(score[m][:2])-fits["sources"][s]["scores"][m][0])>1.e-5:raise ValueError("Decomposition differs")
        p,h=score["physical"],score["neural"];source[s]={k:v.tolist() for k,v in score.items()}
        lines.append(f"| {s} | {p[2]:.1%} | {p[0]:.4f} | {h[0]:.4f} | {p[1]:.4f} | {h[1]:.4f} |")
    (dir/"successor_parts.json").write_text(json.dumps(source,indent=2))
    (dir/"回仓与客户后继.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":main()
