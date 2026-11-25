"""Find the smallest customer mask that permits an audited change of fleet routes."""
from functools import lru_cache
from scipy.optimize import linear_sum_assignment
import numpy as np
from route_columns import DEST,assemble,read


def common_order(a,b):
    @lru_cache(None)
    def lcs(i,j):
        if i==len(a) or j==len(b):return ()
        if a[i]==b[j]:return (a[i],)+lcs(i+1,j+1)
        return max(lcs(i+1,j),lcs(i,j+1),key=len)
    return lcs(0,0)


def support(old,new,already=()):
    given=set(already)
    customers={v for r in old for v in r}-given
    if customers!={v for r in new for v in r}-given:raise ValueError("Different regional customers")
    pairs=[[common_order(tuple(v for v in a if v not in given),tuple(v for v in b if v not in given)) for b in new] for a in old]
    i,j=linear_sum_assignment(-np.array([[len(v) for v in row] for row in pairs]))
    kept=set(v for a,b in zip(i,j) for v in pairs[a][b])
    return sorted(customers-kept)


def main():
    import json
    examples,_=assemble();out=[]
    lines=["# 还原已知有利配车至少需要同时遮去几客","","用二分图匹配决定新旧路线的对应关系，在各配对中保持可见客户的访问次序。表内前两列数量均为允许另开路线、最小化新增遮蔽客户数后的精确下界。","",
           "| 丹麦区域 | 净增 U | 至少同步移动客户数 | 原 40 客掩码之外仍需移动 | 六个旧掩码能覆盖目标的个数 |","|---|---:|---:|---:|---:|"]
    for r in read(DEST/"full_opportunity.json")["rows"]:
        if r["gain"]<=1.e-6:continue
        c=r["case"];old,new=r["parts"],r["winner"]
        masks=[e["mask"] for e in examples if e["case"]==c][6:]
        add=[support(old,new,m) for m in masks];minimum=support(old,new)
        out.append(dict(case=c,group=r["group"],rep=r["rep"],gain=r["gain"],minimum_mask=minimum,missing_by_mask=add))
        lines.append(f"| {c[-1]} / {r['group']} / {r['rep']+1} | {r['gain']:.2f} | {len(minimum)} | {len(add[r['rep']])} | {sum(not a for a in add)}/6 |")
    (DEST/"support.json").write_text(json.dumps(out,indent=2))
    (DEST/"最小联合掩码.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":main()
