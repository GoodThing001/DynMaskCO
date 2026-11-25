"""Repeatability of the same 60-second recipe across two previously registered development runs."""
from pathlib import Path
from statistics import mean,median
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read
from assess import describe
def records(phase):
    cfg=read(HERE/(phase+".json"))
    rows={}
    for i,b in enumerate(cfg["blocks"]):
        a=read(HERE/f"results/{phase}/out/{i}.json")["arms"]
        if a["DISAGREE"]["initial_routes"]!=a["DISAGREE_D"]["initial_routes"]:raise ValueError("Initial solution differs")
        rows[b["case"],b["seed"]]=(a["DISAGREE"],a["DISAGREE_D"])
    return rows
def main():
    first=records("disscreen01");second=records("factorial01")
    if set(first)!=set(second):raise ValueError("Matched case and seed required")
    meta=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    lines=["# 同一案例和种子的重复性","","比较 3 臂筛查与 8 臂消融中相同的 DISAGREE 和 DISAGREE_D 配方。两个批次使用不同物理核和执行顺序，搜索均按单核 60 秒截止。一个配对差是同批神经终值减同规则距离终值。","",
           "| 开发来源 | 首次 / 再次热图差 U | 符号翻转（例×种子） | 两批配对差之差绝对值中位数 U |","|---|---:|---:|---:|"]
    for name in ("VRPTWQ-C2","XTW-p100","GH-p100"):
        keys=sorted(k for k in first if meta[k[0]]["dataset"]==name)
        def diff(data):return {k:data[k][0]["U"]-data[k][1]["U"] for k in keys}
        x,y=diff(first),diff(second)
        for k in keys:
            if first[k][0]["initial_routes"]!=second[k][0]["initial_routes"]:raise ValueError("Initial solution changed")
        flips=sum(np.sign(x[k])!=np.sign(y[k]) for k in keys)
        lines.append(f"| {name} | {describe(x,meta)['mean']:+.2f} / {describe(y,meta)['mean']:+.2f} | {flips}/{len(keys)} | {median(abs(y[k]-x[k]) for k in keys):.2f} |")
    lines+=["","工作量取决于墙钟截点；每轮短时试选的终点还会改变后续算子及路线选择。上述差值刻画实际重复波动，未分解硬件、时间截断和路径分叉各自的贡献。"]
    probe=HERE/"results/trialclock01/analysis.json"
    if probe.exists():
        v=read(probe)
        lines+=["",f"在四个旧训练例的 {v['contexts']} 个完全配对上下文内，只重做 0.04 秒试选，{v['changed_winner']} 个改变赢家，{v['changed_chosen_state']} 个改变传给下一阶段的路线状态。用两次中较少的迭代数重复同一试选，所有终态一致。每种插位算子单次试选的训练中位工作量为 {v['greedy_regret_caps'][0]} 次贪心或 {v['greedy_regret_caps'][1]} 次 regret-2 重构。该诊断将局部路径分叉定位到内层计时截断，整程净收益仍取决于完整求解。"]
    (HERE/"重复性对照.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":main()
