"""Decompose the paired fleet profit of the screened decoder on each Danish geometry."""
from statistics import mean
from collections import defaultdict
import json
from pathlib import Path
from prepare_panel import core

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf8"))

def main():
    folder=HERE/"results/mixscreen01";plan=read(HERE/"mixscreen01.json")
    if read(folder/"analysis.json")["total"]!=132:raise ValueError("Incomplete matched screen")
    rows=defaultdict(list);money=core.load_config()["money"]["distance_U_per_km"]
    for ix,b in enumerate(plan["blocks"]):
        if not b["case"].startswith("dk_"):continue
        q=read(folder/f"out/{ix}.json")["arms"];h,d=q["HEAD"],q["ORDER_SPLIT_D"]
        a,b0=h["endpoint"],d["endpoint"]
        losses=[b0["quality_loss_value_U"]-a["quality_loss_value_U"],money*(b0["distance_km"]-a["distance_km"]),b0["refrigeration_cost_U"]-a["refrigeration_cost_U"]]
        if abs(sum(losses)-(h["U"]-d["U"]))>1.e-6:raise ValueError("Net contribution mismatch")
        rows[b["case"]].append(dict(seed=b["seed"],U=h["U"]-d["U"],quality=losses[0],distance=losses[1],cold=losses[2],
            fleet=a["used_vehicles"]-b0["used_vehicles"],forwards=h["forwards"],forward_s=h["forward_s"],iterations_H=h["iterations"],iterations_D=d["iterations"]))
    lines=["# 丹麦五个几何的 60 秒净效益来源","","两种子等权。各成本列为同规则距离的成本减去 MASKCO 版成本，正值表示 MASKCO 节省成本，三列之和为净效益差。车数差为 MASKCO 减距离。","",
        "| 丹麦几何 | 种子 46001 / 46002 差 U | 平均差 U | 品质损失贡献 U | 里程贡献 U | 制冷贡献 U | 平均车数差 |","|---|---|---:|---:|---:|---:|---:|"]
    for c,v in sorted(rows.items()):
        z={k:mean(r[k] for r in v) for k in v[0]}
        lines.append("| "+c+" | "+" / ".join(f"{r['U']:+.2f}" for r in sorted(v,key=lambda r:r["seed"]))+" | "+" | ".join(f"{z[k]:+.2f}" for k in ("U","quality","distance","cold","fleet"))+" |")
    agg=lambda k:mean(mean(r[k] for r in v) for v in rows.values())
    lines+=["",f"平均每次搜索执行 {agg('forwards'):.1f} 次热图前向，热图、物理特征和解码合计占用 {agg('forward_s'):.2f} 秒。MASKCO 与同规则距离各执行 {agg('iterations_H'):,.0f} 和 {agg('iterations_D'):,.0f} 次重构。"]
    (folder/"丹麦差额.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    (folder/"danish_economics.json").write_text(json.dumps(rows,indent=2))
    print("\n".join(lines))

if __name__=="__main__":main()
