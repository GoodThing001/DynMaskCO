# -*- coding: utf-8 -*-
"""在项目当前状态.md 顶部插入 P4/P5 条目（幂等）。"""
from pathlib import Path

P = Path(r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\项目当前状态.md")
ENTRY = (
    "> **2026-11-26 P4/P5 跨目标扩展完成（P4 Solomon 冷链 486 端点、P5-A CVRP 2716 格、"
    "P5-B VRPTW 1344 格 + 外部方法批次），全部独立审计通过。** P4：ORDER_SPLIT 在 Solomon 冷链 "
    "C1/C2 面板 1/13（2/18），对 ALNS +83.58/+59.63 U，但神经热图净贡献为负（对 ORDER_SPLIT_D −5.32 U）。"
    "P5-A（Uchoa X 100 例，距离目标）：PyVRP 2.18% Gap 全面第 1；本文族 ORDER_SPLIT_D 9.79% 优于 "
    "ALNS 10.56%，神经热图版 ORDER_SPLIT 12.23% 劣于距离版（热图无正贡献，与 P4 一致）。"
    "P5-B（Solomon 56 例，字典序 K,D）：ALNS 最优（K=9.3/D=1039.9），ORDER_SPLIT_D 9.5/1048.8，"
    "ORDER_SPLIT 10.2/1091.6。数据泄漏审计：f0_r0 训练=38 例 R/RC（rc205 除外），验证 C2、折外 C1，"
    "逐例标注。8 个作者模型（AM/POMO/DeepACO/CaR/MVMoE/CaDA/RouteFinder/RRNCO）按作者默认推理接入；"
    "RRNCO 无 CVRP 权重、AM/POMO 等无 VRPTW 权重按预登记 UNSUPPORTED。材料见 "
    "[FINAL_REPORT_P4_P5](../research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/"
    "FINAL_REPORT_P4_P5.md) 与 [交付清单](../research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/DELIVERY.md)。"
)

text = P.read_text(encoding="utf-8")
if "2026-11-26 P4/P5" in text:
    print("already present")
else:
    lines = text.splitlines()
    out = [ENTRY, ""] + lines
    P.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("prepended, lines:", len(out))
