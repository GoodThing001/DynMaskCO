# -*- coding: utf-8 -*-
"""文档索引顶部插入 P4/P5 条目（幂等）。"""
from pathlib import Path

P = Path(r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\docs\文档索引与版本状态.md")
ENTRY = (
    "> **2026-11-26 P4/P5 跨目标扩展完成并推送 GitHub。** P4：Solomon 冷链 486 端点，"
    "ORDER_SPLIT 面板 1/13（2/18），对 ALNS +83.58/+59.63 U，神经热图净贡献为负。"
    "P5-A：Uchoa X 100 例，PyVRP 2.18% 全面第 1；ORDER_SPLIT_D 9.79% 优于 ALNS 10.56%，"
    "神经热图版 12.23% 无正贡献。P5-B：Solomon 56 例，ALNS 最优（9.3/1039.9），"
    "ORDER_SPLIT_D 9.5/1048.8。泄漏审计：f0_r0 训练=38 例 R/RC（rc205 除外）、验证 C2、折外 C1。"
    "材料：[FINAL_REPORT_P4_P5](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/"
    "bottleneck_plan/final_paper_experiments/FINAL_REPORT_P4_P5.md) · "
    "[交付清单](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/DELIVERY.md) · [表12–16/八图]("
    "D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/tables/)。"
)

text = P.read_text(encoding="utf-8")
if "2026-11-26 P4/P5 跨目标扩展" in text:
    print("already present")
else:
    P.write_text(ENTRY + "\n\n" + text, encoding="utf-8")
    print("prepended")
