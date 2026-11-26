# -*- coding: utf-8 -*-
"""更新 16 通道表 §2 的 STD_NORM 坐标规则（中位对齐，最终版）。"""
from pathlib import Path

TABLE = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\neural_diagnostics\16channel_table.md")
GEN = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\gen_16channel_table.py")

OLD_LINE = "- ch0–1：`(x−x0)/coord_scale`，`coord_scale = 1440 × (span/1000)`，span=客户坐标相对仓库的最大跨度，"
NEW_LINE = "- ch0–1：`(x−x0)/coord_scale`，`coord_scale = 实例坐标通道中位(原始单位)/0.0475`（0.0475 = 训练中位）；"

for p in (TABLE, GEN):
    if not p.exists():
        continue
    t = p.read_text(encoding="utf-8")
    if OLD_LINE in t:
        t = t.replace(OLD_LINE, NEW_LINE)
        t = t.replace("即把实例空间跨度映射到训练数据的跨度量级（X 集 span≈1000 → 1440 不变；Solomon span≈100 → 144；",
                      "即把实例坐标通道的分布中位对齐训练中位（X 集 → 约 5600；Solomon → 约 316；冷链实例保持 1440 不变）；")
        p.write_text(t, encoding="utf-8")
        print("patched", p.name)
    else:
        print("no match:", p.name)
