# -*- coding: utf-8 -*-
"""在本轮项目状态与文档索引前置一条 P5-V2 记录（只加不删，保留历史）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
STATE = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md"
INDEX = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md"

BLOCK = (
    "> **2026-11-26 P5 标准任务（CVRP/VRPTW）源码纠错、机制恢复与受控复验完成（P5-V2），"
    "P5-V1 结论归因修正。** 审计复现 V1 标准任务实现缺陷 D1–D5（排序失效/算子失效/静态 Regret-2/"
    "校准缺失/保护位未验可行）与编码偏离 F1–F5；按锁版 C++ 内核逐行重写 V2 内核与池搜索，"
    "并在修复过程中另行发现并修正 V2 自身缺陷 D6（试选缺距离选项、丢弃部分重构、SA/record 偏离）、"
    "D7（共同初解在紧时窗实例漏客致不可行）、D8（校准热图因硬编码 Windows 路径 + 静默回退**从未生效**）。"
    "受控结果：开发面板 240 端点零失败、独立复算零差异——机制恢复相对仅修缺陷 −4.14%（43/48），"
    "完整机制相对冻结 V1 −3.82%；但神经引导在标准目标下仍为负（CVRP +1.09%、VRPTW +0.98%，p<0.01），"
    "编码对齐（需求/容量、坐标中位对齐）无可测效果（p=0.20/0.87）。正式复验 3276 端点（CVRP 1800 + "
    "VRPTW 1008 + STD_NORM 468）+ 外部 PyVRP VRPTW 168 端点全部独立复算通过：CVRP Gap 6.3%–11.9%，"
    "VRPTW 车辆数在 C/R1 类系统性多于 BKS；同协议 PyVRP 同 K 子集距离差仅 +0.67（几乎贴住 BKS），"
    "故本适配不是竞争性求解器，报告如实给出 6–12 个百分点差距。已知限制：Python 内核吞吐比锁版 C++ "
    "低约两个数量级（60 s 内 epochs≈35、iterations≈700–1100），绝对成绩低估锁版机制潜力，"
    "但版本间相对结论在该预算下内部一致。冻结边界未破：P0–P4 冷链实验、锁版机制、f0_r0、P5-V1 全部未改；"
    "32 个 FLEET_PRICE 实例仍未开启；未重训、未按正式结果调参、负结果（含两批作废面板）全部保留。"
    "材料：[P5-V2 目录](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/P5_standard_vrp_v2/) · "
    "[FINAL_REPORT_V2](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/P5_standard_vrp_v2/FINAL_REPORT_V2.md) · "
    "[缺陷与修复](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/P5_standard_vrp_v2/BUGFIX_REPORT.md) · "
    "[表 V2-1…V2-10](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/P5_standard_vrp_v2/tables/) · "
    "[图 F01–F12](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/P5_standard_vrp_v2/figures/)。\n\n"
)

for path in (STATE, INDEX):
    if not path.exists():
        print("skip (missing):", path)
        continue
    text = path.read_text(encoding="utf-8")
    if "P5-V2" in text and "P5 标准任务（CVRP/VRPTW）源码纠错" in text:
        print("already recorded:", path.name)
        continue
    path.write_text(BLOCK + text, encoding="utf-8")
    print("prepended block to", path.name)
