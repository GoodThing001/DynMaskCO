# -*- coding: utf-8 -*-
"""在项目状态与文档索引前置"P5/P5-V2 冻结归档"记录（只加不删）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
STATE = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md"
INDEX = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md"

BLOCK = (
    "> **2026-11-26 P5 / P5-V2 实验终止迭代并完整归档（冻结）。** 标准任务 V1（`P5_standard_vrp`）"
    "与 V2（`P5_standard_vrp_v2`）自本日起不再新增端点、不补跑、不调参、不重训；两棵目录树逐文件 "
    "SHA256 封存于仓库外 `archive/p5_frozen_20261126/`：归档包 `P5_P5V2_FROZEN_20261126.tar.gz`"
    "（11,678,941 字节，SHA256 `77478c5a…98068`）、清单 `MANIFEST.json`（26,280 文件，"
    "SHA256 `ccc57f48…605b86`）、`INDEX.csv`、`ARCHIVE.md`、`ARCHIVE_RECEIPT.json`、"
    "`VERIFY_RESULT.json`（解包自校验：0 缺失、0 不符）、`SHA256SUMS.txt`。"
    "清点：`P5_standard_vrp` 14,560 文件/38.46 MB、`P5_standard_vrp_v2` 11,720 文件/38.46 MB。"
    "冻结前有效端点 3,684（开发 240 + 正式 CVRP 1800 + VRPTW 1008 + STD_NORM 468 + 外部 PyVRP 168，"
    "另诊断 48 单元），全部零失败、逐端点独立复算通过；作废面板 `dev01_prefix_D6`、`dev01_pre_D8` "
    "作为缺陷证据保留、不参与汇总。远端无残留作业，全部产物（含 3,740 个逐端点日志）已下载至本地；"
    "远端 `/tmp` 为易失分区，不再作为存档依赖。冻结边界不变：P0–P4 冷链实验、锁版 ORDER_SPLIT 求解器"
    "（89/89 SHA 一致）、`f0_r0`、P5-V1 源与结果、32 个 FLEET_PRICE 实例均未改动/未开启。"
    "已知未完成项（本轮不再推进）：C++ 内核效率对齐、NDS VRPTW 接入（外部基线为 168 而非 336 端点）。"
    "后续如需迭代必须新建目录与新身份（如 `P5_standard_vrp_v3`）并重新登记协议，不得复用本轮身份或覆盖结果。"
    "材料：[P5-V2 目录](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/P5_standard_vrp_v2/) · "
    "[冻结通知](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/P5_standard_vrp_v2/ARCHIVE_NOTICE.md) · "
    "[最终报告](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/"
    "final_paper_experiments/P5_standard_vrp_v2/FINAL_REPORT_V2.md) · "
    "归档说明：`archive/p5_frozen_20261126/ARCHIVE.md`。\n\n"
)

for path in (STATE, INDEX):
    if not path.exists():
        print("skip (missing):", path)
        continue
    text = path.read_text(encoding="utf-8")
    if "P5 / P5-V2 实验终止迭代并完整归档" in text:
        print("already recorded:", path.name)
        continue
    path.write_text(BLOCK + text, encoding="utf-8")
    print("prepended freeze record to", path.name)
