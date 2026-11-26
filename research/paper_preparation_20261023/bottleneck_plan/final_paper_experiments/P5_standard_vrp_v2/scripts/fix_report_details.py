# -*- coding: utf-8 -*-
"""报告细节修正 + 新增 §4.1（本轮在 V2 自身发现并修复的缺陷 D6–D8）。"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "FINAL_REPORT_V2.md"
t = p.read_text(encoding="utf-8")
fixes = [
    ("> 数据截止：(见 §8 运行记录)。", "> 数据截止：2026-11-26（UTC，正式面板 cvrp/vrptw/norm 全部结束）。"),
    ("- 新增硬截止 `started+seconds−0.05`，保证 Python 内核单次重构超时不再溢出 60 s 在线预算。",
     "- 新增内部分截止 `deadline = seconds − max(0.8, 1.0+0.0015n)` 与硬截止 `deadline − 0.3`，"
     "并在候选槽位粒度插入守卫（每 8 个槽检查一次），保证 Python 内核的单步不可分割操作不再溢出 60 s。"),
    ("表见 `development_results/dev01/tables_dev.md`", "表见 `development_results/dev01_final/tables_dev.md`"),
    ("all 89 个冻结源逐文件 SHA 复核", "全部 89 个冻结源逐文件 SHA 复核"),
    ("- 测试：`standard_vrp/tests` 共 31 项（含 D1–D5 复现、CVRP/VRPTW 固定序列 Split 与暴力枚举对照、\n"
     "  bundle 边界、related/SA、n=200 双变体预算遵守、历史快照键）。",
     "- 测试：`standard_vrp/tests` 共 31 项（含 D1–D5 复现、CVRP/VRPTW 固定序列 Split 与暴力枚举对照、\n"
     "  bundle 边界、related/SA、n=200/n=979 预算遵守、历史快照键）。\n"
     "\n"
     "### 4.1 本轮在 V2 自身发现并修复的缺陷（D6–D8）\n"
     "\n"
     "修复过程中三处**新缺陷**是通过开发面板与正式面板的异常信号定位的，全部按锁版源码修正：\n"
     "\n"
     "| 编号 | 问题 | 发现方式 | 修复 |\n"
     "|---|---|---|---|\n"
     "| D6a–c | 六配置试选缺\"纯距离\"选项、超时丢弃部分重构、SA/record 与锁版不一致 | 开发面板 `ranking_counts` 恒为 `[N,0,0]`，且 STD_BUGFIX 系统性优于 STD_FIX | 逐行对齐锁版 `pool_search.py` L192–229 与 `masked_repair.cpp` |\n"
     "| D7 | 共同初解在紧时窗实例上漏客/超车队上限，导致初解不可行 | VRPTW 阶段所有版本端点 INFEASIBLE | 重写 `nn_witness`（覆盖全部客户 + 车队压缩），开发实例车队上限对齐 Solomon 的 25 |\n"
     "| D8 | 校准热图因硬编码 Windows 路径 + 静默 `except` **从未生效** | 开发面板 `ranking_counts[2]` 恒为 0；诊断脚本首次运行时崩在路径上 | 锁版 `calibrated.json` 入包 + 三源解析 + 缺失即报错；校准异常留痕 |\n"
     "\n"
     "三处缺陷的作废结果分别保留在 `development_results/dev01_prefix_D6/` 与 `dev01_pre_D8/`，\n"
     "证据与最小复现见 `audit/V2_缺陷复现记录.md`。D8 尤其重要：它说明\"校准无效\"的表象曾是因为\n"
     "校准**根本没有被计算过**，与网络能力无关；修复后校准第一次真正参与决策（475 次被采用）。"),
]
for old, new in fixes:
    if old not in t:
        raise SystemExit("missing: " + old[:60])
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("report fixes applied")
