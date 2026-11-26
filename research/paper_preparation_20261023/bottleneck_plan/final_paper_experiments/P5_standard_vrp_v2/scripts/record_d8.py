# -*- coding: utf-8 -*-
"""在 audit/V2_缺陷复现记录.md 追加 D8（校准热图从未生效）。"""
from pathlib import Path

p = Path("audit/V2_缺陷复现记录.md")
t = p.read_text(encoding="utf-8")
add = """

## D8 校准热图从未生效（2026-11-26 正式面板启动后发现）

**现象**：开发面板（D6/D7 修复后、本轮之前）240 端点全部通过独立复算，但
`stats.ranking_counts` 恒为 `{0: N, 1: M, 2: 0}` —— 六配置试选的**第三个选项（校准热图）
一次都没被采用**。

**定位**：`core/calibration.py` 把锁版 `calibrated.json` 的路径写成了**本机 Windows
绝对路径**；远端 Linux 上 `locked_alpha()` 抛 `FileNotFoundError`，而池搜索里的
`try: heat_cal = calibrated_heat(...) except Exception: heat_cal = None` 把它**静默吞掉**，
于是 `options` 退化为 `[None, heat]`，校准分支永久不可达。这正是 D4（校准缺失）在一些
环境下的再次发生，且更隐蔽——因为没有任何异常外泄。

**证据**：本次开发面板（`development_results/dev01_pre_D8/`，240 端点）作废保留；
其 `ranking_counts` 第三项恒为 0；远端 `locked_alpha()` 直接复现该异常
（`p5v2_fixed_candidates.py` 未捕获，第一次运行即崩在 `calibrated.json` 路径上）。

**修复**：

1. 锁版 `calibrated.json` 逐字节拷贝进包内 `standard_vrp/core/calibrated.json`，
   解析顺序 = 包内 → 环境变量 `MASKCO_P5V2_CALIBRATED` → 仓库只读路径；
   三者皆缺则**直接报错**（`FileNotFoundError`），不再返回默认值；
2. 池搜索内的校准异常不再静默：写入 `PredictorStats.errors` 并计入 `fallbacks`；
3. 复核：冒烟 `ranking_counts=[2, 0, 1]`（校准选项被采用的直接证据）；
   `locked_alpha() == 0.25`（与锁版一致）。

**影响**：`dev01_pre_D8` 的 240 端点与当时正在运行的正式面板批次全部作废重跑。
该缺陷也解释了"校准热图无效"的表象——它此前根本没有被计算过。
"""
t = t.rstrip() + add
p.write_text(t, encoding="utf-8")
print("D8 recorded")
