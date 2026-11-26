from pathlib import Path
p = Path("BUGFIX_REPORT.md")
t = p.read_text(encoding="utf-8")
add = """

## D8 校准热图路径与静默回退（2026-11-26 正式面板启动后新增）

- 问题：`core/calibration.py` 使用本机 Windows 绝对路径读取锁版 `calibrated.json`；
  远端 Linux 抛 `FileNotFoundError`，被池搜索的 `except Exception: heat_cal = None` 静默吞掉，
  导致**校准热图从未参与过任何一次排序选择**（`ranking_counts[2]` 恒为 0）。
- 修复：锁版 `calibrated.json` 逐字节入包（`standard_vrp/core/calibrated.json`），
  解析顺序 包内 → 环境变量 → 仓库只读路径，全部缺失则报错；校准异常写入
  `PredictorStats.errors` 并计 `fallbacks`，不再静默。
- 证据：`test_calibration_v2.py`（alpha=0.25、`calibrated_heat` 形状与数值）+
  冒烟 `ranking_counts=[2,0,1]`；`audit/V2_缺陷复现记录.md` D8 段。
"""
if "## D8" not in t:
    t = t.rstrip() + add
    p.write_text(t, encoding="utf-8")
    print("BUGFIX_REPORT updated with D8")
