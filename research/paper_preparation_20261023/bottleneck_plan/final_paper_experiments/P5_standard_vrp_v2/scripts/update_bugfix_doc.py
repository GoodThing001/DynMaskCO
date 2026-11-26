from pathlib import Path
p = Path("BUGFIX_REPORT.md")
t = p.read_text(encoding="utf-8")
old = t[t.index("## 待修复（下一阶段）"):]
new = """## D6 六配置试选阶段与锁版对齐（2026-11-26 复验新增）

首轮开发面板（92/240 端点）出现两个异常信号：

1. `STD_FIX` 的 `ranking_counts` 恒为 `[N,0,0]` —— 校准热图从未被采用；
2. `STD_BUGFIX`（仅缺陷修复 + V1 简化搜索）在 n=100 CVRP 上**系统性优于** `STD_FIX`
   （8/8 例，幅度 0.4%–3.0%）。

回溯锁版 `pool_search.py` L192–218 与 `masked_repair.cpp` L224–268 后定位三处偏离：

| # | 锁版语义 | 修订前 V2 | 后果 |
|---|---|---|---|
| D6a | 试选选项 = `[−d, heat, heat_cal]`（三选项） | `[heat, heat_cal]`（缺纯距离） | 距离排序路径永不可选；机制被静默删除 |
| D6b | 试选取 `repair_batch` 返回值，**保留已完成的部分重构**；仅在 `duration<0.04` 时不记分 | 时间片超时（`_TimeUp`）整次丢弃 | Python 内核下几乎全部试选被丢弃 → 回退到"始终用默认配置"，试选形同虚设 |
| D6c | 接受判定与 `record` 更新在内核内完成，温度为 `2.5·max(.10,1−elapsed/s)` | 接受判定在池搜索层近似实现，温度额外乘 `|best|/n` | SA 行为偏离，等价于更强的贪心屏障 |

修复：

- 重写 `solver/masked_repair.py` 为锁版逐行移植：`choose_slot`（三选项排序 +
  最短**可行**插位保护 + 新开单车候选）、`reconstruct`（剥离后不可行则原样返回父解；
  贪心按掩码给定顺序；Regret 按客户编号升序、严格 `>`）、`repair_batch`
  （每次迭代重洗 → k=min(|group|, regret?10:6+rng()%12) → related 余弦/亲和平方随机
  组掩码 → record/state 与 SA 接受）；
- `pool_search.standard_search`：三选项 × 两算子（`proposals=[(rk,g) for g in (0,1)
  for rk in range(3)]`）、`duration=min(allowance,0.04)`、`TRIAL_TOLERANCE=1.5`、
  温度去掉 `|best|/n`、内层改为 `repair_batch` 时间片（0.10s）；
- 硬截止 `hard = started+seconds−0.05` 贯通试选/内层/证据（Python 内核单次重构
  可达数十毫秒，避免溢出 60s 在线预算）。

证据：`ranking_counts=[0,2,1]`（三选项均被采用，冒烟）、`test_kernels_v2.py`
新增 `test_deadline_large_instance_both_variants`（n=200 两种变体均不超预算）与
`test_repair_batch_related_and_acceptance`（related 掩码 + SA 下 record 单调不劣）。
记录：`audit/V2_原始机制与标准适配差异矩阵.csv` 与 `MECHANISM_PARITY.md §3.1`。

## 待修复（下一阶段）

- C++ 内核适配（效率对齐；当前 Python 内核吞吐低约两个数量级，已登记为已知差异）；
- 正式面板的独立复算审计脚本（`scripts/p5v2_audit.py`）在全部端点上的执行与归档。
"""
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("bugfix report updated")
