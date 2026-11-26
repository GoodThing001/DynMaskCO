import io
from pathlib import Path
p = Path("MECHANISM_PARITY.md")
t = p.read_text(encoding="utf-8")

old_sec3 = t[t.index("## 3. 已知差异（如实登记）"):]
new = """## 3. 已知差异（如实登记）

1. **内核**：V2 目前为 Python 实现（与 V1 同量级），锁版为 C++（约两个数量级更快）。
   60s 预算下 V2 的 `iterations` 远低于锁版；C++ 内核适配见后续阶段（效率对齐）。
   由此带来的**结构性后果**（2026-11-26 修订后登记）：锁版单次重构 40 客约 1–3 ms，
   0.04s 试选窗口可容纳十余次重构；Python 内核单次重构约 10–50 ms，同一窗口只能容纳
   1–4 次重构。因此 V2 的"六配置试选"在语义上正确，但**统计效率低于锁版**。
2. **SA 温度**：V2 已改为锁版公式 `2.5·max(.10,1−elapsed/seconds)`（旧版按 `|best|/n`
   归一化，属偏离，已删除）。
3. **校准热图**：`calibrated_heat(dist, heat)` 与锁版公式一致；三选项
   `[−d, heat, heat_cal]` 全部参与六配置试选（锁版 HYBRID 同款）。
4. **客户插入顺序**：已按锁版修订——贪心模式按掩码给定顺序插入；Regret 模式按客户编号
   升序扫描、严格 `>` 取最大后悔值。`related` 模式在 `repair_batch` 内部实现
   （每偶数次迭代按 `H[anchor,i]+H[i,anchor]` 降序做"平方随机"抽样），与锁版一致。

## 3.1 试选阶段一致性修订（2026-11-26，D6）

首轮开发面板暴露：`STD_FIX` 的 `ranking_counts` 恒为 `[N,0,0]`（校准热图从未被采用），
且 `STD_BUGFIX` 在 n=100 CVRP 上系统性优于 `STD_FIX`。回溯锁版 `pool_search.py` L192–218
与 `masked_repair.cpp` 后确认 V2 存在三处试选阶段偏离：

| # | 锁版语义 | 修订前 V2 | 影响 |
|---|---|---|---|
| D6a | 三选项 `[−d, heat, heat_cal]` | 两选项 `[heat, heat_cal]`（缺纯距离） | 距离排序路径永不可选，等价机制被删 |
| D6b | 试选使用 `repair_batch` 返回值（**保留已完成的部分重构**） | `_TimeUp` 直接丢弃整次试选 | Python 内核下几乎全部试选被丢弃，退化为"永远用默认配置" |
| D6c | 接受判定在内核内按 `new>=u−eps` 或 SA 执行，`record` 先更新 | 在池搜索层用 Scala 近似，且温度乘 `|best|/n` | SA 行为与锁版不一致 |

修订后 `ranking_counts` 三种选项均被采用（冒烟 `[0,2,1]`），试选阶段与锁版逐行对齐；
硬截止（`hard = started+seconds−0.05`）保证 Python 内核单次重构超时不再溢出 60s 预算。

## 4. 冒烟结果（本机 CPU，伪热图；2026-11-26 修订后重跑）

| 实例 | 预算 | epochs | iterations | 跨车块 | 配置采用 | D |
|---|---:|---:|---:|---:|---|---|
| 合成 n=60（CVRP，距离模式） | 5s | 4 | 1229 | — | — | 1264 |
| 合成 n=100（CVRP，距离模式） | 5s | 4 | 1128 | — | — | 1737 |
| 合成 n=80（CVRP，伪热图） | 6s | 5 | 1059 | — | `[0,2,1]`/模式`[3,2]` | 1448 |

> 冒烟使用合成实例与伪热图，仅验证机制与统计；正式实验使用冻结 f0_r0 网络与官方实例。
"""
t = t.replace(old_sec3, new)
p.write_text(t, encoding="utf-8")
print("parity doc updated")
