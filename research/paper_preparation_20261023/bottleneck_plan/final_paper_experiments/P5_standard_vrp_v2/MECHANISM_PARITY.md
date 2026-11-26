# V2 机制一致性报告（MECHANISM_PARITY）

> 对照 `SOURCE_AUDIT.md` 的 M1–M9 与锁版 12 项机制清单，逐项给出 V2 实现位置与验证证据。
> 实现文件：`standard_vrp/solver/masked_repair.py`、`standard_vrp/solver/pool_search.py`、
> `standard_vrp/core/calibration.py`、`standard_vrp/adapters/objective.py`。

## 1. 机制对照

| # | 锁版机制 | V2 实现 | 证据 |
|---|---|---|---|
| M1 | 40 客掩码池 + 自适应双模式（related/random，`.65r+.35·min(3,log1p(gain))`） | `pool_search.standard_search`：`MASK_POOL=40`；`uses/rewards` 与 `argmin/argmax+0.1/√(1+uses)` | 冒烟：`mask_mode_counts` 记录两种模式使用次数 |
| M2 | 冲突池（热图排名靠后的 12 客 + 每人最佳替代 + 组内前 8） | `pool_search.conflict_pool`（1 基排名、有限值屏蔽） | `test_conflict_pool`；冒烟中 active_mode=1 时启用 |
| M3 | 每 epoch 一次前向，重构与跨车共享 | `heat` 每 epoch 计算一次；跨车 `propose_orders` 用同一 `heat` | 冒烟：`前向=epochs`（8=8）、`跨车块=3` 共享 |
| M4 | bundle 2–4 条相近路线（≤24 客）并入掩码参与预测 | `pool_search.bundle` + `hidden` 并入 `pool`（补齐至 40） | `test_bundle_and_orders`；冒烟 `split_blocks≥1` |
| M5 | 六配置短时试选（3 排序 × 2 算子，每配置 ≤0.04s）后采用最优持续 | `options=[dist, heat, calib]` × `greedy/regret2`，`CONFIG_TRIAL_S=0.04`，`trials.sort` 取最优并继续使用 | 冒烟：`ranking_counts=[4,2,0]`（配置被实际采用） |
| M6 | 跨车顺序扰动（随机排列+25% 反转+1–4 次热图引导扰动）+ 固定序列 Split DP | `pool_search.propose_orders`（对齐锁版 `block_split.proposals`）+ `_split_sequence`（`split_cvrp`/`split_vrptw`） | 冒烟：`split_orders=1791`（0.30s 内评估） |
| M7 | 工作解 SA 接受 + 历史最好单调 + 每 4 epoch 重置 | `temperature=2.5·max(.10,1−elapsed/s)·|best|/n`；`child_score ≥ cur` 或 `> best−6·tol` 且 `rand<exp(...)`；`RESET_PERIOD=4` | 冒烟：`improvements>0`、`best` 单调（只接受更优） |
| M8 | 组破坏规模（贪心 6–17 / regret 10） | `k = 10 if regret2 else 6+rng.randrange(12)` | 冒烟：`iterations` 与 `regret_epochs` 计数 |
| M9 | 预测异常留痕（不再静默回退） | `PredictorStats`：attempts/success/failed/fallbacks/forward_s/errors(traceback) | 冒烟：`predictor.success == epochs`；距离模式 `attempts=0` |

## 2. 候选语义（锁版 choose_slot）

| 语义 | 锁版 | V2 |
|---|---|---|
| 候选构造 | 全部路线全部可行插位 + 新开单车 | `choose_slot`：`fits` 过滤 + 新开单车候选始终在列 |
| 贪心 | 距离序前 6 + 热图序再 6，逐一精确评价；不足则距离序补扫 | 同（`greedy_distance_slots=6`、`greedy_heat_slots=6`） |
| Regret | 热图序 top-8 + 最短**可行**插位保护 | 同（`topk=8`、按绕行升序扫第一个可行者、替换末位） |
| 动态后悔值 | 每次插入后重算全部 pending | `masked_repair` while 循环 + `choose_slot(regret=True)`（k(k+1)/2 次调用被测试锁定） |
| 字典序 | 经济价值标量 | VRPTW：`(K,D)` 键 + 后悔值 `K差×1e6+距离差` |

## 3. 已知差异（如实登记）

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
