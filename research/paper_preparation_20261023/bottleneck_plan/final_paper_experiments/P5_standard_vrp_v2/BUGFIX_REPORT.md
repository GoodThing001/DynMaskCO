# V2 缺陷修复清单（BUGFIX_REPORT）

> 对照 `SOURCE_AUDIT.md` 的 D1–D5；所有修复均有单元测试证据（`standard_vrp/tests/`）。
> 修复不改变目标定义、不引入新可调超参数（唯一常数 BIG=1e6 是字典序标量化权重，
> 使 (K,D) 字典序与标量严格一致，非车辆惩罚系数）。

## D1 客户排序生效

- 问题：V1 `remaining = list(removed)` 忽略 ranking。
- 修复：`masked_repair` 贪心严格按 `ranking` 重插（`rank_customers` 输出的顺序）；
  不在 ranking 中的客户按登记顺序补后。
- 证据：`test_masked_repair_v2.py::test_d1_ranking_effective` —
  升序/降序两种 ranking 产生不同的重插顺序（日志逐客户对比）。

## D2 VRPTW 两算子区分

- 问题：V1 VRPTW 分支 `for cust in list(removed)`，greedy/regret2 共用贪心。
- 修复：VRPTW 与 CVRP 共用同一 `choose_slot`/`masked_repair` 实现，按 operator 分派；
  字典序后悔值 = K 差×1e6 + 距离差。
- 证据：`test_d2_vrptw_operators_differ`（重插顺序不同）、`test_vrptw_lex_regret`
  （K 差后悔值 999950 ≫ 同 K 距离差 50）。

## D3 动态 Regret-2

- 问题：V1 `order_key` 一次性排序后逐客户插入（静态）。
- 修复：锁版语义——while pending：对**全部** pending 客户调用 `choose_slot(regret=True)`,
  取后悔值最大者插入，然后重算；平局按登记 ranking 位置确定性决断。
- 证据：`test_d3_dynamic_regret` — `choose_slot` 调用数 = k(k+1)/2（k=10 → 55），
  greedy = k（10）。动态复杂度被测试锁定。

## D4 距离条件残差校准

- 问题：V1 calib 与 heatmap 共用同一原始热图，无校准。
- 修复：新增 `standard_vrp/core/calibration.py`，逐项对齐锁版：
  `spatial=-d/scale`（scale=各行第 8 小距离中位数）、
  `residual=clip(log(rank_D/rank_H),-3,3)`、`heat=spatial+alpha*residual`，
  **alpha=0.25 从锁版 `calibrated.json` 读取核实**；缺失热图（-inf）不产生 NaN。
- 证据：`test_calibration_v2.py`（5 项：一致排名零残差、正/负补偿、±3 截断、
  缺失热图无 NaN、校准与原始热图存在真实差异）。

## D5 最短可行插位保护

- 问题：V1 追加几何绕行最小位置，不验证可行性。
- 修复：锁版语义——按绕行增量升序扫描**全部**插位，取第一个**可行**者
  （容量 + VRPTW 时刻表）；若不在热图 top-k 内则替换末位候选。
- 证据：`test_d5_short_feasible_slot_protection`（构造"几何最短不可行"场景，
  保护位落在可行宽松路线且 regret 有值）。

## 新开单车候选

- 问题：V1 仅当无可行既有插位时才开新车，忽略"新车可能更优"。
- 修复：新开单车作为候选始终参与比较（CVRP：singleton 成本；VRPTW：K+1 → 字典序劣，
  仅在无可行插位时被选中）。
- 证据：`test_new_vehicle_candidate`。

## 热图缺失/NaN 处理

- 修复：`_edge_score` 对 -inf/NaN 返回 None（缺失 → 不参与热图排序，回退距离规则），
  绝不把缺失解释为"模型拒绝该边"，也不让 -inf 参与减法产生 NaN。
- 证据：`test_heat_nan_handling`（全 -inf 热图 → 距离回退、结果可行）。

## D6 六配置试选阶段与锁版对齐（2026-11-26 复验新增）

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

## D8 校准热图路径与静默回退（2026-11-26 正式面板启动后新增）

- 问题：`core/calibration.py` 使用本机 Windows 绝对路径读取锁版 `calibrated.json`；
  远端 Linux 抛 `FileNotFoundError`，被池搜索的 `except Exception: heat_cal = None` 静默吞掉，
  导致**校准热图从未参与过任何一次排序选择**（`ranking_counts[2]` 恒为 0）。
- 修复：锁版 `calibrated.json` 逐字节入包（`standard_vrp/core/calibrated.json`），
  解析顺序 包内 → 环境变量 → 仓库只读路径，全部缺失则报错；校准异常写入
  `PredictorStats.errors` 并计 `fallbacks`，不再静默。
- 证据：`test_calibration_v2.py`（alpha=0.25、`calibrated_heat` 形状与数值）+
  冒烟 `ranking_counts=[2,0,1]`；`audit/V2_缺陷复现记录.md` D8 段。
