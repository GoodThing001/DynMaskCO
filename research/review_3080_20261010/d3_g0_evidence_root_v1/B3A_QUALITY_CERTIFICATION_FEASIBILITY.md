# B3a-P0：质量 / 鱼可售性在线认证可行性审计（只读）

**日期**：2026-10-14 · **基准 commit**：`3038a16` · **性质**：只读审计，**不实现**、不改 `_commit`、不改 B2 模块、不绕过 D8。
**要回答的问题**（用户裁定）：给定**当前冷链状态 + 候选路线 + 已知合同**，能否**保守认证**该计划在执行过程中的鱼可售阈值？

---

## 0. 结论（先给答案）

**可行性：YES —— 而且不需要任何新物理。**

品质轨迹是 `(当前车辆冷链状态, 各段时间, 舱温剖面, 合同)` 的**确定性函数**，而**这条轨迹已经被同一个重放函数驱动**：
`c0_completion_energy`（`run_exp_energy_c0.py:112-151`）在认证能耗时**逐单调用 `transition_segment`**，并且**已经传入** `order_quantity` / `order_temp_class` / `initial_quality` / `segment_distance_units` / `return_to_depot`（`:132-136`、`:147-150`）。
也就是说：**决策时品质演进已经在算了，只是把结果丢掉了**——该函数只 `return float(state.cumulative_energy_kwh)`（`:151`），没有返回 `cargo_manifest` 里每个 lot 的 `quality_remaining`。

因此 D8 的 `CAPABILITY_GAP_BLOCKING` **不是物理上不可为**，而是**接口未暴露 + 保守性尚未定义**。两者都是实现问题，不是建模问题。

**但"可行"≠"照搬终端函数即可"**（用户的保留 2）。下面把差什么、为什么、以及保守性方向讲清楚。三处必须解决：
① 暴露重放后的逐 lot 品质；② 用**当前合同实际生效的 v1 结算公式**（不是 v2）；③ 用**最坏时刻表**评估才叫保守。

---

## 1. 当前生效的品质物理（实测，不是从文档推断）

**本 D3/M0 线跑的是 v1 `QualityConfig`，不是 A-v1 的 v2 绝对感官合同**——这一点必须先钉死，否则会拿错公式：

| 项 | 实测值 |
|---|---|
| `env.coldchain_contract.quality` 类型 | **`QualityConfig`（v1）**，`isinstance(..., QualityConfigV2) == False` |
| v1 字段 | `reference_rate_per_hour=(0.001, 0.002, 0.0035)`、`activation_over_r_kelvin=(3500, 4500, 5500)`、`reference_temperature_k=277.15`、`unsalable_quality_threshold=(0.8, 0.85, 0.9)`、`product_value=(1.0, 1.5, 2.0)` |
| v2 专属字段（`threshold_type` / `dynamics_order` / `initial_value` / `rate_per_hour` / `metric_name`） | **全部 ABSENT** |
| 数据集 `initial_quality` | **三类温区全部为 1.0**（**不是**鱼=9） |

⇒ **可售判据是"比例阈值"而不是"绝对感官分"**：
`salable_c = quality_remaining / initial_quality >= 0.8/0.85/0.9`（`coldchain_state.py:458-468`，v1 结算分支）。

**速率是 Arrhenius 温度相关的一阶衰减**（`coldchain_state.py:128-143`）：
```
rate_c(T) = base_rate_c * exp(activation_over_R_c * (1/277.15 - 1/(T+273.15)))
quality_remaining <- min(quality_remaining, quality_remaining * exp(-rate_c(T) * duration_h))
```
- 温度高于 4 °C ⇒ 速率**更高**（22 °C 时 class-0 约 ×2.16）；低于 4 °C ⇒ 更低（−18 °C 时 class-2 约 ×0.18）。
- `min(...)` 上限保证品质**单调不增**（`:405`）——这条单调性是后面"保守性"论证的基础。
- **未预冷（`precooled=False`）时用田间环境温度**评估平均温度（`:398-402`）⇒ 速率显著更高。这是与 A-v1 预冷决策耦合的一条真实通道。

**结算发生在"返仓交付"时刻**：`salable` 写在交付记录上（`:458-468`），终端检查读的是 `delivered_to_depot` 里的记录（`coldchain_evaluator_a1.py:115-117` 的 `fish_unsalable`）。
⇒ 决策时投影必须投到**返仓后的状态**，而不是"送到客户时"。

---

## 2. 为什么说"不需要新物理"

`c0_completion_energy` 的循环体（`run_exp_energy_c0.py:125-150`）已经包含决策时投影所需的**全部**要素：

| 投影必需要素 | 是否已在重放中 |
|---|---|
| 从**当前**真实车辆冷链状态接续（含已消耗能耗与已装货物品质） | ✅ `state` 即 `v.coldchain_state`（`:169`），且 committed 车把 `committed_next` 前置一次（`certify_plan:176-181`） |
| 行驶时间 | ✅ `d / speed`，`speed = units.speed_kmph / distance_km_per_unit`（`:121`、`:128`） |
| 服务时间 | ✅ `sfinish = sstart + service_time[o]`（`:130`） |
| 发车下界 | ✅ `depart = max(t, reveal_time[o])`（`:127`）— **见 §4 的未来信息问题** |
| 舱温剖面 / 制冷模式 | ✅ `transition_segment` 内部由 `contract.thermal` 与 `active_zone_mask` 决定 |
| 逐单装载与温区 | ✅ `order_quantity` / `order_temp_class` / `initial_quality`（`:131`、`:135`） |
| WAIT 到预约截止 | ✅ `wait_until = env.booking_horizon`（`certify_plan:166`、`c0_completion_energy:139-144`） |
| 返仓段 | ✅ `return_to_depot=True`（`:150`） |
| **逐 lot 品质结果** | ❌ **被丢弃**——只返回 `cumulative_energy_kwh`（`:151`） |

⇒ 唯一的"缺失"是**返回值**：把最终 `state.cargo_manifest` 的逐 lot `quality_remaining`（以及沿途每个交付点的品质）一并返回，即可对同一轨迹套用 v1 结算公式。**没有新方程、没有新参数。**

---

## 3. "保守"必须明确定义——名义轨迹是**乐观**上界

品质单调不增（`:405` 的 `min` 上限；两个动力学都是一阶/零阶衰减）。因此：

> **按计划时刻表重放得到的品质，是该计划可达品质的上界** ⇒ 它是**乐观**投影，不是保守投影。

要让认证**保守**（绝不放过一个可能失败的计划），必须评估**最坏许可时刻表**，而不是名义时刻表：

| 保守化维度 | 具体做法 |
|---|---|
| 每个客户的到达时刻取**最晚许可**（`tw_end`）而非 `tw_start`/最早到达 | 在重放里把 `sstart = max(arrive, tw_start)` 换成 `arrive_worst = tw_end`，并相应推后后续时点 |
| 保留**完整** WAIT 到 `booking_horizon`（不能跳过） | 已经如此（`:139-144`）；这是品质继续衰减的区间，**跳过它就是乐观** |
| 返仓段必须计入 | 已经如此（`:145-150`） |
| 温度取该时刻表的实际剖面（含未预冷时的田间高温） | 同一重放自动给出；**注意未预冷分支会显著抬高速率**（`:398-402`） |
| 承诺路段 | 已前置一次（`certify_plan:176-181`） |
| 多车/多 lot | 逐车逐 lot 分别评估；鱼（按 v1 阈值）取最紧的 `threshold`（class 2 = 0.90） |

**一个必须写进规格的自由度**：是否对"服务期间开门"造成的温度扰动取上界。`transition_segment` 的舱温由合同与制冷模式确定（确定性），因此**不需要额外保守化**；但若未来引入随机舱温，则该维度必须显式加上界。

---

## 4. 用户保留 1 的落实：认证分量的**订单集合合法性**前置

我在 B2 里把补偿项（含"计划订单必须已揭示"）放在四项检查**之后**，因此 `certify_plan` 可能在"尚未验证合法性"的计划上被调用——而 `c0_completion_energy:127` **确实读取 `reveal_time`**。

实测确认：
- 对**已揭示**订单该读取是惰性的（`reveal_time[o] <= clock` ⇒ `max(t, rev) == t`）；
- 但它**不会拒绝**未揭示订单，只是把发车推后 ⇒ 若计划混入未揭示订单，认证会**静默**按未来揭示时刻排程。

⇒ **规格要求（B3b 实现时）**：把"合法性前置"改为**短路顺序**——先 `revealed_only` / `no_served_in_plan` / `no_duplicate_order` / `plan_vehicle_set` / `full_partition`，**全部通过后**才允许调用 `certify_route` 与 `certify_plan`。
这样"所有认证分量只能收到经过合法性检查的订单集合"成为**结构性质**，而不是靠调用顺序碰巧成立。
**本轮不修改已验收的 B2 模块**（B2 已收束），此改动列为 B3b 的**必做项**，见 §6 的 D-B3A-1。

---

## 5. 与终端 `evaluate_trace_a1` 的关系（用户保留 2）

**不能**把 `evaluate_trace_a1` 搬到在线入口，原因是结构性的，不只是"它需要完整轨迹"：

| 差别 | 终端评价器 | 决策时认证器（本设计） |
|---|---|---|
| 输入 | **真实执行后**的 traces（含每个服务的 `coldchain_before/after`） | **当前状态 + 候选计划**（未执行） |
| 时间 | 事后 | 事前 |
| 语义 | "实际交付是否可售" | "**若按此计划执行**（且按最坏许可时刻表）是否可售" |
| 失败含义 | 违反硬约束 | 拒绝该计划的**认证** |

另外两点必须注意：
- 终端检查用 `r.salable is False`（`coldchain_evaluator_a1.py:115`），而 `salable` 只在**交付记录**上被赋值；番茄在 v1 下**没有** `none` 语义（那是 v2），因此 v1 下三类温区**都有**阈值——这与"番茄不评价"的 v2 口径不同，必须按 v1 处理。
- 终端检查还覆盖 `commitment_violation` / `duplicate_service` / `served_not_accepted` 等**非品质**硬约束，那些属于组合认证的其他分量，不能因为要做品质认证就把它们混进来。

---

## 6. 交付给 B3b 的最小规格（本轮只给规格）

**D-B3A-1（顺序，必做）**：组合认证改为**短路顺序**：合法性补偿与分区检查全部通过后，才调用路线与能耗/品质分量。理由见 §4。**需单独授权修改已收束的 B2 模块。**

**D-B3A-2（品质投影，核心新增）**：新增一个**只读**投影函数（建议置于 `simulation/`，名称待定，如 `project_plan_quality`），语义为：
```
project_plan_quality(env, inst_idx, clock, vehicles, plan, contract, *, worst_case_schedule=True)
  -> {vid: [{order_id, temp_class, delivered_quality, initial_quality, ratio, threshold, salable}]}
```
- 复用 `transition_segment`（**不新写物理**），并与 `c0_completion_energy` **共用同一段重放代码路径**以免出现两套时刻表；
- 必须返回**逐 lot 的 `delivered_quality` 与 `threshold`**，并显式标注 `worst_case_schedule` 取值；
- 未预冷订单必须走田间温度分支（`:398-402`）。

**D-B3A-3（认证分量）**：`quality_salability` 三态扩展为四态，且 `installable` 的判据不变（无阻断项才可安装）：
| 状态 | 含义 |
|---|---|
| `CHECKED_SATISFIED` | 注入的在线检查通过 |
| `CHECKED_FAILED` | 注入的在线检查失败 |
| `PROJECTED_SATISFIED_CONSERVATIVE` | **新**：用最坏时刻表投影，全部低于阈值但有裕度 |
| `CAPABILITY_GAP_BLOCKING` | 仍无可用检查（当前状态） |

**D-B3A-4（独立反例，预注册）**：
- **Q1** 名义轨迹可售、**最坏时刻表不可售** ⇒ 认证必须**拒绝**（证明保守性是必需的，不是学术讲究）；
- **Q2** 未预冷订单在田间高温下速率显著更高 ⇒ 认证必须与预冷分支一致；
- **Q3** class-2（阈值 0.90）在同等时长下最先失败 ⇒ 认证必须按**逐温区**阈值判定，不能用一个全局阈值；
- **Q4** 投影结果与**实际执行**后的终端 `salable` 一致（同计划、无重规划场景）⇒ 证明投影语义正确；
- **Q5** 计划内混入未揭示订单 ⇒ 在**合法性前置**下，路线/品质分量**不得被调用**（可观测：分量结果为 `not_run_illegal_order_set`）。

---

## 7. 本文件没有做的事

未实现任何投影函数；未修改 `d3_composed_certifier.py` / `d3_credential.py` / `d3_install_gate.py` / `_commit` / `_plan_and_commit`；未启用写回；未降低 D8 门槛；未启动 E2 / F−D / 32 状态重采集 / G1 / G2 / 最终测试。
