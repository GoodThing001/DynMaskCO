# B3-QP：静态品质下界证明闭合审计（只读）

**日期**：2026-10-14 · **基准 commit**：`8089021`（role-C 材料已并入，**不是** `3d99892`）· **性质**：只读证明闭合审计
**用户裁定**：被证伪的是"最晚出发最差"这一**收紧引理**，**不是静态下界本身**。本轮优先闭合静态下界的证明，而不是增加搜索次数或修改放行状态。

| 产物 | 作用 |
|---|---|
| 本文件 | A1–A5 的源码锚点 + 证明 + 调用路径证据；F-a/F-b/F-c 的最小反例与硬门覆盖；已装/未装两分支；范围声明 |
| `B3QP_ADJUDICATION.md` | **独立只读复核者的裁定**（单独进程，不看我的结论先行；其结论原样收存） |
| `B3QC_LOWER_BOUND_MATRIX.json`（已更新） | BND 扩展为**全部 14 个决策点 / 256 行 / 两分支覆盖** |
| `scripts/tests/test_d3_b3qc_lower_bound.py`（已更新） | BND 现在同时覆盖 `in_cargo(34)` 与 `not_yet_loaded(222)` |

---

## 0. 结论（先说答案）

**问题是**：在 D3/M0 v1 合同与真实执行硬门下，能否严格证明 `Q_i^LB ≤ Q_i^delivery` 对**所有允许轨迹**成立？

| 层次 | 裁定 |
|---|---|
| **条件性蕴含**（若轨迹 ω ∈ Ω，则下界成立） | ✅ **PROVEN**（A1/A2/A3 已由源码闭合；A4 由 Ω 的成员条件给出；F-a 已闭合，F-c 已闭合） |
| **Ω 成员性在生产提交入口被强制执行** | ❌ **NOT_ESTABLISHED**——**没有任何生产提交入口执行这些硬门**；它们只在**事后评估层**被计算（见 §3.4）。这正是你特别提醒的那一点 |
| **总体问题（"对所有允许轨迹"）** | **`NOT_ESTABLISHED`** —— 因为"某次真实执行属于 Ω"目前只是**事后分类**，不是提交时的**保证** |

⇒ 与你的建议一致：**不解除 D8**、`CONSERVATIVENESS_NOT_ESTABLISHED` 保留（但**理由已精确到"缺提交时硬门强制"**，而不是"下界不成立"）。
⇒ 也**不**建议现在收紧 τ（你的顺序：先证明粗上界成立，再单独预声明更紧上界）。

---

## 1. 适用范围与合同版本（**先固定，不事后收缩**）

| 项 | 固定值 |
|---|---|
| 合同版本 | **v1 `QualityConfig`**（一阶、逐温区 Arrhenius、比例阈值 `0.8/0.85/0.9`） |
| 数据集 | `data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz`（sha256 `c35fa683…`） |
| 温区活动掩码 | `(True,)*zones`（`strict_online_env._coldchain_active_mask()`；本配置下从不关闭） |
| 预冷 | `thermal.precooled=True`、`precool_min_temp_class=0`、**无类低于该阈值** ⇒ **未预冷分支不可达** |
| 速度口径 | `units.speed_kmph=1.0`、`units.distance_km_per_unit=1.0` ⇒ 与 env `tw_speed=1.0` **一致** |
| `T^max` | `hard_upper = (30, 8, -12)`（**不含 ambient 放宽**） |
| `τ^max` | `tw_end[0] − now`（采样点为 24.0 h） |

**Ω 的定义与成员条件**见 `B3QC_LOWER_BOUND_DESIGN.md` §1（本轮**未改动**，也未收缩轨迹集合）。

---

## 2. A1–A5：源码锚点 + 证明 + 调用路径证据

### A1 —— 唯一的品质更新形式 ✅ PROVEN
- **锚点**：`coldchain_state.py:404-405`
  ```python
  quality_remaining = lot.quality_remaining * math.exp(-rate * duration_h)
  quality_remaining = min(lot.quality_remaining, max(0.0, quality_remaining))
  ```
- **证明**：`_advance_manifest_quality` 是 v1 下**唯一**写 `quality_remaining` 的路径（`transition_segment:224-230` 按合同类型二选一）；`_delivery_record` 只**读** `lot.quality_remaining`（`:456-468`）。`min(q_old, …)` 只会**降低**，故 `q_new ≤ q_old·exp(−k·Δt)`。新装 lot 的 `quality_remaining = initial_quality`（`:276`）⇒ 起点不高于 `initial_quality`。
- **调用路径证据**：`transition_segment` → `_advance_manifest_quality`（v1 分支）→ 上述两行；无其它写入点。
  **穷尽检索结果**（`quality_remaining\s*=|replace\(lot, quality_remaining`）：全文件仅 4 处——
  `:276`（新建 lot，起点 `=initial_quality`）、**`:404-405/409`（v1 唯一衰减写入点）**、`:489`（`_advance_manifest_quality_v2`，**v2 专属**，本配置不走）、`:466/:512`（只写入交付记录，**只读**）。
  ⇒ v1 下**不存在**第二条改变品质的路径，A1 闭合。
- **边界**：`duration_h <= _EPS` 时直接返回原 manifest（`:478-479`）⇒ 不增，仍满足不等式。

### A2 —— 驱动速率的温度受 `T^max` 约束 ✅ PROVEN（条件性，见 §3.4）
- **锚点**：速率温度来源 `coldchain_state.py:398-402`
  ```python
  average_temperature = 0.5*(temperature_before[c] + temperature_after[c])  # precooled
  average_temperature = contract.thermal.ambient_temperature_c               # 未预冷
  ```
- **证明**：（i）**未预冷分支不可达**（§1，实测 `ambient_branch_samples=0`、`classes_below_floor=[]`），故只需管温区均值；（ii）Ω 含 `temperature_hard_feasible`，而该判据 = `thermal_count == 0`（`coldchain_evaluator.py:358`），`thermal_count` 由逐段 `_thermal_violations` 累加 ⇒ Ω 内每段 `before/after` 都在硬界内 ⇒ 其均值也在 `[hard_low, hard_high]` 内 ⇒ 以 `k` 对 T 非减（A3）得 `k(T_used) ≤ k(hard_upper)`。
- **实测**：**2134** 次速率温度采样、**0** 次越界，最大 18.03 °C。
- **调用路径证据**：`transition_segment:216-218`（算 after）→ `:220-230`（v1 品质推进）→ `:298-299`（越界计数）→ `:301-315`（新状态）。**注意顺序**：计数在品质推进**之后**，因此 A2 的论证依赖"计数覆盖了同一段的 before/after"，见 F-a。

### A3 —— `k(c,T)` 对 T 非减 ✅ PROVEN（解析 + 数值）
- **锚点**：`coldchain_state.py:128-143`
  ```python
  exponent = activation_over_R * (1/T_ref − 1/T_actual);  rate = base_rate * exp(exponent)
  ```
- **证明**：`d(exponent)/dT = activation_over_R / T² > 0`（`activation_over_R=(3500,4500,5500) > 0`，`T>0` 由 `:137-138` 保证），`exp` 单调 ⇒ `k` 对 T 严格递增。
- **数值核对**：三类温区在 `[hard_low, hard_high] ∪ {ambient}` 上取 11 点，全部非减（`A3` PASS）。

### A4 —— Ω 上的在舱时长 ≤ `τ^max` ⚠️ 条件成立，**生产保证未建立**
- **锚点**：返仓上限 `tw_end[0]` 被两处使用：`certify_route:198-202`（计划期 `return_slack`）与终端评估器的 `depot_return_violation`（`coldchain_evaluator_a1`）。
- **证明（条件性）**：`τ^max = tw_end[0] − now`。对**已在舱** lot：`t_load ≤ now`，交付 = 返仓到达 ≤ `tw_end[0]`（Ω 成员条件）⇒ 在舱时长 ≤ `τ^max` ✓。对**尚未取货**订单：`t_load ≥ now` ⇒ 在舱时长 = `t_delivery − t_load ≤ tw_end[0] − now = τ^max` ✓。
- **未建立的部分**：见 §3.4——**没有任何生产提交入口强制"返仓 ≤ tw_end[0]"**，因此"提交的计划会属于 Ω"不是保证，而是事后分类。**这是 A4 唯一未闭合处。**

### A5 —— Ω 由硬门定义 ⇒ 下界以硬门为条件 ⚠️ **生产端强制缺失**
- **锚点**：`hard_gate.py:104-129`（`hard_vector_from_outcome` / `hard_pass`）、`:20-101`（`outcome_protocol_error`）；`coldchain_evaluator.py:358`（`temperature_hard_feasible`）。
- **调用路径证据（关键，已在 HEAD `8089021` 复核）**：`outcome_protocol_error(` / `hard_vector_from_outcome(` 的调用点全部在**评估/校验/诊断**脚本中（`check_drift_parity.py`、`check_hard_gate.py`、`instance_validation.py`、`run_action_oracle.py`），**没有一处位于提交路径**；`certify_plan(` 在活跃树中有 **8 个非测试调用点**——`rolling_optimizer.py:197`、`rrnco_accept_replanner.py:150`、`run_a1_downstream_ablation.py:88`、`run_exp_energy_c0.py:302`、`solver_accept_replanner.py:424`、`pickup_budget_sweep.py:96`、`pickup_rejection_attribution.py:97`，以及 **`d3_composed_certifier.py:233`＝组合认证器自身**。⇒ **实际绕过组合门的路径 = 7 处**（此前 B3a 报告写的"8 处旁路"未扣除认证器自身那一处，此处更正）；且组合认证器 `compose_certification` / `pre_install_reverify` **在 `d3_*` 模块之外没有任何调用方**。
- **裁定**：`Ω 成员性在提交时被保证` = **NOT_ESTABLISHED**。

---

## 3. F-a / F-b / F-c：最小反例与硬门覆盖

### 3.1 F-a（开门升温是否被硬门覆盖）✅ **已闭合（源码级）**
- **最小反例构造尝试**：使开门升温把某温区推过硬上界，并让**后续段**以该温度计算速率——若该升温**从未被计数**，则 A2 垮。
- **源码判定**：`transition_segment` 在 `:243-244` 用 `_apply_door_event` **改写 `temperature_after`**，随后 `:298-299` 调用 `_thermal_violations(temperature_before, temperature_after, duration_h, contract)` —— **用的是开门后的值**；`_thermal_violations` 判定条件是 before/after **任一**越界（`:443-444`）；并且 `:302` 把开门后的值写入新状态，成为**下一段的 before**。
  ⇒ 开门升温**在同一段内即被计数**，且其后续影响也落在下一段的 `before` 检查里。**F-a 不产生未受检的越界**。
- **补充（重要且不利于"强制"叙事）**：`_validate_state`（`:316`）**不校验温度**（只校验维度/载重/容量/closed/唯一性/数量/品质界）⇒ 越界**不会 raise**，只被**累加计数**，最终由**评估器**判失败。这直接支撑 §3.4。
- **实测补充**：对 3 个实例插桩段级越界计数，**段级违规共 0 次**，且**没有任何开门后的 `after` 越出硬上界**。⇒ F-a 的覆盖结论**由源码给出**，其失效模式在实测中**未发生**（若发生则会被计数）。**但这也意味着该分支未被压力测试到**——属"已证明覆盖、未经验证触发"。

### 3.2 F-b（`_force_return_all` 与地平线）⚠️ **机制存在，Ω 定义已排除，但仅事后**
- **锚点**：`strict_online_env.py:658-664`（在 `tw_end[0]` 之后返仓）、`:596`（`horizon = tw_end.max()`，**不是** `tw_end[0]`）。
- **最小反例**：某车在 `tw_end[0]` 之后仍载货 → `_run_loop` 继续（`horizon` 更大）→ 直到 `_force_return_all` 或服务完成才返仓 ⇒ **该轨迹确实可能交付晚于 `tw_end[0]`**，此时 `τ^max` 的上界论证**失效**。
- **判定**：这类轨迹**不在 Ω 内**（`depot_return_violation`），因此**不构成对条件性证明的反例**；但它**构成对"生产运行必然属于 Ω"的反例**。⇒ 与 §3.4 同一结论。

### 3.3 F-c（速度/行驶时间口径）✅ **已闭合（本配置下）**
- **最小反例**：若 `units.speed_kmph / units.distance_km_per_unit ≠ tw_speed`，则认证用的到达时刻与 env 实际执行不同 ⇒ 品质轨迹预测错误 ⇒ `τ^max` 与品质重放都可能失真。
- **实测**：`speed_kmph=1.0`、`distance_km_per_unit=1.0` ⇒ 认证速度 = **1.0** = env `tw_speed=1.0` ⇒ **一致**。
- **限定**：这是**配置相关**的一致性，不是结构性保证。若将来 `tw_speed` 或 `units` 改动，**必须重检**。登记为条件而非已证定理。

### 3.4 ⚠️ 本轮最重要的结构性发现：**Ω 成员性是事后评估，不是提交时强制**
- `_validate_state` 不校验温度（§3.1）；越界只被**计数**，由评估器在**运行结束后**判 `temperature_hard_feasible`。
- `_force_return_all` 允许在 `tw_end[0]` 后返仓并被事后标记（§3.2）。
- 硬门调用点全部位于评估/校验/诊断脚本（A5）；组合认证器**无生产调用方**；`certify_plan` 有 **7 处**非测试路径绕过组合门（8 个调用点减去认证器自身，见 A5）。
⇒ **结论**：`Ω` 是一个**评估层定义的集合**，**不是提交层强制的集合**。因此
1. **条件性证明成立**（ω ∈ Ω ⇒ 下界成立）；
2. **"提交的计划会属于 Ω"未被证明**（也未被证伪——只是**没有任何提交入口承担这个职责**）。
这正是你提醒的"不能因为条件被写进 Ω 就宣称生产运行必然满足它"。**这个缺口无法靠下界本身的数学补上，必须靠一个提交时硬门（D-B3A-6 的"唯一组合提交门"）来补。**

---

## 4. 已在舱 lot 与尚未取货订单：分开口径 ✅ 且**两分支均已实测**

| 分支 | `Q^start` | 在舱时长 | 代码依据 | 实测覆盖 |
|---|---|---|---|---|
| **已在舱** | 真实 `quality_remaining` | `t_delivery − t_load`，其中 `t_load ≤ now` | manifest 中 lot 的真实品质 | **34 行** |
| **尚未取货** | `initial_quality` | `t_delivery − t_load ≤ tw_end[0] − now` | 取货前**不在 manifest**，故不衰减（`:397` 只遍历 `cargo_manifest`；新 lot 在 `:270-278` 以 `quality_remaining=initial_quality` 加入） | **222 行** |

**BND 已从单点扩展为全部 14 个决策点**（`--heavy-cap 14`）：**256 行下界、两分支均覆盖、0 违规**；`τ^max` 在采样点为 24.0 h，`per_class_T_max = {0:30, 1:8, 2:-12}`，最紧比值下界 0.9316。
**覆盖局限（必须写明）**：与交付记录的匹配只能覆盖"出现在下界行里的订单"；对 50 条交付记录而言，**未覆盖的订单未被检验**；且仅 **1 个实例**。

---

## 5. 范围声明：冻结计划 vs 允许重规划（你要求必须声明）

**结论：静态下界对二者都成立，且这一点是结构性的，不是巧合。**
- `T^max` 是**合同常量**（`hard_temperature_bounds_c`），**不依赖计划**；
- `τ^max = tw_end[0] − now` 只用**返仓硬截止**，**不依赖计划**；
- `Q^start` 取自**当前真实状态**（在舱）或**取货时的初始品质**（未装），也**不依赖计划**。
⇒ 无论后续是否发生 reveal 与合法重规划、无论哪条路线被执行、无论某订单被推迟到何时取货（只要 ≥ now），不等式都成立——**前提是该次执行属于 Ω**。
**这正是静态下界在"最晚出发最差"被证伪之后仍然存活的原因**，也是它比任何"计划内最坏时刻表"更强的地方。
**唯一的前提是 §3.4**：Ω 成员性未被提交时强制。因此**重规划本身不破坏下界，破坏它的是"执行结果不属于 Ω 却未被拒绝"**。

---

## 6. 独立只读复核

按你的第 5 项要求，本轮由**独立只读复核者**（单独进程/无我的先验结论）对上述 A1–A5、F-a/F-b/F-c、§3.4 与 §5 逐条裁定，产出 `B3QP_ADJUDICATION.md`，其结论**原样收存、不做修饰**。

---

## 7. 停止条件与后续

**本轮已完成**：① 固定适用范围与合同版本（未收缩轨迹集合）；② A1–A5 的锚点/证明/调用路径；③ F-a/F-b/F-c 的最小反例与覆盖判定；④ 两分支分开检验且均已实测；⑤ 独立复核（另见 `B3QP_ADJUDICATION.md`）。
**未做**：真实安装、`_commit` 修改、解除 D8、引入 `PROJECTED_SATISFIED_CONSERVATIVE`、收紧 `τ^max`、冻结数据/最终种子。

**下一步（按你的顺序，需单独授权）**
1. **补上唯一缺口**：为 D3 目标生产路径建立**唯一组合提交门**，在提交时强制执行 Ω 的硬门（热/时间窗/容量/返仓/承诺）。这**不是数学问题**，而是把"事后评估"变成"提交时拒绝"——完成后 A4/A5 才能真正闭合，D8 才谈得上解除。
2. **再单独预声明更紧的 `τ^max`**（先证明粗上界成立，再收紧；不得把"安全性正确"与"可用性改善"混为一件事）。
3. **D-B3A-7 合同变换独立只读验证器**（低成本）。
4. **暂缓**：D-B3A-3、消除既有 8 处旁路、**B3b 真实安装**。

**研究状态不变**：D3 G0 未通过；G1/G2 未启动；无可归因的 F−D 学习增益；I3 正式负结果 / 旧 D2 `not_qualified` / `stable_online_gain=False`。
