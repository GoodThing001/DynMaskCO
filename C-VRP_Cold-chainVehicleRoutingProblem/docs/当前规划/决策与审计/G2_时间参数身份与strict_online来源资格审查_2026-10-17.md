# G-2 时间参数身份与 strict-online 来源资格审查 · 独立审计报告（2026-10-17）

> **性质：资格审查（qualification），不是新的学习实验，不是新的正式执行。**
> 预声明先于实施登记于 `scripts/q3_qualification/q3_g2_preregistration_2026-10-17.md`；
> 本报告只报告该预声明所列的判据与负控结果，**不新增判据、不放宽判据**。
> 本轮只授权 G-2：未启动 G-1/G-3/C2/G5/G6/C3，未改生产认证器与安装入口，
> 无训练 / GPU / 正式 rollout / 最终测试种子。**未自动 commit/push。**
>
> 本轮的目标不是再拿一个漂亮的 PASS。**允许的结论包含 `NOT_CERTIFIABLE` 与 `REFUSED`**，
> 而且本轮确实产出了这样一条结论（§4.3、§7）。

| 项 | 值 |
|---|---|
| 基准 HEAD | `a71a0c3cede64b15e20f34f7833134d6236bb7a1` |
| 合同 | `default_pilot_contract()`，`contract_hash = 32124bfd4ef73169…` |
| 本轮产出 | 预声明 1 + 身份模块 1 + 控制测试 1 + 本报告 1 |
| 控制结果 | **21/21 PASS，0 SKIP，0 FAIL → `G2_QUALIFICATION_PASS`** |
| 证据 | `results/_q3_next_20261017/g2/controls.json` |

---

## 0. 上一轮遗留记录的两处边界（用户核对，改由本报告承接）

1. **提交链更正**：从 `7ba7300` 到 `a71a0c3` 是**五个**提交
   （`1f9f662` → `0378ce1` → `c09b656` → `427ad54` → `a71a0c3`），其中
   **`1f9f662` 才是移除调用方对象保留的源码提交**；我上一轮汇报里只列了其后四个复现性收尾提交。
   以此处为准。
2. **V-10 的 SKIP 是明确的证据缺口**：它需要四个**未提交**的本地 Q3 套件，故干净 clone 只能
   `INCOMPLETE`。按用户裁定，**不再为这一项追加 G-TMIN 测试**；它留在资格缺口清单里，
   如需正式独立资格再单独授权补齐。`PRE_EXISTING_FAILURE_UNCHANGED` 继续保留
   （`omega10_initialization` 在修复前后同样失败，不能算回归通过）。

---

## 1. 代码身份（执行前登记，全部取**提交 blob**摘要）

```
C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/strict_online_env.py = fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6
C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_state.py = a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e
C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_contract.py = 0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2
C-VRP_Cold-chainVehicleRoutingProblem/scripts/m1b_adapter/m1b_quality_certificate.py = 2aa801d09c26e6851e75ca4c1672ff9f84f1d59d3e5b95d402fcfc8f96ba4d22
C-VRP_Cold-chainVehicleRoutingProblem/scripts/m1b_adapter/m1b_q2_certifier.py = 7674205b34db5fc5ec3acf226acfc972158dcdd3a0b8c61bb2c5ea607a0e1a97
```

basis = **committed blob**（`git show HEAD:<path>`）。认证器摘要与 G-TMIN 勘误登记的最终身份**逐字相同**
⇒ 本轮**没有**改动认证器（控制 C11）。

调用图普查摘要：`eea601d59b306ad1853923d875af17b5b8c30751b90929038cfd881596f17929`
身份摘要：`5fcf1e03966d1aa4fedab1f22ca262d92319ba95d3a67e8e12db5596b85c0d69`

---

## 2. 审查结论一：三层可见性必须分开，而 `horizon` 属于第一层

`strict_online_env.py:454`（`run_resumed` 在 `:596` 同式）：

```python
horizon = float(self.tw_end[inst_idx].max())
```

`tw_end` 是 `(N, nodes)` **完整**数组。**该值在定义上就是未揭示订单截止时间的函数**，因此：

| 层 | 允许知道什么 | 本参数是否属于该层 |
|---|---|---|
| **SIMULATOR-TRUTH**（推进真实世界） | 完整 `tw_end`、完整 `reveal_time`、未来服务完成 | **是**：`horizon` 在这里，合法 |
| **ONLINE-DECISION**（当前决策点可用） | 已揭示订单、公开预约截止、**depot 返回截止 `tw_end[0]`** | 否（除非另有见证） |
| **CERTIFIER-PROOF**（证书可依赖） | 认证器可从合规输入重算的量；借用别的层的量**必须带见证** | 否（除非另有见证） |

**关键区分（本轮核心）**：模拟器为了推进事件**可以**持有隐藏数据；认证器**不可以**把同一数值当作
"已证明的界"。二者是**不同权限**，不能因为"数值一样"就合并。

### 2.1 泄漏是否为空：可机械见证，而不是"数据集里恰好相等"

生成器不变量（`generate_coldchain_data.py:125,131`；`generate_cvrptw_data.py:117,129` 同型）：

```python
tw_start[0], tw_end[0] = 0.0, horizon            # depot 恰为 episode horizon
tw_end[i] = min(horizon, center + width / 2)     # 每个客户被钳制在 horizon 内
```

⇒ `max(tw_end) == tw_end[0] == horizon`，**结构上**成立。实测（控制 C3，只读）：

| 数据域 | 实例数 | `max(tw_end) == tw_end[0]` | 最晚**隐藏**客户截止 |
|---|---|---|---|
| `data/m0_scale/*.npz`（cal 16 + dev_check 16 + train 64） | **96** | **96/96 成立** | **19.5755 < 24.0** |

**分级：`PROVED_CONDITIONAL`。** 条件是"生成器不变量 + 逐实例见证"，
**不是**"定理"，也**不是**"实测所以安全"。一旦某实例出现 `max(tw_end) > tw_end[0]`，
见证失效 ⇒ 该实例的 `horizon` 判 `HIDDEN_DERIVED` ⇒ `NOT_CERTIFIABLE`（负控 N-02 实测触发）。

### 2.2 见证机械化

`classify_horizon()` 对**每个实例**给出 `{tw_end_max, depot_tw_end, hidden_customer_max, witness_equal}`；
`require_public_horizon()` 在 `witness_equal` 为假、或申报值 ≠ `tw_end[0]` 时**拒绝**。
因此"这个界是公开的"不再是一句注释，而是一条可重算的判定。

---

## 3. 审查结论二：`max_iter = 10000` **不是**每辆车温度推进次数的上界（旧解释**低估约 2 倍**）

### 3.1 旧解释

本地试验资产 `q3_time_budget_audit_experimental.py:86-92` 写：

> "`_run_loop` bounds the number of iterations, and every iteration performs at most one advance per
> vehicle, so this is a structural bound on the number of advances" ⇒ `MAX_ITER = 10000`

**该推理不完整**：`_commit` 也会推进（`:272`、`:286`），而 `_plan_and_commit` **每次迭代调用一次**
（`:545`），循环前还有一次（`:469`），终止分支 `_force_return_all`（`:507`）另有两次。

### 3.2 本轮从**静态调用图**重新推导（不执行、不硬编码）

`call_graph_census()` 用 AST 解析 `strict_online_env.py`，得到三处推进调用点与三条结构路径：

| 贡献 | 来源 | 每次/每车次数 | 见证 |
|---|---|---|---|
| `pre_loop` | `run` / `run_resumed` 在循环**前**调用 `_plan_and_commit` 一次 | **1** | `:469`、`:602` |
| `per_iter` | 每迭代一次 `_advance_fleet`（`:519` → `:304`） + 一次 `_commit`（`:655` → `:272`/`:286`） | **2** | 三点均被普查发现 |
| `termination` | `_force_return_all`：`_commit`（+1）与单车 `_advance_fleet`（+1）；调用点紧接 `break` ⇒ 至多一次 | **2** | `:507` + `break` |

**互斥性不是假设，而是见证**：`_commit` 的 `if node == 0:` 分支以 `return` 结束（`:279`），
故 `:272` 与 `:286` 不可能同时执行；普查记录该见证
（`exclusive_via_return_at: 267`）。**见证消失时上界自动变大**（控制 C4b 实测）：
额外插一个推进点 ⇒ `n_max` 20003 → **30004**；删掉 `return` 见证 ⇒ → **30005**。

### 3.3 结果

```
n_max_per_vehicle = 1 + 2 * n_iter + 2 = 1 + 2*10000 + 2 = 20003        (旧：10000)
n_iter <= min(max_iter, 1 + floor(T_max / EPS_EVENT))                   (双约束)
```

| `T_max` | 迭代上界 | 绑定侧 | `n_max` | 区间总 slack（h） |
|---|---|---|---|---|
| `1e-3` | **1001** | **time**（10000 过松） | 2005 | 1.0002e-09 |
| `24` | 10000 | max_iter | **20003** | **1.0711e-09**（旧解释：1.0355e-09） |
| `1e6` | 10000 | max_iter | 20003 | 2.3298e-06 |

**诚实结论**：
- 数值上**旧结论存活** —— 1.07e-9 h 仍比 `Δ* ≈ 0.04 h` 小四个数量级以上，L1 预算结论不变；
- 但**推导不成立** —— 旧 `n` 是**低估**（10000 vs 20003），而预算是**上界**，
  低估即方向不安全。本轮把它换成本轮导出的 20003，并把"用 10000"设为**拒绝**路径
  （`INSUFFICIENT_COUNT_BOUND`，负控 N-07）。

### 3.4 未建立的残余（如实标注 `UNKNOWN`）

- 推导是**上界**，不是紧界：一次迭代里 `_commit` 由于互斥只算 1，但 `_advance_coldchain_to`
  在 span ≤ 1e-9 时**早退**（不计入真实推进）。真实同实例合成观测（控制 G2-SYN，5 节点 2 车）
  为每车 **8** 次、迭代 **6** 次，远低于 20003 —— 但这只是 `DIAGNOSTIC`，**不能**用它把上界改小。
- 若将来有人把 replanner/hook 改成**直接调用** `env._advance_coldchain_to`，该上界会失效：
  当前全仓只有 `d3_execution_authority.py` / `d3_write_exclusion.py` 的**包装器**，它们转发原函数、
  不新增推进；已登记为需要复检的接缝。

---

## 4. 审查结论三：四类时间参数的**字段—来源—可见性—认证资格矩阵**

| # | 字段 | 定义来源 | 可见性 | 是否携带未揭示信息 | 认证资格 | 资格缺失时 |
|---|---|---|---|---|---|---|
| 1 | `horizon / T_max`（`:454, :596`） | `tw_end.max()` 全数组 | **SIMULATOR-TRUTH** | **是** | **仅在有见证时**可用于在线认证；替代量 = `tw_end[0]` | `NOT_CERTIFIABLE`（`HIDDEN_DERIVED_HORIZON`） |
| 2 | `tw_end[0]`（被采纳的时域上界） | depot 返回截止时间窗 | **ONLINE-DECISION** | 否 | 可用（前提是第 1 行见证成立） | 无替代量，`NOT_CERTIFIABLE` |
| 3 | `n`（每车温度推进次数） | **静态调用图**导出；`max_iter` 只是循环上限 | CERTIFIER-PROOF | 否 | 用 `1+2·n_iter+2`，否则拒绝 | `REFUSED`（`INSUFFICIENT_COUNT_BOUND`） |
| 4 | `hours_per_time_unit`（`contract:36, :521`） | `UnitScale`，**在 `contract_hash` 内** | CERTIFIER-PROOF | 否 | 仅当**恰为 1.0** | `NOT_CERTIFIABLE`（`TIME_UNIT_CONVERSION_NOT_BUDGETED`） |
| 5 | `eps_skip = 1e-9`（`:219`） | 字面量 | CERTIFIER-PROOF | 否 | 可用；进预算（每区间一个常数） | 摘要变动 |
| 6 | `eps_backward = 1e-6`（`:217`） | 字面量 | CERTIFIER-PROOF | 否 | 可用；**不进**预算 | 摘要变动 |
| 7 | `eps_event = 1e-6`（`:306,:490,:506,:514`） | 字面量 | CERTIFIER-PROOF | 否 | 可用；**不进**预算 | 摘要变动 |

**为什么第 6、7 行不进预算**（必须写清，否则等于漏项）：`eps_backward` 只决定"倒退多小可以静默跳过"
—— 跳过时**锚点不动**，故不会重复计入冷却；`eps_event` 只移动**事件时刻**，而一个区间内的时长**望远镜求和**
恰为 `clock_next − clock_prev`，与事件具体何时触发无关。

**为什么只有 `eps_skip` 进预算**：它是唯一"少算了一段时长"的路径（`end <= start + 1e-9` 直接 return），
故按"每区间一个常数"计入，与旧审计一致。

---

## 5. 唯一主对照（公开替换对）结果

**A 臂 = 现状参数**（`tw_end.max()`, `n = max_iter`）；**B 臂 = 公开替换**（`tw_end[0]`, 导出上界）。
判据只有方向：**B 不得更宽松**。96 个登记实例全体（控制 C1）：

| 量 | A 臂 | B 臂 | 更宽松数 |
|---|---|---|---|
| `H_max`（h） | `24.000001` | `24.000001` | **0/96** |
| `n` | 10000 | **20003** | 0/96 |
| 区间 slack（h） | 1.0355e-09 | **1.0711e-09** | 0/96 |
| 每类 `q_lb`（见下方口径说明） | `8.54807998117` | `8.54807998117` | 0/96 |

**`q_lb` 的口径必须写清（本轮发现的第四个缺口）**：默认 pilot 合同是 **v1 `ColdChainContract`**，
而 `certify_quality_lb` 的入参来自 `check_quality_classes_are_certifiable`，它返回
**`CONTRACT_QUALITY_CONFIG_UNSUPPORTED`（`quality config is not QualityConfigV2`）** ——
**即：用默认合同根本无法评估第 7 道质量门**。因此上表 `q_lb` 一行的数值是用**声明过的合成常量**
（`q0_safe = 9`、`rate_per_hour = 0.01883`、`threshold = 5`，即鲻鱼鱼类零阶率）跑出来的，
**不是**合同常量；两臂 `H_max` 逐位相同 ⇒ `q_lb` 必然相同（`q_lb = max(0, q0 − k·H_max)`
对 `H_max` 单调）。该数值与 G-TMIN 记录的 `8.54807998017` 在第 9 位小数不同，原因是
`0.01883` 是合同精确率的**四舍五入写法**，不代表任何新结论。

**读法**：`H_max` 与 `q_lb` 两臂**逐位相同**（因为见证成立），故**隐藏派生的那个值对证书不是承重的**；
唯一变化是 `n` 变大（更保守）。这正是"泄漏为空 + 预算被修正"的组合结论，
而**不是**"所以可以不看来源"。若某实例见证失效，B 臂根本不存在（`NOT_CERTIFIABLE`）。

---

## 6. 共享时间身份规范草案（`G2_TIME_IDENTITY_V1`）

### 6.1 对象

```
{
  schema/version/not_for_production/production_consumable,
  contract: {contract_hash, hours_per_time_unit, distance_km_per_unit},
  code:     {files: {5 个源文件 blob 摘要}, basis, max_iter, census_digest},
  time_unit:{hours_per_time_unit, identity_bound, certifiable},
  horizon:  {value, defined_source, admissible_source, classification,
             witness_equal, leak_magnitude},
  advance_budget: {n_max_per_vehicle_at_max_iter, per_iteration, pre_loop, termination,
                   formula, iteration_formula, inherited_reading_rejected},
  tolerances: {eps_skip, eps_backward, eps_event, in_slack_budget, not_in_slack_budget, why_not},
  domain:   {dataset_glob, n_instances, splits, horizon_value},
  identity_digest
}
```

`identity_digest = sha256(规范 JSON)`，覆盖**全部**字段。控制 C9 实测：重建得同一摘要；
单独改 `max_iter`、改 `horizon.value`、改任一源码摘要 ⇒ **三个摘要全部改变**。

### 6.2 判定阶梯

| 情形 | 判定 | 实测 |
|---|---|---|
| 时域不是上界（观测时钟越界） | `REFUSED: HORIZON_NOT_AN_UPPER_BOUND` | N-01 |
| 见证失效（隐藏订单截止更晚） | `NOT_CERTIFIABLE: HIDDEN_DERIVED_HORIZON` | N-02 |
| 单位非 1.0 | `NOT_CERTIFIABLE: TIME_UNIT_CONVERSION_NOT_BUDGETED` | N-03 |
| 区间内发生快照恢复 | `REFUSED: SNAPSHOT_RESTORE_INSIDE_INTERVAL` | N-04 |
| 推进次数越界 | `NOT_CERTIFIABLE: COUNT_BOUND_VIOLATED` | N-05 |
| 身份漂移 | `REFUSED: IDENTITY_MISMATCH` | N-06 |
| 沿用 `n = max_iter` | `REFUSED: INSUFFICIENT_COUNT_BOUND` | N-07 |
| 时域非有限 / 非正 | `REFUSED: HORIZON_INVALID` | N-08 |
| 单位合同配旧身份 | `REFUSED`（`IDENTITY_MISMATCH` + 单位码） | N-09 |

**九条负控全部命中（9/9），无一条静默通过。**

### 6.3 适用范围（必须随身份一起引用）

`domain` 只覆盖 `data/m0_scale` 三个 split（96 实例，horizon 24.0）与
`default_pilot_contract()`（`hpu = 1.0`）。**换数据集、换合同、换 `max_iter`、改 `strict_online_env.py`
任一处，身份即失效**，必须重建并重新登记。

---

## 7. 最重要的负面结论：认证器的时域参数**目前没有生产来源**，在线资格 = `UNKNOWN`

这不是本轮的推测，是两条源码事实（预声明 F-7/F-8，控制 C11 复核）：

1. `M1bQ2TrustFixCertifier` **零生产调用点** —— 全仓库只在测试里构造；
2. `T_end` 的全部实际取值是测试字面量 `24.0`；正式入口 `d3_l1_l1_formal_run.py:469` 传
   `certify_fn=None`（即根本不认证）。

**因此**："以 `T_max = 24` 认证过某状态"这件事，**当前没有一条生产链把它绑定到实例**。
本轮**没有**修这个缺口（不属 G-2 授权范围：要动认证器/安装入口），只把它的**形状**写清楚：

| 缺口 | 状态 | 修复所需 |
|---|---|---|
| `T_end` 无生产调用点 | **`UNKNOWN`** | 需要一条调用链：实例 → `classify_horizon` 见证 → 认证器；属 G-1 或单独授权 |
| `certify_fn=None` 跳过认证 | **`UNKNOWN`**（既有 G0 A-2 缺口，未变） | 接线时必须同时保留 fail-closed |
| **默认 pilot 合同是 v1，第 7 道质量门要求 `QualityConfigV2`** ⇒ `check_quality_classes_are_certifiable` 返回 `CONTRACT_QUALITY_CONFIG_UNSUPPORTED`，**默认合同无法喂 `certify_quality_lb`** | **`UNKNOWN`（本轮新发现）** | 需要一份**已通过身份认证的 v2 合同**；在此之前，任何"用默认合同算出的 `q_lb`"都只是分析，不是证书 |
| V-10 取代行台账不可复现 | **已知证据缺口** | 需提交四个本地 Q3 套件（用户已裁定暂不追加） |

**分级边界**：`PROVED_CONDITIONAL` 只用于"依赖生成器不变量 + 逐实例见证"的命题；
`DIAGNOSTIC` 用于合成观测（G2-SYN）；`UNKNOWN` 用于上面三项。
**没有**任何一条 `PROVED`（无条件定理）。

---

## 8. 与"是否具备下一阶段实施资格"的结论

**结论：G-2 通过，且下一阶段（G-1 精确时间身份）具备实施资格——但有两个前置约束。**

1. **可以进入 G-1**：身份对象、见证机制、调用图上界、九条负控均已就位，
   G-1 只需把 `T_end` 从"测试字面量"变成"身份对象里的 `horizon.value`"，并在认证器**外侧**
   接线（不改认证器算术）。
2. **前置约束 A**：G-1 **必须**先解决"零生产调用点"，否则 `T_end` 仍然是别的层的数；
   建议的最小形态是 `certify_fn` 工厂：见证不成立就返回拒绝函数，而不是返回 `None`。
3. **前置约束 B**：`n` 必须使用本轮导出的 20003（或按当时的调用图重算），
   **不得**再回到 `max_iter`。这条已由 `INSUFFICIENT_COUNT_BOUND` 强制。
4. **前置约束 C（新增发现）**：任何涉及 `q_lb` 的资格讨论，都要先有一份**已认证的 v2 合同**；
   用默认 v1 pilot 合同算出来的 `q_lb` 只能标注为分析结果。否则 G-1 会把一个
   "合同层面根本不可评估"的量当作已绑定。

**不建议**在本轮结论之上做任何扩张：不启动 G-3/C2/G5/G6/C3；不改认证器；不追认任何正式资格。

---

## 9. 需要用户签署的事项（逐条说明为什么必须由人决定）

| # | 事项 | 为什么必须人决定 | 建议 |
|---|---|---|---|
| **S-1** | 是否接受 **`n = 20003`** 取代 `max_iter = 10000` 作为推进次数上界 | 这会**改变一个已登记的预算推导**（虽然数值结论不变）。改动既有登记属于签署事项，不是工程选择 | 接受（旧值是低估，方向上不安全） |
| **S-2** | 是否接受 **"`horizon` 在见证成立时可用，否则不可认证"** 这一资格口径 | 它决定了以后能否用 `tw_end.max()`；等价于承认"数值相等不等于权限相同" | 接受 |
| **S-3** | 是否授权 **`T_end` 生产接线**（G-1 前置约束 A） | 涉及**认证器外侧的生产调用链**与 `certify_fn` 语义，属安装/认证路径 | 建议**单独一轮**授权，不在 G-1 内顺手做 |
| **S-4** | 是否要求 **G-1 在接线后重跑全部 L1 预算**（新 `n`） | 会触发"是否产生新证据身份"的判断；本轮**不**重跑任何正式批次 | 建议要求重算预算、**不**重跑实验 |
| **S-5** | V-10 的四个本地套件**是否入库** | 入库会扩大公开证据面（且其中含试验资产），属用户裁量 | 暂不入库，留作已知缺口 |
| **S-6** | 是否接受本轮**不**把 `NOT_CERTIFIABLE`/`UNKNOWN` 视为"G-2 未通过" | 这是判据语义问题，必须事先由人认可，否则下一轮会被读成失败 | 接受：三态是设计目标 |
| **S-7** | 是否要求**先取得一份已通过身份认证的 v2 合同**，再讨论任何 `q_lb` | 默认 pilot 合同是 v1，第 7 道门直接 `CONTRACT_QUALITY_CONFIG_UNSUPPORTED`；在此之前 `q_lb` 只能做分析，不能当证书 | 建议要求先出 v2 合同身份，再谈 `q_lb` 的资格 |

---

## 10. 状态（全部不变）

```
GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED   （永久，不追认）
GTMIN 正式验收 FAIL / 未闭合                           （不变）
THERMAL_CONTINUOUS_NOT_ESTABLISHED
Q3_FORMAL_FREEZE_NOT_AUTHORISED
M2_NOT_AUTHORISED
L1 BLOCKED
L2_NOT_ESTABLISHED / A3_NOT_ESTABLISHED
stable_online_gain = False
HISTORICAL_BACKBONE_UNVERIFIED
C0 AWAITING_USER_BUDGET_RULING
t_min 恒 0（无可信见证机制）
最终测试种子 SEALED
```

**本轮未做**：G-1 / G-3 / C2 / G5 / G6 / C3；未改生产认证器与安装入口；未训练；无 GPU；
未跑正式 rollout；未碰最终测试种子；**未 commit / 未 push**。

**明确不主张**：Q3 连续温度认证已建立；时间参数已在生产链路上绑定；`max_iter` 曾是正确的上界；
V-10 台账可复现；或本轮任何一条无条件 `PROVED`。
