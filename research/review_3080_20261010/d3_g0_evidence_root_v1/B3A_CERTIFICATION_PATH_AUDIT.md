# B3a：认证路径与旁路审计（只读）

**日期**：2026-10-14 · **基准 commit**：`3038a16` · **性质**：只读审计。
**用户保留 1**：SCOPE 测试通过**不等于**完整证明 strict-online 无泄漏；必须
(a) 审计组合认证器的**传递依赖**，特别是 `c0_completion_energy` 对 `reveal_time` 的读取；
(b) 确认所有认证分量**只能收到经过合法性检查的订单集合**，且**不存在绕过组合入口的认证路径**。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 传递依赖里是否有**未来信息**读取？ | **有且仅有一处**：`c0_completion_energy:127` 读 `env.reveal_time`。其余分量只读**静态订单属性**与当前状态 |
| 认证分量是否**只收到合法订单集合**？ | ❌ **现状不成立**：补偿项（含"已揭示"）位于四项检查**之后**，`certify_plan` 可能在未验证的计划上被调用 ⇒ 列为 **B3b 必做项 D-B3A-1**（短路顺序） |
| 是否存在**绕过组合入口**的认证路径？ | ❌ **存在**：`certify_plan` 在活跃树里有 **8 个非测试调用点**直接调用它。B2 组合认证器**目前没有任何生产调用方** |
| `certify_route` 是否只服务于认证？ | ❌ 它同时服务于**候选枚举**（`action_contract.py:399`/`:417`）⇒ 改其语义会影响枚举，属耦合 |

---

## 1. 组合认证器的传递依赖（逐函数，含读取的 env 字段）

| 分量 | 函数 | 位置 | 读取的 env / 状态 | 是否含未来信息 |
|---|---|---|---|---|
| 1 计划对账 | `validate_repair_plan` | `mpre_policy.py:83-132` | **不读 env**（只用 `plans` / `base_plans` / `mask_set` / `protected`） | 否 |
| 2 所有权分区 | `validate_full_partition` | `dynmaskco_cc_context.py:180-230` | **不读 env**（`served_mask` 由调用方传入；`universe` 由调用方按 `reveal_time<=clock` 构造） | 否（且构造 `universe` 用的是"已揭示"判据，不是未来） |
| 3 路线可行性 | `certify_route` | `action_contract.py:142-203` | `env.capacity`、`env.coldchain_contract.thermal.supported_temp_classes`、`env.temp_class`、`env.tw_end`、`env.tw_start`、`env.demands`、`env.service_time`、`env.dist_mat` | **否**——全部是**静态订单属性**与静态矩阵，与揭示时刻无关 |
| 4 能耗预算 | `certify_plan` → `c0_completion_energy` | `run_exp_energy_c0.py:154-199` / `:112-151` | 同上，**外加 `env.reveal_time`（`:127`）**、`env.booking_horizon`（`:166`）、`env.dataset['temp_class']`、`v.coldchain_state`/`status`/`committed_next`/`current_node` | **是：唯一一处** |
| 补偿 | `_compensations`（本模块） | `d3_composed_certifier.py:151-167` | `env.reveal_time`（判"已揭示"）、`v.status` | 否（用 reveal_time 做**合法性**判断，不是拿未来信息当输入） |

**对唯一那处的判定**（与 B3a-P0 §4 一致）：
- 对**已揭示**订单惰性（`max(t, rev) == t`）；
- 但**不会拒绝**未揭示订单 ⇒ 若合法性未前置，认证会**静默**按未来揭示时刻排程；
- ⇒ 缓解措施不能只是"记录风险"，必须是**结构性的短路顺序**。

**审计方法说明（可复核）**：上表由逐函数读源码得出；`reveal_time` 的检索覆盖 `d3_composed_certifier.py` / `d3_credential.py` / `d3_install_gate.py` 及其四个被调函数的函数体。**未发现其它未来信息读取**（无 `visible_mask` 之外的揭示依赖、无未来需求/时间窗的"预测性"读取）。

---

## 2. 合法性前置：现状**不成立**（B3b 必做）

B2 的实现顺序是：
```
1 repair_plan → 2 full_partition → 3 route_feasible → 4 energy_budget → 补偿(revealed_only/served/…)
```
因此 `certify_plan`（分量 4）**可能在含未揭示或已服务订单的计划上被调用**，而其内部会
- `:127` 按 `reveal_time` 静默推后发车（未来信息）；
- `:173` 静默丢弃已服务订单。

⇒ **要求**：改为**短路顺序**，合法性检查**全部通过后**才允许调用路线/能耗/品质分量：
```
0 plan_vehicle_set → 1 revealed_only → 2 no_served_in_plan → 3 no_duplicate_order
  → 4 full_partition → 5 repair_plan → 6 route_feasible → 7 energy/quality
```
未通过时，路线/能耗/品质分量**不得被调用**，并在结果里记为 `not_run_illegal_order_set`（与 D-B3A-4 的 Q5 对应）。
**这是对已收束 B2 模块的修改，需单独授权。**

---

## 3. 旁路审计：**组合入口不是唯一认证路径**

`certify_plan(` 在活跃树中的调用点（排除定义与归档副本）：

| 文件:行 | 性质 |
|---|---|
| `evaluation/rolling_optimizer.py:197` | 生产滚动优化器 |
| `evaluation/rrnco_accept_replanner.py:150` | 接受决策 replanner |
| `evaluation/solver_accept_replanner.py:424` | OR-Tools 接受 replanner |
| `evaluation/run_exp_energy_c0.py:302` | `C0ReserveReplanner._try_insert_certified`（计划级回滚包装） |
| `evaluation/run_a1_downstream_ablation.py:88` | 消融实验入口 |
| `research_20261010/pickup_budget_sweep.py:96` | 预算扫描 |
| `research_20261010/pickup_rejection_attribution.py:97` | 诊断 |
| `simulation/d3_composed_certifier.py:156` | **B2 组合认证器（本轮新增）** |
| `tests/test_a1_saa_correctness.py:147/161/1701` | 测试 |

**事实**：
1. **8 个非测试调用点**直接调用**只查能耗**的 `certify_plan`，它们**不经过** B2 的组合门，也**不做**身份绑定。
2. **B2 的三个模块（`d3_credential` / `d3_composed_certifier` / `d3_install_gate`）目前没有任何生产调用方**——与 `d3_geo_adapter` 相同：只有我自己的测试引用。
3. 另有一族**间接**调用：继承 `C0ReserveReplanner._try_insert_certified` 的 replanner（`run_exp_rule_control.py`、`scenario_saa.py`、`a1_strong_controls.py`、`train_a1_rollout_v.py`、`i3_counterfactual_teacher.py` 等）——它们同样绕过组合门。

⇒ **"不存在绕过组合入口的认证路径"在当前代码库里是假的。**
这不否定 B2 的价值（它证明**组合门本身**可用），但必须记录：**接入生产路径**是 B3b 的工作，不是既有事实。
**不得**在任何后续文档里把"我们实现了组合认证器"写成"所有认证都走组合认证器"。

`certify_route(` 的调用点同样重要：`action_contract.py:399` 与 `:417` 位于**候选枚举**内部（`_enumerate_from_plans`），另加 `run_perishability_oracle.py:65`。
⇒ `certify_route` 是**共享原语**（枚举 + 认证），**改它的语义会同时改变候选集合**；B3b 若要加强它，必须同时重跑枚举相关回归。

---

## 4. 与用户保留 3 的接口

`contract_provenance.transform_check`（见 `B3A_SUBMIT_BOUNDARY_AND_RECEIPT_DESIGN.md` §4）必须在**提交入口**执行
`apply_objective_profile(C_raw, p).contract_hash == env.coldchain_contract.contract_hash`，
并在 `raw_contract_source == 'unavailable'` 时记 `NOT_VERIFIED`。
仅记录三个哈希**不构成**变换证明。

---

## 5. 交付给 B3b 的清单（本轮只列，不实现）

| 编号 | 事项 | 需授权类型 |
|---|---|---|
| **D-B3A-1** | 组合认证改为**短路顺序**，合法性前置；分量不得见非法订单集合 | 修改已收束的 B2 模块 |
| **D-B3A-2** | 新增只读品质投影 `project_plan_quality(..., worst_case_schedule=True)` | 新增 `simulation/` 模块 |
| **D-B3A-3** | `quality_salability` 增 `PROJECTED_SATISFIED_CONSERVATIVE`；`installable` 判据不变 | 修改 B2 模块 |
| **D-B3A-4** | 预注册 Q1–Q5 反例（见 B3a-P0 §6） | 测试 |
| **D-B3A-5** | 提交边界观察器（方案 a+b）与回执 schema 实现 | 新增模块；方案 (c) 需改 env |
| **D-B3A-6** | 把组合门接入生产 replanner 调用路径（消除 8 处旁路） | 生产路径变更，需单独授权 |
| **D-B3A-7** | `transform_check` 强制执行 | 修改提交入口 |

---

## 6. 本文件没有做的事

未修改任何模块；未接入生产路径；未改 `certify_plan` / `certify_route` / `_commit` / env；未启用写回；未启动科学实验。
