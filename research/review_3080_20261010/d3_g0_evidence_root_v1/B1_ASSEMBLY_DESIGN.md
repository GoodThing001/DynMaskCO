# B1 真实状态装配层设计（只读设计，不含实现）

**日期**：2026-10-14 · **基准 commit**：`51896c3` · **性质**：**只读设计**。本文件不实现任何东西，不改生产源码。
**授权边界**：按用户裁决，先审设计，之后**单独批准实现**；本批**不自动启动** F−D、E2、32 状态重采集、G1/G2 或最终测试。
**结论先行**：装配层所需的**零件基本都已存在**，缺的不是算法而是**边界**——一个把「状态 → 候选池 → 共同 legal → 评分 → 完整 C0 → 真实安装」串起来、并在安装点做身份校验与回执的事务边界。

---

## 0. 一句话诊断

现有 `d3_geo_adapter.py` 已经证明可以**拒绝**非法输入，但它**没有生产调用方**，而且它看到的只是"计划"这一层。
真实的状态提交发生在 `StrictOnlineEnv._commit()` 内部，**晚于** replanner 返回、**不在**适配器可见范围内，且 `_commit()` **不做任何可行性校验**（`strict_online_env.py:260-299`）。
所以"安装函数被调用"与"真实状态已提交"目前**在代码上无法区分** —— 这是 B1/B3 要解决的核心问题。

---

## 1. 已存在的零件（全部亲自读过，附 file:line）

| 层 | 函数 / 类 | 位置 | 作用 |
|---|---|---|---|
| 状态载体 | `VehicleState`（19 字段） | `strict_online_env.py:74-92` | status / current_node / ready_time / current_load / committed_next / committed_arrive / committed_finish / mutable_suffix / served_route / dispatch_time / return_finish / needs_replan / replan_reason / coldchain_state / coldchain_updated_time / committed_coldchain_before / coldchain_accounting_anchor |
| replanner 接口 | `Replanner.plan(...)` | `strict_online_env.py:95-109` | 契约：`[c1,…,0]` 连续服务后返仓；`[0]` 立即返仓；`[]` = WAIT（**env 只 commit 第一个**） |
| 决策点准备 | `prepare_decision_point(clock, vehicles)` | `:608-621` | **会改状态**：idle→`ready_time=clock`；ready→`max(ready_time, clock)` |
| 计划+提交 | `_plan_and_commit(...)` | `:623-656` | prepare → visible_ids → replan_ids → `replanner.plan` → force_suffix 覆盖 → **提交循环** `:650-656` |
| **真实提交** | `_commit(inst_idx, clock, v, traces, node)` | `:260-299` | 不可逆地写 `status/committed_next/committed_arrive/committed_finish/ready_time/coldchain_state/dispatch_time`；**无任何 TW/容量/能耗校验** |
| 预留客户 | `get_reserved_customers(vehicles)` | `:232-246` | committed_next + **所有车** mutable_suffix tail |
| 未来揭示 | `has_future_reveal(inst_idx, clock, served_mask)` | `:248-258` | WAIT vs CLOSE |
| **状态失效事件** | reveal 分支 | `:523-528` | `needs_replan=True`、`replan_reason='reveal'`、**`mutable_suffix = []`** |
| 快照钩子 | `_maybe_snapshot` → `snapshot_hook(...)` | `:541-554` | 在 `_plan_and_commit` **之前**调用 |
| 快照续跑 | `run_resumed(snapshot)` | `:573` | 从快照恢复并跑完 |
| 状态身份 | `snapshot_state_hash(snapshot)` | `recourse_snapshot.py:68-77` | 确定性 sha256；**跳过 `state_hash` 键本身**；键排序；对 route 顺序敏感 |
| 快照捕获 | `capture_recourse_snapshot(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask, all_customers=None)` | `:80-211` | 写 `state_hash`（`:210`）与 **`coldchain_contract_hash`**（`:168`） |
| 快照恢复 | `restore_recourse_snapshot(snapshot)` | `:214-225` | **严格**：缺 `state_hash` 拒绝；不一致 raise；另校验 `coldchain_contract_hash`（`:326`） |
| 合同身份 | `ColdChainContract.contract_hash` / `ColdChainContractV2.contract_hash` | `coldchain_contract.py:302` / `:658` | 64-hex |
| 目标身份 | `ObjectiveProfile.profile_hash` | `coldchain_contract.py:454` | 64-hex |
| 可信候选池 | `decision_pool_from_vehicles(vehicles, served_mask, visible_ids)` | `dynmaskco_cc_context.py:106-121` | 锁 committed_next + **非 replan 车** tail |
| 统一 mask 池 | `mask_candidate_pool(vehicles, served_mask, visible_ids, protected, plans)` | `:124-152` | decision_pool − protected 车 tail |
| mask 作用域断言 | `validate_mask_scope(vehicles, served_mask, visible_ids, plans, mask, mutable_ids, protected)` | `:155-177` | ⊆ decision_pool、非 committed_next、非 protected tail（**已不再断言 ⊆ mutable 车 plan suffix**） |
| 全分区校验 | `validate_full_partition(committed, plans, deferred, universe, served_mask=None, future_fn=None)` | `:180-230` | committed ⊎ suffix ⊎ deferred 互斥且覆盖；含 `served_planned` / `future_planned` |
| 共同 legal | `enumerate_legal_actions(env, inst_idx, plans, remaining, mutable_ids)` | `mpre_policy.py:18-26` | 需 **env + inst_idx**；逐个候选过 `cand.feasible` |
| 候选枚举 | `enumerate_actions_from_plans(env, inst_idx, plans, customer, allowed_vehicle_ids=None)` | `action_contract.py:365` | — |
| 单路线认证 | `certify_route(env, inst_idx, anchor_node, anchor_time, anchor_load, suffix)` | `action_contract.py:142` | — |
| 计划构造 | `build_vehicle_plans(env, inst_idx, vehicles)` | `action_contract.py:91` | vehicles → FleetPlan |
| 应用动作 | `apply_action(plans, action, src_vid=None, allowed_vehicle_ids=None)` | `action_contract.py:218` | **第 3 个位置参数是 `src_vid`**；新建对象 |
| 计划身份 | `plan_hash(plans)` | `action_contract.py:301` | (vid, anchor_node, suffix) 的 16-hex |
| 完整对账 | `validate_repair_plan(plans, base_plans, mask_set, protected_vehicle_ids)` | `mpre_policy.py:83-132` | 车辆集合/anchor、exact-once、非 mask 顺序、protected 完整计划；`:116 extra` 是 gate B 的成因 |
| 历史策略 | `stepwise_sample(score_fn, extract_fn, env, inst_idx, plans, mask_set, mutable_ids, …)` | `mpre_policy.py:34-70` | F 的原始路径，返回**部分计划**（不 fail-closed） |
| **真实 C0 能耗认证** | `certify_plan(env, inst_idx, clock, vehicles, served_mask, plan, contract, B)` | `run_exp_energy_c0.py:154-199` | 返回 **`(feasible, total_energy_kwh)`**；含 committed leg + WAIT 到 booking_end + 返仓。**纯函数，不能改状态** |
| TW/容量可行性 | `C0ReserveReplanner._route_feasible(self, env, inst_idx, cur, cur_time, load, route)` | `run_exp_energy_c0.py:245-258` | `certify_plan` **不查** TW/容量，这一条才查 |
| 计划级回滚包装 | `C0ReserveReplanner._try_insert_certified(...)` | `run_exp_energy_c0.py:296-307` | 已存在的**计划级**回滚：`:298` 存 `saved_plan`，`:304` 捕获 `ValueError`，`:307` 恢复 |
| C0 能耗积分 | `c0_completion_energy(...)` | `run_exp_energy_c0.py:112-151` | 经 `transition_segment` 返回 `new_state`，无副作用 |
| 终端全链校验 | `evaluate_trace_a1(traces, dataset_instance, contract, accepted_set, rejected_set, B, reject_penalty_by_class)` | `coldchain_evaluator_a1.py:36-165` | 返回 dict（`hard_feasible` `:147`）+ 14 类失败码（commitment/duplicate/capacity/TW/served_before_reveal/depot/fish_unsalable/budget…） |
| 硬门 | `hard_vector_from_outcome` / `hard_pass` / `service_ok` / `outcome_protocol_error` / `select_improving` | `hard_gate.py:104/127/132/20/154` | 12 个布尔键；`select_improving` 返回 `(cost, action_id)` 或 None |
| **D3 安装门（本批已有）** | `InstallLedger` / `run_repair_transaction` | `d3_geo_adapter.py:586` / `:708` | `finish(...)` 门：`REFUSAL_REASONS` `:122-124`、`INSTALL_REQUIRES_CERTIFICATION` `:119`、`self.install_fn(plan)` `:689`、`_copy_plan` `:157`、**STILL OPEN 声明** `:54-60` |
| 快照捕获钩子 | `PrePrepareContextCollector(deindex, save_snapshot=False)` + `.hook(...)` | `feature_local_replanner.py:69/79/85-121` | 只读；按 `(inst, event, clock)` 缓存；`get` 未命中即 raise。**可保存完整 snapshot** |
| 强制注入计划 | `env.force_suffix` → `v.mutable_suffix = list(force_suffix[key])` | `strict_online_env.py:639-643`（配 `:590`）+ `counterfactual_teacher.py:112-145` | 已有"绕过 replanner 直接注入计划"的机制；`_snapshot_with_force` 只写快照副本 |
| replanner 状态导出 | `JF1HRepairReplanner.export_state()` / `restore_state(state)` | `jf1h_repair.py:77-85` / `:87-113` | **只导出** `{version, baseline, slack_vehicles, deferred_customers, status_inst}`；不碰 env/vehicles |
| incumbent 计算 | `compute_incumbent_plans(env, snapshot, continuation)` | `dynmaskco_cc_context.py:86-97` | 需 `env.prepare_decision_point` + `continuation.plan` |
| mutable 车集合 | `mutable_scope(vehicles)` | `dynmaskco_cc_context.py:100-104` | idle/ready 且 needs_replan |
| 真实写回点 | `v.mutable_suffix = …` | `feature_local_replanner.py:413`、`dynmaskco_cc_replanner.py:212`、`cc_lns_replanner.py:278`、`jf1h_repair.py:331-333`、`event_plan_replanner.py:196`、`cc_swap.py:308` | 每个 replanner 各自写回 |

---

## 2. 真实调用链规格（8 段）

> 记号：`[现有]` 已实现；`[缺]` 装配层必须新增；`[风险]` 现实现有需要决策的语义问题。

**① 状态载体** —— `[现有]`
`vehicles: list[VehicleState]` + `traces: list[VehicleTrace]` + `served_mask: bool[N]` + `env: StrictOnlineEnv` + `inst_idx` + `clock` + `event_id`。
快照与恢复由此处定义，是唯一权威运行态。

**② 决策点准备** —— `[现有]` + `[风险 R1]`
`env.prepare_decision_point(clock, vehicles)`（`:608`）会**改写** `ready_time`。
而 `_maybe_snapshot`（`:541`）在 `_plan_and_commit`（`:545`）**之前**触发，也就是说：
**快照钩子看到的是 pre-prepare 状态，而计划是在 post-prepare 状态上算出来的。**
→ B2 必须声明"状态身份锚点"到底是哪一个；两种可选：
 (a) 在**认证时刻**（post-prepare）捕获身份 —— 与计划真正依据的状态一致，但快照钩子里的身份与之不同；
 (b) 把身份锚点声明为 pre-prepare，并在校验时**确定性地重放** `prepare_decision_point`。
**建议 (a)**，理由：身份必须覆盖计划真正依赖的状态；pre-prepare 身份会把"准备步骤"的差异误报为状态漂移。
（此问题与既有的 "online context 必须是 pre-prepare" 结论不冲突：那一条约束的是**特征**，这里约束的是**身份**。）

**③ 可信候选池** —— `[现有]` + `[风险 R2]`
两套"锁定"口径并存且**不等价**：
- `decision_pool_from_vehicles`（`:106`）：锁 committed_next + **非 replan 车**的 tail（replan 车 tail 可重分配）；
- `get_reserved_customers`（`:232`）：锁 committed_next + **所有车**的 tail。
`mask_candidate_pool`（`:124`）基于前者，再加 `protected` 车 tail。
→ 装配层必须**显式声明权威口径**，并在同一决策点把两者都记录（不等价性作为观测项，而不是让其中一套静默生效）。建议权威 = `mask_candidate_pool`（已有单测与 B 语义），`get_reserved_customers` 作为**交叉校验**：若 `pool_B ∩ get_reserved_customers() ≠ ∅` 则记录。

**④ B 语义池** —— `[现有，但无生产调用方]`
`d3_geo_adapter.build_pool_B(vehicles, served_mask, visible_ids, plans, mutable_ids, protected=frozenset(), pool_fn=None)`。
装配层负责注入**真实**的 `vehicles / served_mask / visible_ids / plans`，并记录 `diag`（含 `empty`，空池按原样记录、不丢状态）。

**⑤ 共同 legal** —— `[现有]` + `[缺]`
`mpre_policy.enumerate_legal_actions(env, inst_idx, plans, remaining, mutable_ids)`。
**所有臂必须传同一个 `env` / `inst_idx` / `plans` / `remaining` / `mutable_ids`** —— 这是"动作改变"可归因的前提。
装配层需新增：在每次 `legal` 生成后记录 `legal_order`（客户 id + 车辆 id），以便事后证明两臂的候选集一致。

**⑥ 评分** —— `[现有]`
`extract_fn(P, remaining, legal)` → state dict；`score_fn(state)` → `scores[len(legal)]`。
D-Geo（`geo_score`）与 F（`select_historical_argmax`）都在适配器内，且已由 S1/S2 守卫。

**⑦ 完整 C0 认证** —— `[现有零件]` + `[缺组合]` + `[风险 R3]` ← **本节最重要的修正**

**`certify_plan` 不是"完整认证"，它只查能耗，而且是一个纯函数。**

- `certify_plan(...)`（`run_exp_energy_c0.py:154-199`）只认证**能量预算** `total ≤ B + 1e-9`，返回 `(bool, float)`。它从真实 `v.coldchain_state` 接续（`:176-181` 把 `committed_next` 前置以避免双重计数），经 `c0_completion_energy`（`:112-151`）/ `transition_segment` 积分。**全程无赋值、无副作用**，因此它**不能**充当"安装"。
- 它**不查** TW 与容量 → 那是 `C0ReserveReplanner._route_feasible`（`:245-258`）；
  它**不查** 质量/可售/承诺/重复服务 → 那是 `evaluate_trace_a1`（`coldchain_evaluator_a1.py:36-165`）；
  它**不查** 所有权分区 → 那是 `validate_full_partition`（`dynmaskco_cc_context.py:180-230`）；
  它**不查** 计划对账（exact-once / 非 mask 顺序 / protected 不变）→ 那是 `validate_repair_plan`（`mpre_policy.py:83-132`）。
- 仓库自己也记录了这条限制：`01_取货线/文档/协议与审计/方向A全链路核查与正向增益改进清单_2026-09-28.md:128` ——「certify_plan 只查能耗」。

⇒ **装配层的"提交门"必须是一个组合**，而不是一次 `certify_plan` 调用。建议的组合与顺序（任一项不过即拒绝）：

| 序 | 检查 | 函数 | 覆盖 |
|---|---|---|---|
| 1 | 计划对账 | `validate_repair_plan(plans, base_plans, mask_set, protected_vehicle_ids)` | exact-once、非 mask 顺序、protected 完整计划、车辆集合/anchor |
| 2 | 所有权分区 | `validate_full_partition(committed, plans, deferred, universe, served_mask, future_fn)` | committed/已服务/未来客户不进计划；互斥且覆盖 |
| 3 | 路线可行性 | `C0ReserveReplanner._route_feasible(...)` | TW + 容量 |
| 4 | 能耗预算 | `certify_plan(...)` | `energy_kwh ≤ B` |
| 5 | （可选，终端） | `evaluate_trace_a1(...)` | 仅在整个 episode 结束时可用，非决策点可用 |

- `[缺包装]` D3 安装门期望 `certify_fn(P) -> (ok, detail)`，而 `certify_plan` 返回 `(bool, float)` ⇒ 装配层需要**薄包装**把组合结果与 `total_energy_kwh / B / 余量` 放进 `detail`。
- `[风险 R3]` 这些函数分散在**实验入口脚本**（`run_exp_energy_c0.py`）与库模块（`coldchain_evaluator_a1.py` 在 `evaluation/`）中，`simulation/` 层没有统一认证器。**建议原样 import、不搬移、不复制**（搬移/复制会改源码身份并制造第二个认证器），并把该耦合登记为已知风险；是否上收留待单独批准。

**已有可复用的"先校验后写入"先例**（装配层应当对齐它们，而不是另发明一套）：
- `feature_local_replanner.py:386-388` 先 `validate_full_partition` 通过，才在 `:407-414` 写 `v.mutable_suffix`（写入点 `:413`）；
- `cc_lns_replanner.py:264` 在写入前重算 `_partition_ok`，`:267 if ok:` 才执行写回循环 `:268-278`（写入点 `:278`），随后 `:279` 同步 deferred；
- `C0ReserveReplanner._try_insert_certified`（`run_exp_energy_c0.py:296-307`）是**计划级**的插入+回滚包装。

**已有"绕过 replanner 注入计划"的机制**（B1 可用于构造状态级反例，同时是一条风险）：`env.force_suffix` → `strict_online_env.py:639-643` 直接把 `v.mutable_suffix` 覆盖为注入计划（配 `:590` 的 `run_resumed`），`counterfactual_teacher._snapshot_with_force`（`counterfactual_teacher.py:112-145`）只在深拷贝副本上写。**风险**：`force_suffix` 绕过 WAIT 逻辑（`:639-646` 注释自陈），因此不能用它来断言"安装路径的合法性"。 

**⑧ 实际安装** —— `[现有]` + `[缺事务边界]` ← **本批的核心缺口**
安装 = ①replanner 写 `v.mutable_suffix`；②`_plan_and_commit` 的提交循环（`:650-656`）pop 第一个节点并 `_commit`。
`_commit`（`:260-299`）**不做任何校验**：它直接按计划头算到达/服务时间并写 `committed_next`。
→ 因此"合法性"必须在**pop 之前**校验，而 pop 由 env 在自己的循环里做，适配器看不见。
**装配层必须新增**：包住 `_plan_and_commit` 的一个校验点（见 §4），以及安装回执（§7）。

---

## 3. 候选池与承诺护栏的现状语义（B1 事实基础）

- `committed_next` 非空 ⇒ 该客户**不可再分配**（`decision_pool_from_vehicles:115-116`、`get_reserved_customers:241-242`）。
- `status == 'committed'` 的车正在执行一段 leg；`served_route` 记录**已服务**节点；`mutable_suffix` 是**未提交**计划。
- `served_mask` 为 `True` 的客户**不得进入计划**：`validate_full_partition` 的 `served_planned` 分类（`:213-214`）正是为此。
- `protected`（预留车）的 tail 由 `mask_candidate_pool(protected=…)` 排除；`validate_repair_plan` 的第 5 条（`mpre_policy.py:126-130`）校验 protected 车计划**完整不变**。
- 状态失效的**真实事件**：reveal（`:523-528`）清空所有非 closed 车的 `mutable_suffix` 并置 `needs_replan=True`。
  这意味着**上一个事件点认证过的计划在下一次 reveal 后必须失效**，而目前**没有任何东西记录或检查这件事**。

---

## 4. 安装点的精确定位：「回调被调用」≠「真实状态已提交」

**现状（代码事实）**
1. 适配器的 `InstallLedger.install_fn` 是**注入回调**。它被调用，只证明**回调执行了**。
2. 真实状态提交发生在 `_commit`，由 `_plan_and_commit` 在 **replanner 返回之后**驱动，**适配器无法观察**。
3. `_commit` **不校验** TW / 容量 / 能耗 / 客户是否仍可见 —— 它照单执行计划头。

**设计要求：装配层必须区分四态，并分别留证**

| 状态 | 含义 | 判据 |
|---|---|---|
| `callback_invoked` | 安装回调被调用 | `InstallLedger.n_installs == 1`，回执存在 |
| `suffix_written` | 计划已写入 live `VehicleState.mutable_suffix` | 逐车 `list(v.mutable_suffix) == 计划 suffix`（写回后、commit 前） |
| `head_committed` | env 已 pop 并 `_commit` 首个节点 | `v.committed_next == 计划头` 且 `v.status == 'committed'`（或 `0` ⇒ `returning/idle`） |
| `verified_consistent` | 提交后状态与计划一致且合法 | `served_route` / `trace.services` 反映该客户；`validate_full_partition` 通过 |

**可证伪的失败模式（本批要设计的反例之一）**
`callback_invoked = True` 但 `head_committed = False`：写入的 `mutable_suffix` 在下一次 reveal 被清空、或 `v.status` 已变成 `committed/closed` 使提交循环跳过该车 —— 安装回调"成功"，真实状态**没有任何改变**。这在当前代码里**不会被任何断言捕获**。

**全仓唯一一处"安装前后对比真实状态"的地方**（说明这一层确实缺失）：
`run_feature_local_writeback_check.py:143-151` 捕获了 live `v.mutable_suffix` 的前后值并断言相等 —— 但它①只覆盖 **reject 路径**，②是一个**测试脚本**，不是运行时闸门。
另外两处"看似"接近但严格说不够：
- `cc_lns_replanner.py:287-291` 记录 `writeback_changed` / `writeback_partition_ok`，但 `writeback_partition_ok` 是在**写入之前**基于候选算出来的（`:264`），不是写后复验；
- `feature_local_replanner.py:217-229` 记录 `baseline_before` / `baseline_after` / `learned_after` 三个**计划哈希**，是事后身份记录，没有回滚也没有复验。
**结论**：运行期"安装已提交且与认证计划一致"这一命题，目前**没有任何运行时证据**。这正是 B3 回执存在的理由。

---

## 5. B1 最小状态级反例设计（设计，不实现）

**共同前置**：本地用 `data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz`（或 dev_check）构造真实 `StrictOnlineEnv` + JF1-H-F `continuation`，经 `compute_incumbent_plans(env, snapshot, continuation)`（`dynmaskco_cc_context.py:86`）取得**真实** incumbent 与 live `vehicles`。
**本地可行性**：`data/m0_scale/*.npz` **已存在于本机**（train 1 667 788 B / cal 420 584 B / dev_check 420 657 B），因此 B1 **不需要服务器**。

| # | 反例 | 构造 | 断言的不变量 | 预期观测（修复前/未装配） |
|---|---|---|---|---|
| **C1** | 已承诺订单 | 取一个 `status='committed'` 且 `committed_next=c` 的决策点；把 `c` 放进 mask | `c ∉ pool_B` 且 mask 含 `c` 必须被拒绝 | `decision_pool_from_vehicles` 排除 `c`；`validate_mask_scope` 抛 `AssertionError`；**但** `d3_geo_adapter` 自身不检查 committed（计划级 API）⇒ 若装配层不调用作用域断言，`c` 可被重规划 |
| **C2** | 冻结已执行前缀 | 选 `served_route` 非空 / `served_mask` 有 True 的决策点；把已服务客户放进计划 | 已服务客户不得进入任何计划 | `validate_full_partition(served_mask=…)` 的 `served_planned` 非空 ⇒ `ok=False`；`_commit` **不会**阻止它 |
| **C3** | 保护车辆 | 设 `protected = {预留车 id}`（`env.get_reserved_customers` 语义 + `reserve` 配置），把该车 tail 客户放进 mask | `protected` 车 head/tail 不被改动 | `mask_candidate_pool(protected=…)` 排除；`validate_repair_plan` 第 5 条报 `protected_vehicle_changed` |
| **C4** | **状态版本变化** | 在事件 *k* 对计划 P 做认证（记录 `state_hash`），然后推进到事件 *k+1*（若为 reveal，`mutable_suffix` 被清空 `:528`），再尝试安装 P | 认证后状态变化 ⇒ 安装必须被拒绝 | **当前无任何检查**：`state_hash` 一致性与计划安装**没有绑定**；P 会被静默丢弃（tail 已清空）或写入错误状态 |
| **C5** | **认证后提交失败** | 认证通过后，构造"计划头不再合法"的情形（头客户已 served / 车辆已 `closed` / `ready_time` 被推进）再走提交循环 | 提交前必须复验；不得提交非法头 | `_commit`（`:260`）**零校验**：它照单提交。`_plan_and_commit` 的循环也只在 `status in ('idle','ready')` 时 pop ⇒ `closed` 车**静默跳过**，`callback_invoked=True` 而 `head_committed=False` |

**为什么不能用合成假件**：C1–C5 全部依赖 `committed_next` / `served_route` / `status` / `needs_replan` 与**真实 env 的提交循环**的交互。Red Phase 的 A-3b 尝试用一个伪造的 `VehiclePlan.committed_next` 属性，**没有触及真实机制**（已记为 `NOT_VERIFIABLE`）。B1 必须禁止同类做法：反例要跑真实 `StrictOnlineEnv`。

**每条反例的产物要求**：`state_hash` / `contract_hash` / `pool_B` / `legal_order` / 选择的动作 / 四态安装标志 —— 与 B2/B3 共用同一份记录结构。

---

## 6. B2 三重身份绑定设计（state, contract, plan）

**用户裁决**：只绑定 `plan_hash` 不足够 —— 认证时用到的**车辆状态**或 **C0 合同版本**可能已变化，而计划哈希不变。

**修正后的判断（两轮只读盘点 + 我逐条复核后）**：三个分量的"原料"分散存在，但**没有任何一个地方同时算出它们**，而且**实时决策路径上目前不存在状态身份**。具体：

| 分量 | 现状 | 证据 |
|---|---|---|
| **状态** | ⚠️ **实时路径上没有状态身份**。全仓唯一的状态哈希函数 `snapshot_state_hash(snapshot)`（`recourse_snapshot.py:68-77`）哈希的是**快照 dict**，不是 live `vehicles`；它唯一的产线调用者是 `capture_recourse_snapshot`（`:210`），而后者**只经 `env.snapshot_hook` 触发**，该钩子**默认为 `None`**（`strict_online_env.py:179`），仅 teacher export / oracle / 诊断驱动注册。`visible_state.py`、`coldchain_visible_features.py`、`dynmaskco_cc_context.py`、各 replanner **全都没有** `state_hash` 相关调用（`state_id` 全仓 0 命中）。全仓**唯一**"状态变化后重新哈希"的先例是 `counterfactual_teacher.py:144`（强制 suffix 后给**副本**重新打哈希）。 | 已复核 |
| **合同** | ✅ 身份很强（覆盖 `asdict` 全字段，含 `parameter_provenance` 的敏感区间），但**从 state 不可达**：`VisibleState` 无 contract 字段，`evaluate_visible_plan(vis, plan_suffixes, contract, objective)` 把 contract 作为**独立参数**；决策代码里 `env.coldchain_contract` **0 命中**。只有 replanner 手里有 `env`，因此**只有装配层能读到**。另：`objective_profile_hash` **全仓不存在**，只有手写 8 键的 `ObjectiveProfile.profile_hash`。 | `coldchain_contract.py:301-306` / `:453-465`；`visible_state.py:45-74` |
| **计划** | ⚠️ `plan_hash` 只哈希 `(dict key, anchor_node, repr(tuple(suffix)))`，**不含 `anchor_time` / `anchor_load`**，且**截断到 16 hex（64 bit）**。⇒ 同一辆车、同 `anchor_node`、同 suffix、但**锚点时刻不同**的两个计划会**哈希相同** —— 计划身份对"时钟/锚点漂移"是**盲的**。`event_plan_candidate.full_plan_hash`（`:32-35`，截断 24 hex）**包含** anchor_time/load，是更强的备选。 | `action_contract.py:301-309`；`event_plan_candidate.py:32-35` |

⇒ **B2 的第一件工作不是"绑定"，而是先把三个身份在安装边界上变成可计算的**：
1. 装配层必须**显式**在认证时刻捕获一次状态身份（`capture_recourse_snapshot` 需要 `env / inst_idx / clock / event_id / reveal_idx / vehicles / traces / served_mask`，其中 `traces` 与 `env.force_suffix` 都在 env 手里）——**不能依赖 `snapshot_hook` 默认存在**；
2. 装配层必须**显式**读 `env.coldchain_contract.contract_hash`（+ 若启用 profile，`apply_objective_profile(...).contract_hash` 与 `profile_hash`）；
3. 计划身份需要**加强**：要么改用/补一个覆盖 `anchor_time`/`anchor_load` 的更强 digest，要么在凭据里**显式**记录这两个字段。**注意**：直接替换现有安装门的 `plan_hash` 判定是一次**语义变更**，需要它自己的测试与契约更新（与 batch 1 的 3 条被取代断言同一类处理方式）。

**其它必须写进实现规格的事实**
- `_route_feasible` **同名两处**：`visible_state.py:163`（可见计划评估用）与 `run_exp_energy_c0.py:245`（`C0ReserveReplanner` 用）。装配层**必须显式指名**要哪一个，不能靠 import 巧合。
- `validate_mask_scope(...)` 的 `mutable_ids` 参数**只出现在签名行**（`dynmaskco_cc_context.py:155`），函数体**从未使用** —— 一个"看起来在约束、实际不生效"的陷阱，装配层不得依赖它来保证 mask ⊆ mutable 车计划。
- `ColdChainContractV2` **没有** `from_manifest` / `load_coldchain_contract_v2`（全仓缺失）⇒ 合同漂移反例只能在**进程内**构造 v2 合同（`tests/test_quality_v2.py` 就是这么做的），不能用 manifest 往返。
- 快照字段 `replanner_state` 来自 `JF1HRepairReplanner.export_state()`，只有 `{version, baseline, slack_vehicles, deferred_customers, status_inst}`；`FeatureLocalReplanner` **不 override**，所以它的 `decision_log` / `event_plans` / `_accepts_total` **不进快照**（见 §7）。
- `state_hash` 在 `restore_vehicles_from_json`（`dynmaskco_cc_context.py:34-74`）里被**刻意跳过**（JSON 往返不可复现）——这一点决定了下面那条关键约束。

**关键约束（必须写进实现规格）**：`state_hash` **不能从 JSON 往返后的快照重算**。
`recourse_snapshot.py` 的模块 docstring 与 `dynmaskco_cc_context.py:8-9, 35-38` 都明确记录：JSON 往返后 `state_hash` 不复现（numpy array → list 的序列化差异），因此 `restore_vehicles_from_json` **刻意不做** `state_hash` 校验。
⇒ **状态身份必须在同一进程、同一对象上捕获与复算**（用 `capture_recourse_snapshot` 的同一函数，输入同一批 live 对象）。把它从磁盘上的 JSON 重算会得到**不同的哈希**，从而产生假阳性拒绝。

**认证凭据（certification credential）设计**

```
Credential = {
  state_hash,                     # snapshot_state_hash(snapshot captured from LIVE state)
  state_hash_source,              # 'assembly_capture' | 'env_snapshot_hook'  (不得假设 hook 存在)
  state_capture_anchor,           # 'post_prepare' | 'pre_prepare'  (必须声明)
  contract_hash,                  # env.coldchain_contract.contract_hash
  effective_contract_hash,        # apply_objective_profile(contract, profile).contract_hash
  profile_hash,                   # ObjectiveProfile.profile_hash (若启用 profile; 不是 objective_profile_hash)
  inst_idx, clock, event_id,      # 决策点坐标
  plan_hash,                      # action_contract.plan_hash(P)  <- 16 hex, 不含 anchor_time/load
  plan_hash_strong,               # 覆盖 anchor_time/anchor_load 的 digest（见 §6 第 3 条）
  anchor_fields,                  # {vid: (anchor_node, anchor_time, anchor_load)} 显式留证
  mask_order, mutable_ids,        # 计划的来源范围
  certifier_identity,             # module.qualname（已在 S4 记录）
  c0_detail,                      # certify_plan 的 (feasible, total_energy_kwh) 与 B
  certification_components,       # §2⑦ 的组合认证逐项结果（repair_plan/full_partition/route_feasible/energy）
  deadline_ok,
}
```

**安装门要求（在现有 S3/S4 之上增加两项）**
1. `state_hash` 复算**相等**（同一进程、同一批 live 对象、同一 anchor）；
2. `contract_hash` / `effective_contract_hash` / `profile_hash` 全部相等；
3. 已有的：`plan_hash` 相等、certifier 身份非空、`deadline_ok is not False`、`failure_reason is None`。

**反例（B2 专属）**
- **D1 状态漂移**：认证后推进 `clock`（或触发 reveal 清空 tail）而不改计划 ⇒ 必须因 `state_hash` 不等而拒绝。
  ⚠️ **现状强化**：因为**实时路径上没有状态身份**（§6 表），D1 在**装配完成前根本无法表达** —— 这正是"先做状态身份可计算性"的理由。
- **D2 合同漂移**：认证后换成 `ColdChainContractV2`（或不同 `ObjectiveProfile`）而计划不变 ⇒ 必须因 `contract_hash` / `profile_hash` 不等而拒绝。
  构造可行性：`ColdChainContractV2` 与 v1 的 `contract_hash` 不同（`coldchain_contract.py:657-662`）；**但 v2 没有 manifest 往返路径**（`from_manifest` / `load_coldchain_contract_v2` 全仓缺失）⇒ 反例必须**进程内**构造。
- **D3 锚点歧义**：同一决策点分别以 pre-prepare 与 post-prepare 捕获身份 ⇒ 两者**必然不等**；用来证明"锚点必须声明"，而不是当成缺陷。
- **D4 计划身份的盲区（新增，来自只读盘点）**：构造两个计划，`anchor_node` 与 `suffix` 完全相同但 `anchor_time` 不同（等价地：状态漂移改了 `ready_time`/`committed_finish` 却没改 suffix）⇒ 当前 `plan_hash` **相等**，现有安装门**会放行**。用来证明"只绑 `plan_hash` 不足够"是**可复现的**，而不是理论担忧。

---

## 7. B3 持久化安装回执设计

**要求**：记录 `P_before` / `P_proposed` / `P_certified` / `P_after_install`，记录**真实执行结果**与身份校验；必须特别区分"安装函数被调用"与"真实状态已成功提交"。

**为何需要"真实执行结果"**：见 §4 —— `install_fn` 被调用与 `_commit` 生效之间隔着一整个 env 提交循环。

**回执结构（建议）**

```
InstallReceipt = {
  receipt_schema_version,
  # ── 四个计划 ──────────────────────────────────────────────
  P_before:        {plan_hash, plans},            # pop 之前 env 里的计划（FleetPlan）
  P_proposed:      {plan_hash, plans, source},    # 修复器产出（未认证）
  P_certified:     {plan_hash, credential_hash},  # 通过 C0 的那个计划 + 凭据摘要
  P_after_install: {plan_hash, plans},            # 提交后从 live vehicles 重建的计划
  # ── 真实执行结果（四态，§4）──────────────────────────────
  callback_invoked: bool,
  suffix_written:   {vid: [nodes]} | None,
  head_committed:   {vid: {'node': n, 'status': ..., 'committed_next': ...}},
  verified_consistent: bool,
  # ── 身份 ────────────────────────────────────────────────
  credential: Credential,                          # §6
  identity_recheck: {state_hash_ok, contract_hash_ok, plan_hash_ok},
  # ── 判定 ────────────────────────────────────────────────
  outcome: 'installed_and_committed' | 'callback_invoked_but_not_committed' | 'refused' | 'rolled_back',
  refusal_or_failure_reason,
}
```

**"可独立重新加载、重新计算身份，与实际安装状态一致"的可检验定义**
1. 从回执里的 `P_before` + `credential` 能**重算** `state_hash` 与 `plan_hash`，且与回执记录相等（进程内复算；跨进程需同时保存 `capture_recourse_snapshot` 所需的 live 对象序列 —— 见下条）；
2. `P_after_install` 能从 `P_certified` + 提交循环**确定性复现**（同一 `clock`、同一 commit 循环规则）；
3. `head_committed` 里的 `committed_next` 与 **live 状态**一致；
4. 独立复核程序能仅凭回执 + 该决策点的状态捕获，判断"是否真的提交了"，**不需要**相信任何布尔标志。
**未决**：跨进程复算 `state_hash` 需要把 `capture_recourse_snapshot` 的输入（vehicles/traces/served_mask）按**同一序列化规则**持久化 —— 而 JSON 往返已知会破坏哈希（§6）。**这一条是本设计最大的未决项**，需要单独决策：要么定义一套"可复现的落盘序列化"（并在其上重新固定 `state_hash` 公式与 schema 版本），要么把回执的"独立重算"限定为**同进程/同运行内**并把跨进程复算标记为未实现。
**不得**用一个"看起来很确定"的哈希掩盖这一点。

**B3 现状缺口（实测，必须一并解决才算"完整回执"）**
1. **replanner 内态没有随快照走**：`JF1HRepairReplanner.export_state()`（`jf1h_repair.py:77-85`）只导出 `{version, baseline, slack_vehicles, deferred_customers, status_inst}`；而 `FeatureLocalReplanner` **不 override** `export_state`/`restore_state` ⇒ 它的 `_accepts_total` / `decision_log` / `event_plans` / `_decision_seq` **不进快照**。这意味着"从回执独立重载"连**决策级**状态都还原不了。
2. **`P_before`/`P_after` 只在 accept 路径记录**：`feature_local_replanner.py:419-422` 才写 `P_before`/`P_after`；reject 路径（`:371-373`、`:398-401`）不记。而 B3 要求四个计划**在所有路径**都有记录（含拒绝与失败），否则"为什么没提交"无法复核。
3. **计划级回滚 ≠ 状态级回滚**：`InstallLedger.installed_plans` / `installs` 是**计划的**值拷贝（`_copy_plan`，`d3_geo_adapter.py:157`、存储 `:691-696`），**不是** vehicle/env 状态。全仓**没有**任何 replanner 保存"修复前 + 修复后的完整状态对"（唯一的全状态快照 `capture_recourse_snapshot` 是在 replanner 之前由 `snapshot_hook` 采的，`strict_online_env.py:549-554`，并非按每次修复配对）。
   `env.run_resumed(snapshot)`（`:573-606`）原则上能重建完整 live 状态，但**没有**任何 replanner 去采一对前后快照并比较 —— 因此**状态级回滚目前不可实现**，只能做到"计划级回滚 + 状态身份拒绝"。

---

## 8. 风险与未决问题（需你裁决）

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| **R1** | 身份锚点 pre-prepare vs post-prepare（§2②） | 假阳性拒绝 / 身份覆盖不到计划真正依赖的状态 | 选 post-prepare，并在凭据中声明 `state_capture_anchor` |
| **R2** | `decision_pool_from_vehicles` 与 `get_reserved_customers` 口径不等价（§2③） | 候选池边界不唯一 ⇒ 归因可疑 | 显式声明权威 = `mask_candidate_pool`，另一套作交叉校验并记录差集 |
| **R3** | 真实 C0 认证器位于实验入口脚本 `run_exp_energy_c0.py`（§2⑦） | 装配耦合实验入口；上收会改源码身份 | 先原样 import；上收需单独批准 |
| **R4** | `state_hash` 跨 JSON 往返不可复现（§6/§7） | 回执的"独立重载"能力受限 | 明确限定为同进程复算，或另立"可复现落盘序列化"作为独立议题 |
| **R5** | `_commit` 零校验（§4） | 非法计划头会被真实提交 | 装配层在 pop 之前做一次完整复验（`validate_full_partition` + `validate_repair_plan` + C0 复算） |
| **R6** | 32 状态资产仍只有索引（任务 B 结论） | 恢复需要新执行 | 本设计**不绕过**该资格要求；B1 使用**本地 data/*.npz 新构造**的决策点，不声称恢复历史 32 状态 |
| **R7** | **实时路径没有状态身份**，`env.snapshot_hook` 默认 `None`（`strict_online_env.py:179`） | "状态漂移"当前不可观测、不可拒绝 | 装配层**自己**在认证时刻调 `capture_recourse_snapshot`，并把 `state_hash_source` 记入凭据；不得假设 hook 已注册 |
| **R8** | `plan_hash` **不含** `anchor_time`/`anchor_load` 且截断 64 bit；`full_plan_hash` 含但截断 96 bit | 只绑 `plan_hash` 时"计划相同、锚点已变"会被放行 | 凭据同时记录 `plan_hash_strong` + `anchor_fields`；替换安装门判定属**语义变更**，需自己的契约测试 |
| **R9** | `ColdChainContractV2` 无 manifest 往返；`objective_profile_hash` 不存在（只有手写 8 键的 `profile_hash`） | 合同身份只能进程内比对；profile 字段增加时 `profile_hash` 可能漏字段 | 反例进程内构造；在凭据里记录 `profile_hash` 的**键集**，以便日后发现漂移 |
| **R10** | 状态级回滚目前**不可实现**（无 replanner 保存前后状态对；只有计划级回滚 + 状态身份拒绝） | "回滚到已承诺状态"只能靠**拒绝**而非**恢复** | 回执的 `outcome` 只用 `refused` / `rolled_back(plan_level)` 两种措辞；**不得**声称恢复过 committed 状态 |
| **R11** | `validate_mask_scope` 的 `mutable_ids` 形参**从未被使用**；`_route_feasible` 同名两处（`visible_state.py:163` / `run_exp_energy_c0.py:245`） | 误以为 mask ⊆ mutable 车计划已被断言；误用另一个 `_route_feasible` | 装配层显式加"mask ⊆ mutable 车计划 suffix"的**自有**断言；组合认证里**指名** `C0ReserveReplanner._route_feasible` |

---

## 9. 本文件明确**不做**的事

- 不实现任何装配代码、不写状态级测试、不加回执落盘 —— 等你**单独批准实现**。
- 不改 `d3_geo_adapter.py`（`51896c3` 之后未再改动，sha256 仍为 `099c3517…`）。
- 不启动 F−D / E2 / 32 状态重采集 / G1 / G2 / 最终测试种子。
- 不修改任何冻结树、历史裁决或历史测试文件。
- 不把 32 状态索引当作可恢复状态使用。

---

## 9b. 只读盘点与交叉核对（本设计的取证方式）

本设计由**两个独立的只读盘点**加上我对每条载荷锚点的**亲自复读**共同支撑；两份盘点在所有重叠事实上一致，且都未创建/修改任何文件。

**盘点 1（状态 / 合同身份）**：`coldchain_contract.py`、`visible_state.py`、`coldchain_visible_features.py`、`recourse_snapshot.py`、`dynmaskco_cc_context.py`、`action_contract.py` + 全树标识符搜索（`state_hash` / `state_id` / `state_version` / `snapshot_hash` / `identity`）。

**盘点 2（认证与提交面）**：`rolling_optimizer.py`、`coldchain_evaluator_a1.py`、`hard_gate.py`、`jf1h_repair.py`、`feature_local_replanner.py`、`cc_lns_replanner.py`。

**盘点带来的三处关键修正**（我原先的设计初稿偏乐观）：
1. **"完整 C0 认证"不是 `certify_plan`** —— 它只查能耗且是纯函数；完整门是四项组合（§2⑦）。
2. **实时路径上没有状态身份** —— `snapshot_state_hash` 哈希的是快照 dict，而 `env.snapshot_hook` 默认 `None`（`strict_online_env.py:179`）；`state_id` 全仓 0 命中，各 replanner/`visible_state.py`/`coldchain_visible_features.py` 中 `state_hash` 0 命中。
3. **`plan_hash` 对锚点漂移是盲的** —— 它只哈希 `(dict key, anchor_node, repr(tuple(suffix)))` 且截断到 16 hex；新增 D4 反例把"只绑 `plan_hash` 不足够"从理论担忧变成**可复现**。

**我亲自复核过、并因此写入本文的锚点**：`strict_online_env.py:74-92/95-109/232-258/260-299/523-528/538-554/573/608-656/648-656`；`recourse_snapshot.py:68-77/80-211/214-225/168/210/326`；`dynmaskco_cc_context.py:34-74/86-97/100-104/106-121/124-152/155-177/180-230`；`mpre_policy.py:18-26/34-70/83-132`；`action_contract.py:66-76/91/142/218/288-298/301-309/355-374`；`d3_geo_adapter.py:54-60/119/122-124/157/560/586/628-705/689/708`；`run_exp_energy_c0.py:112-151/154-199/245-258/296-307/382-393`；`coldchain_evaluator_a1.py:20-33/36-165`；`hard_gate.py:20/104/127/132/149/154`；`jf1h_repair.py:25-31/77-85/87-113/205/207/331-333`；`feature_local_replanner.py:69/79/85-121/204-229/269-291/308-423/407-414`；`cc_lns_replanner.py:109-196/231-295/264-291`；`visible_state.py:32-42/45-74/163-186/219`；`coldchain_visible_features.py:249-276/282-295`；`counterfactual_teacher.py:112-145`；`event_plan_candidate.py:32-35`。
**未能核实或不存在的**（盘点明确报告 + 我确认）：`urgent-defer` 命名的函数不存在；`objective_profile_hash` 不存在（只有 `profile_hash`）；`ColdChainContractV2.from_manifest` / v2 loader 不存在；`_snapshot_with_force` 不在 `dynmaskco_cc_context.py`（在 `counterfactual_teacher.py`）；任何携带 vehicle/env 状态的 `export_state`/`restore_state` 不存在；任何安装边界上的强制 C0 闸门（A-v1 层）不存在；持久化 `P_after_install` 不存在。

---

## 10. 建议的最小实现顺序（待批准）

1. **装配骨架（只读驱动）**：`strict_online_env` 之上加一个 `snapshot_hook` 驱动的装配入口，串起 ①→⑦，**先只记录不干预**（不安装），产出与真实 baseline 逐位一致的对照（no-op 等价性）。
2. **B1 状态级反例**：实现 §5 的 C1–C5，每条都必须**在真实 env 上**跑，且修复前应给出可观测的违规/未知。
3. **B2 凭据 + 安装门**：实现 §6 的 `Credential` 与三重身份校验，配 D1/D2/D3 反例。
4. **B3 回执**：实现 §7，并对 R4 做出明确取舍（限定同进程，或另立落盘序列化议题）。
5. **独立验收**：新契约测试对**旧三条被取代断言**给出明确替代命题（已在 Green Phase 建立 P/G 对照模式，复用该模式），并重跑全部仍适用的工程回归。
