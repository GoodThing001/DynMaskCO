# B2 静态核查（实现前，只读）

**日期**：2026-10-14 · **基准 commit**：`2176af6` · **性质**：只读静态/经验核查，**不实现**。
**用户要求的顺序**：先静态核查 → 再定 D1–D8 → 最后实现组合认证器。本文件是第一步。
**核查对象**：`run_exp_energy_c0.py`（`certify_plan` / `c0_completion_energy`）、`action_contract.certify_route` /
`build_vehicle_plans`、`coldchain_contract.apply_objective_profile`、以及 import 副作用。
**方法**：直接读源码 + 在子进程里做经验测量（不是从文档推断）。

---

## 1. 风险①：`run_exp_energy_c0` 的 import 顶层副作用 —— 已实测

| 检查 | 结果 |
|---|---|
| import 成功且耗时 | ✅ 0.151 s |
| 是否执行实验 | ✅ **否**：`main(argv=None)` 在 `:582`，`if __name__ == "__main__":` 在 `:715` |
| 是否写文件 | ✅ **否**：`scripts/`(486 文件) / `results/`(18775) / `data/`(123) 的文件数与最新 mtime **前后完全相同**（`changed: false`） |
| 全局副作用 | ⚠️ **有**：import 会**向 `sys.path` 插入 3 条**——`scripts/`、`scripts/simulation/`、`scripts/coldchain/`（`:27/:30/:33`） |
| 传递依赖 | ⚠️ 会传递导入另一个实验入口 `run_exp_reserve`（`from run_exp_reserve import REV, PVAL, K_EFF, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT, ...`） |

**结论**：**可以 import 复用其认证函数**（无数据/实验副作用），但：
1. 它有**顺序相关**的 `sys.path` 副作用 ⇒ 装配层**必须自己显式建立路径**，不得依赖"import 顺便把路径配好了"（B1 的测试就是这么做的）；
2. 它耦合到 `run_exp_reserve`（另一个实验入口），因此**不得**把"import 成功"当作"该模块可长期作为库使用"的证明——这一点登记为已知耦合（R3，B1 设计已记录）。

---

## 2. 风险②：合同与有效 profile 是否真正一致 —— 已实测：**它们不同，必须分别登记**

`env.coldchain_contract` 拿到的是**已应用目标配置后的有效合同**（gate 的 `_make_env` 先 `apply_objective_profile(...)` 再构造 env），因此**原始合同从 env 不可达**。

实测（本地 `results/o0cc/scale_v2/objective_profile.json`）：

| 项 | 值 |
|---|---|
| `raw contract_hash`（`default_pilot_contract()`） | `32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0` |
| `effective contract_hash`（`apply_objective_profile(raw, prof)`） | `edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833` |
| 两者是否相同 | **不同** |
| `objective` 是否改变 | **是** |
| `quality` 是否改变 | 否（本 profile 只改 objective） |
| `profile_hash` | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |

**落地要求**：`CertificationCredentialV1` 必须**同时**登记 `contract_hash`（原始，**由调用方注入**）与
`effective_contract_hash`（从 env 读）。若调用方未提供原始合同，必须记录
`raw_contract_reachable: false`，**不得**把有效合同冒充原始合同，也不得留空默认通过。

---

## 3. 风险③：路线能耗是否包含已承诺路段与返仓 —— 已读源码确认

`certify_plan(env, inst_idx, clock, vehicles, served_mask, plan, contract, B)`（`run_exp_energy_c0.py:154-199`）：

- **已承诺路段**：`:176-181` 对 `status=='committed'` 车把 `v.committed_next` **前置一次**
  （`route = [v.committed_next] + [o for o in plan[vid] if not served_mask[o] and o != v.committed_next]`），
  即**恰好一次**，不重复计入。
- **WAIT 到预约截止**：`c0_completion_energy:139-144`，服务完 route 后若早于 `wait_until`（`env.booking_horizon`）就制冷等待。
- **返仓段**：`c0_completion_energy:145-150`，`return_to_depot=True` 计入最后一段。
- **已 closed 车**：`:170-172` 直接累加 `cc.cumulative_energy_kwh` 后 `continue`。
- **未 dispatched 车**：`:196-198` 用 `dispatch_vehicle(create_vehicle_state(contract))` 从当前 clock 起算。
- **判据**：`:199` `return total <= B + 1e-9, total`。

**与 `build_vehicle_plans` 的一致性（防重复计数）**：`action_contract.py:91-108` 对 `committed` 车设
`anchor_node = committed_next`、`anchor_load = current_load + demand[committed_next]`，而
`suffix = mutable_suffix` 去掉 0 ⇒ **suffix 不含已承诺路段**。所以把 `suffix` 当作 `certify_plan` 的
`route` 传入是**正确**的（承诺段由 `certify_plan` 自己前置一次）。
另外 `build_vehicle_plans` `:95-96` **丢弃 closed 车**，与 `certify_plan` 的 closed 分支**互补而不冲突**。

**结论**：就"已承诺 + WAIT + 返仓"而言，能耗口径是完整的，**未发现重复计入**。
但下面第 4 条发现两个**静默**行为，必须由组合认证器补上显式拒绝。

---

## 4. 风险④：重复计入 / 使用未来信息 —— 发现 **1 处未来信息读取 + 2 处静默**

### F1（未来信息，必须补偿）`c0_completion_energy:127`
```python
depart = max(t, float(rev[o]))        # rev = env.reveal_time
```
它**读取 `reveal_time`**（未来信息字段）作为发车时刻的下界。
对**已揭示**订单该下界是惰性的（`rev[o] <= clock`）；但它**不会拒绝未揭示订单**——只是把发车推后。
⇒ 若计划里混入未揭示订单，认证会**静默**按未来揭示时刻排程，而不是报错。
**补偿要求（B2-2）**：组合认证器必须**自己**断言"计划中的每个订单在 `clock` 时刻已揭示"
（`env.reveal_time[inst_idx, o] <= clock + 1e-6`），否则拒绝。这正是用户警告的那类污染。

### F2（静默丢弃）`certify_plan:173 / :178-180`
`route = [o for o in plan[vid] if not served_mask[o]]` —— **已服务订单被静默剔除**而不是拒绝。
⇒ 组合认证器必须用 `validate_full_partition(served_mask=...)` 显式拒绝 `served_planned`。

### F3（静默跳过）`certify_plan` 的 closed 分支
`cc.closed` 车直接累加能耗、不检查其计划。`build_vehicle_plans` 也不给它计划条目 ⇒ 二者互补，
但**如果调用方自己构造 `plan` 并给 closed 车塞了 route，该 route 会被完全忽略而不报错**。
⇒ 组合认证器必须断言 `plan` 的车辆集合与 `build_vehicle_plans` 的集合一致（或至少 ⊆ 非 closed 车）。

---

## 5. 质量 / 鱼可售性：决策时**没有**可执行的在线检查 ⇒ 必须标记能力缺口

**实测证据**：

| 检查点 | 位置 | 覆盖 |
|---|---|---|
| 温区支持 | `action_contract.certify_route:178-183` | `temp_class not in supported_classes` ⇒ `feasible=False`, `coldchain_reject_reason='unsupported_temperature_class'` |
| TW | `:184-186` | `arrive > tw_end` ⇒ `'tw'` |
| 容量 | `:188-193` | `load > capacity` ⇒ `'capacity'` |
| 返仓 | `:198-202` | `return_slack < 0` ⇒ `'return'` |
| **能耗投影** | `:171` | `'projected_energy_kwh': None` |
| **质量损失投影** | `:172` | `'projected_quality_loss': None` |
| **热裕度投影** | `:173` | `'projected_thermal_margin': None` |
| 鱼可售（零阶感官 ≥5） | `coldchain_evaluator_a1.evaluate_trace_a1:115-117` | **终端**（需要完整 trace），决策时不可用 |

⇒ **决策时刻唯一可执行的冷链约束是"温区是否支持"**（静态订单属性）；
**品质衰减与鱼可售性在决策点没有可执行检查**。

**按用户裁决**：B2-2 必须把 `quality_salability` 记为 **能力缺口 / 阻断**，
**不得**因为"四个函数都调用了"就默认合格。凭据中该分量取值为
`CAPABILITY_GAP_BLOCKING`（三态之一），而非 `True`/`False`。

---

## 6. 由此确定的 B2 设计约束（供第 2、3 步）

1. **凭据**同时登记 raw / effective 合同、profile、`plan_hash_legacy` **与** `plan_digest_anchor_v1`、
   `anchor_fields`、决策坐标、`source_identity`、以及**认证结果**（含能力缺口三态）。
2. **组合认证** = 四项检查 **+ 三项补偿**（F1 已揭示性、F2 served、F3 计划车辆集合）
   **+ 质量/可售能力缺口标记**。任一项不过 ⇒ 拒绝；能力缺口 ⇒ **阻断**（不是通过）。
3. **安装前复验**：`H(S)/H(C)/H(P)` 三者全部相等才放行，且**三重身份相等不能替代**超时检查、
   完整认证、实际安装结果检查。
4. **`state_identity_information_scope`**：必须显式声明状态身份只依赖**已揭示**信息与真实可提交状态，
   并用测试证明**没有引入未来信息**（见第 3 步的 D-scope 检查）。
5. 不改 `_commit` / `_plan_and_commit`；不启用真实写回；不声称状态级 rollback；不声称跨进程可复核。

---

## 7. 本文件没有做的事

未实现任何 B2 代码；未修改任何生产文件；未启动 E2 / F−D / 32 状态重采集 / G1 / G2 / 最终测试。
B1 与 Green Phase 的历史测试、记录与冻结源码均未改动。
