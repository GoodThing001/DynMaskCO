# C2 预声明 · 勘误 01：消费对象绑定、数据依赖完整性、strict-online 来源（2026-10-17，追加式）

> **性质：勘误（amendment），追加式。本文件先于任何新增诊断运行而登记。**
> 不改写 C2 七要素预声明、不改写接口审计 7/7 记录、不改任何生产 helper。
> **本轮只批准 C2 只读资格审查**：不做 C2 实现、不做 G6 持久化、不做 C3、不接线生产。
> 基准 HEAD：`fe18e070cb57b53ab2b942450b3613b45a02cf5a`。

---

## 1. 最重要的缺口：四方摘要相同 **不** 蕴含四方认证了同一个状态

### 1.1 反例（无需哈希碰撞、无需篡改 digest）

设冻结状态 `S0`，四方各自计算 `identity(S0)` ⇒ 四份摘要一致，原判据 **C1 通过**。但：

```
M1-a  实际执行时读取活动环境 env1        （≠ S0）
C0    实际执行时读取另一个状态 env2      （≠ S0）
M1-b  使用 S0 的快照与 projection
安装方 使用 S0
```

此时 `I_A(S0) = I_B(S0) = I_C0(S0) = I_install(S0)` 成立，
但 `S_A^consumed ≠ S_C0^consumed` **完全可以同时成立**。

⇒ **C1 只证明"四份身份计算对相同输入一致且敏感"，不证明"四个认证器实际读取的就是 S0"。**

### 1.2 新增资格要求 **Q-C2-1：`CONSUMED_STATE_BOUND_TO_FROZEN_IDENTITY`**

必须证明下列四条**同时**成立，缺一即 `CONSUMED_STATE_UNBOUND`：

| # | 要求 | 机械形式（下一阶段实现） |
|---|---|---|
| a | 各认证器实际读取的数据来自**已绑定的冻结状态** | 认证器只接受冻结体或冻结体的**授权投影**，不接受原始 `env`/调用方 mapping |
| b | 或使用**可独立验证、来源于该状态**的授权投影 | 投影携带"由哪个冻结身份派生"的绑定，且投影可被独立复核 |
| c | 认证前后的状态身份**没有未经记录的变化** | 认证前后各复算一次冻结体身份；不一致 ⇒ 拒绝（沿用 G-3 L-2 的做法） |
| d | 各层声明的**读取范围**能支持其认证结论 | 声明"我读了 X 才能说 Y"；若 Y 依赖 X′ ∉ X ⇒ 拒绝 |

**失败条件**：四方摘要全部相同，但任一认证器实际消费的状态与冻结体不同，
而 C2 仍判为"共同身份成立" ⇒ **C2 核心主张尚未成立**。

**这是本勘误的核心**：C2 若不补这一条，就会重演 G-3 的问题——
**身份证明是对的，但真正使用的对象不是被证明的对象**。

---

## 2. 接口审计的可推断范围必须限定（不撤销测量）

`test_q3_c2_interface_audit.py` 的方法是：**在事先选定的字段名集合内**，对目标函数体做
`ast.walk` 的**直接属性读取统计**，并按 `RECOMPUTE_CALLS` 名称表判断"是否复算身份"。
它**不是**完整数据流/调用图分析。因此 7/7 的三条结论必须按下列方式收窄：

| 既有表述 | 允许的解释（收窄后） |
|---|---|
| 三方状态字段交集为 0 | **在预先选定的 `STATE_FIELDS` 集合内**，三个目标函数的**直接**属性读取交集为 0 |
| 只有 M1-b 复算身份 | **在脚本列出的 `RECOMPUTE_CALLS` 名称中**，只有 M1-b 命中 |
| 生产入口不存在三方联合消费 | 静态检索**未发现既定的调用形态**；**不等于**证明所有动态/间接路径都不存在 |

**具体证据**：`action_contract.py:79-84` 的 `_travel()` 内部访问 `env.dist_mat` 与 `env.tw_speed`，
而 `certify_route()` 的直接属性计数里**没有**这两个字段（它通过 `_travel` 传递读取）。
⇒ 直接统计**系统性低估**了真实读取集合。

**登记**：`C2_INTERFACE_AUDIT_OBSERVED`（7/7）**保留**，但**不得**升级为
"完整数据依赖已证明"。另：审计的 **A-5** 用事先写好的 `role` 字符串描述各层职责，
**不是**机械验证的语义结论。

### 2.1 新增资格要求 **Q-C2-2：`DATA_DEPENDENCY_COMPLETENESS`**

必须区分并分别标注三类读取：

1. **直接读取**（函数体内的属性/下标访问）；
2. **传递读取**（经由被调 helper，如 `_travel` → `env.dist_mat`）；
3. **未覆盖路径**（动态分派、间接调用、回调）。

并以第 2 类为**必要条件**：若某认证结论依赖传递读取得到的字段，而该字段未列入声明，
则该层的"读取范围"声明**不成立**。

---

## 3. C0 的 strict-online 数据访问必须单独审查

`run_exp_energy_c0.py:154-201` 的 `certify_plan()` 直接取得**完整实例数组**：

```python
dist = env.dist_mat[inst_idx];  tws = env.tw_start[inst_idx]
st_  = env.service_time[inst_idx];  rev = env.reveal_time[inst_idx]
dem  = env.demands[inst_idx];  tc = env.dataset['temp_class'][inst_idx]
```

这不是明确的"已揭示订单投影"。虽然下游主要按计划内订单索引计算，
**不能**据此断言已发生未来信息影响；但同样**不能**把 C0 的原始 `env` 访问路径
认定为 strict-online 合格来源。

### 3.1 新增资格要求 **Q-C2-3：`C0_VISIBLE_PROJECTION_PROVENANCE`**

必须核实三件事：

| # | 核实项 | 失败条件 |
|---|---|---|
| a | 进入身份与能耗认证的**数据来源** | 来源是完整实例数组而非可见投影，且未声明 |
| b | 被**实际索引**的订单集合 | 索引集合不受 `reveal_time <= clock` 约束 |
| c | **冻结时刻的可见性边界** | 认证所需字段无法由合法可见信息提供 |

三者任一不成立 ⇒ 沿预声明 **`REFUSED: HIDDEN_FUTURE_FIELD_REQUIRED`**，
**不得**为了四元组完整性去读取未揭示订单。

---

## 4. G-2 未合格的阻断范围（资格区分，必须保留）

| 命题 | 是否成立 |
|---|---|
| G-2 未合格 ⇒ **不得签发可安装的联合认证** | **成立**（判据 C10 / `JOINT_NOT_CERTIFIABLE`） |
| G-2 未合格 ⇒ **逻辑上禁止一切不产生资格的隔离身份诊断** | **不成立** |

即：隔离身份诊断（不产出可安装结论）**可以**在 G-2 未合格时进行；
是否实施，由本轮只读审查与后续单独授权决定。

---

## 5. 新增负控（**先登记，后运行**）

| 编号 | 注入 | 预期 |
|---|---|---|
| **Q1-N1** | 四方报告摘要全相同，但其中一方实际消费的状态已被改变 | 现有 C1 **通过**（反例成立）；**Q-C2-1** 必须判 `CONSUMED_STATE_UNBOUND` |
| **Q1-N2** | 同一反例，但该方消费的是**授权投影**（携带派生绑定） | 允许通过（须能独立复核投影来源） |
| **Q1-N3** | 认证后冻结体被改，且变化未被记录 | 拒绝（前后复算不一致） |
| **Q2-N1** | 认证结论依赖 `_travel` 传递读取的字段，但该字段未列入声明 | 该层"读取范围"声明不成立 |
| **Q2-N2** | 把 AST 直接属性统计当作完整调用链证明 | `INFERENCE_SCOPE_OVERSTATED` |
| **Q3-N1** | C0 认证所需字段只能从未揭示订单取得 | `REFUSED: HIDDEN_FUTURE_FIELD_REQUIRED` |
| **Q3-N2** | C0 的索引集合超出 `reveal_time <= clock` 的可见集合 | `C0_VISIBLE_PROJECTION_PROVENANCE` 不成立 |

---

## 6. 阶段裁决登记

| 项 | 状态 |
|---|---|
| C2 七要素预声明 | **已交付** |
| C2 接口静态审计 | **已完成 7/7 诊断，保留限定范围**（不得升级为完整依赖证明） |
| C2 四方共同身份 | **尚未建立**（Q-C2-1 未证明） |
| C2 隔离实现 | **暂缓授权**：先审查消费对象绑定 |
| C3 联合认证 / 生产接线 | **继续禁止** |
| G6 / 生产安装事务 | **不建议现在扩展** |

---

## 7. 状态（不变）

`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`（永久，不追认）、
**V-10 干净 checkout FAIL**、`THERMAL_CONTINUOUS_NOT_ESTABLISHED`、
`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、L1 `BLOCKED`、`L2/A3_NOT_ESTABLISHED`、
`stable_online_gain=False`、`HISTORICAL_BACKBONE_UNVERIFIED`、C0 `AWAITING_USER_BUDGET_RULING`、
`t_min` 恒 0、最终测试种子 `SEALED`、`PRODUCTION_NOT_QUALIFIED`、
B-1…B-9 全开（B-9a 有条件闭合、B-9b 划出 G-3）。

---

## 8. 实测附录：已登记负控的运行结果（追加，先登记后运行的顺序已遵守）

> 本附录在 §5 的负控登记**之后**写入。探针：`scripts/tests/test_q3_c2_consumed_state_probe.py`
> （只读，**8/8**，`C2_CONSUMED_STATE_GAP_CONFIRMED`）；证据
> `results/_q3_next_20261017/c2/consumed_state_probe.json`（未入库）。

### 8.1 Q1-N1：核心反例成立（**C1 通过，但消费状态不同**）

| 责任方 | 报告的摘要 | 实际消费状态的摘要 |
|---|---|---|
| M1-a | `c621a98d74b5cef2…` | `c621a98d74b5cef2…` |
| M1-b | `c621a98d74b5cef2…` | `c621a98d74b5cef2…` |
| **C0** | `c621a98d74b5cef2…` | **`a5d949da6f9f…`（不同）** |
| 安装方 | `c621a98d74b5cef2…` | `c621a98d74b5cef2…` |

⇒ **四份报告摘要逐字相同（C1 判据满足），而四份消费状态不同。**
无需哈希碰撞、无需篡改 digest ⇒ **Q-C2-1（`CONSUMED_STATE_BOUND_TO_FROZEN_IDENTITY`）
是独立于 C1 的必要条件，C1 不能替代它。**

### 8.2 Q1-N2：合法形态可实现（授权投影）

携带"由哪个冻结身份派生"的投影可被**独立复核**：对真实冻结体通过，对已变体拒绝
（`proj_ok=True / bare_ok=False`）。⇒ 绑定必须**随对象一起传递**，而不是并行传递一个摘要。

### 8.3 Q1-N3：只有复算能发现冻结后被改

绕过冻结（`object.__setattr__`）后身份 `c621a98d74b5cef2` → `7e4b46af31eca6a0`：
**只存旧摘要的一方看不见**，**前后各复算一次的一方看得见** ⇒ Q-C2-1 的 (c) 条是探测器。

### 8.4 Q2-N1：接口审计低估传递读取（实测）

`certify_route` 的**直接**属性集合为
`{capacity, demands, service_time, temp_class, tw_end, tw_start}`；
`env.dist_mat` 只通过 `_travel`/`_dist` **传递读取**，**不在**上述集合内。
⇒ 直接属性统计是**下界**；审计的 7/7 只能标 `C2_INTERFACE_AUDIT_OBSERVED`。

### 8.5 Q2-N2：标签已自带限定

审计产物的 verdict 为 `C2_INTERFACE_AUDIT_OBSERVED`（非 PROVEN）。

### 8.6 Q3-N1 / Q3-N2：C0 无可见性门（源码级）

| 项 | 实测 |
|---|---|
| 完整实例数组读取 | `env.dist_mat[inst_idx]`、`env.tw_start[inst_idx]`、`env.service_time[inst_idx]`、`env.reveal_time[inst_idx]`、`env.demands[inst_idx]`、`env.dataset['temp_class'][inst_idx]` |
| `reveal_time` 使用位置 | 仅 1 处（`:163`），随后作为 `rev` 传入下游用于**调度语义** |
| 与 `clock` 的比较 | **0 处** ⇒ **`has_visibility_gate = False`** |
| 索引集合来源 | 调用方的 `plan.get(v.vehicle_id, [])` 与 `v.committed_next`，**不是**可见性过滤 |

⇒ `C0_VISIBLE_PROJECTION_PROVENANCE`（Q-C2-3）在源码层面**未成立**；
这**不等于**任何既有运行发生了未来信息影响（探针不作此主张），但 C2 **不得**把 C0 的原始
`env` 访问路径认定为 strict-online 合格来源。

### 8.7 结论

| 项 | 状态 |
|---|---|
| C1（四方摘要一致 + 1 ulp 联动） | **必要但不充分** |
| Q-C2-1 消费对象绑定 | **未建立**（反例成立） |
| Q-C2-2 数据依赖完整性 | **未建立**（直接统计为下界） |
| Q-C2-3 C0 来源 | **未成立**（无可见性门） |
| C2 四方共同身份 | **尚未建立** |
