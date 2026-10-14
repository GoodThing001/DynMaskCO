# D3 装配接口独立审计（P0 任务一）

**日期**：2026-10-14 · **基准提交**：`d6d17a041b305caf3d9940a090b49763da1d87b3`（执行时 HEAD 与 `origin/main` 一致，工作树仅剩无 `.gitmodules` 映射的幽灵子模块）
**性质**：只读审计 + **仅对新增代码**施加最小修复。**未改历史代码、未改任何冻结目录**、未训练、未采集、未运行 F−D。
**方法**：先写反例测试并**在修复前运行**（失败原件保留），再修复，再用同一套测试复验。

---

## 0. 结论

用户指出的四项装配风险**全部成立**。修复前反例套件 **2 通过 / 8 失败**；修复后 **10 通过 / 0 失败**；
回归套件由 32/0 变为 **33/0**（新增 1 条 epsilon 契约断言）。
原件：`raw/test_risks_PREFIX_failures.txt`（修复前失败）、`raw/test_risks_POSTFIX.txt`（修复后全绿）。

| # | 风险 | 成立 | 严重度 | 处置 |
|---|---|---|---|---|
| **R1** | `pool_fn` 注入接口形同虚设 | ✅ 实测 `UnboundLocalError` | 中 | 已修 + 反例 |
| **R2** | `n_solver_calls` 语义错位（实为动作应用次数） | ✅ 实测计数 3 而真实 solver 调用 0 | 中 | 已修 + 反例 |
| **R3** | F 与 D-Geo 的 tie-break **不相同** ⇒ 不是单因素 | ✅ 构造性反例（见 §3） | **高** | 已修：默认 `eps=0`，并加"是否单因素"标记 |
| **R4** | 失败时返回**部分修复计划**，调用方可能误装 | ✅ 实测返回 `{1:[4,4]}` 而原计划 `{1:[4]}` | **高** | 已修：全部失败路径回退原计划 |

## 1. R1 `pool_fn` 注入路径

**源码（修复前 `d3_geo_adapter.py:102-104`）**

```python
102:  if pool_fn is None:
103:      from dynmaskco_cc_context import mask_candidate_pool
104:  pool = [int(c) for c in mask_candidate_pool(vehicles, served_mask, visible_ids, ...)]
```

**构造性反例**：传入非空 `pool_fn` ⇒ 第 103 行不执行 ⇒ 第 104 行 `UnboundLocalError:
cannot access local variable 'mask_candidate_pool' where it is not associated with a value`，
且注入的 `pool_fn` **从未被调用**。

**最小修复**：`from dynmaskco_cc_context import mask_candidate_pool as pool_fn`（把默认实现绑到同名局部名），
随后**无条件**调用 `pool_fn(...)`。修复后反例断言：注入函数被调用 **1 次**、返回顺序 `[7,8]` 被保留。

## 2. R2 `n_solver_calls` 计数语义

**源码（修复前）**：`budget.bump('n_solver_calls')` 位于每次 `apply_fn()` 之后 ⇒ 它数的是**动作应用次数**。
本适配器**从不调用任何 solver**，却报出 `n_solver_calls = 3`。

**为什么必须分开**：用户裁定指出 `n_eval`、模型评分、候选枚举、完整 C0 认证与 solver 调用
**不是天然等成本的工作单位**。把"应用次数"记成"solver 调用"会让一个比较**声称等 solver 工作量**，
而两臂实际做的量不同。

**最小修复**：`Budget.KEYS` 拆为六项 ——
`n_actions_enumerated / n_actions_scored / n_candidate_plans_certified / **n_action_applications** /
**n_solver_calls** / wall_seconds`；`apply_fn` 后只加 `n_action_applications`；
`n_solver_calls` 由装配层用 `report_solver_calls(n)` **显式上报**，适配器内恒为 0 并在 `snapshot()` 里写明原因。
**历史记录不改**（旧记录本就不含此字段）。

## 3. R3 F 与 D-Geo 的 tie-break 差异（本轮最重要的发现）

### 3.1 差异究竟在哪

| 臂 | 规则 | 平局如何解决 |
|---|---|---|
| **F** | `np.argmax(scores)` = **首个最大值** | 按 `legal` 下标顺序（`legal` 若按规范键排序，则等价于**最小规范键**） |
| **D-Geo** | 统一最小参照 + 规范键 | 容差带 `eps` 内的候选，取**最小规范键** |

⇒ 二者**平局键序相同**（在规范顺序下），**真正的差别是容差带 `eps`**：
`eps = 0` 时两者**结构上完全一致**（`argmax(−Δd)` ≡ `argmin(Δd)` 且平局键序相同）；
`eps > 0` 时 D-Geo 会把"严格更差但在 eps 内"的候选也纳入平局，**可能选出与 F 不同的动作**。

### 3.2 构造性反例（证明差异真实存在）

```
候选 0：customer 4（规范键更小），Δd = 1.0 + 0.5e-12   ← 距离更大
候选 1：customer 5（规范键更大），Δd = 1.0             ← 距离更小
```

* F：`argmax([−1.0000000000005, −1.0]) = 1` ⇒ 选 customer 5（严格最小距离）
* D-Geo（`eps=1e-9`）：`T = {0,1}` ⇒ 取最小规范键 ⇒ 选 **customer 4** ⇒ **与 F 不同**
* D-Geo（`eps=0`）：`T = {1}` ⇒ 选 **customer 5** ⇒ **与 F 相同**

⚠️ 首次构造我写反了（让"更大距离"同时拥有"更大键"），两臂恰好同选、反例不成立；
已按上述方向重写并在测试里加 `assert keys[0] < keys[1]` 固定构造方向。

### 3.3 处置（**不修改 F 的历史行为**）

* **默认 `eps = 0`**（`EPS_D_PRIMARY_REL/ABS = 0.0`）：此时 D-Geo 的平局解决与 F **结构一致**，
  F 与 D-Geo **只剩评分向量这一个因素不同** ⇒ 满足"只替换评分函数"的预声明。
* `select_geo` 每步输出 **`single_factor_with_F`** 标记：`eps == 0` 为 `True`，
  `eps > 0` 为 `False` 且 `eps_is_provisional=True`。
* `eps > 0` 的容差版本**降级为注册的稳健性变体**，**不得**用作主对照；
  若非要当主对照，则比较必须如实改称"**学习评分策略 vs 几何评分策略 + 容差**"，
  **不得**再宣称只有评分数值这一个因素不同。
* `unified_minimum_tie_set` / `select_geo` 的默认值随之改为 0；
  原暂定值 `1e-9 / 1e-12` 保留为 `EPS_D_REL/ABS_PROVISIONAL`（仅变体使用），
  **仍未被冻结**（冻结前须完成 `D3_NUMERIC_STABILITY`）。

### 3.4 仍需在装配层补的三件事（用户要求，属"下一步"而非本轮）

1. 固定并登记**共同动作顺序**（规范键序），并在两臂运行记录里落盘该顺序；
2. 独立证明 **F 的历史选择行为未发生未经声明的变化**（本轮的 X1/X2 只覆盖适配器自身，
   **不能自动追认**装配层）；
3. 记录**距离平局、logits 精确平局与所选动作**（三个量分别落盘），
   以便事后区分"因容差改变"与"因评分改变"。

## 4. R4 失败路径可能返回部分修复计划

**源码（修复前）**：`timeout` / `no_legal` / `nan` 三条路径都 `return P, ...`，而 `P` 是**工作副本**，
在 ≥1 步之后**已不是原计划**。实测：mask `[4,5]`、第 2 步 `no_legal` ⇒ 返回 `{1: [4, 4], ...}`
而原计划是 `{1: [4], ...}`。仅记录 `failure_reason` **不足以**阻止调用方安装半修计划
（`_identity_apply` 这类不改计划的实现会掩盖该问题，故反例用了会真改计划的 `apply_fn`）。

**最小修复**：新增内部 `_rollback(reason)`，**所有**失败路径统一返回
`dict(plans)`（**调用方传入的原计划**），并把工作副本作为证据留在
`diag['partial_plan_at_failure']` + `diag['rollback']=True` + `diag['rollback_to']`。
成功路径（`failure_reason=None`）仍返回修好的计划。

⚠️ **仍未闭合**（须装配层保证）：适配器的回退只是"返回原计划"，
**真正的保证**是**调用方**在 `failure_reason is not None` 时**不得安装**返回计划。
该不变量必须由装配层的不变量测试覆盖，本轮只做到"不把部分计划当作结果返回"。

## 5. 验证

| 套件 | 修复前 | 修复后 |
|---|---|---|
| `scripts/tests/test_d3_adapter_risks.py`（本轮新增，R1–R4 反例） | **2 / 8 失败** | **10 / 0** |
| `scripts/tests/test_d3_geo_adapter.py`（T-G1..T-G9 + X1/X2） | 32 / 0 | **33 / 0** |

回归中唯一的变化是 T-G5 的 epsilon 契约断言：默认 `eps=0` 后
`eps_is_provisional` 为 `False`、`single_factor_with_F` 为 `True`；
已把该断言改为**分别覆盖两条分支**（默认=单因素；显式正容差=暂定且非单因素），
**没有**改容差或放宽判据来让测试变绿。

## 6. 本审计**不**主张

* 不改写历史裁决；不主张 F 与 D-Geo 有任何效果差异；
* 不把"工程实现完成"升级为"主模型有效性获得证明"；
* 本轮通过**不**等于装配层合格：§3.4 与 §4 的装配层不变量仍是前置条件。
