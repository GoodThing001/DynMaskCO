# D3 G0 剩余装配风险审计（任务 A，只读）

**日期**：2026-10-14 · **基准**：执行时最新 `origin/main` · **性质**：只读代码审查；**不改代码、不运行新科学实验**
**被审**：`scripts/simulation/d3_geo_adapter.py`（498 行）、`scripts/simulation/mpre_policy.py`、
`scripts/simulation/action_contract.py`、`test_d3_adapter_risks.py`（R1–R4）、
`test_d3_geo_adapter.py`（T-G1..T-G9 + X1/X2）、`test_d3_assembly_invariants.py`（A1–A4）
**已闭合项**（不重复提出修复）：R1 `pool_fn` 注入、R2 预算单位分离、R3 默认容差与主判 tie-break、
R4 失败回退、N3 GEO 非确定性拒绝、认证后超时、安装事务 API —— 均有测试，标注 **CLOSED**。

---

## 1. 分类总表

| # | 风险 | 分类 | 依据行 |
|---|---|---|---|
| A-1a | 评分向量长度 **短于** `legal` ⇒ **静默只取前缀**、选错动作（不报错） | **CONFIRMED** | 366 / 368 / 373 / 385 |
| A-1b | 评分向量长度 **长于** `legal` ⇒ `IndexError`（fail-loud） | CONFIRMED（fail-loud，可接受） | 385 |
| A-1c | 评分向量为 2-D ⇒ `np.argmax` 展平后静默错位 | **CONFIRMED** | 366 / 373 |
| A-1d | GEO 路径的 `deltas` 与 `keys` 长度不一致（自定义 state 供应方） | **OPEN** | 226–244 / 382–384 |
| A-1e | `state['vehicle_target_vid']` 短于 `legal` ⇒ `IndexError`（fail-loud） | MITIGATED | 382 / 386 / 398 |
| A-2 | **`certify_fn=None` 会跳过全部认证**并返回成功 ⇒ 事务层会安装未认证计划 | **CONFIRMED（G0 阻断）** | 412 / 427(成功返回) / 457 |
| A-3a | 共享 `apply_fn` 就地修改原计划 | MITIGATED（`apply_action` 新建对象） | 338 / 405 |
| A-3b | 适配器**自身不校验**承诺/已执行前缀/保护车未被改动 | **OPEN** | 405（无校验） |
| A-4a | 被安装的计划就是被认证的那个对象 | MITIGATED（同一对象） | 412–413 / 480 |
| A-4b | **未记录认证计划/安装计划的身份哈希**；且账本按**引用**保存计划 | **NOT_VERIFIABLE** | 459 / 无 hash 字段 |
| A-5a | 事务捕获异常 ⇒ 0 安装 | MITIGATED（A3 覆盖） | 471–477 |
| A-5b | **重复调用 `finish` 会重复安装**（无幂等守卫） | **CONFIRMED** | 449–461 |
| A-5c | **无安装回执持久化** ⇒ `P_after_install` 不存在 | **CONFIRMED** | 447 / 460（仅内存） |
| A-5d | `rollback_fn=None` 时仍计一次 rollback，但**没有任何状态被恢复** | **OPEN** | 452–455 |
| A-6 | `if not legal:` 分支后有**两行不可达代码** | CONFIRMED（无行为影响） | 361–364 |

## 2. A-1 评分长度 / shape 与 `legal` 索引一致性

`366: scores = np.asarray(score_fn(state))`；`368: if not np.isfinite(scores).all()` —— **只查有限性，不查长度与 shape**；
`373: idx = select_historical_argmax(scores)` ⇒ `385: legal[idx]`。

**反例设计（本轮仅设计，不实施）**

| 反例 | 构造 | 预期可观察后果 |
|---|---|---|
| C1-len-short | `legal` 3 个候选，`score_fn` 返回长度 1 的向量 | **不报错**；`argmax` 得 0 ⇒ 只对候选 0 打分即决定动作 ⇒ **静默选错**（最严重） |
| C1-len-long | 返回长度 5 | `legal[3]` ⇒ `IndexError`（fail-loud） |
| C1-2d | 返回 `(3,1)` 数组 | `np.argmax` 展平 ⇒ 若展平后总分 ≥ `len(legal)` 可静默错位 |
| C1-geo-mismatch | 自定义 `state[GEO_SCORE_KEY]` 比 `legal` 短 | `T` 内索引可能 ≥ `len(keys)` ⇒ `IndexError`；若更长则尾随候选被**静默忽略** |

**最小修复方向（下一轮，不在本轮实施）**：在 366 之后加断言
`len(scores) == len(legal) and np.asarray(scores).ndim == 1`，不符即 `_rollback('score_shape')`，
并把该原因加入 `FAILURE_REASONS`；GEO 路径同时断言 `len(state[GEO_SCORE_KEY]) == len(legal)`。
**注意**：改共享路径会改变 F 的行为判据，须按新决策策略登记 + 等价测试。

## 3. A-2 `certify_fn=None` 进入生产安装路径（**G0 阻断**）

`412: if certify_fn is not None:` ⇒ 为 `None` 时**完全不做认证**，函数在末尾以
`failure_reason=None` 正常返回 ⇒ `run_repair_transaction` → `ledger.finish` → `install_fn(plan)` **安装**。

**该风险已被现有通过的测试实际演示**：A3 的"成功路径恰好安装一次"用例（`test_d3_assembly_invariants.py`）
**没有传 `certify_fn`**，却断言 `n_installs == 1` ⇒ 即"未认证也安装"。G0 要求"完整 C0 认证"，
故这是**已证实且阻断 G0** 的风险。

**反例设计**：`run_repair_transaction(certify_fn=None, ...)` 在成功路径下 `n_installs == 1` 且
`n_candidate_plans_certified == 0` ⇒ 安装次数 > 认证次数。
**最小修复方向**：为"允许安装"的入口引入必需的 `certify_fn`（缺失即 `ValueError`），
或在 `InstallLedger.finish` 增加 `certified` 标志的强制要求；把"无认证路径"仅保留给显式的诊断入口。

## 4. A-3 `apply_fn` 与原计划 / 冻结承诺

* `338: P = dict(plans)` 为**浅拷贝**；`405: P = apply_fn(P, a, allowed_vehicle_ids=mutable_ids)`。
  `action_contract.apply_action` 会新建 `VehiclePlan` 对象 ⇒ 调用方的计划对象**未被就地修改**
  ⇒ **MITIGATED**（仅对共享 `apply_fn`）。
* 但适配器**自身**不检查 `committed_next`、已执行前缀、保护车是否被保持；它完全依赖注入的
  `legal_fn`（掩码池）与调用方的 `validate_repair_plan`。若调用方传入一个**包含 committed 客户**
  的 `legal_fn`，适配器会照常应用。⇒ **OPEN（适配器边界）**。
**反例设计**：构造 `legal_fn` 返回一个 `committed_next` 客户的插入动作 ⇒ 断言适配器应拒绝
（当前不会拒绝）。

## 5. A-4 认证计划与真实安装计划的身份一致性

* 认证的对象（`413` 的 `certify_fn(P)`）与成功返回的对象、以及 `ledger.finish(plan=...)` 传给
  `install_fn` 的对象是**同一个** ⇒ 语义上一致 ✔ **MITIGATED**。
* 但**没有任何身份哈希**在认证点与安装点被记录；`459: self.installed_plans.append(plan)`
  保存的是**对象引用** ⇒ 之后任何就地修改都会改变"账本以为安装了什么"。
  事后**无法验证**"安装的计划 == 认证的计划"⇒ **NOT_VERIFIABLE**。
**反例设计**：认证 → `plan[vid].suffix` 就地改一位 → 断言 `installed_plans[0]` 与认证快照应当不等
（当前无任何断言会触发）。
**最小修复方向**：认证时记录 `plan_hash(certified_plan)`，安装时重算并要求相等；
账本保存**不可变副本/hash** 而非引用。

## 6. A-5 事务异常 / 重复安装 / 缺失回执

* 异常 ⇒ `exception:<Type>` + 0 安装 ✔（A3 覆盖）**MITIGATED**
* **重复安装**：`finish` 无幂等守卫 ⇒ 调两次装两次 ⇒ **CONFIRMED**。
  反例：`ledger.finish(...)` 连调两次（成功路径）⇒ 断言 `n_installs` 应为 1，当前为 2。
* **缺失回执**：`events` / `installed_plans` 仅存内存，**不落盘** ⇒ `P_after_install` 无持久证据
  ⇒ **CONFIRMED**（与既有字段缺口一致）。
* **空回滚**：`rollback_fn=None` 时仍 `n_rollbacks += 1` ⇒ 记为"已回滚"但**无状态被恢复**
  ⇒ **OPEN**（应在 `rollback_fn is None` 时显式标记 `rollback_unverified`，或要求必填）。

## 7. A-6 不可达代码（附带发现）

```
361:  if not legal:
362:      return _rollback('no_legal')
363:      diag['n_steps'] = len(steps)        <- 不可达
364:      return _rollback('no_legal')        <- 不可达
```
行为不受影响，但属实现残留；建议下一轮清理（本轮只读，不修改）。

## 8. 已有测试覆盖（CLOSED，不重复提出）

| 项 | 测试 | 状态 |
|---|---|---|
| R1 `pool_fn` 注入 | R1 反例 | CLOSED |
| R2 预算六项分离 | R2 反例 | CLOSED |
| R3 默认 `eps=0` / legal-order-first / `single_factor_with_F` | T-G5 / R3 / A1 / A2 | CLOSED |
| R4 失败回退原计划 | R4 反例 / A3 | CLOSED |
| N3 GEO 非确定性拒绝 | A1 | CLOSED |
| 认证后超时判 timeout | A4 | CLOSED |
| 安装事务 API 存在且异常 0 安装 | A3 | CLOSED |
| **评分长度/shape 一致性** | **无** | **OPEN（A-1a/c）** |
| **certify_fn=None 不得安装** | **无（且现有用例反证）** | **OPEN（A-2）** |
| **重复安装幂等** | **无** | **OPEN（A-5b）** |
| **认证/安装计划身份哈希** | **无** | **OPEN（A-4b）** |

## 9. 本轮不实施

以上仅给**最小反例设计**与修复方向；**未修改任何代码**，**未运行任何新实验**。
所有 OPEN/CONFIRMED 项构成 **G0 未闭合**的一部分；G0 通过前不得进入 G1/G2。
