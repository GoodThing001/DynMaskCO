# G-3 预声明 · 勘误 04：B-9b 划出范围、L-5 与回滚资格差异（2026-10-17，追加式）

> **性质：勘误（amendment），追加式，只读审计先于本文件登记。**
> **不改写** 17/17 记录、勘误 03、隔离候选源码与工程诊断报告；不改任何生产 helper。
> 基准 HEAD：`0e38936a0b2d4e5a01ed518dbe67fa1b300c65ee`。
> 审计证据：`results/_q3_next_20261017/g3/l5_rollback_audit.json`（未入库；只读，6/6 观测）。

---

## 1. B-9b 的治理裁决：选 **(c)**，并在 G-3 内保留 **(a)**

| 选项 | 内容 | 裁决 |
|---|---|---|
| (a) | 把 `INSTALL_OUTCOME_UNVERIFIED` 固化为终态并写入适用域 | **采纳**：G-3 内无法证明真实安装结果的路径一律维持该终态，**不得**宣称"没有安装"或"回滚成功" |
| (b) | 在 G-3 隔离候选中引入可复核的回滚契约 | **不采纳**：会让一个专注"计划身份"的研究杠杆承担真实执行器、状态持久化与外部副作用的职责，扩大验证范围并引入混杂 |
| (c) | 把真实安装事务原子性划出 G-3，交由**单独预声明**的生产安装器设计 | **采纳**：B-9b 不再是 G-3 的判据 |

**理由**：即使"回滚后状态相等"成立，也**不**等于所有外部副作用已撤销（例如已下单、已写盘、
已推进外部状态机）。因此原子性既不能在 G-3 内证明，也不应在 G-3 内假装解决。

**G-3 只对它能够建立的性质负责**：

1. 计划**内容身份**（完整、无歧义、可逐字段核对）；
2. **冻结对象一致性**（认证对象 = 使用对象 = 归档对象）；
3. **认证—安装交接证据**（四档结论各自的可证内容）。

---

## 2. 资格缺口 1：L-5 锚点交叉核验**未强制**，且容差制未预声明

### 2.1 已核实的三处源码事实

| # | 事实 | 位置 |
|---|---|---|
| **G-1a** | `install_with_frozen_identity(..., frozen_state=None, require_frozen_state=False)`：**默认不做锚点交叉核验**，安装照样继续 | 候选 `:367-368` + `if require_frozen_state or frozen_state is not None:` |
| **G-1b** | 交叉核验用的是**绝对容差** `1e-12`（时间）与 `1e-9`（载货）；在 24.0 h 处 `1e-12 ≈ 281.5 ulp`，而 G-1 的判据是 **1 ulp 必须被识别** | 候选 `:350`、`:356` |
| **G-1c** | committed 车辆的期望载货取**调用方提供的 `load`**，**没有**独立重算待接订单需求量 | 候选 `:343`（注释亦承认） |

### 2.2 实测（审计行 L5-1 / L5-2 / L5-3）

| 行 | 观测 |
|---|---|
| **L5-1** | 传入 `anchor_time=12345.0`、`anchor_load=999.0` 的"凭空锚点"，**不**提供冻结状态 ⇒ `INSTALL_COMPLETED`，回调被调用 1 次，`refusals == []` |
| **L5-2** | 接受 1 ulp 与 1e-13（≈28 ulp）差异；仅 2e-12 被拒 ⇒ **有效粒度 ≈ 281.5 ulp**；载货同理（1e-10 过、2e-9 拒） |
| **L5-3** | 调用方同时给出"计划 `anchor_load=999.0`"与"状态 `load=999.0`" ⇒ 交叉核验**通过** ⇒ 它验的是**一致性**，不是**来源** |

### 2.3 结论与登记

> **B-9a 由"已闭合"降级为"有条件闭合（conditionally closed）"。**
> 深度冻结与计划内容比对**有隔离诊断支持**（17/17）；
> **L-5 的唯一来源约束尚未闭合**，且在默认参数下**根本不执行**。

**下一次隔离审计前必须完成（不得静默取容差）**：

1. 把冻结状态设为**强制输入**（缺失即 `REFUSED: FROZEN_STATE_MISSING`，**不得**继续安装）；
2. **预先声明**锚点比对的要求：**位级一致**（推荐，与 G-1 一致）还是**经独立证明的容差一致**；
   若是后者，容差必须**登记**并给出其**上界推导**，不得写死 `1e-12`；
3. committed 车辆的载货**必须**由冻结状态独立重算（`current_load + demand[committed_next]`），
   不接受调用方直接给 `load`。

---

## 3. 资格缺口 2：`INSTALL_ROLLED_BACK` 的实现弱于预声明的证明条件

### 3.1 事实

勘误 03 §2 的登记条件是"**回滚完成且回滚后状态经过复核**"。
候选 `:463` 实际是：

```python
receipt['rollback_verified'] = bool(rollback_fn(frozen))
```

**只要回滚函数返回真值就升级为 `INSTALL_ROLLED_BACK`，没有任何独立的后回滚状态验证器。**

### 3.2 实测（审计行 RB-1 / RB-2）

| 行 | 观测 |
|---|---|
| **RB-1** | 一个**什么也不做**、只 `return True` 的回滚函数 ⇒ `outcome = INSTALL_ROLLED_BACK`、`rollback_verified = True`、`rollback_did_anything = False` |
| **RB-2** | 回滚函数 `return False` ⇒ `INSTALL_OUTCOME_UNVERIFIED`（阶梯本身是有效的） |

### 3.3 结论与登记

> **`INSTALL_ROLLED_BACK` 只能读作"回滚函数报告成功"，不能读作"回滚完成"。**
> 报告已诚实披露（`proves` 文本含 "not independently verified"），
> **但披露局限不等于满足原判据**——这是**实现弱于登记**，必须登记。

**后续生产规格的处置**（不在 G-3 实现）：

1. 把**回滚函数自报成功**与**独立验证结果**拆成两个字段
   （`rollback_reported_success` / `rollback_state_verified`）；
2. 在真正验证前维持结果为**未知**，不得使用"回滚完成"措辞；
3. 独立状态复核是将来研究 B-9b 的**必要**组成，但**不充分**（见 §1 的理由）。

**本轮不改候选源码**：改名会同时改动已入库的 17/17 语义；正确做法是追加登记，
并在下一次隔离审计时连同 §2 的三项一起处理。

---

## 4. 本轮不做什么

- 不在 G-3 内引入回滚框架（不选 (b)）；
- 不改候选源码、不改 17/17 记录、不改勘误 03；
- 不改生产 helper、不接线、不真实安装；
- 不启动正式实验 / 训练 / GPU / 最终测试种子。

---

## 5. 状态（不变）

`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`（永久，不追认）、
**V-10 干净 checkout FAIL**、`THERMAL_CONTINUOUS_NOT_ESTABLISHED`、
`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、L1 `BLOCKED`、`L2/A3_NOT_ESTABLISHED`、
`stable_online_gain=False`、`HISTORICAL_BACKBONE_UNVERIFIED`、C0 `AWAITING_USER_BUDGET_RULING`、
`t_min` 恒 0、最终测试种子 `SEALED`、`PRODUCTION_NOT_QUALIFIED`、B-1…B-9 全开
（其中 **B-9a 有条件闭合、B-9b 已划出 G-3 范围**）。
