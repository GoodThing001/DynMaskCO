# D-Geo 实现规格与 T 系列测试设计（**待裁定冻结；不授权运行比较**）

**日期**：2026-10-14 · **状态**：`SPEC — approved to draft, NOT approved to run`
**授权**：用户 2026-10-14 裁定②「批准起草实现规格，使用统一动作枚举，只替换评分函数」；
允许**设计、静态审查与测试准备**；**不**放行 F−D 比较。
**作用域**：用户裁定①选 **B（训练期严格语义）** 作为首个固定状态归因实验的主定义。
**禁止**：不修改 `MASKCO_code/`；不修改任何旧冻结树 / gate / checkpoint / 封存结果；不启动训练或采集。

---

## 1. D-Geo 是什么、不是什么

**是**：现有共同逐步修复框架（`mpre_policy.stepwise_sample`）里的一个**非学习评分器**。
**不是**：新求解器；不自己做客户选择；不引入新的动作枚举、可行性过滤或认证路径。

唯一被替换的东西：**评分函数**。

## 2. 已核实的接口事实（决定了实现形态）

`scripts/simulation/mpre_policy.py`：

```python
def stepwise_sample(score_fn, extract_fn, env, inst_idx, plans, mask_set, mutable_ids, ...):
    P = _plan_copy(plans); remaining = list(mask_set)
    while remaining:
        legal = enumerate_legal_actions(env, inst_idx, P, remaining, mutable_ids)   # [(c, action, cand)]
        if not legal: return P, steps, records, 'no_legal'
        state  = extract_fn(P, remaining, legal)
        scores = np.asarray(score_fn(state))                                        # len == len(legal)
        if deterministic: idx = int(np.argmax(scores))                              # 首个最大值
        ...
```

⇒ **`score_fn` 只收到 `state`**（用户判断正确）。因此增量距离**必须**由**共同的 `extract_fn`** 暴露，
且与 `legal` **索引一一对应**。

⚠️ 同一源码还暴露两个**必须由规格钉死**的风险：

1. `remaining = list(mask_set)` —— 若调用方传 `set`，**动作顺序无保证**；而 `argmax` 取**首个最大值**
   ⇒ 平局结果取决于 `legal` 的顺序 ⇒ **tie-break 事实上不确定**。
2. `enumerate_legal_actions` 的顺序是 `for c in remaining: for cand in cands:` —— 同样继承上面的顺序。

## 3. 最小接口（唯一允许的评分入口）

```python
GEO_SCORE_KEY = 'geo_incremental_distance'      # state 中的键，长度 == len(legal)，按 legal 顺序对齐

def geo_score(state):
    """D-Geo：s_D(a) = −Δd(a)。只读 state 中由共同 extract_fn 产出的增量行驶距离。"""
    d = state[GEO_SCORE_KEY]
    return [-float(x) for x in d]
```

**`state[GEO_SCORE_KEY]` 的产生方式（硬性）**：在**共享** `extract_fn` 内，对 `legal` 中每个
`(c, action, cand)` 计算**同一环境返回的增量行驶距离**，按 `legal` 顺序追加。
**不得**为新臂另写一条枚举/评分旁路。

## 4. 五条硬性要求（用户裁定②原文落库）

| # | 要求 | 可检查形式 |
|---|---|---|
| **G1** | **共同候选机制**：F 与 D-Geo 使用相同的 `enumerate_legal_actions()`、同一初始计划、同一 **B 类掩码**、同一可变车辆集合、同一硬可行过滤；D-Geo **不另外选择客户** | 两臂的 `legal` 在**共同初始状态与相同部分计划**上必须逐项相同（含顺序） |
| **G2** | **评分口径**：`incremental_distance` 必须是**同一环境返回的增量行驶距离**；**不得**混入冷链复合成本、shadow utility、接拒罚金 | `GEO_SCORE_KEY` 的取值须能与环境的距离口径逐点对上；出现其它成本字段即判实现缺陷 |
| **G3** | **固定 tie-break**：事前规定动作的**完整规范排序**；距离相同或浮点近似相同时必须确定，**不得**依赖容器偶然遍历顺序 | 见 §5（顺序钉死 + 显式平局规则 + 容差策略） |
| **G4** | **完整认证**：候选须经共同的计划对账 + 分区检查 + **C0 硬认证**；**不得**仅凭插入动作的局部 `feasible=True` 认定整条计划合格 | 记录 `validate_repair_plan` / `validate_full_partition` / `certify_plan` 三者的结果，任一不过即不采纳 |
| **G5** | **失败也入账**：空候选、无法完成修复、超时、原计划回退**均记录**；**不得**只比较两臂都成功的样本 | 逐状态记录 `failure_reason ∈ {None,'no_legal','nan','timeout','rollback','cert_fail'}` 与两臂各自的终态 |

## 5. 顺序与平局（G3 的具体化，必须写死）

1. **规范顺序**：传入 `stepwise_sample` 的 `mask_set` 必须是**有序序列**（建议按客户 id 升序），
   **不得**传裸 `set`；`enumerate_legal_actions` 的 `legal` 顺序由该序列唯一决定，须**落盘记录顺序**。
2. **显式平局键**：平局时按**完整规范排序**决定，键序建议
   `(customer_id, vehicle_id, slot.kind, slot.anchor, position, predecessor, successor)`，全为确定性整数字段。
3. **容差策略**：当两动作的 `|Δd| ≤ eps_d`（`eps_d` 事前写死，建议 `1e-9` 相对 + `1e-12` 绝对）时
   视为**并列**，**按第 2 条的键序裁决**，**不**用浮点比较结果裁决。
   ⚠️ 理由：本项目已记录定容 padding 会引入 ~1e-5 级真实重结合噪声并能翻转 argmax。
4. **确定性回归**：同输入重复运行（同进程 / 跨进程各 ≥20 次）须给出**逐字节相同**的 `steps` 序列。

## 6. 动作分歧不是身份不匹配（方法学要点，用户裁定②）

两臂第一步动作不同后，后续 `legal` 集合**自然可能不同**——这是**策略导致的状态分叉**，
**不**自动构成实验身份不匹配。

⇒ G1 的"共享动作集合"机械等价**只施加于**：**共同初始状态**与**相同部分计划**（即分叉之前）。
⇒ 分叉之后必须**逐状态记录**两臂各自的 `legal` 集合指纹、所选动作与分叉点，
但**不得**要求其继续相等，**也不得**把分叉本身当作失败。

## 7. R2（regret-2）的正确措辞（修正 D3 草案）

草案 §1 写"三臂共享同一 tie-break"是**错的**。R2 有自己的**客户选择**与 **regret 排序**机制。

| | 共享 | 不共享 |
|---|---|---|
| R2 vs D-Geo | 原始状态、掩码（B 语义）、合同、评价器、C0 认证 | **决策规则**：R2 有客户选择 + regret 排序；D-Geo 只按增量距离评分 |

⇒ 正确表述：**三臂共享状态/掩码/合同/认证/评价器；只有 F 与 D-Geo 共享"动作枚举 + tie-break"**，
R2 作为**独立的较强非学习对照**报告，**不得**假装采用与 D-Geo 相同的决策规则。

## 8. B 类掩码：必须在**采样之前**构造（用户裁定①的关键实现要求）

**禁止**：从含 deferred 客户的原池随机采样，再靠 B 的断言失败去筛状态。

**要求**：采样前，用**同一规则**为**所有臂**构造严格可用的掩码范围：

```text
mutable_plan_customers = { x : x in plan.suffix(v), x>0, v in mutable_ids }      # 训练期 B 语义
pool_B = [c for c in mask_candidate_pool(...) if c in mutable_plan_customers]   # 保序
若 pool_B 为空 ⇒ 原样记录该状态为空池，不得事后剔除该状态
```

**源码依据（为什么 A 不可行）**：`mpre_policy.validate_repair_plan` 第 116/131 行
`detail['extra'] = sorted(cur_customers - base_customers)`、`ok = not any(detail.values())`
⇒ **任何**"原计划没有的客户"都会进 `extra` 并使 `ok=False`，**整条计划被拒**。
deferred 客户按定义不在任何 `base_plans` suffix 中 ⇒ 用 A 门插入它**必然**被对账器拒绝。
故 A 与既有修复对账器是**结构性冲突**，不是"支持不全"；A 保留为后续
"deferred 客户进入路线重构"的**独立扩展方向**（需先扩展对账器语义）。

**声明限制**：本作用域限定是**新的预声明**，**不得**声称完整重放了历史训练时的采样分布，
**不得**用于覆盖旧结果。

## 9. T 系列测试设计（纯离线、无实验、不产出效果结论）

| 编号 | 断言 | 判据 |
|---|---|---|
| **T-G1** | 索引一一对应 | `len(geo_score(state)) == len(legal)`，且第 i 项的增量距离与**独立复算**的第 i 个 `(c, action, cand)` 一致 |
| **T-G2** | 评分口径纯净 | 改冷链复合成本 / shadow utility / 接拒罚金**不影响** `geo_score` 输出（注入式反例） |
| **T-G3** | B 掩码前置 | 掩码构造发生在采样**之前**；含 deferred 的原池经 `pool_B` 过滤后**不含** deferred；**空池原样记录** |
| **T-G4** | 顺序钉死 | `mask_set` 传 `set` ⇒ 触发拒绝或规范化；同一有序输入重复 ≥20 次（含跨进程）`steps` **逐字节相同** |
| **T-G5** | 平局裁决 | 构造 `|Δd| ≤ eps_d` 的两动作 ⇒ 由**键序**裁决，且与浮点比较结果**无关**（人为翻转浮点顺序仍得同一动作） |
| **T-G6** | 完整认证 | 局部 `feasible=True` 但整条计划对账/分区/C0 任一不过 ⇒ **不采纳**，记 `cert_fail` |
| **T-G7** | 失败入账 | `no_legal` / `nan` / `timeout` / `rollback` 四类各造一个反例 ⇒ 均被记录且计入分母 |
| **T-G8** | 分叉语义 | 两臂第一步不同后，后续 `legal` 不相等 ⇒ **不判 FAIL**；但分叉点与两臂 `legal` 指纹必须落盘 |
| **T-G9** | 源码身份 | 新冻结树的逐文件 SHA 与运行环境记录齐备；旧树**零改动** |

**T 系列只判"实现是否正确"，一律不读效用、不判效果。**

## 10. 资格层次（不得混层）

* **D3-Engineering**（本规格服务的层次）：B 掩码、D-Geo 评分、动作枚举、共同 C0、源码身份、字段记录机制。
* **D3-Science**（**未授权**）：学习贡献与在线效用，需 E2 训练资格 + 前瞻采集资格 + 完整 8+2 秒发布门 + 统计预声明。

⇒ D3-Engineering **不得**追认旧 D2；**不得**因为候选动作不同就声称科学增益成立。
