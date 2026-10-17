# G-2 时间参数身份与 strict-online 来源资格审查 · 七要素预声明（2026-10-17）

> **性质：预声明（pre-registration），先于实施登记。** 本文件在写任何 G-2 代码之前登记假设、
> 唯一主对照、判据、参数与次数、预期方向、失败条件、资源门与代码身份。
> **本轮只授权 G-2。** 不启动 G-1/G-3/C2/G5/G6/C3；不修改生产认证器与安装入口；
> 不训练、不用 GPU、不跑正式 rollout、不碰最终测试种子；**不自动 commit/push**。
>
> **本轮的判据不是"再拿到一个漂亮的 PASS"，而是排除认证链里最危险的一种状态：
> 数学判据正确，但它的时间参数来自未来信息，或它的身份无法对应真实执行过程。**
> 因此本轮**允许**的结论包含 `NOT_CERTIFIABLE` / `REFUSED`，并且这些结论**不视为失败**。

---

## 0. 代码身份（执行前登记，全部取**提交 blob**摘要）

> 依据上一轮勘误 §4.2：`core.autocrlf=true` 下工作区字节不是可复现身份，故一律用
> `git show HEAD:<path>` 的 blob 摘要。本轮六个被审文件的 blob 与工作区字节**恰好相等**（已核）。

| 项 | 值 |
|---|---|
| HEAD | `a71a0c3cede64b15e20f34f7833134d6236bb7a1` |
| `scripts/simulation/strict_online_env.py` | `fd17fe77f8375463…`（blob） |
| `scripts/coldchain/coldchain_state.py` | `a2ee98dcb7290f52…`（blob） |
| `scripts/coldchain/coldchain_contract.py` | `0fb9e6e8fcff11c5…`（blob） |
| `scripts/m1b_adapter/m1b_quality_certificate.py` | `2aa801d09c26e685…`（blob） |
| `scripts/m1b_adapter/m1b_q2_certifier.py` | `7674205b34db5fc5…`（blob，= 上一轮 G-TMIN 最终身份） |
| `scripts/coldchain/q3_time_budget_audit_experimental.py` | `a20e1430c726cacb…`（**未跟踪**，工作区字节；本地试验资产） |
| 候选模块（待创建） | `scripts/coldchain/q3_g2_time_identity_experimental.py` |
| 测试（待创建） | `scripts/tests/test_q3_g2_time_identity.py` |

完整 64-hex 摘要存放在报告 §1 的 pin 块（源码里不放裸 64-hex，避免暂存区密钥扫描误报）。

---

## 1. 假设（assumption）

### 1.1 读源码阶段**已核实**的事实（不是预测，是源码事实；负控只负责把它们机械化）

| 编号 | 事实 |
|---|---|
| **F-1** | `horizon = float(self.tw_end[inst_idx].max())`（`strict_online_env.py:454`；`run_resumed` 同式在 `:596`）。`tw_end` 是 `(N, nodes)` **完整**数组，**包含尚未揭示订单**的时间窗。 |
| **F-2** | `max_iter = 10000`（`:481`）是 `_run_loop` 的**迭代**上限，源码里**没有**任何注释或断言把它等同于"每辆车温度推进次数上限"。 |
| **F-3** | `_advance_coldchain_to` 在 env 内只有三处调用：`:272`、`:286`（都在 `_commit` 内，互斥分支）、`:304`（`_advance_fleet` 内，每车一次）。 |
| **F-4** | `_plan_and_commit` 在 `:469`（循环前一次）、`:545`（**每次迭代一次**）、`:602`（`run_resumed`）被调用；其内 `for v in vehicles` 对每车**至多一次** `_commit`（`:655`）。 |
| **F-5** | `_force_return_all`（`:658`）在终止分支被调用**一次**，内部对单车做 `_commit(...,0)` + 一次单车 `_advance_fleet`。 |
| **F-6** | 三个容差：跳过阈值 `1e-9`（`:219`）、倒退容忍 `1e-6`（`:217`）、事件/终止比较 `1e-6`（`:306/:490/:506/:514`）。 |
| **F-7** | `M1bQ2TrustFixCertifier` **零生产调用点**：全仓库只在测试里构造；正式入口 `d3_l1_l1_formal_run.py:469` 传 `certify_fn=None`。 |
| **F-8** | `T_end` 的全部实际取值是测试字面量 `24.0`，**没有**任何生产代码把它绑定到实例或环境。 |
| **F-9** | `hours_per_time_unit` 位于 `UnitScale` 内，而 `contract_hash = sha256(json(asdict(contract)))` ⇒ **单位系数已被合同身份覆盖**；`coldchain_state.time_to_hours` 用 `duration * hpu`。 |
| **F-10** | 数据生成器对每个客户**钳制** `tw_end[i] = min(horizon, center + width/2)`，且 `tw_end[0] = horizon`（`generate_coldchain_data.py:125,131`；`generate_cvrptw_data.py:117,129` 同型）。 |
| **F-11** | 已登记数据域（`data/m0_scale` 三个 split，**96** 个实例）实测：`tw_end.max(axis=1) == tw_end[:,0] = 24.0` **96/96**；最晚的**隐藏**客户时间窗 19.5755 < 24.0。 |

### 1.2 待检验假设（负控负责判真伪）

- **H-1（horizon 定义含未来信息，但数值上可能为空泄漏）**：`horizon` **定义**上依赖未揭示订单，
  故其**来源**是 `HIDDEN_DERIVED`；但在 F-10 的生成器不变量 + F-11 的实测下，它与
  **公开量** `tw_end[0]` 逐位相等，因而在该数据域内**携带的未来信息为空**。
  若某实例出现 `tw_end.max() > tw_end[0]`，则泄漏**非空**，该实例的时间上界**不得**用于在线认证。
- **H-2（`max_iter` 不等价于每车推进次数上界）**：真实上界是
  `n_advance ≤ 1 + 2·n_iter + 2`（前置 `_plan_and_commit` 1 次 + 每次迭代 `_advance_fleet` 与
  `_commit` 各 1 次 + 终止分支 2 次），故 `n_max = 2·max_iter + 3 = 20003`，**约为旧解释的 2 倍**。
  旧预算把 `n = 10000` 当作上界，属于**低估**（方向上不安全），但数值影响仍应远小于 `Δ*`。
- **H-3（迭代数另有时间上界）**：因每次迭代 `clock` 严格增加 `> 1e-6`，
  `n_iter ≤ 1 + floor(T_max / 1e-6)`；与 `max_iter` 取小。该式使**小 horizon** 下的 10000 上限变紧。
- **H-4（单位系数身份已绑定，但非 1.0 不可认证）**：`hpu` 在 `contract_hash` 内（F-9），
  故**身份**可绑；但 `hpu != 1.0` 时 `fl(duration)·hpu` 的转换误差**未被预算**，
  因此非 1.0 一律 `NOT_CERTIFIABLE`（与既有 L1-5 一致，本轮只复核不新主张）。
- **H-5（认证器目前对在线不可用）**：因 F-7/F-8，`T_end` 的值**只有测试字面量来源**。
  故"以 `T_max=24` 认证"这件事，**当前**不具备 online 资格；本轮的目标是给它一个**可机械验证的来源**。

---

## 2. 唯一主对照（the single primary control）

> **主对照 = 公开替换对（public-substitution pair）。**
> 对每个登记实例，计算两遍完全相同的证书链：
> **A 臂（现状）** 用 `T_max = tw_end.max()`（隐藏派生的原始值）与 `n = max_iter`（旧解释）；
> **B 臂（公开替换）** 用 `T_max = tw_end[0]`（公开量，且以 `tw_end.max() == tw_end[0]` 为机械见证）
> 与 `n = 2·max_iter + 3`（从调用图导出的上界）。
>
> **通过条件只有一条方向**：`B 臂不得比 A 臂更宽松`——
> `H_max(B) ≥ H_max(A)`、`q_lb(B) ≤ q_lb(A)`、`slack(B) ≥ slack(A)`、`n(B) ≥ n(A)`。

**为什么只有一个主对照**：本轮问题是**资格**问题，不是精度问题。任何"更准/更好"的对照都会把
"这个参数到底能不能用于认证"变成"这个参数算得好不好"，而后者不是本轮授权范围。
所有其它检查（三层可见性矩阵、调用图普查、负控）都是**辅助判据**，不得替代主对照。

---

## 3. 判据（criteria，全部机械化、可否决）

| # | 判据 | 通过条件 |
|---|---|---|
| **C1**（主） | 公开替换不更宽松 | 96/96 实例上 `H_max(B) ≥ H_max(A)`、`q_lb(B) ≤ q_lb(A)`、`n(B) ≥ n(A)`、`slack(B) ≥ slack(A)` |
| **C2** | 来源分类完整 | 四类时间参数（horizon / 推进次数 / 单位系数 / 容差）各自给出 `{字段, 来源, 可见性, 认证资格}`，**无空缺**；三层（模拟器内部 / strict-online 决策 / 认证器可证明）**逐层分开登记** |
| **C3** | horizon 见证可机械核验 | 公开路径必须有 `tw_end.max() == tw_end[0]` 的**逐实例见证**；见证缺失或不等 ⇒ 该实例判 `HIDDEN_DERIVED` 且**不可认证** |
| **C4** | 次数上界来自调用图 | `n_max` 由**静态调用图普查**导出（不是硬编码常数）；普查结果含三个调用点 `:272/:286/:304` 与两条路径（每迭代 2 次、前置 1 次、终止 2 次） |
| **C5** | 旧解释被否决 | 以 `n = max_iter` 构造的预算被**显式拒绝**（`INSUFFICIENT_COUNT_BOUND`），不得静默通过 |
| **C6** | 迭代上界双约束 | 报告 `n_iter ≤ min(max_iter, 1 + floor(T_max/1e-6))`，并对 `T_max ∈ {1e-3, 24, 1e6}` 三个取值给出各自绑定的那一侧 |
| **C7** | 单位系数 | `hpu == 1.0` ⇒ 可认证；`hpu != 1.0` ⇒ `NOT_CERTIFIABLE`，且 `hpu` 改变会改变 `contract_hash`（身份不可复用） |
| **C8** | 容差登记 | 三个容差（`1e-9` / `1e-6` 倒退 / `1e-6` 事件）逐个登记，并说明**只有 `1e-9` 进入预算**、两个 `1e-6` 影响**事件时刻而非时长和**（时长望远镜求和仍精确） |
| **C9** | 身份可机械重建 | 身份摘要 = 规范 JSON 的 sha256，内容含六个源码 blob 摘要 + `max_iter` + 调用图普查 + 三个容差 + horizon 来源分类；**任一字段变动即摘要变动**，与登记值不符 ⇒ `REFUSED` |
| **C10** | 负控全部命中 | §4 的 N-01…N-09 **九项**各自给出预期的 `REFUSED` / `NOT_CERTIFIABLE`，无一项静默通过 |
| **C11** | 不修改生产件 | 六个被审文件的 blob 摘要与 §0 **逐一不变**；不导入 `d3_geo_adapter.run_repair_transaction`；不实例化生产认证器 |
| **C12** | 结论分级诚实 | 每项结论标注 `PROVED_CONDITIONAL` / `DIAGNOSTIC` / `UNKNOWN`；**不得**把 `DIAGNOSTIC` 写成 `PROVED`；`UNKNOWN` 必须列出缺什么 |

---

## 4. 参数与次数（parameters and counts，预先固定，不得事后增删）

| 项 | 值 |
|---|---|
| 静态调用图普查对象 | `strict_online_env.py` 一个文件（AST 解析，不执行） |
| 数据实例数 | **96**（`data/m0_scale` 三个 split：cal 16 + dev_check 16 + train 64），**只读** |
| 合成实例 | **≤ 4 个**（3–5 节点，2 辆车，手工构造，用于 N-02/N-05 与次数诊断） |
| 时间上界取值 | **3 个**：`1e-3`、`24.0`、`1e6`（C6 的绑定侧对比） |
| `hpu` 取值 | **2 个**：`1.0`、`2.0` |
| 负控条数 | **9 条**（下） |
| 正控条数 | **2 条**：P-01 公开替换对（96 实例）、P-02 合成实例上次数实测 ≤ 导出上界 |
| 重复次数 | 纯函数，每项 **1 次**；合成实例的推进次数诊断最多 **2 次** |
| 随机种子 | 合成实例固定 `seed=20261017`；不引入随机性到判据 |

**九条独立负控（实现前固定）**

| 编号 | 注入 | 预期判定 |
|---|---|---|
| **N-01** | `horizon` 取真值的一半（不覆盖 episode） | `REFUSED: HORIZON_NOT_AN_UPPER_BOUND` |
| **N-02** | 合成实例令隐藏客户 `tw_end > tw_end[0]` | `NOT_CERTIFIABLE: HIDDEN_DERIVED_HORIZON` |
| **N-03** | `hpu = 2.0` | `NOT_CERTIFIABLE: TIME_UNIT_CONVERSION_NOT_BUDGETED` |
| **N-04** | `resumed_between_doors = True` | `REFUSED: SNAPSHOT_RESTORE_INSIDE_INTERVAL` |
| **N-05** | 实测/注入推进次数 `> 导出上界` | `NOT_CERTIFIABLE: COUNT_BOUND_VIOLATED` |
| **N-06** | 篡改身份（改 `max_iter` 或改一个源码摘要） | `REFUSED: IDENTITY_MISMATCH` |
| **N-07** | 用旧解释 `n = max_iter` 构造预算 | `REFUSED: INSUFFICIENT_COUNT_BOUND` |
| **N-08** | `horizon ∈ {nan, -1.0, 0.0}` | `REFUSED: HORIZON_INVALID` |
| **N-09** | 把"非 1.0 单位"的合同与"登记身份"配对 | `REFUSED: CONTRACT_IDENTITY_MISMATCH` |

---

## 5. 预期方向（expected direction）

| 量 | 预期 |
|---|---|
| `T_max`（公开路径） | 与原型**逐位相同**（96/96 为 `24.0`），因 F-10/F-11 |
| `H_max` | B 臂 = A 臂（`24.000001`）；**B 臂不得更小** |
| `q_lb` | B 臂 = A 臂（`8.54807998017`）；**B 臂不得更高** |
| `n_max` | 由 10000 变 **20003**（`2·max_iter + 3`），即旧预算**低估 2 倍** |
| 总 slack | 由 ~1.0e-9 h 变 ~1.0e-9 h（`1e-9 + 20003·ulp/2`），**量级不变**，仍远小于 `Δ* ≈ 0.04 h` |
| 结论 | **旧数值结论存活**（L1 预算远小于阈值），但**推导不成立**：旧 `n` 是**低估**，必须换成本轮导出的上界 |
| horizon 资格 | 该数据域内 `PUBLIC_BY_WITNESS`（`PROVED_CONDITIONAL`：依赖生成器不变量 + 逐实例见证）；**不是**无条件 `PROVED` |
| 认证器在线资格 | **仍为 `UNKNOWN`**：`T_end` 无生产调用点（F-7/F-8）这一缺口**本轮不修** |

---

## 6. 失败条件（failure conditions，任一成立即判本轮 FAIL）

| 编号 | 条件 |
|---|---|
| **F-A** | 主对照 C1 出现 B 臂更宽松（`H_max` 变小 / `q_lb` 变高 / `n` 变小 / `slack` 变小） |
| **F-B** | 任一条负控**未**给出预期判定（静默通过 = 最严重失败） |
| **F-C** | `n_max` 仍以硬编码常数出现（不是调用图导出），或调用图普查漏掉 `:272/:286/:304` 任一处 |
| **F-D** | 身份摘要不能机械重建，或改动 `max_iter` 后摘要**不变** |
| **F-E** | 六个被审文件任一摘要发生变化（说明"只读"被破坏） |
| **F-F** | 出现 `PROVED` 标注但论证依赖实测数据而非不变量 |
| **F-G** | 为了让某项通过而降低判据、放宽容差或删除负控 |
| **F-H** | 越界启动 G-1/G-3/C2/G5/G6/C3、训练、GPU、正式 rollout、最终测试种子 |
| **F-I** | 自动 commit/push |

**任一失败：记录 `BLOCKED_WITH_REASON`，不降低判据再试。**

---

## 7. 资源门（resource gate）

| 项 | 上限 |
|---|---|
| 运行时间 | 测试套件 **< 120 s**（纯 CPU） |
| 内存 | < 400 MB |
| GPU / 训练 | **禁止** |
| 数据读取 | **只读** `data/m0_scale` 三个 npz；不得写、不得改 |
| 合成实例 | **≤ 4 个**，3–5 节点；不得扩成批量实验 |
| 正式 rollout | **禁止**（合成实例的推进次数诊断不算 rollout：无模型、无 replanner、无评分） |
| 外部网络 | 禁止 |
| 生产路径 | 不导入安装入口、不改认证器、不实例化带 `verified_t_min` 的调用 |

---

## 8. 交付物与"不做什么"

**交付（对应用户六项要求）**
1. 本预声明 + `G2_时间参数身份与strict_on_line来源资格审查_2026-10-17.md`（独立审计报告）；
2. **字段—来源—可见性—认证资格矩阵**（四类时间参数 × 三层可见性）；
3. **每辆车温度推进次数上界的推导**（静态调用图 + 双约束迭代上界）或未建立理由；
4. **共享时间身份规范草案** + 九条负控实现与结果；
5. **需要用户签署的事项**（列表，逐条说明为什么必须由人决定）；
6. **G-2 是否具备下一阶段实施资格**的结论。

**明确不做**
- 不实现 G-1（精确时间身份）与 G-3（候选计划身份）；
- 不改 `m1b_q2_certifier.py` / `m1b_quality_certificate.py` / `strict_online_env.py`；
- 不把 `NOT_CERTIFIABLE` 写成 FAIL 或反过来；不追认任何既有正式资格；
- 不自动 commit/push。

---

## 9. 与状态的关系

本预声明**不改变任何科研状态**：
`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`（永久，不追认）、
`THERMAL_CONTINUOUS_NOT_ESTABLISHED`、`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、
L1 `BLOCKED`、`L2/A3_NOT_ESTABLISHED`、`stable_online_gain=False`、`HISTORICAL_BACKBONE_UNVERIFIED`、
C0 `AWAITING_USER_BUDGET_RULING`、`t_min` 恒 0（无可信见证机制）、最终测试种子 `SEALED`
——全部保持不变。
