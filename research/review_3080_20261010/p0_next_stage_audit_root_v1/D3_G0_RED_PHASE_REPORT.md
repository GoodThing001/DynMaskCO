# D3-G0 失败反例先行（Red Phase）报告

**日期**：2026-10-14 · **基准 commit**：`dbc5971`（本地 `HEAD` == `origin/main`，起点干净）
**性质**：只**新增测试**；**不改生产源码、不改历史测试、不动冻结树/实验数据/裁决文件**
**被测生产文件**：`C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_geo_adapter.py`
（498 行，sha256 `f7480925f339555982e12bed58600e5b8dd96825f077a88ddf9857b80af676bb`，**本轮前后未变**）
**新增文件**（4 个，全部为本轮产物）：

| 文件 | 字节 | sha256 |
|---|---|---|
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_d3_g0_red_phase.py` | 28,070 | `779cafaf7876dc13ca11ef2bc8e83a11f595b9d15ef931c7d1636ca5816e01d9` |
| `research/review_3080_20261010/p0_next_stage_audit_root_v1/D3_G0_RED_PHASE_MATRIX.json` | 8,990 | `327a9fb35fd683ecaac282b0b4e6c0962e2ecdfca8dfc55489a6c183a3758107` |
| `research/review_3080_20261010/p0_next_stage_audit_root_v1/raw/G0_red_phase_stdout.txt` | 2,047 | `225adcae9e74b9f11622cdc27af444f75c4bb52d2cb4ff5498ba4c6825e1c9ee` |
| `research/review_3080_20261010/p0_next_stage_audit_root_v1/raw/G0_red_phase_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`（空） |

---

## 0. 结论摘要

**12 个反例全部有确定结论**，停止条件达成，**不自动进入修复**：

| 判定 | 数量 | 条目 |
|---|---|---|
| `COUNTEREXAMPLE_REPRODUCED` | **9** | G0-01(A-2)、G0-02(A-1a)、G0-03(A-1c)、G0-04(A-1b)、G0-05(A-5b)、G0-06(A-4b)、G0-07(A-1d)、G0-09(A-5d)、G0-10(A-5c) |
| `NOT_VERIFIABLE` | **1** | G0-08(A-3b) —— 保持 **OPEN**，**未被证伪** |
| `CONTROL_PASSES` | **2** | G0-C1、G0-C2 |
| `TEST_INVALID` / `NOT_REPRODUCED` / `FAIL_LOUD_ACCEPTABLE` | 0 | — |

`harness_valid: true`（无 `TEST_INVALID` 且两个阳性对照均通过）。退出码 0，stderr 空。

**四项相对上一轮只读审计（`D3_G0_RESIDUAL_RISK_AUDIT.md`）的**升降级**（均由反例推动，非人工改判）：

| 审计项 | 审计分类 | Red Phase 结论 | 变化原因 |
|---|---|---|---|
| A-1b | CONFIRMED（fail-loud，可接受） | **COUNTEREXAMPLE_REPRODUCED** | 只有展平 argmax **越界**时才 fail-loud；**落在合法区间内则静默成功**，多余评分被丢弃。故“可靠地 fail-loud”不成立 |
| A-1d | **OPEN** | **COUNTEREXAMPLE_REPRODUCED** | GEO 路径的 `deltas` 短于 `legal` 时静默忽略真实最优候选，`failure_reason=None` |
| A-4b | **NOT_VERIFIABLE** | **COUNTEREXAMPLE_REPRODUCED** | 改为可执行的“身份字段存在性 + 引用别名”双重证据：身份字段为空 **且** 外部改写会改变账本内容 |
| A-5d | **OPEN** | **COUNTEREXAMPLE_REPRODUCED** | `rollback_fn=None` 时 `n_rollbacks` 照加、且无任何 unverified 标记 |

**最重要的一条限定（必须与生产资格分开读）**：`d3_geo_adapter.py` 在**当前仓库内没有任何生产调用方** —— 全仓仅测试文件 import 它（另有 `build_d3_identity.py` 只做源码身份登记）。
因此这 9 条反例证明的是「**适配器不强制自己的注入式接口契约**」，**不是**「某条正在运行的生产路径正在错决策」。装配层尚未编写，而**恰恰是这个尚未编写的装配层负责选择这些供给方** —— 这正是它们是 G0 阻断项的原因，但不能把它们表述成“线上错误正在发生”。

---

## 1. 做法：可证伪优先

### 1.1 每条反例先写「应当成立的不变量」，再用**注入的假供给方**触发

所有输入都是合成数据 + 注入的钩子（`score_fn` / `extract_fn` / `legal_fn` / `apply_fn` / `certify_fn`）：
**不加载 MaskCO 检查点、不使用科学数据、不训练**。矩阵 JSON 中 `production_source_modified=false`、`scientific_data_used=false`、`checkpoint_loaded=false`。

### 1.2 判定词汇表（防止无关异常冒充反例）

| 判定 | 含义 |
|---|---|
| `COUNTEREXAMPLE_REPRODUCED` | **指定的不变量**被违反，且违反方式与预测一致 |
| `FAIL_LOUD_ACCEPTABLE` | 源码抛出**预期的**错误（可接受，不算缺陷） |
| `NOT_REPRODUCED` | 不变量成立（该项可关闭） |
| `NOT_VERIFIABLE` | 用注入假件**无法观测**该不变量（保持 OPEN，不得据此判定通过或失败） |
| `TEST_INVALID` | 发生了**非预期/无关**异常（是测试问题，**不是**生产发现） |

`run_guarded()` 包裹每个用例：任何逃逸异常一律记为 `TEST_INVALID` 并带 traceback 尾部，**绝不**被算作反例。

### 1.3 两个**阳性对照**（把测试有效性与生产 G0 资格分开）

- **G0-C1**（对应 A-2）：一个**合规的** stub 账本，在 `certified=False` 时必须拒绝安装。观测 `stub_installs=0, stub_refused=1` → `CONTROL_PASSES`。
  含义：harness 有能力区分合规实现，故 G0-01 的失败是**生产发现**，不是测试假象。
- **G0-C2**（对应 A-1a / A-1c）：`len(scores)==n_legal and ndim==1` 这一简单断言可机械检出短向量与 2-D 输入（`good_accepted=true, short_rejected=true, 2d_rejected=true`）→ `CONTROL_PASSES`。
  含义：A-1a/A-1c 只需一条断言即可防住，缺口是真实的。

### 1.4 本轮修掉的两个**自身**harness 缺陷（已留痕，未美化）

1. 首轮 `G0-02` 用 `mask_set=[4]` 导致 `legal` 长度为 1，使 A-1a 静默退化成 A-1b 的情形 → 改为 `mask_set=[4,5,6]`。
2. 首轮用 `dir(ledger)` 找身份字段，命中 Python 自带的 `__hash__`，报出假字段 → 改为 `vars(ledger)` 并排除 `__` 前缀。
3. 首轮 `G0-03` 归类为 `TEST_INVALID`（`ValueError: The truth value of an array ... is ambiguous`）。复查后确认**这不是无关异常**：2-D 评分既不被处理也不被拒绝，只是崩溃点发生在适配器**自带的证据记录器**（`d3_geo_adapter.py:402-403` 的 `sv == scores[idx]` 在数组上做真值判断）而非我原先预测的 `legal[idx]`。故改为**分层测量**：主证据 = 选择原语的静默错位（可在原语层直接证明），次表现 = 完整适配器在该输入下的实际行为。
4. 首轮曾把矩阵 JSON 写到错误的仓库深度（`C-VRP_.../research/...` 的游离路径）→ 已改为向上寻根到 `MASKCO_code/`；**该游离文件已删除**，未进入任何提交。

---

## 2. 结果总表

| 用例 | 审计项 | 断言的不变量 | 判定 | 关键观测 |
|---|---|---|---|---|
| G0-01 | A-2 | 安装要求 `n_candidate_plans_certified ≥ 1` | **REPRODUCED** | `n_installs=1`，`n_candidate_plans_certified=0`，事务返回 `true` |
| G0-02 | A-1a | `len(scores)==len(legal)` 或 fail-closed | **REPRODUCED** | 评分长度 1 / `n_legal=3`，`failure_reason=null`，选中列表第 0 位客户 4 |
| G0-03 | A-1c | `ndim(scores)==1` 或 fail-closed | **REPRODUCED** | 原语返回**展平** argmax `2`，而矩阵语义指向第 `1` 行；完整适配器随后在自身记录器抛 `ValueError` |
| G0-04 | A-1b | 过长评分 fail-closed **或**绝不被静默部分使用 | **REPRODUCED** | 2 个候选 / 4 个评分：argmax 落在合法区间内 → **静默成功**；越界 → `IndexError` |
| G0-05 | A-5b | 每笔事务 `n_installs ≤ 1` | **REPRODUCED** | `finish` 调两次 → `n_installs=2`，`installed_len=2` |
| G0-06 | A-4b | 记录认证/安装计划身份，且账本条目不可变 | **REPRODUCED** | `identity_fields_recorded=[]`；外部改写计划后账本条目由 `{1:[4]}` 变为 `{1:[999]}` |
| G0-07 | A-1d | `len(deltas)==len(legal)` 或 fail-closed | **REPRODUCED** | 3 个候选中真实最优是客户 6，`deltas` 被截断为 2 → 静默选中客户 4，`failure_reason=null` |
| G0-08 | A-3b | 不安装会重规划已承诺客户的计划 | **NOT_VERIFIABLE** | 注入的 `committed_next` 是伪造属性；真实承诺护栏在 `VehicleState` + mask 池 → **A-3b 保持 OPEN** |
| G0-09 | A-5d | `rollback_fn=None` 的回退被标记为未验证 | **REPRODUCED** | `n_rollbacks=1`，事件 `{'action':'rollback','reason':'no_legal'}`，`rollback_unverified_flagged=false` |
| G0-10 | A-5c | 成功安装产出持久化 `P_after_install` 回执 | **REPRODUCED** | `persistence_api_surface=[]`，事件仅在内存 |
| G0-C1 | 对照(A-2) | 合规实现不得安装未认证计划 | **CONTROL_PASSES** | 合规 stub `installs=0 / refused=1` |
| G0-C2 | 对照(A-1a/A-1c) | 长度/shape 契约可机械检出 | **CONTROL_PASSES** | 合规被接受，短向量与 2-D 均被拒 |

---

## 3. 逐项证据

### G0-01 · A-2（认证可被整体跳过）—— 最严重
`d3_geo_adapter.py:412 if certify_fn is not None:`：`certify_fn` 默认 `None`，为 `None` 时**整段认证被跳过**，函数在 `:427` 正常返回 `failure_reason=None`，事务层 `:457` 随即安装。
反例：`run_repair_transaction(..., certify_fn=None)` → `n_installs=1` 且 `n_candidate_plans_certified=0`，返回 `True`。
即：**“已认证”与“已安装”之间不存在强制的因果链**，只由调用方是否传参决定。

### G0-02 · A-1a（评分短于 `legal` ⇒ 静默取前缀）
`:366 scores = np.asarray(score_fn(state))`；`:368` **只查有限性**，不查长度；`:373` `np.argmax` 在长度为 1 的向量上恒返回 0 → `:385 legal[0]`。
反例：`score_fn` 返回 `[0.0]` 而 `legal` 有 3 项 → 选中 `legal[0]`（客户 4），`failure_reason=null`，无任何告警。评分供给方是注入接口，适配器完全不校验其长度。

### G0-03 · A-1c（2-D 评分：先静默错位，后自身崩溃）
**主证据（原语层，可直接证明）**：`select_historical_argmax(np.array([[0,0],[9,0],[1,0]]))` 返回 `2`，而矩阵语义指向第 `1` 行 —— 展平索引 `row*K+col` 与候选索引语义**不是同一件事**。`:366` 无 `ndim` 检查，`:373 np.argmax` 直接展平。
**次表现（完整适配器）**：同一 2-D 输入下，`legal[2]` 恰好存在（3 个候选），于是流程继续，直到适配器**自带的证据记录器** `:402-403`（`[i for i, sv in enumerate(scores) if sv == scores[idx]]`）在数组真值判断上抛 `ValueError`。
**判定依据**：不变量“`ndim==1` 或 fail-closed”被违反 —— 2-D 输入既未被处理也未被拒绝；两次运行（2 候选 / 3 候选）分别得到「崩溃」与「静默错位」，说明失败形态取决于候选数量，而非契约。
**注**：此崩溃点与我事前的预测（`legal[idx]` 处 `IndexError`）不同，已如实记录差值而非改写预期。

### G0-04 · A-1b（过长评分不是可靠 fail-loud —— 对审计的**升级**）
反例：`legal` 长 2，评分长 4。
- argmax = 0（`[5,4,3,2]`）→ **无异常、静默成功**，`scores[2:4]` 被丢弃；
- argmax = 1（`[1,5,4,3]`）→ 同样静默成功；
- argmax = 2（`[1,2,9,3]`）→ `IndexError`。

即 fail-loud 与否**取决于最大值恰好落在哪**，而这是数据依存的。上一轮审计「CONFIRMED（fail-loud，可接受）」因此不成立，本轮升级为反例。

### G0-05 · A-5b（`finish` 不幂等）
`:449-461` 无任何已安装/已结束状态。反例：同一账本连续两次 `finish(failure_reason=None, ...)` → 两次都返回 `True`，`n_installs=2`，`installed_plans` 长度 2。
后果：重复调用（重试、事件重放、并发回调）会重复安装并重复计数。

### G0-06 · A-4b（无身份记录 + 按引用保存）
`:459 self.installed_plans.append(plan)`：既没有计划哈希，也没有值拷贝。
**证据类型明示**：这是**接口面存在性 + 运行时别名证明**，**不是**哈希比对（因为没有哈希可比）。
反例：安装成功后 `diag` 与 `vars(ledger)` 中 `hash`/`identity` 相关字段为空列表；随后把账本持有的计划对象做外部替换，账本内容由 `{1:[4],2:[],3:[5]}` 变为 `{1:[999],2:[],3:[5]}` —— **账本会“改口”说它安装过别的东西**。
上限：这只能证明身份无法核对，不能证明历史上某个被安装计划真的被改过。

### G0-07 · A-1d（GEO 距离向量索引失配 ⇒ 静默错过真实最优）
`:238 unified_minimum_tie_set` 对传入的 `deltas` 取 `d_min = min(d)` 并只在**该向量长度内**建 tie 集；`:384 select_geo(state[GEO_SCORE_KEY], ...)` 从不校验向量长度与 `legal` 一致。
反例：注入的 `extract_fn` 把 `GEO_SCORE_KEY` 截断为前 2 项（3 个候选，真实最优为增量距离 1.0 的客户 6）→ 静默选中客户 4，`failure_reason=null`。
**暴露前提（必须写清）**：截断来自一个非合规的注入 `extract_fn`；仓库自带的 `make_geo_extract_fn` 产出的向量是对齐的。本条证明的是**适配器不校验注入状态供给方的索引契约**，即防御性契约缺口；在装配层选定供给方之前，它是真实的 G0 风险，但当前不存在可触发的生产路径。

### G0-08 · A-3b（**无法验证，保持 OPEN**）
反例尝试：在 `VehiclePlan` 上人工挂 `committed_next=7`，再让修复事务把客户 7 插回后缀 → 观测到“已承诺客户出现在安装后的后缀中”，且事务返回 `True`。
**如实重分类**：`VehiclePlan` **没有** `committed_next` 字段；真实承诺位于 `VehicleState` 并由 mask 池排除，**在本适配器的计划级 API 之外**。故该观测**不能**证伪真实流水线，记为 `NOT_VERIFIABLE`。
**A-3b 维持 OPEN**；要判定它需要状态级 harness（构造真实 `VehicleState`，经 `mask_candidate_pool` 再进适配器）。**本轮不做**（超出「只加测试」的授权范围与合成假件能力）。

### G0-09 · A-5d（无回退实现的回退仍被计数）
`:452-455`：`failure_reason is not None` 时 `n_rollbacks += 1`，仅当 `rollback_fn is not None` 才真正调用；`rollback_fn=None` 时**什么都不恢复，也不打标记**。
反例：`InstallLedger(install_fn=..., rollback_fn=None)` + `finish(failure_reason='no_legal', ...)` → `n_rollbacks=1`，事件 `{'action':'rollback','reason':'no_legal'}`，`rollback_unverified_flagged=false`。
后果：该计数器会被读成“安全地回退了 1 次”，实际是“声称回退但没有任何状态被恢复”。

### G0-10 · A-5c（无持久化回执）
`:441-461` 的接口面里不存在任何 `receipt`/`persist`/`write`/`path`/`save`/`dump` 方法或参数（`persistence_api_surface=[]`），事件只存在于 `self.events`。
反例：成功安装后无法得到可离开进程的 `P_after_install`。**证据类型**：接口面证明 —— 连可供测试的回执路径参数都不存在。

### 未重复计数的已闭合项（遵守「已 CLOSED 项不得重算」）
R1 `pool_fn` 注入、R2 预算单位分离、R3 默认容差/主判 tie-break、R4 失败回退（fail-closed 返回原始计划）、N3 GEO 非确定性拒绝、认证后超时、安装事务 API、A-1e（`IndexError` fail-loud，MITIGATED）、A-3a（`apply_action` 新建对象，MITIGATED）、A-4a（认证对象即被安装对象，MITIGATED）、A-5a（异常 ⇒ 0 安装，MITIGATED）、A-6（`:363-364` 两行不可达代码，无行为影响）—— **本轮均未重测，也不计入统计**。

---

## 4. 注册反例登记表 —— 每项均有确定结论

| 注册项 | 期望失败触发于 | 是否由指定不变量触发 | 结论 |
|---|---|---|---|
| A-2 | 未认证即安装 | 是（`certify_fn=None` 直通安装） | 反例成立 |
| A-1a | 评分短于 `legal` | 是（无长度校验，静默取前缀） | 反例成立 |
| A-1b | 评分长于 `legal` | 是（in-range 分支静默） | 反例成立（升级） |
| A-1c | 2-D 评分 | 是（无 ndim 校验；原语层错位可证） | 反例成立 |
| A-1d | GEO 距离向量失配 | 是（无索引对齐校验） | 反例成立（升级） |
| A-3b | 安装重规划已承诺客户 | **否**（伪造属性，未触达真实机制） | **不可验证 / 保持 OPEN** |
| A-4b | 身份缺失 + 引用别名 | 是（接口面无身份字段；运行时别名已证） | 反例成立（升级） |
| A-5b | 重复安装 | 是（无幂等守卫） | 反例成立 |
| A-5c | 无持久回执 | 是（接口面为空） | 反例成立 |
| A-5d | 无实现的回退被计数 | 是（无 unverified 标记） | 反例成立 |

**停止条件已达成**：上面每一行都有确定结论，没有悬置项。**未执行任何修复**，也未进入下一阶段。

---

## 5. 测试有效性 vs 生产 G0 资格（必须分开陈述）

**测试有效性（本轮的自我声明，可以给结论）**
- 12 个用例全部执行完毕，退出码 0，stderr 为空（0 字节）。
- 两个阳性对照均通过 → harness 能区分合规实现与不合规实现。
- 0 个 `TEST_INVALID` → 无逃逸异常被误当反例；`harness_valid: true`。
- 判定词汇表把「预期 fail-loud」与「反例」分开，`FAIL_LOUD_ACCEPTABLE` 与 `COUNTEREXAMPLE_REPRODUCED` 不是同一个桶。
- 未为迁就现状修改任何测试期望（`G0-03` 的预测与实际不符时，改的是**测量分层**与**证据描述**，不是判定阈值）。

**生产 G0 资格（本轮不给通过结论）**
- 本轮**没有**证明任何生产路径正在错决策：`d3_geo_adapter.py` 当前**无生产调用方**，装配层尚未编写。
- 9 条反例的暴露路径全部经过**注入式接口**（`score_fn` / `extract_fn` / `certify_fn` / `rollback_fn` / 账本调用纪律）。它们证明：**适配器不强制自己的接口契约，也不强制安装事务的安全不变量**。
- 因此本轮的准确表述是：**“若装配层按现状接入这些钩子，则存在 9 条已复现的 G0 缺口”**；而不是“G0 已被违反”或“G0 已通过”。
- 修复授权独立于本轮：**本轮不修**。

---

## 6. 未做 / 未授权（明确边界）

- 未修改生产源码（`d3_geo_adapter.py` 哈希前后一致），未修改任何历史测试、冻结树、实验数据、裁决文件。
- 未训练、未采集、未做 F−D 对比、未重采 32 状态、未跑 G1 smoke / G2 batch、未触碰最终测试种子。
- 未加载任何检查点，未使用科学数据；全部为合成数据 + 注入假件。
- **未修复上述任何缺口**，即使 9 条全部复现 —— 修复需要单独授权。
- 未把 `NOT_VERIFIABLE` 的 A-3b 写成通过或失败。

---

## 7. 复现

```powershell
# 必须在带 jax 的解释器下运行（mpre_policy 链路 import jax）
cd D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
& D:\AA_Py\A_Anaconda\python.exe -B scripts\tests\test_d3_g0_red_phase.py   # exit 0
```
脚本向 stdout 打印 12 行判定表与 `verdict_counts` / `harness_valid`，并写出
`research/review_3080_20261010/p0_next_stage_audit_root_v1/D3_G0_RED_PHASE_MATRIX.json`（仓库根由向上寻根到 `MASKCO_code/` 解析）。
原始输出留档：`raw/G0_red_phase_stdout.txt`、`raw/G0_red_phase_stderr.txt`（空）。
