# D3-G0 / F Execution Readiness：V1.19 勘误与不变量级对照

- 基准 HEAD：`0ec1cce`（本轮提交后见文末）
- 本轮新证据：`scripts/tests/test_d3_g0_invariants_v119.py` →
  `v119_evidence/D3_V119_G0_INVARIANT_DIFFERENTIAL.json`（**8 行 / 8 PASS**，**可执行反例**，非字符串检查）
- 未执行：真实 F 新批次、F−D-Geo 比较、E2 训练、L2/L3、40 日批次、最终测试；未重采 64 状态、未补跑 V1.11 的 42 条
- 科学增益：**`NOT_ESTABLISHED`**

## 1. 勘误（保留 V1.17 原件，不改写）

| V1.17 的错误 | 勘误 |
|---|---|
| "G0 残余风险 24 条" | **撤回**：那是 `D3_G0_RESIDUAL_RISK_AUDIT.md` 中前 24 条以 `#`/`-`/`*` 开头且长度 > 4 的行，**不是** 24 条编号独立、判据完整的未闭合风险。真实条数须按不变量逐条核对（见 §3） |
| 统计类别"1 CLOSED / 1 BLOCKED / 6 PENDING" | 更正为 **`CLOSED_FOR_ENGINEERING` 1 / `BLOCKED` 1 / `RECORDED_BLOCKED_ROUND` 1 / `PENDING`·`DRAFT_PENDING_USER` 6 = 9**（A2 漏列为一类） |
| 把旧 OPEN 项当作"当前源码未修复" | **撤回**：V1.17 只做源码字符串检索 + 归档字段搜索 + 旧文档标题读取，**不等于**对当前生产实现的差分测试。本轮以**可执行反例**重做（§2） |

## 2. 可执行 G0 差分证据（8/8 PASS，全部为真实调用）

| 不变量 | 负例 | 实测 |
|---|---|---|
| **I0** 接口契约存在 | — | `INSTALL_REQUIRES_CERTIFICATION=true`；`FAILURE_REASONS` 含 `score_shape/score_len/no_certifier`；`REFUSAL_REASONS` 含 `already_finished/certification_missing/…` |
| **G0-01** 无认证器 | 旧行为：静默安装**未认证**计划 | **拒绝于采样前**：`n_installs=0`、`n_pre_sampling_refusals=1`、返回 `False` |
| **G0-02** 分数向量契约 | 2-D 向量；过短；过长 | `score_shape`；`score_len`；`score_len` —— **三者 `n_steps=0`**（不再把短向量当前缀静默评分；`score_contract.violation` 被记录） |
| **G0-03** 认证/安装身份 | 认证哈希与待安装计划不一致 | `certified_plan_identity_mismatch`，台账同时记录 `installed_plan_hash`/`certified_plan_hash`/`certifier`，`n_installs=0` |
| **G0-04** 一账本一事务 | 同一台账二次 `finish()` | 首次**恰好一次**安装成功；二次 `already_finished` 拒绝，**安装数仍为 1** |
| **G0-05** 三种结局互不混淆 | 无 `rollback_fn`；有 `rollback_fn`；采样前拒绝 | `n_rollback_unverified=1` / `n_state_restorations=1` / `n_pre_sampling_refusals=1`，三者**分列** |
| **G0-06a** 品质能力缺口 | 以组合认证器宣称"完整认证" | `d3_composed_certifier` **自身声明**缺口：*"no CONSERVATIVE decision-time quality/salability certification exists"*，且模块带 `CAPABILITY…` 清单 → **不得**据此出"已完整认证"回执 |
| **G0-06b** 预算 | 用任意数字充当正式 C0 预算 | **64 个 dev_check 状态无已注册预算** → `BLOCKED / AWAITING_USER_BUDGET_RULING`，代理不得填数 |

## 3. 不变量级对照：`历史缺陷 → 现有修复 → 现有验收 → 剩余缺口`

| 不变量 | 历史缺陷 | 现有修复 | 现有验收 | 剩余缺口 |
|---|---|---|---|---|
| 认证必需 | 无 `certify_fn` 仍安装（G0-01） | `d3_geo_adapter.py:721-725` 采样前拒绝 | **本轮 G0-01 负例** + `test_d3_g0_{red,green}_phase.py` | 无（接口层闭合） |
| 分数契约 | 短向量静默按前缀评分；2-D 被 `np.argmax` 展平 | `:476-484` `score_shape`/`score_len` | **本轮 G0-02** + B3AT1 | 无（接口层闭合） |
| 认证↔安装身份 | 只凭布尔 `certified` | `:668-675` 五项校验含 `certified_plan_identity_mismatch` | **本轮 G0-03** | 无（接口层闭合） |
| 事务幂等 | 可重复 `finish` | `:632-639` `already_finished` | **本轮 G0-04** | 无 |
| 回滚语义 | 回滚"成功"无法区分是否真恢复 | `:640-664` 三态分列 | **本轮 G0-05** | 无 |
| 写者排他 | AT2A 记录过绕过路径 | `d3_write_exclusion.py`（B3-AT2B-2a：作用域 + policy 随作用域传递） | `test_d3_b3at2b2a_write_exclusion.py`（14 行） | **真实状态上的写入唯一性**未建立 |
| 组合认证 | 缺少状态/合同绑定的组合认证 | `d3_composed_certifier.py` 有组合结构，可调 `certify_plan` 查能源预算 | B2/B3QC/B3QP 系列 | **品质/可售能力缺口**（模块自述）；**真实状态+合同绑定**未建立 |
| 安装回执 | 无 `P_*` 持久回执 | 台账事件（`installs`/`events`）**非**持久回执 | 本轮仅见台账内事件 | **`P_before/P_proposed/P_certified/P_after_install` 持久化 + 独立复核未实现** |
| 能源预算 | `B=300` 仅为历史 CAL 夹具 | — | — | **64 状态无已注册预算 → `BLOCKED`** |
| D8 安装回执 | `CAPABILITY_GAP_BLOCKING` | 部分组件 | — | **未闭合** |

**结论（与你裁定一致）**：**G0-Safety 一部分接口安全修复已实现且已有历史验收**；
**仍未建立**的是——真实状态与合同上的组合认证、写入唯一性、持久安装回执、D8 所需完整硬约束、品质认证与 C0 预算裁定。
**不得**从头重做 G0 接口层（会污染既有安全语义）。

## 4. C0 预算调查（任务 C）与责任划分

- **未找到**适用于这 64 个 dev_check 状态的**已注册 C0 预算**；`test_d3_b2_certification.py` 中的 `B_BUDGET=300` 属**历史 CAL 认证夹具**，
  `SAA` 系列测试里的 `budget=1e9` 属**隔离夹具**；两者**均不得**升级为当前主档协议。
  （说明：本轮字符串检索未在该文件中命中 `B_BUDGET = 300` 的精确写法，故**不声称**该夹具必然存在或不含该常量——此为检索局限。）
- 预算候选方案（**仅提案，`AWAITING_USER_BUDGET_RULING`**）：须写明**来源、单位、适用数据集与物理合同、可证伪假设、验证方法、适用条件**；
  在裁定前，认证器只允许在**明确标记 synthetic-only** 的隔离测试中验证**拒绝机制**，**不得**作为 dev_check 正式能源认证，**不得**安装真实状态。
- 你给出的责任划分已采纳：调查/复用/反例/提案由我完成；**预算最终冻结与正式执行由你授权**。

## 5. 本轮完成 / 未完成 / 阻塞

**完成**：§1 勘误；§2 可执行 G0 差分（8/8）；§3 不变量级对照；§4 预算调查与责任划分；V1.17 原件保留。
**未完成（诚实列出，不伪装闭环）**：
1. B 任务中剩余的**真实状态组合认证**、**写入唯一性（真实状态）**、**持久安装回执 + 独立复核**、**TOCTOU/进程中断/部分提交正反例** —— 均需在隔离环境与真实 `VehicleState` 上实现，本轮未完成；
2. **V1.18 可签署执行预登记**：只有 V1.17 §6 的要点草案，**未**形成含冻结 sha/预算/调用次数的正式版；
3. **E 任务四份科学预登记**（L1/L2/E2/L3）：仍为 `DRAFT_PENDING_USER`，未补全既有冻结参数；
4. **权威状态文件**（`项目当前状态.md`、`执行进度表.md`）顶部仍停在 V1.10–V1.14，**未**追加 V1.16/V1.17/V1.19（属用户拥有文档，未擅自改写）。

**阻塞（需你裁定）**：C0 正式预算（`AWAITING_USER_BUDGET_RULING`）；D8 安装能力解阻；E2 预登记前置条件。

## 6. Claim limits

工程资格进展**不等于**主模型增益。未改上游 `MASKCO_code/`、冻结树与冻结数据；历史 FAIL/BLOCKED 原件全部保留。
`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、`stable_online_gain=False`、
I3 正式负结果（ΔJ = −144.680，95% CI [−191.109, −100.125]）与旧 D2 `not_qualified` 全部不变。
