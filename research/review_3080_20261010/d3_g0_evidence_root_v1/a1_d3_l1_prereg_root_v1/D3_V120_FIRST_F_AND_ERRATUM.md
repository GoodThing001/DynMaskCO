# D3 V1.20：V1.19 勘误 + 组合认证器真实执行 + 预算来源（修正匹配）+ F-first 分叉

- 起点 SHA `40150f6`；本轮新证据 `scripts/tests/d3_l1_v120_forward_and_l1_readiness.py` →
  `v120_evidence/D3_V120_READINESS.json`（**3 行 / 2 PASS / 1 FAIL**，失败原因见 §3）
- 未执行：真实 F 新批次、F−D-Geo 比较、E2 训练、L2/L3、40 日批次、最终测试；未重采 64 状态
- 科学增益：**`NOT_ESTABLISHED`**

## 1. V1.19 勘误（按你的两点 + 检索错误）

| 你的指正 | 承认与更正 |
|---|---|
| G0-06a 只查能力声明与源码文本，**未执行**组合认证器 | **成立**。V1.19 的 G0-06a 属"能力声明检查"，不能称为真实状态下的认证/拒绝路径通过 |
| G0-06b 是"预填 `None` 再检查仍为 `None`"，不是全仓预算审计 | **成立**。改标为 **`ARCHIVED_BUDGET_GAP_CONFIRMED / CURRENT_PROVENANCE_NOT_EXHAUSTIVELY_SEARCHED`** |
| `test_d3_b2_certification.py` 实含 `CAPACITY, NUM_VEHICLES, B_BUDGET = 50.0, 25, 300.0` | **成立**。V1.19 用精确串 `'B_BUDGET = 300'` 匹配 → 错报 `false`；V1.20 改用**元组赋值感知**的正则，命中该行（`:45`）。300.0 仍是**历史 CAL 夹具**，**不得**移植为 64 个 dev_check 状态的正式预算 |

**影响范围**：仅限 V1.19 部分验收的解释；**不推翻** G0-01–G0-05 的工程修复，也不改变任何历史科学裁定。

## 2. 组合认证器：**已真实执行**（不是读文本）

- 签名（实测）：`compose_certification(env, inst_idx, clock, vehicles, served_mask, plans, *, mask_set, protected, deferred, B, contract, base_plans=None, …)`
- `CAPABILITY_GAPS = ('quality_salability',)`；`installable = ok and not blocking_gaps`
- 本轮在**真实 day-0 状态**（采集器捕获 → 恢复车辆/`served_mask`）上实际调用，传入
  **明确标记 `SYNTHETIC_ONLY_NOT_A_FORMAL_C0_BUDGET` 的 B**（你允许用于验证拒绝机制）：
  返回 `installable = false`、`ok = false`，`reasons = [full_partition_failed,
  legality_short_circuit:full_partition, repair_plan_unverifiable]`。

**诚实的边界**：本次拒绝来自**合法性/分区门**（我手工构造的计划不是合法修复计划），
**尚未**把拒绝单独归因到 `quality_salability`。因此 G0-06a 目前是
**`EXECUTABLE_AND_REFUSES / QUALITY_GAP_ATTRIBUTION_NOT_ISOLATED`**：
下一步需用**合法修复计划**驱动同一调用，并断言 `blocking_capability_gaps == ['quality_salability']`。

## 3. 本轮新增失败与真实分母（全部保留）

| 序 | 失败 | 原因 | 处置 |
|---|---|---|---|
| 1 | G0-06a 首跑 `TypeError: missing 'B' and 'contract'` | 我把必填关键字参数写漏 | 补上 synthetic-only `B` 与 `contract=eff` |
| 2 | G0-06a 二跑 `ModuleNotFoundError: run_exp_energy_c0` | 认证器依赖 `scripts/evaluation/` 未在 `sys.path` | 加入 `scripts/evaluation`、`scripts/models` |
| 3 | G0-06a 三跑 **FAIL（保留）** | 拒绝原因归属未被隔离（合法性门先触发） | 记为限制，不伪装通过 |

**分母**：V1.20 = 3 行（2 PASS / 1 FAIL）；三次修复尝试全部留在本轮记录内，未改写为通过。

## 4. C0 预算（工作包 3.1–3.2）：待你裁定

- **修正后的来源审计**（有限范围）：命中 9 处预算声明 ——
  历史 CAL 夹具 `B_BUDGET = 300.0`（`test_d3_b2_certification.py:45`，元组赋值）与
  `SAA` 系列的 `budget=1e9` **隔离夹具**；**没有**适用于 64 个 dev_check 状态的**已登记协议预算**。
- 标签：**`ARCHIVED_BUDGET_GAP_CONFIRMED / CURRENT_PROVENANCE_NOT_EXHAUSTIVELY_SEARCHED`**（本轮为有界搜索，**不声称**穷尽）。
- 候选规则（**提案，`AWAITING_USER_BUDGET_RULING`**）：须独立给出**来源、单位、适用数据集与物理合同、
  可证伪假设、验证方法、适用条件**；裁定前只允许 synthetic-only 预算验证**拒绝机制**。

## 5. D8 / G-L / G-C 分项判词（按你的"三个可证伪问题"）

| 项 | 判词 |
|---|---|
| **当前状态能否认证**（G-L 提交前安全） | **BLOCKED**：接口层已闭合（V1.19 可执行反例），但**真实状态上的合法性/分区/能源组合认证**未通过（§2 的 `full_partition_failed` 即为此类）；须以合法修复计划复现 |
| **认证凭据能否绑定安装**（G-C 的绑定面） | **PARTIAL**：`certified↔installed` 身份不一致**必拒**（V1.19 G0-03 实测）；但**持久安装回执** `P_before/P_proposed/P_certified/P_after_install` 与**独立重读**未实现 |
| **已认证约束能否贯穿实际执行**（G-C 后续安全） | **BLOCKED**：写者排他仅有隔离工程证据（B3-AT2B-2a 有限收束），真实状态写入唯一性未建立 |
| **品质/可售保守保证** | **BLOCKED**：组合认证器自述无保守的决策时品质/可售认证，能力缺口阻断 `installable` |

**结论**：D8 保持 **BLOCKED**，按你的要求**只做**上述三个可证伪问题的受限解阻，**不**无边界扩展。

## 6. 工作包 1 / 2 的签署资格（**均未达到**）

- **V1.18（真实 F 前向执行版预登记）**：**`READY_FOR_USER_SIGNATURE = NO`**。
  已有：V1.17 §6 要点草案、V1.16 整轮撤销监督器、V1.14 相位持久化、隔离端到端验收 6/6。
  **仍缺**：与 V1.11 隔离的新 `run_id` 正式冻结、64 状态 `state_hash` 清单冻结、
  checkpoint/backbone/训练状态重建身份冻结、调用次数与初始化/单槽/整轮预算冻结、
  Windows 新目标名持久化的端到端验收在**真实 F worker** 上的接线。
- **L1（同掩码机制实验）**：**`READY_FOR_USER_SIGNATURE = NO`**；本轮仅完成
  **L1-1 共享输入可构造性**（真实 day-0 状态：`state_hash`、`clock`、生效合同、登记 D-Geo 评分器），
  并把 `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED` 标签固化；
  两臂统一枚举/输出协议、三个判定（首步分歧 / 完整修复完成 / 认证可执行计划）与
  空池·`score_len`·`score_shape`·分叉后 legal 不一致·认证失败·超时**合成反例**尚未实现。

## 7. E2 前置条件（不训练）

唯一变动因素（预训练 vs 随机初始化）+ 训练身份（数据/源码/种子/预算）+ 合格基线 = **未冻结**；
`E2` 保持**缺失**，本轮**不启动训练**，预登记仍是草案。

## 8. 下一项需要你批准的**唯一具体科学执行**

> **V1.18 真实 F 前向批次（64 状态，新 `run_id`，与 V1.11 完全隔离）** ——
> 仅在 V1.18 执行版预登记签署、且上述身份/预算/调用次数冻结后执行；
> **禁止**把新结果拼接为 V1.11 缺失的 42 条。

在此之前不需要你的其他批准；C0 预算裁定可在该批次之后进行（F 前向不依赖 C0 预算）。

## 9. Claim limits

未改上游 `MASKCO_code/`、冻结树、冻结数据、已消费采集分母与旧失败证据；历史 FAIL/BLOCKED 原件全部保留。
`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、`stable_online_gain=False`、
I3 正式负结果（ΔJ = −144.680，95% CI [−191.109, −100.125]）与旧 D2 `not_qualified` 全部不变。
