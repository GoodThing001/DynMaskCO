# D3 主模型资格集中解阻：V1.17 统一汇报

- 基准 HEAD：`cf5a091`（`main` 与 `codex/o0cc-cal-phys` 同步）
- 本轮可重放审计：`scripts/tests/d3_l1_v117_gap_audit.py` → `v117_evidence/D3_V117_GAP_AUDIT.json`
- **未执行**：真实 F 新批次、F−D-Geo 正式比较、E2 训练、L2/L3 科学批次、40 日批次、最终测试；
  未重采 64 状态、未补跑 V1.11 的 42 条缺失项
- 科学增益：**`NOT_ESTABLISHED`**

## 1. 资格矩阵（机械重算，非引用）

| # | 项 | 状态 | 依据 |
|---|---|---|---|
| A1 | V1.11 `PermissionError` 机理与 V1.12+ 修复 | **CLOSED_FOR_ENGINEERING** | `d3_l1_v111_real_f_forward.py:38` 用 `os.replace(temp, path)` 替换进度文件；修复改为**发布新目标名**并把 `PermissionError` 当作**有界、被记录的读取失败**（相位日志 + 注入式验收 `d3_l1_v112_phase_journal_acceptance.py` 主动注入 `PermissionError`） |
| A2 | 真实 F 前向结果 | **RECORDED_BLOCKED_ROUND** | 归档 `pass=22`、`failed_or_missing=42`；**未外推**、**未补跑** |
| B2 | 64 状态的 C0 预算 / 受绑定认证器 | **BLOCKED** | 归档值 `BLOCKED_MISSING_REGISTERED_BUDGET_AND_BOUND_CERTIFIER`、`NOT_RUN_MISSING_REGISTERED_BUDGET_AND_STATE_CONTRACT_BOUND_CERTIFIER` → **无有效依据，不得猜预算、不得用无限预算** |
| G0 | 残余风险（空 `certify_fn` 绕过、score shape/length、保护车辆与承诺篡改、认证/安装 SHA 不一致、重复安装、回滚无效） | **PENDING_ENGINEERING** | `D3_G0_RESIDUAL_RISK_AUDIT.md`（24 条标题）；每项需**隔离合成正反例**后方可宣称闭合 |
| B5 | 安装回执 `P_before/P_proposed/P_certified/P_after_install` | **PENDING** | 未实现；需隔离副本 + 独立复核安装后状态；**无授权的在线写回** |
| C1 | L1 F−D-Geo 七要素预登记（更新） | **DRAFT_PENDING_USER** | 须在**新 64 状态资产**上重新冻结共享初始状态/mask/legal 原序/合同/候选预算 |
| C2 | L2 新鲜状态决策质量方案 | **DRAFT_PENDING_USER** | 须用事前登记的**新鲜开发日**，不得与 64 状态重复记账 |
| C3 | E2 学习贡献对照预登记 | **DRAFT_PENDING_USER** | 唯一变动因素（预训练 vs 随机初始化）与训练身份须先冻结；当前**无合格 E2** |
| C4 | L3 在线效用 + A1/A2/A3 统计判据核对表 | **DRAFT_PENDING_USER** | 预算/阈值/种子/资源门**未冻结**，代理**不得自行填数** |

**汇总**：9 行 → **CLOSED 1 / BLOCKED 1 / PENDING 6**。

## 2. 本轮修改清单与交付

| 文件 | 动作 |
|---|---|
| `scripts/tests/d3_l1_v116_forward_control.py` | 新增尾部身份记录发布 + **`round_verdict()` 整轮撤销**语义（逐槽 PASS 降级为临时） |
| `scripts/tests/d3_l1_v116_forward_control_acceptance.py` | 新增 **6 场景合成验收**（真实子进程） |
| `scripts/tests/d3_l1_v117_gap_audit.py` | 新增**只读缺口审计器**（本汇报的机械依据） |
| `v116_evidence/A1_D3_L1_V116_FORWARD_CONTROL_ACCEPTANCE.json` | 6/6 PASS 原件 |
| `v117_evidence/D3_V117_GAP_AUDIT.json` | 9 行资格矩阵原件 |
| `A1_D3_L1_STAGE_REPORT_V116.md`、本文件 | 阶段与统一汇报 |

## 3. 修复前负例 → 修复后正例（保留原始失败）

| 项 | 修复前 | 修复后 |
|---|---|---|
| **整轮撤销**（你点名的风险） | `reconcile_rows()` 的逐槽 `qualified` 不参考整轮控制结果 → 64 个 PASS 行 + 尾部身份失败仍像"通过的轮次" | `round_verdict()` 要求「尾部记录有效 ∧ `control.ok` ∧ 64 行合格」，否则**全部降级**并持久化 `ROUND_REVOKED.json`；**S2**（尾部被删）实测 64→**0** |
| 结束身份失败 | 只置 `reason`，不影响行资格 | **S4**：worker 运行中改写被监视文件 → `HARD_STOP_IDENTITY_END` + `ROUND_REVOKED` |
| 槽内阻塞 | 无真实强杀验收 | **S6**：真实子进程被监督器按 `slot_span_deadline` 强杀 + 撤销 + 终止/未到达状态 |
| 行数不足 / 尾部损坏 | 未覆盖 | **S5** `INCOMPLETE_ROWS`、**S3** 尾部损坏 → 均撤销 |

**我自己的缺陷（保留原文，未改写为通过）**：S4 首版期望哈希写错（在 **start** 就 `HARD_STOP_IDENTITY`，未测到 end 路径）；
第二版忘传 `mode=mutate_watched`（worker 未改写文件）；V1.17 审计器两处路径/引号错误（`simulation/` 猜错、补丁多留引号、PowerShell `\"` 破损）。
以上均记录在案并由后续运行修正。

## 4. 身份 SHA（不得外推）

运行期身份由审计器逐文件重算，直接可读于 `v117_evidence/D3_V117_GAP_AUDIT.json` 与
`v116_evidence/A1_D3_L1_V116_FORWARD_CONTROL_ACCEPTANCE.json`（含 `tool_sha256`、`control_sha256`）。
既有登记值（本轮复用，未改）：F checkpoint `f5acf83f…903a7f5`、backbone `03615b08…c6d38db`、
dev_check NPZ `66573ddc…39c89`、profile `df1f7a6e…25b6`、生效合同 `edc1b2ea…4833`（base `32124bfd…30b0`、
校准来源 `0f00af5d…ce28`）。本文件**不**填写未经本轮计算的任何数值。

## 5. 剩余缺口（阻断项，按优先级）

1. **C0 预算与状态/合同绑定认证**（B2 **BLOCKED**）：需先给出**已登记预算**与**受绑定的认证器**，否则不可认证；
2. **G0 残余风险逐条隔离正反例**（PENDING_ENGINEERING）；
3. **B5 安装回执**（PENDING，需隔离副本）；
4. **D8 安装回执 / E2**（未闭合）；
5. **F 流程资格**：即使合成链路已闭合，**真实 F 64/64** 仍是独立事项，须走 §6 的签署。

## 6. 下一次需要你签署的执行版预登记（草案要点，**未授权执行**）

**`A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_18`** 应冻结：

1. **新 `run_id`**，与 V1.11 **完全隔离**；**禁止**把新结果拼接为 V1.11 缺失的 42 条；
2. **输入身份**：64 状态清单（原序）+ 每状态 `state_hash` + dev_check NPZ SHA + `booking_horizon=24.0` + 生效合同 `edc1b2ea…4833`；
3. **模型身份**：checkpoint SHA、backbone SHA、训练状态恢复口径、**有序 legal 不变**、评分长度/形状/有限性/**重复逐位一致**；
4. **调用次数与预算**：每状态前向次数、初始化/单状态/整批墙钟上限、超时语义（含 Windows 文件替换的有界重试上限）；
5. **失败语义**：`FAILED_*` 原因分类、**失败不可重采**、尾部身份失败 ⇒ **整轮撤销**（V1.16 语义）；
6. **源码与资产 SHA**：worker/supervisor/比较器/相位日志/持久化工具 + 数据/profile/合同；
7. **产出**：逐状态评分原件 + 1546 项独立只读复核 + 整轮裁决（`ROUND_PASS` / `ROUND_REVOKED`）。

## 7. Claim limits

工程资格进展**不等于**主模型增益；`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、
`stable_online_gain=False`、I3 正式负结果（ΔJ = −144.680，95% CI [−191.109, −100.125]）与旧 D2
`not_qualified` 全部不变。未修改上游 `MASKCO_code/`、冻结树与冻结数据；历史失败原件全部保留。
