# CODEX_AUTONOMOUS_RESEARCH_LOOP

> 长期自主科研迭代账本。**新会话从此文件 + `项目当前状态.md` + 进度表顶部恢复**，不重新盘点项目。

- 最终使命：以严谨、可证伪、可复现的持续研究，探索并**验证**MaskCO 在 strict-online 动态冷链取货—回仓问题上的稳定正向在线效用增益；
  若证据不支持，则交付完整、可复核的负结果或机制发现。
- 当前 HEAD（最近一次本账本更新时）：见 §1「已消费正式实验」与最后一次 commit。
- 权威顺序：最新用户复核 > `项目当前状态.md` > 进度表顶部 > 本账本 > 历史报告。

## 0. 最近收束（新会话从这里续接）

- **V1.18(A) 归档资产冻结已闭合（commit `52362db`）**：`scripts/tests/d3_l1_v118_freeze_check.py`
  已改为读**真实** V1.9 采集归档 `v19_evidence/formal_collection/STATE_LEDGER.jsonl`
  （行语义 = `event == 'PUBLISHED'`，不是 `status`）：64 行按**发布顺序**（= 规范 `(day_id, position)` 顺序，
  绝不重排）、64/64 带 `state_hash`、128/128 snapshot+validation 原始文件 SHA 现场复算一致、
  完整性位（`bitwise_parity_ok`/`projection_causal`/`suffix_boundary_sha_equal`/
  `ledger_integrity_ok`/`registered_before_capture`）全为真、`run_id` 唯一。
  证据 `v118_evidence/D3_V118_FREEZE_CHECK.json` = **4 行 / 4 PASS / 0 FAIL**。
- 同时**实测**采集预算（仅作溯源，**不是**新运行冻结值）：`slot_span_wall_s` min/median/max =
  0.625 / 1.422 / 3.328 s（n=64）；`day_rollout_wall_s` max = 0.519 s（n=16）。
- 事故与恢复（如实记录，未隐藏、未降低强度）：一次工作树补丁损坏了该脚本（丢失 `main()`、argparse 与
  V118-1/2 身份检查）；已从 `1b2bc9b` **逐字节恢复**并只重放定位器改写。
- **仍为 `NOT_READY`，未产生任何新科学结论**：缺隔离新 `run_id`、新运行自身的预算冻结、
  真实 F worker 上的 Windows 新目标持久化验收、绑定新 run 身份的只读独立复核器。

## 1. 已消费的正式实验次数（**不得复用、不得重跑**）

| 实验 | 次数 | 状态 | 证据 |
|---|---|---|---|
| F-free 一次性状态采集（64 槽位） | **1（已用尽）** | 64/64 发布，独立验收 1498/1498，未补采 | `v19_evidence/`、`l1_gates/A1_D3_L1_FORMAL_64_INDEPENDENT_AUDIT_V19.json` |
| 真实 F 前向批次 V1.11 | **1（失败，已用尽）** | 22/64 完成、42 未完成、整轮 `BLOCKED`；**禁止拼接补跑** | `v111_evidence/REAL_F_FORWARD_QUALIFICATION_V111.json` |
| 服务器 F-free 计时样本 | 1（已用尽） | `recorded_Q_server=NOT_PASSED`，事后身份 `MATCH` | `l1_gates/A1_D3_L1_SERVER_TIMING_*.json` |
| 真实 F−D-Geo 机制比较（L1） | **0（未授权）** | — | — |
| E2 训练 | **0（未授权）** | — | — |
| L2/L3、40 日批次、最终测试 | **0（未授权）** | — | — |

## 2. 当前真实阻断（按优先级）

1. **V1.18 真实 F 64 状态评分资格**：`READY_FOR_USER_SIGNATURE = NO`（缺隔离 run_id、64 状态身份清单冻结、
   checkpoint/backbone/训练状态身份冻结、调用次数与时间预算冻结、真实 F worker 上的 Windows 持久化验收）。
2. **L1 同掩码机制归因**：仅 L1-1 共享输入可构造性通过；两臂协议与六个合成反例未实现。
3. **C0 正式预算**：`ARCHIVED_BUDGET_GAP_CONFIRMED / CURRENT_PROVENANCE_NOT_EXHAUSTIVELY_SEARCHED` → `AWAITING_USER_BUDGET_RULING`。
4. **D8**：`BLOCKED`（G-L 状态认证未过；G-C 真实写入唯一性未建立；持久安装回执未实现；品质保守保证缺失）。
5. **E2**：前置条件未冻结，未训练。
6. **L2/L3 与 A1/A2/A3**：未启动。

## 3. 已完成并可独立复核的工作包

| 工作包 | 结论 | 证据 |
|---|---|---|
| P0-A 逐位比较器 | 16/16 反例门通过（含 ±0、NaN payload、NumPy dtype/标量、容器类型） | `v116_evidence`…（见 `l1_gates/A1_D3_L1_BITWISE_PARITY_GATE.json`） |
| P0-B 因果扰动（方案 B） | `DIST_MAT_GENERATION_CONTRACT = VERIFIED`（16/16 天逐位）；扰动由扰动后坐标按 float32 重算 | `l1_gates/A1_D3_L1_P0B_CAUSAL_PERTURBATION.json` |
| P0-C 采集工程 | 64/64 发布 + 1498/1498；V1.8 起 6 套门 | `l1_gates/A1_D3_L1_P0C_*.json` |
| V1.16 监督器整轮撤销 | 6/6 合成验收（尾部删除/损坏/结束身份/行数不足/真实强杀） | `v116_evidence/A1_D3_L1_V116_FORWARD_CONTROL_ACCEPTANCE.json` |
| V1.19 G0 可执行差分 | 8/8；G0-01–G0-05 接口层闭合 | `v119_evidence/D3_V119_G0_INVARIANT_DIFFERENTIAL.json` |
| V1.20 组合认证器**真实执行** | `installable=false`；但拒绝归因于合法性/分区门，品质缺口归属**未隔离** | `v120_evidence/D3_V120_READINESS.json` |

## 4. 重要失败与真实分母（原件保留，未改写）

- V1.20 = 3 行 2 PASS 1 FAIL：漏必填 kwargs → `sys.path` 缺 `scripts/evaluation` → 归属未隔离（FAIL 保留）。
- V1.19 = 8/8，但含两处**解释**缺陷（G0-06a 属能力声明检查、G0-06b 属预填空值检查）+ `B_BUDGET` 匹配错误 → V1.20 勘误。
- V1.11 F 前向 22/64：Windows 进度文件替换 `PermissionError`；V1.12+ 以"发布新目标名 + 有界读取"修复。
- 较早：P0-B 首版把 float64 与 float32 计算路径差异误判为"外部路网矩阵"（V1.4 已更正）。

## 5. 下一个可自动执行任务（无需用户授权）

1. **V1.18 冻结前检查**：核对 64 状态清单顺序/`state_hash`、数据/profile/合同/checkpoint/backbone 身份、
   supervisor 与相位持久化接线 → 产出可签署执行预登记（`scripts/tests/d3_l1_v118_freeze_check.py`）。
2. **L1 两臂协议实现**（不运行真实比较）：共享 snapshot/P0/mask/ordered legal/budget 的两臂枚举与输出协议，
   三个判定分离，六个合成反例（空池、`score_len`、`score_shape`、分叉后 legal、认证失败、超时）。
3. **D8 受限三问**：合法修复计划的真实组合认证；`P_*` 持久回执 + 独立重读；真实状态写入唯一性。
4. **E2/L2/L3 预登记草案**（不训练、不跑批次）。

## 6. 需要用户签署的事项（唯一清单）

1. **V1.18 真实 F 前向批次**（64 状态、新 `run_id`、与 V1.11 完全隔离、禁止拼接 42 条）——**唯一待批的科学执行**；
2. **C0 正式预算**（`AWAITING_USER_BUDGET_RULING`，代理不得自定）；
3. L1 正式机制比较（在 V1.18 通过、L1 预登记签署后）；
4. E2 训练、L2/L3、40 日批次与最终测试（各自独立授权）。

## 7. 硬约束（每次执行前自检）

`MASKCO_code/` 永不修改；不改冻结数据/冻结树/历史 FAIL 原件/用户正式裁决；扩展代码不加 `__init__.py`；
strict-online 只用已揭示信息；已执行前缀与承诺冻结、每单恰好服务一次；C0 硬认证超预算不提交；
`distance_cost` 与 `coldchain_cost` 严格分离；效用 = 收入 − 燃油 − 拒绝损失（品质不货币化）；
开发证据与最终测试隔离；正式运行须先有七要素预登记 + 身份冻结 + 资源门 + 授权；
不得事后调判据/加次数/挑结果；不得用 `PENDING`/`None`/无限预算/历史 CAL 夹具/synthetic-only 冒充正式值；
不得把合成 64 行 PASS 冒充真实 F 64 状态 PASS。

## 8. 汇报单位

以**完整工作包**或**真正需要人工决定的科学关口**为汇报单位；不以单个脚本/测试/commit 为单位；
不为形成汇报而中断仍可自主完成的工作。汇报固定回答六问（见总指令 §10）。
