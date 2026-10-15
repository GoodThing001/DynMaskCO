# CODEX_AUTONOMOUS_RESEARCH_LOOP

> 长期自主科研迭代账本。**新会话从此文件 + `项目当前状态.md` + 进度表顶部恢复**，不重新盘点项目。

- 最终使命：以严谨、可证伪、可复现的持续研究，探索并**验证** MaskCO 在 strict-online 动态冷链取货—回仓问题上的稳定正向在线效用增益；
  若证据不支持，则交付完整、可复核的负结果或机制发现。
- 当前 HEAD（最近一次本账本更新时）：见 §1「已消费正式实验」与最后一次 commit。
- 权威顺序：最新用户复核 > `项目当前状态.md` > 进度表顶部 > 本账本 > 历史报告。

## 0. 最近收束（新会话从这里续接）

### 0.0 本轮第二阶段（同一会话续接）：G0 反例可执行化 + 两份预登记草案 + L2/L3 草案

在 §0.1 之后继续，全部已 push：

| commit | 内容 |
|---|---|
| `3b6afe75` | **G0 残余风险反例可执行化**：8 个最小反例跑在**当前代码**上，区分 `FIX_CONFIRMED`（风险已关闭）与 `RISK_CONFIRMED_OPEN`（风险经执行确认**仍开放**）⇒ **7 FIX_CONFIRMED / 1 RISK_CONFIRMED_OPEN** |
| `3993e34e` | **E2 训练预登记草案**（`A1_D3_L1_E2_TRAINING_PREREGISTRATION_V1_18.md`，**不执行**） |
| `b336f6e6` | **`P_after_install` 四字段前瞻采集预登记草案**（**不执行**） |
| `4c3fb5bb` | **L2/L3 层级升级预登记草案**（**不执行**） |

**G0 反例结论（逐条，可复核 `D3_G0_RESIDUAL_RISK_COUNTEREXAMPLES.json`）**

| 风险 | 反例 | 判定 |
|---|---|---|
| A-2 | 无认证器的事务 | **FIX_CONFIRMED**：`no_certifier` 拒绝、0 次安装 |
| A-2 | 认证 A 安装 B | **FIX_CONFIRMED**：`certified_plan_identity_mismatch` 拒绝 |
| A-4b | 安装后改动调用方计划对象 | **FIX_CONFIRMED**：台账存**值拷贝**，记录不被改写 |
| A-4b | 计划身份不可计算 | **FIX_CONFIRMED**：拒绝，不以 `None` 身份安装 |
| A-5d | 无 `rollback_fn` | **FIX_CONFIRMED**：记 `rollback_unverified`，**不**记"已恢复" |
| A-5d | 有 `rollback_fn` | **FIX_CONFIRMED**：`committed_state_restored=True`，恰好调用 1 次 |
| A-5b | 重复 `finish` | **FIX_CONFIRMED**：第二次 `already_finished` 拒绝，总数 1 次安装 |
| **A-3b** | 掩码含**已提交**客户 | **RISK_CONFIRMED_OPEN**：采样器**照做**（计划中已提交客户被移动），适配器**无**已执行前缀守卫；真正的守卫在 `VehicleState` + B 池 |

**如实留档的自身缺陷**：A-3b 第一版反例把探针绕经安装门，产生了**无关失败导致的假"detected"**；
该次运行**不作为证据保留**，已改写为**直接机制检查**，并在产物里把 A-3b 单列为
`RISK_CONFIRMED_STILL_OPEN`，**不计入已关闭项**。

**三份草案的实质内容**

- **E2**：单因素 = backbone 初始权重来源。新增 **E2-7（已对源码核实，不是假设）**：
  `mpre_trained.py` 的 `c_adapter` / `delta_head` 都是 `MLP(..., zero_last=True)`，末层
  kernel/bias 为 `zeros_init()`、隐藏层取自 rngs 流 ⇒ 若 E2-rand 只随机化 backbone 而两层初始化
  不同，则"预训练 vs 随机"会与"零末层 vs 随机初始化""seed A vs seed B"**同时**变化。规格**强制**
  两层在两侧使用**同一初始化器与同一 seed**，并要求**参数名/形状/dtype 逐项相等**的机械证据。
  同时重述 `--seed` **不是** E2（backbone 仍载入预训练）。
- **`P_after_install`**：回执通道已闭合（7/7），`P_after_install` **只**允许来自通过独立重读的回执，
  **禁止**由 `P_certified`/`P_proposed` 推算；并如实列出**未闭合**三项：环境自身 `_plan_and_commit`
  不可观测、`A-3b` 仍开放、状态级回滚未被回执证明。
- **L2/L3**：三层不可跳级逐层写死前置门；质量口径**三列分离**（`candidate`/`accepted`/`closed_loop`，
  `candidate` **不得**称"部署"）；L2 完整 C0 认证**依赖用户裁定 B**；L3 需**新鲜开发日** + 等预算
  + 冻结统计计划。

**仍未完成、且本轮未做**：2×2 统一场景输入适配器**设计**（最小清单第 7 项，纯设计）；
32 状态确定性重采集（最小清单第 3 项，本身是一次新执行，需另行授权）。

### 0.1 本轮第一阶段（V1.18 + L1 + D8 + C0 搜索）**全部完成**

自 `f73d2c88` 续接，**V1.18 达到 `READY_FOR_USER_SIGNATURE`，L1 工程协议闭合**。三项已 push：

| commit | 内容 |
|---|---|
| `e8907eb` | **V1.18**：隔离 run_id、真实 F 证据推导的预算冻结、V1.14 相位日志 + V1.16 整轮撤销接入 F worker、9 场景合成端到端验收（9/9）、独立只读复核器（9/9 + 监督自检 9/9）、七要素执行版预登记、`--readiness` 动态计算 |
| `8de9a3e` | **L1 两臂协议**：共享输入单因素机器可检验、三判定分离、六个合成反例（10/10）、可重放验收器、独立只读审计 17/17、七要素执行版预登记 |
| `eeb8dbec` | **D8 持久安装回执 + 独立重读**（G0 A-5c 关闭，7/7）；不动 `d3_geo_adapter.py` 一个字节 |

**V1.18 现状（可直接复核）**

- `run_id = d3l1-real-f-forward-20261015-v118`，与 V1.9 / V1.11 文本不相交；
  绑定 64 状态（台账发布顺序，不重排）+ 模型 / 骨干 / 数据 / profile / 基础+有效合同 /
  14 依赖模块身份；运行期 watch 集 = 10 代码文件 + ckpt + backbone + npz + profile = **14 路径**。
- 调用次数 **129**（1 warm-up + 64×2，沿用 V1.11 协议）；预算**由 V1.11 真实 F 实测推导**：
  槽跨度 max/median = 0.344 / 0.234 s（n=22）、模型初始化 6.344 s、首次编译 6.781 s、
  22 状态批墙钟 19.562 s ⇒ 全 64 状态外推 **70.03 s**；冻结上限 **60 / 10 / 180 s**
  （余量 4.57× / 29.07× / 2.57×，**不放宽**）。V1.9 的 F-free 计时**只作溯源**，明确排除。
- V1.11 根因（固定目标名 `STAGE.json` 被读者持有 ⇒ `WinError 5` ⇒ 22/64）**结构性消除**：
  每次发布到**从未存在过**的新目标名并**写后读回校验**；读取有界，持久拒绝与损坏记录均为**硬错误**。
- 验收：`v118_evidence/SYNTHETIC_END_TO_END.json` **9/9 PASS**（读者全持句柄、瞬时共享拒绝、
  损坏记录、真实截止强杀、真实进程树强杀、尾部身份篡改、尾部缺失、63/64 行、评分长度坏行）；
  V1.16 回归 **6/6**。
- 独立复核器：`v118_evidence/INDEPENDENT_READONLY_VERIFIER.json` = `VERIFIED_NOT_EXECUTED` 9/9
  （真实 F 未跑，如实不是 PASS），
  `..._SUPERVISED_SELFTEST.json` = 64 合成行**独立判定器 64/64 与生产判定器逐行一致**。
- `--readiness`：**`READY_FOR_USER_SIGNATURE`，`missing_before_signature = []`**，
  同时如实记录 `real_F_forward_executed = false` —— 前置条件已备齐，待签署的就是那一次执行。
- 唯一待批科学执行：**V1.18 真实 F 64 状态前向**（新 run_id，**禁止拼接 V1.11 缺失的 42 条**）。

**L1 现状（可直接复核）**

- `scripts/simulation/d3_l1_two_arm_protocol.py`：共享输入（单因素三重机械检查：同一 state 对象、
  运行前后 state 摘要三处相等、`DECLARED_VARYING_FIELDS = ('scorer_id',)`）；
  `legal_fingerprint` 同时用于共享身份与每臂逐步轨迹（避免两套编码互相比较）；
  V1 / V2 / V3 **三个独立判定**；空池 = `NOT_APPLICABLE_EMPTY_MASK`（**不是**"一致"）；
  认证器缺失时 V3 = `NOT_RUN`（既非 True 也非 False）。
- 验收 `l1_gates/A1_D3_L1_TWO_ARM_PROTOCOL_ACCEPTANCE.json` **10/10**；空池、`score_len`、
  `score_shape`、分叉后 legal、认证失败、超时六类反例全部命中；
  `no_legal` 如实记为 **`NOT_EXERCISED`**（64 状态中每个可见订单都至少 1 个可行插入，
  该分支在本数据集不可自然触发，**不冒充通过**）。
- 独立审计 `l1_gates/A1_D3_L1_TWO_ARM_INDEPENDENT_AUDIT.json` **17/17**：
  **不导入**协议 / geo_adapter / 验收器，只读 trace JSON，独立重算 V1/V2/V3 并与记录值对拍。
- L1 正式比较仍 `BLOCKED`，**唯一硬前置 = V1.18 真实 F 前向**（不是协议缺失）。

**D8 / C0 本轮新增**

- D8：`scripts/simulation/d3_install_receipt.py` + `d3_install_boundary.py`（**再导出**冻结的
  `pre_install_reverify` 并以对象身份证明未被重实现）+ 7 场景验收 7/7。
  回执**只在 ledger 确实安装恰一个计划时**写出，存全部 `plan_hash` 所需字段以便读者**自行重算**
  （不靠声明）；R1 由**独立进程**重读。**仍不证明**：环境自身 `_plan_and_commit` 的提交、
  状态级回滚、已安装计划的品质/能耗性质。
- C0 预算：`C0_BUDGET_PROVENANCE_SEARCH.json` —— 有界穷举（脚本 / D3 证据 / data / docs /
  research / CC_Compare 审计分列），**184 命中：DOC_MENTION 165 / TEST_LOCAL 12 /
  NON_PORTABLE_FIXTURE 7 / 协议可达候选 0**，判定
  **`SEARCH_EXHAUSTIVE_NO_REGISTERED_PROTOCOL_BUDGET_FOUND`**（搜索半边已闭合）。
  **勘误并留档**：第一版分类器按"上下文含 protocol 一词"判类，把 29 处**文档提及**误升为
  `REGISTERED_PROTOCOL_BUDGET`；已在报告 `corrected_from` 中原样保留该误判并给出更正后的分类器。
  预算裁定仍 **`AWAITING_USER_BUDGET_RULING`**（代理不得自定）。

### 0.2 上一轮收束（V1.18(A)，commit `52362db`）

- V1.18(A) 归档资产冻结 **4/4 PASS**：`STATE_LEDGER.jsonl` 的 `PUBLISHED` 行 64 条按发布顺序、
  64/64 带 `state_hash`、128/128 snapshot+validation 原件 SHA 现场复算一致、完整性位全真、
  `run_id` 唯一。证据 `v118_evidence/D3_V118_FREEZE_CHECK.json`。
- 同时实测采集预算（**仅溯源**）：`slot_span_wall_s` 0.625/1.422/3.328 s（n=64）、
  `day_rollout_wall_s` max 0.519 s（n=16）。
- 事故与恢复（如实记录）：一次工作树补丁损坏了该脚本；已从 `1b2bc9b` **逐字节恢复**并只重放定位器改写。
- 本轮已把该节列出的五项缺失**全部闭合**（见 §0.1）。

## 1. 已消费的正式实验次数（**不得复用、不得重跑**）

| 实验 | 次数 | 状态 | 证据 |
|---|---|---|---|
| F-free 一次性状态采集（64 槽位） | **1（已用尽）** | 64/64 发布，独立验收 1498/1498，未补采 | `v19_evidence/`、`l1_gates/A1_D3_L1_FORMAL_64_INDEPENDENT_AUDIT_V19.json` |
| 真实 F 前向批次 V1.11 | **1（失败，已用尽）** | 22/64 完成、42 未完成、整轮 `BLOCKED`；**禁止拼接补跑** | `v111_evidence/REAL_F_FORWARD_QUALIFICATION_V111.json` |
| 服务器 F-free 计时样本 | 1（已用尽） | `recorded_Q_server=NOT_PASSED`，事后身份 `MATCH` | `l1_gates/A1_D3_L1_SERVER_TIMING_*.json` |
| 真实 F−D-Geo 机制比较（L1） | **0（未授权）** | 协议已闭合，**待 V1.18 通过后另行签署** | `l1_gates/A1_D3_L1_TWO_ARM_*.json` |
| E2 训练 | **0（未授权）** | — | — |
| L2/L3、40 日批次、最终测试 | **0（未授权）** | — | — |

## 2. 当前真实阻断（按优先级）

1. **V1.18 真实 F 64 状态前向**：工程资格已 `READY_FOR_USER_SIGNATURE`，**唯一待批的科学执行**。
   未执行前 L1 正式比较不可启动。
2. **L1 正式机制比较**：协议与反例已闭合；硬前置 = 第 1 项。
3. **C0 正式预算**：搜索半边已穷举闭合（无注册协议预算），裁定仍 `AWAITING_USER_BUDGET_RULING`。
4. **D8**：持久回执已实现（A-5c 关闭）；**仍阻断**于 G-L 状态认证、G-C 真实写入唯一性、
   品质保守保证，以及"环境自身提交不可观测"。
5. **E2**：前置条件未冻结，未训练。
6. **L2/L3 与 A1/A2/A3**：未启动。

## 3. 已完成并可独立复核的工作包

| 工作包 | 结论 | 证据 |
|---|---|---|
| P0-A 逐位比较器 | 16/16 反例门通过 | `l1_gates/A1_D3_L1_BITWISE_PARITY_GATE.json` |
| P0-B 因果扰动（方案 B） | `DIST_MAT_GENERATION_CONTRACT = VERIFIED`（16/16 天逐位） | `l1_gates/A1_D3_L1_P0B_CAUSAL_PERTURBATION.json` |
| P0-C 采集工程 | 64/64 发布 + 1498/1498；V1.8 起 6 套门 | `l1_gates/A1_D3_L1_P0C_*.json` |
| V1.16 监督器整轮撤销 | 6/6 合成验收 | `v116_evidence/A1_D3_L1_V116_FORWARD_CONTROL_ACCEPTANCE.json` |
| V1.19 G0 可执行差分 | 8/8；G0-01–G0-05 接口层闭合 | `v119_evidence/D3_V119_G0_INVARIANT_DIFFERENTIAL.json` |
| V1.20 组合认证器**真实执行** | `installable=false`；品质缺口归属**未隔离** | `v120_evidence/D3_V120_READINESS.json` |
| **V1.18 F 前向工程资格** | `READY_FOR_USER_SIGNATURE`；9/9 + 复核 9/9 | `v118_evidence/SYNTHETIC_END_TO_END.json`、`INDEPENDENT_READONLY_VERIFIER*.json` |
| **L1 两臂协议** | 10/10 + 独立审计 17/17 | `l1_gates/A1_D3_L1_TWO_ARM_*.json` |
| **D8 持久安装回执** | A-5c 关闭；7/7 | `B3A_receipt_evidence/D3_INSTALL_RECEIPT_ACCEPTANCE.json` |
| **C0 预算溯源穷举** | `SEARCH_EXHAUSTIVE_NO_REGISTERED_PROTOCOL_BUDGET_FOUND` | `C0_BUDGET_PROVENANCE_SEARCH.json` |
| **G0 残余风险反例可执行化** | 7 FIX_CONFIRMED / 1 RISK_CONFIRMED_OPEN（A-3b 仍开放） | `D3_G0_RESIDUAL_RISK_COUNTEREXAMPLES.json` |
| **E2 预登记草案** | 规格已交付（含 E2-7 零末层强制项，源码已核）；**未训练** | `A1_D3_L1_E2_TRAINING_PREREGISTRATION_V1_18.md` |
| **`P_after_install` 采集预登记草案** | 规格已交付；回执通道 7/7，**未采集** | `A1_D3_L1_P_AFTER_INSTALL_COLLECTION_PREREGISTRATION_V1_18.md` |
| **L2/L3 升级预登记草案** | 规格已交付；**未执行** | `A1_D3_L1_L2_L3_ESCALATION_PREREGISTRATION_V1_18.md` |

## 4. 重要失败与真实分母（原件保留，未改写）

- V1.20 = 3 行 2 PASS 1 FAIL：漏必填 kwargs → `sys.path` 缺 `scripts/evaluation` → 归属未隔离。
- V1.19 = 8/8，但含两处**解释**缺陷 + `B_BUDGET` 匹配错误 → V1.20 勘误；本轮 C0 搜索在报告
  `corrected_from` 中再次留档一版**分类器**误判（29 处文档提及被误升为注册预算）。
- V1.11 F 前向 22/64：Windows 进度文件替换 `PermissionError`；V1.18 起以"新目标名 + 写后校验 +
  有界读取"结构性修复（9/9 场景 + V1.16 6/6 回归）。
- `reconcile_rows` 本轮新增可选 `rows_dir` / `row_glob`（默认值复现历史行为，V1.16 6/6 通过）；
  **一次实现错误（默认 `rows_dir` 从 `'scores'` 误改为 `out`）已被 V1.16 回归当场抓住并修正**，
  未隐藏。
- **A-3b 反例第一版**把探针绕经安装门，得出**无关失败导致的假"detected"**；该次运行**不作为证据保留**，
  已改写为直接机制检查，并把 A-3b 在产物中单列为 `RISK_CONFIRMED_STILL_OPEN`。
- **C0 搜索第一版分类器**按"上下文含 protocol 一词"判类，把 29 处**文档提及**误升为注册预算；
  已在报告 `corrected_from` 中原样保留并给出更正分类器。
- 较早：P0-B 首版把 float64 与 float32 计算路径差异误判为"外部路网矩阵"（V1.4 已更正）。

## 5. 下一个可自动执行任务（无需用户授权）

1. **V1.18 真实 F 前向**（唯一待批；未获签署前不得执行）——签署后：
   `python scripts/simulation/d3_l1_v118_forward_runner.py --supervise --out-dir <run_v118> --authorization <file>`，
   随后 `--readiness` 与 `d3_l1_v118_independent_verifier.py` 复验。
2. **L1 正式机制比较**（V1.18 通过 + L1 预登记签署后）；运行预算须按 L1 预登记 §6 用**真实 F 实测**
   冻结，不得引用 V1.9 计时。
3. **D8 受限三问**：G-L 状态认证；G-C 真实写入唯一性；品质保守保证（回执已不再是缺口）。
4. **E2/L2/L3 与 A1/A2/A3 预登记草案**（不训练、不跑批次）。
   —— **已完成**：E2 → `A1_D3_L1_E2_TRAINING_PREREGISTRATION_V1_18.md`；
   `P_after_install` → `A1_D3_L1_P_AFTER_INSTALL_COLLECTION_PREREGISTRATION_V1_18.md`；
   L2/L3 → `A1_D3_L1_L2_L3_ESCALATION_PREREGISTRATION_V1_18.md`（**均未执行**）。
5. **2×2 统一场景输入适配器设计**（最小清单第 7 项，纯设计；Role C 现绕 cond_hist 场景流，
   接 MaskCO 须走**一条**决策代码路径）。
6. **32 状态确定性重采集**（最小清单第 3 项）：**本身是一次新执行**，须先获授权并冻结清单/预算；
   合同/profile 身份须在该预登记内显式记录（本轮已把 `effective_contract_hash` 与 `profile_hash`
   写入 V1.18/L1/E2 身份块，不再以"缺口"表述）。
7. 若需扩充 `no_legal` 覆盖：需先有**非活跃**且无可行插入的订单来源，并保持 strict-online 因果
   （本数据集不可自然触发，已如实记录）。

## 6. 需要用户签署的事项（唯一清单）

1. **V1.18 真实 F 前向批次**（64 状态、`run_id=d3l1-real-f-forward-20261015-v118`、与 V1.11 完全隔离、
   禁止拼接 42 条）——**唯一待批的科学执行**；
2. **C0 正式预算**（`AWAITING_USER_BUDGET_RULING`，代理不得自定）；
3. **L1 正式机制比较**（在 V1.18 通过、L1 预登记签署后）；
4. E2 训练、L2/L3、40 日批次与最终测试（各自独立授权）。

## 7. 硬约束（每次执行前自检）

`MASKCO_code/` 永不修改；不改冻结数据/冻结树/历史 FAIL 原件/用户正式裁决；扩展代码不加 `__init__.py`；
strict-online 只用已揭示信息；已执行前缀与承诺冻结、每单恰好服务一次；C0 硬认证超预算不提交；
`distance_cost` 与 `coldchain_cost` 严格分离；效用 = 收入 − 燃油 − 拒绝损失（品质不货币化）；
开发证据与最终测试隔离；正式运行须先有七要素预登记 + 身份冻结 + 资源门 + 授权；
不得事后调判据/加次数/挑结果；不得用 `PENDING`/`None`/无限预算/历史 CAL 夹具/synthetic-only 冒充正式值；
不得把合成 64 行 PASS 冒充真实 F 64 状态 PASS；**未测试的分支记 `NOT_EXERCISED`，不得记 PASS**。

## 8. 汇报单位

以**完整工作包**或**真正需要人工决定的科学关口**为汇报单位；不以单个脚本/测试/commit 为单位；
不为形成汇报而中断仍可自主完成的工作。汇报固定回答六问（见总指令 §10）。
