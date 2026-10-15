# V1.18：真实输入 F 前向执行版预登记（七要素）

- 轮次：`A1-D3-L1` / 任务 `V1.18 real-F forward engineering qualification`
- 状态：`EXECUTION_PREREG_FROZEN / READY_FOR_USER_SIGNATURE`（**本文件不授权、也未执行任何真实 F 前向**）
- 上位于：`A1_D3_L1_REAL_F_FORWARD_QUALIFICATION_PREREGISTRATION_V1_11.md`（同一调用协议；V1.18 只修工程缺口，不改科学口径）
- 绑定运行：`run_id = d3l1-real-f-forward-20261015-v118`
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**

---

## 0. 本文件回答的问题

V1.11 的**唯一一次**真实 F 前向已消费并**整轮失败**（22/64 完成、42 未完成，根因见 §4）。
V1.18 不是"再跑一次同样的事"：它把 V1.11 暴露的三处工程缺口逐个闭合，并把新运行的
调用次数、时间预算与身份**冻结在一个可机械复核的架构里**，然后把**唯一待批的科学执行**
交回用户签署。本文件是那一份可签署的执行版预登记。

**八项签署前必备条件**（全部由 `d3_l1_v118_forward_runner.py --readiness` **从磁盘工件动态计算**，
不存在手写 ready 标志）：

| # | 条件 | 证据工件 | 当前 |
|---|---|---|---|
| 1 | 隔离的新 `run_id`，与 V1.9 / V1.11 文本不相交 | `run_v118/RUN_IDENTITY_*.json` | **满足** |
| 2 | 64 状态清单按台账**发布顺序**冻结、逐状态 `state_hash`、与 manifest 零分歧 | 同上 `state_list` | **满足** |
| 3 | 模型 / 骨干 / 数据 / profile / 合同 / 依赖源码身份逐字节复核 | 同上 `identities` + `dependency_identity` | **满足** |
| 4 | 调用次数与初始化 / 槽 / 批时间预算**由真实 F 实测证据推导**并冻结 | 同上 `call_schedule` + `budgets` | **满足** |
| 5 | 真实 worker / 监督器路径上的**新目标名持久化**端到端验收（合成 scorer） | `v118_evidence/SYNTHETIC_END_TO_END.json`（9/9） | **满足** |
| 6 | 与新 `run_id` 及冻结 SHA 绑定的**独立只读复核器**，且其行级判定器已被证明可达判定 | `INDEPENDENT_READONLY_VERIFIER.json` + `..._SUPERVISED_SELFTEST.json` | **满足** |
| 7 | V1.14 相位日志 + V1.16 整轮撤销监督器接入新 F worker 路径 | `SUPERVISION_RESULT_*.json` / `ROUND_*.json` / 9 场景 | **满足** |
| 8 | 真实 F 前向本身 | —— | **待签署**（本文件即为此） |

`--readiness` 现返回 `READY_FOR_USER_SIGNATURE`（`missing_before_signature = []`），
同时如实记录 `real_F_forward_executed = false`：**前置条件已备齐，正是要请用户授权那一次执行**。

---

## 1. 要素① 状态来源：**已闭合**（不再是 `STATE_ASSET_GAP`）

- 来源：V1.9 一次性 F-free 采集的 **64 个已发布状态**，位于
  `v19_evidence/formal_collection/`（`STATE_LEDGER.jsonl` + `snapshots/` + `validation/`）。
- **顺序**：`STATE_LEDGER.jsonl` 的 `event == 'PUBLISHED'` 追加顺序，**即**规范
  `(day_id, position)` 顺序；本运行**从不重排**（重排会冻结一个与采集器发布的不同对象）。
- **逐状态身份**：每行带 `state_hash`；64/64 现场复算通过；与
  `STATE_MANIFEST_64_V19.json` 的 `state_hash / snapshot_sha256 / validation_sha256` **零分歧**。
- **状态绑定**：`state_binding_sha256 = sha256(json([[day, pos, state_hash], ...]))`
  = `f5eedd1027342496939e7…`（由独立复核器二次推导，与运行身份一致）。
- **输入张量**：直接复用 V1.10 已验收的 64 个 `original` 派生输入
  （`v110_evidence/derived/day{DD}_pos{P}.pkl`，逐文件 SHA 复核），
  本运行**不重新派生**输入（重新派生是一次新推导，可能静默不同）。
- 每个期望行的 `input_sha256 / legal_sha256 / n_legal / Nv` 都来自 V1.10 报告，
  汇总为 `FORWARD_TARGETS`（`targets_sha256 = 37e23f7567e3391a…`）。
- 已用于开发的数据只支持"在这些状态上 F 的评分接口是否可用"，**不是**独立科学验证；
  L2 必须另用事前登记的新鲜开发日。
- **未**重新采集、**未**替换任何槽位、**未**从 64 中剔除任何状态（全部分母保留）。

## 2. 要素② 唯一比较：**本阶段零比较**

本阶段运行**只做一件事**：对 64 个已冻结输入各调用真实 F 评分两次。因此

```
mask_scope      = B                      # 与 mask_candidate_pool 的既定作用域一致（本阶段不再构造掩码）
eps_rel/eps_abs = 0.0 / 0.0              # D-Geo 本阶段不调用
tie_break       = legal_order_first      # F 的历史 first-maximum argmax 语义，原样保持
F:  s_F(a) = restored MaskCO logits      a_F = first argmax_{a in legal} s_F(a)
D-Geo: 不调用（0 次）
```

`F_D_Geo_comparisons = 0`、`install_calls = 0`、`training_runs = 0`、`new_rollouts = 0`
由运行报告机械累加，不是声明。

## 3. 要素③ 判定：**只判定 F 评分接口的工程可用性**

本阶段**不产生**任何分歧、增益、显著性、学习贡献或行为结论。记录：
逐状态两次前向的 `dtype / shape / bytes` 逐位一致、finite、长度等于有序 legal 长度、
原始 numpy 数组及其十六进制、两次墙钟、`span`、input/legal 指纹。
**明确不主张**：不达 L3 完整 8+2 秒在线发布门；不主张 F 优于/劣于任何对照。

## 4. 要素④ 日志：全部落盘到**从不重名的目标**

V1.11 的失败链（原件保留，未改写）：

```
worker: os.replace(STAGE.json.tmp, STAGE.json)      # 固定目标名
supervisor: latest(phases) -> 读同一个固定名
=> Windows: 读者持有句柄时 os.replace 报 [WinError 5]，槽 5/2 中止，整批 22/64
```
（`v111_evidence/WORKER_FAILURE.json` 原文即为该 `PermissionError`。）

V1.18 的两处结构性修复：

- **R-D1（写方）**：`publish_new` 只写**从未存在过的**目标名
  `prefix_<uuid>.json`（`open(..., 'xb')`），临时名再 `os.replace` 到一个**不存在**的目标
  ⇒ 不存在"读者持有我要替换的目标"这一情形，竞态在结构上不可能发生。
- **R-D2（读方）**：`VerifiedReader` 只按**文件名集合**读取，临时文件从不参与；
  仅 `PermissionError`（共享冲突）在**有界窗口**内重试，持久拒绝是**硬错误**，
  损坏记录也是**硬错误**（不得当成"还没写"）。

另加：**写入后读回校验**（`verify=True` 重新读该目标并逐字节比较 sha256）；
不一致即 `PermanentWriteError` 并把该文件作为隔离证据留在原地。

日志记录组（沿用要素④原表并落到实际字段）：

| 记录组 | 实际工件 |
|---|---|
| 状态身份 | `FORWARD_TARGETS_*.json`：`day_id/position/state_hash/input_sha256/legal_sha256/n_legal/Nv` |
| 模型身份 | `RUN_IDENTITY_*.json`：F ckpt SHA、backbone SHA、训练元数据、依赖源码 SHA、`label` |
| 共同条件 | 同上 `contracts.effective_contract_hash`、`state_binding_sha256` |
| 逐步动作 | **本阶段无动作**（只评分，不 `argmax` 到执行）——`first_argmax_rule` 只作规则审计 |
| 执行结果 | `MODEL_LOAD_*.json`（83 训练叶子逐位恢复、6+6 参数组、`restored_exact`） |
| 资源 | 每槽 `forward_first_wall_s / forward_repeat_wall_s / span_wall_s / F_calls`；相位日志 `mono` 时间线 |
| 尾部身份 | `phases/identity_start_*.json` 与 `phases/identity_end_*.json`：14 个受 watch 文件的起止 SHA |

JAX 首次编译与模型初始化**单独计时并保留**（`WARMUP_*.json` 的 `wall_s`、
`MODEL_LOAD_*.json` 的 `initialization_wall_s`），且**不得**据此声称满足 L3 在线时限。

## 5. 要素⑤ 失败语义与停止规则（fail-closed）

- 任一槽的身份 / 结构 / finite / shape / 逐位重复不一致 ⇒ **该槽 FAILED**，
  之后所有槽 `NOT_ATTEMPTED_AFTER_TERMINATION`，**不重试、不追加、不替换**。
- 超时按**真实监督器截止**处理（槽上限 / 初始化上限 / 批上限），由 `ProcessTree`
  （Windows Job Object）强杀并回收，`tree_active_after` 必须为 0。
- **整轮撤销（V1.16，V1.18 已接入）**：即使 64 行全部 PASS 落盘，只要
  尾部身份记录缺失/损坏、结束身份不符、或行数不足，
  `round_verdict` 就把**全部行降级为 `qualified=False`** 并落盘 `ROUND_REVOKED.json`。
  V1.18 额外要求尾部身份**载荷**覆盖 14 个冻结路径（`TAIL_IDENTITY_PAYLOAD_INVALID`）。
- 失败原件一律保留；`ROUND_VERDICT.json` 只作索引，权威记录是 `ROUND_*.json`。

## 6. 要素⑥ 预算与资源门：**由真实 F 实测证据推导，不编造**

**允许的证据只有一条**：V1.11 那次**真实 F** 的实测样本（22 个已 F 完成状态、44 次前向调用）。
V1.9 的 F-free 采集计时**只作采集资源参考**，**不得**替代 F 初始化或 F 前向预算（已在运行身份里
单独存为 `measured_budget_observed_by_v19_collector`，并显式标注该限制）。

实测（`v111_evidence/scores/*.json` 现场重算）：

| 量 | n | min | median | max |
|---|---|---|---|---|
| 首次前向 `forward_first_wall_s` | 22 | 0.0780 | 0.1250 | 0.2340 |
| 重复前向 `forward_repeat_wall_s` | 22 | 0.0780 | 0.0940 | 0.1410 |
| 每槽两次评分跨度 `span_wall_s` | 22 | 0.1560 | 0.2340 | **0.3440** |
| 模型初始化 | 1 | — | — | 6.3440 |
| 首次编译前向（warm-up） | 1 | — | — | 6.7810 |
| 批墙钟（截至最终结果写入前，22 状态） | 1 | — | — | 19.5620 |

冻结（**不高于** V1.11 已登记的工程上限；本运行**不放宽**任何一项）：

| 上限 | 冻结值 | 推导规则 | 观测余量 |
|---|---|---|---|
| `initialize_compile_s` | **60.0** | V1.11 工程上限，覆盖实测 模型初始化 6.344 + 首次编译 6.781 = 13.125 | **4.57×** |
| `slot_span_s` | **10.0** | V1.11 工程上限，对照实测槽最大 0.344 | **29.07×** |
| `batch_s` | **180.0** | V1.11 工程上限；须容纳 64 状态外推 | 外推 70.03 ⇒ **2.57×** |

外推规则（**标注为外推，不是测量**）：`per_state_obs = 实测批墙钟 / 22`，
`projected = per_state_obs × 64 + 初始化 + warm-up = 0.8892 × 64 + 13.125 = 70.03 s`。
独立复核器**从原始 V1.11 记录重新推导**同一组数字并逐项断言"观测统计落在上限内"
（`recompute_budget_freeze`），因此"冻结值有没有被事后放宽"是可机械检验的。

**调用次数（事前固定，沿用 V1.11 协议）**：

```
warmup_calls      = 1        （用第 1 个状态的输入做一次编译/初始化前向）
calls_per_state   = 2        （同一输入恰好两次，用于逐位重复性）
n_states          = 64
max_forward_calls = 129      = 1 + 64×2
```

**不达 L3 声明**：以上均为**离线接口检查的工程上限**，**不宣称**通过 L3 完整在线 8+2 秒发布门。
执行前已有**一次计时资格运行**的替代物：V1.11 的真实 F 样本本身即单位成本来源；
V1.18 **不**为取预算而新跑任何状态——样本已存在，直接复算。

## 7. 要素⑦ 源码与资产身份（逐条实测，运行前后各查一次）

| 类别 | 冻结值 | 实测 |
|---|---|---|
| F 权重 | `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt` | `f5acf83fa2519020…903a7f5` ✅ |
| F backbone | `MASKCO_code/ckpts/cvrp100.ckpt` | `03615b08a55125a8…` ✅ |
| 标签 | `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED` | 强制保留 |
| backbone 标签 | `BACKBONE_PATH_DOCUMENTED_IN_SCRIPT / TRAINING_EXECUTION_BINDING_UNVERIFIED / HISTORICAL_BACKBONE_SHA_UNVERIFIED` | 强制保留（复核意见已否决提升） |
| 数据 | `data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz` | `66573ddc653c863f…` ✅ |
| profile | `results/o0cc/scale_v2/objective_profile.json` | `63e56212174c9ce3…`，`profile_hash=df1f7a6e…` ✅ |
| 有效合同 | `apply_objective_profile(default_pilot_contract(), profile)` | `edc1b2ea3914ddba…` ✅（基础合同 `32124bfd…`） |
| 合同/协议身份件 | `l1_gates/A1_D3_L1_CONTRACT_PROTOCOL_IDENTITY.json` | `f44e3456db5fac67…` ✅ |
| 状态清单 | `v19_evidence/STATE_MANIFEST_64_V19.json` | `2340cc56c83da49c…` ✅ |
| V1.9 运行时身份 | `v19_evidence/RUNTIME_IDENTITY_V19.json` | `7f1b18613c0c5950…` ✅ |
| 依赖源码 | 14 模块逐文件 SHA（`RUNTIME_IDENTITY_V19.modules`） | `dependency_identity_sha256` 冻结 ✅ |
| 运行期 watch 集 | 10 个代码文件 + ckpt + backbone + npz + profile = **14 个路径** | 起止 SHA 闭合（`identity_start` / `identity_end`）✅ |
| 明确排除 | 并行 `CC_Compare` 提交不属于 D3 身份 | — |

新增/接线的工程文件（本次产出，本身进入 watch 集）：

- `scripts/simulation/d3_l1_v118_run_identity.py`（运行身份 + 预算推导 + 动态缺失项）
- `scripts/simulation/d3_l1_v118_persistence.py`（新目标名持久化 + 有界读取 + 写后校验）
- `scripts/simulation/d3_l1_v118_targets.py`（期望行 / 身份目标 / 行级独立判定）
- `scripts/simulation/d3_l1_v118_forward_runner.py`（`--freeze-identity` / `--supervise` / `--worker` / `--readiness`）
- `scripts/tests/d3_l1_v118_independent_verifier.py`（独立只读复核器 + 自检夹具）
- `scripts/tests/d3_l1_v118_synthetic_worker.py`（合成 scorer worker）
- `scripts/tests/d3_l1_v118_end_to_end_acceptance.py`（9 场景端到端验收）

复用的生产路径（未改语义，`reconcile_rows` 增加可选 `rows_dir` / `row_glob`，
默认值复现历史行为，V1.16 6/6 回归通过）：
`scripts/tests/d3_l1_v116_forward_control.py`、`scripts/tests/d3_l1_v112_phase_journal.py`、
`scripts/simulation/d3_l1_p0c_v18.py`（`ProcessTree`）。

---

## 8. 执行边界（签署后仍受约束）

**签署此预登记授权的内容**：以 `run_id = d3l1-real-f-forward-20261015-v118`
在上述冻结身份下执行**一次** 64 状态真实 F 评分接口检查（最多 129 次前向），
产出只能用于"真实输入下 F 评分接口工程可用性"这一结论。

**签署此预登记不授权**：F−D-Geo 机制比较（L1 正式运行）、E2 训练、L2/L3、
40 日批次、最终测试、任何安装/写入真实状态、任何效用或显著性主张。

**执行一次后即消费**：无论 PASS 或 FAIL 都保留原件，不重跑、不补样、不追加状态；
V1.11 的 42 条未完成状态**禁止拼接**（新运行是独立整轮，不是 V1.11 的续跑）。

## 9. Claim limits

1. 本文件是**规格 + 工程资格**，不是结果；本文件产出时真实 F 前向调用数 = **0**。
2. 通过只支持"真实输入下 F 评分接口工程可用"，**不**支持任何算法优势、学习增益或在线效用。
3. F 的 backbone 训练期身份未证；即使通过，也只能声明"工程验证"一类结论。
4. 复用已用于开发的状态**不是**独立科学验证；L2 必须另用事前登记的新鲜开发日。
5. 工程上限（60 / 10 / 180 s）**不是** L3 在线发布门，**不得**被引用为满足 8+2 秒。
6. I3 正式负结果（配对差 −144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、
   D3 G0 未通过、D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False`
   **全部不变**。
