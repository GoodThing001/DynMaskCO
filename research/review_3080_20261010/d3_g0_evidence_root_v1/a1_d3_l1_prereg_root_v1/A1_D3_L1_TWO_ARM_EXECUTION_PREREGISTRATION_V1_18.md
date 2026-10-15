# A1-D3-L1 两臂机制比较执行版预登记（七要素）· V1.18-L1

- 轮次：`A1-D3-L1-MECHANISM-PREREG`
- 状态：`EXECUTION_PREREG_FROZEN / ARMS_SCORING_BLOCKED_ON_V1_18`（**本文件不授权、也未执行任何真实 F−D-Geo 比较**）
- 上位于：`A1_D3_L1_MECHANISM_DIAGNOSTIC_PREREGISTRATION.md`（七要素原文；**不改写其任何判据**）、
  `A1_D3_L1_STATE_RECOLLECTION_PREREGISTRATION.md`（要素① A2 分支）、
  `A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_18.md`（F 侧前置）
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**

---

## 0. 本文件闭合的是"要素①可执行 + 两臂协议可执行"，不是比较结果

V1.17 的 L1 状态是「**仅 L1-1 共享输入可构造性通过；两臂协议与六个合成反例未实现**」。
本轮把两臂协议**实现并验收**（10/10 + 独立只读复核 17/17），但**F 臂无法评分**：真实 F 前向
是唯一待签署项（见 V1.18 预登记）。因此：

- **工程协议：已闭合**（§2–§7 全部有可执行证据）；
- **两臂正式比较：仍 `BLOCKED`**，阻断项是 **V1.18 真实 F 前向未执行**，不是协议缺失；
- 本文件**不**把任何合成结果写成 F 的行为。

## 1. 要素① 状态来源：**A2 分支已落地，且已绑定真实冻结状态**

- 载体：V1.9 一次性 F-free 采集的 **64 个已发布状态**（`v19_evidence/formal_collection/`），
  按 `STATE_LEDGER.jsonl` **发布顺序**冻结、逐状态 `state_hash`、与 manifest 零分歧
  （V1.18 独立复核器 `V1-3` 二次推导一致，`state_binding_sha256 = f5eedd10…`）。
- 输入张量：复用 V1.10 已验收的 64 个 `original` 输入（`FORWARD_TARGETS`，
  `targets_sha256 = 37e23f7567e3391a…`）。
- 状态物化：**复用 `d3_l1_v110_input_qualification.materialize`**（同一已验收码路），
  不新写一条并行实现；因此"两臂的 P0/车辆/掩码/池/合法序"与 V1.10 审计过的对象同源。
- `previously_used_for_development = true`：这些状态**只支持机制差异诊断**，
  **不是**独立科学验证；L2 必须另用事前登记的新鲜开发日。
- `empty_pool`：本状态集合中 **B 池为空是真实存在的**（V1.9 只读覆盖已记录 pre-plan 尾段池为空）；
  协议要求空池**原样入分母**，不得剔除状态（见 §5、C2）。

## 2. 要素② 唯一比较（单因素）：**机器可检验，不是声明**

```
mask_scope      = B                    # d3_geo_adapter.build_pool_B，池在采样前按 mutable_ids 计划后缀过滤
eps_rel         = 0.0                  # EPS_D_PRIMARY_REL
eps_abs         = 0.0                  # EPS_D_PRIMARY_ABS
tie_break       = legal_order_first    # TIE_BREAK_LEGAL_ORDER；canonical_key 仅作审计列
F:      s_F(a) = restored MaskCO logits           a_F = first argmax_{a in legal} s_F(a)
D-Geo:  s_D(a) = -incremental_distance(a)         a_D = first argmax_{a in legal} s_D(a)
```

单因素由 **`assert_single_factor` 的机械化三重检查**保证（`d3_l1_two_arm_protocol.py`）：

1. **同一对象**：两臂收到的是**同一个 state 字典**（同一批 numpy 张量引用），不是两份拷贝；
2. **状态摘要前后一致**：每臂运行前后各算一次 `state_sha256`，必须三处相等
   （`shared_key_before == 臂前 == 臂后 == shared_key`）——任何一臂原地改写状态都会被抓住；
3. **唯一声明差异**：`DECLARED_VARYING_FIELDS = ('scorer_id',)`，只有评分函数允许不同。

**共享输入身份**（每次运行都落盘）：
`shared_input_sha256`（state_identity + common_conditions + legal_identity + tensor_identity +
budget + contract + certifier）、`legal_sha256`（**有序**合法候选逐项指纹）、
`mask_sha256`、`mutable_ids`、`protected_ids`、`contract_hash`、每张量 `array_digest`
（dtype + shape + 原始字节 sha256，**从不容差比较**）。

**分叉语义**（用户裁定，原样保留）：两臂一旦首步不同，后续部分计划不同、下一步 `legal`
可能自然不同 —— 这是策略作用的结果，**不属输入身份不匹配**。机械可比性**只在首步**检查：
`legal_first_step_sha256` 必须两臂相等且等于共享 `legal_sha256`；分叉后各自记录自己的
legal 指纹（`legal_trace`，每步一条），**永不用于判 FAIL**。

## 3. 要素③ 机制判定：**三个独立判定，永不合并**

| 判定 | 定义 | 字段 |
|---|---|---|
| **V1 `first_step_divergence`** | 两臂首步动作（**六字段**：customer / vehicle_id / slot_kind / slot_anchor / position / predecessor / successor）是否不同 | `verdict ∈ {DIVERGENT, IDENTICAL, NOT_COMPARABLE, NOT_APPLICABLE_EMPTY_MASK}`、`diverging_fields`、`first_divergence_position`（只可能为 0） |
| **V2 `full_repair_completed`** | **每臂**：是否走完整个掩码且 `failure_reason is None`（**不含**认证） | `{arm: {full_repair_completed, failure_reason, n_steps, returned_original_plan}}` |
| **V3 `certified_executable_plan`** | **每臂**：是否存在**通过认证**的完整计划；未提供认证器时**记 `NOT_RUN`**，既非 True 也非 False | `{arm: {certified_executable_plan, certified_plan_hash, certifier}}` |

典型反例必须如实分开记录：**两臂首步不同，但某臂在第二步回退** ⇒ 只能记为**首步分歧诊断**，
**不得**算作"F 成功生成了不同的可执行完整计划"。V1 的 `NOT_COMPARABLE` 与
`NOT_APPLICABLE_EMPTY_MASK` 都是**独立取值**，都**不得**被读作"两臂一致"。

L1 **不要求**统计 CI 或效用胜出门槛；L1 结果**不得**被追认为 L2 显著性证据。

## 4. 要素④ 日志：记录组与字段（全部落盘，可第三方重判）

| 记录组 | 实际字段 / 工件 |
|---|---|
| 状态身份 | `state_identity`：`inst_idx`、`day`、`clock`、`state_hash`、`reveal_time`、`visible_fingerprint`、`n_visible` |
| 模型身份 | `arm_identities`：每臂 `scorer_id`、`label`、`callable`（module.qualname）；F 的 checkpoint / backbone SHA 见 V1.18 运行身份 |
| 共同条件 | `mask_order` / `mask_sha256` / `mutable_ids` / `protected_ids` / `pool_B` + `pool_B_rule` / `capacity` / `tw_max` / `num_vehicles` / `n_legal` / `legal_sha256` / `tie_break` / `eps_*` / `selector_*` |
| 逐步动作 | 每臂 `records`：`chosen`、`customer`、`vehicle_id`、`incremental_distance`、`legal_order`、`canonical_keys`、`distance_ties`、`score_exact_ties`、`selection.rule`；每臂 `steps`：完整**六字段**动作 + 增量距离；每臂 `legal_trace`：逐步 `remaining` / `n_legal` / `legal_sha256` |
| 执行结果 | 每臂 `failure_reason`、`n_steps`、`plan_after`、`plan_hash_after`、`returned_original_plan`、`certified` / `certified_plan_hash` / `certifier`、`diag`（含 `score_contract` / `geo_contract` / `rollback` / `partial_plan_at_failure`） |
| 资源 | 每臂 `budget.snapshot()`（`n_actions_enumerated` / `n_actions_scored` / `n_candidate_plans_certified` / `n_action_applications` / `n_solver_calls` / `wall_seconds`）、`wall_clock_limit_s` |
| 重放 | `replay_digest`：把 audit / V1 / V2 / V3 / plan_after / plan_hash 去掉计时后取摘要；同输入必须复现同一 digest |

JAX 首次编译与模型初始化**单独计时并保留**；L1 **不得**据此声称满足 L3 完整 8+2 秒在线门。

## 5. 要素⑤ 失败语义与停止规则（fail-closed，全部保留在完整分母中）

失败词表沿用 `d3_geo_adapter.FAILURE_REASONS`：
`no_legal` / `nan` / `timeout` / `rollback` / `cert_fail` / `score_shape` / `score_len` /
`geo_len` / `no_certifier`。规则：

- **任一失败臂返回调用方原计划**（`returned_original_plan = True`），部分计划只作
  `diag.partial_plan_at_failure` 证据；**不得**安装半修复计划；
- 失败**不得**被记成"两臂一致"，**不得**记成"某臂落后"；
- 超时以 `timeout` + `diag.timed_out` 表示，**不剔除状态、不追加运行直到出现有利结果**；
- 空池（`mask_order == []`）⇒ V1 `NOT_APPLICABLE_EMPTY_MASK`、零步、计划不变、
  `legal_trace` 为空；该状态**原样留在分母**；
- 认证失败 ⇒ V3 该臂 False + `failure_reason='cert_fail'`；未提供认证器 ⇒ V3 `NOT_RUN`；
- 评分器抛异常 ⇒ 协议路径**向上抛出**（不吞），**事务路径**
  （`run_repair_transaction`）把它转成 `exception:<Type>` 的**回滚**，安装次数必须为 **0**。

### 六类合成反例（用户要求）与实际覆盖

| 用户要求的反例 | 场景 | 结果 |
|---|---|---|
| 空池 | C2（`mask_order == []`） | PASS：`NOT_APPLICABLE_EMPTY_MASK`、零步、计划不变 |
| `score_len` | C3（评分向量短 1） | PASS：`score_len` 干净拒绝，绝不前缀 argmax |
| `score_shape` | C4（2-D 评分向量） | PASS：`score_shape` 干净拒绝，绝不 flatten |
| 分叉后 legal | C5（强制首步分歧） | PASS：首步 legal 两臂相等，分叉后 legal 指纹记录但**不比较** |
| 认证失败 | C6（恒拒认证器） | PASS：`cert_fail`、V3 `any_certified=False`、原计划返回、认证确实被调用 2 次 |
| 超时 | C7（评分器超墙钟上限） | PASS：`timeout`、`timed_out=True`、原计划返回、V2 False |
| （额外）评分器异常 | C8 | PASS：协议路径抛出；事务路径 `exception:ValueError` + 回滚 + **0 次安装** |
| （额外）`no_legal` | C2b | **`NOT_EXERCISED`**：在全部 64 个冻结状态中对每个可见订单都至少存在 1 个可行插入（`states_probed=64`），该分支在本数据集上**不可自然触发**；如实记为未覆盖，**不冒充通过** |

证据：`l1_gates/A1_D3_L1_TWO_ARM_PROTOCOL_ACCEPTANCE.json`（10/10）、
`l1_gates/A1_D3_L1_TWO_ARM_TRACES.json`（完整两臂报告）、
`l1_gates/A1_D3_L1_TWO_ARM_INDEPENDENT_AUDIT.json`（17/17）。

**可重放验收器**：`scripts/tests/d3_l1_two_arm_acceptance.py`（确定性；`replay_digest` 在 C1 中
对同一输入重跑一次并要求完全相等）。
**独立只读审计**：`scripts/tests/d3_l1_two_arm_readonly_audit.py` **不导入**协议、不导入
geo_adapter、不导入验收器，只读 trace JSON，**独立重算 V1/V2/V3** 并与记录值逐项对拍
（A2/A3/A4），并独立核验单因素（A5）、分叉纪律（A6）、空池纪律（A7）、无科学主张（A8）。

## 6. 要素⑥ 预算与资源门：**两臂等预算，先冻结后运行**

- 每臂使用**同一个** `wall_clock_limit_s`；`Budget` 逐项分开计数
  （枚举 / 评分 / 认证 / 动作应用 / 求解器调用 / 墙钟），因为**单位成本不等**；
- `n_solver_calls` 由装配层显式上报（`report_solver_calls`），适配器本身不调用求解器，
  不得把"动作应用次数"冒充"求解器调用次数"；
- **正式比较的运行预算尚未冻结**，因为正式比较依赖 V1.18 真实 F 前向完成后的真实 F 评分耗时；
  届时以**同样的证据规则**冻结（实测 F 单位成本 + 明确推导规则 + 余量报告），
  **不得**引用 V1.9 F-free 采集计时，**不得**事后放宽；
- 既有 D3 的 10.0 s 单决策口径**不得**在结果产生后临时放宽；离线加载诊断**不得**宣称通过完整在线时限。

## 7. 要素⑦ 源码与资产身份

| 类别 | 冻结值 |
|---|---|
| 状态清单 | `v19_evidence/STATE_MANIFEST_64_V19.json` sha `2340cc56c83da49c…` |
| 状态台账 | `v19_evidence/formal_collection/STATE_LEDGER.jsonl` sha `4de96c18ed5a6f2d…` |
| 运行期源码 watch 集 | V1.18 的 10 个代码文件 + ckpt + backbone + npz + profile（14 路径，起止 SHA 闭合） |
| 协议实现 | `scripts/simulation/d3_l1_two_arm_protocol.py` |
| D-Geo 适配器 | `scripts/simulation/d3_geo_adapter.py`（`SELECTOR_GEO` / `eps=0` / `legal_order_first`） |
| 状态物化 | `scripts/tests/d3_l1_v110_input_qualification.py`（`materialize`，V1.10 已验收码路） |
| 验收器 | `scripts/tests/d3_l1_two_arm_acceptance.py` |
| 独立审计 | `scripts/tests/d3_l1_two_arm_readonly_audit.py` |
| 数据 | `data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz` sha `66573ddc653c863f…` |
| 合同 | 有效合同 `edc1b2ea3914ddba…`（基础 `32124bfd…`） |
| 明确排除 | 并行 `CC_Compare` 提交不属于 D3 身份 |

四份工件的 SHA 已写进验收 JSON（`tool_sha256` / `protocol_sha256` / `geo_adapter_sha256` /
`v110_materializer_sha256` / `trace_sha256`），并由独立审计的 A9 交叉核验
"验收摘要指向的 trace 摘要 == 实际 trace 摘要"。

## 8. 执行前置门（全部满足才可申请 L1 正式运行）

- [x] 两臂共享输入协议实现且单因素机器可检验（C1 + 独立审计 A5）
- [x] 三判定分离实现（C1/C5/C6 + 独立审计 A2/A3/A4）
- [x] 六类合成反例实现（C2–C7；`no_legal` 记 `NOT_EXERCISED` 并如实说明原因）
- [x] 可重放验收器（`replay_digest` 相等）
- [x] 独立只读审计 17/17，且**不导入**被测协议
- [ ] **V1.18 真实 F 前向完成并通过**（唯一硬前置；见 V1.18 预登记）
- [ ] 正式比较的调用次数与墙钟预算按 §6 冻结
- [ ] 状态重采集规格如适用（本 L1 用 A2 分支的 V1.9 资产，已有 `state_hash` / 可见信息指纹）
- [ ] backbone 身份标签用户裁定（当前保持 `HISTORICAL_BACKBONE_SHA_UNVERIFIED`）
- [ ] 用户签署本预登记

## 9. Claim limits

1. 本文件是**规格 + 工程资格**；本文件产出时真实 F−D-Geo 比较次数 = **0**，真实 F 前向 = **0**。
2. 两臂验收中的 F 臂是**合成确定性 logits**（`SYNTHETIC_F_NOT_REAL_F`），
   **任何**合成分歧/一致都**不是** F 的行为，**不得**被引用为 F 的机制结论。
3. L1 只输出机制差异清单；**不是**算法优势、学习增益或在线效用；首步分歧**不等于**不同的可执行计划。
4. 复用已用于开发的状态**不是**独立科学验证；L2 必须另用事前登记的新鲜开发日。
5. `no_legal` 一类分支在本数据集上不可自然触发，已如实记为 `NOT_EXERCISED`。
6. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、D3 G0 未通过、
   D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
