# A1-D3-L1 正式机制比较执行版预登记（七要素）· L1-v2

- 轮次：`A1-D3-L1-MECHANISM-PREREG`（**正式执行版**）
- 状态：`EXECUTION_PREREG_FROZEN / READY_FOR_USER_SIGNATURE`（**本文件不授权、也未执行正式比较**）
- 绑定运行：`run_id = d3l1-l1-two-arm-20261016-l1v2`
- 取代：`A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md`（SHA `2e73cac4787b3ba4092d8cd4546332d2b63514724163cba5e1ba1a19dd82c54e`）
  —— **原件逐字节保留、不改写**。取代理由：V1 §7 登记旧的 **34 条 F-only** watch 集，
  而冻结身份现为 **42 条 L1 watch**；预登记与身份矛盾时**不得**沿用其 SHA 作为签署依据。
- 机器可读清单（本文件 §7 由它生成，二者不可能不一致）：
- 机器可读清单（本文件 §7 由它生成）：`l1_gates/L1_V2_PREREGISTRATION_MANIFEST.json`
  - 清单 sha256（对**不含本行摘要字段**的清单本体计算）：`b42326c4e33416a8daf59dfb789b95d6ced20f22a2eb249d19419c7ca9cf06de`
- 上位（**判据原样继承，不事后修改**）：
  `A1_D3_L1_MECHANISM_DIAGNOSTIC_PREREGISTRATION.md`（唯一研究问题、三判定分离、分叉语义、失败分母）、
  `A1_D3_L1_TWO_ARM_EXECUTION_PREREGISTRATION_V1_18.md`（共享输入单因素、六类反例、可重放验收）
- 前置事实：**V1.22 真实 F 前向已执行并通过（64/64，独立裁决 6/6，独立只读复核 `VERIFIED`）**，
  一次性授权回执恰一条。**本预登记不重跑 F 资格批次。**
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**

---

## 0. L1 的唯一研究问题（原样继承）

> 在**完全相同的合法动作集合**中，F（真实 MaskCO 权重评分）与 D-Geo（`−Δd` 评分）
> 是否产生**不同的可执行修复选择**？

L1 **只**回答"是否存在**可归因的动作分歧**"。L1 的分歧**不得**被追认为增益、显著性、质量或在线效用；
那是 L2/L3，须各自独立授权。

## 1. 要素① 状态来源

| 项 | 冻结内容 |
|---|---|
| 状态集合 | **复用 V1.9 冻结的 64 个 dev_check 状态**，台账**发布顺序**（`event='PUBLISHED'`），不重排 |
| 数量 | `64`，且 `EXPECTED_STATES = 64` 由裁决器强制 |
| 身份 | `state_binding_sha256 = f5eedd1027342496939e7257e64f2d25dd6f7cd8787e731b6aa9dde213eefee1` |
| 前向目标 | `forward_targets_sha256 = 37e23f7567e3391a4a3360203a3d58e77c2e16b32c7e3514c65c8aa15d5ade08` |
| 空池 | **原样保留**并记 `empty_pool`；该状态留在分母 |
| 失败状态 | **全部留在分母**；不替换、不补采、不重跑 |
| 开发用途标注 | 这些状态**已用于开发**，L1 结论**只**支持"在这些状态上两评分器的机制差异"，**不是**独立科学验证；L2 必须另用事前登记的新鲜开发日 |

状态的物化路径与 V1.10 一致（同一 `materialize` 码路），不新写并行实现。

## 2. 要素② 唯一比较（单因素）

```
mask_scope      = B                      # build_pool_B：按 mutable_ids 计划后缀在采样前过滤
eps_rel         = 0.0                    # EPS_D_PRIMARY_REL
eps_abs         = 0.0                    # EPS_D_PRIMARY_ABS
tie_break       = legal_order_first      # F 的历史 first-maximum argmax 语义
F:      s_F(a) = restored MaskCO logits          a_F = first argmax_{a in legal} s_F(a)
D-Geo:  s_D(a) = -incremental_distance(a)        a_D = first argmax_{a in legal} s_D(a)
```

**两臂必须共享（机器可检验）**：初始快照、`P0`、strict-online 可见信息、**B 池**、
**初始有序 legal**（环境枚举器原序）、`mask`、合同、`tie_break`、**可用资源**。
**唯一声明差异** = `scorer_id`（`DECLARED_VARYING_FIELDS = ('scorer_id',)`）。

### 2.1 两臂执行顺序与共享 10 秒预算（**本次新增的事前登记规则**）

两臂**顺序执行**，顺序冻结为 `['F', 'D-Geo']`，且**共享同一个单调计时起点**
（`wall_clock_scope = per_state_both_arms_share_one_10s_allowance`）。

**机械后果**：第二臂的剩余时间 = `10.0 s −（第一臂已花时间）`，
故**第二臂的时间预算取决于第一臂的成本**。这是**仪器属性**，不是任一评分器的属性。

**事前解释规则（不得事后依据结果调整）**：

```
a state whose ONLY inter-arm difference is a `timeout` on the SECOND arm, while the arms ran in the frozen sequential order under the shared per-state allowance, is classified as ORDER_ATTRIBUTABLE.  Its V2 difference is reported but may NOT be read as a scorer mechanism advantage: the second arm was charged for time it did not spend.  A `timeout` on the FIRST arm is never order-attributable.  Every affected state keeps its record and stays in the 64-state denominator; none is dropped, replaced or re-run.
```

- **V1 不受该顺序影响**：`V1` 比较**第一步**，而第一步之前**没有任何评分调用**，
  故此时第二臂花费为 0、任何截止都不可能触发。**V1 在构造上顺序不变**；
  隔离合成反例 O4 用同一状态在两种顺序下复算确认：两臂都产出第一步时，
  **所选动作与首步 legal 指纹完全相同**。
- **V2 可能受该顺序影响**：`V2` 比较**逐臂完成情况**，而这正是共享截止所约束的对象，
  故**第二臂的 `timeout` 可能由顺序造成**。隔离合成反例对（O2/O3）：同一状态、同一掩码、
  同一有序 legal、两臂自身成本与调用上限全部不变，**唯一变化的是哪一臂先跑** ——
  O2 两臂完成，O3 第二臂 `timeout` 且**回退原计划**。
- **顺序还可能使 V1 变为 `NOT_COMPARABLE`**：若第一臂吃掉全部额度，第二臂在**第一次评分调用**
  即被 fail-closed 拒绝，于是**没有第一步可比较**。该情形记为 `NOT_COMPARABLE`
  （含义恰为"两臂无法比较"），**不是**评分器差异；O4 单独记录此情形。
- **禁止的解读**：由执行顺序造成的超时差异**不得**被解释为评分器机制优势；
  该解读由 `classify_order_confound` 机械排除，并在裁决中列出受影响状态。

**最小公平性处理 = 上述事前解释规则 + 机械分类**。**不放宽**任何上限、**不增加**正式运行次数、
**不改动**任一臂的评分器或调用上限。受影响状态**全部保留在 64 状态分母中**，不剔除、不替换、不重跑。

三项机械化检查（沿用 V1.18 协议，不新造）：①两臂收到**同一个 state 对象**；
②state 摘要运行前后三处相等；③首步 legal 指纹两臂相等且等于共享值。

**分叉语义**（原样继承）：两臂首步不同后，后续部分计划与 legal 可自然不同；**只在首步检查身份**，
分叉后各自的 legal 指纹**逐条记录、永不用于判 FAIL**。

## 3. 要素③ 三判定（独立报告，永不合并）

| 判定 | 定义 | 取值 |
|---|---|---|
| **V1 首步动作分歧** | 两臂首步六字段动作是否不同 | `DIVERGENT` / `IDENTICAL` / `NOT_COMPARABLE` / `NOT_APPLICABLE_EMPTY_MASK` / `NOT_ATTEMPTED` |
| **V2 完整修复** | **每臂**是否走完整个掩码且 `failure_reason is None`（**不含认证**）；**并附 `v2_order_attribution`** | 逐臂布尔 + `failure_reason` + 顺序归因 |
| **V3 可认证计划** | **本次选择 `NOT_RUN`**：不提供认证器、不做正式 C0 认证、**不安装任何计划** | `NOT_RUN`（记录"无认证器"这一事实） |

**V3 = NOT_RUN 的直接后果（必须如实表述）**：本次**只允许**讨论
「候选动作分歧」与「**未认证**的完整修复差异」；
**不得**称为"已认证的可执行计划差异"。任何"可执行/可认证"措辞在本轮均不成立。

## 4. 要素④ 日志（逐槽与逐步记录组）

| 记录组 | 必需字段 |
|---|---|
| 状态身份 | `day_id`、`position`、`state_hash`、`clock`、可见信息指纹 |
| 模型身份 | F checkpoint SHA、backbone SHA、训练元数据、`label`（`HISTORICAL_BACKBONE_UNVERIFIED`） |
| 共同条件 | `mask_order`、`mask_sha256`、`mutable_ids`、`protected_ids`、`pool_B`、`contract_hash`、首步 `legal_sha256` |
| 逐步动作 | 两臂 `steps` 六字段动作、`records`（含 `legal_order`、`canonical_keys`、`score_exact_ties`）、**逐步 legal 指纹** |
| **逐步评分对应** | **逐次评分调用**记录：该步 legal 指纹、logits 长度、logits 摘要、dtype ⇒ 使"跨步复用"可被检出 |
| 执行结果 | 每臂 `failure_reason`、`n_steps`、`plan_after`、`plan_hash_after`、`returned_original_plan`、`diag` |
| **顺序归因** | `arm_order`、`wall_budget_source`、`wall_clock_scope`、`v2_order_attribution`（每状态） |
| 资源 | 每臂 `BudgetGate` 快照（调用数、墙钟、是否越限）与 `Budget` 六项计数 |
| 重放 | `replay_digest`：audit / V1 / V2 / V3 / plan_after / plan_hash 去计时后取摘要 |

**JAX 首次编译与模型初始化单独计时并保留**；**不得**据此声称满足 L3 完整 8+2 秒在线门。

## 5. 要素⑤ 失败语义与停止规则（fail-closed，全部留在分母）

- 失败词表沿用 `d3_geo_adapter.FAILURE_REASONS`：`no_legal` / `nan` / `timeout` / `rollback` /
  `cert_fail` / `score_shape` / `score_len` / `geo_len` / `no_certifier`；本轮另加
  `budget_calls`（调用数越限，**fail-closed 回退原计划**）。
- 任一失败臂**返回调用方原计划**；部分计划只作 `diag.partial_plan_at_failure` 证据；**不得安装半修复计划**。
- 超时按 `timeout` 记账并保留；**不得**把失败记成"两臂一致"，**不得**记成"某臂落后"。
- **超时状态的顺序归因**（§2.1）：`v2_order_attribution` 取
  `NOT_ORDER_ATTRIBUTABLE` / `ORDER_ATTRIBUTABLE` / `ORDER_INDETERMINATE`；
  后两者**不得**读作机制差异，但**状态照常留在分母**。
- 空池 ⇒ V1 `NOT_APPLICABLE_EMPTY_MASK`、零步、计划不变，**该状态留在分母**。
- **禁止**：为改善结果重跑、追加状态、剔除失败状态、放宽上限、拼接任何历史记录。

## 6. 要素⑥ 预算与资源门（**由 V1.22 实测推导，含明确缺口**）

### 6.1 实测事实（V1.22，`REAL_F_FORWARD_ADJUDICATION_V122.json`）

| 量 | 值 | 含义 |
|---|---|---|
| 单状态跨度 max | 0.422 s | **仅**一次状态内的**两次首步**评分，**不是**多步 repair 成本 |
| worker 墙钟 | 33.0 s | 128 次评分 + 初始化/编译 |
| 初始化+编译 | 6.688 s | 单独计时 |
| 记录前向数 | 129 | 1 warm-up + 128 |

⇒ 单次 F 调用成本（由墙钟反推）：`(33.0 − 6.688) / 128 ≈ 0.2057 s`；**最坏单次**取
`max(0.422/2, 0.2057) = 0.211 s`。

### 6.2 步数会计（本次冻结的核心）

```
m = 掩码步数 = min(2, |B 池|)          # 与 V1.10 已验收输入物化所用的掩码规则一致
每状态 F 评分调用数 = m                  # 每步一次
每状态 D-Geo 评分调用数 = m              # 每步一次
F 调用上限 = 64 × m = 128
D-Geo 调用上限 = 64 × m = 128
```

**两臂上限完全相同**；**不因 D-Geo 更快而给 F 任何额外时间或调用额度**。

### 6.3 冻结上限（**本次修复未改动任何一条**）

| 上限 | 值 | 推导 |
|---|---|---|
| 每臂每状态评分调用 | `m = 2` | 步数会计（精确量，非估计） |
| 每臂总评分调用 | `128` | `64 × m` |
| 单决策（每状态）墙钟 | **10.0 s** | **沿用既有 D3 单决策口径，不放宽**；两臂**共享**该额度（§2.1） |
| 初始化+编译 | `60.0 s` | 沿用既有工程上限（V1.22 实测 6.688 s） |
| 整批墙钟 | `180.0 s` | 沿用既有工程上限；`64 × 2 × 0.211 s ≈ 27 s`（F 侧）+ D-Geo（向量读取，量级更小） |

### 6.4 **明确的缺口（不得用乘数掩盖，也不得为此开未授权试跑）**

1. **第 2…m 步的单位成本从未实测**：V1.22 每状态**只在首步**评分一次，多步 repair 无直接计时样本。
   由于掩码消耗后合法集变小，**首步成本是每次调用的上界**，但**总调用数与墙钟不因此被确定**。
2. **不得**为填补该缺口而运行任何真实 F 试跑（那需要另行授权）；**本预登记不编造乘数**。
3. 若签署人要求更紧的墙钟上限，可用的**唯一**合法手段是**降低 `m`**（减少步数），而不是猜测单步耗时。
4. **离线机制诊断不得冒称通过在线时限**；既有 D3 的 10 s 单决策口径是**工程上限**，
   **不是** L3 的 8+2 秒发布门。
5. **顺序混杂**（§2.1）的残余不确定性如实记录：第一臂成本与第二臂剩余额度**实际消耗**，
   只有在真实运行中才能观察到；本预登记给出的**不是**对其数值的预测，而是**解读规则**。

## 7. 要素⑦ 源码与资产身份（**导出闭包，逐文件 SHA**）

- **运行期 watch 集 = 42 条**（38 个源码文件
  + 4 项非源码资产），由
  `scripts/data/d3_l1_closure.py` 从 **L1 正式入口**出发按传递本地 import 图导出。
  **V1 §7 登记的 34 条 F-only watch 集已作废**：它冻结了决定 F 前向的代码，
  **一条**决定 L1 比较的代码都没有冻结（`l1_sources_in_closure == []`）。
- 闭包模块自身也是被冻结路径之一（自指必须被覆盖，否则闭包可被改写而不被检出）。
- **源码身份摘要** `source_identity_sha256 = 6139249946497ac1dde4848cda7531df85446667075fb50c012607c275d05d0c`
  —— 对整个 watch 集取一次规范摘要，凭据以此**绑定执行代码**。
- **逐文件 SHA（42 条，完整列表）**：

| 路径（仓库相对） | SHA-256 |
|---|---|
| `C-VRP_Cold-chainVehicleRoutingProblem/data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz` | `66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89` |
| `C-VRP_Cold-chainVehicleRoutingProblem/results/m0_scale/mpre_reinforce_s42_clean/model.ckpt` | `f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5` |
| `C-VRP_Cold-chainVehicleRoutingProblem/results/o0cc/scale_v2/objective_profile.json` | `63e56212174c9ce371b8a5b1fada6a549d7bc47527fdd1b0b13a7bbe90db57e8` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_contract.py` | `0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_state.py` | `a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/coldchain_visible_features.py` | `76f1d1e13a9432153554298565e7ed53e60f5667515a5dd9466505372cb227b0` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/d3_l1_closure.py` | `32f80a20eb3fd12fa3682b9e4b0ca8314450e31f6e97fc60763ca79bf6a30a73` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/d3_l1_v121_dependency_closure.py` | `f3429c202c7623c5151dc9e5751d04b48c4c5984e209fd752c6b04e1638f88b2` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/repair_state.py` | `daf1d8ea6b5b7ace865fc130c8be85879567fa6fc7d7642bc8a1241b28511148` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/cvrptw_utils.py` | `1fd6aa427bd242227963972c9a439dd94c5b6ca3cb5dbcef37bda3f2f315a73f` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/dynmaskco_cc_graph.py` | `0101e7c31ab8c95eefbc0daa9dd41e5a7df5ec4552a69744a8847f4f9bfba93d` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/mpre.py` | `585f0296e31fc98060f393cad023a6c028c70bbbdb8bcc0c6d27f46f03b3e57c` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/mpre_trained.py` | `c19352745e6e246dc855a3ac2b0f7827b188edb829e2104b428ed0a1c0d545f2` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/project_paths.py` | `3afd25befe673c05318489b83ae259971b0da8b894a8efe7b88e1c8ad84f6658` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/action_contract.py` | `9d64273aa3df98cb931c0c8c4b0d0112c134c13e080d3b4f1615b8ae06d2b53b` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/cc_lns_replanner.py` | `ae634f5323595aafda85aaf8e99eb4e240173106d60b2523bb033ae603e22f25` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_geo_adapter.py` | `099c3517472d886ece4ea3e929df2ec54d0e459b2a2e2e07da5b658acc07adc5` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_causal_perturbation.py` | `d11e3a08f7d494595dfef97aa8ef134bfacf23aafd0f1ef09f10ff6caf606d5a` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_l1_formal_run.py` | `284f3cbc32164fc4b8796580e7133df502003fab3d72e6042245d6b18cbd888e` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_p0c_v18.py` | `1d897b8c0d0b65fe09c8230c4e506e157466a24cde157215479b7c10520f50f7` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_real_arm.py` | `af73f788a615a7e8b29a2d6de4781de88ca39f4acba6f34d94a1b01c88fe1672` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_state_collector.py` | `fff9158855ca75c1469025a99432ca9c6eb6369147f5ec42aeb5bf1cf96230c4` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_two_arm_protocol.py` | `1a10e8065d4744df3171cc7a4055657a604c9a8646101e9d546fbe45d87b693a` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v118_authorization.py` | `abe3252afe2be867f86ece46f77675979893e8f699bcfc55822716f92803e30a` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v118_persistence.py` | `3ccbf94a336bec14863542a613419a9071fbf7045526ee9b19a1740252c961f7` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v118_run_identity.py` | `9c9f787fa8e3210a45156dd4608713a0104714f96e4e64cc00e194a096cc8ec3` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v118_targets.py` | `bfff7d14c8f865fa1876e3f742aae75c49a60b7ac41a42f5e865479f5bb37ba0` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v120_execution_context.py` | `71a7a80661ce697a62787e6f8c64b42428963f52e4eb47f1c0e6d4e3bdd4a2dd` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/dynmaskco_cc_context.py` | `4eebb2a51a024a9e881ce7100bbe568219e1afa5ac8c6d33c707924a36ff1685` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/jf1h_repair.py` | `233f8439536a2af676fe389b012b74c4fa8cf458fde7aef226b084342ca089ee` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/joint_fleet.py` | `7dafcd6d82bc7cc602d5353b8fd3afdd790c1fec6b8aefd449988652be707935` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/mpre_policy.py` | `5b03e4cae812fa20785e68ce837be3a615aadb44ab29b7dbdc4b95f104a4a191` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/mpre_replanner.py` | `0f879d85e9ed3e6ce55dfa40b5582e427a4f05e8215670115d50a2c5d9163ea0` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/mtrained_replanner.py` | `8bc177c1da57b3420537609e5f0b7fa69f489754fac88997fb8731ea6202f777` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/recourse_snapshot.py` | `a2c13f75566e383312b369fd8cf9453a41a22c7eeeb8f4cd38021c8baa965537` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/strict_online_env.py` | `fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/visible_state.py` | `710fe75ca1fc08aedf59bf46e263c6de62ae24315e873aa41b2547d42e57e046` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/d3_l1_v110_input_qualification.py` | `a9f4078db0340105db9b1b122c876843f73a28a0218a377250bd4626d27122a1` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/d3_l1_v112_phase_journal.py` | `b89954ad298c0b937e7b93055a2abddcb5c12b5a2282e4b5252edebb54891a2f` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/d3_l1_v116_forward_control.py` | `e082f672147b5cfc3a2ee16e689a03653d94d5665081f352f57397fb0aae8cf4` |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/d3_l1_v18_readonly_audit.py` | `5d03ef0711dff196a1f1e7bbcc80632d4fa5091b53dfbdc10ba731485f130fad` |
| `MASKCO_code/ckpts/cvrp100.ckpt` | `03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db` |

- **非源码资产（4 项，显式根）**：

| 路径（仓库相对） | SHA-256 |
|---|---|
| `C-VRP_Cold-chainVehicleRoutingProblem/data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz` | `66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89` |
| `C-VRP_Cold-chainVehicleRoutingProblem/results/m0_scale/mpre_reinforce_s42_clean/model.ckpt` | `f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5` |
| `C-VRP_Cold-chainVehicleRoutingProblem/results/o0cc/scale_v2/objective_profile.json` | `63e56212174c9ce371b8a5b1fada6a549d7bc47527fdd1b0b13a7bbe90db57e8` |
| `MASKCO_code/ckpts/cvrp100.ckpt` | `03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db` |

- **关键 L1 决策模块（必须在册，缺一即拒绝）**：

- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_l1_formal_run.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_real_arm.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_two_arm_protocol.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_geo_adapter.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/d3_l1_v110_input_qualification.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v118_authorization.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v120_execution_context.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/d3_l1_v116_forward_control.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/d3_l1_v112_phase_journal.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v118_persistence.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v118_targets.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_l1_v118_run_identity.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/d3_l1_v121_dependency_closure.py`
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/d3_l1_closure.py`

- **明确排除**：测试 / 验收 / 证据 / 只读审计工具按 `EXCLUDED_WITH_REASON` **逐条记录理由**后排除；
  并行 `CC_Compare` 提交不属于 D3 身份。
- 第三方库版本以运行身份 `packages` 记录，不以文件 SHA 表达。
- **冻结校验同源**：`d3_l1_closure.cross_check_watch_set` 由身份与冻结门共用；
  闭包任一文件缺失即 `NOT_READY`，**不存在**静默回退到 F-only watch 的分支。
- **消费前实时校验**：监督器对**磁盘实时重算**的 watch 集授权，
  故冻结之后被修改的代码在**消费一次性凭据之前**即被阻断；消费后再校验一次。
- `--build` 时漂移 = **0** 条（无）。

## 8. 执行边界与消费规则

**授权**：以 `run_id = d3l1-l1-two-arm-20261016-l1v2` 在上述冻结身份与预算下执行**一次**
64 状态、最多 `m=2` 步、每臂最多 128 次评分的
**F−D-Geo 同掩码机制比较**（离线、不安装、V3 = `NOT_RUN`）。

**唯一执行命令**：

```
python scripts/simulation/d3_l1_l1_formal_run.py --supervise --out-dir <frozen L1 run directory> --authorization <signed credential> --preregistration research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md
```

**不授权**：E2 训练、L2、L3、40 日批次、最终测试、任何安装/写入真实状态、任何效用或显著性主张、
C0 正式预算裁定。

**消费规则**：PASS / FAIL / 超时 / 中途强杀**均消费**本次唯一机会；不重跑、不追加状态、
不拼接任何历史记录；崩溃时保留原件、不静默重启整批。

## 9. 本次可形成与不可形成的科学结论

**可以形成**：
1. 在完全相同的初始有序 legal 上，F 与 D-Geo **首步动作是否分歧**（含分歧字段与分叉位置）；
2. **每臂**是否产生**完整（未认证）修复**，并**区分**该差异是机制差异还是**顺序/共享额度**所致（§2.1）；
3. 失败 / 超时 / 回滚 / 空池的**分布**，且全部状态留在分母；
4. **顺序归因计数**：`ORDER_ATTRIBUTABLE` / `ORDER_INDETERMINATE` / `NOT_ORDER_ATTRIBUTABLE` 各多少状态。

**不可以形成**：
1. 任一修复是**可认证**的（V3 = `NOT_RUN`）；
2. 任何**决策质量**或**在线效用**差异；
3. 任何**显著性**、**学习贡献**或 **MaskCO** 结论；
4. 关于**历史训练期 backbone** 的任何陈述；
5. 把 `ORDER_ATTRIBUTABLE` 状态的 V2 差异当作**评分器机制优势**。

## 10. Claim limits

1. 本文件是**规格 + 工程资格**；产出时 L1 正式比较次数 = **0**。
2. 复用的 64 状态**曾用于开发**，**不是**独立科学验证；L2 必须另用新鲜开发日。
3. 离线机制诊断**不满足** L3 在线发布门。
4. C0 正式预算**继续作为独立待裁定项**（`NON_BLOCKING_UNRESOLVED`），本轮不假装已解决。
5. V1.22 的 PASS 只证明**评分接口**在真实输入上可用，**不**蕴含任何分歧或增益。
6. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、D3 G0 未通过、
   D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
7. **本文件替代 V1 作为签署对象**；V1 原件、旧签署包与 V2 证据**全部逐字节保留**。
8. **人工批准字段不由任何工具代填**；本文件本身**不是**授权。
