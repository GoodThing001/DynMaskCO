# A1-D3-L1 正式机制比较执行版预登记（七要素）· L1-v1

- 轮次：`A1-D3-L1-MECHANISM-PREREG`（**正式执行版**）
- 状态：`EXECUTION_PREREG_FROZEN / READY_FOR_USER_SIGNATURE`（**本文件不授权、也未执行正式比较**）
- 绑定运行：`run_id = d3l1-l1-two-arm-20261015-l1v1`
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
| 身份 | `STATE_LEDGER.jsonl` 与 `STATE_MANIFEST_64_V19.json` 的 sha256 见 §7；逐状态 `state_hash` |
| 顺序摘要 | `state_binding_sha256`（= 64 条 `(day_id, position, state_hash)` 的规范摘要） |
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
**初始有序 legal**（环境枚举器原序）、`mask`、合同、`tie_break`、**可用资源**（同一 `BudgetGate` 上限）。
**唯一声明差异** = `scorer_id`（`DECLARED_VARYING_FIELDS = ('scorer_id',)`）。

三项机械化检查（沿用 V1.18 协议，不新造）：①两臂收到**同一个 state 对象**；
②state 摘要运行前后三处相等；③首步 legal 指纹两臂相等且等于共享值。

**分叉语义**（原样继承）：两臂首步不同后，后续部分计划与 legal 可自然不同；**只在首步检查身份**，
分叉后各自的 legal 指纹**逐条记录、永不用于判 FAIL**。

## 3. 要素③ 三判定（独立报告，永不合并）

| 判定 | 定义 | 取值 |
|---|---|---|
| **V1 首步动作分歧** | 两臂首步六字段动作是否不同 | `DIVERGENT` / `IDENTICAL` / `NOT_COMPARABLE` / `NOT_APPLICABLE_EMPTY_MASK` |
| **V2 完整修复** | **每臂**是否走完整个掩码且 `failure_reason is None`（**不含认证**） | 逐臂布尔 + `failure_reason` |
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
| 资源 | 每臂 `BudgetGate` 快照（调用数、墙钟、是否越限）与 `Budget` 六项计数 |
| 重放 | `replay_digest`：audit / V1 / V2 / V3 / plan_after / plan_hash 去计时后取摘要 |

**JAX 首次编译与模型初始化单独计时并保留**；**不得**据此声称满足 L3 完整 8+2 秒在线门。

## 5. 要素⑤ 失败语义与停止规则（fail-closed，全部留在分母）

- 失败词表沿用 `d3_geo_adapter.FAILURE_REASONS`：`no_legal` / `nan` / `timeout` / `rollback` /
  `cert_fail` / `score_shape` / `score_len` / `geo_len` / `no_certifier`；本轮另加
  `budget_calls`（调用数越限，**fail-closed 回退原计划**）。
- 任一失败臂**返回调用方原计划**；部分计划只作 `diag.partial_plan_at_failure` 证据；**不得安装半修复计划**。
- 超时按 `timeout` 记账并保留；**不得**把失败记成"两臂一致"，**不得**记成"某臂落后"。
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

### 6.3 冻结上限

| 上限 | 值 | 推导 |
|---|---|---|
| 每臂每状态评分调用 | `m = 2` | 步数会计（精确量，非估计） |
| 每臂总评分调用 | `128` | `64 × m` |
| 单决策（每状态）墙钟 | **10.0 s** | **沿用既有 D3 单决策口径，不放宽**；由 `wall_clock_limit_s` 与 `BudgetGate` 双重把关 |
| 初始化+编译 | `60.0 s` | 沿用既有工程上限（V1.22 实测 6.688 s） |
| 整批墙钟 | `180.0 s` | 沿用既有工程上限；`64 × 2 × 0.211 s ≈ 27 s`（F 侧）+ D-Geo（向量读取，量级更小） |

### 6.4 **明确的缺口（不得用乘数掩盖，也不得为此开未授权试跑）**

1. **第 2…m 步的单位成本从未实测**：V1.22 每状态**只在首步**评分一次，多步 repair 无直接计时样本。
   由于掩码消耗后合法集变小，**首步成本是每次调用的上界**，但**总调用数与墙钟不因此被确定**。
2. **不得**为填补该缺口而运行任何真实 F 试跑（那需要另行授权）；**本预登记不编造乘数**。
3. 若签署人要求更紧的墙钟上限，可用的**唯一**合法手段是**降低 `m`**（减少步数），而不是猜测单步耗时。
4. **离线机制诊断不得冒称通过在线时限**；既有 D3 的 10 s 单决策口径是**工程上限**，
   **不是** L3 的 8+2 秒发布门。

## 7. 要素⑦ 源码与资产身份（**导出闭包，逐文件 SHA**）

- **运行期 watch 集 = 34 条**（30 个源码文件 + 4 项非源码资产），由
  `scripts/data/d3_l1_v121_dependency_closure.py` 从**执行路径种子**导出；L1 自身源码
  （`d3_l1_two_arm_protocol.py`、`d3_geo_adapter.py`、`d3_l1_real_arm.py`、
  `d3_l1_real_arm_acceptance.py`）与状态物化路径**均在其中**。
- 逐文件 SHA 见签署包 `l1_gates/L1_FORMAL_EXECUTION_SIGNING_PACKAGE.json`
  `6_identity_freeze.watch_paths_with_sha256`；同一清单也由 L1 运行时的独立复核器重算。
- **明确排除**：并行 `CC_Compare` 提交不属于 D3 身份；测试/验收/证据工具按
  `EXCLUDED_WITH_REASON` 逐条记录理由后排除。
- 第三方库版本以运行身份 `packages` 记录，不以文件 SHA 表达。

## 8. 执行边界与消费规则

**授权**：以 `run_id = d3l1-l1-two-arm-20261015-l1v1` 在上述冻结身份与预算下执行**一次**
64 状态、最多 `m=2` 步、每臂最多 128 次评分的 **F−D-Geo 同掩码机制比较**（离线、不安装、V3 = `NOT_RUN`）。

**不授权**：E2 训练、L2、L3、40 日批次、最终测试、任何安装/写入真实状态、任何效用或显著性主张、
C0 正式预算裁定。

**消费规则**：PASS / FAIL / 超时 / 中途强杀**均消费**本次唯一机会；不重跑、不追加状态、
不拼接任何历史记录；崩溃时保留原件、不静默重启整批。

## 9. 本次可形成与不可形成的科学结论

**可以形成**：
1. 在完全相同的初始有序 legal 上，F 与 D-Geo **首步动作是否分歧**（含分歧字段与分叉位置）；
2. **每臂**是否产生**完整（未认证）修复**；
3. 失败 / 超时 / 回滚 / 空池的**分布**，且全部状态留在分母。

**不可以形成**：
1. 任一修复是**可认证**的（V3 = `NOT_RUN`）；
2. 任何**决策质量**或**在线效用**差异；
3. 任何**显著性**、**学习贡献**或 **MaskCO** 结论；
4. 关于**历史训练期 backbone** 的任何陈述。

## 10. Claim limits

1. 本文件是**规格 + 工程资格**；产出时 L1 正式比较次数 = **0**。
2. 复用的 64 状态**曾用于开发**，**不是**独立科学验证；L2 必须另用新鲜开发日。
3. 离线机制诊断**不满足** L3 在线发布门。
4. C0 正式预算**继续作为独立待裁定项**（`NON_BLOCKING_UNRESOLVED`），本轮不假装已解决。
5. V1.22 的 PASS 只证明**评分接口**在真实输入上可用，**不**蕴含任何分歧或增益。
6. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、D3 G0 未通过、
   D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
