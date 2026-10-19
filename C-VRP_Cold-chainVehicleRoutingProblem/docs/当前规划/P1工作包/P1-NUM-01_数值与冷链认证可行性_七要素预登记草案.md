# P1-NUM-01 · 数值与冷链认证可行性 · 七要素预登记草案

> **状态：`DRAFT_NOT_AUTHORISED`** · `PREREG_DRAFT_FOR_USER_SIGNATURE` · 2026-10-18
> 基线 HEAD `918d9e1128d00e8fbd45ef485d0898625af51cd5`
> **本文件只交付设计规格。它不授权、也未执行任何实现、训练、批次、比较或安装。**
> 上位：[`科研方法创新主控文档`](../研究设计/科研方法创新主控文档.md)（历史路线 B，仅取其方法学约束）；
> 证据与裁决以 `项目当前状态.md` → `执行进度表.md` 顶部 → `docs/当前规划/决策与审计/` 为准。
> **不改写任何既有裁决。** 历史 `Q3-S3 NO-GO`、`G-TMIN FAIL`、`L1 BLOCKED`、`L2/L3 NOT AUTHORISED`、
> `FINAL-TEST SEALED` 全部原样保留（见 §9）。

---

## 0. 与 Q3-S3 的区别（**本包不得重开已 NO-GO 的原工作包**）

| 维度 | **Q3-S3（已 NO-GO，禁止重开）** | **P1-NUM-01（本包）** |
|---|---|---|
| 被认证的数值对象 | **既有生产执行器**（`coldchain_state.transition_segment`，float64 + 平台 `math.exp`） | **一个新的、独立的数值合同 C′**；生产执行器字节**不得修改** |
| 量化范围 | **对所有分段成立**的外扩界（`f_level_established`） | **不需要**对所有分段成立；只需在**事先冻结的域**上成立且域可判定 |
| 失败根因 | κ（平台 `math.exp` 的一致的相对误差界）不可建立 → `EXP_ERROR_BOUND_NOT_ESTABLISHED` | 把"可否证明平台 libm 精度"从**充分必要条件**降为**若干可选充分条件之一**（§2 的 T1–T5） |
| 结论状态 | **NO-GO，两轮止损用尽，资格扩张终止** | 全新问题；**若三条路线全部不可行，本包也必须给出 NO-GO**（§6） |
| 证据 | `results/q3_s3_float_outward_bound/controls.json`（24 计划 / 14 PROVED_CONDITIONAL_F / 10 NOT_CERTIFIABLE） | **不得复用** S3 的 24 个计划作为通过证据；必须新的冻结计划集 |

### 0.1 为什么"换个问法"是合法的、不是翻案

S3 的**核心引理 S3-IMP** 说的是：任一**非平凡且对所有分段成立**的外扩界，**必然蕴含**关于平台 `math.exp` 的一致相对精度陈述。
本包**不接受**"对所有分段成立"这个前提，因此**不落入** S3-IMP 的约束域。这是**改变问题**，**不是**推翻 S3 的结论。

S3 的两轮止损、`f_level_established=false`、`EXP_ERROR_BOUND_NOT_ESTABLISHED` **一字不改**。

### 0.2 明确禁止事项（本包的红线）

1. **不得**修改 `scripts/coldchain/coldchain_state.py`（`a2ee98dc…`）或 `coldchain_contract.py`（`0fb9e6e8…`）的字节。
2. **不得**把 S3 的 `f_level_established` 由 `false` 改为 `true`，或把任意 S3 行重述为 PASS。
3. **不得**用"实测没出事"替代数值保证 —— S3-N11 已逐条拒绝四类来源：
   `EMPIRICAL_MEASURED` / `FINITE_GRID_CONFIRMED` / `COMMUNITY_CONVENTION` / `IEEE754_INFERENCE`。
4. **不得**在没有用户裁定的能耗预算 `B` 时宣称"完整认证"（§6 前置 P0-2）。

---

## 要素① 可证伪假设

> **H1**：存在一个**独立于 Q3-S3** 的数值合同 C′，使得对**事先冻结**的计划集合 S：
> (a) C′ 的数值保证**不依赖**任何不可核验的平台 libm 断言（满足 §2 T1–T5 之一）；
> (b) 在 C′ 下 **可认证计划率** ≥ 事前登记阈值 `θ_cert`；
> (c) **单计划认证墙钟** ≤ 既有登记的 40 ms 上限（S3 `G-T1` 登记值），**单决策** ≤ 10 s（strict-online 口径）；
> (d) 对 C′ 拒绝的计划，其拒绝码落在**事前冻结的拒绝码表**内（§3.2）。

**否证条件（事前固定，满足任一即 H1 被否证）**：

1. **数值保证不可得**：T1–T5 中**没有一条**能在**不修改生产执行器**的前提下给出可核验保证；
2. **率门未过**：`certified / |S| < θ_cert`；
3. **预算门未过**：中位或最大值任一项超过 40 ms/计划，或触达 10 s/决策；
4. **拒绝码越界**：出现拒绝码表之外的拒绝，或出现"以拒绝规避不可认证"的占比超过 `θ_refuse`；
5. **身份不符**：合同 / 执行器 / profile / 数据哈希与 §8 不符 → **不启动**，记 `BLOCKED`（不是 FAIL，更不是 PASS）。

**H1 不是**"证明生产执行器安全"，**不是**"通过 Q3"，**不是**"取得生产资格"。它只回答"新数值合同这条路走不走得通"。

---

## 要素② 唯一主对照

> **唯一主对照 = 受限域认证器 `C_restrict`**：对**含任意短 gap 的计划一律拒绝**（不进入 F-level 扩宽），
> 其余判据与 C′ **逐条相同**（同一冻结计划集、同一合同参数、同一拒绝码表前 5 项、同一墙钟口径）。

**为什么是它**：C_restrict 是"**不投入任何数值工作也能安全**"的基线。若 C′ 不能显著多于 C_restrict 的可认证集合，
则**新数值合同没有价值** —— 这正是本包要否证的东西。

| 项 | 冻结值 |
|---|---|
| 分母 | 冻结计划集 S 的**全部**计划（**不得**因拒绝而缩小） |
| 主统计量 | 可认证率差 `Δ_rate = rate(C′) − rate(C_restrict)` |
| 不确定度 | 按**实例/日聚类** bootstrap，不对计划视作独立样本 |
| `C_restrict` 的短 gap 判据 | 沿用 S2 已登记口径：判定路径**不含** `math.exp/log/pow`，比较两端为 `Fraction` |
| 强制并列报告臂（非对照） | R-level 结果（S1 的实值括区）作为**上限参照**，**不得**当对照、**不得**称"已认证" |

**共享项（必须机器可检验）**：同一计划集、同一合同、同一 `n_max` 计算方式、同一域常量、同一拒绝码表。

---

## 要素③ 主判据与统计方法

### 3.1 三口径必须分列（沿用本仓既有规定，不得混用）

| 口径 | 定义 | 使用限制 |
|---|---|---|
| `real_level` | R-level 实值括区是否接受 | **不是**认证；**不得**称"已认证" |
| `certified` | C′ 下**完整**通过（数值 + 温度 + 品质 + 能耗 + strict-online） | **本包主口径** |
| `production` | 生产可用 | **本包不涉及**；`PRODUCTION_NOT_QUALIFIED` 不变 |

### 3.2 拒绝码表（事前冻结，禁止事后扩张）

沿用 S1/S3 已登记的拒绝码词汇，本包只**子集化**、**不新增**：
`NONPOSITIVE_HEAT_TRANSFER_OUTSIDE_DOMAIN` / `DRIFT_CONDITION_VIOLATED` / `EMPTY_BRACKET` /
`PREFIX_CLOCK_INCONSISTENT` / `SERVES_UNREVEALED_NODE` / `COMMITTED_SET_MISMATCH` /
`DEPOT_IN_STOP_LIST` / `ZONE_INACTIVE_OUTSIDE_DECLARED_DOMAIN` / `NON_FINITE_*` /
`EXP_ERROR_BOUND_NOT_ESTABLISHED`（**仅允许 C_restrict 使用此码**；C′ 若使用即视为未建立保证）。

### 3.3 阈值：**本文件不填数字**（`USER_DECISION_REQUIRED`）

| 阈值 | 是否可推导 | 依据 |
|---|---|---|
| 单计划认证墙钟 40 ms | ✅ **可用既有登记值** | S3 `G-T1` 登记：`g_t1_seconds=0.04`，实测中位 2.4387 ms / 最大 4.1886 ms（n=60） |
| 单决策 10 s | ✅ **可用既有口径** | `project` 既有 strict-online 决策时钟 |
| **率门 `θ_cert`** | ❌ **不存在已登记值** | S3 的 24 计划集给出 14/24=0.583 仅作**规模参照**，**不得**直接当通过线 |
| **拒绝规避上限 `θ_refuse`** | ❌ **不存在已登记值** | 需用户裁定 |
| **计划集规模 `|S|`** | ❌ 需用户裁定 | S3 用 24；本包不得复用那 24 个 |

> **不得**把任何探索性标准写成正式通过线。上述两处 `❌` 在签署前必须由用户填数，否则本包**不可启动**。

### 3.4 分母保留规则（事前固定）

| 情形 | 处理 |
|---|---|
| C′ 拒绝 | **留在分母**，按拒绝码分类单列 |
| C_restrict 拒绝 | 同上 |
| 认证超时 | **留在分母**，记 `timeout`；**不剔除、不追加运行** |
| 计划集内异常 | **留在分母**，记 `FAILED_EXCEPTION` 并给根因；**不插补** |

---

## 要素④ 数据、参数、种子及允许的次数

| 项 | 冻结内容 |
|---|---|
| 计划集 S | **新冻结**；与 S3 的 24 计划**零交集**（须机械核对计划哈希）；每计划登记 `plan_hash` + 生成码路 + 可见信息指纹 |
| 数据 | `data/m0_scale/dcc_50_r1_edod05_{train,cal,dev_check}_teacher.npz`；`data/baseline/50_node/{train,val,test}` |
| 合同 | `ColdChainContractV2()`，hash `f70c91632f782188d07950832ec03a3bb94a7c561572cbd56edf7b8d76d53fc0`（与 S1/S2/S3 同一合同，**不得**改） |
| profile | `results/o0cc/scale_v2/objective_profile.json` |
| 域常量 | `eps_f_c = 0.001`（S3 已登记）；`Δ*` 与 `Θ_z` 沿用 S1/S3 已证值 |
| `n_max` | **运行时重算** `1 + 2*max_iter + 2`（S3: `max_iter=10000` → 20003），**不得**硬编码结果 |
| 种子 | **新种子**，签署前冻结；**不得**沿用 S3 的 `seed=20261017`（那会使 S3 与 P1 无法区分） |
| 允许次数 | 冻结于要素⑥；一次机会消费规则见 §7 |
| 开发/确认隔离 | 计划集**只作开发**；`20260927`/`20260928` **保持 SEALED** |

---

## 要素⑤ 预期方向和失败条件

**事前声明方向**：基于 S3 row[17] 的 24 计划分布（**14 PROVED_CONDITIONAL_F / 10 NOT_CERTIFIABLE，
且 10 个全部先在 R-level 被拒、`n_rejected_at_float_level_only = 0`**），本包的**预期是**：

- C′ 相对 C_restrict 的**可认证率提升有限**（历史分布显示 S3 的拒绝**全部**发生在实值层，F-level 加宽**从未**是唯一拒绝理由）；
- 因此 **`Δ_rate ≈ 0` 是很有可能的、且有价值的结果**；
- 若如此，结论是"**数值投资不是当前瓶颈**"，而不是"数值问题不存在"。

**必须同时声明**：`EXP_ERROR_BOUND_NOT_ESTABLISHED` 仍然成立；本包**不**改变 S3 的 NO-GO。

**失败条件**：§要素① 的五条否证条件；外加
6. 任何"为通过而换判据/换种子/换计划集" → 该批次**作废**；
7. 任何把 `real_level` 结果当 `certified` 报告的表述 → 该报告**作废**。

---

## 要素⑥ 资源门、时间上限与停止条件

| 门 | 值 | 来源 |
|---|---|---|
| 单计划认证墙钟（中位 + 最大，**两项都报**） | **40 ms** | S3 `G-T1` 登记值 |
| 单决策墙钟 | **10 s** | strict-online 既有口径 |
| 认证器调用 | **计入**同一预算 | 沿用 L2 要素⑥ |
| 总墙钟 / 总计划集工作量 | `USER_DECISION_REQUIRED` | 无登记值 |

**停止条件（fail-closed）**

1. 任一身份项与 §8 不符 → **立即停止，转 `BLOCKED`**，需新身份，不得静默沿用；
2. 首次不可恢复的协议错误 → 停止并**保留原件**，不重跑、不调参；
3. 超时按 `timeout` 记账，**不剔除计划、不追加运行直到有利**；
4. 若在实施前发现 T1–T5 全不可行 → **直接 NO-GO**，**不实施**（这正是本包要提前回答的问题）；
5. 发现任何"改生产执行器即可过关"的诱惑 → **停止**（§0.2 红线 1）。

---

## 要素⑦ 完整源码、数据、合同、权重及运行身份

| 类别 | 冻结内容（**均为既有值，本包不改**） |
|---|---|
| 执行器（**只读**） | `scripts/coldchain/coldchain_state.py` = `a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e`<br>`scripts/coldchain/coldchain_contract.py` = `0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2` |
| Q3 层模块（**只读，不得改**） | S3 `source_sha256` 已登记 7 项，含 `q3_g2_time_identity_experimental.py = c2cf6c64…`、`q3_s1_route_bracket_experimental.py` 等（见 `results/q3_s3_float_outward_bound/controls.json` 的 `source_sha256`） |
| 合同 | `f70c91632f782188d07950832ec03a3bb94a7c561572cbd56edf7b8d76d53fc0` |
| 平台身份 | S3 `platform` 块：CPython `math.exp` = **C 库 exp**（CPython 不实现 exp），`float_info{mant_dig:53, epsilon:2.220446049250313e-16}`；**必须随报告原样复现** |
| 数据 | `data/m0_scale/*` + `data/baseline/*`；哈希清单 `data/baseline/DATA_MANIFEST.sha256`（**已在库**，30/30 本地核验通过） |
| profile | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |
| 认证器 | **必须记录 `certifier_identity`**；且**必须有生产调用者**（见 §6 前置 P0-5） |
| 运行环境 | Python/JAX/Flax 版本、CPU/GPU、`JAX_PLATFORMS` |
| 独立只读验收 | 与源码身份同批产出，由**不导入被测模块**的工具执行 |

---

## 6. Go/No-Go 分叉与**终止前置**

### 6.1 必须先行建立的前置（缺一即终止，不得"先跑再看"）

| # | 前置 | 当前事实 | 缺则后果 |
|---|---|---|---|
| **P0-1** | 数值保证来源：`FORMAL_SPECIFICATION` / `VENDOR_DOCUMENTED` / `CERTIFIED_IMPLEMENTATION` **三选一**；**或**声明一条完全不触发 `math.exp` 的受限域合同 | S3 已登记**三类都拿不到**，四类经验/惯例来源全被拒 | **终止**（这正是 S3 NO-GO 的直接原因，不得绕过） |
| **P0-2** | 能耗预算 `B` 由用户裁定 | `AWAITING_USER_BUDGET_RULING`（C0 预算缺口，10 项待裁） | 无法做"完整认证"；只能做**不含能耗门**的缩减认证，须显式改名 |
| **P0-3** | `horizon` / `max_iter` 的**生产来源** | G-2 判定为 `UNKNOWN`（"认证器的时域参数目前没有生产来源"），且 `G2_QUALIFICATION_PASS` 已被改名 `G2_DIAGNOSTIC_COMPLETE_WITH_ERRATA` | `n_max=20003` 无生产依据 → 预算项不可签 |
| **P0-4** | `t_min` 是否进入认证路径 | G-TMIN **FAIL 永久（不追认）**，生产 `t_min` 恒 0 | 若进入，需**新裁定**；不得沿用旧判定 |
| **P0-5** | 认证器接线 | `compose_certification` **目前没有任何生产调用点**（仅测试调用），已登记 `CERTIFIER_WIRING_GAP` | 认证无生产消费者 → 结论只能是 `DIAGNOSTIC_ONLY` |

### 6.2 三条候选路线与分叉

| 路线 | 内容 | Go 条件 | 成本/风险 |
|---|---|---|---|
| **A · 有理算术闭环** | 认证全在 `Fraction` 里做；判定路径**零超越函数**（S2 已证明可行） | 计划集内**全部**计划可判定；短 gap 数 ≤ 每决策容量 | 低 / 风险：S2 已登记"40 ms 只覆盖约 **37 个短 gap**/决策"；超出即超预算 |
| **B · 双执行器一致性** | 另建**有理参考实现**，证明与生产执行器的差异 ≤ 可计算界 | 差异界可**证明**（不是实测） | 高 / 风险：等价性可能本身不可得，且**不得**改生产执行器 |
| **C · 受限域合同** | 只认证"门后无短 gap"的计划，其余拒绝 | 拒绝率可接受（`θ_refuse` 由用户定） | 低 / 风险：**这就是 C_restrict 本身**，若 C′≡C_restrict 则 H1 无意义 |

**分叉规则（事前固定）**

```
P0-1 不成立                    → NO-GO（terminate，不实施）
P0-2..P0-5 任一不成立          → BLOCKED（不得当 FAIL，不得当 PASS）
P0 全成立 → 实施路线 A
  A 的 Δ_rate 不显著            → 试路线 B（同一计划集、同一判据）
  B 亦不成立                    → 试路线 C；若 C′≡C_restrict → H1 否证 → NO-GO
  预算门在任一阶段不过          → 该路线 NO-GO，记录，不降级为"近似可接受"
```

**止损**：每条路线各**一次**机会；**不补跑、不拼接、不换种子**。两轮止损用尽即终止本包（沿用 Q3 的止损纪律）。

### 6.3 最小可行实验（MVE）

| 项 | 内容 |
|---|---|
| 目的 | **只回答 P0-1 与路线可行性**，不做任何方法比较 |
| 规模 | 计划集 `|S|` 由用户定（**建议量级参照 S3 的 24，但不得复用其内容**） |
| 运行 | **纯 NumPy / 纯 CPU**，无 GPU、无训练、无 rollout |
| 产出 | ① 每条路线"可否在不改执行器的前提下给出保证"的判定 ② 拒绝码分布 ③ 单计划墙钟中位/最大 ④ 若可实施，C′ vs C_restrict 的可认证率差 |
| 明确不做 | 不训练、不比较学习方法、不产生任何生产资格表述 |

---

## 7. 消费规则

- 正式批次一经启动，PASS/FAIL/超时/中途强杀**均消费**该机会；
- **不重跑、不补跑、不拼接**任何历史记录（含 S3 的任何行）；
- 崩溃时保留原件、不静默重启整批；
- 所有结论必须附**拒绝码分布**与**分母**；缺任一即不得报告结论。

---

## 8. 资格阻断（会造成本包无法启动的缺失）

| 缺失 | 现状 | 阻断级别 |
|---|---|---|
| 数值保证来源（P0-1） | 当前平台三类来源均不可得 | **硬阻断** |
| 能耗预算 `B`（P0-2） | `AWAITING_USER_BUDGET_RULING` | **硬阻断**（完整认证） |
| `horizon`/`max_iter` 生产来源（P0-3） | `UNKNOWN` | **硬阻断**（n_max 依据） |
| 认证器生产调用者（P0-5） | `CERTIFIER_WIRING_GAP` | 软阻断（只能出 `DIAGNOSTIC_ONLY`） |
| 真实数据 | `data/` 本机在（2.17 GB），但**不在 git**；本轮已入库 4 个身份清单 | 本机可做；换机需带外提供 |
| 权重 | **不需要**（本包纯 CPU / 纯 NumPy） | 无 |

**注意**：本包的硬阻断**不是在等数据，而是在等裁定与来源**。在 P0-1/P0-2/P0-3 未解决前，
本包**只能停在设计**，不得进入实现。

---

## 9. 继承的不可改写状态（本包不触碰）

`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`（永久，不追认）· **V-10 干净 checkout FAIL** ·
`FLOAT_EXECUTOR_OUTWARD_BOUND_NOT_ESTABLISHED` · `ANALYTIC_REAL_BOUND_CONDITIONAL` ·
`THERMAL_CONTINUOUS_NOT_ESTABLISHED` · `Q3_FORMAL_FREEZE_NOT_AUTHORISED` · **Q3-S3 NO-GO（两轮止损用尽）** ·
`M2_NOT_AUTHORISED` · L1 `BLOCKED` · L2/L3 `NOT_ESTABLISHED` / `NOT_AUTHORISED` ·
`stable_online_gain=False` · `HISTORICAL_BACKBONE_UNVERIFIED` · C0 `AWAITING_USER_BUDGET_RULING` ·
`t_min` 恒 0 · **最终测试种子 `SEALED`** · `PRODUCTION_NOT_QUALIFIED` · B-1…B-9 全开。

## 10. Claim limits

- 本包若 GO，只证明"**某个新的数值合同在一次登记批次内可行**"，**不**证明生产安全、**不**建立 `PRODUCTION_NOT_QUALIFIED` 之外的任何资格、**不**恢复 Q3 的连续温度认证。
- 本包若 NO-GO，等价于"S3 之后，数值路线在当前平台与不动执行器的约束下仍走不通"，**不**等价于"冷链认证不可能"。
- 任何跨平台结果**不得**外推（`math.exp` 的行为是平台属性）。
