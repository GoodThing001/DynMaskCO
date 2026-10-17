# G-3 计划身份与认证—安装边界一致性 · 七要素预声明（2026-10-17）

> **性质：预声明（pre-registration），先于实施登记。本轮只授权隔离规格，不授权实现。**
> 不修改 `action_contract.py` / `d3_geo_adapter.py` / `m1b_immutable_snapshot.py` / 冻结树 / 数据；
> 无 GPU、无训练、无正式 rollout、无最终测试种子；**不自动 commit/push**。
>
> **唯一主问题**：现有 `action_contract.plan_hash()` 未绑定 `anchor_time` 与 `anchor_load`，且把
> SHA256 截短为 16 hex；`m1b_immutable_snapshot.plan_identity_v2()` 虽包含二者，却用 `round(..., 12)`。
> **主反例：认证计划 A 与实际安装计划 B 物理上不同，但 `plan_hash` 相同。**
> 本轮只研究**严格计划身份**与**认证—安装边界一致性**，不扩大到 Q3 数学证明或新路线优化。

---

## 0. 源码身份（执行前登记，全部取**提交 blob**摘要）

| 项 | 值 |
|---|---|
| 基准远端 HEAD | `a71a0c3cede64b15e20f34f7833134d6236bb7a1` |
| `scripts/simulation/action_contract.py`（**被审**） | `9d64273aa3df98cb931c0c8c4b0d0112c134c13e080d3b4f1615b8ae06d2b53b` |
| `scripts/simulation/d3_geo_adapter.py`（**被审**） | `099c3517472d886ece4ea3e929df2ec54d0e459b2a2e2e07da5b658acc07adc5` |
| `scripts/m1b_adapter/m1b_immutable_snapshot.py`（**被审**） | `38acab48996211191c60ca88edc9190c327815381fe2b287ec2d7f8b06432c21` |
| `scripts/simulation/strict_online_env.py`（引用） | `fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6` |
| 缺陷复现（只读，本轮） | `scripts/tests/test_q3_g3_plan_identity_gap.py` |
| 候选（**本轮不创建**） | `scripts/q3_qualification/q3_g3_plan_identity_candidate.py` |

---

## 1. 假设（assumption）

**H-1（`plan_hash` 不绑定时间/载货）**：`action_contract.py:301-309` 只 `h.update()` 了
`vid`、`p.anchor_node`、`repr(tuple(p.suffix))`，且返回 `hexdigest()[:16]`。
而 `VehiclePlan`（`:67-73`）**有** `anchor_time`、`anchor_load`。

**H-2（主反例可实现且材料级）**：同一 `vid/anchor_node/suffix`、
`anchor_time 24.0 → 30.0`、`anchor_load 12.5 → 25.0`，`plan_hash` **完全相同**。
**已实测**：`6c83833a6d143f50` vs `6c83833a6d143f50`（G-03）。

**H-3（第二 helper 更严但仍非无损）**：`plan_identity_v2`（`m1b_immutable_snapshot.py:33-53`）
含四字段，但 `round(x, 12)`；在 24.0 处 `ulp ≈ 3.55e-15`，
该舍入半径 ≈ **141 ulp**，故 1 ulp 漂移被吸收（G-05）。

**H-4（缺陷到达安装边界）**：`d3_geo_adapter.py:151-152` 直接
`from action_contract import plan_hash` 并 `return plan_hash(plans)`，
认证与安装两侧的 `certified_plan_hash` / `installed_hash` 都由它算出（21 处引用，G-06）。

**H-5（"只加字段"不足以构成资格）**：把 `anchor_time`/`anchor_load` 加进哈希只解决**编码**；
若认证方算的是"它以为自己认证的计划"、安装方算的是"它以为在装的东西"，
两侧**读的不是同一个对象**，身份一致仍可能是**不同物理计划的一致假象**。G-3 必须同时规定**唯一来源**。

**H-6（G1/G2/G3 职责必须分开）**：G1 = 时间身份无损；G2 = 执行预算前提的**独立资格**（含
`T_clock_sup`、`n_max`、距离模型域与 k）；G3 = 计划身份 + 认证—安装对象一致。
**身份一致绝不意味着 G2 预算或 Q3 连续温度认证成立。**

---

## 2. 唯一主对照（the single primary control）

> **主对照 = 现有 `action_contract.plan_hash()`。**
> 对每一对被测计划 `(A, B)`，同时计算：
> **(a) 旧身份** `plan_hash(A)`、`plan_hash(B)`；
> **(b) 候选身份** `G3_PLAN_IDENTITY_V1(A)`、`G3_PLAN_IDENTITY_V1(B)`。
> 判据只有一条方向：**凡旧身份判"相同"而 A、B 物理上不同者，候选必须判"不同"**；
> 凡旧身份判"不同"者，候选**不得**判"相同"（不得放松）。

**为什么只有一个主对照**：本轮问题是"旧身份是否单射、是否覆盖完整物理计划"。
引入第二个对照（例如"候选是否更快/更短/更可读"）会把资格问题变成实现品味问题。
G-1 的无损编码（`struct.pack('>d').hex()`）只是**候选实现所复用的编码约定**，不作为对照。

---

## 3. 判据（criteria，全部机械化、可否决）

| # | 判据 | 通过条件 |
|---|---|---|
| **C1**（主） | 主反例被候选分离 | 对 G-03 的主反例对：`plan_hash` 相同（已实测），候选**不同**；且对旧身份能分辨的差异（suffix / anchor_node / 车辆集合）候选**也**分辨 |
| **C2** | 无损编码 | `anchor_time` / `anchor_load` 用位型编码（`struct.pack('>d').hex()` 或等价的二进制规范化），**禁止** `round` / 十进制格式化；1 ulp 漂移必须改变身份 |
| **C3** | null 与非法值 | 缺失字段 ⇒ 独立哨兵；`nan`/`±inf`/`-0.0`/非数值 ⇒ `REFUSED`（沿用 G-1 的拒绝码体系） |
| **C4** | 完整摘要 | SHA256 **完整 64 hex**；截短必须显式登记并提供完整摘要（截短前缀不得单独充当身份） |
| **C5** | 完整有序 suffix | suffix 逐元素、保序、含尾部 `0` 语义；重复节点/别名键必须被拒绝 |
| **C6** | 车辆身份 | `vehicle_id` 与其**来源**（fleet 状态）绑定；重复键 ⇒ `REFUSED: DUPLICATE_VEHICLE_KEY` |
| **C7** | 已执行前缀与承诺 | `served` 前缀 + `committed_next` 序列进身份；跨事件复用 ⇒ `REFUSED` |
| **C8** | 实例 / 事件 / 状态 | `instance_id`、`event_id`、可信状态身份进身份；跨实例 ⇒ `CROSS_INSTANCE_CONFUSION` |
| **C9** | 合同 / 代码 / 单位 | `contract_hash`、`hours_per_time_unit`、被审源码 blob 摘要进身份 |
| **C10** | G-2 字段显式化 | 身份中区分：距离模型类型、`multiplier_ceiling`、**扰动重复次数 k**、**k 的来源**、
单位、预算资格；**不得**仅凭调用方给的 `ceiling` 自称 k 已核验 |
| **C11** | 认证—安装独立复算 | 安装方**独立**读候选计划并**重算**身份，`certified_plan_hash` 仅作交叉核对；不一致 ⇒ `REFUSED` |
| **C12** | 身份 ≠ 资格 | 身份一致但 G-2 未合格时，`PLAN_IDENTITY_CONSISTENT = True` 而 `INSTALLABLE = False`（判据 `G2_BUDGET_NOT_QUALIFIED`） |

---

## 4. 参数与次数 + 负控（实现前固定，不得事后增删）

| 项 | 值 |
|---|---|
| 合成计划对 | **≤ 16 对**（每类负控一对 + 主反例一对） |
| 数据 | **不读** `data/`；合成车队规模 ≤ 3 车 × suffix ≤ 4 节点 |
| 一 ulp 邻域 | `math.nextafter(x, ±inf)`，`anchor_time` / `anchor_load` 各 2 次 |
| 重复次数 | 纯函数，每项 **1 次** |
| 随机种子 | 不使用（全部确定性构造） |

| 编号 | 注入 | 预期 |
|---|---|---|
| **N-1** | 相同 suffix，`anchor_time` 差 **1 ulp** | `IDENTITY_MISMATCH`（旧身份为 `IDENTICAL`，此为**主反例家族**） |
| **N-2** | 相同 suffix，`anchor_load` 不同（差 1 ulp 与差 1e-9 各一次） | `IDENTITY_MISMATCH` |
| **N-3** | 不同 `vehicle_id` | 身份不同 |
| **N-4** | 不同 `instance_id` | `CROSS_INSTANCE_CONFUSION` |
| **N-5** | 不同已执行前缀（移动一个已服务节点） | `EXECUTED_PREFIX_CHANGED` |
| **N-6** | 认证后计划被改（suffix 交换两项） | `POST_CERTIFICATION_STATE_CHANGED` |
| **N-7** | 跨事件复用旧证书（`event_id` 变、其余不变） | `REFUSED: STALE_CERTIFICATE` |
| **N-8** | 重复 `vehicle_id` 键 / 别名键（`0` 与 `0.0` 与 `'0'`） | `REFUSED: DUPLICATE_VEHICLE_KEY` / `AMBIGUOUS_KEY` |
| **N-9** | `anchor_time ∈ {nan, inf, -inf, -0.0, None, 'soon'}` | `NON_FINITE_TIME` / `NEGATIVE_ZERO_TIME` / `NON_NUMERIC_TIME` / `MISSING_TIME` |
| **N-10** | 缺 k / `k` 未声明但距离模型含 `long_tail_perturbation` | `NOT_CERTIFIABLE: PERTURB_MULTIPLICITY_UNDECLARED` |
| **N-11** | 伪造 `multiplier_ceiling`（调用方给 1.0，而 k=2、asym 开启） | `REFUSED: CEILING_NOT_DERIVED_FROM_K` |
| **N-12** | 身份完全一致但 G-2 未合格 | `INSTALLABLE = False`（`G2_BUDGET_NOT_QUALIFIED`） |
| **N-13** | 安装方自行复算与 `certified_plan_hash` 不一致 | `REFUSED: CERTIFIED_INSTALLED_MISMATCH` |
| **N-14** | suffix 含重复节点 / 越界节点 | `REFUSED: SUFFIX_NOT_A_PERMUTATION` |

---

## 5. 预期方向（expected direction）

| 量 | 预期 |
|---|---|
| 主反例 | 旧身份 `IDENTICAL`、候选 `MISMATCH`（G-03 已实测旧身份相同） |
| 旧身份能分辨的三类差异 | 候选**同样**分辨（不得放松） |
| `anchor_time` / `anchor_load` | 1 ulp 即变身份；无任何十进制舍入 |
| 身份长度 | 完整 64 hex（不为"好读"而截短） |
| `k` 的表述 | 存在**结构上界** `k ≤ n_edge_perturb`，但**没有**已认证且足够紧的实用上界 |
| 浮点裕量 | 解析实值上界为 `ANALYTIC_REAL_BOUND_CONDITIONAL`；**浮点外扩上界未建立**（见 §6 与报告 §0） |
| 生产替换 | **不做**；候选隔离，生产件摘要不变 |

---

## 6. 失败条件（任一成立即判 G-3 规格不合格）

| 编号 | 条件 |
|---|---|
| **G3-F1** | 主反例在候选下仍判"相同" |
| **G3-F2** | 候选把旧身份能分辨的任一差异判成"相同"（放松） |
| **G3-F3** | 候选仍使用 `round` / 十进制格式化规范化 `anchor_time` / `anchor_load` |
| **G3-F4** | 身份被截短却未同时提供完整摘要 |
| **G3-F5** | 身份字段没有登记**唯一来源**（允许"多源同值"而不说明谁权威） |
| **G3-F6** | 安装方复算被设计成"信任 sampler 自报的 `certified_plan_hash`" |
| **G3-F7** | 把身份一致写成"G-2 预算合格"或"Q3 连续温度认证成立" |
| **G3-F8** | 规格中出现"解析上界已被证明覆盖 float32 执行器"的表述 |
| **G3-F9** | 修改 `action_contract.py` / `d3_geo_adapter.py` / `m1b_immutable_snapshot.py` / 冻结树 / 数据 |
| **G3-F10** | 自动 commit/push |

---

## 7. 资源门

| 项 | 上限 |
|---|---|
| 运行时间 | 只读探针 **< 30 s**（纯 CPU） |
| 数据读取 | 禁止读 `data/` |
| GPU / 训练 / 正式 rollout | 禁止 |
| 生产件 | 只读；不得修改、不得接线 |
| 网络 | 禁止 |

---

## 8. `G3_PLAN_IDENTITY_V1` 草案（隔离，不实现生产接线）

### 8.1 编码规则

| 规则 | 内容 |
|---|---|
| **E-1** | 时间与载货量：`struct.pack('>d', x).hex()`（位型）+ 源 dtype；**禁止** `round`/`%f`/`Decimal` 量化 |
| **E-2** | 时间必须 `finite ∧ ≥ 0 ∧ 非 -0.0`；载货量 `finite ∧ ≥ 0`；违反 ⇒ `REFUSED` |
| **E-3** | 缺失 ⇒ 哨兵 `'null'`（与 `0.0` 位型不同） |
| **E-4** | 整数（`vehicle_id`、`anchor_node`、suffix 元素、`instance_id`、`event_id`）用十进制字符串 |
| **E-5** | suffix：**保全序**的列表，逐元素编码；**不排序**（顺序即语义） |
| **E-6** | 摘要：`sha256(canonical JSON)` **完整 64 hex**；如需短前缀，**只能**作为显示，必须同时记录完整摘要 |

### 8.2 字段与唯一来源

| 字段 | 唯一来源 | 备注 |
|---|---|---|
| `vehicle_id` 集合 | fleet 状态的 active 车（`action_contract.build_plans` 的输入） | 重复/别名键 ⇒ 拒绝 |
| `anchor_node` | `VehiclePlan.anchor_node` | 与 `current_node`/`committed_next` 的对应关系必须登记 |
| `anchor_time` | `VehiclePlan.anchor_time`（来自 `env` 的 `ready_time`/`clock`） | **位型编码**；来源链须可追 |
| `anchor_load` | `VehiclePlan.anchor_load`（来自冷链状态总量） | 位型编码 |
| `suffix` | `VehiclePlan.suffix`（有序，含尾部 `0` 语义） | 不排序 |
| `served_prefix` / `committed_next` | `served_mask` / 车辆承诺字段 | 冻结前缀 |
| `instance_id` / `event_id` | env | 跨实例/跨事件拒绝 |
| `clock` / `anchor`（冷链） | env 与 `coldchain_updated_time` | **两者分离**（G-1 C7） |
| `contract_hash` / `hours_per_time_unit` | 合同 | 非 1.0 ⇒ `NOT_CERTIFIABLE` |
| 距离模型 `models` | 域声明 | 白名单 |
| `multiplier_ceiling` | **由 k 与模型推导**（`q3_g2d…analytic_multiplier()`） | **不得**由调用方自由给 |
| `perturb_multiplicity k` + **k 来源** | 域声明 | 未声明 ⇒ `PERTURB_MULTIPLICITY_UNDECLARED` |
| 代码身份 | 被审源码 blob 摘要 | 漂移 ⇒ 身份变 |

### 8.3 与 G-1 / G-2 的职责分离

```
G1  time identity     : 无损时间/锚点编码（已隔离交付，26/26）
G2  budget qualification: T_clock_sup、n_max、距离模型域、k、监视器（诊断完成，生产未合格）
G3  plan identity     : 计划对象 + 认证—安装对象一致（本规格）
```

**禁止的合流**：G3 身份一致 ⇒ 不得推出 G2 预算合格；G2 预算合格 ⇒ 不得推出 Q3 连续温度成立。

---

## 9. 独立验证者设计（认证—安装边界）

| 要求 | 设计 |
|---|---|
| 认证方 | 对**真正接受认证的那个对象**计算并回报完整身份（完整 64 hex + 逐字段），不得回报"我以为的计划" |
| 安装方 | **独立**读取候选计划 → **重算**身份 → 与认证方回报比对；`certified_plan_hash` **只**作交叉核对，不作依据 |
| 对象不可变性 | 认证与安装之间计划对象必须不可变（值拷贝 + 只读视图）；可变共享引用 ⇒ `REFUSED: PLAN_MUTATED_BETWEEN` |
| 执行前缀冻结 | 前缀变化 ⇒ `EXECUTED_PREFIX_CHANGED` |
| 状态一致 | 认证后状态变化 ⇒ `POST_CERTIFICATION_STATE_CHANGED`（沿用 `verify_state_unchanged` 的语义，编码换成无损） |
| 拒绝后无安装 | 任何 `REFUSED` ⇒ `n_installs = 0` 且安装回调未被调用 |

**可证明 / 不可证明（预先声明）**：
- **可证明**：两侧身份由**同一规范**、**同一字段来源**、**同一编码**算出，且比对覆盖完整摘要。
- **不可证明（本轮）**：进程外篡改、内存级共享可变对象、跨进程时序；这些**只能**靠不可变性设计与拒绝后无安装来缩小，**不得**声称已证明。

---

## 10. 资格阻断清单（G-3 结束前必须保持"未解除"）

| # | 阻断 | 状态 |
|---|---|---|
| B-1 | `T_clock_sup` 生产来源不存在 | 未解除 |
| B-2 | 距离模型域未签署；k 无实用紧上界 | 未解除 |
| B-3 | 浮点外扩上界未建立（float32 执行器） | 未解除 |
| B-4 | 默认 v1 合同喂不了质量门 | 未解除 |
| B-5 | 生产 `plan_hash` 未替换（本轮只做规格） | 未解除 |
| B-6 | 监视器盲区（安装前抓取引用等） | 未解除 |
| B-7 | V-10 台账不可复现 | 未解除 |
| B-8 | 全部本地证据未入库 | 未解除 |

---

## 11. 状态（不变）

`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`（永久，不追认）、
`THERMAL_CONTINUOUS_NOT_ESTABLISHED`、`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、
L1 `BLOCKED`、`L2/A3_NOT_ESTABLISHED`、`stable_online_gain=False`、
`HISTORICAL_BACKBONE_UNVERIFIED`、C0 `AWAITING_USER_BUDGET_RULING`、`t_min` 恒 0、
最终测试种子 `SEALED`。
