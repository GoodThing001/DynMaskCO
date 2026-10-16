# DynMaskCO-CC：M1-a 原型实施边界约束（S-15～S-17 增补 + S-12 精确化）（2026-10-16）

**基准**：`d314e94658ce2ddcbe978385f02421792e770aab`（用户已核对该 HEAD）。

**性质**：**设计约束**文件。记录**拟授权**的 M1-a 原型实施要求。
**本条消息不构成授权**，本文件**也不申请、不消费**授权。

**本轮实际动作**：**只**做静态读取与设计增补。
**未**实现原型，**未**运行任何合成测试（含 S-1…S-17 全部 `NOT_RUN`），
**未**加载 F 权重，**未**真实前向，**未**采集状态，**未**训练，**未** rollout，**未**安装计划。

**与既有交件的关系**：不重复撰写、不复审、不覆盖
[M1-a 独立复核与补充](DynMaskCO_CC_M1-a独立复核与补充_2026-10-16.md)（下称"**补充件**"）
及其审查的
[方法学复核与 M1-a 设计](DynMaskCO_CC_方法学复核与M1-a设计_2026-10-16.md)（下称"**设计件**"）。
本文件**只做增补与精确化**，沿用二者的
`REQUIRED` 检查集、I1/I2/I3 不变量、fail-closed 语义与合成验收框架。

---

## 1. S-15：非法计划**不得进入**品质回调

### 1.1 要求（用户原文约束）

使用**合成的未来订单、已服务订单及重复订单**案例。要求：

- legality gate **拒绝**；
- **品质检查调用次数为 0**；
- **不得**"先调用品质实现、再把结果标记为不合格"。

### 1.2 为什么这是**可静态保证**的结构性要求（源码依据）

`compose_certification` 的 legality-first 短路使"0 次调用"在**正常路径**上**构造性成立**：

| 事实 | 位置 |
|---|---|
| `DOWNSTREAM = ('route_feasible', 'energy_budget', 'quality_projection')` | `d3_composed_certifier.py:110` |
| 四项 legality gate 任一失败 ⇒ `_not_run(DOWNSTREAM, …)` | `:136-147` |
| 四项检查与**品质分支**整体位于 `if legality_ok:` 之内 | `:206`…`:283` 的缩进层级 |
| `quality_check_fn` **唯一**调用点 | `:291`（在该 `if` 之内） |
| 全仓库 `quality_check_fn` 生产侧仅此一处 | `:57`（形参）、`:291`（调用）、`:283`（判空） |

⇒ **非法计划到达 `:291` 是不可能的**。

### 1.3 因此 S-15 的**真正**检验对象：不是"是否拒绝"，而是**调用计数与顺序**

本用例**必须**断言**调用次数恰为 0**，而非仅断言"返回 not ok"。
理由：若未来有人在 `:206` 之前插入早调用，或把品质分支移出 `if legality_ok:`，
"结果不合格"仍会通过，但**非法数据已经流过品质实现**——这正是要防的。

**机械检验设计**：

```python
# 设计伪代码；本轮 NOT_RUN。
calls = []
def spy_quality(env_, inst_, clock_, vehicles_, served_mask_, plans_):
    calls.append({'clock': clock_, 'plans': _structural_fingerprint(plans_),
                  'n_orders': _count_orders(plans_)})
    return True, {'probe': 'SPY_ONLY_NEVER_A_CERTIFICATE'}

for case in (FUTURE_ORDER, SERVED_ORDER, DUPLICATE_ORDER):
    rep = compose_certification(<ctx with case>, quality_check_fn=spy_quality)
    assert rep['legality_ok'] is False
    assert rep['installable'] is False
    assert rep['quality']['state'] == QUALITY_CAPABILITY_GAP   # 未被换成 CHECKED_*
    assert len(calls) == 0, 'ILLEGAL_PLAN_REACHED_QUALITY_IMPL'
    assert rep['components']['quality_projection'].get('not_run_illegal_order_set') is True
```

**判定词表（新增，事前固定）**：

| 结果 | 含义 |
|---|---|
| `LEGALITY_FIRST_HOLDS` | 三例均拒绝、调用恰 0、品质状态未被改写 |
| `ILLEGAL_PLAN_REACHED_QUALITY_IMPL` | **任一**用例调用次数 > 0 ⇒ **硬失败**，不得以"结果仍不合格"辩护 |
| `REPORT_SCHEMA_INVALID` | 必需分量缺失或 `ok` 非 bool |

**补充断言**：三例中 `quality` 字段**必须**仍为 `QUALITY_CAPABILITY_GAP`。
若出现 `CHECKED_SATISFIED`/`CHECKED_FAILED`，说明品质分支被**非法执行**过。

---

## 2. S-16：品质回调返回值**严格类型检查**

### 2.1 要求

对以下情形**一律 fail-closed**：

1. `qok` 返回**非布尔值**；
2. **缺失证据**（无实现身份 / 无适用域 / 无合同版本 / 无证明引用）；
3. **身份不符**（实现 SHA / 合同版本与登记不符）;
4. **报告结构错误**（非二元组、非映射、缺必需键）。

**不得**通过 Python truthiness 把非布尔结果解释为 `QUALITY_OK`。

### 2.2 为什么这是**真实**风险（源码依据）

| 事实 | 位置 |
|---|---|
| `qok, qdetail = quality_check_fn(...)` —— **直接解包，无类型检查** | `d3_composed_certifier.py:291` |
| `quality = {'state': QUALITY_OK if qok else QUALITY_FAILED, 'detail': qdetail}` —— **真值判断** | `:292` |
| `blocking_gaps = [] if qok else []` —— **只要 truthy 即清空阻断缺口** | `:293` |
| `installable = bool(ok and not blocking_gaps)` | `:298` |

⇒ **当前实现在类型层面是不设防的**：`qok` 返回 `"no"`（非空字符串）、`[0]`、
`{'state': 'ok'}`、`numpy.bool_(True)`、甚至 `1` 都会被判为**通过**，
并**清空阻断缺口**、使 `installable` 可为 `True`。

这不是假想缺陷：既有测试已证明**任意注入的回调能把 `installable` 翻成 `True`**
（`test_d3_b3ac_gate_refusal.py:591/600/603-605/616`，逐字标注 `PROBE_ONLY_GATE_NOT_A_WALL`）。

### 2.3 严格验证器设计（适配器**必须**先行拦截，不依赖 composer）

```python
# 设计伪代码；本轮 NOT_RUN。
def strict_quality_verdict(result, ctx):
    # 1) 结构：必须恰为二元组
    if not (isinstance(result, tuple) and len(result) == 2):
        return REFUSE('QUALITY_RESULT_NOT_A_PAIR')
    qok, qdetail = result
    # 2) 类型：必须恰为内置 bool（排除 numpy.bool_、int、字符串、容器）
    if type(qok) is not bool:
        return REFUSE('QUALITY_QOK_NOT_BUILTIN_BOOL')
    # 3) 证据：必须为映射且含全部必需键
    if not isinstance(qdetail, dict):
        return REFUSE('QUALITY_DETAIL_NOT_A_MAPPING')
    missing = [k for k in REQUIRED_QUALITY_EVIDENCE if k not in qdetail]
    if missing:
        return REFUSE('QUALITY_EVIDENCE_MISSING')
    # 4) 身份：实现/合同/适用域必须与注册表登记逐字段相等
    if qdetail['impl_sha256'] != ctx.registered_quality_impl_sha256:
        return REFUSE('QUALITY_IMPL_IDENTITY_MISMATCH')
    if qdetail['contract_version'] != ctx.contract_version:
        return REFUSE('QUALITY_CONTRACT_VERSION_MISMATCH')
    if qdetail['applicable_domain_id'] != ctx.registered_domain_id:
        return REFUSE('QUALITY_DOMAIN_IDENTITY_MISMATCH')
    # 5) 域成员性：必须显式声明且为真（不得缺省视为通过）
    if qdetail.get('omega_membership_checked') is not True:
        return REFUSE('QUALITY_OMEGA_MEMBERSHIP_NOT_CHECKED')
    # 6) 仅在 qok 为 True 时才允许 QUALITY_OK
    return ('QUALITY_OK' if qok else REFUSE('QUALITY_CHECKED_FAILED'))
```

`REQUIRED_QUALITY_EVIDENCE`（最小集，实施时冻结）：
`impl_sha256`、`contract_version`、`applicable_domain_id`、
`conservativeness_proof_ref`、`omega_membership_checked`。

**三条硬约束**：

- **T1 无 truthiness**：判定**只能**来自 `type(qok) is bool` 与注册表**逐字段比对**；
  **禁止** `if qok:`、`bool(qok)`、`qok == True`（后者会放过 `1`）。
- **T2 缺省即拒绝**：任何字段**缺失**一律拒绝；**不得**用 `.get(k, default_ok)`。
  （`omega_membership_checked` 尤其**不得**缺省为 `True`。）
- **T3 适配器先行**：严格检查**必须**在适配器内完成，
  **不得**依赖 composer ⇒ 因为 composer **当前不做**这些检查（§2.2）。

**判定词表（新增）**：`QUALITY_RESULT_NOT_A_PAIR` /
`QUALITY_QOK_NOT_BUILTIN_BOOL` / `QUALITY_DETAIL_NOT_A_MAPPING` /
`QUALITY_EVIDENCE_MISSING` / `QUALITY_IMPL_IDENTITY_MISMATCH` /
`QUALITY_CONTRACT_VERSION_MISMATCH` / `QUALITY_DOMAIN_IDENTITY_MISMATCH` /
`QUALITY_OMEGA_MEMBERSHIP_NOT_CHECKED` / `QUALITY_CHECKED_FAILED`。

**S-16 负控用例**（至少 9 个，须各自**独立**拒绝且**均不得**产生 `QUALITY_OK`）：
真值型非布尔 6 例（`"no"` / `[0]` / `{'state':'ok'}` / `1` / `0.0` / `numpy.bool_(True)`）
+ 结构错误 1 例 + 证据缺失 1 例 + 身份不符 1 例。

---

## 3. S-17：适用域 Ω 成员检查与 **strict-online 信息隔离**

### 3.1 要求

1. 使用合成的**域内/域外**上下文验证 **Ω 成员检查**；
2. **未经认证**的品质实现**不得**进入正式注册表；
3. **禁止**品质回调直接读取**决策时刻不可见**的订单数据。

### 3.2 域内/域外检查设计

```python
# 设计伪代码；本轮 NOT_RUN。
# ctx_in  : 已登记 Ω 的成员（合同版本、温区、车队、日范围、horizon 均在册）
# ctx_out_*: 逐项越界 —— 合同版本不符 / 温区不在册 / 车队规模不符 /
#            日范围越界 / horizon 不符 / 订单数越界
cert_in  = compose_certification(<ctx_in>,  quality_check_fn=registered_impl)
for ctx in ctx_out_cases:
    cert_out = compose_certification(<ctx>, quality_check_fn=registered_impl)
    assert cert_out['installable'] is False
    assert 'QUALITY_DOMAIN_MEMBERSHIP_FAILED' in _reasons(cert_out)
assert cert_in['quality']['state'] == QUALITY_CAPABILITY_GAP   # 见 §3.4：仍不可安装
```

**关键设计点**：Ω 是**轨迹集合**，**不得**以"当前状态落在某个温度区间"冒充成员性
（沿用设计件 §5.1 末尾的既有约束）。
成员性检查**必须**在**运行时**执行，且其依据（Ω 版本 + 检查函数身份）写入收据。

### 3.3 注册表门（防止未认证实现进入）

正式注册表**必须**拒绝：

| 拒绝类型 | 条件 |
|---|---|
| `TEST_ONLY_REGISTRY_REFUSED` | 条目标记 `TEST_ONLY` / `PROBE_ONLY` / `SYNTHETIC_ONLY` |
| `UNCERTIFIED_IMPL_REFUSED` | 无保守性证明引用，或证明适用域 ≠ 条目适用域 |
| `NO_OMEGA_DECLARATION_REFUSED` | 条目未声明 Ω 或其检查函数身份 |
| `IMPL_SHA_MISMATCH_REFUSED` | 磁盘实现 SHA ≠ 条目登记 SHA |

⇒ **合成验收中使用的假品质实现，其注册表条目必须被标记 `TEST_ONLY`
并在此门被拒绝**——即"假实现**能**接线"与"假实现**能**产生正式结论"**必须**被分开验证。

### 3.4 🔴 `M1A_SYNTHETIC_ENGINEERING_ACCEPTED` 与品质能力的关系（用户红线）

用户明确要求：合成验收**只能**验证接口与拒绝机制，
**不得**使用假品质实现宣称已经具备真实决策期品质认证能力。

**机械落实（本文件新增硬约束）**：

- 在**无真实已审核品质实现**的情况下，**即使全部合成用例通过**，
  `quality.state` **仍必须**为 `QUALITY_CAPABILITY_GAP`，
  `installable` **仍必须**为 `False`；
- 任何合成正控**不得**以"`installable=True`"作为通过条件；
  唯一例外是**显式标记 `TEST_ONLY` 的接线探针**，其结果**不得**写入任何正式结论；
- ⇒ **`M1A_SYNTHETIC_ENGINEERING_ACCEPTED` 与品质能力完全解耦**：
  前者是**工程**状态，后者是**科学能力**状态。**不得**由前者推出后者。

### 3.5 strict-online 信息隔离：可机械检验的边界

**仓库既有正面先例**（不是新发明）：可见投影原语
`coord_normalize_visible`（定义 `cvrptw_utils.py:15`）与
非可见投影 `coord_normalize` **成对存在**，后者**会泄漏未来节点**——
根 `CLAUDE.md:83` 的硬不变量第 3 条逐字记载：
「use `coord_normalize_visible` (never `coord_normalize`, which leaks future nodes)」。

**因此 S-17 的隔离检查按"最小权限 + 事后审计"双轨设计**：

| 轨 | 机制 |
|---|---|
| **最小权限（事前）** | 回调**不接收** `env`（或只接收**过滤后的只读视图**）；上下文以**显式 allow-list** 传递：`clock`、`inst_idx`、`served_mask`、`visible_mask`、可见订单属性、车辆 CC/cargo 状态、`contract`。**不得**传递原始 `env.reveal_time`、未来订单坐标/需求/温区 |
| **事后审计** | 回调返回的 `qdetail` **必须**列出其读取的订单 ID 集合 `read_order_ids`；适配器比对：任一 ID 满足 `reveal_time[id] > clock + 1e-6` ⇒ 拒绝 `QUALITY_READ_UNREVEALED_ORDER` |
| **负控** | 合成一个**故意读取未揭示订单**的假实现 ⇒ 必须被 `QUALITY_READ_UNREVEALED_ORDER` 拒绝；再合成一个**只读可见订单**的假实现 ⇒ 允许进入严格验证器（但**仍不得**产生 `QUALITY_OK`，见 §3.4） |

**判定词表（新增）**：`QUALITY_READ_UNREVEALED_ORDER`（硬失败）/
`QUALITY_DOMAIN_MEMBERSHIP_FAILED` / `TEST_ONLY_REGISTRY_REFUSED` /
`UNCERTIFIED_IMPL_REFUSED` / `NO_OMEGA_DECLARATION_REFUSED` /
`IMPL_SHA_MISMATCH_REFUSED`。

**红线**：事后审计是**补充**，**不得**以其存在为由放宽事前最小权限。

---

## 4. S-12 精确化：`base_plans` 的来源与别名，**而非**内容差异

### 4.1 用户对原设计的更正（接受）

原 S-12 写作"`base_plans=Pc` ⇒ 必须拒绝"。该表述**过强且会误判**：

- 合法候选**完全可能**与 `P0` 内容相等（例如 mask 重构后收敛回同一计划，
  或选中的候选恰为 KEEP）；
- 因此把"`P0` 与 `Pc` 内容相等"当作作弊证据，会**把合法候选判为违规**。

### 4.2 精确化后的 S-12（替代原表述）

**检验对象 = 来源与对象别名（provenance & aliasing），不是值相等。**

```python
# 设计伪代码；本轮 NOT_RUN。
# P0 来源必须是独立冻结的 P0 工件，且与候选生成路径隔离。
base = ctx.P0_source            # 独立冻结来源（含自身 SHA 与取得路径）
cand = generate_candidate(...)  # 候选生成路径

# (a) 别名检查：绝不允许两者是同一对象或其副本视图
assert base is not cand,                                   'S12_BASE_IS_CANDIDATE_ALIAS'
assert id(base) != id(cand),                               'S12_SAME_OBJECT_ID'
assert _shares_mutable_storage(base, cand) is False,       'S12_SHARED_MUTABLE_STORAGE'
if isinstance(base, np.ndarray) or isinstance(cand, np.ndarray):
    assert not np.shares_memory(base, cand),               'S12_SHARES_MEMORY'

# (b) 来源检查：base 必须来自冻结 P0 工件，其 SHA 与登记一致
assert ctx.P0_provenance == 'FROZEN_P0_ARTIFACT',          'S12_BASE_NOT_FROM_FROZEN_SOURCE'
assert ctx.P0_sha256 == ctx.registered_P0_sha256,          'S12_BASE_SHA_MISMATCH'

# (c) 隔离检查：候选生成路径不得写入 P0
p0_hash_before = _structural_hash(base)
_ = generate_candidate(...)
assert _structural_hash(base) == p0_hash_before,           'S12_CANDIDATE_PATH_MUTATED_P0'

# (d) 【明确不检查】base 与 cand 的内容是否相等 —— 相等是合法的
```

### 4.3 判定词表（新增，事前固定）

| 判定 | 含义 |
|---|---|
| `S12_PROVENANCE_ISOLATED` | 来源为冻结 P0 工件、SHA 一致、无别名、无共享存储、生成路径未改写 P0 |
| `S12_BASE_IS_CANDIDATE_ALIAS` | 同一对象 ⇒ **硬失败** |
| `S12_SHARED_MUTABLE_STORAGE` / `S12_SHARES_MEMORY` | 共享可变存储 ⇒ **硬失败** |
| `S12_BASE_NOT_FROM_FROZEN_SOURCE` | 来源非冻结工件 ⇒ **硬失败** |
| `S12_BASE_SHA_MISMATCH` | 与登记 SHA 不符 ⇒ **硬失败** |
| `S12_CANDIDATE_PATH_MUTATED_P0` | 候选生成改写了 P0 ⇒ **硬失败** |

**正控必须包含**：`P0` 与 `Pc` **内容完全相等**的合法候选 ⇒
**必须**判 `S12_PROVENANCE_ISOLATED`（**通过**），
以证明本用例**不再**把内容相等当违规。

**与既有代码依据的一致性**：composer 对缺失 `base_plans` 记
`repair_plan_unverifiable`（`:176-181`），并在文档中写明
"默认成候选自身会使该检查**空洞化**"（`:73-76` 注释）——
本精确化正是把该注释的**意图**变成可机械检验的来源/别名断言。

---

## 5. 实施与授权边界

### 5.1 实施路径（用户要求）

- **必须**在新的**非冻结扩展路径**完成；
- **不修改** `MASKCO_code/`、L1 冻结树、既有实验结果；
- **不修改**已签署预登记与既有正式裁决；
- 沿用仓库既有约定：扩展路径用 `sys.path.insert`，**不新建 `__init__.py`**；
- 交付走 `tools/git_deliver.py`（保留密钥扫描，**不使用** `--skip-secret-scan`）。

### 5.2 本阶段**禁止**（用户逐条要求，本文件照录）

真实 F 前向；L2/L3 比较；真实两天烟测；新状态采集；在线 rollout；计划安装。

### 5.3 交付物（若原型通过全部隔离合成测试）

代码、测试、**失败词表**、**独立只读复核**、**剩余资格缺口**。

---

## 6. 状态词表（用户指定，本文件**保留不激活**）

若原型通过全部隔离合成测试，状态**最多**记为：

```
M1A_SYNTHETIC_ENGINEERING_ACCEPTED
```

**不得**记录为：`FULL_INSTALLABILITY_ESTABLISHED`、`L2_PASS`、`A3_PASS`。

**本文件对本词表的三条机械约束**：

1. **本轮未达成、未记录、未激活**该状态——它是**待授权执行后**才可能达到的**工程**状态；
2. 该状态**只表示接口与拒绝机制在**隔离合成环境下成立，
   **不表示**品质能力、不表示完整可安装性、不表示任何科学结论；
3. 与 §3.4 一致：即使该状态达成，`quality.state` **仍**须为 `QUALITY_CAPABILITY_GAP`、
   `installable` **仍**须为 `False`。

---

## 7. 本轮边界与保留项

**本轮已做**：静态读取与源码核验；S-15/S-16/S-17 设计增补；S-12 精确化；
实施与授权边界记录；状态词表约束。

**本轮未做**：实现原型；运行任何合成用例（S-1…S-17 **全部 `NOT_RUN`**）；
真实 F 前向；L1 重跑或补跑；L2/L3 正式比较；真实两天烟测；新状态采集；
E2 训练；在线 rollout；计划安装；修改 `MASKCO_code/`、扩展冻结树、冻结数据或既有结果；
使用最终测试种子 `20260927`/`20260928`；`--skip-secret-scan` 绕过。

**保留不变**：`stable_online_gain=False`；`HISTORICAL_BACKBONE_UNVERIFIED`；
I3 负结果（`−144.680 [−191.109, −100.125]`）；D2 `not_qualified`；D3 G0 未通过；
D8 `CAPABILITY_GAP_BLOCKING`；E2 无合格结果；C0 `AWAITING_USER_BUDGET_RULING`；
**L1 正式批次 `BLOCKED`（绝不改写为 PASS）**。

**未闭合资格门（沿用补充件 §6，本轮无变化）**：
G-1 决策期品质认证；G-2 生产调用点；G-3 预算 `B`；G-4 主质量指标五要素；
G-5 共同决策器代次；G-6 `mutable_ids` 强制；G-7 L1 `BLOCKED` 进入资格；
G-8 假回调可翻转 `installable`。

**执行前置**：真正实施原型及运行上述测试**之前，仍需用户另行明确授权**。
**本文件不申请也不消费该授权。**
