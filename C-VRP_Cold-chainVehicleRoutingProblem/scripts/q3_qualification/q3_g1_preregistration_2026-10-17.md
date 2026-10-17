# G-1 精确时间身份 · 七要素预声明与隔离候选规格（2026-10-17）

> **性质：预声明（pre-registration），先于实施登记。本文件不含任何 G-1 实现。**
> G-1 只在**隔离候选 + 轻量负控**范围内获准启动；**不授权生产接线**、不改生产认证器与安装入口。
> 本轮（G-2 勘误轮）只做到"规格 + 编码可行性探针"，真正的候选留到下一轮。
> 不训练、无 GPU、无正式 rollout、不碰最终测试种子；**不自动 commit/push**。

---

## 0. 代码身份（执行前登记，全部取**提交 blob**摘要）

| 项 | 值 |
|---|---|
| 基准 HEAD | `a71a0c3cede64b15e20f34f7833134d6236bb7a1` |
| `scripts/m1b_adapter/m1b_immutable_snapshot.py`（**被替换对象**） | `38acab48996211191c60ca88edc9190c327815381fe2b287ec2d7f8b06432c21` |
| `scripts/m1b_adapter/m1b_q2_certifier.py` | `7674205b34db5fc5ec3acf226acfc972158dcdd3a0b8c61bb2c5ea607a0e1a97` |
| `scripts/coldchain/coldchain_state.py` | `a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e` |
| `scripts/coldchain/coldchain_contract.py` | `0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2` |
| `scripts/simulation/strict_online_env.py` | `fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6` |
| G-2 已获资格的时间预算 | `n_max = 20003`（`max_iter` 单界，见 G-2 勘误 E-03）；`T_clock_sup = horizon + travel_bound`（E-01/E-02） |
| 隔离候选（**待创建，本轮不创建**） | `scripts/q3_qualification/q3_g1_time_identity_candidate.py` |
| 测试（**待创建**） | `scripts/tests/test_q3_g1_time_identity.py` |

---

## 1. 假设（assumption）

**H-1（当前身份是"过滤器"，不是"规范形式"）**：`snapshot_state_identity`
（`m1b_immutable_snapshot.py:176`）用 `round(x, 9)` 规范化四个字段
（`ready_time`、`current_load`、`committed_finish`、`return_finish`）。
`round(x,9)` **不是单射**：在 24.0 附近，两个相差 1e-12 与 2e-12 的 double **映射到同一个字符串**。
**已实测（G-1 编码探针 P-01/P-02）**：该窗口相当于 **281474 ulp**，
即当前身份对"1 ulp 漂移"这类告警**完全失明**。

**H-2（可无损编码存在且平凡）**：`struct.pack('>d', x).hex()` 对大端 IEEE-754 位型做精确编码，
`float.fromhex` / `struct.unpack` 均可逐位还原；`-0.0`、`nan`、`±inf` 都有不同的位型可被**检出**。
**已实测（P-03/P-04/P-05/P-08）**：8 个手选样本 + 1000 点全域采样全部逐位往返。

**H-3（"同状态等价"与"1 ulp 分离"必须同时成立）**：好的身份既不能**漏报**
（1 ulp 漂移必须改变摘要，P-06 已证可行），也不能**误报**
（同一状态的两次重建/序列化必须给出同一摘要）。二者缺一，身份就不合格。

**H-4（`-0.0` 是策略问题，不是编码问题）**：`-0.0 == 0.0` 数值相等但位型不同。
`repr(round(-0.0, 9))` **保留符号**（我初稿断言它会塌缩，实测**不成立**，见 P-04 的更正）。
故身份必须**显式规定**"时间字段拒绝 `-0.0`"，而不能指望哈希或比较自动发现。

**H-5（时间身份必须绑定 G-2 已获资格的预算）**：`n_max = 20003` 与 `T_clock_sup` 是 G-2（含勘误）
**有条件资格**的产物；G-1 若自带一套或沿用旧 `max_iter`，则两份身份互相矛盾。

**H-6（生产替换必须隔离）**：G-1 的候选**不得**直接改 `m1b_immutable_snapshot.py`；
生产函数保持字节不变，候选以**独立模块 + 对照**形式存在，直到单独授权接线。

---

## 2. 唯一主对照（the single primary control）

> **主对照 = 同状态双表示对 / 一 ulp 邻域对（one pair, two obligations）。**
> 对每个登记状态 `S`：
> **(a) 不变性** `identity(rebuild(S)) == identity(S)`（不得误报）；
> **(b) 分离性** 对任一绑定字段取 `nextafter(x, +∞)` 得到 `S'`，必须
> `identity(S') != identity(S)`（不得漏报）。

**为什么只有一个主对照**：本轮判据是"身份是否是**单射且稳定**的规范形式"。这两个方向是同一张
判定表的正反面；任何第三个对照（例如"身份是否更快/更短"）都会把资格问题变成工程口味问题。
编码探针（P-01…P-08）只是**前提可行性**，不得替代主对照。

---

## 3. 判据（criteria，全部机械化、可否决）

| # | 判据 | 通过条件 |
|---|---|---|
| **C1**（主） | 单射且稳定 | 主对照 (a)(b) 对全部登记状态成立；`n_false_positive = n_false_negative = 0` |
| **C2** | 无损 | 任一绑定字段经 encode→decode 逐位相等（含 `0.30000000000000004`、`24.000001`） |
| **C3** | `round(x,9)` 被弃用 | 候选**不**调用 `round`/`format`/`%.9f` 作为规范形式；且候选通过"1 ulp 分离"这一当前实现无法通过的用例 |
| **C4** | 缺失值不被当作 0 | `None`（无 `committed_finish` / `return_finish`）必须编码为独立哨兵，与 `0.0` 区分 |
| **C5** | 单位绑定 | `hours_per_time_unit` / `distance_km_per_unit` 进摘要；改单位 ⇒ 摘要变 ⇒ `REFUSED` |
| **C6** | 合同身份绑定 | `contract_hash` 进摘要；伪合同配旧身份 ⇒ `REFUSED` |
| **C7** | 事件时钟绑定 | 事件 `clock` 与冷链**锚点** `coldchain_updated_time` 都进摘要（两者可不等，必须分开记） |
| **C8** | 时间上界绑定 | `T_clock_sup` 与 `n_max` 进摘要；**缺失即 `NOT_CERTIFIABLE`**，不得回落到 `horizon` 或 `max_iter` |
| **C9** | 实例/计划绑定 | `instance_id`、`event_id`、已执行前缀（`served` + `committed` 序列）进摘要；跨实例混淆 ⇒ 摘要不同 |
| **C10** | 恢复路径排除 | 区间内发生快照恢复 ⇒ `REFUSED`（锚点可能被外部注入） |
| **C11** | 生产件不动 | `m1b_immutable_snapshot.py` 与其余四个登记文件的 blob 摘要**逐字不变** |
| **C12** | 结论分级 | 每项标 `PROVED_CONDITIONAL` / `DIAGNOSTIC` / `UNKNOWN`；`UNKNOWN` 必须列出缺什么 |

---

## 4. 参数与次数 + **十一条负控**（实现前固定，不得事后增删）

| 项 | 值 |
|---|---|
| 主对照输入 | 已登记状态：合成 **≤ 5** 个（3–6 节点）+ 每状态至多 **4** 个绑定字段的一 ulp 邻域 |
| 一 ulp 邻域 | `math.nextafter(x, ±inf)`，每字段 2 次 |
| 编码实现 | `struct.pack('>d').hex()`（主）+ `float.hex()`（回显） |
| 重复次数 | 纯函数，每项 **1 次** |
| 数据 | 不读 `data/`（合成实例自构造） |
| 随机种子 | `seed = 20261017`（若需抖点） |

| 编号 | 注入 | 预期 |
|---|---|---|
| **N-1** | 任一绑定时间取 `nextafter`（1 ulp） | 摘要改变（`STATE_IDENTITY_DRIFT`） |
| **N-2** | 时间字段取 `-0.0` | `REFUSED: NEGATIVE_ZERO_TIME` |
| **N-3** | 时间字段取 `nan` / `inf` / `-inf` | `REFUSED: NON_FINITE_TIME` |
| **N-4** | `hpu` 由 1.0 改 2.0（合同随之变） | `REFUSED: CONTRACT_IDENTITY_MISMATCH` 且 `NOT_CERTIFIABLE: TIME_UNIT_CONVERSION_NOT_BUDGETED` |
| **N-5** | 同 clock 不同 `instance_id` | 摘要不同（`CROSS_INSTANCE_CONFUSION`） |
| **N-6** | 区间内快照恢复（`resumed_between_doors=True`） | `REFUSED: SNAPSHOT_RESTORE_INSIDE_INTERVAL` |
| **N-7** | 认证后状态被改（`state_identity_after != before`） | `REFUSED: POST_CERTIFICATION_STATE_CHANGED` |
| **N-8** | 已执行前缀改动（移动/替换已服务节点） | `REFUSED: EXECUTED_PREFIX_CHANGED` |
| **N-9** | 缺 `T_clock_sup` 或缺 `n_max` | `NOT_CERTIFIABLE: TRUSTED_TIME_BOUND_MISSING` |
| **N-10** | 缺 `contract_hash` 或与登记不符 | `REFUSED: CONTRACT_IDENTITY_UNBOUND` |
| **N-11** | 把候选的规范化函数换回 `round(x,9)` | 必须被**自检**发现并 `REFUSED: LOSSY_CANONICALISER` |

---

## 5. 预期方向（expected direction）

| 量 | 预期 |
|---|---|
| 主对照误报数 | **0**（重建/序列化后摘要稳定） |
| 主对照漏报数 | **0**（任一绑定字段 1 ulp 漂移即改变摘要） |
| 当前实现对照 | `round(x,9)` 在 1 ulp 漂移下**摘要不变**（P-06 实测 `round9_equal = True`）⇒ 旧身份**必须**被取代，而不是被修补 |
| `-0.0` | `REFUSED`（策略），编码本身可表示 |
| 缺时间上界 | `NOT_CERTIFIABLE`（不得回落） |
| 生产件 | 摘要逐一不变 |

---

## 6. 失败条件（任一成立即判 G-1 FAIL）

| 编号 | 条件 |
|---|---|
| **G1-F1** | 主对照出现任一次误报或漏报 |
| **G1-F2** | 任一负控 N-1…N-11 未给出预期判定（静默通过 = 最严重失败） |
| **G1-F3** | 候选仍以 `round(x,9)`/十进制格式化为规范形式 |
| **G1-F4** | `None` 与 `0.0` 在摘要中不可区分 |
| **G1-F5** | 缺 `T_clock_sup` 时回落到 `horizon` 或 `max_iter` |
| **G1-F6** | 修改了 `m1b_immutable_snapshot.py` 或其余登记文件（生产件必须不动） |
| **G1-F7** | 把候选接入任何生产路径（认证器、安装入口、`certify_fn`） |
| **G1-F8** | 出现无条件 `PROVED` 标注，或把 `DIAGNOSTIC` 写成 `PROVED` |
| **G1-F9** | 自动 commit/push |

---

## 7. 资源门

| 项 | 上限 |
|---|---|
| 运行时间 | **< 60 s**（纯 CPU） |
| 数据读取 | 禁止读 `data/`（合成实例自构造） |
| 合成实例 | ≤ 5 个，3–6 节点 |
| GPU / 训练 / 正式 rollout | 禁止 |
| 生产路径 | 不导入安装入口、不改认证器、不接线 |
| 网络 | 禁止 |

---

## 8. 隔离候选规格（`G1_TIME_IDENTITY_V1`）

### 8.1 编码规则（**这是 G-1 与 `round(x,9)` 的分界线**）

| 规则 | 内容 |
|---|---|
| **R-1** | 时间与载荷一律用 `struct.pack('>d', x).hex()`（大端 IEEE-754 位型）作为**规范形式**；`float.hex()` 只作人类可读回显，**不参与摘要** |
| **R-2** | 时间字段必须 `finite ∧ x >= 0 ∧ not (-0.0)`；违反 ⇒ `REFUSED`，**绝不**编码后再事后解释 |
| **R-3** | 缺失值（`committed_finish is None`）编码为独立哨兵 `'null'`，与 `0.0` 的位型（`'0000000000000000'`）**不同** |
| **R-4** | 禁止 `round` / `format` / `%.9f` / `Decimal` 量化作为规范形式；候选自检时若发现即 `REFUSED: LOSSY_CANONICALISER` |
| **R-5** | 整数（`instance_id`、`event_id`、`n_max`）用十进制字符串，不经过浮点 |

### 8.2 绑定字段（摘要覆盖全部）

```
G1_TIME_IDENTITY_V1 = {
  schema, version, not_for_production, production_consumable,
  bits:    {clock, anchor, ready_time, current_load,
            committed_finish | 'null', return_finish | 'null',
            segment_durations: [..]},                       # 每个都是 R-1 位型
  execution: {instance_id, event_id, served_prefix: [..], committed_next: [..]},
  binding:  {contract_hash, hours_per_time_unit, distance_km_per_unit,
             T_clock_sup, n_max_per_vehicle,
             source: {5 个 blob 摘要}, census_digest},
  domain:   {dataset_glob, horizon_value, witness_equal},
  identity_digest                                        # sha256(规范 JSON)
}
```

### 8.3 与生产件的关系

- 候选模块**只读**生产件，不改一个字节（判据 C11）；
- 候选提供 `state_identity_v2(snap)` 与 `verify_state_unchanged_v2(...)`，
  并**并列**调用生产 `snapshot_state_identity` 以便对照；
- 生产替换需要**单独预声明 + 单独授权**（本文件**不**授权）。

### 8.4 编码可行性探针（本轮已完成，不是 G-1 实现）

`scripts/tests/test_q3_g1_encoding_probe.py`（8/8 PASS）只回答一个前提问题：无损编码是否存在、代价多大。

| 行 | 结论 |
|---|---|
| P-01 | `round(x,9)` 把 24.0+1e-12 与 24.0+2e-12 映射到同一字符串，位型编码可区分 |
| P-02 | 该窗口 = **281474 ulp**（`ulp(24) ≈ 3.55e-15`） |
| P-03 | 9 个样本逐位往返（含 `0.1`、`0.30000000000000004`） |
| P-04 | `-0.0` 位型可区分；**更正**：`repr(round(-0.0,9))` 保留符号，故这是**策略**问题 |
| P-05 | `nan`/`±inf` 可检出且位型互异 |
| P-06 | 1 ulp 漂移改变摘要，而 `round(x,9)` 不变 |
| P-07 | 七个绑定组件逐个改动都改变摘要 |
| P-08 | 1000 点全域采样逐位可逆 |

**分级**：`DIAGNOSTIC`（这是编码可行性探针，**不是** G-1 身份合格证明）。

---

## 9. 与状态的关系

本预声明**不改变任何科研状态**：`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`、
Q3 连续温度未建立、`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、L1 `BLOCKED`、
`L2/A3_NOT_ESTABLISHED`、`stable_online_gain=False`、`HISTORICAL_BACKBONE_UNVERIFIED`、
C0 `AWAITING_USER_BUDGET_RULING`、`t_min` 恒 0、最终测试种子 `SEALED` —— 全部不变。
