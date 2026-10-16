# DynMaskCO-CC：M1-a 原型实施边界约束·定向源码事实勘误（2026-10-16）

**勘误对象**：[M1-a 原型实施边界约束](DynMaskCO_CC_M1-a原型实施边界约束_2026-10-16.md)（提交 `495d8df`）。

**性质**：**定向**源码事实勘误。**只**更正该文件中三处与源码不符的论断。
**不**重新开展全项目审计，**不**重写、**不**删除、**不**覆盖原件与既有裁决。
**未**实现代码，**未**运行 S-1～S-17（全部 `NOT_RUN`），**未**执行真实模型或在线实验。

**基准**：`495d8dfd4f949b7ffb02a0b534ad41d47ac46b4f`。

---

## 1. 🔴 勘误一（严重）：S-15 的缩进判断错误，"零调用构造性保证"**撤销**

### 1.1 原件错在哪

原件 §1.2 断言品质分支位于 `if legality_ok:` **之内**，据此得出
"非法计划到达 `:291` 是**不可能的**"、"零调用**构造性成立**"。**该结论是错的。**

### 1.2 准确源码行号与缩进层级

| 行号 | 缩进 | 内容 |
|---|---|---|
| `:206` | **4** | `if legality_ok:` —— 块开始 |
| `:221-230` | 8 | `route_feasible` 分量 |
| `:232-243` | 8 | `energy_budget` 分量 |
| `:246-269` | 8 | `quality_projection` 分量（**只读投影**，非品质认证） |
| `:270` | **4** | `else:` —— `if legality_ok:` 的 else |
| `:271-277` | 8 | 非法集合时把三个分量标 `not_run_illegal_order_set` |
| **`:277`** | — | **`if legality_ok:` / `else:` 结构在此结束** |
| `:283` | **4** | **`if quality_check_fn is None:` —— 新的顶层块，不在任何 legality 分支内** |
| `:289` | 8 | `blocking_gaps = list(CAPABILITY_GAPS)` |
| `:290` | 4 | `else:` |
| **`:291`** | **8** | **`qok, qdetail = quality_check_fn(env, inst_idx, clock, vehicles, served_mask, plans)`** |
| `:292-295` | 8 | 三态赋值 / `blocking_gaps` / `if not qok` |

⇒ **`:283-295` 缩进为 4 空格，与 `:206` 同级**，位于 `if legality_ok:` **之外**。
⇒ **`quality_check_fn` 会被无关于 legality 地调用**：
即使四项 legality gate 与 `full_partition` 全部失败（`plans` 含未来订单、已服务订单、重复订单），
只要调用方**注入了** `quality_check_fn`，**`:291` 仍会执行**，非法 `plans` **已流过品质实现**。

### 1.3 撤销与重新定位

- **撤销**："legality-first 短路 ⇒ 品质回调零调用是结构性保证"这一结论；
  同时撤销原件 §1.2 表格中"四项检查与品质分支整体位于 `if legality_ok:` 之内"一行。
- **`d3_composed_certifier.py` 的既有设计**：`:307-309` 的 `note` 自述
  「legality gates run FIRST and, when one fails, the route / energy / quality components are
  NOT RUN」——该自述对 **`quality_projection`**（`:246`，确在块内）**成立**，
  对 **`quality_check_fn` 正式分支**（`:283-295`，在块外）**不成立**。
  ⇒ 这是**既有实现的已知缺陷**（自述与结构不一致），**不是**本文件新引入的判断。
- **直接 Composer 测试 ⇒ 归入"已知缺陷反例"**（已存在的直接注入路径）：

  | 路径 | 行号 |
  |---|---|
  | 测试封装直接调用 composer，并转发 `quality_check_fn` | `scripts/tests/test_d3_b3ac_gate_refusal.py:154-157`（签名 `:139` 默认 `None`） |
  | 探针直接调用 composer 并注入 `_probe_quality` | 同上 `:596-600`（探针定义 `:591-593`） |
  | 另一直接注入点 | `scripts/tests/test_d3_b3at1_transaction.py:258-262` |
  | 失败品质直接注入 | `scripts/tests/test_d3_b2_certification.py:360-362` |

- **新适配器 ⇒ 零调用改为"未来必须通过的验收要求"**：
  适配器**必须自行**保证"非法计划不进入品质实现"，
  **不得**依赖 composer 的结构。**该保证在实施前不存在。**

### 1.4 修正后的 S-15 设计（替换原件 §1.3）

必须**分成两个调用**，才能把"缺陷存在"与"适配器修复"分开证明：

```python
# 设计伪代码；本轮 NOT_RUN。
calls = []
def spy_quality(env_, inst_, clock_, vehicles_, served_mask_, plans_):
    calls.append(_structural_fingerprint(plans_))
    return True, {'probe': 'SPY_ONLY_NEVER_A_CERTIFICATE'}

for case in (FUTURE_ORDER, SERVED_ORDER, DUPLICATE_ORDER):

    # (A) 已知缺陷反例：直接 Composer + 注入回调
    repA = compose_certification(<ctx with case>, quality_check_fn=spy_quality)
    assert repA['legality_ok'] is False
    assert repA['installable'] is False
    # 如实记录既有实现的真实行为：调用次数被观测，不假定为 0
    observed_A = len(calls); calls.clear()
    # 期望 observed_A >= 1  ⇒ 记录为 KNOWN_DEFECT_ILLEGAL_PLAN_REACHED_QUALITY_IMPL

    # (B) 适配器要求：同一非法计划走适配器
    verdict = adapter.quality_stage(<ctx with case>, registry)   # 见 §3 修正设计
    assert verdict is REFUSED
    assert len(calls) == 0, 'ADAPTER_MUST_NOT_CALL_QUALITY_ON_ILLEGAL_PLAN'
```

**判定词表（修正，事前固定）**：

| 判定 | 含义 |
|---|---|
| `KNOWN_DEFECT_ILLEGAL_PLAN_REACHED_QUALITY_IMPL` | **既有 Composer** 在非法计划上仍调用注入回调（**(A) 的预期结果**，如实记录既有缺陷） |
| `ADAPTER_LEGALITY_FIRST_HOLDS` | **适配器**在非法计划上调用次数恰 0（**(B) 的通过条件**） |
| `ADAPTER_ILLEGAL_PLAN_REACHED_QUALITY_IMPL` | **适配器**调用次数 > 0 ⇒ **硬失败** |

**红线**：**(B) 不得**用 (A) 的结果代替；
**(A) 的高调用次数不得**被当作缺陷已修复的证据，反之 **`quality.state` 仍为 `QUALITY_CAPABILITY_GAP`**
也**不得**被当作"未调用"的证据（两者是不同的量）。

---

## 2. 勘误二：S-16 的 `blocking_gaps` 是**无条件**清空

### 2.1 原件错在哪

原件 §2.2 表内写"**只要 truthy 即清空阻断缺口**"。**表述不准**。

### 2.2 准确源码行号

```python
:293        blocking_gaps = [] if qok else []        # 两个分支都是 []
```

⇒ **无论 `qok` 为何值，`blocking_gaps` 都被赋为 `[]`**，是**无条件清空**，
**与 `qok` 的真值无关**。原件的"truthy"描述会误导读者以为 falsy 时缺口保留——**并不保留**。

### 2.3 后果更严重，且**与 `installable` 的判定路径有关**

`installable = bool(ok and not blocking_gaps)`（`:298`）。
走 `:290 else` 分支时 `blocking_gaps` 恒为 `[]`，故 `installable` **只**由 `ok` 决定；
而 `ok = not reasons`（`:297`），`reasons` 仅在 `:294-295` 的 `if not qok:` 为真时**才**追加
`quality_failed`
⇒ **`qok` 为 truthy 即 `installable=True`**（前提是其余检查全过）。

**同时必须如实指出（避免把本勘误读成新缺口）**：
三态赋值 `QUALITY_OK if qok else QUALITY_FAILED`（`:292`）**保留**了"检查未通过"这一信号，
因此即使 `blocking_gaps` 被无条件清空，该实现的**行为**仍是 fail-closed
（`qok` falsy ⇒ `:294` 追加 `quality_failed` ⇒ `ok=False` ⇒ `installable=False`）。
⇒ §2 的真正问题**不是**"缺口被清空导致漏放"，
**而是**：`qok` 的**类型合法性完全没有被检查**，
且"清空是无条件的"使得**无法**从 `blocking_capability_gaps` 或 `quality.state` 反推类型是否合法。

### 2.4 明确保留：`0.0` 是 falsy 非布尔，**仍须**被严格类型检查拒绝

- Python 中 `0.0` 属 **falsy**，若按 `:292` 的三态赋值会被判为 `QUALITY_FAILED`，
  若按 `:298` 的路径则**不**追加 `quality_failed` 以外的差异——但关键在于：
  **`0.0` 不是 `bool`**，它**既不是**合格的 `True` **也不是**合格的 `False`。
- ⇒ 原件 §2 的 **T1（无 truthiness，必须 `type(qok) is bool`）继续有效**，
  且 `0.0` **必须**被 `QUALITY_QOK_NOT_BUILTIN_BOOL` 拒绝。
- ⇒ 同样地，`0`、`''`、`[]`、`None` 这些 falsy 非布尔值**不得**被解释为"品质检查失败"，
  而应被解释为**回调返回类型非法**（`QUALITY_QOK_NOT_BUILTIN_BOOL`）。
  **"失败"与"非法"是两种不同的拒绝理由，不得合并。**

### 2.5 因此 `:293` 的无条件清空**不改变 T1 的必要性**

恰恰相反：由于清空是**无条件**的，"`qok` 的类型合法性"**无法**从
`blocking_capability_gaps` 或 `quality.state` 反推
⇒ **严格类型检查是唯一防线**，且**必须**在适配器内完成。

---

## 3. 勘误三：S-17 **不得**绕过适配器直接验证注册表

### 3.1 原件错在哪

原件 §3.2/§3.4 的用例把 `registered_impl` 直接作为 `quality_check_fn=` 传给 `compose_certification`
来测"注册表门"与"Ω 成员检查"。**该入口是错的**，因为：

| 事实 | 依据 |
|---|---|
| **Composer 没有注册表概念** | `compose_certification` 形参中无注册表；`:291` 直接调用所传可调用对象，**不校验实现身份、合同版本、适用域** |
| **Composer 接收原始 `env`** | `:291` 实参为 `env, inst_idx, clock, vehicles, served_mask, plans` ⇒ 回调**直接拿到原始环境对象**（含 `env.reveal_time` 等未来信息），**没有**最小可见信息接口 |
| ⇒ 因此"注册表审核"与"信息隔离"**都无法**经由 Composer 验证 | 它们**必须**由适配器层实现并验证 |

### 3.2 事后 `read_order_ids` **不得**冒充事前的信息隔离证明

原件 §3.5 把"事后审计"列为隔离机制之一，措辞使其像是**证明**。**更正**：

- **事前最小权限**是**唯一**的隔离**实施**机制：适配器的品质接口
  **不得**接收原始 `env`，只接收**显式 allow-list**（`clock`、`inst_idx`、`served_mask`、
  `visible_mask`、可见订单属性、车辆 CC/cargo 状态、`contract`）；
- **事后 `read_order_ids` 审计**只是**次级探测器**，
  它**能够**发现"回调声明读了未揭示订单"，但**不能**证明"回调没有读别的"——
  回调可以不诚实申报，也可以读取未被申报的字段；
- ⇒ **不得**以"事后审计通过"作为信息隔离的证明。两者是**探测**与**实施**的区别，**不可互换**。

### 3.3 修正后的 S-17 测试入口

```python
# 设计伪代码；本轮 NOT_RUN。
# (A) 注册表门：在适配器层验证，不经 Composer 的 quality_check_fn
registry = AdapterQualityRegistry()          # 正式注册表
for entry in (TEST_ONLY_ENTRY, UNCERTIFIED_ENTRY,
              NO_OMEGA_ENTRY, SHA_MISMATCH_ENTRY):
    assert registry.admit(entry) is REFUSED  # 逐一拒绝，理由见词表
assert registry.admit(AUDITED_ENTRY) is ADMITTED

# (B) Ω 成员检查：在适配器层，用域内/域外上下文
for ctx in (ctx_in, ctx_out_contract, ctx_out_zone, ctx_out_fleet,
            ctx_out_day, ctx_out_horizon, ctx_out_orders):
    v = adapter.quality_stage(ctx, registry)
    assert v is REFUSED if ctx is not ctx_in else v is not ADMITTED
    # 域外理由须含 QUALITY_DOMAIN_MEMBERSHIP_FAILED

# (C) 信息隔离：检查接口签名本身，而非行为
assert 'env' not in adapter_quality_signature_params()   # 事前最小权限
assert set(adapter_quality_signature_params()) <= ALLOWLIST
# (D) 事后审计仅作次级探测，明确标注 NOT_A_PROOF
```

**判定词表（修正）**：沿用 `TEST_ONLY_REGISTRY_REFUSED` / `UNCERTIFIED_IMPL_REFUSED` /
`NO_OMEGA_DECLARATION_REFUSED` / `IMPL_SHA_MISMATCH_REFUSED` /
`QUALITY_DOMAIN_MEMBERSHIP_FAILED`；
新增 `QUALITY_READ_UNREVEALED_ORDER`，并**标注为**
`SECONDARY_DETECTOR_NOT_AN_ISOLATION_PROOF`。

**红线**：`QUALITY_READ_UNREVEALED_ORDER` 的**存在**不得被引用来论证隔离已成立；
隔离成立**只能**由 (C) 的接口边界证明。

---

## 4. 采纳用户建议：**两段式**适配器设计（替换原件 §3.2 的单段式）

用户建议正确且与 §1 的缺陷直接对应。修正后的执行顺序：

```text
第一段：compose_certification(..., quality_check_fn=None)      # 见下
        ⇒ 完成合法性、路线、能耗检查
        ⇒ 预期 quality.state == QUALITY_CAPABILITY_GAP（因为传 None）
        ⇒ 全部通过之前，绝不进入第二段

第二段（仅当第一段合法性/路线/能耗全部通过）：
        经【注册表】解析已审核品质实现 + 【最小可见信息接口】执行独立品质认证
        ⇒ 该接口不接收原始 env
```

**四条硬约束**：

1. **第一段必须显式传 `quality_check_fn=None`**——这同时**规避**了 §1 的缺陷
   （传 `None` 时走 `:283-289`，`:291` **不会**执行，故非法计划**不会**到达任何品质实现）；
2. **第二段只在第一段通过后执行**——不得并行、不得提前、不得因"反正会拒绝"而先跑；
3. **第二段不经 Composer**——注册表审核、Ω 成员检查、身份比对、严格类型检查
   **全部**在适配器层完成（§2、§3）；
4. **测试回调的成功不得当作正式品质能力**——沿用原件 §3.4 的约束：
   无真实已审核实现时，**即使全部合成用例通过**，
   `quality.state` 仍须为 `QUALITY_CAPABILITY_GAP`、`installable` 仍须为 `False`。

⇒ 该设计使 §1 的缺陷**不再可达**（因为第一段传 `None`），
但 **§1 的缺陷仍然存在**于 `d3_composed_certifier.py` 中，
**未修复、不应被描述为已修复**，且**不得**用它作为"零调用已保证"的依据。

---

## 5. 勘误清单与剩余阻断

### 5.1 本次更正的三处

| # | 原件位置 | 错误 | 更正 |
|---|---|---|---|
| 1 | §1.2 表 + §1.3 | 品质分支在 `if legality_ok:` 内、"零调用构造性成立" | **撤销**；`:283-295` 在 `:206` 块**之外**；`:291` 可对非法计划调用；归入**已知缺陷反例**；新适配器零调用为**待通过验收要求** |
| 2 | §2.2 表 | "只要 truthy 即清空 `blocking_gaps`" | **无条件清空**（`:293` `[] if qok else []`）；`0.0` 等 falsy 非布尔**仍须**被 T1 拒绝；"失败"与"非法"分列 |
| 3 | §3.2 / §3.4 / §3.5 | 经 Composer 直接验证注册表与 Ω；事后审计被当作隔离证明 | Composer **无注册表概念**、**接收原始 `env`**；测试入口移至**适配器层**；`read_order_ids` 降为**次级探测器**，**不得**冒充事前隔离证明 |

**原件中仍然有效、未被本次更正触及的部分**：S-15 的三类非法输入构造
（未来/已服务/重复）；S-16 的 T1/T2/T3 三条约束与 9 个负控；
S-17 的四类注册表拒绝理由与 Ω 成员性不得以温度区间冒充的约束；
S-12 的来源/别名精确化（§4）；§5 实施与授权边界；§6 状态词表约束。

### 5.2 剩余阻断（本轮无变化，**未因本勘误而减少**）

| # | 阻断 | 状态 |
|---|---|---|
| **G-1** | 决策期品质/鱼可售认证 | `CAPABILITY_GAP_BLOCKING`，保守性未建立 |
| **G-2** | 生产调用点 | `compose_certification` 无生产调用者 |
| **G-3** | C0 预算 `B` | `AWAITING_USER_BUDGET_RULING` |
| **G-4** | L2 主质量指标五要素 | 全 `UNRESOLVED` |
| **G-5** | 共同决策器代次 | 两代并存，工作树版与驱动器不兼容 |
| **G-6** | `mutable_ids` 强制 | composer 形参零引用，须适配器自行强制 |
| **G-7** | L1 `BLOCKED` 进入资格 | 须用户显式裁定 |
| **G-8** | 假回调可翻转 `installable` | 仓库自测已确认 `PROBE_ONLY_GATE_NOT_A_WALL` |
| **G-9（新增）** | **Composer 品质分支不受 legality 门控** | `:283-295` 在 `if legality_ok:` 之外 ⇒ 注入回调时非法计划可达品质实现；**既有缺陷，未修复** |

⇒ **G-9 是本次勘误的净新增阻断**：它使"完整认证"的任何主张
在适配器层实现之前**更**不可成立。

### 5.3 保留不变

`stable_online_gain=False`；`HISTORICAL_BACKBONE_UNVERIFIED`；
I3 负结果（`−144.680 [−191.109, −100.125]`）；D2 `not_qualified`；D3 G0 未通过；
D8 `CAPABILITY_GAP_BLOCKING`；E2 无合格结果；C0 `AWAITING_USER_BUDGET_RULING`；
**L1 正式批次 `BLOCKED`（绝不改写为 PASS）**；
**未修复** `d3_composed_certifier.py`；未改扩展冻结树、冻结数据或既有结果。

**未授权则不动**：**未经用户明确授权，不启动 M1-a 原型实施。**
本文件**不申请也不消费**授权。
