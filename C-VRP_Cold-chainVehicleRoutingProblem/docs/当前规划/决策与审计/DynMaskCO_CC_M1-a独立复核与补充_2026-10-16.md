# DynMaskCO-CC：M1-a 设计独立复核与补充（2026-10-16）

**审查基准**：`6aae352b2baa19f2b50cd359e66bd4c98a169145`（执行前已 `git fetch`；
本地 `HEAD`、`origin/main`、`origin/codex/o0cc-cal-phys` 三者相同；工作区仅
`CC_Compare/MAPT_code` 既有脏状态，不属本任务）。

**性质**：**新增**的独立静态复核与接口补充。**不改写、不覆盖**任何既有交件——
包括 M0 五份交件与
[方法学复核与 M1-a 设计](DynMaskCO_CC_方法学复核与M1-a设计_2026-10-16.md)（下称"既有附录"）。
不是预登记签署、不是工程验收、不是科学执行授权。

**本轮实际边界**：只做静态读取、文档勘误与设计。**未**执行科研/仿真/模型模块，
**未**加载 F 权重，**未**做真实或合成前向，**未**采集状态，**未**训练，**未**安装计划。
本文件所述一切合成用例均为**待授权方案**。

---

## 0. 先说清一件事实：本任务的主要交件已存在

用户本次任务要求"成果单独新增『方法学复核与 M1-a 设计』文档"。检索发现该文件
**已于 `6aae352` 提交并已推送**（commit date `2026-10-16 13:27:49 +0800`），
282 行，覆盖范围与本次任务高度重合。

因此本轮**不重复撰写**，改为：

1. 对该文件做**独立静态复核**（它自己未做独立复核）；
2. 补正其中**两处行号/措辞不精确**；
3. 补上该文件未记录、但对 M1-a 接线有**直接**影响的**两项源码事实**。

**未提交任何新授权，也未因其存在而跳过复核。**

---

## 1. 独立静态复核结果

复核方法：仅用标准库（`hashlib` / `ast` / `io` / `re` / `os` / `json`），
**不导入**任何科研、仿真或模型模块。复核器为一次性脚本，**不写入仓库**。

### 1.1 身份复核（10/10 全等）

既有附录 §8 登记了 10 个输入文件的工作树原始字节 SHA256。逐条重算：

| 文件 | 结论 |
|---|---|
| `scripts/simulation/d3_composed_certifier.py` | **MATCH** |
| `scripts/simulation/d3_geo_adapter.py` | **MATCH** |
| `scripts/simulation/d3_l1_two_arm_protocol.py` | **MATCH** |
| M0 总体路线图 / L2 草案 / L3 草案 / C0-基线清单 / 资格矩阵 JSON | **5/5 MATCH** |
| `项目当前状态.md` / `执行进度表.md` | **2/2 MATCH** |

⇒ 既有附录所审查的输入**与当前工作树逐字节相同**，其结论**未因后续提交而失效**；
同时这也**独立证明**本轮未改动上述任一文件（含 M0 五件）。

### 1.2 内容复核（逐条与源码对齐）

| # | 既有附录的断言 | 源码实测 | 判定 |
|---|---|---|---|
| 1 | composer 返回 dict，`installable` 在返回中 | 返回键实测 12 个，**含 `ok` 与 `installable`**（`d3_composed_certifier.py:299-309`） | **成立** |
| 2 | 回调接口 `certify_fn(P) -> (ok, detail)` | `d3_geo_adapter.py:554` `ok, detail = certify_fn(P)` | **成立** |
| 3 | 回调缺省为 `None` 在 `:553` | 实测解包在 **`:554`**，`certify_fn=None` 形参默认在 **`:405`** | **行号需精确化**（见 §2.1） |
| 4 | sampler 在检查 `ok` 之前就设 `diag['certified']=True` | `:558` `diag['certified'] = True`，`:561` `if not ok:`，`:563` `return _rollback('cert_fail')` ⇒ **赋值确在 ok 检查之前** | **成立** |
| 5 | `not None` 分支另有 `:575` 置 `certified=False` | `:571 else:` / `:575 diag['certified'] = False`（注释说明"无认证器时纯候选"） | **成立** |
| 6 | `mutable_ids` 只有形参、函数体零引用 | AST 全函数体扫描：标识符出现**恰 1 次**，类型 `ast.arg`（声明），**位置 `:57`**；`Load`/`Store`/`Attribute`/callsite-kwarg **各 0 次**；全文文本出现亦仅 1 次 | **成立（强于原文）** |
| 7 | 无认证器时 `quality.state = QUALITY_CAPABILITY_GAP` | `:283-289`：`quality_check_fn is None` ⇒ `{'state': QUALITY_CAPABILITY_GAP}` 且 `blocking_gaps = list(CAPABILITY_GAPS)` | **成立** |
| 8 | `installable = bool(ok and not blocking_gaps)` | `:298` 逐字相同 | **成立** |
| 9 | 安装边界无认证器即拒绝 | `d3_geo_adapter.py:710` `INSTALL_REQUIRES_CERTIFICATION = True`；`:721-725` `certify_fn is None` ⇒ `ledger.finish(failure_reason='no_certifier', …, refused_before_sampling=True)` | **成立** |
| 10 | 离线 L2 不调用 `run_repair_transaction` / `InstallLedger.finish` | 二者定义于 **`d3_geo_adapter.py:708` 与 `:586`**（**不在** `d3_commit_transaction.py`） | **成立，但位置需精确化**（见 §2.2） |
| 11 | V3 无认证器记 `NOT_RUN`，不记 True/False | `d3_l1_two_arm_protocol.py:310`、`:385` 默认 `certify_fn=None`；`:531-534` `certify_fn is None` ⇒ `v3 = {a: {'certified_executable_plan': 'NOT_RUN'}}`，`any_certified='NOT_RUN'` | **成立** |
| 12 | 文档 14 个相对链接均存在、代码围栏配对 | 相对链接 **14/14 存在**；` ``` ` 围栏 **4 个（偶数，配对）** | **成立** |
| 13 | `B=710.6548152249378` 只作 I3 旧实验来源数字 | 见 §3.1——**成立，但证据应更硬** | **成立** |

**复核结论**：既有附录的**结论性断言 13/13 成立**，无须更正其实质内容；
发现 **2 处行号/位置表述不精确**（§2）与 **2 项未记录的源码事实**（§3）。

### 1.3 复核过程自曝：我自己的第一版复核器有两处错误，已更正

按"不以退出码为证据"的既有要求，如实记录：

- **错误一（假阴性）**：第一版把"`certified=True` 是否先于 ok 检查"实现为比较
  `:558` 与 `if not ok:` 的**下一个分支**（`:571 else:`）⇒ 得出 `False`，
  会**错误地否定**既有附录的成立断言。改为直接比较 `:558` 与 `:561` 后确认**原文正确**。
- **错误二（过宽）**：第一版 `C6` 检索 `710.654` 时只搜 `scripts/` 全目录并断言"不存在"，
  实际存在 2 个命中 ⇒ 该检项**判据本身有误**。改判见 §3.1。

两处均为**我方复核器**缺陷，不是既有附录的缺陷；在此留档以证明复核不是走过场。

---

## 2. 对既有附录的两处精确化（不改变其结论）

### 2.1 回调行号

既有附录 §2.2 写"`d3_geo_adapter.stepwise_sample_d3(..., certify_fn=...)` 在 **:553–563** 解包"。
实测：**`:553` 是 `if certify_fn is not None:`**，**解包在 `:554`**；
`certify_fn=None` 的**形参默认值在 `:405`**。
⇒ 建议表述为"**`:554` 解包；形参默认 `None` 见 `:405`；失败回滚 `:563`**"。

### 2.2 安装事务的归属文件

既有附录 §5.3 提到 `run_repair_transaction` 与 `InstallLedger.finish`，未给路径。
实测二者**均定义在 `scripts/simulation/d3_geo_adapter.py`**（`:708`、`:586`），
**不在 `d3_commit_transaction.py`**（该文件 627 行内 `install_fn` 出现 **0** 次）。
⇒ 若后续实现要"确保离线 L2 不触达安装路径"，**必须**以 `d3_geo_adapter` 为拦截对象。

---

## 3. 既有附录未记录、但直接影响 M1-a 接线的事实

### 3.1 `710.6548152249378` **确实存在于生产脚本目录**，且自带 `budget_provenance`

既有附录 §5.4 说该值"只保留为 I3 旧实验来源数字"。**结论正确，但证据可以更硬**——
它不是只存在于文档里，而是**写在可执行脚本中**：

| 位置 | 内容（逐字） |
|---|---|
| `scripts/evaluation/diag_i3_teacher_probe.py:33-37` | `SCALE200_CONFIG = {…'orders': 200, 'capture_ordinals': [24, 99, 174], 'budget': 710.6548152249378, 'cooling_share': 3.7208946347769882, 'vehicles': 15, 'budget_provenance': 'Existing 200 training-day rho=0.60 budget; fixed before probe', …}` |
| `scripts/evaluation/_debug_vote2.py:81-83` | `rp = SaaReplanner(budget=710.6548152249378, capacity=50.0, …, cooling_share=3.7208946347769882, sampler=sampler, …)` |

**这为什么重要**：该数值**带出处标注**（"现有 200 训练日 rho=0.60 预算；探针前固定"），
即它**正是既有 `(1−ρ)·mean(train 全接 C0 能耗)` 派生口径的一次具体实例**，
**不是**协议级常量。⇒ 与附件 4 §4.3 的分类**一致**，且**进一步**说明：

- 任何人若"照抄数字"建 L2 预算，抄到的是**一个探针配置里的派生值**，
  **不是**绑定 64 `dev_check` 状态的注册预算；
- 该值所在的两个脚本是**诊断/调试脚本**（`diag_`、`_debug_` 前缀），
  **不构成**注册协议路径。

**M1-a 强制要求（新增）**：`CertificationContext.B` **不得**从
`diag_*` / `_debug_*` / 测试夹具 / profile 的 `energy_scale` 派生；
必须来自**版本化注册配置**并携带协议、车队、日范围与状态绑定 SHA。
**本文件不登记也不建议 `B` 的取值**——裁定权在用户。

### 3.2 🔴 任意注入的"品质检查"**确实能**把 `installable` 翻成 `True`——**仓库自己的测试就是这样断言的**

这是既有附录 §2.3 第 4 条（"测试假回调不能成为正式品质能力证明"）在**源码层面的完整证据链**，
既有附录只给了结论，未给这条最直接的证据。实测：

| 位置 | 内容 |
|---|---|
| `d3_composed_certifier.py:290-296` | `quality_check_fn` 非 `None` ⇒ `qok, qdetail = quality_check_fn(...)`；`quality.state = QUALITY_OK if qok else QUALITY_FAILED`；`blocking_gaps = [] if qok else []` ⇒ **只要 `qok=True`，阻断缺口即清空** |
| `d3_composed_certifier.py:298` | `installable = bool(ok and not blocking_gaps)` ⇒ 其余检查全通过时 **`installable=True`** |
| `scripts/tests/test_d3_b3ac_gate_refusal.py:591` | `def _probe_quality(env_, inst_, clock_, vehicles_, served_mask_, plans_):` —— **一个探针假回调** |
| 同上 `:600` | `quality_check_fn=_probe_quality) if clean else None` |
| 同上 `:603-605` | `'AC-R10 PROBE ONLY: installable is reachable solely via an injected quality check'`；断言值 `'PROBE_ONLY_GATE_NOT_A_WALL' if (cert_probe['installable'] is True and cert_legal['installable'] is False) else 'UNEXPECTED'` |
| 同上 `:616` | `'a fake quality check is what flips \`installable\`. That injection is '` |

⇒ **不是我的推断**：仓库**自己**的测试把"假品质回调把 `installable` 翻成 True"
**断言为预期结果**，并**明确命名为"探针专用门，不是墙"**（`PROBE_ONLY_GATE_NOT_A_WALL`）。

**对 M1-a 的直接约束（新增，强制）**：

1. L2 的认证回调**必须**由**审核过的品质实现注册表**解析 `quality_check_fn`；
   **禁止**由方法臂、命令行参数、环境变量或测试注入；
2. 回调实现**必须**携带：实现身份 SHA、适用合同版本、**适用域 Ω**、
   保守性证明引用，并**在运行时做 Ω 成员检查**；
3. 缺少上述任一 ⇒ 回调**必须**按"无品质能力"处理（等价于 `quality_check_fn is None`），
   **绝不允许**"因为传了个函数所以缺口消失"；
4. **验收负控必须包含**："把 `_probe_quality` 一类的假回调接到正式入口 ⇒ 必须被**拒绝**"。
   仅断言"假回调能翻转 `installable`"**不足以**证明防线存在——那只是复述了漏洞。

---

## 4. 三项方法学问题的审查结果（用户要求的第 1 项）

### 4.1 任务一 A：L2 完整认证的真实含义

**四层必须分开（结论）**：

| 层 | 判定函数 | 单位/判据 | 当前可用性 |
|---|---|---|---|
| ① C0 能耗预算认证 | `run_exp_energy_c0.certify_plan` | kWh；`total <= B + 1e-9` | **可用**（纯函数，仅能源） |
| ② 完整路线与承诺合法性 | `compose_certification` 的四项 legality gate + `repair_plan` / `full_partition` / `route_feasible` | 布尔 + 逐车原因 | **可用**（有生产实现，**无生产调用点**） |
| ③ 决策期品质/鱼可售认证 | `quality_check_fn` | **无已证明的保守实现** | **不可用** ⇒ `CAPABILITY_GAP_BLOCKING` |
| ④ `installable` 最终判定 | `bool(ok and not blocking_gaps)` | 布尔 | **恒为 False**（因③） |

**接口差异（已复核）**：`compose_certification` 返回 **dict（12 键）**；
回调契约为 **`(ok, detail)` 元组**。⇒ **不能直接接线**，
且**不得**用 `bool(report)` 或 `report['ok']` 代替完整资格——
`ok=True` 而 `installable=False` 是**可达的合法状态**（`:283-298`）。

**对用户要求的正面回答**：
**`FULL_INSTALLABILITY_NOT_AVAILABLE`** —— 品质认证不能闭合，
因此**保留能源与路线认证的有限诊断能力**，
**但其通过率不得用作 L2 完整质量主门的通过率**。
名称本身尚未存在于代码（实测 `scripts/` 下 0 命中），
M1-a **只**把它定义为**回调返回的 detail 标签**，不新建第四个状态常量，
以免与既有 `QUALITY_OK`/`QUALITY_FAILED`/`QUALITY_CAPABILITY_GAP` 三态冲突。

### 4.2 任务一 B：H1 裁决词表

既有附录 §3.1 的四词表（`PASS` / `NOT_ESTABLISHED` / `NEGATIVE_DIRECTION` / `BLOCKED`）
经复核**采用**，并补充三条**机械化**约束（本文件新增）：

1. **CI 跨零 ≠ 统计等价**，也 **≠ 学习机制被证伪**；
   负点估计 + CI 跨零 ⇒ `NOT_ESTABLISHED` + 另报 `observed_direction=negative`。
2. **主质量指标五要素全部 `UNRESOLVED`**（函数 / 单位 / 方向 / 认证失败处理 / 预声明阈值）。
   ⇒ **在五要素闭合前，任何 H1 正负裁决都是 `BLOCKED`**，
   不是 `NOT_ESTABLISHED`，更不是 `NEGATIVE_DIRECTION`。
3. **"失败留分母"≠ 已定义全分母估计量**；仅取两臂都认证通过的子集**有选择偏差**，
   只能作**附加描述**，**不得**悄然升为主分析。

### 4.3 任务一 C：L3 决策层分辨率

既有附录 §4 的四组指标（接拒动作 / 计划结构 / 费用分解 / 时间与回退）经复核**采用**；
并补充三条（本文件新增）：

1. **接单向量相同不蕴含无分辨率**：相同接单可对应不同车辆、顺序、回仓时刻、燃油、
   品质、超时与回退。⇒ 原 §8bis 的"接单集合差异率"**单独不足以**判定不可分辨。
2. **CI 跨零不是"等价"或"不可分辨"的充分证明**——与 §4.2 第 1 条同源；
   要看**等价界**是否事前定义。
3. **这些层只用于解释，永不替代预登记的主效用比较**；
   且**不得**用于事后挑选天数或替换主门。

---

## 5. M1-a 完整认证最小实现设计（用户要求的第 2 项）

### 5.1 参数映射 `compose_certification ← CertificationContext`（复核 + 补强）

| composer 形参 | 来源 | 必须前置核验 |
|---|---|---|
| `env, inst_idx, clock` | 冻结状态的**只读**视图 | 节点数、单位、epoch；**只**允许揭示判定与可见/已承诺属性 |
| `vehicles, served_mask` | 原状态真实车辆/CC/cargo 的只读副本 | **不得**从候选计划重造 |
| `plans` | 完成候选 `Pc` | 节点合法、全承诺覆盖、结构边界先验 |
| `mask_set, protected, deferred` | 同一固定状态、同一规则导出 | 记录 hash；`deferred` 须单独满足承诺覆盖 |
| `B, contract` | **版本化注册配置** + 真实合同对象 | **禁止**来自 `diag_*`/`_debug_*`/夹具/profile scale（**见 §3.1**） |
| `base_plans` | 独立、已核对完整的 `P0` | **必填**；缺省即 `repair_plan_unverifiable`（`:176-181`） |
| `mutable_ids` | 同一动作空间导出 | **适配器须自行强制**——composer **完全未使用**该参数（**§1.2 第 6 条**） |
| `quality_check_fn` | **审核过的品质实现注册表** | 见 **§3.2** 四条强制约束 |
| `project_quality` | 保留只读投影为**有限诊断** | 其通过**不取代**品质证书（`:280-282`） |
| `stress_quality_schedule` | 记录 schedule 身份 | **不得**把 latest schedule 自动声明为最坏/保守 |

### 5.2 必需状态上下文（`CertificationContext`，最小集）

| # | 内容 | 关键约束 |
|---|---|---|
| 1 | **完整 `P0`** | 与 `Pwork`、mask 后 `partial`、候选 `Pc` **分开保存**；**禁止** `base_plans=Pc` 自证 |
| 2 | **完整承诺 `A_t`** | 已接受/已拒绝/待决/deferred 显式集合；**不得**从"可见集合"或某个 suffix 推断 |
| 3 | **可见信息** | `state_id`/epoch、`clock`、`served_mask`、`visible_mask`、外生揭示身份、真实 ID 映射、可见订单属性；不可见属性**不得**传入 |
| 4 | **修改边界** | `mask_set`、`protected`、`mutable_ids`（**自行强制**）、动作空间、平局规则、基线 hash |
| 5 | **物理与业务合同** | contract/version/hash、单位、速度、booking horizon、等待/服务/回仓/卸货语义、CC state、C0 与 evaluator 身份 |
| 6 | **预算与品质资格** | 有限有效 `B`（kWh）+ 登记来源与状态绑定 SHA；品质实现的**身份/合同版本/Ω/成员检查** |
| 7 | **运行与证据身份** | 两臂共同资源预算、截止时刻、独立时钟起点、输入/源码/候选/指标/接受规则身份、**追加式收据**、异常去向 |

### 5.3 返回值映射：dict → `(ok, detail)` 的**唯一**合法映射

```python
# 设计伪代码；不是可运行入口，不构成任何验收结果。
REQUIRED = ('plan_vehicle_set', 'revealed_only', 'no_served_in_plan',
            'no_duplicate_order', 'full_partition', 'repair_plan',
            'route_feasible', 'energy_budget')

def callback(Pc):
    pre = validate_context_plan_and_deadline(ctx, Pc)      # 独立前置门
    if not pre.ok:
        return False, receipt(pre, Pc)                     # 不进入 composer
    try:
        rep = compose_certification(...)                   # §5.1 映射
        if not schema_ok(rep, REQUIRED):                   # 缺键/非 bool/not_run
            return False, receipt('REPORT_SCHEMA_INVALID', Pc, rep)
        if rep['quality']['state'] != QUALITY_OK:           # 品质缺口
            return False, receipt('FULL_INSTALLABILITY_NOT_AVAILABLE',
                                  Pc, rep, original_gap=rep['quality']['state'])
        full_ok = bool(rep['ok'] is True and rep['installable'] is True
                       and rep['legality_ok'] is True
                       and not rep['blocking_capability_gaps']
                       and all(rep['components'][k]['ok'] is True for k in REQUIRED))
        if not durable_receipt(full_ok, Pc, rep):
            abort('RECEIPT_NOT_DURABLE')                   # 硬停止，不返回无证据 True
        return full_ok, receipt(full_ok, Pc, rep)
    except BaseException as exc:                           # 注意 composer 内部已 catch BaseException
        return False, receipt('TYPED_EXCEPTION', Pc, type(exc).__name__)
```

**三条强制不变量**：

- **I1（不自证）**：`base_plans` 必须独立提供且先于候选通过校验；
- **I2（缺口优先）**：`quality.state != QUALITY_OK` ⇒ **无条件**返回 `False`，
  **即使** `rep['ok'] is True` 且 `rep['installable'] is False` 之外的其余检查全绿；
- **I3（证据先于结论）**：收据落盘失败或来源身份变化 ⇒ **中止**，**不返回 `True`**。
  `compose_certification` 内部 `_guard` 会捕获 `BaseException`（`:87`），
  故**必须**有**独立监督器**处理终止语义，**不得**仅凭回调布尔值认定完成。

### 5.4 fail-closed 语义（覆盖用户点名的六类违规）

| 事件 | 必须结果 | 代码依据 |
|---|---|---|
| **认证失败**（任一项 not ok） | 拒绝 + 逐项原因 | `:297-298` `ok = not reasons` |
| **品质能力缺口** | **不可安装**；保留能源/路线有限诊断 | `:283-289`、`:298` |
| **时间窗违规** | `route_feasible=False`；**不得**靠能源通过覆盖 | `certify_route` 于 `:210-218` |
| **重复订单** | `no_duplicate_order=False`；**后续组件 `not_run`** | `:128-134`、`:145-146`、`:271-277` |
| **容量越限** | `route_feasible=False`（accept 侧更严，`capacity−1e-4`） | `:207-230` |
| **回仓与能耗违规** | 回仓 ⇒ `route_feasible=False`；能耗 ⇒ `energy_budget=False` | `:210-243` |
| **非法/未来/已服务/错误车辆** | 前置拒绝；`legality_short_circuit`；**不删节点后重判** | `:112-147` |
| **`P0` 缺失 / 自证** | `repair_plan_unverifiable` | `:176-181` |
| **超时（认证前后）** | 失败/硬停止；**保留候选与不完整收据** | 回调侧 `:567-570` + 适配器 |

---

## 6. 哪些资格门仍未闭合（用户要求的第 3 项）

| # | 未闭合项 | 当前状态 | 阻塞对象 |
|---|---|---|---|
| **G-1** | **决策期品质/鱼可售认证** | `CAPABILITY_GAP_BLOCKING`；保守性未建立 | **L2 完整质量主门** |
| **G-2** | **生产调用点** | `compose_certification` **无生产调用者**：全 `scripts/` 调用点共 **32 处**，其中 **30 处在 `scripts/tests/*`**，另 2 处为**定义自身**（`d3_composed_certifier.py:55`）与 `d3_commit_transaction.py:359` 的**标签字符串** `certifier = 'compose_certification+pre_install_reverify'`（**不是**调用，grep 符号会**误报**） | L2 的 V3 非 `NOT_RUN` |
| **G-3** | **C0 预算 `B`** | `AWAITING_USER_BUDGET_RULING` | L2 能源门 |
| **G-4** | **L2 主质量指标五要素** | 函数/单位/方向/失败处理/阈值 **全 `UNRESOLVED`** | H1 裁决 |
| **G-5** | **共同决策器代次** | 两代并存；工作树版**与自己的驱动器不兼容** | L3 身份冻结 |
| **G-6** | **`mutable_ids` 未实现** | composer 形参零引用 ⇒ 修改边界**必须**由适配器自行强制 | 承诺/修改边界 |
| **G-7** | **L1 `BLOCKED` 的进入资格** | 须用户显式裁定；**不得**改为 PASS | L2 前置门 |
| **G-8** | **假品质回调可翻转 `installable`** | 仓库自测已确认为 `PROBE_ONLY_GATE_NOT_A_WALL` | 任何"完整认证"主张 |

**结论**：`FULL_INSTALLABILITY_NOT_AVAILABLE` 维持。
**不得**以其有限诊断能力替代 L2 完整质量主门（与既有附录 §1 一致）。

---

## 7. 隔离合成验收方案（**本轮 NOT_RUN**，待授权）

在既有附录 §7.1 的九组矩阵之上，**新增五组**（全部为待授权方案，本轮**未执行**）：

| 组 | 构造 | 预期判定 |
|---|---|---|
| **S-10 假回调拒入** | 把 `_probe_quality` 式假回调接到**正式**入口 | **必须拒绝**；仅断言"能翻转 `installable`"**不算**通过（**§3.2**） |
| **S-11 `mutable_ids` 强制** | 在 `mutable_ids` 之外的车辆上移动订单 | 适配器**必须**拒绝（composer 不会拒绝——**§1.2 第 6 条**） |
| **S-12 自证基线** | `base_plans=Pc` | **必须**拒绝（I1） |
| **S-13 缺口优先** | 所有检查全绿 + 品质缺口 | **必须** `False` 且 detail 含 `FULL_INSTALLABILITY_NOT_AVAILABLE`（I2） |
| **S-14 证据不可落盘** | 收据写盘失败 | **必须**中止，**不得**返回 `True`（I3） |

**下一次正式资格门（依赖顺序，未授权）**：
① 适配器原型 + 合成接线验收（含 S-10…S-14）+ 独立只读复核 ⇒
② 真实品质实现 + Ω 证明 + 运行时成员检查闭合 ⇒
③ 用户签署主指标五要素、`δ_L2`、统计/安全门、`B` 来源与绑定 ⇒
④ 新协议的状态/数据身份与 L1 `BLOCKED` 下的进入资格 ⇒
⑤ **另行**的正式执行签署。**本文件不预先授予**。

---

## 8. R / D-Geo / cond_hist 的层次归属（用户第 2-6 项）

| 名称 | 层次 | 能否直接做固定状态 L2 配对 |
|---|---|---|
| **D-Geo** | 固定 mask/动作空间内的**规则打分**，逐步重构 | 仅当 `P0`/mask/动作空间/接受器/完整认证/指标**全部固定**后才成立；**旧 L1 首步分歧 ≠ L2 正效应** |
| **R（regret-2 / CC-LNS）** | `cc_lns_replanner.py` 中的**完整搜索包装**（自有 mask、regret-2 重构、平局规则、循环预算、`J_vis` 接受器） | **不能**直接配对——旧 R 包装 ≠ 相同 mask 下的一次打分替换；须声明新变体身份与资源口径 |
| **`cond_hist`** | **系统层**（条件场景 + SAA + 接拒/重规划） | **不是** `certify_fn` 或固定 mask 修复臂；**不能**强行加入同质 L2 标量配对 |

⇒ 修正 M0"R 与 `cond_hist` 均为 L2 必备平行臂"的表述：
**保留基线清单**，但**先做层次兼容性审查**；本轮不实现适配。

---

## 9. 静态复核记录与身份

| 复核项 | 方法 | 结果 |
|---|---|---|
| 10 个输入文件字节身份 | `hashlib.sha256` 原始字节 | **10/10 MATCH** |
| `mutable_ids` 引用 | `ast` 全函数体扫描（Name/Attribute/keyword/arg 四类）+ 文本 | 仅 **1 处**，`ast.arg` `:57` |
| composer 返回结构 | `ast` 解析 `Return` 的 `Dict` 键 | **12 键**，含 `ok`/`installable` |
| 回调解包与顺序 | 行级正则 + 窗口打印 | 解包 `:554`；`certified=True` `:558` **先于** `if not ok:` `:561` |
| V3 语义 | 行级正则 | `:531-534` `NOT_RUN` |
| 品质三态常量 | 正则 + AST | `QUALITY_OK='CHECKED_SATISFIED'` / `QUALITY_FAILED='CHECKED_FAILED'` / `QUALITY_CAPABILITY_GAP='CAPABILITY_GAP_BLOCKING'` |
| `FULL_INSTALLABILITY_NOT_AVAILABLE` | `scripts/` 全目录文本扫描 | **0 命中**（尚未存在于代码） |
| `710.6548152249378` | `scripts/` 全目录文本扫描 | **2 命中**（`diag_i3_teacher_probe.py:35`、`_debug_vote2.py:81`） |
| 安装边界 | 行级正则 + 窗口 | `INSTALL_REQUIRES_CERTIFICATION = True`（`:710`）；`:721-725` 拒绝 |
| `run_repair_transaction`/`InstallLedger` | `def`/`class` 扫描 | **`d3_geo_adapter.py:708` / `:586`** |
| 既有附录 14 链接 / 围栏 | `os.path.exists` / 正则 | **14/14 存在**；围栏 **4（配对）** |

**本轮未做**：任何真实 MaskCO 评分；L1 重跑或补跑；L2/L3 正式比较；
新的两天真实烟测；新状态采集；E2 训练；在线 rollout；任何计划安装；
修改冻结树或冻结数据；使用最终测试种子 `20260927`/`20260928`；改写已签署预登记或既有正式裁决；
`--skip-secret-scan` 绕过。

**保留不变**：`stable_online_gain=False`；`HISTORICAL_BACKBONE_UNVERIFIED`；
I3 负结果（`−144.680 [−191.109, −100.125]`）；D2 `not_qualified`；D3 G0 未通过；
D8 `CAPABILITY_GAP_BLOCKING`；E2 无合格结果；C0 `AWAITING_USER_BUDGET_RULING`；
**L1 正式批次 `BLOCKED`（绝不改写为 PASS）**。

**未提交任何新授权；本文件不申请也不消费授权。**
