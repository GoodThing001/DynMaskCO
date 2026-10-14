# B3-QP 独立只读复核者裁定（原样收存）

**复核者**：独立只读子进程（无我的先验结论）· **日期**：2026-10-14 · **基准**：`6d99940`
**复核方式**：只读；它把作者的验收套件在 `Path.write_text` 被 shim 的情况下重跑（拦截矩阵写出，`rc=0`，7/7 PASS，判定与 `raw/B3QC_lower_bound_stdout.txt` 一致），并自行插桩 `_force_return_all`、做数据集/合同内省与最大舱温数值扫描。
**本文件声明**：以下裁定**原样收存**，不作修饰。作者的回应与更正列于 §5。

---

## §1 逐项裁定

### 1. A1 — **PROVEN（源码穷尽）**
活跃树中 `quality_remaining` 的全部写入点：`:276`（新 lot 起点 = `initial_quality`）、**`:404-405` + `:409`（v1 唯一衰减写入）**、`:489`（`_advance_manifest_quality_v2`，v2 专属，分支在 `transition_segment:220-230` 互斥）、`:466`/`:512`（只**读**入 `DeliveredCargoRecord`）。`delivered_to_depot` 只有一个写入点（`:305` 取自 `:291`）。无加项、无额外因子。`min(...)` 只降 ⇒ `q_new ≤ q_old·exp(-k·dt)`。

> **最强反论**：**证明这件事的不是测试**。`test_d3_b3qc_lower_bound.py:151-153` 是**源码子串匹配**，即使再追加第三条写入路径它**仍然会通过**。给出证明的是穷尽检索，不是那个检查。

### 2. A2 — **PROVEN，但测量是同义反复；真正的实质是"分支不可达"**
顺序（F-a 问题）：`_apply_door_event`（`:243-244`）**改写 `temperature_after`**；`_thermal_violations` 在 `:298-299` 对**改写后的值**调用；该值在 `:302` 被存入状态并成为下一段的 `temperature_before`（`:216`）。⇒ 开门越界**在其出现的那一段内即被计数**，且后续段所依据的值正是那个被计数的值。**F-a 不漏。** 也不存在"段内极值被漏掉"：`_advance_temperatures` 单调（传热 ≥ 0，制冷是 `max(target, ·)` 夹紧），极值必在端点。合同校验 `low < target < high` 使 `dispatch_vehicle` 起点也不可能越界。

其数值扫描：任一段内最大舱温**恰为 target**（18.0 / 4.0 / −18.0）；作者记录的 `max_rate_temperature_c = 18.0333` 恰为 `18.0 + door_heat[0]/thermal_capacity[0] = 0.01/0.30`。

> **但是**：`ambient_branch_reachable()` 为 `False`，所以所测断言退化为"`before/after ≤ hard_upper` 时其均值 `≤ hard_upper`"——**结构上不可能失败，证据价值为零**。且记录器（`d3_quality_lower_bound.py:327-344`）**重新实现**了分支选择，而不是从生产帧读取，仪器可能与它审计的代码漂移。A2 的真实内容是"不可达"这一条加上 A3。

### 3. A3 — **PROVEN（退化情形不是漏洞）**
`d(exponent)/dT = activation_over_R / T² > 0`，`activation_over_R=(3500,4500,5500) > 0`（`:136-143`）。退化情形：`T ≤ 0 K ⇒ ValueError`（`:136-138`）⇒ 在其定义域上恒非减；合同使 T 远离 0 K。`duration_h ≤ _EPS` 与空 manifest 短路（`:393`）⇒ 零时长贡献恰为 0。数值核对：三类温区在 `[hard_low, hard_high] ∪ {ambient}` 上均单调。

### 4. A4 — **仅在更正后的表述下 PROVEN；作者的措辞是错的**
正确表述：交付记录在**返仓到达参数**处生成（`:286-291`，`_delivery_record` 用 `end_time`），而 `return_arrival ≤ tw_end[0]` 是 Ω 条件，故某 lot 的**装货→交付**时长为 `≤ tw_end[0] − now`（对任意 `now ≤ t_load`）——这正是界所需要的。对**已在舱** lot，该量**不是**"从 now 到交付的在舱时长"，而是**被 `tau_max` 界定且严格小于**它。`_advance_coldchain_to`（`strict_online_env.py:209-230`）无空档：每次推进都与同区间的 `transition_segment` 配对。

> **反论 / 新发现**：`_force_return_all`（`:658-664`）设 `v.return_finish = depart + travel`（`:273`），但只把冷链推进到 `depart`（`:272`），随后 `_advance_fleet(..., finish, ...)` 以 `depart_time=arrival_time=service_finish=finish`（`:391-402`）关闭货物。**记录到的卸载发生在"出发时刻"，比物理早了一个返仓行驶时间**——对界是保守的，但它**静默截断了记录物理中的货物衰减**。B3-QC 的两份文档都未提及。而被引为 A4 锚点的 `certify_plan` **根本没有引用 `tw_end[0]`**。它在 8 个 CAL 实例上插桩 `_force_return_all`：**0 次调用**；最大返仓到达 15.97 h vs `tw_end[0]=24`；0 次迟返；0 次热违规。

### 5. A5 — **NOT_ESTABLISHED（决定性缺口），且作为执行引擎的性质被 FALSIFIED**
独立核实（非采信作者）：`outcome_protocol_error` / `hard_vector_from_outcome` **在任何提交路径上都没有调用者**（仅测试与 `check_drift_parity.py`、`check_hard_gate.py`、`instance_validation.py`、`run_action_oracle.py`、`audit_baseline_service.py`、`freeze_coldchain_scale.py`）。`compose_certification` 与 `pre_install_reverify` **零非测试调用者**，且 `d3_composed_certifier.py:283-298` 因声明的 `quality_salability` 能力缺口**无条件返回 `installable=False`**。`_commit`（`:260-299`）与 `_plan_and_commit`（`:623-656`）**完全不做可行性检查**；replanner 写什么就提交并执行什么，失败只能由评估器事后发现。`certify_plan`（`run_exp_energy_c0.py:154-199`）**只查能耗**（甚至不查 `tw_end[0]`）；`certify_route`（`action_contract.py:200-203`）查 TW/容量/返仓，但只被 `d3_composed_certifier.py:210`（无生产调用方）与 `action_contract` 自身的修复循环调用。

⇒ TW/容量/返仓在 A-v1 生产线上成立**只是因为 `C0ReserveReplanner` 恰好自己检查了**（`run_exp_energy_c0.py:245-258`），而**热与品质在任何地方都没有可执行的在线检查**。Ω 是**评估层**集合。裁定："提交的计划必然属于 Ω" = **NOT_ESTABLISHED**；作为执行引擎的性质 = **FALSIFIED**。

> 作者关于"7 处真实旁路"的更正**正确**：8 个非测试 `certify_plan` 调用点，其中一个就是认证器自身（`d3_composed_certifier.py:233`）⇒ 7 处。

### 6. 计划无关性 — **成立**，并被锐化
`T_max` 是合同常量、`tau_max` 只用 `tw_end[0]` 与时钟（`:222-226`）⇒ 重规划、订单被丢弃、更晚装货都不能破坏**对确实被交付的 lot** 的不等式。失效模式**从不是计划**，而是成员性（§5）。作者漏掉的一个（可用性、非安全性）陷阱：`bound_for_state:273-286` 会为**每个被计划**的订单产出一行界，因此**后来被拒绝的订单也会得到一行永不被检验的界**。

### 7. 已装 / 未装 — 代码正确，但**已装分支被检验不足**
确认：只有 `cargo_manifest` 中的 lot 衰减（`:397`）；未装订单根本不在 manifest 中，并以 `quality_remaining = initial_quality` 在 `:270-278` 入舱。矩阵 `branch_coverage`：`in_cargo=34`、`not_yet_loaded=222`。**但** 14 个决策点中有 4 个 `in_cargo` 行数为 0，且整套检查只跑**1 个实例**。

> **作者未声明的隐藏耦合**：`bound_for_state:278` 硬编码 `iq = 1.0` 而非读 `env.initial_quality[inst_idx, o]`。仅因该数据集 `initial_quality` 处处为 1.0 才正确（三个 `m0_scale` npz 均已核对）。在 `initial_quality < 1` 的数据集上，**界会静默高于真实交付品质**。

---

## §2 总体裁定（拆成三条陈述）

| 陈述 | 裁定 |
|---|---|
| 条件蕴含 `ω ∈ Ω ⟹ Q_i^LB ≤ Q_i^delivery` | **PROVEN**（源码闭合：`:404-405` + `:142` + `:398-402` + `:298-299` + 返仓到达门） |
| Ω 成员性在生产提交入口被强制 | **NOT_ESTABLISHED**（不存在这样的代码路径；组合门无生产调用方且因能力缺口自我阻断） |
| 所问的问题（"对**所有**允许轨迹严格证明"） | **NOT_ESTABLISHED** |

**下界成立的精确条件**：轨迹 ω 满足 (i) `temperature_hard_feasible`（`coldchain_evaluator.py:358`）、(ii) `return_arrival ≤ tw_end[0]`、(iii) 未装分支 `t_load ≥ now` **且** 数据集 `initial_quality = 1.0`（或代码改用 `env.initial_quality`）、(iv) **未预冷分支不可达，或 ambient ≤ 每个出现类别的 hard_upper**（此处 `ambient=25` 高于 class 1(8) 与 class 2(−12)，故收紧形式**只因 `precool_min_temp_class = 0` 而成立**）。

**两个独立原因使该下界无法闭合认证目标**：(a) 提交时无任何东西强制 Ω；(b) **即便承认 Ω，测量显示该下界从不触发**：`k(T_max)·24 h = 0.0709 / 0.0605 / 0.0249` ⇒ 比值下界 ≥ 0.9316 / 0.9413 / 0.9754，全部高于 0.8/0.85/0.9；矩阵在**每个**决策点都报 `n_unsalable_lower_bound = 0`，6 个 CAL 实例的最小实测交付品质为 0.972（界 0.932）。⇒ 收益（若有）只能来自**更紧的 `tau_max`**，而非更多搜索或仅加一道提交门。

---

## §3 未经检验 / 检验不足的假设与最小验证测试（复核者原列）

| # | 假设 | 最小测试 |
|---|---|---|
| A-c1 | 未装分支 `Q_start = initial_quality`，而代码用字面 `1.0`（`:278`） | 对每个 npz 断言 `np.all(env.initial_quality == 1.0)`，或改用 `env.initial_quality[inst,o]`。**最廉价的潜在证伪点** |
| A-c2 | 每个被交付订单都被某个决策点的某行界覆盖 | 交付 id 集合与界行 id 并集的差集，显式报告；当前从不打印不匹配数 |
| A-c3 | A1 无第三条写入路径 | 变异测试：给 `_advance_manifest_quality` 追加一个虚假因子，确认套件**失败**（今天会通过） |
| A-c4 | 无"段内热极值被漏掉" | 对每个真实段的内部点采样 `_advance_temperatures`，与 `hard_upper` 比较 |
| A-c5 | `_force_return_all` 语义（卸载在出发 vs 到达） | 构造一次触发它的运行，断言记录 `return_arrival` 与货物时钟截断的差异 |
| A-c6 | Ω 成员性在提交时被强制 | 用故意不可行的 suffix（超 TW / 超容量）跑 D3 生产路径，证明它在**执行前被拒绝**。今天它被执行，只被事后贴标签 |
| A-c7 | `_validate_state` 不会因热越界抛错 | 断言 `thermal_violation_count > 0` 可在无异常下达到（`:533-557` 不校验温度） |
| A-c8 | 容量下限从不绑定 | 解析检查 `zone_load[c]/thermal_capacity[c] ≤ hard_upper[c] − target[c]` |
| A-c9 | BND/ADV 的普适性 | 多实例 BND（≥8）、多约束 lot 的 ADV、非零 `booking_horizon` 的 WAIT 组合；今天 ADV 是 1 实例 × 1 订单 × 9 网格且只有 **2 个不同最小值**（按作者自己的判据即 `INEFFECTIVE_SEARCH`） |

---

## §4 复核者指出作者未报告的问题（原样）

1. **A2 的仪器是复述，检查是同义反复**：作者把"2134 采样、0 违规"当作载荷证据；在 ambient 分支不可达时该比较**不可能失败**。
2. **潜在死缺陷**：`quality_lower_bound()` 默认 `ambient_reachable=True`（`:234-237`），任何直接调用都会静默施加设计说不需要的 ambient 放宽，与设计矛盾；只有 `bound_for_state` 绕开了它。
3. **文档间状态漂移**：A4 在 `B3QC_LOWER_BOUND_DESIGN.md:103` 是"未做端到端独立验证"，却在 `B3QP_LOWER_BOUND_PROOF_AUDIT.md` 被提升为 PROVEN。
4. **未记录的货物时钟截断**（§4）——记录品质比物理更乐观的唯一之处，且对 A4 的"交付 = 返仓到达"表述是载荷性的。
5. **未声明的副条件 `ambient ≤ hard_upper`**（§2）。
6. **Ω 文本过时**：`omega_definition()['precool_branch']` 仍称两分支都在 Ω 且"由 ambient 放宽覆盖"，与实现的"不放宽"路径矛盾。
7. **BND 覆盖比名字暗示的窄得多**：`falsify_bound_against_delivered:362-374` 在交付 id 不在界行中时静默 `continue`；`--heavy-cap 14` 只覆盖运行**前 14 个决策点**（最后捕获 clock 3.112，运行继续到 ~16）⇒ 定向过滤 + 短捕获，50 条交付中未知份额从未被比较。
8. **"全部已捕获决策点"字面正确但易被误读**为"整段运行"。
9. **该下界作为认证装置在数值上是空的**（从不低于任何阈值；真实数据上余量约 0.04）。作者称 τ 是"偏松主因"，但未说明**不会有任何订单被拒绝**。
10. A5 上作者是对的，独立重数确认（8 个非测试调用点、含认证器自身、7 处真实旁路）。

**复核者结论**：条件不等式**确实可证**，且在 Ω 上**未发现任何健全性反例**——但**它不是所问的问题**，且缺口**不是数学问题**。`Ω` 是**评估层**集合，没有提交时执行者；而且即便承认 Ω，该下界**从不触发任何阈值**。建议保持 D8 关闭，分别记录两条判定（条件下界 **PROVEN** 并附四条显式副条件；提交时 Ω 强制 **NOT_ESTABLISHED**），并把**更紧的 `tau_max`**——而非提交门或更多搜索——视为让下界具备认证价值的唯一路径。

---

## §5 作者回应与已完成的更正（本轮）

| 复核者指出的问题 | 处置 |
|---|---|
| A-c3：A1 检查是子串匹配 | ✅ **已修**：新增 `A1X` 检查**枚举**全部写入点并与期望集合 `{276,404,405,409,489}` 精确比对 |
| §4-1：A2 仪器是复述 | ✅ **已修**：新增 `RATE` 检查**插桩生产函数 `quality_rate_per_hour` 本身**（记录其真实入参），不再自行重推分支选择 |
| §4-2：`quality_lower_bound` 默认 `ambient_reachable=True` | ✅ **已修**：默认改为 `False`，并在 docstring 写明必须显式声明可达性 |
| §4-5 / §2(iv)：副条件未声明 | ✅ **已修**：新增 `tightened_bound_side_conditions()` 并在 `SIDE` 检查中报告——**ambient(25) 高于 class 1(8)/class 2(−12) 的 hard_upper，收紧形式只因这些类别保持预冷而成立** |
| §4-6：Ω 文本过时 | ✅ **已修**：`omega_definition()` 的 `precool_branch` 改为与实现一致并标注已被独立复核更正 |
| A-c1 / §1-7：`iq=1.0` 硬编码 | ✅ **已修**：改读 `env.initial_quality[inst,o]` 并记录 `iq_source`；`IQ` 检查确认来源为数据集 |
| A-c2 / §4-7：覆盖从未量化 | ✅ **已修**：新增 `COV` 检查，**双向**报告：**50 条交付中 32 条被覆盖（64%），18 条从未被任何界行覆盖**（已列出）；并说明原因是"只捕获前 14 个决策点" |
| §4-9：下界数值上是空的 | ✅ **已记录**：新增 `BIND` 检查，报告**逐类是否触发阈值**——**全部未触发**（0.9316/0.9413/0.9754 vs 0.8/0.85/0.9），并写明"收益必须来自更紧的 τ" |
| §1-4 / §4-4：`_force_return_all` 截断未报告 | ✅ **已记录**：新增 `FORCE` 检查（本实例 **0 次调用**，最大返仓到达 15.97 h vs 24 h，0 迟返），并写明该截断使**记录**品质比物理乐观，**削弱**（而非加强）BND |
| §1-4：`certify_plan` 不引用 `tw_end[0]`，却被引为 A4 锚点 | ✅ **已更正**（见 `B3QP_LOWER_BOUND_PROOF_AUDIT.md` §2/A4 与 §6） |
| §1-4：A4 措辞错误 | ✅ **已更正**：在舱 lot 的量是"**被 `tau_max` 界定**"，不是"从 now 起算的在舱时长" |
| §4-3：文档间状态漂移（A4） | ✅ **已更正**：`B3QP_LOWER_BOUND_PROOF_AUDIT.md` 的 A4 不再被并入无条件 PROVEN；改为"**更正后表述下 PROVEN，且以 Ω 成员性为条件**" |
| A-c4～A-c9 | ⏳ **未做**（超出本轮授权范围，登记为后续项） |

**未变更**：D8 保持关闭；`CONSERVATIVENESS_NOT_ESTABLISHED` 保留；未引入 `PROJECTED_SATISFIED_CONSERVATIVE`；未真实安装、未改 `_commit`、未触碰冻结数据与最终种子。
