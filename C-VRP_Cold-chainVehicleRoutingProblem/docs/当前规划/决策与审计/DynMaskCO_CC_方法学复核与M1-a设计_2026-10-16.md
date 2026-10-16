# DynMaskCO-CC：M0 方法学复核与 M1-a 设计附录

日期：2026-10-16。审查基准：`a7c1118dfba26f96e5f08f4b7517c4d27fc988ea`。

性质：新增静态审查与技术设计；不是预登记签署、工程验收或科学执行授权。原 M0 五份交件、已签署预登记、历史裁决和原始证据均保留。本附录纠正推断并提出后续接口要求，不修改原文，不把建议写成已实现能力。

## 1. 范围、依据与当前结论

实施前已 fetch；本地 HEAD、`origin/main`、`origin/codex/o0cc-cal-phys` 均为上述基准。工作区已有 `CC_Compare/MAPT_code` 脏状态，不属于本任务。交付只增加本文件。

权威依据依次为[项目当前状态](../../../项目当前状态.md)、[执行进度表](../结果与进度/执行进度表.md)、既有决策/审计与用户本次边界。M0 审查对象：

- [总体路线图](DynMaskCO_CC_下一阶段总体科研路线图_2026-10-16.md) §4.1、§5.1。
- [L2 草案](L2_可认证决策质量_七要素预登记草案_2026-10-16.md) §1、§3、§7bis、§10。
- [L3 草案](L3_strict_online效用_预登记设计草案_2026-10-16.md) §8bis 及末尾关于 CI 的限制。
- [C0 预算与强基线清单](C0预算与强基线_资格缺口及用户决策清单_2026-10-16.md) §2、§6、§8。
- [资格矩阵](DynMaskCO_CC_下一阶段资格矩阵_2026-10-16.json)。

结论：**`FULL_INSTALLABILITY_NOT_AVAILABLE`**。现有能耗、路线与修复一致性检查可提供有限诊断，但决策期品质/鱼可售的正式能力缺口尚未闭合。当前不能据此进入 L2 完整质量主门。H1 也不能在主指标未定义、认证未就绪时作正负统计裁决。

本轮没有执行科研模块、真实/合成验收、模型前向、rollout、采集、训练或计划安装。后文所有合成用例都是待授权方案。L1 已接受 Option A 的诊断收口，但正式 `BLOCKED` 不变；机会已消耗，不重跑、不补样、不启用 Option B。本附录不把 L1 改成 PASS，也不由其 BLOCKED 自动推出新 L2 的正负结论。

## 2. 勘误 A：四层认证不能合并为一个布尔值

### 2.1 现有代码的实际保证

| 层次 | 真实调用与保证 | 不包含的保证 |
|---|---|---|
| C0 能耗预算 | `run_exp_energy_c0.certify_plan`，计算车队已发生及计划完成的能耗，与 B 比较，单位 kWh | 不独立检查 TW、容量、完整承诺或鱼可售；实现按既有 `B + 1e-9` 判据，不能把该能耗数值判据扩展为其他比较容差 |
| 路线及承诺合法性 | composer 先检查车辆、揭示、已服务、重复，再组合 `validate_full_partition`、`validate_repair_plan`、`certify_route` | 不证明输入基线与真实完整承诺相符；允许 `deferred` 的分区不等于每个已接订单都有可执行服务安排；温区兼容不等于温度轨迹或可售认证 |
| 决策期品质/鱼可售 | 未传入真实品质认证时，`quality.state = QUALITY_CAPABILITY_GAP`，实际字符串为 `CAPABILITY_GAP_BLOCKING` | 只读品质投影不是已证明适用的保守认证；名义投影通过不能消除缺口 |
| 最终可安装资格 | composer 返回 `installable = bool(ok and not blocking_gaps)` | 不是实际安装、D8 持久化回执或用户执行授权；还依赖输入上下文及品质回调本身可信 |

关键源码：[composer](../../../scripts/simulation/d3_composed_certifier.py) :55–58、:112–243、:283–309；[能耗认证](../../../scripts/evaluation/run_exp_energy_c0.py) :112–199；[路线认证](../../../scripts/simulation/action_contract.py) :142–203；[修复器回调](../../../scripts/simulation/d3_geo_adapter.py) :553–579；[双臂协议](../../../scripts/simulation/d3_l1_two_arm_protocol.py) :310–350、:475–481、:530–543。

`certify_plan` 会滤去已服务节点、补入已承诺下一段；`c0_completion_energy` 会按揭示时刻抬高出发时刻，但不检查 TW 上界。composer 的前置合法性检查有实际作用，不能直接绕过后仅调用能耗函数。完整认证还须证明已承诺段只计一次、锚点与真实车辆相符、等待/回仓/卸货语义一致，不能只凭函数名认定这些前提已满足。

### 2.2 dict 与回调接口的差异

`compose_certification(...) -> dict`；`d3_geo_adapter.stepwise_sample_d3(..., certify_fn=...)` 在 :553–563 解包 `certify_fn(P) -> (ok, detail)`。不能直接接入 composer，也不能返回 `bool(report)` 或仅返回 `report['ok']`。

从代码可以构造如下逻辑状态：四项基础检查通过、无其他 reasons，品质回调缺失，则 `ok=True`、`installable=False`、品质能力缺口仍在。这是静态分支结论，本轮未执行该反例。若只映射 `ok`，将错误放过完整资格门。

另外，sampler 在检查回调返回的 ok 之前就设置 `diag['certified']=True`，所以它表示“调用过回调”，不能单独作为完整认证通过证据。成功的 detail 没有完整保存在该 diag 中；失败/超时返回输入计划，不能从返回对象反推出被拒候选的内容。

原 L1 的真实调用者 `d3_l1_l1_formal_run.py` :417–430 传入的是完整 `P0`，并另存 partial。本附录不声称原 L1 把 partial 当成了安全回退。新接口仍必须区分完整基线、修复工作计划与待认证候选，防止未来调用者混淆。

### 2.3 还缺什么

- composer 的 `mutable_ids` 只有形参，函数体没有使用；传入此参数不产生车辆修改范围检查。必须另行验证不可修改车辆、锚点和承诺前缀。
- `plan_vehicle_set` 直接检查多出的车辆；相对于 base 的 keyset 检查不能证明 base 自身没有漏掉真实车辆。必须先验证完整 P0。
- `full_partition` 可以把可见未服务订单放进 deferred。可见订单不等于已接受订单，承诺集合必须独立提供与核对；不能把已接受订单移到 deferred 后称为“完整可行”。
- composer 接受任意 `quality_check_fn`，其返回 True 只会产生接口层的 `CHECKED_SATISFIED`。测试假回调不能成为正式品质能力证明。正式入口须绑定已审核的实现、适用域与证明身份，禁止外部任意注入。
- 既有质量投影标记保守性未建立。[既有条件下界审计](../../../../research/review_3080_20261010/d3_g0_evidence_root_v1/B3QP_LOWER_BOUND_PROOF_AUDIT.md) 的论证在限定 Ω 内、附副条件成立，但生产端没有强制成员性；不得把这一限制改写成“全部下界已被否定”，更不得跳过运行时适用域检查。v1 品质比例结论不能未经证明迁移到 v2 鱼感官分值。

因此 M0 中的“真实完整认证”应收紧为：**真实基础检查已存在，完整可安装认证尚不可用**。只保留有限诊断，不把能源/路线通过率当作 L2 完整质量主门通过率。

## 3. 勘误 B：H1 词表及主质量指标

### 3.1 修正原推断

总体路线图 §5.1 与 L2 草案 §1 把“不显著/未达到门槛”写成 H1 被证伪，推断过强。以下是供新协议采用的裁决词表，**不追溯修改任何既有正式裁决**。

先统一正向差值为“F 更好时 Δ > 0”，再按先决资格、主检验、方向依次判定：

| 词条 | 使用条件 | 允许的解释 |
|---|---|---|
| `BLOCKED` | 身份、完整认证、B、主指标、状态/数据资格或预声明阈值等先决项缺失 | 不能执行或裁决该主检验；记录缺项为 `UNRESOLVED`，不转成负效应 |
| `PASS` | 全部资格成立，预登记主指标均值达到用户签署的 δ_L2、95% CI 下界 > 0，并满足已声明认证失败/安全门 | 只支持已登记方法、资源及状态范围的正效应 |
| `NOT_ESTABLISHED` | 资格成立但正效应门未达到，例如 CI 跨零/触零、精度不足，或均值未达 δ_L2 | 未建立预声明效应；不是统计等价，不证明学习机制普遍无用 |
| `NEGATIVE_DIRECTION` | 建议新协议以 Δ < 0 且 95% CI 上界 < 0 作为充分方向证据；精确判据仍待签署 | 对登记比较显示反向效应，不外推为整个 MaskCO 范式被证伪 |

负的点估计但 CI 跨零，应为 `NOT_ESTABLISHED`，另报 observed_direction=negative。均值为正但未达 δ，不是负方向。认证率若显著下降，应按预登记的独立安全/可用性端点处理，不能自动推导“所有学习分歧均无价值”。非劣效/等价需预先定义边界和对应检验，本次不新增边界或等价结论。

### 3.2 L2 主质量指标审计

| 必需定义 | M0 现状 | 审查状态 |
|---|---|---|
| 唯一函数 | 提出 candidate / accepted / certified 等层次，但未绑定唯一可复现的主标量函数和版本 | `UNRESOLVED` |
| 单位 | 未区分原始距离、规范化 coldchain_cost、不同货物品质单位与 A-v1 货币效用 | `UNRESOLVED` |
| 方向 | “quality 越高越好”与 accepted=`min(J0,J)` 的成本方向并存，Δ 尚未唯一映射 | `UNRESOLVED` |
| 认证失败处理 | 声明保留固定分母，但“仅 fully certified 的质量均值”没有给失败/超时/缺失的主标量映射 | `UNRESOLVED` |
| 预声明阈值 | δ_L2 待定，不能继承 A3 δ3 或其他阶段门槛 | `UNRESOLVED` |

实例聚类、配对与 bootstrap 的描述不能补足以上定义。“失败留分母”不等于已定义全分母估计量；仅对两臂都认证通过的子集计算质量，可能有选择偏差，只能作额外描述，不能悄然成为主分析。

### 3.3 候选指标与取舍（未选定）

| 候选 | 函数/单位/正向差值建议 | 局限及所需决定 |
|---|---|---|
| 可见计划规范化成本 C | 候选入口 `visible_state.evaluate_visible_plan`；合同规范化式 `D/D_scale + λq·Q/Q_scale + λe·E/E_scale`，一致尺度下无量纲；Δ=`C_DGeo − C_F` | 该入口会补全剩余可见订单；必须明确是完整承诺计划还是包含补全策略的估计量，不能直接称为固定候选自身质量；物理认证另作硬门 |
| 剩余完成距离 D | 固定已发生常量与回仓范围后，原始距离单位；Δ=`D_DGeo − D_F` | 可隔离路径效率，但不是燃油、coldchain_cost 或在线净效用，不能单独支持冷链品质增益 |
| 货物品质/可售指标 | 按真实合同版本分别报告品质比例、鱼感官分值或可售率；明确每类方向与阈值 | 不同单位不能直接平均；若做标量需签署权重与业务解释；鱼可售硬约束不能自动货币化 |
| 完整认证成功率 | 每个固定状态的 0/1，单位为比例，Δ=`rate_F − rate_DGeo` | 反映可用性，不是通过后的决策质量；不能替代主质量函数或掩盖品质能力缺口 |
| 含安全回退的计划成本 | 若 P0 本身完整认证，按预声明接受规则定义有效成本：候选完整认证且被接受才取候选成本，否则取 P0 成本；正向差值为 DGeo 减 F | 可定义固定分母的部署提案价值，但必须保留候选失败标签；P0 未认证/指标缺失时不能填 0 或默认为安全，品质全局缺口仍为 BLOCKED |

可优先讨论“含安全回退的成本 + 独立完整认证率/失败分类”，但本次不选主指标、不签署 δ_L2、不把这些候选写进正式预登记。candidate 与 accepted 的效果须分开；若共同接受器拒绝所有候选，不能把候选差异当成有效计划收益。[coldchain_contract.py](../../../scripts/coldchain/coldchain_contract.py) :542–558 的规范化成本与 A-v1 的 revenue−fuel−rejection loss 是不同函数。

## 4. 勘误 C：L3 决策层分辨率

总体路线图 §4.1 与 L3 草案 §8bis 不能据“接单集合相同”或“CI 跨零”推出整个在线系统不可分辨，或任何修复差异都不可能产生效用。相同接单也可能对应不同车辆、顺序、回仓时间、燃油、品质、超时与回退。L3 草案后文“不把跨零当作等价”的限制应同时约束 §8bis。

建议下面四组解释性记录；它们不替代预登记的日级主效用比较，不增加抽样机会，也不据其筛天。

| 诊断层 | 最小记录 | 比较解释 |
|---|---|---|
| 接拒动作 | 逐可见订单 accept/reject/defer/unanswered、决策时刻、理由；资源失败与业务拒绝分开 | 按外生 day/order_id/reveal event 对齐；策略分叉后的内生事件不强行当成同状态配对 |
| 计划结构 | 完整车辆归属、锚点、承诺前缀、suffix 顺序/有向边、WAIT/RETURN、到达/服务/卸货时刻、cargo manifest | 比较完整结构而非仅接单向量或短 hash；车辆标签与物理车辆差异须区分，结构规范化仅作诊断且事先声明，不用于放宽旧 parity |
| 费用分解 | 日级 revenue、fuel、rejection loss，以及纯距离、能耗 kWh、品质/硬违规记录 | 按原合同核对 `ΔJ = Δrevenue − Δfuel − Δrejection_loss`；不同费用抵消不代表无决策差异；能耗与品质不擅自加入 J |
| 时间/回退 | 各阶段预算起止、实际 wall time、超时率、异常、fallback 原因、回退 P0 身份及其资格 | 相同动作可能源于共同回退；超时不得并入普通业务拒绝或静默丢失 |

诊断按日/实例聚类，缺失显式报告，不作事后多指标挑选来替换主门。若主效用 CI 跨零，只能说该登记检验未建立正效应，不能证明统计等价或全系统无分辨率。即使观察样本中完整动作/计划逐项相同，也只支持该样本和该实现的观察一致性，不能推出总体等价。若主检验本身合格，接单向量相同也不单独取消其效用解释；机制归因则仍需与记录能力相称。

## 5. M1-a 最小认证适配器设计

### 5.1 不可变状态上下文

每个固定状态建立 `CertificationContext`；两臂使用独立只读副本但内容身份相同，禁止共享可变 env/车辆/计划对象造成串扰。必须提供：

1. **完整 P0**：修复前 incumbent，完整车辆 keyset、物理锚点和 suffix；独立于修复工作输入 `Pwork`、mask 删除后的 partial、完成候选 `Pc` 保存。P0 与真实车辆/承诺状态先校验，不允许 `base_plans=Pc` 自证。
2. **完整承诺**：已接受 A_t、已拒绝、待决与 deferred 的显式集合；已服务、不可变执行前缀、在途 committed_next/剩余段、已装货 cargo、回仓/卸货状态及车辆容量。不能从“可见集合”或某个 suffix 推断完整 A_t。快照顶层未显式导出该业务集合，opaque replanner state 是否包含它也必须逐字段验证，不能假定已具备。
3. **可见信息**：state_id/epoch、inst_idx、clock、served_mask、visible_mask、外生揭示身份和真实 ID 映射、可见订单的坐标/需求/TW/服务时长/温区/初始品质；不可见属性不可传给修复器。composer 所需揭示判定应由只读门控提供，不能因此开放未来属性。先验证 ID 类型、取值范围、映射唯一性，再索引数组，避免负索引或隐式截断。
4. **修改边界**：mask_set、protected、mutable_ids、动作空间/车辆集合、平局规则、完整基线 hash；未 mask 的所有权与相对顺序、不可修改车辆全计划和全部锚点/承诺段必须保持。
5. **物理与业务合同**：精确 contract/version/hash、单位换算、速度、booking horizon、等待/服务/回仓/卸货语义、CC state、温区支持、实际 C0 及 evaluator 身份。核对初始 anchor_load 在容量内，不能依赖非空 suffix 循环间接检查；核对能耗等待至 horizon 与路线回仓时限使用相容的时间语义。既有类型/位级恢复口径不变，不用新的规范化 hash 冒充旧 bitwise parity。
6. **预算及品质资格**：有限且有效的 B（kWh）、其协议登记/适用车队与日范围/状态绑定 SHA；已审核品质认证实现身份、合同版本、证明/适用域 Ω 及成员检查。缺任一必要资格必须拒绝完整通过。
7. **运行/证据身份**：两臂共同资源预算、截止时刻与独立时钟起点、输入/源码/候选/指标/接受规则身份、追加式认证收据与异常去向。运行后核对输入 epoch/hash 未变，并由独立只读复核核对收据。

已接受未完成订单必须恰好由不可变 committed 段或完整候选 suffix 覆盖一次；已取货 cargo 必须具有合法回仓卸货完成路径。未接受可见订单可以按协议留待决/拒绝。若协议允许已接受订单延后，须提供已认证的完整承诺完成安排；单独写入 deferred 不能视作履约。上下文缺失时是能力/输入缺口，不擅自修订承诺语义。

Ω 是轨迹集合，不能仅以“当前状态属于某个温度区间”冒充成员性保证。品质证书必须写明量词：是认证固定 Pc 的完整完成轨迹，还是对后续所有允许重规划/等待的轨迹给保守保证。前者不能自动替代后者；后者须说明后续每次提交及执行段如何持续强制热学/回仓等硬门，且不得把 force-return 违规轨迹当成已排除却不实际约束。若仅能做名义固定计划投影或生产端强制仍缺失，应继续保留能力缺口与 D8 阻断。

### 5.2 参数映射

| composer 参数 | 来源及前置条件 |
|---|---|
| `env, inst_idx, clock` | 冻结决策状态的只读环境；只允许揭示判定及可见/已承诺订单属性，核对节点数、单位与 epoch |
| `vehicles, served_mask` | 原状态真实车辆/CC/cargo 与服务状态的只读副本，不从候选重造 |
| `plans` | 完成的 `Pc`，先核对节点合法、全承诺覆盖与结构边界 |
| `mask_set, protected, deferred` | 相同固定状态、相同规则导出并记录 hash；deferred 单独满足承诺覆盖规则 |
| `B, contract` | 新协议签署后的预算绑定与真实合同对象；不从测试夹具、profile 的 scale 或旧实验数字猜测 |
| `base_plans` | 独立、已核对完整的 `P0`；必填 |
| `mutable_ids` | 同一动作空间导出；适配器独立强制检查，因为当前 composer 不使用该参数 |
| `quality_check_fn` | 仅由已审核、版本/证明/适用域均匹配的品质实现注册表解析；不能由方法臂或命令行随意传入 |
| `project_quality` | 可保留真实只读投影作为有限诊断，其通过不取代品质证书 |
| `stress_quality_schedule` | 必须记录采用的 schedule 身份；不得把 latest schedule 自动声明为最坏情况或保守证明 |

### 5.3 返回映射及调用链伪代码（未实现）

```text
固定状态材料 -> 核对完整 P0/承诺/可见边界/合同与身份
            -> 两臂同构 CertificationContext
            -> stepwise_sample_d3(..., certify_fn=make_callback(ctx))
            -> Pc 的完整认证收据 -> 离线指标/共同接受规则
```

```python
# 设计伪代码；不是可运行入口，也不构成任何验收结果。
def callback(Pc):
    pre = validate_context_plan_and_deadline(ctx, Pc)
    if not pre.ok:
        return False, persist_refusal(pre, ctx, Pc)
    try:
        report = compose_certification(
            ctx.env_view, ctx.inst_idx, ctx.clock, ctx.vehicles,
            ctx.served_mask, Pc, mask_set=ctx.mask_set,
            protected=ctx.protected, deferred=ctx.deferred,
            B=ctx.B, contract=ctx.contract, base_plans=ctx.P0,
            mutable_ids=ctx.mutable_ids,
            quality_check_fn=approved_quality_impl_or_none(ctx),
            project_quality=True,
            stress_quality_schedule=ctx.stress_quality_schedule)
        schema_ok = validate_report_schema_and_required_components(report)
        capability_ok = approved_quality_scope_holds(ctx, report)
        post = validate_unchanged_state_plan_and_deadline(ctx, Pc)
        full_ok = bool(
            schema_ok and capability_ok and post.ok
            and report['ok'] is True and report['installable'] is True
            and report['legality_ok'] is True
            and report['quality']['state'] == 'CHECKED_SATISFIED'
            and not report['blocking_capability_gaps']
            and all(report['components'][k]['ok'] is True
                    for k in REQUIRED_LEGALITY_AND_CERTIFICATION_CHECKS))
        receipt = persist_full_report_and_candidate(ctx, Pc, report, full_ok)
        require_durable_receipt_and_deadline(receipt, ctx)
        return full_ok, receipt
    except Exception as exc:
        return False, persist_typed_exception_or_abort(ctx, Pc, exc)
```

`REQUIRED_LEGALITY_AND_CERTIFICATION_CHECKS` 是四项 legality gate 加 `repair_plan/full_partition/route_feasible/energy_budget`；缺必需字段、ok 非 bool、not_run、必需物理值/B/E/指标非有限或报告矛盾均不能默认通过。空路线的诊断 slack 可有约定的无穷值，须按字段语义区分，不能把这种哨兵当作物理量合格。品质 scope 的审核身份是额外必要条件，不能由字符串 `CHECKED_SATISFIED` 自证。

品质缺口时返回 `(False, detail)`，detail 明确 `FULL_INSTALLABILITY_NOT_AVAILABLE` 与原始 `CAPABILITY_GAP_BLOCKING`，仍保存合法范围内的能源/路线结果。预检查拒绝非法 ID/未来/重复等时，不强行继续能耗或品质调用。详细收据保存原始 report、P0/Pc 完整结构与 SHA、承诺/可见/mask/车辆/B 登记/合同/品质证明身份、时钟、失败路径；成功 detail 也必须持久化，不能只靠 sampler diag。

收据写盘失败、来源变化或状态/计划被修改应中止资格运行；不能返回无证据的 True。普通认证异常显式拒绝，账本/身份错误升级为硬停止；中断/终止应保留终态并向外传播，不能吞成成功。既有 composer 内部 `_guard` 会捕获 BaseException，后续实现须确保独立监督器及终止状态优先，不能仅凭回调布尔值认定完成。超时覆盖认证与收据落盘，完成后再越界仍拒绝，不放宽原预算。

新 L2 必须分别记录“完成 mask 重构”“认证是否通过”“共同接受器是否接受”和“实际安装”。现协议 `V2` 受 failure_reason 影响，`V3` 依赖回调语义；不改旧 L1 的 V2/V3 定义或裁决，只在新身份下增设明确字段。候选在回调中被拒时仍保存其原件，不能把 sampler 返回的回退计划记成候选已认证。

**离线 L2 不调用 `run_repair_transaction` 或 `InstallLedger.finish`**：后者可触发真实 `install_fn`。本设计输出离线证书，不安装计划，也不以空安装回调生成 D8 合格结论。完整证书将来也不等于实际执行授权。

### 5.4 fail-closed 语义

| 输入/认证事件 | 必须结果与证据 |
|---|---|
| 品质回调缺失、适用域未证明/不匹配、测试假回调 | 完整资格不可用；保留 gap，不以投影或假 True 消除 |
| TW/容量/温区/回仓不合法 | 对应路线拒绝，保存 vehicle/order/reason；不以能耗足够覆盖 |
| 重复、未来、已服务、无效 ID、错误车辆 | 前置拒绝，后续检查 not_run；不删除节点后重判 |
| P0 缺失、基线自证、承诺漏项、accepted 仅在 deferred | 输入/覆盖缺口拒绝；不从候选反填 P0 或 A_t |
| committed 段重复计数、物理锚点变化、不可修改车辆被更改 | 承诺/修改边界拒绝，记录首个结构差异 |
| 能耗越限、非有限 B/E、B 未登记或合同/单位不符 | 能源失败或身份硬停止；B 不由 profile 的归一化尺度生成 |
| 品质函数异常、缺字段、非 bool/非有限输出 | 拒绝并记录类型与路径；绝不 bool(dict) 或以缺失当满足 |
| 认证前后超时、证据落盘失败、上下文改变 | 失败/硬停止；保留候选及不完整收据，不转有效结果，不重采补样 |

`B=710.6548152249378` 只保留为 I3 旧实验来源数字，不登记为新 L2 预算；原夹具 300.0、隔离值 700/1e9 也不能继承。新预算应单独以版本化配置绑定协议/车队/日与状态身份，不改旧合同或冻结树。

## 6. R、D-Geo、cond_hist 的比较层次

| 名称 | 现有层次 | 能否直接做固定状态 L2 配对 |
|---|---|---|
| D-Geo | 固定 mask/动作空间内的规则打分与逐步重构；与 F 的比较可只改变打分源 | 只有共同 P0、mask、合法动作、资源、接受器、完整认证及指标均固定后才成立；旧 L1 分歧不等于新的 L2 正效应 |
| R / CC-LNS | `cc_lns_replanner.py` 中有自有 mask、regret-2 重构、增量距离/动作 ID 平局规则、循环预算及 J_vis 接受器的完整搜索包装 | 旧 R 包装不等于相同 mask 下的一个打分替换；若提取 R-core，须声明新变体/身份/资源口径，不能借用旧 R 数值作直接配对 |
| cond_hist | `scripts/evaluation/scenario_saa.py` 所处的条件场景、SAA 与接拒/重规划系统层 | 不是 `certify_fn` 或固定 mask 修复臂；不能强行添加为同质 L2 标量配对臂。将来若共同完整系统只换修复器，须另定系统身份与协议，不能自动等同历史 cond_hist |

因此修正 M0 将 R 与 cond_hist 都直接列为 L2 必备平行臂的表述：保留基线清单，但先做接口/层次兼容性审查；不在本轮实现适配或改动比较路线。旧 backbone 仍未完成当前部署资格，历史结果不能替代新身份认证。

## 7. 隔离合成验收方案与正式资格依赖

### 7.1 待授权的合成方案，当前 NOT_RUN

使用新建、非冻结、无 F 权重/模型前向、无真实数据/状态采集的确定性微型状态。使用显式标记 `SYNTHETIC_ONLY` 的有限 B 和物理参数；不消耗任何正式采集或 L1 槽位，不安装计划。预期矩阵如下：

| 用例组 | 构造 | 预期判定 |
|---|---|---|
| 接口负控 | composer dict 直接接回调；只取 ok；certified=True 但 ok=False | 均不得作为完整通过；保留原始输出 |
| 基础正控/品质缺口 | 合法路线、真实基础函数通过，quality=None | 能源/路线有限诊断通过；完整资格仍 False / `FULL_INSTALLABILITY_NOT_AVAILABLE` |
| 回调接线正控 | 仅 harness 的 TEST_ONLY 假品质 True/False | 只验 tuple 与收据传递；正式资格注册表必须拒绝 TEST_ONLY，不能产出正式 installable 正结论 |
| 独立约束负控 | 分别注入 TW、容量、回仓、能耗越限；重复/未来/已服务/越界 ID | 对应失败；非法集合后续组件不运行；不修数据后重判 |
| 承诺负控 | accepted 漏项/放 deferred、在途重复、缺 cargo 返回、P0/Pwork 混用 | 覆盖/物理状态拒绝，不能静默缩小承诺 |
| 边界负控 | 非 mask 移动、非 mutable 车辆变化、锚点/前缀改变、漏车辆 | 各自拒绝并给出定位证据 |
| 身份负控 | B 登记/合同/品质证明版本/适用域/状态 epoch/候选发生变化 | 硬停止或拒绝完整资格；原始失败保留 |
| 异常/资源负控 | 认证抛错、非 bool、NaN/Inf、认证后超时、收据写盘失败 | 全部拒绝/中止；无无证据 True，零安装 |
| 证据复核 | 独立只读进程复算输入/候选/源码/收据 SHA、结果计数与未运行项 | 不一致 fail-closed，不删失败，不挑选通过用例 |

未来只有真实品质实现及适用域证明通过独立资格审查后，才可安排“完整认证正控”。若尚无该实现，合成接线验收即使通过也必须继续标记完整能力不可用，不能用假认证器补齐正控。

### 7.2 下一次正式资格门（未授权）

依赖顺序是：

1. 上下文与合成接口工程验收；保留当前品质阻断，独立确认零安装。
2. 决策期品质/鱼可售认证的真实实现、适用合同、保守性证明与运行时 Ω 成员检查闭合；完整承诺与回仓认证也必须闭合。缺口存在则停在有限诊断，不能请求正式 L2 质量比较。
3. 用户选择并签署主指标函数/单位/方向、candidate/accepted 层次、全分母失败估计量、δ_L2、统计/安全门、B 的来源与绑定。不能用 I3 或 A3 数字替代。
4. 新协议明确可用状态和数据身份、L1 BLOCKED 条件下的独立进入资格、等资源与共同接受器、所有实际源码/模型身份及独立复核。若未来需要 Mmax/shape 修复，只能在新身份中单独审查，不回改 L1 或冻结树，也不由本设计授权真实前向。
5. 上述门全部满足后，另行请求清楚限定范围的正式执行签署；本附录不预先授予该权限，不自动推进 M1-b、M2、M3、M4 或 L3。

### 7.3 最小用户决策集合

当前仅交付 M1-a 设计。下一步最小授权建议为：**在新的非冻结路径实现认证适配器原型，并运行不加载 F 的隔离合成接线/失败闭环验收和独立只读复核**。范围须继续排除真实数据前向、L1 重跑、在线 rollout、新状态、两天烟测及任何安装。

这一工程授权不要求今天替正式 L2 选 B 或 δ；合成值只属于测试。待真实品质能力可行后，第二组需要用户决定的是物理合同/品质认证适用域和 B 登记；第三组才是质量主指标、失败估计量、δ、统计门及新状态/进入资格。最后的真实执行必须单独签署，不能从前述设计或工程授权推定。

## 8. 静态复核记录与身份

审查方法：读取源码、权威状态、M0 五件及已有裁决；使用标准库 AST、文本/JSON 解析与 SHA256，不导入执行任何科研/仿真/模型模块。交付另运行仓库安全扫描与 Git 工具。静态确认 dict/tuple 差异、quality-gap 分支、`mutable_ids` 未使用、sampler certified 标记时序与安装事务调用点。以下是工作树原始字节 SHA256（不假定与 Git 行尾规范化后的 blob 相同）。

| 输入文件 | SHA256 |
|---|---|
| `scripts/simulation/d3_composed_certifier.py` | `b65ddf6545a84b656cd2e9dd541cea266d2d6bdbc1cd56abd76f7780104aea01` |
| `scripts/simulation/d3_geo_adapter.py` | `099c3517472d886ece4ea3e929df2ec54d0e459b2a2e2e07da5b658acc07adc5` |
| `scripts/simulation/d3_l1_two_arm_protocol.py` | `1a10e8065d4744df3171cc7a4055657a604c9a8646101e9d546fbe45d87b693a` |
| M0 总体路线图 | `f47f61d4524e4be4179955cc3e085d94fa1a1d47c25078c50ed3a485b829a137` |
| M0 L2 草案 | `4be18f8b74a9e5ef11d44539890afd73e046d3ab858e2c3426f57b520756938a` |
| M0 L3 草案 | `b0af9b7906374a1e13ebbaa64d8ac33a08ba4b81260f7e70b2351de38e08526c` |
| M0 C0/基线清单 | `f389648ed3102f4f6146b3b97cd1a3b60c120265a4805903ea4006edf7c5f074` |
| M0 资格矩阵 JSON | `a46d928e652154aba97e2c5d270aafebafe2114c51245b50ee27ad123ed9cef4` |
| 项目当前状态 | `1331bf0ff1973e260cdbbf9c61e3487827209bfe1e5691731c364be3dc07b8e5` |
| 执行进度表 | `1a0e7a3421bbdfb17ad585daff2d4638a6cd936f38abf4a311b70c7bb992fec8` |

交付前静态检查：本文件 14 个相对文件链接均存在，代码围栏配对；AST 确认 `mutable_ids` 在 composer 函数体零引用；表中 10 个输入文件的审查前后 SHA 完全相同。提交前再次 fetch，两远端分支仍为审查基准，暂存区原为空。以上仅为文档/静态检查结果，不是合成或真实运行验收。

安全扫描的初次零文件覆盖报告未被接受；随后在可读取暂存区的进程中，以 UTF-8 和关闭 Git 路径转义配置运行原扫描器，确认实际扫描本文件 1 件、无豁免、0 finding。保留交付工具默认扫描，不绕过、不修改扫描规则。

以上身份用于定位本次静态审查输入，不是新运行合同或科学资格 PASS。单独交付本文件，保留历史树、原始 FAIL 与现有 L1 BLOCKED；完整品质门、主指标及预算仍未闭合，不申请正式实验执行。
