# D3 完整动作一致性审计（P0 补充，工程证据）

- 轮次：`A1_D3_ACTION_CONSISTENCY_AUDIT`（承接 `A1_D3_FIXVALID_5273739d`）
- 触发：用户复核发现上一轮证据存在**记录缺陷**，要求"补一项针对性审计"
- 状态：`ACTION_CONSISTENCY_AUDIT = COMPLETE`（10 行 / 10 PASS / 0 FAIL）
- 未改动：`freeze_tree_5273739d/`、`raw/` 下既有原始结果、历史测试、`MASKCO_code/`；本轮**只新增**审计工具与审计记录
- `algorithmic_gain_claim: none`；`F_vs_D_Geo_effect: NOT_COMPARED`；L2/L3 仍未授权

## 1. 用户指出的两处证据缺陷：确认

### 1.1 `apply_calls` 引用污染（确认）

上一轮 `verify_legal_control()` 把**模块级可变列表本身**放进了归档记录（`detail['apply_calls'] = APPLY_CALLS`），而后续场景会 `APPLY_CALLS.clear()`。归档因此出现自相矛盾：run1 记录的 `chosen_step.position=1`、当时算出的 `applied_index_from_apply_fn=0`，但同一记录里 `apply_calls` 的动作却是**后一个场景**的动作（position=0）。用户判断正确：该字段**不能**作为可靠证据。

### 1.2 探针不是清洁控制（确认，并给出机制）

归档 `C_coherence_probe` 的 `failure_reason = "score_len"`，同时 `n_steps = 1`。本次审计复现并解释该机制：

| 场景 | 评分器 | failure_reason | n_steps | n_applied |
|---|---|---|---|---|
| C1 两客户夹具 `mask_set=[4,5]` | 手工常数 `[1.0, 1.0]` | `score_len` | 1 | 1 |
| C2 同一夹具 | 适配器自带 `G.geo_score`（长度自匹配） | `null`（自然终止） | 2 | 2 |

机制：`stepwise_sample_d3` 是**多步**装配，每一步的合法动作数都要求与该步评分向量等长；手工常数向量只能匹配第一步，后续步触发 `score_len` 回滚，而**先前步的 steps 仍被返回**。因此"`score_len` + `n_steps=1`"是**评分向量长度**问题，不是夹具非法，也不是实现缺陷；但上一轮把它当作"完成全流程的对照"来判读是**错的**。

## 2. 修正后的审计设计

1. **每场景独立 `Recorder`**，`snapshot()` 取 `deepcopy`；场景之间不共享任何列表对象。
2. **逐字段比较**实际传给 `apply_fn` 的 `FleetAction` 与 steps 记录中的完整动作：`customer / slot_kind / slot_anchor / position / predecessor / successor`（6 字段，比上一轮只比 customer 严格）。
3. **两个场景分别判定**：A = 单步合法对照（同客户、两合法动作、精确并列）；B = 完整事务流（`run_repair_transaction` + 仅测试用认证器）。
4. 评分器一律使用**长度自匹配**的 `G.geo_score`（精确并列由两个动作 `incremental_distance` 相同保证）。
5. D3 模块**从冻结树导入**（`--d3-sim/--d3-tests` 指向 `freeze_tree_5273739d`），审计记录里保存各模块 `__file__` 与 sha256 作为身份证据。
6. 记录器带 12 次调用上限守卫，防止多步装配无界循环（触发即 FAIL 并记录，不隐藏）。

## 3. 结果（原始记录：`audit/A1_D3_ACTION_CONSISTENCY_AUDIT_{stdout,stderr}.txt` + `.json`）

| 行 | 检查 | 结果 | 关键数据 |
|---|---|---|---|
| A0 | 无异常、未触发守卫 | PASS | n_applied=1 |
| A1 | 合法夹具无契约违例原因，且每步都有被记录的应用动作 | PASS | `failure_reason=null`、n_steps=1、n_applied=1 |
| A2 | **逐字段**：应用动作 == steps 记录动作（6/6 字段） | PASS | 两侧均 `customer=5, slot_kind='anchored', slot_anchor=3, position=1, predecessor=0, successor=0` |
| A3 | 计划层：装配返回的计划 == 由记录动作重建的计划 | PASS | `48d80e0fcf4584e8` == `48d80e0fcf4584e8` |
| B1 | 事务流完成：1 次安装、认证器 1 次、无异常 | PASS | `n_installs=1`、`returned=True` |
| B2 | **跨模式**：事务内应用的动作 == 装配层应用/记录的动作（6 字段） | PASS | 两侧完全相同 |
| B3 | **三方绑定**：由应用动作重建的计划身份 == 认证身份 == 真正交给 `install_fn` 的计划身份 | PASS | `48d80e0fcf4584e8` 三方一致，台账另记 `plan_hash == certified_plan_hash` |
| C1 | 复现上一轮 `score_len`（多步 artefact） | PASS | `score_len`、n_steps=1 |
| C2 | 同一夹具换长度自匹配评分器即不再 `score_len` | PASS | `null`、n_steps=2 |
| D1 | 隔离证明：A 的归档快照在 B 运行前后逐字节一致，A/B 列表对象不同 | PASS | `identical_before_after=true`、`distinct_lists=true` |

冻结树身份（审计实际导入的模块）：`d3_geo_adapter.py` sha256 `099c3517472d886e…`（即登记身份）、`mpre_policy.py` `5b03e4cae812fa20…`、`action_contract.py` `9d64273aa3df98cb…`、`test_d3_geo_adapter.py` `80ab32e96e20c4de…`。

## 4. 结论

**实际传给 `apply_fn` 的动作与 steps 记录的动作逐字段一致（单步合法对照与完整事务流两条路径都成立），且"应用动作 → 认证身份 → 安装身份"三方绑定。** 因此上一轮的证据缺口是**纯记录缺陷**（引用污染 + 非清洁控制），**不是**装配器执行了错误动作。

按用户规则应当明确：若逐字段比较出现不一致，本条必须判为**实现缺陷**；实测一致，故本条按"证据缺口已收束"处理，且**未**据此获得任何算法结论。

## 5. Claim limits（本轮不证明的）

1. 仅覆盖**本夹具族**（同客户两锚定动作、双客户两动作）与**当前适配器实现**；不是全装配路径的不变式证明。
2. 事务流使用的是**仅测试用确定性认证器**（非 C0 物理/合同认证），D8 通道仍未打通（台账无 receipt）。
3. 多步装配的**终止原因**（自然 `no_legal`）只在两个小夹具上观察；大规模/多掩码路径未审计。
4. 审计工具本身的**三次调试迭代**（`AttributeError: n_calls`、`rec[...]` 下标笔误、B3 比较键含 `incumbent` 导致重建未执行）未单独归档原始输出，仅在此逐条记录原因；最终运行 exit 0，前两次失败**未**被改写为通过（B3 失败那一次的内容已包含在本文档与最终 JSON 的 `previous_round_defects_confirmed` 之外的记录里）。
5. 不涉及任何 F−D-Geo 效果、指标、阈值；`PENDING` 值仍未冻结。

## 6. 建议的下一轮（P1，需用户确认后才开工）

P1 = 制定 **F−D-Geo L1 机制诊断可执行规格**（仅机制/可执行动作差异，不做统计断言）。开工前需用户确认的开放项：

1. **状态来源**：L1 用哪些开发日/状态（是否必须为"未被前序批次使用过、身份完整、可恢复"的新鲜状态；若无，数据源协议需先修订授权）。
2. **F 侧输入**：F 使用训练好的 MaskCO checkpoint 评分的**具体加载口径**（`mpre_reinforce_s42_clean`，sha `f5acf83f…`）与是否需要 E2 训练结果（当前 E2 无合格训练结果 → L1 是否只做"权重已加载、评分可复现"的机制验证）。
3. **共同合法动作集与 tie-break**：`eps=0` + `legal_order_first` 已定为主规则；L1 需确认是否允许同时记录 `canonical_key` 变体作为审计列。
4. **记录字段与停止规则**：完整动作差异（6 字段）、掩码与信息身份、失败原因、运行预算与超时如何处理（超时是否计入失败）。
