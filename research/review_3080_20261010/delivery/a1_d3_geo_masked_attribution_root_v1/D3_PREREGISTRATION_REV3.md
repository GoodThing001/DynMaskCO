# D3 预声明 REVISION 3（修订冲突项与冻结取值；**不授权执行**）

- **状态**：`DRAFT_FOR_USER_SIGNATURE` —— 未签字前不运行 F−D-Geo 科学比较。
- **取代范围**：仅取代 `D3_PREREGISTRATION_REV2.md` §9 中**与最新 D3 装配裁决冲突**的平局取值。
- **原件保留**：`D3_PREREGISTRATION_REV2.md` 与 `D3_PROSPECTIVE_PREREGISTRATION.md` **字节保留、不改写**；历史裁决不覆盖。
- **依据**：`research/review_3080_20261010/p0_next_stage_audit_root_v1/D3_ASSEMBLY_INVARIANTS.md`
  （§N1 装配层默认值、§N2 `eps=0` 仍不足以保证平局一致）、
  `p0_next_stage_audit_root_v1/D3_NEXT_STAGE_QUALIFICATION_MATRIX.md`、
  本批 P1 的 `A1_D3_ELIGIBILITY_MATRIX.json`、`A1_D3_FREEZE_MANIFEST_7b79ac57.json`。

## 1. REV3 修正的冲突项（逐条对照）

| 项 | REV2 §9 原取值 | REV3（现行有效） | 依据 |
|---|---|---|---|
| `eps_d`（主对照） | 相对 `1e-9` + 绝对 `1e-12` | **`EPS_D_PRIMARY_REL = EPS_D_PRIMARY_ABS = 0.0`** | 装配审计：`single_factor_with_F` 仅在 `eps == 0` 时为真 |
| tie-break（主对照） | 规范动作键序 `(customer_id, vehicle_id, slot.kind, slot.anchor, position, predecessor, successor)` | **`TIE_BREAK_LEGAL_ORDER = 'legal_order_first'`**（沿用 F 的历史 `legal` 顺序与首个最大值规则；D-Geo 在 `eps=0` 时也按同一 legal 顺序取第一个最小距离动作） | §N2 + 用户裁定"唯一主对照保留 F 的历史 legal 顺序与首个最大值规则" |
| 暂定 eps | — | `EPS_D_REL_PROVISIONAL = 1e-9`、`EPS_D_ABS_PROVISIONAL = 1e-12` **只允许非主对照的稳健性变体显式启用** | §N1 修复说明 |
| 规范键序 | 主判 | 降级为 `TIE_BREAK_CANONICAL`，**仅用于审计与变体** | §N2 |

**关键（N2）**：`eps = 0` **本身不足以**保证平局一致——当实际 `legal` 顺序与规范键序不一致且出现精确同分时，两臂仍可能选不同动作。
因此主对照必须**同时**满足 `eps == 0` **且** `tie_break == legal_order_first`；该组合是 `single_factor_with_F = True` 的**唯一**条件。

**实现现状（本批只读核对符号存在）**：`scripts/simulation/d3_geo_adapter.py` 中
`EPS_D_PRIMARY_REL/ABS = 0.0`、`EPS_D_*_PROVISIONAL`、`TIE_BREAK_LEGAL_ORDER/TIE_BREAK_CANONICAL`、
`select_geo(..., tie_break=TIE_BREAK_LEGAL_ORDER)` 均已存在；装配层（`stepwise_sample_d3`）的默认值
一致性以 T-G 回归为准，**若回归出现失败，按失败原样记录，不得事后修改冻结文件后声称通过**。

## 2. 三层资格（不得混层，不得互相追认）

| 层 | 允许做什么 | 判据 | 禁止 |
|---|---|---|---|
| **L1 机制诊断** | 同掩码、同候选集合下是否产生**可执行动作分歧**；报告完整分歧与失败原因 | 只报分歧/一致与原因；**不宣称统计优势** | 不得据 L1 声称任何增益 |
| **L2 固定状态质量评价** | 事前固定状态集、指标、聚类单位与阈值后做正式推断 | 可比性门 → 决策质量指标（合法/非法、C0 认证结果、候选价值分布）；预登记阈值与 CI | 不得事后改判据/改阈值/挑档；不得把一步分歧当优势 |
| **L3 在线效用** | 另行预注册；逐日配对 + 完整实际提交证据 | 需 G0/G1/G2、E2、`P_after_install`、等时预算全部就位 | 不得用 L2 结果追认 L3 |

## 3. 唯一主对照

`Δ = F − D_Geo`（唯一主对照）。只有 F 与 D-Geo 共享**动作枚举 + tie-break**；`F−R2`、`D-Geo−R2` 为次要比较，运行后不得从表中挑最有利者。不得事后替换主对照、不得为 F 更换共同决策器。

## 4. 统计单元与状态来源

- **独立单元 = 开发日（instance/day）**，**不是**同一天内的多个固定状态；同一天内多状态属重复测量，必须按日聚类（day-clustered bootstrap）。
- 状态集必须在看到任何臂结果**之前**固定；来源必须是**新鲜、可恢复、独立预登记**的状态。
- `results/m0_scale/sgbs_fixed_cal_v2` / `sgbs_fixed_train_v2` 及其历史 32 条状态：**仅 L1 诊断**，缺完整恢复信息且已用于历史开发，**不得**充当 L2/L3 科学证据。

## 5. 效用指标与阈值

- **L3 效用口径**（沿用 A-v1）：`J = revenue − fuel − reject_loss`；品质**不货币化**；鱼类可售为**硬约束**；C0 预算由 `certify_plan` 认证。
- **L2 指标**：合法/非法、C0 认证结果、候选价值分布、两臂首步分歧（分歧不判 FAIL）。
- **阈值**：任何未获来源支持的数值一律标 **`PENDING`**；**不得**为凑出结论填写未经来源支持的阈值或 CI 门槛。

## 6. 停止规则

1. REV3 未签字 ⇒ 不运行 L2/L3；
2. 可比性门任一不等 ⇒ `not_comparable`，停止并修身份，不作结论；
3. 任一硬约束被放宽（品质货币化 / 鱼可售软化 / C0 放宽）⇒ 该臂作废并按违规记录；
4. 两臂等时或等工作量不成立 ⇒ 不得声称等时合格；
5. 出现未预声明的杠杆、决策器或数据 ⇒ 停止并另行预声明；
6. 达成本上限 ⇒ 报告 `not_qualified`，不扩预算；
7. 回归失败（含装配不变量）⇒ 原样记录，**不得**以修改冻结文件的方式消除失败。

## 7. 签字前仍需用户裁定的取值

| 项 | 现状 |
|---|---|
| `eps_d` / tie-break（主对照） | 已由装配裁决固定为 `0/0` + `legal_order_first`；**请确认接受** |
| 统计单元 | 本 REV3 提议"开发日为独立单元、按日聚类"；**请确认** |
| 状态集与来源 | `PENDING`（需新鲜、可恢复、独立预登记的状态集） |
| 正式阈值与 CI 门槛 | `PENDING` |
| 等时/等工作量预算上限 | `PENDING` |
| E2 训练对照 | 缺失；在补齐前不得作学习/预训练贡献主张 |

## 8. 非声称

不声称 MaskCO 主模型在线增益、不声称学习或预训练贡献（E2 缺失）、不声称 P2/G-L/G-C、不做
A1→A2 或敏感性→主档追认、不改写 I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）与旧 D2
`not_qualified`、不把工程安全机制当作算法创新。`algorithmic_gain_claim: none`。
