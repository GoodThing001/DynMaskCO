# B3-Q：品质投影与合法性短路工程验收报告

**日期**：2026-10-14 · **基准 commit**：`f3b78f2` · **授权**：受限 B3-Q（用户裁决 2026-10-14）
**执行顺序**：**先冻结修订后的反例**（`B3Q_COUNTEREXAMPLE_SPEC.json`）**再实现**。
**边界**：无真实写回、未改 `_commit` / `_plan_and_commit`、未解除 D8、未触碰冻结数据与最终测试种子、未启动科学实验。

| 产物 | 作用 |
|---|---|
| `research/…/B3Q_COUNTEREXAMPLE_SPEC.json` | **实现前冻结**的 Q1–Q5 + L1–L3 + C1 预注册（含你的保守性更正） |
| `C-VRP_…/scripts/simulation/d3_quality_projection.py` | **D-B3A-2** 只读品质投影（复用 `transition_segment`，**不宣称保守**） |
| `C-VRP_…/scripts/simulation/d3_composed_certifier.py` | **D-B3A-1** 合法性短路（已改，v2_legality_first） |
| `C-VRP_…/scripts/tests/test_d3_b3q_quality_projection.py` | 12/12 PASS，exit 0，stderr 0 字节 |
| `research/…/B3Q_PROJECTION_MATRIX.json` + `raw/` | 机器可读矩阵与原始输出 |

---

## 0. 验收结果：**12/12 PASS**，且**保守性标签保持 NOT_ESTABLISHED**

| 项 | 结果 | 关键实测 |
|---|---|---|
| **Q1** 压力时刻表可行性 | ✅ | **压力时刻表在 3 辆载货车（0/1/2）上全部不可行**——如客户 12 的"最晚可达到达"为 12.55 而 `tw_end`=11.42；客户 2 为 13.02 vs 7.77。结果被标为**不可行**且**未用作界** |
| **Q2** 未预冷田间温度分支 | ✅ | 同一路线下把 lot 的 `precooled` 翻为 False，**恰好 1 单受影响**（车 1 / 订单 32，品质下降 **4.58e-05**）——分支被正确遵循 |
| **Q3** 逐温区阈值分别应用 | ✅ | 28 条记录全部满足 `threshold == unsalable_quality_threshold[temp_class]`；**刻意不断言**"class N 总是先失败" |
| **Q4** 投影 == 实际执行 | ✅ | 25/25 单在 `(vehicle, order, delivered_quality, salable)` 上**逐位相等**（在深拷贝上经生产机制真实执行） |
| **Q5 / L1** 合法性短路 | ✅ | 四类非法输入各自被**自己的门**拒绝并命名；下游 `route_feasible`/`energy_budget`/**`quality_projection`** 全部记 `not_run_illegal_order_set` |
| **L2** 合法输入判定不变 | ✅ | `tw`/`capacity`/`energy` 仍各自按原样拒绝，且"能耗失败时路线仍可行"仍成立 |
| **L3** 非法输入不崩溃 | ✅ | 越界客户 + 缺席车辆 → 干净拒绝，无异常逃逸 |
| **C1** 有界证伪搜索 | ✅（**结论为"未证伪"，但搜索无效**） | 5 行网格给出**同一个最小值** ⇒ `INEFFECTIVE_SEARCH` |
| **SUF** 投影通过不解除阻断 | ✅ | `installable=False`、`CAPABILITY_GAP_BLOCKING` 保持；`projection_state=PROJECTED_SATISFIED_UNPROVEN` |

---

## 1. 你的三点反对：**第 1 条被实测证实**，第 2/3 条已写入设计约束

我原稿的推论「品质单调不增 ⇒ 每个客户取 `tw_end` 即最坏品质」已被撤回。实测结果：

**第 1 条（可造出 Ω 外轨迹）——证实。** 把每个客户推到"最晚可达到达"后，**三辆载货车全部不可行**：
```
customer 12: latest arrival = 12.552  >  tw_end = 11.415   (unreachable)
customer  2: latest arrival = 13.023  >  tw_end =  7.773   (unreachable)
```
⇒ 该压力时刻表**根本不在合法执行轨迹集合内**，因此**不能作为界**。实现里它是被**检查**的（`stress_schedule_admissible`），失败时只记 `inadmissible_stress_vehicles` 并拒绝把它当界。

**第 2 条（等待会改变舱温/制冷模式，不能只看时间单调）——已写入约束。** 品质对**时间**与**温度**各自单调，但**不复合**成"时刻表集合上的界"：舱温剖面本身是时刻表的函数。因此我不再声明任何基于"时间单调"的界。

**第 3 条（各 lot 进入冷链时刻不同）——已写入约束。** 最大化某客户的服务时刻**不同时**最大化所有 lot 的损失，故不做逐客户独立最坏化的复合。

**标签**：`CONSERVATIVENESS_NOT_ESTABLISHED`（在投影与矩阵中都是**常量**），并附上未证明的不等式说明；`PROJECTED_SATISFIED_CONSERVATIVE` **未被引入**（D-B3A-3 按你的裁决**暂缓**）。

---

## 2. C1 的诚实结论：**搜索无效，因此"未发现反例"不构成任何支持**

C1 的网格（按 `tw_end` 缩放 1.0/0.75/0.5/0.25/0.0）**五行给出了完全相同的最小值** `0.986138525549`。

⇒ 我把它标为 **`INEFFECTIVE_SEARCH`**：搜索**没有真正移动**那个约束性 lot 的时刻表，所以
- **不得**把"未发现反例"读成对界的支持；
- 也**不得**因此升级标签。

**为什么无效**：均匀缩放 `tw_end` 并未针对**约束性 lot** 构造对抗时刻表（且压力模式本身已不可行，缩放只让到达更早、品质更高）。**有意义的保守性检验需要另设计**（见 §5）。

---

## 3. 实现过程中的一处**报告缺陷**（已修）

Q2 第一版用**排序后的前 5 个**品质值做展示，而差异发生在第 6 个之后，导致打印出来**看起来完全相同**。已改为**按 (vehicle, order) 逐单比较**并输出最大差值，才暴露出真实差异（车 1 / 订单 32，4.58e-05）。这是**报告口径**问题，不是物理问题——但它本来足以让人误判 Q2 为"无影响"。

---

## 4. D-B3A-1 落地：合法性短路（结构性，而非靠调用顺序）

`d3_composed_certifier.py` 现在顺序为：
```
plan_vehicle_set → revealed_only → no_served_in_plan → no_duplicate_order
  → full_partition → repair_plan → route_feasible → energy_budget → quality_projection
```
任一路径门失败 ⇒ 下游三分量**记 `not_run_illegal_order_set` 且根本不运行**。
⇒ "所有认证分量只收到经过合法性检查的订单集合"现在是**结构性质**，`certify_plan` 内部 `:127` 的 `reveal_time` 静默重排与 `:173` 的静默丢弃**不再有机会被触发**（Q5/L1 可证）。

**L2 证明重排没有改变合法输入的判定**（`tw`/`capacity`/`energy` 三例仍按原样拒绝，且能耗失败时路线仍可行）。

---

## 5. 仍未建立的东西（不得读作已解决）

| 项 | 状态 | 说明 |
|---|---|---|
| **保守品质认证** | ❌ **未建立** | 缺 `Q_cert,i ≤ Q_i(ω) ∀ω∈Ω_admissible` 的可检验证明；标签保持 `NOT_ESTABLISHED` |
| 有意义的保守性证伪设计 | ❌ 未做 | 需要**针对约束性 lot 的对抗时刻表**（在 Ω 内搜索 + 可行性修复），或解析式界论证；当前 C1 网格已自证无效 |
| 短路修复对既有合法用例之外的回归 | 部分 | L2 覆盖 B2 的 tw/capacity/energy 三例；**未**重跑 B2 全量 D1–D8 与新顺序的组合（见 §7 建议） |
| D-B3A-3 状态 | **暂缓**（按裁决） | 需先有保守下界 |
| D-B3A-6 旁路消除 | **暂缓**（按裁决） | 8 处调用点属不同实验路线/历史基线，统一改写会改源码身份与比较协议 |
| D-B3A-7 合同变换验证器 | 规格已给 | 本轮未实现（按你的裁决"先建立可计算的来源证明"，属独立只读验证器） |

---

## 6. 合同版本资格声明（论文风险，按要求写入）

本轮**全部**结果适用于 **v1 `QualityConfig`**（比例阈值 `0.8/0.85/0.9`、`initial_quality` 三类温区均为 1.0、Arrhenius 舱温速率），矩阵中每条投影都带
`applicable_contract_version = {quality_config_class: QualityConfig, contract_schema_version: coldchain-contract-v1, is_v2: false}`。

> **在 D3/M0 上证明 v1 品质投影正确，并不自动证明 A-v1 动态冷链主任务上的鱼可售认证正确。** 两者是不同物理与不同指标语义（v1 比例 vs v2 绝对感官分，鱼 9/≥5）。**未来若切换 v2，必须独立证明质量模型与硬约束实现，不得复用 v1 的通过结论。**

---

## 7. 停止条件已达成（按你的裁定）

短路修复 ✅、投影接口 ✅、预声明反例验收 ✅ ⇒ **停止**。未进入真实安装、未改 `_commit`、未触碰冻结数据与最终测试种子。

**下一步（需你裁决，本报告不自动推进）**：
1. **保守性论证的独立设计**（P0）：针对约束性 lot 的 Ω 内对抗搜索 + 可行性修复，或解析界；若无法建立 ⇒ 如实输出 `CONSERVATIVENESS_NOT_ESTABLISHED` 并保持阻断。
2. **B2 全量回归在短路顺序下重跑**（P0，低风险）：确认 D1–D8 在新顺序下仍全通过。
3. **D-B3A-7 合同变换独立只读验证器**（P1）。
4. B3a 观察器/回执的**无写回测试实现**（P1，已批准但本轮未做）。
5. G0 独立验收 / 资源门（后续）；E2、F−D、正式效果批次**暂缓**（G0 未通过，不具执行资格）。

---

## 8. 研究状态（不变）

**D3 G0 未通过**；G1/G2 未启动；**无可归因的 F−D 学习增益**；I3 正式负结果 / 旧 D2 `not_qualified` / `stable_online_gain=False` 保持原样。
