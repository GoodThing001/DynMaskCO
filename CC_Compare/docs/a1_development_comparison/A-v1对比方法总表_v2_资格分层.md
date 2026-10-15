# A-v1 对比方法总表 · v2 资格分层版（版本化展示件）

> **版本化展示件，不覆盖旧总表。** 旧总表与旧 `ARTIFACT_MANIFEST.json` **逐字节保留**（见 `ARTIFACT_MANIFEST_v2_资格分层.json`）。
>
> 本件按用户 2026-10-14 裁定制作：**三个维度分开**，**不筛除任何行**，主档优先、`p_c=0` 单独作敏感性。
>
> ⚠️ **本件不改变任何负结果**：主档同批 `I3 − cond_hist = −144.680 [−191.109, −100.125]`（同批 `cond_hist = 600.997`）原样保留。

## 0. 三个维度（**不要合并成一个标签**）

| 维度 | 字段 | 取值 | 解决的问题 |
|---|---|---|---|
| 配对范围 | `pairing_group` | `A` / `B` / `NONE` | 哪些结果能在**同批**比较 |
| 证据资格 | `evidence_tier` | `DEGENERATE / NOT_VERIFIABLE / STRICT_PASS / CORE_PASS / DIAGNOSTIC / REFERENCE` | 该行证据**能支持什么结论** |
| 方法保真度 | `fidelity` | `native / solver-adapted / ordering-adapted / incompatible` | 是否代表**原论文完整算法** |
| 因果核验 | `causal_gate_status` | `VERIFIED / IDENTITY_REGISTERED_ONLY / NOT_VERIFIABLE_FROM_ARCHIVED_GATE / NOT_APPLICABLE` | **登记 ≠ 验证** |

**判定规则（可机械执行）**：

1. `evidence_tier` 按**优先级顺序**判定（`REFERENCE` 为保留值，只给配对参照臂本身）：
   `Q_nondegen=False` ⇒ `DEGENERATE`；**`Q_identity=False`** ⇒ `NOT_VERIFIABLE`；`Q_formal_strict=True` ⇒ `STRICT_PASS`；`Q_formal_core=True` ⇒ `CORE_PASS`；否则 `DIAGNOSTIC`。
2. ⚠️ **`Q_identity=None` 是结构性不适用，不是否决**：纯规则 / 基线臂**没有** checkpoint 身份块 ⇒ 单列 `identity` 显示 `N/A（…）`。上一版把 `None` 也判成 `NOT_VERIFIABLE`，会把 `Q_formal_strict=True` 的规则臂错标为"不可核验"（与 `Q_method_health=null` 是**同一类错误**）。
3. **`pairing_group` 与 `evidence_tier` 相互独立**：有配对关系**不等于**严格合格（例：`I3` = 配对组 B + `CORE_PASS`，且有超时缺口），严格合格也**不等于**与其他组可配对。
4. **`Q_method_health=None` 是结构性不适用**（内部基线/规则臂的 gate 不含该判据），**不得**强制要求等于 `True`，也**不得**因此降级。
5. **`booking_horizon` 缺口是跨状态的因果风险**：本件单列 `causal_gate_status`；**不得**用"已登记"替代"已验证"。
6. **`λ=3` 单列 `DEGENERATE`**（可观察的退化行为），**不得**并入"不可核验"（会让读者误以为原件缺失）。
7. `pairing_group` 标注**不代表**超时/健康资格通过；超时缺口与身份问题在「风险标记」列**逐行**显式给出。

## 1. 主档 `p_c=(5,10,15)`（**正式主结论口径**）

> **这是正式论文主结论的档位。** 排序**仅**用于阅读；跨 `pairing_group` 的数值**不得**直接做差。

| 原# | 方法 | 方法保真度 | 配对组 | 证据资格 | 因果核验 | 依赖闭包 | 身份 | 方法健康 | Utility | Δ vs cond_hist | 超时/天 | 风险标记 |
|---:|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| 1 | ◆ C1 cond_hist（SAA 强基线） | `native` | **A** | **REFERENCE** | `NOT_APPLICABLE` | N/A（结构性不适用） | N/A（未在该矩阵中单列，身份以 §2 基线表为准） | N/A（未在该矩阵中单列） | 631.4 [522.7, 727.6] | 0.0 [0.0, 0.0] | — | 配对参照臂本身（不作优势判定对象） |
| 2 | ★ Ours: MaskCO A3（掩码重构 · 预训练权重） | `native` | **NONE** | **NOT_VERIFIABLE** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | False | N/A（结构性不适用） | 458.8 [356.1, 557.8] | -158.5 [-203.4, -114.7] | 18.85 | 超时>0；身份问题：shared_source_sha256_mismatch；配对来源**跨源**（shared_source_sha256_mismatch）⇒ 禁止用于任何优势判定 |
| 3 | ★ Ours: MaskCO I3（决策感知） | `native` | **B** | **CORE_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | N/A（结构性不适用） | 456.3 [369.8, 538.0] | -144.7 [-191.1, -100.1] | 37.33 | 超时>0；**有同批配对证据**，但**超时资格存在缺口**；配对对象是同批 cond_hist=600.997 |
| 4 | rule: value/energy λ=2 | `native` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | N/A（结构性不适用） | N/A（纯规则臂：无学习权重，gate 结构性没有 _pairing_ident） | N/A（结构性不适用） | 180.6 [118.1, 243.2] | -450.8 [-570.5, -319.9] | 0.00 | — |
| 5 | rule: myopic | `native` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | N/A（结构性不适用） | N/A（纯规则臂：无学习权重，gate 结构性没有 _pairing_ident） | N/A（结构性不适用） | 30.5 [-94.4, 149.7] | -600.9 [-638.5, -560.6] | 0.00 | — |
| 6 | OR-Tools-RH accept | `solver-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | 20.9 [-102.8, 139.7] | -610.5 [-647.4, -569.4] | 0.00 | — |
| 8 | RouteFinder accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -26.1 [-152.1, 93.4] | -657.5 [-699.9, -611.9] | 0.00 | — |
| 9 | MVMoE accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -38.6 [-165.6, 83.4] | -670.0 [-722.6, -619.0] | 0.00 | — |
| 7 | SGBS accept | `ordering-adapted` | **A** | **CORE_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -52.8 [-182.8, 69.5] | -684.2 [-732.1, -640.5] | 9.07 | 超时>0 |
| 10 | POMO accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -68.1 [-190.0, 48.6] | -699.5 [-752.3, -650.5] | 0.00 | 与 SGBS / Omni-VRP **共用同一权重文件** ⇒ 非独立模型 |
| 11 | AttentionModel accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -72.9 [-196.8, 46.6] | -704.3 [-756.6, -652.9] | 0.00 | — |
| 12 | RRNCO-Ordering accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -73.2 [-203.1, 54.4] | -704.6 [-768.5, -646.1] | 0.00 | — |
| 13 | CO-enriched-ML accept（纯学习排序） | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -74.3 [-210.8, 53.0] | -705.7 [-760.7, -651.3] | 0.00 | — |
| 15 | Sym-NCO accept | `ordering-adapted` | **A** | **CORE_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -76.1 [-217.8, 55.3] | -707.5 [-772.9, -644.8] | 0.23 | 超时>0 |
| 14 | CO-enriched-ML accept（+官方 PC-HGS） | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -80.5 [-218.0, 48.3] | -711.9 [-778.9, -655.2] | 0.00 | — |
| 16 | Omni-VRP accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -85.7 [-209.6, 33.8] | -717.1 [-776.5, -662.0] | 0.00 | 与 POMO **共用同一权重文件** |
| 19 | DeepACO accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -100.0 [-231.2, 30.8] | -731.4 [-797.1, -665.8] | 0.00 | **跨次不可复现**（ACO 未固定采样 seed） |
| 17 | MAPT accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -102.2 [-224.6, 18.5] | -733.6 [-789.0, -679.8] | 0.00 | **弱介入**：模型只决定前 ~3 位、尾部 EDD ⇒ 不得当作 MAPT 方法能力 |
| 20 | CaDA accept | `ordering-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -190.3 [-327.8, -54.5] | -821.7 [-892.2, -755.2] | 0.00 | — |
| 18 | Learn-Improvement-Heuristics accept | `ordering-adapted` | **A** | **CORE_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -335.1 [-536.5, -144.0] | -966.6 [-1143.1, -807.0] | 54.23 | 超时>0；**跨次不可复现**（清缓存后两次调用不一致） |
| 21 | PyVRP-RH accept | `solver-adapted` | **A** | **STRICT_PASS** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | False | True | True | -569.8 [-660.7, -482.9] | -1201.2 [-1286.6, -1117.5] | 0.00 | **展示层身份矛盾**：数值来自修复腿（须 0.11.3），旧标注写 0.14.0 ⇒ 见 manifest |
| 22 | rule: value/energy λ=3 | `native` | **A** | **DEGENERATE** | `NOT_VERIFIABLE_FROM_ARCHIVED_GATE` | N/A（结构性不适用） | N/A（纯规则臂：无学习权重，gate 结构性没有 _pairing_ident） | N/A（结构性不适用） | -1943.1 [-2051.4, -1841.0] | -2574.5 [-2611.9, -2538.0] | 0.00 | **全拒单（0 接单）的退化规则**，不是"性能很差的规则" |

## 2. 敏感性档 `p_c=0`（**不是**主结论口径）

> ⚠️ `p_c=0` 绝对效用更高**只是评分函数差异**（接拒构造**不接收** `p_c`），**不得**充当头条贡献证据。

| 原# | 方法 | 方法保真度 | 配对组 | 证据资格 | 因果核验 | 依赖闭包 | 身份 | 方法健康 | Utility | Δ vs cond_hist | 超时/天 | 风险标记 |
|---:|---|---|---|---|---|---|---|---|---:|---:|---:|---|

## 4. 逐行原始数值（**逐字保留**，供一致性核对）

| 原# | Utility p_c=0 | Utility 主档 | Utility p_c=(10,20,30) | Δ(p_c=0) | Δ(主档) | 超时/天 p_c=0 | 超时/天 主档 | 硬可行率 p_c=0 | 服务单/天 | 能耗 kWh/天 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1703.7 [1680.9, 1726.9] | 631.4 [522.7, 727.6] | -435.6 [-651.0, -239.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | — | — | — | — | — |
| 2 | 1577.9 [1558.5, 1597.0] | 458.8 [356.1, 557.8] | -704.1 [-904.6, -510.5] | -116.7 [-138.6, -96.0] | -158.5 [-203.4, -114.7] | 19.82 | 18.85 | 1.000 | 89.2 | 709.8 |
| 3 | — | 456.3 [369.8, 538.0] | — | — | -144.7 [-191.1, -100.1] | — | 37.33 | — | — | — |
| 4 | 1403.3 [1314.0, 1498.0] | 180.6 [118.1, 243.2] | — | -300.5 [-379.8, -214.5] | -450.8 [-570.5, -319.9] | 0.00 | 0.00 | 1.000 | 63.5 | 508.9 |
| 5 | 1305.2 [1284.1, 1325.7] | 30.5 [-94.4, 149.7] | — | -398.5 [-423.9, -370.8] | -600.9 [-638.5, -560.6] | 0.00 | 0.00 | 1.000 | 69.0 | 710.1 |
| 6 | 1300.7 [1281.2, 1320.6] | 20.9 [-102.8, 139.7] | -1258.8 [-1492.3, -1037.5] | -403.1 [-428.4, -375.3] | -610.5 [-647.4, -569.4] | 0.00 | 0.00 | 1.000 | 68.5 | 710.2 |
| 7 | 1266.5 [1244.2, 1289.9] | -52.8 [-182.8, 69.5] | — | -437.3 [-463.8, -409.0] | -684.2 [-732.1, -640.5] | 12.03 | 9.07 | 1.000 | 67.1 | 710.2 |
| 8 | 1266.2 [1244.1, 1287.5] | -26.1 [-152.1, 93.4] | — | -437.6 [-466.3, -408.7] | -657.5 [-699.9, -611.9] | 0.00 | 0.00 | 1.000 | 67.6 | 710.1 |
| 9 | 1257.7 [1231.3, 1284.0] | -38.6 [-165.6, 83.4] | — | -446.1 [-478.4, -414.7] | -670.0 [-722.6, -619.0] | 0.00 | 0.00 | 1.000 | 66.9 | 710.1 |
| 10 | 1238.8 [1214.8, 1261.5] | -68.1 [-190.0, 48.6] | — | -464.9 [-498.3, -432.8] | -699.5 [-752.3, -650.5] | 0.00 | 0.00 | 1.000 | 66.0 | 710.3 |
| 11 | 1236.2 [1207.3, 1263.2] | -72.9 [-196.8, 46.6] | — | -467.5 [-501.3, -433.7] | -704.3 [-756.6, -652.9] | 0.00 | 0.00 | 1.000 | 65.4 | 710.3 |
| 12 | 1235.4 [1202.4, 1265.2] | -73.2 [-203.1, 54.4] | — | -468.3 [-511.9, -428.0] | -704.6 [-768.5, -646.1] | 0.00 | 0.00 | 1.000 | 65.7 | 710.3 |
| 13 | 1234.5 [1204.9, 1263.3] | -74.3 [-210.8, 53.0] | — | -469.3 [-505.9, -432.4] | -705.7 [-760.7, -651.3] | 0.00 | 0.00 | 1.000 | 65.8 | 710.3 |
| 14 | 1230.5 [1197.6, 1259.7] | -80.5 [-218.0, 48.3] | — | -473.2 [-516.7, -434.1] | -711.9 [-778.9, -655.2] | 0.00 | 0.00 | 1.000 | 65.6 | 710.3 |
| 15 | 1229.3 [1196.0, 1261.3] | -76.1 [-217.8, 55.3] | — | -474.5 [-515.9, -435.0] | -707.5 [-772.9, -644.8] | 0.07 | 0.23 | 1.000 | 65.3 | 710.1 |
| 16 | 1226.9 [1195.8, 1254.5] | -85.7 [-209.6, 33.8] | — | -476.8 [-512.8, -442.0] | -717.1 [-776.5, -662.0] | 0.00 | 0.00 | 1.000 | 65.0 | 710.3 |
| 17 | 1215.4 [1189.0, 1240.8] | -102.2 [-224.6, 18.5] | — | -488.3 [-526.0, -451.5] | -733.6 [-789.0, -679.8] | 0.00 | 0.00 | 1.000 | 64.7 | 710.3 |
| 18 | 1215.0 [1149.2, 1269.0] | -335.1 [-536.5, -144.0] | — | -488.7 [-560.8, -428.3] | -966.6 [-1143.1, -807.0] | 25.35 | 54.23 | 1.000 | 64.2 | 685.1 |
| 19 | 1213.6 [1175.9, 1250.2] | -100.0 [-231.2, 30.8] | — | -490.2 [-534.8, -446.3] | -731.4 [-797.1, -665.8] | 0.00 | 0.00 | 1.000 | 64.8 | 710.1 |
| 20 | 1157.0 [1120.6, 1190.8] | -190.3 [-327.8, -54.5] | — | -546.7 [-592.1, -502.5] | -821.7 [-892.2, -755.2] | 0.00 | 0.00 | 1.000 | 62.0 | 710.3 |
| 21 | 909.2 [856.8, 960.4] | -569.8 [-660.7, -482.9] | -2048.8 [-2223.5, -1876.5] | -794.5 [-852.0, -739.4] | -1201.2 [-1286.6, -1117.5] | 0.00 | 0.00 | 1.000 | 47.9 | 710.3 |
| 22 | 0.0 [0.0, 0.0] | -1943.1 [-2051.4, -1841.0] | — | -1703.7 [-1726.9, -1680.9] | -2574.5 [-2611.9, -2538.0] | 0.00 | 0.00 | 1.000 | 0.0 | 0.0 |

## 5. PyVRP：实现身份拆分（**只读核验结论**）

| 字段 | 内容 |
|---|---|
| `method` | PyVRP-RH accept |
| `adaptation_arm_impl` | A-v1 `SolverAcceptReplanner` + PyVRP HGS |
| `repair_leg_runtime_version` | 0.11.3 |
| `independent_frozen_method` | PyVRP-RH-D，wheel `0.14.0` |
| `result_attribution` | A-v1 修复腿，**不是** PyVRP-RH-D 的独立冻结实验 |
| `identity_evidence_status` | REPORTED / HISTORICAL_RUNTIME_NOT_INDEPENDENTLY_VERIFIED |

> **不得**用现在执行一次 `import pyvrp; print(__version__)` 反推历史运行时版本。应先检查历史启动命令、解释器绝对路径、环境清单、原始日志，以及当时是否记录了 package/wheel SHA。
> 若历史未封存包版本 ⇒ 保持「**报告称 0.11.3、历史环境未独立核验**」。
> 出处：`读表注意与已知限定.md` §2.5、`docs/a1_development_comparison/README.md` §13.3、`audit/INDEPENDENT_READONLY_ADJUDICATION_2026-10-14.md`。

## 6. 机制不匹配（**未适配**，无任何数值）

| 方法 | 原因 |
|---|---|
| Learning-to-Delegate | 回归目标是 HGS 子问题距离、规模 n≥500；A-v1 单决策可见客户 ≤90 |
| PIP-constraint | 单车辆 TSPTW/TSPDL，env 无车队维度；无法对接 15 车接单+路由决策 |

## 7. 本件**不**做什么

- **不**改旧总表、旧 `ARTIFACT_MANIFEST.json`、任何封存运行原件与既有 gate；
- **不**筛除 `S3/S4` 行后再重新排名（全部 22 行**逐行保留**）；
- **不**运行求解器、**不**重算效用、**不**新增/改写任何数值；
- **不**构成优势主张；主档最可信结论仍是**负结果**：`I3 − cond_hist = −144.680 [−191.109, −100.125]`。

---

_生成器：`CC_Compare/tools/render_qualification_tiered_table.py`；旧总表工作树 SHA `7d94a4d783f60a21`。_

> **可复现性**：本件是「旧总表解析 + 资格矩阵」的**纯函数**（页脚**不含**生成时的 commit），因此重跑生成器必须得到**逐字节相同**的 SHA。生成时的来源提交只记在 `ARTIFACT_MANIFEST_v2_资格分层.json` 的 `source_commit` 字段。
