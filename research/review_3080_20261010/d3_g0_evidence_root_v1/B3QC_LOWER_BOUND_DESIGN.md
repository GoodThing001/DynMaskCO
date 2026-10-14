# B3-QC：品质下界可证明性设计（只读）

**日期**：2026-10-14 · **基准 commit**：`3c2ad49` · **性质**：**只读**设计与审计。不实现安装、不改 `_commit`、不解除 D8、不触碰冻结数据与最终种子。
**三态裁决**：`PROVEN / FALSIFIED / NOT_ESTABLISHED`；**搜索次数永远不得升级为 `PROVEN`**。
**最终判定**：**`NOT_ESTABLISHED`**（本轮还**证伪了一个引理**，见 §3.4）。

| 产物 | 作用 |
|---|---|
| `scripts/simulation/d3_quality_lower_bound.py` | 候选解析下界 + Ω 定义 + 假设清单 + **已证伪引理登记** + 反例工具（1-D 可行出发扫描、真实速率温度插桩） |
| `scripts/tests/test_d3_b3qc_lower_bound.py` | A1/A2/A3/OM/BND/ADV/VER 七项验收（7/7 PASS，exit 0，stderr 0 字节） |
| `B3QC_LOWER_BOUND_MATRIX.json` + `raw/` | 机器可读矩阵与原始输出 |

---

## 1. 交付物 1：Ω 的正式定义（`Omega_admissible_d3m0_v1`）

**由生产配置固定的部分**
- `active_zone_mask = (True,)*zones`：`strict_online_env._coldchain_active_mask()` 在该配置下**从不关闭任何温区** ⇒ **未制冷漂移分支不在 Ω 内**。
- 品质模型 = **v1 `QualityConfig`**（一阶、逐温区 Arrhenius、比例阈值）。
- 预冷分支：`lot.precooled = thermal.precooled and temp_class >= precool_min_temp_class`（`coldchain_state.py:261-262`）。实测 `precooled=True, precool_min_temp_class=0` ⇒ **所有 lot 都被预冷**（见 §2.3）。

**成员条件（全部为生产硬门，见 `hard_gate`）**
| 维度 | 条件 |
|---|---|
| 热 | `temperature_hard_feasible == True`（无段前/段后温度越出 `hard_temperature_bounds_c`；`coldchain_evaluator.py:358`） |
| 时间窗 | 每次到达 ≤ `tw_end`；只能通过等 `tw_start` 推迟服务，**不得早于前一段服务结束出发** |
| 容量 | 每次取货后累计载重 ≤ `shared_vehicle_capacity` |
| 返仓 | 返仓到达 ≤ `tw_end[0]`（否则 `depot_return_violation`） |
| 承诺 | 已承诺 leg 不可撤销；订单恰好被取一次 |
| 顺序 | **等待只表现为"不派车"**（`mutable_suffix == []`），计划 suffix 的顺序在 Ω 内保持不变 |
| 时钟 | 时间不可倒退（`_advance_coldchain_to` 会抛错） |
| 地平线 | 需要靠 `_force_return_all` 在 `tw_end[0]` 之后返仓的轨迹 **不在 Ω 内** |

**Ω 内的自由度**：① WAIT 的次数与长度（受 `booking_horizon` 约束）；② 每段在"上一段服务结束 ~ 下一 `tw_end`"之间的**出发时刻**；③ 实际发生的开门事件（每次取货一次）与剩余取货次数；④ 实际制冷占空比 ⇒ 每段真实舱温。

**不在 Ω 内**：热违规、超 `tw_end` 到达、容量溢出、`tw_end[0]` 之后返仓、撤销承诺、重排 suffix、关闭活跃温区。

> **关键结构性观察（本轮最重要的设计结论）**：因为**等待不能发生在途中**（一旦 commit 就继续执行），**给定一个固定计划，Ω 内的时刻表自由度退化为"出发/等待时刻"这一维**。这把"对时刻表集合证明不等式"从高维问题压缩成 **1 维**——但见 §3.4：那条一维单调引理**被证伪了**。

---

## 2. 交付物 2：温度与剩余时长上界能否从合同与 strict-online 状态推出

### 2.1 热力学事实（读源码 + 实测）
`_advance_temperatures`（`coldchain_state.py:353`）：
```
natural = ambient + (current - ambient) * exp(-coefficient * duration_h)
若温区活跃:  updated = max(target, cooled), cooled = natural - cooling_rate * duration_h
否则:        updated = natural
```
⇒ 活跃制冷下 **`updated >= target`**，且 `natural` 落在 `current` 与 `ambient` 之间。**开门事件是向上跳变**：`warmed = current + door_heat[zone]/thermal_capacity[zone]`（且它在**品质更新之后**施加，因此影响的是**下一段**的 `before`）。

合同实测值：`target=(18, 4, −18)`、`hard_bounds=((0,30),(0,8),(−25,−12))`、`ambient=25.0`、`thermal_capacity=(0.3,0.4,0.5)`、`door_heat=(0.01,0.035,0.06)`、`cooling_rate=(2,5,8)`、`heat_transfer=(0.2,0.16,0.12)`。

### 2.2 温度上界：**可以推出**（并已实测未被违反）
品质速率使用的温度只有两个来源（`coldchain_state.py:398-402`）：
- 预冷分支：**该 lot 所属温区的段前/段后均值**；
- 未预冷分支：**`thermal.ambient_temperature_c`**。

而 Ω 含 `temperature_hard_feasible` ⇒ 段前/段后温度都在硬界内。**因此**
```
T_i^max = hard_upper[class_i]                    若未预冷分支不可达
        = max(hard_upper[class_i], ambient)      否则
```
**实测**：2134 次速率温度采样，**0 次越界**，最大速率温度 **18.03 °C**。

### 2.3 未预冷分支**不可达** ⇒ 环境温度放宽**不必要**（重要的收紧）
实测 `precooled=True`、`precool_min_temp_class=0`，且数据集中**所有** `temp_class ≥ 0` ⇒ **每个 lot 都被预冷**，`ambient` 分支**永远不会被走到**（插桩里 `ambient_branch_samples = 0`）。
⇒ 把 `max(hard_upper, ambient)` 里的 ambient 放宽**去掉**是**保安全**的收紧。效果（见 §6）：
| 每类 `T^max` | 放宽时 | 收紧后 |
|---|---|---|
| class 0（硬上界 30 / 环境 25） | 30 | **30** |
| class 1（硬上界 8 / 环境 25） | 25 | **8** |
| class 2（硬上界 −12 / 环境 25） | 25 | **−12** |

> ⚠️ **同时必须纠正一处此前的表述**：B3-Q 的 Q2 通过"把 `precooled` 手动翻为 False"验证了该物理分支，但那**不是当前配置下可达的输入**。Q2 检验的是**分支实现**，不是可达配置——这一点在此更正并登记。

### 2.4 剩余时长上界：**可以推出**，但很松
Ω 要求返仓到达 ≤ `tw_end[0]`（否则 `depot_return_violation`）⇒ 在 `t_now` 时已在舱内（或将在 ≥ `t_now` 时装舱）的 lot，其在舱时间
```
tau_i^max = max(0, tw_end[0] - t_now)
```
实测在采样决策点 `tau_max = 24.0 h`——**极松**（整个回仓时间窗）。这是下界偏保守的主要原因，也是后续可改进的方向（例如用"计划剩余服务 + 返仓"的实际上界叠加 WAIT 上界，而不是整窗）。

---

## 3. 交付物 3：候选解析下界——假设、推导、反例、失效条件

### 3.1 候选（你给出的结构，按 v1 物理具体化）
```
Q_i^LB = Q_i^start * exp( -k_i(T_i^max) * tau_i^max )
Q_i^start = quality_remaining     若 lot 已在 cargo_manifest
          = initial_quality       若订单尚未取货（取货前不衰减）
salable_i^LB :  Q_i^LB / initial_quality_i >= unsalable_quality_threshold[class_i]
```

### 3.2 假设（每条都可证伪，`assumptions()` 中逐条登记）
| id | 内容 | 依据 | 已被检验 |
|---|---|---|---|
| **A1** | v1 每次更新都是 `q ← min(q, q·exp(−k·Δt))`，无其它因子/加项 | `:404-405` | ✅ 结构检查通过 |
| **A2** | Ω 上驱动速率的温度 ≤ `T_i^max` | `:398-402` + `temperature_hard_feasible` | ✅ 2134 采样 0 越界 |
| **A3** | `k(class, T)` 对 T 非减（Arrhenius，正活化能） | `:128-143` | ✅ 三类温区在硬界+环境范围内均单调 |
| **A4** | Ω 上在舱时长 ≤ `tw_end[0] − now` | 返仓硬门 | ⚠️ 仅有硬门依据，**未做端到端独立验证** |
| **A5** | **Ω 由硬门定义** ⇒ 下界**以硬门为条件**；放宽任何硬门都必须重新推导 `T^max`/`tau^max` | `hard_gate` | 声明性；是**资格条件**而非经验命题 |

### 3.3 失效条件（不假设，去证伪）
| id | 失效路径 | 现状 |
|---|---|---|
| **F-a** | 开门升温使某温区越出硬界，而该越界值**未被任何段的 before/after 检查看到** | 未发现（2134 采样 0 越界）；但**只覆盖了 1 个实例的运行** |
| **F-b** | `_force_return_all`（`strict_online_env.py:658-664`）在 `tw_end[0]` 之后返仓——该轨迹不在 Ω 内，但若被当作合法则 `tau^max` 论证崩塌 | 已在 Ω 定义中排除；**未做端到端抽查** |
| **F-c** | `dist_mat` 与真实行驶时间不一致（`tw_speed`/asymmetry）使实际到达晚于任何由 `dist_mat` 推得的上界 | 未检验 |
| **F-d** | 跨温区耦合（rate 用的是 lot 自己的温区；若未来模型耦合温区则 A2 需重推） | 当前模型不耦合 |
| **F-e** | **"最晚可行出发 = 最坏品质"的一维单调引理** | ❌ **已被本轮证伪**，见 §3.4 |

### 3.4 ⚠️ 本轮**证伪**了一条关键引理（`L-departure-monotone`）

我原本打算用一条更紧的设计替代 §3.1 的静态下界：既然固定计划的 Ω 自由度只有出发时刻一维，那么**最坏情形 = 最晚可行出发**，用重放取该点的品质即可（远比 `tau^max=24 h` 紧）。

**1-D 可行出发扫描直接证伪了它**（真实决策点，约束性 lot = 车 0 / 订单 5）：
```
max_admissible_delay = 5.992 h     (9/9 网格点均可行)
realised binding service start : 8.4506 ×8 个点, 最后一个 8.6108
delivered_quality              : 0.997133 ×8 个点, 最后一个 0.997234   ← 反而升高
quality_monotone_nonincreasing_in_delay = false
```
**原因**：把出发推迟到使车辆**正好在 `tw_start` 到达**时，`service_start = max(arrive, tw_start)` 不再移动，而**取货前的空闲消失**，于是**后装 lot 的在舱时间被缩短** ⇒ 品质**升高**。
**结论**：更紧的"最晚出发 = 最坏"设计**不得使用**。登记为常量 `FALSIFIED_LEMMAS['L-departure-monotone']`，防止后续文档悄悄复活它。

⇒ 存活下来的候选只有 §3.1 的**静态 T/τ 下界**，且其判定为 **`NOT_ESTABLISHED`**。

---

## 4. 交付物 4：对抗搜索方案（"反例工具"，不是证明）

**设计**：给定**固定计划**，对**约束性 lot 所属车辆**做**一维可行出发扫描**：
1. 用二分求**最大可行延迟**（每次重放检查"所有到达 ≤ `tw_end`"且"返仓 ≤ `tw_end[0]`"）；
2. 在 `[0, max_delay]` 上取网格，**只保留可行点**；
3. 逐点**重放整条路线**（`transition_segment`，含 WAIT 与返仓），记录**该 lot 的真实服务开始时刻**与**真实交付品质**；
4. 与候选下界比较：有低于下界的点 ⇒ **FALSIFIED**；无 ⇒ 仅"未证伪"。
5. 并**评估搜索有效性**：若所有网格点给出同一个最小值 ⇒ `INEFFECTIVE_SEARCH`（"未发现反例"**连弱证据都不是**）。

**实测**：`n_feasible=9 / n_infeasible=0`、`VARIED_SEARCH`、并**证伪了 §3.4 的引理**——即该工具确实产生了有效反例，而不是空转。

**已知局限**：仅 1 个决策点 × 1 个约束 lot × 9 个网格点；未扫多实例、未扫"哪些 lot 是约束性的"、未覆盖 `booking_horizon` 下的 WAIT 组合。

---

## 5. 交付物 5：`PROVEN / FALSIFIED / NOT_ESTABLISHED` 三态裁决规则

`Falsification.verdict()` 的实现即规则：
- **`FALSIFIED`**：存在任一反例（下界被低于、假设被违反、搜索找到更低品质）。
- **`NOT_ESTABLISHED`**：无反例 ⇒ **仍然不是证明**，因为"在采样轨迹与有界网格上成立"不蕴含"对 Ω 全体成立"。
- **`PROVEN`**：只有给出 `Q_i^LB ≤ Q_i(ω) ∀ω∈Ω` 的**证明**才可授予；**本模块没有任何通往 `PROVEN` 的代码路径**。
- **搜索次数无关**：`VER` 检查显式验证"1 万次搜索后判定仍为 `NOT_ESTABLISHED`"。

**实测**：`VER` PASS——通过检查后仍 `NOT_ESTABLISHED`；1 万次搜索后仍 `NOT_ESTABLISHED`；有反例时 `FALSIFIED`；`PROVEN` 不可达。

---

## 6. 可用性诊断（**独立于安全判据，不得用于放宽安全下界**）

在采样决策点（25 个订单）上：

| 指标 | 放宽 ambient（含不可达分支） | **收紧后（当前）** |
|---|---|---|
| 下界判为**不可售**的订单数 | **7 / 25** | **0 / 25** |
| 最紧的比值下界 | 0.7119 | **0.9316** |
| 与阈值（0.8/0.85/0.9）的关系 | 大量低于阈值 | 全部高于阈值 |
| `delivered_quality >= Q_i^LB` 违规数 | 0 / 25 覆盖 | **0**（25 个有下界的订单全部满足） |

**读法**：收紧后的下界在这个样本上**不再否决任何实际可售订单**；但这是**可用性诊断**，**不构成**安全证明，**也不得**反过来作为调整安全下界的理由。另需注意 `BND` 只覆盖了首个决策点上的 25 个订单（对 50 条交付记录做匹配）——**覆盖不完整**，属已知局限。

---

## 7. 版本门（不可省略）

全部结果只适用于 **v1 `QualityConfig`**（一阶、比例阈值 `0.8/0.85/0.9`、初始品质 1.0、Arrhenius 舱温速率）。
**若最终科学主任务采用 v2 绝对感官 + 零阶鱼品质模型，必须另立 v2 的下界论证，不得沿用本文件的任何资格结论。**

---

## 8. 停止条件与后续（需你裁决）

**停止条件已达成**：Ω 定义 ✅、T/τ 可推性 ✅、候选下界（假设/推导/反例/失效条件）✅、对抗搜索方案 ✅、三态裁决规则 ✅，且已完成**独立于实现的证伪检验**。
**未做**：真实安装、`_commit` 修改、解除 D8、冻结数据/最终种子。

**下一步（我建议的顺序）**
1. **P0｜下界的证明或放弃**：要么补齐 A4 的端到端依据并把 F-a..F-c 逐一证伪（多实例、多约束 lot、含 `booking_horizon` 的 WAIT 组合），要么明确宣告该静态下界**不可证明**并保留 `installable=False`。**不得**通过扩大搜索次数或挑选容易通过的路线来改变结论。
2. **P0｜收紧 `tau^max`**：现在的 `24 h` 是整窗上界，是下界偏松的主因；若能用"计划剩余服务+返仓+WAIT 上界"替代，可显著提升可用性而不损失安全性。
3. **P1｜D-B3A-7 合同变换独立只读验证器**（低成本、可独立执行）。
4. **P1｜无写回观察器与回执的测试实现**（已批准，独立工作包）。
5. **暂缓**：D-B3A-3（`PROJECTED_SATISFIED_CONSERVATIVE`）、D-B3A-6（消除 8 处旁路）、**B3b 真实安装**。

**研究状态不变**：**D3 G0 未通过**；G1/G2 未启动；**无可归因的 F−D 学习增益**；I3 正式负结果 / 旧 D2 `not_qualified` / `stable_online_gain=False`。
