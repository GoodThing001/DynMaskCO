# B 静态耦合 MCVRP：协议设计与无训练筛选诊断

> **执行回传（2026-09-23，对抗验证）：本协议的耦合物理设计被全数驳回，本文件保留为「attempt #1」设计记录，不作为有效方向。** 三个耦合方案（门冲击跨区耦合 / 单共享车厢 / 最小改动）被对抗复核驳回，理由一致且硬：①门冲击需放大 40–70× 才能材料（信号制造，相对现有 0.07–0.12°C 尖峰）；②货物本身是巨大热缓冲，门冲击动的是空气不是产品温度，把 8°C 空气尖峰作用到产品品质正是"0.2→0.003"同类简化误差；③跨区耦合仍是单调可加的逐单门暴露和，规则可捕获（`bool` 字段 `quality_trajectory_integral` 还会使 `validate()` 崩溃）。筛选冒烟（即便把门冲击放大到不物理的 8°C）G_S≈1.5–2% → **NO-HEADROOM**。**结论：门冲击耦合不是可行机制；B 的耦合源需重新选定。** 生产契约字段（`door_cross_zone_fraction`/`quality_trajectory_integral`）已从 `coldchain_contract.py`/`coldchain_state.py` 回退；筛选脚本 `run_coupled_mcvrp_screen.py` 保留但依赖已回退字段、已标记 SUPERSEDED。

日期：2026-09-23。性质：执行态协议文档（先 B 后 A 的第一步）。上承 `决策与审计/协议重设计决策_2026-09-23.md`。硬约束不变：**MaskCO 为核心方法、冷链物流为领域**；不修改 `../MASKCO_code/`；全部新代码在扩展工作区内。

本步 = 协议设计 + 无训练筛选诊断（纯 NumPy + C0 迁移引擎，1 天内出结论，出结论前不碰训练）。

---

## 1. 耦合物理：ThermalConfig 字段新增（精确）

全部为 **research-extension 字段**，默认值关闭，生产 C0 contract 行为 bit-identical（沿用 `diurnal_amplitude_c`/`precool_*` 的既有模式；激活后改变 `contract_hash`，进生产需 re-freeze 身份）。

| 字段 | 类型 | 默认 | 单位 | provenance |
|---|---|---|---|---|
| `thermal.door_cross_zone_fraction` | `float` | `0.0` | dimensionless | `literature`；Giliberto & Paradiso arXiv:2504.15741 shared-compartment door shock；灵敏度 `[0.0, 0.8]` |
| `thermal.quality_trajectory_integral` | `bool` | `False` | —（方法开关） | 无（同 `precooled`/`use_station` 布尔开关，无物理量纲） |

**QualityConfig 无字段变更**。品质参数沿用 pilot；耦合由「路由时长 + 门冲击」产生（多小时路线让 Arrhenius 衰减累积、开门冲击注入温度尖峰）。`reference_rate_per_hour` 的放大只作为「加强物理」杠杆之一，不动默认。

**附带变更**：`thermal.door_heat_kwh` 的 provenance 灵敏度区间从 `[0.0, 0.15]` 放宽到 `[0.0, 8.0]` kWh/opening（文献开门冲击 5–15 K → 1–6 kWh/次，Stellingwerf et al. 2018），因为耦合 B 采用文献量级开门热，超出 pilot 区间。

## 2. 精确方程改动（`scripts/coldchain/coldchain_state.py`）

### 2.1 `_apply_door_event`（跨区开门冲击，原 #L400）
- 原：只有 `opened_zone` 加 `door_heat_kwh[zone]`，其余为 0。
- 新：`opened_zone` 加 `door_heat_kwh[zone]`；每个 `j ≠ opened` 加 `door_cross_zone_fraction × door_heat_kwh[opened]`；`total_heat` 累加所有区注入热；恢复能耗 `heat / COP(current)` 按区累加。
- `door_cross_zone_fraction = 0` 时与原实现 bit-identical。

### 2.2 `_advance_manifest_quality`（真实轨迹温度代替前后均值，原 #L373）
- 签名新增参数 `active_zone_mask`（由 `transition_segment` 的 `zone_mask` 传入）。
- `quality_trajectory_integral = False`：中点均值 `rate(0.5·(T_before+T_after))·duration`（旧行为）。
- `= True`：`quality_remaining = Q0 · exp(−∫₀^d rate(T_zone(t)) dt)`，`T_zone(t)` 由新增 `_segment_temperature_c` 重建（指数趋近环境温度 + 主动制冷线性逼近 target 并截断），用 5 点 Gauss-Legendre 数值积分（新增 `_segment_decay_integral`，纯 stdlib，无 numpy）。
- 未预冷 lot 仍用环境温度（田间温度），不随车厢轨迹变化。

### 2.3 `_advance_temperatures`：**不改**
其方程（指数自然趋近 + 主动区线性制冷到 setpoint）保持不变；新增的 `_segment_temperature_c` 只是复用它的轨迹模型做积分，不改变端点温度计算。

## 3. 无训练筛选诊断

脚本：`scripts/evaluation/run_coupled_mcvrp_screen.py`（已实现，冒烟通过）。

### 3.1 实例
静态 MCVRP：depot 居中，客户均匀撒在 100×100 km²，`temp_class ∈ {0,1,2}` 均匀，`demand ~ U[1,5]`，`capacity=45`（≈15 停/车），`speed=40 km/h`（每段 0.5–1 h，多停路线累计数小时让品质衰减可感知）。**第一屏无时间窗**——隔离耦合物理头腔；紧 TW 是 B 立项后的第二杠杆（TW 自身是强组合约束，会混淆耦合信号）。

### 3.2 四个求解器（共享距离结构，只在优化目标上分）
- **D**：距离 cheapest-insertion（距离基线）。
- **Dplus**：D + 距离目标局部搜索（2-opt + relocate）＝距离控制（隔离纯距离头腔）。
- **R**：Dplus + 规则族 best-of `{no-op(=Dplus 本身), serve-by-class asc, serve-by-class desc}`。类序规则**保持类内距离序**（非稻草人），只重排同温类停靠。
- **S**：Dplus + 耦合目标 J 的局部搜索（2-opt + relocate，first-improvement，`--budget` 上限）。

### 3.3 目标与归一化
`J = distance/D̄ + quality/Q̄ + energy/Ē`，其中 `D̄/Q̄/Ē` 是对 **D 解**的每分量均值（筛选级 devmean，λ_q=λ_e=1）。**不是**冻结生产目标（生产目标用校准 profile）。物理量级扫描 `dT_door ∈ {0,4,8}`°C（`door_heat_kwh = dT × thermal_capacity`）。

### 3.4 测量与精确三分判定
对每实例算 `G_X = (J_Dplus − J_X)/J_Dplus`（正=X 更优）；`G_S`＝搜索头腔、`G_R`＝最佳规则头腔。paired bootstrap 1000 次取 95% CI。

| 判定 | 精确规则 | 阈值 |
|---|---|---|
| **no-headroom** | `G_S < 0.02` | 2%（材料性下限） |
| **rule-capturable** | `G_S ≥ 0.02` 且 `G_R ≥ 0.80 × G_S` | 规则回收 ≥80% 头腔 |
| **viable** | `G_S ≥ 0.02` 且 `G_R < 0.80 × G_S` | 头腔材料且非规则可捕获 |

### 3.5 当前冒烟结果（n=60、8 实例、budget=300，非正式结论）
`G_S`：dt=0 → 1.31%，dt=4 → 2.05%（CI 1.03–3.17%），dt=8 → 1.55%（CI 0.81–2.56%）；`G_R = 0`（类序规则族被距离路由支配）。→ 名义 dt=8 判 **NO-HEADROOM**（临界：1.55% vs 2% 阈值）。**但注意**：此冒烟把门冲击放大到不物理的 4–8°C（信号制造）、且用空气轨迹作用到产品品质（货物是热缓冲）——即便这样头腔也仅 1.5–2%，进一步确认门冲击耦合不可行；见顶部执行回传。

## 4. 会判死 / 失效的警告清单

1. **阈值任意性**：2%/80% 是 pilot 阈值；换阈值可翻转判定 → 冻结前必须做阈值敏感性。
2. **归一化口径**：筛选级 devmean 非生产目标；改 λ_q/λ_e 会改胜负 → 结论只对声明的归一化成立。
3. **Dplus 非距离最优**：若 `H_DDplus`（距离头腔）未收敛，`G_S` 会混入距离头腔 → 每次报告 `H_DDplus`，并要求其 ≪ `G_S`。
4. **搜索预算不足**：S 是 first-improvement + budget 上限，`G_S` 是头腔**下界**；预算太小低估头腔 → 报预算并做预算敏感性。
5. **门冲击量级是主杠杆**：no-headroom 时先加强 reference_rate / 路线长度 / cross_fraction，再改判定。
6. **随机种子耦合**：实例随机生成与 bootstrap 种子耦合 → 多 seed 复跑确认。
7. **规则族封闭性**：R 只含 3 规则；判 viable 前必须穷尽「简单规则」家族（按品质期限、按类分区等）。
8. **contract_hash 变更**：door_heat 区间放宽 + 新字段改变 hash；任何一项进生产 → re-freeze 1152 实例身份（硬不变式 5）。
9. **时间窗缺失**：第一屏无 TW；B 立项时加 TW 必须**重筛**（TW 会改变头腔结构与规则可捕获性）。

## 5. B 立项后的完整基准（指针，非本步范围）

100–300 客户、10–20 车（路线 10–30 停）、紧 TW + 文献开门冲击 + 交叉耦合；精确 B&P（Hülagü & Ciancio 2025）/ ALNS（Chen/Liu/Langevin 2019）/ HGS 基线；MaskCO 条件构造对照。贡献 = 首个**物理耦合**冷链基准 + 非线性目标（MILP 需分段线性化、MaskCO 原生）下的掩码构造。

一句话总结：**开门冲击交叉耦合物理（文献锚定）× 多停长路线（密度）× 筛选级 devmean 目标——三层叠加是可测量的耦合头腔来源；2%/80% 三分判定是质量闸门。当前参数化给出临界 NO-HEADROOM，下一步是加强物理量级复筛，而非判死或立项。**
