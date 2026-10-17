# Q3 第一轮七要素预声明：完整路线分割无关括区（S1）与在线认证成本（S2）

> **性质**：预声明（本文件在**实施之前**冻结）。冻结时点基准 HEAD：`ea6319f3a9ce386204fdf443642d612ddbf66287`。
> **授权范围**：仅 Q3 第一轮 S1 + S2。**不含** S3（执行器浮点外扩）、不含 G-2 规格收口、不含 M1-b 合同 v2、
> 不含 C3/G6/安装事务、不含训练/GPU/正式 rollout/最终种子、不含正式 L2/L3、不含任何 S-A…S-G 签署内容。
> **授权来源**：`M1_L2最小可行性与成本裁决建议_2026-10-17.md` §4.2–§4.3、§7、§10.2 与用户本轮任务书。
> **本文件不修改**任何冻结实验、历史裁决、`MASKCO_code/`、生产认证器或安装入口。

---

## E1 目标与可证伪命题

### E1.1 目标

为 L2 的"可认证决策质量"提供**决策时刻**（strict-online，不看未来揭示时钟）对**整条候选路线**的连续温度条件，
并给出该条件的**在线计算成本**定界。这是 `Q3-1`（括区未覆盖完整路线）与 `Q3-3`（括区在线可算性未给预算）的闭合尝试。

### E1.2 待检验命题（H-S1）

> **H-S1**：对任意在决策时刻合法的候选路线（含**冻结前缀**、**不可取消承诺**、**WAIT**、**返仓**、以及
> **mid-tour 入口状态**），存在一个**只由计划与实例数组决定**、**不读取任何未来揭示时钟**的实值区间
> `[lo, hi]`（"分割无关括区"），使得执行器在**每一个门事件**上的**实值**温度落在其中；
> 且任一 zone 的"门事件上界"给出该 zone 在整个路线上的上界。

### E1.3 待检验命题（H-S2）

> **H-S2**：该括区的两端可在**不引入 libm（`math.exp` / `math.log` 正确性）任何假设**的前提下，
> 以精确有理算术判定；且单次"全车队候选"认证的实测耗时 ≤ E5.3 预登记上限。

### E1.4 可证伪点（任一成立即本轮 FAIL）

* **F-a**：合法候选路线中存在一个门事件，其执行器实值温度**落在括区之外**（E4 的有限控制中出现 1 例即 FAIL）。
* **F-b**：判定路径（返回裁决的函数）**调用了 `math.exp` / `math.log` / `math.pow`**（AST 机械检查）。
* **F-c**：单次全车队认证中位耗时 > E5.3 上限。
* **F-d**：括区的两端在同一次判定中出现 `lo > hi`（空界）而未 REFUSED。
* **F-e**：`n_exp` 的闭合式与实测 `math.exp` 调用数**不相等**（任一样本）。
* **F-f**：本轮的"证明部分"被写成依赖数值抽样（即把 E4 的有限控制当作定理）。

---

## E2 对象、判据与裁决阶梯

### E2.1 记号（全部由本文件定义，实施必须逐字对应）

| 符号 | 定义 |
|---|---|
| `C` | 决策时刻时钟（strict-online 当前 clock） |
| `R = (n_1 … n_m)` | 某车辆的**候选后缀**（`mutable_suffix`），`n_m` 可为 0（返仓）或空（WAIT） |
| `P = (p_1 … p_q)` | 某车辆的**冻结前缀**（`committed_next` 这一段已固定、不可撤销的 leg），可为空 |
| `c_v` | 车辆 `v` 当前的 `current_node` |
| `ρ_v` | 车辆 `v` 的 `ready_time`（决策点准备之后） |
| `z(n)` | 实例数组 `temp_class[n]`（门所在的 zone） |
| `δ_z` | 精确有理数 `door_heat_kwh[z] / thermal_capacity_kwh_per_c[z]`（**不是**浮点商） |
| `T_∞z` | 精确有理数 `A - c_z/h_z` |
| `margin_z` | 精确有理数 `c_z - h_z·max(0, A - τ_z)`（D3 域条件要求 > 0） |
| `Θ_z` | 唯一的 `t` 使 `F_t(τ_z + δ_z) = τ_z`，即 `(1/h_z)·ln((τ_z+δ_z-T_∞z)/(τ_z-T_∞z))` |
| `Φ_Δ(T)` | 执行器单段无门推进的**实值**闭式 `A + (T-A)e^{-hΔ} - cΔ`（**无** `max(τ,·)`） |
| `F_Δ(T)` | ODE 流 `T_∞ + (T-T_∞)e^{-hΔ}` |
| 门时钟 | 第 `i` 次 `temp_class = z` 的服务完成时刻 `f_i` |
| gap | 同一车辆同一 zone 相邻两个门时钟之差（第一个门的前一个 gap 从 `C` 起算） |

**门时钟的计划决定性（S1 的核心引理，需在报告中给证明）**：

```
d_1 = max(ρ_v, C)                      # _commit: depart = max(ready_time, clock)
a_i = d_i + travel(n_{i-1} → n_i)      # _commit: arrive = depart + travel
s_i = max(a_i, tw_start[n_i])          # _commit: service_start = max(arrive, tw_start)
f_i = s_i + service_time[n_i]          # _commit: finish = service_start + service_time
d_{i+1} = f_i                          # _plan_and_commit 在同一 clock 上 pop 下一个后缀元素
```

⇒ **gap 总长只依赖（冻结前缀, 计划, 实例数组）**，不依赖任何未来揭示时刻。
**声明域限制（必须写入裁决记录）**：若在某个已 committed leg 内部发生 reveal，`mutable_suffix` 被清空
（`strict_online_env.py:528`），则该"计划被如实执行"的前提失效；本证书是**条件式计划证书**
（"若该计划被如实执行"），不是轨迹证书。

### E2.2 分割无关括区（实施必须逐字对应）

对每个 `(v, z)`，按 gap 顺序传播两个有理端点 `(lo, hi)`：

```
入口:      lo_0 = hi_0 = entry interval 的下/上端（fresh 车 = [τ_z, τ_z]；mid-tour 车 = 声明的入口区间）
每个 gap Δ:
   若 lo == hi == τ_z:                     # 已坍缩，R 上永久停在 τ_z（需 margin_z > 0）
       端点不变（0 次 exp）
   否则若 Δ ≥ Θ_z^hi:                       # 已证返回，坍缩到 τ_z（0 次 exp）
       lo ← τ_z,  hi ← τ_z
   否则:                                    # 需要精确括号（1 次 exp 括号，同时供两端使用）
       E = exp_neg_enclosure(h_z·Δ)          # 有理 [E_lo, E_hi]，对**真**指数成立
       hi ← max(τ_z, T_∞z + (hi - T_∞z)·E_hi)     # ODE 端（L-G）
       lo ← max(τ_z, A + (lo - A)·E_hi - c_z·Δ)   # 最粗分割端（BR-1；因 lo < A，取 E_hi 得最下端）
   若该 gap 以 zone-z 的门结束:
       lo ← lo + δ_z,  hi ← hi + δ_z
       （并记录门后端点；`hi` 即在此时刻之前所有存值上界）
```

**band 判据**：`sup_z := max over zone-z 门 of hi_after_door`（含入口端点的 `hi_0`）；
`band_ok_z ⟺ sup_z ≤ τ_z + δ_z`（精确有理比较，**无容差**）。
**hard 判据**：`sup_z ≤ hard_high_c[z]` 且 `entry ≥ hard_low_c[z]`（精确有理比较）。

### E2.3 裁决阶梯（沿用 plan-layer 既有四级，禁止新增词汇）

| 裁决 | 条件 |
|---|---|
| `REFUSED` | 身份不匹配 / 域外参数 / 畸形输入 / 服务未揭示节点 / 承诺集不符 / 空界（`lo > hi`）/ 前缀时钟与重算不符 / 非有限值 |
| `NOT_CERTIFIABLE` | 某 zone 的 `band_ok_z` 为假，或入口已越 hard bound |
| `DIAGNOSTIC_ONLY` | 算术完成但某个 binding 缺失（入口状态未绑定 / 承诺集未绑定 / source identity 未绑定） |
| `PROVED_CONDITIONAL` | 全部 binding 就位且每个 zone 的 `band_ok_z` 为真 |

**明确**：band 严格位于 hard bound 之内（逐 zone 数值见证，随报告给出），故 **band 是绑定判据**，
hard 判据只作为附加报告，不用于裁决升级。

### E2.4 本轮明确**不**建立的东西

* **F-level（执行器 float）**：括区是**实值**区间；"执行器存下来的 double 落在该实值区间内"这一步需要
  `math.exp` 的误差界 κ 或位等价重放，**属 S3，本轮不做**。因此本轮裁决只能写"实值安全"。
* **段内（连续）剖面**：执行器不定义段内轨迹，本轮不主张。
* **计划可行性/经济性**：TW、容量、能耗预算、品质均不在本轮判据内。

---

## E3 数据、合同与源码身份（冻结）

### E3.1 合同

`CONTRACT = ColdChainContractV2()`（默认构造，与 Q3 既有审核一致），`contract_hash = f70c91632f782188d07950832ec03a3bb94a7c561572cbd56edf7b8d76d53fc0`。

| zone | τ | δ (精确 δ) | A | h | c | T_∞ | margin | Θ (有理区间中心，见 S2) | band | hard |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 18.0 | 1/30 | 25.0 | 0.2 | 2.0 | 15.0 | 0.6 | ≈0.055249 h | [18, 18.0333] | [0, 30] |
| 1 | 4.0 | 0.0875 | 25.0 | 0.16 | 5.0 | −6.25 | 1.64 | ≈0.053127 h | [4, 4.0875] | [0, 8] |
| 2 | −18.0 | 0.12 | 25.0 | 0.12 | 8.0 | −41.6667 | 2.84 | ≈0.042147 h | [−18, −17.88] | [−25, −12] |

### E3.2 数据

* **唯一真实数据**：`data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz`（16 实例 × 51 节点）。
  只读其公开数组（`coords / dist_mat / tw_start / tw_end / service_time / temp_class / demands /
  reveal_time / initial_quality`）。**不读** `data/m0_scale` 的教师标注用于任何判据（本轮不需要）。
* **不使用** `data/baseline/**`、`data/heldout/**`、`data/archive/**`、最终测试种子。
* **合成控制**：允许并**要求**使用合成实例与合成合同（域外参数、退化合同），生成规则必须写死在测试模块里、
  随机种子固定并在报告中登记；合成结果只能支撑"判据可达/失败边界存在"，不支撑任何性能主张。

### E3.3 冻结源码身份（**committed blob sha256**，`git show HEAD:<path>` 的字节）

```
C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_state.py = a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e
C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_contract.py = 0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2
C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/strict_online_env.py = fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6
C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/q3_strict_float_experimental.py = 56e8a323135bbd3db0978a2a700a1d94fe8e8433cc71709e45dc5d37f13a76e6
C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/q3_float_replay_qualification_experimental.py = c29244366fac65258a512c00e2463dcc479d43718c5ff13cf8f26468857095e3
C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/q3_event_segmentation_experimental.py = aef0c8d35c34fe9596e8ac7f5fbfedba75c350f6ba9d21b0c6f88eebb4ee99c9
C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/q3_plan_layer_experimental.py = 4a9643725c2bbfe58bce1cae306d47647b75120a0f840fee9d2e083435b91f43
```

**注意**：`coldchain_state.py` / `coldchain_contract.py` / `strict_online_env.py` 三条与
`Q3冻结身份登记_2026-10-17.md` 登记的 64-hex **逐字一致**（该表登记值经本轮复核确为 committed blob 摘要）。

本轮**新增**模块（实施后回填，形态为"报告登记、源码不写字面量"）：

```
scripts/coldchain/q3_s1_route_bracket_experimental.py      （待实施后回填）
scripts/coldchain/q3_s2_online_cost_experimental.py        （待实施后回填）
scripts/tests/test_q3_s1_route_bracket.py                  （待实施后回填）
scripts/tests/test_q3_s2_online_cost.py                    （待实施后回填）
```

### E3.4 运行环境

Windows / CPython 3.12 / numpy；**无 GPU、无训练、无 rollout**。S2 的计时为**本机隔离计时**
（`time.perf_counter`，预热后取中位数），登记为"本机诊断"，不得当作跨平台保证。

---

## E4 控制行花名册（**先声明，后运行**）

### E4.1 S1 行（`scripts/tests/test_q3_s1_route_bracket.py`）

| ID | 类型 | 预登记期望 | 内容 |
|---|---|---|---|
| `S1-P1` | 正控 | `PROVED_CONDITIONAL` 且 `n_short_gaps == 0` | 真实实例 0 的贪心合法路线（fresh 车、含 WAIT 与返仓），所有 gap ≥ Θ^hi |
| `S1-P2` | 正控（数值确认） | 1500 条随机分割轨迹，**0 例外** | 对每个 gap 取 1…8 个随机切点、按执行器逐段推进，门事件实值必须落在括区内 |
| `S1-P3` | 正控（端到端） | 门时钟**逐位相等**，0 偏差 | 把 `reveal_time` 全置 0（合成无揭示实例）+ 一次性 replanner，使计划被如实执行；重算门时钟 vs `env` 记录 `service_finish` |
| `S1-P4` | 正控 | 括区宽度 = 0（所有 gap 长于 Θ） | 长 gap 坍缩到 τ 后 `hi - lo == 0` |
| `S1-P5` | 正控 | 短 gap 可被 mid-tour 低入口状态认证 | 入口在 `[τ, τ+δ)` 内部时，`Δ < Θ` 仍 `band_ok`（P5 的二元条件会拒绝）——证明括区**严格推广** P5 |
| `S1-P6` | 正控 | 前缀（committed leg）参与且重算一致 | 给定非空冻结前缀，重算时钟与调用方自报时钟逐位相同才 `PROVED` |
| `S1-N1` | 负控 | `NOT_CERTIFIABLE` 且 `surplus > 0` | 人为构造 `Δ < Θ` 的短 gap（fresh 入口）⇒ band 越界 |
| `S1-N2` | 负控 | `NOT_CERTIFIABLE` | 入口温度越出 hard bound（如 zone0 入口 40 ℃） |
| `S1-N3` | 负控 | `REFUSED` / `NONPOSITIVE_HEAT_TRANSFER_OUTSIDE_DOMAIN` | 合成合同 `h = 0`（域外） |
| `S1-N4` | 负控 | `REFUSED` / `DRIFT_CONDITION_VIOLATED` | 合成合同 `margin ≤ 0`（域外） |
| `S1-N5` | 负控 | `REFUSED` / `EMPTY_BRACKET` | 注入 `lo > hi` 的人工端点 ⇒ 必须拒绝，**不得**静默交换 |
| `S1-N6` | 负控 | `REFUSED` | 计划服务未揭示节点 / 承诺集不符 / 路线内含节点 0 / 重复节点 |
| `S1-N7` | 负控 | `REFUSED` | 前缀时钟与重算不符（自报 `committed_finish` 被篡改） |
| `S1-N8` | 负控 | `REFUSED` | `active_zone_mask` 含 `False`（域外） |
| `S1-N9` | 负控 | `DIAGNOSTIC_ONLY` | 入口状态未绑定 / 承诺集未绑定（binding 缺失不得升级为 PROVED） |
| `S1-N10` | 负控 | `REFUSED` | 非有限值（`nan`/`inf`）出现在 `tw_start` / `service_time` / 入口 |

**退化界（"空界"）专项**：`δ_z = 0`（`door_heat_kwh[z] = 0`）时 `Θ_z = 0`、括区宽度恒 0，
`S1-D1` 行要求此时任何合法计划都 `PROVED_CONDITIONAL`、`n_exp == 0` 且 `n_exp_thresholds == 0`
（因为 `δ_z = 0 ⇒ K_z = 1 ⇒ ln K = 0` 精确，无需任何 exp 括号）——这是"界非空但不约束"的正向边界。

### E4.2 S2 行（`scripts/tests/test_q3_s2_online_cost.py`）

| ID | 类型 | 预登记期望 | 内容 |
|---|---|---|---|
| `S2-1` | 正控 | 100/100 等式成立 | 实测 `math.exp` 调用数 == 闭合式 `n_exp(plan) = #S`（`S` = 需精确括号的 gap 集，逐计划计数；Θ 阈值属合同级一次性成本 `#S_th = #{z : δ_z > 0}`，单独计数） |
| `S2-2` | 正控 | 坍缩路径 0 次 exp | 全部 gap ≥ Θ^hi 时 `n_exp == 0`（逐计划），且 `n_exp_thresholds == #S_th` |
| `S2-3` | 正控（机械） | AST 判定路径无 `math.exp`/`math.log`/`math.pow` | 对 S1/S2 的裁决函数做 AST 静态检查（F-b 的可证伪检查） |
| `S2-4` | 正控 | 精确有理数，无浮点阈值 | 裁决比较的两端均为 `Fraction`（类型断言） |
| `S2-5` | 正控 | ≤ 上限 | 单次全车队认证中位耗时 ≤ E5.3 上限（隔离计时，预热后 200 次） |
| `S2-6` | 正控 | 线性 | `n_stops` 从 10 增到 40 时实测耗时 `t(40) ≤ 6·t(10)`（闭合式为线性；不要求逐点单调，避免常数噪声导致假 FAIL） |
| `S2-7` | 正控（上界） | 实测 == 闭合式上界 | 全短 gap 最坏形态：`n_exp == n_doors + Z·A`（`Z` zone 数，`A` 有非空路线的车数） |
| `S2-N1` | 负控 | 计数必须**真的**被测到 | 把闭合式故意写错一个常数，`S2-1` 必须 FAIL（证明该行不是自证） |

### E4.3 探针（**不进入判据**）

`Θ` 有理区间与 `math.log` 浮点值之差、`δ_real` 与 `δ_float` 之差、括区余量分布：只作报告，作 PASS/FAIL 依据**无效**。

---

## E5 接受规则

### E5.1 S1 PASS 条件（全部满足）

1. `S1-P1…P6`、`S1-D1` 全部 `ok=True`；
2. `S1-N1…N10` 全部给出**预登记的那个**裁决与拒绝码（不是"任意拒绝"）；
3. `S1-P2` 的 1500 条随机分割轨迹 **0 例外**；
4. `S1-P3` 的真实 `env` 门时钟复现 **0 偏差**。

### E5.2 S2 PASS 条件（全部满足）

1. `S2-1`、`S2-7` 计数等式 **100/100**；
2. `S2-3` AST 检查通过（判定路径**不调用** `math.exp`/`math.log`/`math.pow`）；
3. `S2-4` 裁决比较两端为精确有理；
4. `S2-5`、`S2-6` 成本门通过。

### E5.3 成本上限（**现在登记，不得事后调整**）

* 决策预算 `B_dec = 10.0 s`（来源：本仓库 L2 相关臂统一使用的 `time_limit = 10.0` 每次决策软截止；
  已在 `diag_a1_*` / `diag_adjudicate.py` 中以 `time_limit != 10.0` 作为配置合格性检查）。
* 每次决策允许认证的候选数 `C_ADMISSIBLE = 250`（来源：仓库既有 LNS/rolling 臂 `K = 10` 轮 × 最多 25 车
  = 250 次计划草案/决策）。
* ⇒ **单次"全车队候选"认证上限 `t_max = B_dec / C_ADMISSIBLE = 40 ms`**。
* 计时口径：`time.perf_counter`，先预热 20 次，再取 **200 次中位数**；单进程、无 I/O、无 numpy 调用在计时区内。

### E5.4 证明 vs 确认（**禁止混写**）

* **PROVED（引理级，本轮需给证明）**：BR-1（细化抬高）、L-F（单步支配）、L-G（ODE 流为分割无关上界）、
  地板不降低、两端的单调性、门时钟的计划决定性、`Θ` 有理区间的正确性（`ln` 级数尾界）。
* **CONFIRMED（有限数值确认，不构成证明）**：`S1-P2` 的 1500 条随机分割、`S1-P3` 的真实轨迹复现、
  `S2-1` 的计数等式、`S2-5` 的计时。
* 报告中必须对每个结论标注 `PROVED` / `CONFIRMED` / `NOT_ESTABLISHED` 三者之一。

---

## E6 失败与止损规则

1. **任一预登记行 FAIL ⇒ 本轮 S1/S2 判 FAIL**。禁止通过以下任一方式追求 PASS：
   放松 band、给 band 加容差、改用 hard bound 代替 band、缩小 gap 集、剔除不利计划、
   追加运行次数、"再试一次不同的随机种子"。
2. **禁止把数值零例外写成证明**（F-f）。若报告中出现该表述，本轮结论作废。
3. **S2 若只能给出需要 κ 的浮点界 ⇒ 判 FAIL**，记为"实值界存在但决策时刻不可判定"，
   **不得**把 S3 的输入前移当作本轮成果。
4. **止损联动**：按 `M1_L2最小可行性与成本裁决建议_2026-10-17.md` §10.2，若两轮授权内 S1+S2 与 S3
   不能同时建立 ⇒ 转 **NO-GO**，不放松物理标准、不无限扩张。本轮只消耗其中一轮。
5. 完成后**立即停止**，等待用户对下一项的明确授权。**本轮不自动授权 S3。**

---

## E7 禁止事项与不主张

### E7.1 禁止

* 不修改任何冻结实验、历史裁决、`MASKCO_code/`、生产认证器、安装入口；
* 不开启 C3 / G6 / 训练 / GPU / 正式 rollout / 真实安装 / 最终测试种子 / 正式 L2/L3；
* 不新增生产可消费凭据（`production_consumable` 恒 `False`）；
* 不重开 C2 / G-3 / G-2 全系列；不为消除每个 `UNKNOWN` 新建模块；
* 不在源码中写裸 64-hex 摘要（仓库约定：摘要放报告）；
* 不用 `--skip-secret-scan`、不 force push、不改写历史、不 `git add -A` 暂存 `results/**`。

### E7.2 不主张

* 不主张 F-level（执行器 float）认证；不主张段内连续剖面安全；
* 不主张 S3 已建立；不主张 L2 可启动；不主张任何前置已合格；
* 不主张任何性能优势、学习增量或在线效用；
* 不主张本轮结果可迁移到未检验的实例规模、节点数、数据族或距 models；
* 不主张"数值抽样零违例 ⇒ 数学证明"。

---

## 附：本轮交付物清单

1. 本预声明（先冻结、先推送）；
2. `scripts/coldchain/q3_s1_route_bracket_experimental.py`；
3. `scripts/coldchain/q3_s2_online_cost_experimental.py`；
4. `scripts/tests/test_q3_s1_route_bracket.py`（E4.1 全部行）；
5. `scripts/tests/test_q3_s2_online_cost.py`（E4.2 全部行）；
6. 独立 Go/No-Go 报告（含 `PROVED`/`CONFIRMED`/`NOT_ESTABLISHED` 标注与 **S3 全部必要条件清单**）；
7. 两份权威状态文件的**增量**条目（保留全部历史）。
