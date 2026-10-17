# Q3 第二轮七要素预声明：S3 执行器浮点外扩界（最后一轮 Q3 隔离资格研究）

> **性质**：预声明（本文件在**实施之前**冻结）。冻结时点基准 HEAD：`ff243c1de711b5f81f05901bc712ee02a32aa186`。
> **授权性质**：**最后一个** Q3 隔离资格研究轮次。**不授权**生产替换或正式 L2 实验。
> **止损协议位置**：`M1_L2最小可行性与成本裁决建议_2026-10-17.md` §10.2 的两轮止损**第二轮（共两轮）**。
> 本轮结束**即** GO / NO-GO；**不追加第三轮**；**不允许**把 S3 拆成多个资格战役。
> **本轮不修改**任何冻结实验、历史裁决、`MASKCO_code/`、生产认证器、安装入口、C2/G-3。

---

## E1 目标、命题与可证伪点

### E1.1 唯一主目标

> 获得一把**决策时刻可计算**、**保守包含实际浮点执行器结果**、且**对全部允许的未来事件分段成立**的
> **执行器浮点（F-level）外扩界**。

**不允许**以事后（partition 已知）的**位等价重放**替代**决策前认证**。
理由已 PROVED：分段时钟 = `min(下一 reveal, 预约截止, 各 finish)`，其中 reveal 是未来信息，
且分段改变终值（1 段 `4.054313` vs 4 段 `4.054433`，差 `1.20e-04 ℃`）⇒ 逐位重放**不是**决策时刻可认证的性质。
S3 的"位等价重放"分支**在本轮被明确删除**，不再作为备选。

### E1.2 核心通过条件（形式化，与用户任务书一致）

```
∀ P ∈ P_declared,  ∀ partition π ∈ Π(P):   T_executor(P, π) ∈ [ lo_S3(P), hi_S3(P) ]
且 hi_S3(P) ≤ τ_z + δ_z + ε_F  （band + 预登记容差）
且 lo_S3(P) ≥ τ_z − ε_F
且 hi_S3(P) ≤ hard_high_z,  lo_S3(P) ≥ hard_low_z
```

其中 `Π(P)` 是执行器在该计划上**全部允许的分段**（由 `n_max` 界定件数），
`lo_S3/hi_S3` 必须**只由（冻结前缀, 计划, 实例数组, 合同, 声明的 κ）**决定。

### E1.3 待检验命题

* **H-S3a（可构造性）**：存在把 S1 的实值括区外扩到 F-level 的**误差预算**，其每一项都可独立核验，
  且在 `κ` 给定时给出**非空**且落在 band 内的外扩界。
* **H-S3b（不可回避性）**：任何对外扩界的有效主张**必须**依赖一个关于平台 `math.exp` 的
  **一致相对精度**陈述；该陈述**不能**从执行器合同、IEEE-754 或 C 标准推出。
* **H-S3c（κ 门槛）**：存在 `κ*`，使 `κ ≤ κ*` 时 H-S3a 的结论成立；且 `κ*` 远大于任何现实 libm 的
  实测误差（从而"缺的只是**保证**，不是**能力**"）。

### E1.4 可证伪点（任一成立即 S3 FAIL，且按 E6 转 NO-GO）

| # | 可证伪点 |
|---|---|
| **F-a** | 存在合法计划与合法分段，使执行器存储温度**越出** `[lo_S3, hi_S3]`（有限控制中 1 例即 FAIL） |
| **F-b** | 外扩界**空**（`hi_S3` 超 band+`ε_F`，或 `lo_S3` 低到无信息）而没有在预声明计划集上被判 `NOT_CERTIFIABLE` |
| **F-c** | 误差预算**漏计**任一项（κ / 四则舍入 / 地板 / 门处 δ / 时间差与单位换算 / 多段累计 / 跳过段 / 未知分段）|
| **F-d** | 在外扩界中使用**未来分段信息**（必须机械检查：`lo_S3/hi_S3` 的计算不读 reveal 时刻） |
| **F-e** | 把**经验 κ**（实测值）当作证明填入参数表并宣布认证成功 |
| **F-f** | 事后放宽 `ε_F`、`G-T1=40 ms`、或减少候选数/样本集后宣称原协议通过 |

---

## E2 对象、判据与裁决阶梯

### E2.1 κ 的对象与**可接受来源**（这是本轮最关键的一条）

`κ` 的定义：对**声明输入域** `Dκ` 内的一切 `x`，
`|fl_exp(x) − exp(x)| ≤ κ · exp(x)`，其中 `fl_exp` 是**目标平台上被冻结执行器实际调用的** `math.exp`。

**可接受来源（只有三种，且必须在预声明中指名）**：

| 代号 | 来源 | 判据 |
|---|---|---|
| `FORMAL_SPECIFICATION` | 该平台的 libm 有**正式规范**规定 `exp` 的误差界，且适用域覆盖 `Dκ` | 须给出规范文本标识与条款 |
| `VENDOR_DOCUMENTED` | 平台厂商文档给出 `exp` 的最大误差（ulp 或相对），且覆盖 `Dκ` | 须给出文档标识与适用条件 |
| `CERTIFIED_IMPLEMENTATION` | 执行器调用的 exp **不是**平台 libm，而是一个**自带证明**的实现 | 须给出证明与被冻结的调用路径（**本轮不允许**，因执行器为冻结源码） |

**明确排除（不得使用）**：实测/经验值、有限网格检验、统计论证、"所有现实 libm 都满足"、
"行业惯例"、`IEEE-754` 推理（见 E2.6）、S1 的实值括区、位等价重放。

**若三种来源均不成立** ⇒ 结论必须是 `EXP_ERROR_BOUND_NOT_ESTABLISHED`，
且 **`f_level_established = False`**，**即使算术全部算完且落在 band 内**也不得升级。

### E2.2 F-level 误差预算（逐项，全部给出闭合式）

对 zone `z`，在一个 **gap**（S1 的同一个 gap 对象，不另立口径）上：

| 项 | 闭合式 | 依据 |
|---|---|---|
| **T-κ** | `κ · S_z · n_pieces_max` | 每段 `E` 的相对误差 `κ` 经 `∂v/∂E = (v_{n}-A)` 传播；`S_z = A − τ_z` 是可达的最大 `|v−A|` |
| **T-round** | `n_pieces_max · n_ops · u · V_max` | 每段约 `n_ops = 7` 次四则运算，`u = 2^-53`（`L-B`），`V_max = max(A, |τ|+δ)` |
| **T-door** | `n_doors_z · n_ops_door · u · V_max` | 门处 `heat/capacity` 除法 + 加法各一次舍入 |
| **T-time** | `u · V_max` | `end − depart` 与 `× hours_per_time_unit` 两次舍入（`coldchain_state.py:209`、`coldchain_contract.py:518-521`） |
| **T-skip** | `margin_z · n_max · ε_time`（`ε_time = 1e-9 h`） | `coldchain_state.py:361-362`：`duration_h ≤ _EPS` 时**直接返回原值**，即**不做任何冷却**；此类"跳过段"最多 `n_max` 个 |
| **T-floor** | `0` | `max(τ,·)` 精确、且只会抬高，方向已知（**必须显式登记为 0，不得省略**） |
| **T-unknown** | `0`（在 T-κ 与 T-skip 已被 `n_max` 覆盖的意义上） | 未知分段只影响"件数"，而件数已由 `n_max` 界定 ⇒ **无需读未来时钟** |

**总预算**：`E_z = T-κ + T-round + T-door + T-time + T-skip`（精确有理数相加，**无浮点**）。
**外扩界**：`lo_S3_z = max(τ_z, lo_R_z) − E_z`，`hi_S3_z = max(τ_z, hi_R_z) + E_z`，
其中 `lo_R/hi_R` 直接来自 **S1 已冻结的括区**（同一 gap 列表、同一传播，**不重算、不改判据**）。

`n_max = 20003`（每次 rollout 每车的 advance 次数上界）**不在本文件中手写**，
必须由 `q3_g2_time_identity_experimental.call_graph_census` 在运行时重算并登记
（这正是 G-2 的结构上界第一次被真正消费）。

### E2.3 `ε_F`（预登记常数，不得事后调整）

**`ε_F = 1.0e-3 ℃`**。理由（先登记）：
* 它比最窄的 band 宽度（zone 0 的 `δ_0 ≈ 3.33e-2 ℃`）小 **33×**，比 zone 2 的 `0.12 ℃` 小 **120×**；
* 它高于声明 `κ ≤ 1e-9` 时的最坏误差预算（见 E2.5）**约 6 个数量级**，因此不是"量身定做"；
* 它是温度的**物理分辨率量级**（10⁻³ ℃），低于任何冷链决策会关心的差异。

### E2.4 裁决阶梯（沿用既有四级词汇，另加一个**只在 F-level 出现的**修饰位）

| 裁决 | 条件 |
|---|---|
| `REFUSED` | κ 无合法来源 / 合同域外 / 非有限输入 / 时间单位非法 / 身份不一致 / 空区间 |
| `NOT_CERTIFIABLE` | `hi_S3_z > τ_z + δ_z + ε_F` 或越 hard bound（**外扩界非空但不足以认证**）|
| `DIAGNOSTIC_ONLY` | 算术完成但 binding 缺失（入口状态未绑定 / 承诺集未绑定 / 真实状态来源未绑定 / `n_max` 未重算）|
| `PROVED_CONDITIONAL_F` | 全部 binding 就位、κ 来源合法、`hi_S3/lo_S3` 落在预登记界内 |

**修饰位 `f_level_established`**：只有当 `κ` 来源属 E2.1 的合法三类**且** `κ ≤ κ*` **且**裁决为
`PROVED_CONDITIONAL_F` 时才为 `True`。**其余任何情形恒为 `False`**，包括"κ 凭经验取值而算术通过"。

### E2.5 预登记的导出量（由合同常数推导，非测量 → 允许在实施前写死）

`u = 2^-53 = 1.1102e-16`，`V_max = 25.0 ℃`，`n_max = 20003`，`n_ops = 7`，`ε_time = 1e-9`：

| zone | `S_z = A − τ_z` | `margin_z` | `T-skip` | **`κ*_z`（令 `E_z ≤ ε_F`）** |
|---|---|---|---|---|
| 0 | 7 | 0.6 | 1.200e-05 | ≈ **7.06e-09** |
| 1 | 21 | 1.64 | 3.281e-05 | ≈ **2.30e-09** |
| 2 | 43 | 2.84 | 5.681e-05 | ≈ **1.10e-09** |

⇒ **预登记期望：`κ* = min_z κ*_z ≈ 1.10e-09`（zone 2 绑定）**，且 `T-skip` 是**主导项**
（比 T-round 大约 5 个数量级）⇒ **F-level 的第一瓶颈不是 `math.exp` 的精度，而是执行器的
`duration_h ≤ 1e-9` 早退**。实施必须复现该结论；若实施给出不同的绑定项，须在报告中解释。

### E2.6 不可回避性（本轮要给出的**定理级**结论）

> **引理 S3-IMP**：设证书必须对某 gap 的**全部允许分段**成立。则任一非平凡的一致的
> F-level 上界都蕴含一个关于 `fl_exp` 的**一致相对精度**陈述；反之，**不使用**任何一致精度陈述时，
> 唯一有效上界是 `fl_exp ∈ [0,1]`，它要求每个无门段 ≥ `(A−τ_z)/c_z`（= **3.5 / 4.2 / 5.375 h**），
> 远大于任何合法 gap ⇒ **外扩界为空**。

⇒ S3 的成败**完全归结为 κ 的资格**。这一点必须写成定理并配一个可执行对照（`S3-N1`/`S3-P1`），
不得写成"还缺一个细节"。

---

## E3 数据、合同、平台与源码身份（冻结）

### E3.1 平台身份（**必须在运行时读出并登记**，不得抄写）

`sys.version` / `platform.platform()` / `platform.machine()` / `sys.float_info` /
`math.exp` 的语义来源（CPython `mathmodule.c` 调用 C 库 `exp`，CPython 自身不实现 exp）。
冻结时点实测：**CPython 3.12.7（Anaconda, MSC v.1929 64-bit AMD64），Windows-11-10.0.26200**，
IEEE-754 binary64（`mant_dig = 53`，`rounds = 1`）。

### E3.2 合同与数据

* `CONTRACT = ColdChainContractV2()`，`contract_hash = f70c91632f782188d07950832ec03a3bb94a7c561572cbd56edf7b8d76d53fc0`。
* 真实数据（**增补/非门控**，与第一轮同一处置）：`data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz`。
  **门控证据一律为合成实例**（固定种子），以免重蹈证据本地性缺口。
* **不使用** `data/baseline/**`、`data/heldout/**`、`data/archive/**`、最终测试种子。

### E3.3 冻结源码身份（committed blob sha256）

```
scripts/coldchain/coldchain_state.py                          = a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e
scripts/coldchain/coldchain_contract.py                       = 0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2
scripts/simulation/strict_online_env.py                       = fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6
scripts/coldchain/q3_strict_float_experimental.py             = 56e8a323135bbd3db0978a2a700a1d94fe8e8433cc71709e45dc5d37f13a76e6
scripts/coldchain/q3_s1_route_bracket_experimental.py         = 8db4bc292a67d012871c807d0c0eb6605fb4db0d8b45bf3c731cd762ed19be83
scripts/coldchain/q3_s2_online_cost_experimental.py           = 5da844d327ac55789bdb6e858527428cee73926e5d6bbf3932ed6f49dd8a466c
scripts/coldchain/q3_g2_time_identity_experimental.py         = （运行时重算并登记，本文件不手写）
```

本轮新增模块（实施后回填，形态同第一轮：摘要放报告、源码不写裸 64-hex）：

```
scripts/coldchain/q3_s3_float_outward_bound_experimental.py   （待实施后回填）
scripts/tests/test_q3_s3_float_outward_bound.py               （待实施后回填）
```

---

## E4 控制行花名册（**先声明，后运行**）

### E4.1 正控

| ID | 预登记期望 | 内容 |
|---|---|---|
| `S3-P1` | `REFUSED / EXP_ERROR_BOUND_NOT_ESTABLISHED`，`f_level_established=False` | 不给 κ ⇒ 必须拒绝，且**不得**给出任何 F-level 结论 |
| `S3-P2` | `PROVED_CONDITIONAL_F`（**条件于声明的 κ**） | 给定一个**来源被显式标注为 `DECLARED_ASSUMPTION_NOT_PROVED`** 的 κ，机制给出非空外扩界；该行**只**证明"机制可用"，**不**证明 κ 成立 |
| `S3-P3` | 网格 0 违例（**CONFIRMED，非 PROVED**） | 在声明输入域上实测 `fl_exp` 相对误差；并与 `κ = 1 ulp` 比对；报告中必须标注 `CONFIRMED` |
| `S3-P4` | 复现 E2.5 的 `κ* ≈ 1.10e-09` 与主导项 = `T-skip` | 门槛与主导项必须由代码算出，不是抄写 |
| `S3-P5` | 与 `SF.exp_neg_enclosure` 一致 | 预算中的有理数运算与既有 rigour primitive 不冲突 |

### E4.2 负控（覆盖用户任务书 §4 的 8 条 + 本轮新增）

| ID | 用户要求对应 | 预登记期望 |
|---|---|---|
| `S3-N1` | ① κ 缺失 | `REFUSED / EXP_ERROR_BOUND_NOT_ESTABLISHED`；`f_level_established=False` |
| `S3-N2` | ② κ 小于实际误差 | 把 κ 设为 **0** 时，**必须**在执行器 float 值越界处被检出（`NOT_CERTIFIABLE` 或 containment 违例） |
| `S3-N3` | ③ 实值通过但执行器越界 | S1 实值 `PROVED` 的计划，若执行器 float 值越出实值括区（第一轮实测 247/1500），**不得**被判 F-level 通过 |
| `S3-N4` | ④ 未知分段改变终值 | 同一 gap 的不同分段给出不同终值 ⇒ **单条事后轨迹不能代替全集**；控制必须给出"单轨迹 vs 全分段上确界"的差 |
| `S3-N5` | ⑤ 门加法累计舍入 | 从预算中**移除 `T-door`** 后必须能被检出（对拍出越界） |
| `S3-N6` | ⑥ 大量短 gap 使界失效 | 短 gap 足够多使 `hi_S3 > τ+δ+ε_F` 时必须 `NOT_CERTIFIABLE`，**不得**静默通过 |
| `S3-N7` | ⑦ 无效合同/非有限/错误时间单位 | `REFUSED`，逐个子例给出**具体**码 |
| `S3-N8` | ⑧ 状态来源与身份 | `state_identity` / `source_identity` 缺失或不一致 ⇒ `DIAGNOSTIC_ONLY` / `REFUSED`，**不得**升级 |
| `S3-N9` | **新增**（本轮发现） | **跳过段**：构造把 gap 切成大量 `≤1e-9` 片段的分段，实测执行器值**高于** S1 的 ODE 上端 ⇒ 证明"R-level 上界不是 F-level 上界"，且 `T-skip` 必须存在 |
| `S3-N10` | **新增** | **未来信息机械检查**：`lo_S3/hi_S3` 的计算路径不得读取 `reveal_time`（AST + 运行期双重检查） |
| `S3-N11` | **新增** | κ 来源为**经验值**时必须被 `REFUSED`（`KAPPA_SOURCE_NOT_ACCEPTABLE`），即使算术通过 |

### E4.3 非空性与成本

| ID | 预登记期望 |
|---|---|
| `S3-E1` | 在**事先固定**的合成计划集（固定种子、固定规模、不挑选）上：给定声明 κ 时 `PROVED_CONDITIONAL_F` 的比例、失败原因、以及 `hi_S3` 的最大余量**全部报告**；**不得**剔除不利计划 |
| `S3-E2` | 单候选 `G-T1 ≤ 40 ms`（**中位数与最大值都报**） |
| `S3-E3` | `G-T2`（完整 10 s 在线决策）**未测得**时必须写 `FULL_ONLINE_10S_BUDGET_UNMEASURED`，**不得**算作零成本 |

---

## E5 接受规则与 GO/NO-GO

### E5.1 S3 PASS 条件（全部满足）

1. `S3-P1…P5`、`S3-N1…N11`、`S3-E1…E3` 全部给出**预登记的那个**结果；
2. **κ 来源合法**（E2.1 三类之一），且 `κ ≤ κ*`；
3. 预声明计划集上 **非空**且 `hi_S3 ≤ τ+δ+ε_F`；
4. `G-T1` 通过，且 `G-T2` 若未测得则被如实标注 `UNMEASURED`。

### E5.2 GO / NO-GO（与用户任务书 §6 一致）

* **GO**：数学或**可核验平台保证**支持 F-level 外扩；所有适用输入与舍入路径均被覆盖；
  未知未来分段无需泄漏；非空性与事前成本门通过。
* **NO-GO**：**κ 无可靠依据**、未知分段不能保守覆盖、外扩界没有认证能力、时间预算未过，
  或出现不可修复的协议违规。

**预登记判据**：`S3-P1` 的期望就是 `REFUSED`。因此**若 κ 无合法来源，本轮必然是 NO-GO**，
且这不是"意外"，而是 E2.1 的直接推论。**不得**为把 NO-GO 变成 GO 而修改 E2.1 的来源清单。

### E5.3 时间门（承接复核意见二）

* **`G-T1` 单候选资格门 = 40 ms**，按"每次全车队候选认证"计量，**中位数与最大值都报**，**不得事后放宽**。
* **`G-T2` 完整在线决策门 = 10 s**：本轮**只**登记"哪些部分已计时、哪些未计时"；
  **未测得的部分不得算作零成本**；不得宣称 10 s 已满足。

---

## E6 失败与止损

1. 任一预登记行 FAIL ⇒ 按 F-a…F-f 判本轮 FAIL，**不得**通过放松 `ε_F`、改 κ 来源清单、
   减少候选数、缩小/更换样本集、追加运行来追求 GO。
2. **κ 无合法来源 ⇒ NO-GO**（这是预登记的期望路径之一，不是失败后的解释）。
3. **NO-GO 即终止** Q3 资格扩张：**不改写**第一轮结果、**不追加**第三轮、**不重开** G-2/C2/G-3。
4. 完成后**立即停止**，等待用户对下一项的明确授权。
5. 若出现**不可修复的协议违规**（例如外扩界实际读了未来分段），本轮**立即**判 NO-GO 并在报告中点名。

---

## E7 禁止事项与不主张

### E7.1 禁止

* 修改任何冻结实验、历史裁决、`MASKCO_code/`、生产认证器、安装入口、C2/G-3；
* 启动 M1-b 生产接线、C3/G6、真实安装、训练、GPU、正式 L2/L3、最终测试种子；
* 用事后位等价重放替代决策前认证；
* 把经验 κ 当证明；为通过而事后调整 `ε_F`/`κ`/样本集/候选数/时间门；
* 在源码写裸 64-hex；使用 `--skip-secret-scan`；force push；改写历史；`git add -A` 暂存 `results/**`。

### E7.2 不主张（承接复核意见三）

* **不**主张段内（连续）物理温度剖面安全；执行器**不定义**段内轨迹；
* **不**主张订单品质可售性、能耗预算、最终联合认证资格；
* **不**主张 `THERMAL_CONTINUOUS_NOT_ESTABLISHED`、`PRODUCTION_NOT_QUALIFIED`、
  `Q3_FORMAL_FREEZE_NOT_AUTHORISED`、L2 可启动等状态发生变化；
* **不**主张 S3 通过即"连续温度资格全部闭合"；
* **不**把有限网格检验写成证明；**不**把"现实 libm 都满足"写成数学保证。

---

## 附：本轮交付物清单

1. 本预声明（**先冻结、先推送**）；
2. 复核意见与资格偏差登记（同一次提交）；
3. `scripts/coldchain/q3_s3_float_outward_bound_experimental.py`；
4. `scripts/tests/test_q3_s3_float_outward_bound.py`（E4 全部行）；
5. 独立 Go/No-Go 报告（含 `PROVED` / `CONFIRMED` / `NOT_ESTABLISHED` 标注、
   `EXP_ERROR_BOUND_NOT_ESTABLISHED` 登记、以及"若要有第三轮所需的**精确**前置"的清单）；
6. 两份权威状态文件的**增量**条目（保留全部历史）。
