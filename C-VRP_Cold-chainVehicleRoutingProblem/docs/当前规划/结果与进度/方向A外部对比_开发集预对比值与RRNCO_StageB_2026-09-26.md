# 方向 A 外部对比方法：开发集预对比值 + RRNCO Stage B 结果（2026-09-26）

> ## ⚠️ 作废声明（2026-09-29 追加，优先于本文档其余一切内容）
>
> 本文档 §1–§11 的全部 A-v1 协议数值（ortools_v2/pyvrp_v2、2×2 消融、gatefix_*）均为
> **R1（预检/完成时间语义分歧）与 A-05e（求解器初始载重钉死少乘 INT_SCALE）两重根因修复前
> 的批次，按 [战役协议附录](../决策与审计/A-v1正向优化战役协议_2026-09-26.md) 已全部作废，
> **不得引用**。仅保留：§2/§10 的 RRNCO Stage B + R1 冻结（旧 DCC-VRP 距离协议，与 A-v1 无关，
> 仍有效）。
>
> **当前唯一有效的 A-v1 外部求解器对比**（执行端 2026-09-29 跑完，助手独立复算一致；
> 开发集 seed 20260926，40 天，修正下游 R1/R2/R3 + A-05e）：
>
> | 臂 | 主档配对差 vs cond_hist | 接单/天 | 结论 |
> |---|---|---:|---|
> | OR-Tools accept（修正容量） | **−287.7 [−319.3, −255.7]** | 68.5 | C1 系统显著胜（三档全正） |
> | PyVRP accept（修正容量） | **−878.4 [−964.6, −790.2]** | 47.9 | C1 系统显著胜（三档全正） |
>
> 裁决层级 ①「系统 vs 公平强对照」在开发集已闭合（`results/a1_ext_a05e_{ortools,pyvrp}/README.md`）；
> 层级 ②「cond vs uncond 条件信息门」在开发 20260926 已过（C1 +71.5 [13.5,132.1]）但验证种子
> 稳健性未过（+16.5/+11.4，CI 跨零），S3 系列杠杆逐批推进中（当前 S3-2 密度打包运行中）。
> 最终测试 20260927/28 保持不碰、只测一次。

---

> **v2 更新（2026-09-26 晚）**：新增「下游 2×2 归因消融」（router × vote），结论升级——
> **SAA 前瞻投票在锁定下游上与「无前瞻 myopic 接单」相比显著为负（+229/+345/+461 @ 三档 p_c，
> greedy 与 OR-Tools 路由器一致）**；myopic 臂用满预算服务 ~69 单/天，SAA 臂只用 58–68% 预算。
> 详见下方 §4.3 与 §7。原 §1–§4.2 内容保留（其中 solver_meta 计数为 v1 有缺陷版本，v2 已纠正，
> 逐日决策数据两版逐位一致）。
>
> 执行依据：[方向A外部对比方法_选取与执行计划_2026-09-26.md](方向A外部对比方法_选取与执行计划_2026-09-26.md)。
> 全部结果 = **开发证据**（seed 20260926，40 天，已查看）；论文测试日 seed 20260927/28 未碰。
> 不动 CC_Compare 冻结链、不改 `MASKCO_code/`、不修改步骤 2/3 协议。

## 1. 交付清单

| 交付 | 位置 | 状态 |
|---|---|---|
| 决策文档 | `docs/当前规划/决策与审计/方向A外部对比方法_选取与执行计划_2026-09-26.md` | ✅ |
| OR-Tools/PyVRP A-v1 接单适配 | `scripts/evaluation/solver_accept_replanner.py` + `run_a1_external_accept.py` | ✅ 本地烟测 + 40 天开发集 |
| OR-Tools 40 天开发集 | `results/a1_external_dev/ortools/gate.json`（本地+服务器） | ✅ hard_feasible=1.0，0 超时 |
| PyVRP 40 天开发集 | `results/a1_external_dev/pyvrp/gate.json` | ✅ hard_feasible=1.0，0 超时 |
| RRNCO Stage B（真模型） | `CC_Compare/RRNCO/dcc_rh_v4/results/r0_5_run_a.json` | ✅ verdict=PASS |
| 服务器环境 | `/home/hzeng/envs/a1_ext`（py3.11 + numpy + ortools 9.11.4210 + pyvrp 0.14.0） | ✅ |

适配实现中修掉的三个坑（已固化在代码）：OR-Tools 客户真实 id 与模型节点号方向颠倒（越界 UB → 原生崩溃）；
PyVRP `Client.location` 每次返回新 wrapper（`Model.data()` 按对象 id 对账 → KeyError，须保留 add_location 原引用）；
PyVRP 路由中 depot activity 与首位客户共用 `idx=0`（须 `is_client()` 门控，否则 depot 被误映射为客户）。

## 2. RRNCO-Ordering-RH-D Stage B（真实模型验证）＝ PASS

服务器 GPU（`cc_compare` env：torch 2.11.0+cu128 / rl4co 0.6.0 / tensordict 0.13.0，checkpoint
`epoch_199.ckpt` sha256 `b1ff3191…` 与 SERVER_ENVIRONMENT 一致），`run_r0_5.py` 七门全过：

- **B3 确定性**（同子问题两次 order 逐字段一致）、**B4 三快照 + 公共链**（初始部分揭示 / 行程中间
  anchor+load / 多车冻结前缀；公共 Bridge late-reveal 6/6 complete、0 fallback、hard vector 全 true）、
  **B5 可变规模**（pool 1/2/5/10/25/50 + 空 pool + anchor-only；N<25 自动 replacement=True 抽样）、
  **B6 未来扰动**（坐标/需求/服务/TW/数量五类变形下 ordering/raw_actions/tensor hash 全部不变）、
  **B7 模型贡献**（真实 ckpt 与 edd/nearest/fixed/shuffle/uniform 五对照全部改变决策）。
- run_id `802c3a4e7a1a471bb366d0101a7ff801`，verdict **PASS**。
- 含义：RRNCO 真实模型按 R0.5 清单完成适配验证 → 按协议可升 `GO_R1_RRNCO_ORDERING_RH_D`
  （R1 三层身份冻结按计划文档要求**须用户授权**，本轮未冻结）。

## 3. 开发集 40 天对比值（seed 20260926，按天配对，day-clustered bootstrap）

| 臂 | p_c=0 效用 | p_c=(5,10,15) | p_c=(10,20,30) | 服务/天 | 拒单/天 | 能耗 kWh（B=710.7） |
|---|---:|---:|---:|---:|---:|---:|
| **ortools_accept** | **+1292.8 [1270,1314]** | **+9.4 [−118,+131]** | **−1273.9** | **68.5** | 125.5 | **710.3（100%）** |
| cond_hist | +1046.0 | −314.5 [−395,−233] | −1598.7 | 54.6 | 139.4 | 484.3（68%） |
| explicit_feat | +961.4 | −494.4 | −1883.5 | 48.5 | 145.5 | 446.1（63%） |
| pyvrp_accept | +905.3 | −575.7 [−659,−491] | −2056.7 | 47.6 | 146.4 | 710.3（100%） |
| uncond_hist | +858.9 | −628.8 | −2121.5 | 44.3 | 149.7 | 409.7（58%） |

配对差（n=40，CI 严格异于 0 者标 ★）：

| 配对差 | p_c=0 | p_c=(5,10,15) |
|---|---:|---:|
| ortools_accept − cond_hist | **+246.8 [181.6, 313.0] ★** | **+324.0 [226.3, 422.4] ★** |
| ortools_accept − explicit_feat | +331.5 [222.9, 435.8] ★ | +503.8 [352.9, 648.7] ★ |
| ortools_accept − uncond_hist | +433.9 [346.3, 521.6] ★ | +638.3 [505.8, 771.9] ★ |
| pyvrp_accept − cond_hist | **−140.8 [−194.0, −89.0] ★** | **−261.2 [−340.8, −185.7] ★** |
| pyvrp_accept − uncond_hist | +46.4 [−23.9, +118.7] | +53.1 [−53.1, +159.4] |

（cond_hist − uncond_hist = +187.1（p_c=0）/ +314.3（主 p_c），与步骤 2 门一致——本表重算自同一逐日行。）

## 4. 两个决定性发现（开发证据，决定论文对照策略）

### 4.1 「myopic + 强路由」不是一个良定义的单一基线——两个冻结级求解器给出相反符号

- 同一接单规则（完整覆盖可行 + 真实 C0 `certify_plan` ≤ B）、同一信息边界、同一评价器，
  仅求解器不同：**OR-Tools accept 显著胜 cond_hist（p_c=0 +247 ★），PyVRP accept 显著负于
  cond_hist（p_c=0 −141 ★）**。逐日服务数相关仅 0.37（68.5 vs 47.6/天）。
- 机制（本地诊断，两求解器同轨迹逐 accept 记录）：对**相同已接受池**，PyVRP（HGS）计划的
  C0 完成能耗系统性高于 OR-Tools（GLS）计划——accept 时刻均值 665 vs 623 kWh，pool 10–20 档
  572.6 vs 479.0 kWh（差 ≈ 6–13% B）→ PyVRP 侧更多 `certify_budget` 拒绝（1528 vs 1280/2000 决策）
  → 接单轨迹分叉。**绿色预算下「距离最优路由」≠「能耗可行路由」**：距离目标不含车辆级待命制冷，
  两个都近距离最优的求解器在能耗结构上系统性分叉。
- 论文口径：外部经典臂必须**两个都报**并声明「经典 RH 接单基线对路由器敏感」；不能只挑
  OR-Tools 或只挑 PyVRP。

### 4.2 OR-Tools accept 是当前开发集上最强的非学习臂——SAA 臂的预算预留明显偏保守

- 两求解器臂把预算用满（710.3 ≈ B），SAA 三臂只用 58–68%（409.7–493.9）却服务更少
  （44–56 vs 68.5/天）。即 SAA 共享下游的**未来预算预留门槛（est λ）过度保守**：以历史边际
  value/e 拒掉大量订单后，终局还有 ~200 kWh 预算闲置。
- 含义：(a) 这是步骤 2 门结论之外的**下游口径级发现**（cond−uncond 配对差不受影响，但三臂
  绝对效用都被低估）；(b) 论文若只报「MaskCO > cond_hist」不足以宣称优于经典 RH——**必须同表
  报 OR-Tools accept 并胜出，或显式声明预算预留校准**；(c) 一个低成本的下一步 = 校准 SAA 接单
  的 λ 预留（如按剩余预算/剩余时长动态化），预期能缩小大部分差距。

## 5. 下一步建议（按优先级）

1. **（决策）步骤 3 门结果**：random 臂 maskco−cond_hist = −96.9（未过门）；explicit/pretrained
   臂仍在服务器运行。三档全不过 → 按协议收束步骤 3；外部求解器臂转为负结果论文的强基线资产。
2. **SAA 预算预留校准**（无训练、低风险）：把「OR-Tools accept 用满预算 +247」的差距定位到
   λ 预留，做一档敏感性（λ×0.5 / 动态剩余预算比例），在开发集上验证。
3. **（可选，干净归因）solver-router + SAA 臂**：把求解器路由接进 SAA 下游（替换贪心插入），
   分离「路由强度」与「场景信息」的各自贡献（K=10 × 50 决策/天 × 40 天，10s/决策内可完成）。
4. **RRNCO R1 冻结**：Stage B 已 PASS，按 dcc_rh_v4 协议需用户授权三层身份冻结；正式 A-v1
   接单协议下的 RRNCO 对比（ordering 偏好接入接单决策）另立适配。
5. **最终同批 TEST**（模型定稿后）：全部主方法 + 内部臂 + 外部臂在 seed 20260927/28 上同批重跑；
   本轮一切数字只作开发证据。

## 6. 纪律核查（全部通过）

- 论文测试日 20260927/28：未跑（`--final-test` 未使用）。
- 服务器资源：新增 12×2 workers（96 核负载峰值 ~24），不与步骤 3 三臂互扰；GPU 用 GPU 1
  （步骤 3 的 cvrp100 臂在 GPU 0）。
- 冻结链：CC_Compare `common/`、PyVRP/OR-Tools `dcc_vrp/` 冻结身份未动；新代码全部在
  扩展工作区 + `a1_ext` 新 venv。
- 确定性：两求解器独立重跑逐日逐位一致（本地 2 天烟测 A==B + 服务器 v2 与 v1 逐日逐位一致）。

## 7. v2 新增：下游 2×2 归因消融（router × vote，2026-09-26 晚）

**方法**：`run_a1_downstream_ablation.py` 的 `AblSaaReplanner`（SaaReplanner 子类）只改
「接单侧计划来源」（router ∈ greedy/ortools/pyvrp）与「SAA 投票开关」（vote on/off）；
场景流与步骤 2 cond_hist 完全一致（arm_seed=7002、k=10、K=10、10s/决策、同 certify/效用）。
**忠实性验证**：greedy_vote_cond 重跑与步骤 2 cond_hist 逐日逐位一致（40/40 天 PASS）——
消融 harness 与生产口径对齐。

**主表（40 天，p_c=(5,10,15)）**：

| 臂 | 效用 [95% CI] | 服务/天 | 能耗(B=710.7) |
|---|---:|---:|---:|
| **greedy_novote**（myopic，无前瞻） | **+30.5 [−103, +161]** | **69.0** | 710.1（100%） |
| ortools_novote（=ortools_accept v2） | +9.4 [−125, +143] | 68.5 | 710.3（100%） |
| greedy_vote_cond（=cond_hist，主对手） | −314.5 [−396, −230] | 56.1 | 493.9（70%） |
| ortools_vote_cond | −309.5 [−393, −228] | 55.5 | 471.0（66%） |
| pyvrp_vote_cond | −439.5 [−519, −358] | 51.3 | 508.7（72%） |
| pyvrp_novote（=pyvrp_accept v2） | −575.7 [−662, −487] | 47.6 | 710.3（100%） |

**配对差（vote 效应 / router 效应，CI 严格异于 0 标 ★）**：

| 配对差 | p_c=0 | p_c=(5,10,15) | p_c=(10,20,30) |
|---|---:|---:|---:|
| greedy_novote − greedy_vote（**vote 效应，greedy**） | **+229.0 ★** | **+345.0 ★** | **+461.0 ★** |
| ortools_novote − ortools_vote（**vote 效应，ortools**） | **+211.3 ★** | **+319.0 ★** | **+426.6 ★** |
| pyvrp_novote − pyvrp_vote（vote 效应，pyvrp） | −90.5 ★ | −136.2 ★ | −182.0 ★ |
| greedy_novote − ortools_novote（router 效应，novote） | +12.4 ★ | +21.1 ★ | +29.7 ★ |
| greedy_vote − ortools_vote（router 效应，vote） | −5.3 | −5.0 | −4.8 |

**结论（开发证据，方向 A 的核心质疑）**：
1. **SAA 前瞻投票在锁定下游上是决策有害的**：关闭投票（纯 myopic 接单）在 greedy 与
   OR-Tools 路由器下稳定 +211~+461/天（三档 p_c 全严格为正）。myopic 臂把预算用满（100%），
   SAA 臂只用 58–70%。
2. **vote 效应方向依赖路由器**：对 pyvrp（其距离最优计划的 C0 能耗系统性偏高）投票反而
   +90~+182 有益——前瞻的作用是「保护一个能耗失能的路由器」，而非「改善决策」。
3. 机制：`_sim_scenario` 的 est 预算门槛（est≈5.5 kWh/单 vs 真实 ~10.4 kWh/单）与影子计划
   真实能耗硬检查（-inf）的组合使投票在多数场景下退化为「拒绝」，造成系统性过度拒单
   （SAA 三臂预算闲置 30–42%）。
4. **含义**：步骤 2 门（cond−uncond=+314）只在「投票框架内部」成立；相对无前瞻强基线
   （myopic + greedy/OR-Tools），当前锁定下游的前瞻没有绝对收益。论文若走「场景前瞻提升
   接单决策」，必须先解决此下游校准问题（见 §8），或改写主张。

## 8. 修订后的下一步（按优先级，需用户决策）

1. **（决策）步骤 3 门**：random 臂重跑确认 maskco−cond_hist = −96.9（未过门）；**explicit 臂
   崩溃**（`ValueError: sum(pvals[...,:-1]) > 1.0`，在 maskco 采样器内）；pretrained 臂仍在服务器
   运行。三档全不过 → 按协议收束步骤 3。
2. **（决策）下游校准可行性**：§7.3 的过度拒单机制是可修复的（选项：`_sim_scenario` 双分支
   同为 -inf 时回退 myopic；est 门槛改真实边际能耗；影子计划不再把全部场景订单塞入硬能耗
   检查）。修复属于**修改锁定下游** → 需重新锁定并重跑步骤 2 门，必须用户裁定。
3. **（并行，无训练）**：若走「负结果/方法论」叙事，现有 2×2 已是完整的强基线矩阵
   （myopic/greedy/ortools/pyvrp × vote），可直接支撑论文对比表。
4. RRNCO R1 三层身份冻结（需用户授权）+ RRNCO ordering 接入 A-v1 接单协议（另立适配）。
5. 最终同批 TEST（模型定稿后，seed 20260927/28）；本轮一切数字只作开发证据。

## 9. 工程注记（v2 修复与事故记录）

- **solver_meta 计数缺陷（已修）**：v1 外部臂的 n_solves/reject_reasons 因 `_reset_if_new` 按天
  清零 + 多 worker 聚合错误而低估（旧 2000 vs 真实 7761/40 天）。v2 按天抓增量聚合。
  **逐日决策数据（效用/服务/拒单/能耗）v1==v2 逐位一致**，§3 表格数值不受影响。
- **消融 harness 两处修复**：①worker 切分数据集导致采样器 rng 的 inst_idx 变成局部索引 →
  场景流与步骤 2 不一致（35/40 天分叉）——改为全数据集 + 全局实例索引；②PyVRP
  `Client.location` 新 wrapper / depot 与客户共用 idx 等坑（见 §1）。
- **⚠️ 事故（需用户知悉）**：2026-09-26 14:32 本地 `scripts/evaluation/scenario_saa.py` 存在一处
  非本助手所作的修改（mtime 14:32:33，行为与步骤 2 口径不同：直接重跑 cond_hist 日 0 仅接 4 单
  vs 步骤 2 的 56 单）。助手在诊断中用 `_sftp_get` 拉取服务器副本时**覆盖了该本地版本**
  （服务器副本经验证与步骤 2 逐位一致，已同时留在本地与服务器）。若该 14:32 版本含用户未
  提交的编辑，请从编辑器本地历史恢复；助手侧无备份。
## 10. 追加（2026-09-26 晚）：RRNCO R1 冻结完成 + 下游修复等待状态

### 10.1 RRNCO-Ordering-RH-D R1 三层身份冻结 = 完成 ✅

- Stage B 双跑复验：`b2_injection_gate` **29/29 PASS**；`run_r0_5` run_a（802c3a4e…）/
  run_b（0551f33e…）七门全 PASS、决策证据逐位一致；checkpoint sha256（b1ff3191…）全链一致；
  `verify_r0_5 --results` **ALL PASS**（服务器 + 本地）。
- 冻结链（单向，无循环）：`SOURCE_MANIFEST.json`（903b2ebe…，15 文件）
  ← `FROZEN_CONFIG.json`（rev1，0945bf5a…，BackendConfig/协调器规则/证书容差）
  ← `FREEZE_SEAL.json`（rev1）。`verify_freeze` **ALL PASS（服务器 + 本地）**。
- 判定升为 `GO_R1_RRNCO_ORDERING_RH_D`（记录于 R0_ADAPTABILITY_REPORT.md 顶部）。
- 边界：冻结对象 = 旧 strict-online DCC-VRP 协议适配；A-v1 accept 协议的 RRNCO 接单适配
  另立工作区，不进入本 seal。上游 rrnco 本地 checkout 可漂移（只读）；compute 身份以服务器
  SERVER_ENVIRONMENT 记录的 commit（823d510d…）为准。
- 注：本地曾有一版以本地字节生成的冻结链，因本地 adapter 文件与服务器字节漂移（CRLF/同步差异）
  在服务器核验失败——已改为**以服务器实际执行字节为准**重新生成全链，本地已对齐。

### 10.2 下游修复等待状态（用户裁定 B 方案）

- 用户的 15:52 审计修正（T2-T5 + `INFEASIBLE_SCENARIO_PENALTY=-1e6`）已同步本地+服务器；
  实测该版本日 0 接单数从步骤 2 的 56 骤降至 4（-1e6 有限惩罚把「任一场景不可行」的计数差
  放大成硬决策）。**按用户裁定暂停一切 scenario_saa 依赖的运行**，等修复收敛后再重跑
  门 + 2×2。
- 已就绪（收敛后一条命令即可跑）：`run_a1_downstream_ablation.py`（通用 CLI：
  sampler×router×vote×sim-mode×est-scale，逐档 p_c 独立轨迹）+ `saa_calibrated.py`
  （fixA/fixC 变体，若用户修复未覆盖可作对照）+ `_analyze_ablation.py`。

## 11. 修正后下游的全量重跑（2026-09-26 深夜）＝ 方向 A 前提的决定性负证据

### 11.1 下游修复链（用户授权「你自己改」）

在用户 15:52 审计修正（T2 预检/完成阶段截止复查、T3 影子起点对账、T4 −1e6 有限惩罚、
T5 采样时间计入时限、①收入单计、③est 基线含真实已耗能耗）之上，助手修两处（已写入
`scenario_saa.py`，带日期注释）：
1. **影子车队修复**：T3 曾跳过「未派车且空计划」的车辆 → 影子车队消失（rej 分支 st={} →
   未来订单全拒、acc 分支只剩计划车 → 塞少量即不可行）→ 投票崩溃为全拒（日 0 只接 4 单）。
   修复 = 未派车全部参与（空路线在 `_complete_u` 本就 0 能耗）。
2. **终局能耗惩罚移除**：`_sim_scenario` 对「影子计划能耗 > B」置 −1e6 → 影子塞满未来一天后
   几乎恒成立 → 场景相关价值（plan_rev/fuel/rej_loss）被淹没 → 投票退化为「p_c>0 全接 /
   p_c=0 全拒」。修复 = 只有真正不可行（TW/容量/返仓，energy is None）才置惩罚；est 门槛
   即未来策略预算规则；真实 C0 预算仍由实际接单的 certify_plan 强制。

### 11.2 修正后下游：门 + 2×2（40 天，逐日逐位）

| 臂（p_c=(5,10,15)） | 效用 | 服务/天 | 预算使用 |
|---|---:|---:|---:|
| greedy_vote_cond | +30.5 | 69.0 | 100% |
| **greedy_novote** | **+30.5** | **69.0** | **100%** |
| ortools_vote_cond | +9.4 | 68.5 | 100% |
| ortools_novote | +9.4 | 68.5 | 100% |
| pyvrp_vote_cond | −575.7 | 47.6 | 100% |
| pyvrp_novote | −575.7 | 47.6 | 100% |

- **cond − uncond（门主比较）= +0.00，CI [0.00, 0.00]，逐日逐位完全一致**（主 p_c 与
  p_c=(10,20,30) 均如此；p_c=0 为 −28.7 [−51.7, −7.3]，cond 略差）。
- **vote 臂 ≡ 对应 novote 臂，逐日逐位一致**（greedy/ortools/pyvrp 三个路由器全部验证）。
- 含义：**修正后的共享下游上，SAA 场景前瞻对决策零贡献**——接单结果完全由
  「真实 C0 认证可行的 myopic 接单」决定，条件/无条件采样器无差别。步骤 2 门在旧下游上的
  +314 条件信息效应未通过正确性审计：修正后为 0（p_c=0 甚至为负）。
- p_c=0 档：投票退化为近全拒（服务 ~7.5/天，边际未来订单价值 > 当前订单 → 拒），且 cond 略差
  于 uncond——与「条件信息有价值」的假设方向相反。

### 11.3 结论（开发证据，方向 A 决策闭环）

1. 旧下游的「条件信息增益」是旧实现机制（−inf 传染 + 收入双计 + 影子车队缺失）的产物；
   经 T2-T5 审计 + 两处修复后，**条件场景前瞻在修正下游上没有正决策价值**。
2. 最强非学习臂 = myopic（greedy 或 OR-Tools 路由器，预算用满 100%，服务 ~69 单/天）。
3. 步骤 3（maskco）各臂均跑在旧下游上，其结果（random 臂 −96.9 未过门等）相对修正下游已
   失效；修正下游上 maskco 最多与 myopic 打平。
4. **建议**：方向 A 按「修正后下游的负结果」收束——负结果/方法论论文路线
   （`negative_result_paper/` 已具备完整框架）以本 2×2 + 修正链为核心证据。
   不主张继续调下游以制造正信号（等同 p-hacking）。

## 12. 追加（2026-09-29）：A-v1 接单协议下的全对比方法臂（进行中，开发集 40 天）

同一在线任务/信息边界/硬认证/10s 预算下，各外部方法的 accept 臂（40 天 seed 20260926，
按天配对 day-clustered bootstrap，vs 步骤 2 三臂）。**本表为当前有效口径**（R1+A-05e 之后）。

> ⚠️ **2026-09-30 批次裁定（时限语义修复后）**：修复前完成的 RRNCO 两臂与全部在跑臂
> = **诊断批次**（旧版「先提交接单、再记录超时」——越限决策仍会接单；RRNCO 主档实测
> timeout 日 = {0,10,20,30}，每 worker 首日 ×1 = provider 懒加载落入首个决策，其余 36 天
> 0 次）。正式裁决待修复后代码（越限=保旧计划+拒单，推理/认证后各查一次；provider
> 预算外预载；入口+provider 源码纳入启动封存）统一重跑。下表数字仅诊断。

### 12.1 已完成臂（诊断批次数字，正式重跑前不入裁决）

| 臂（方法 × p_c） | utility | vs cond_hist | 备注 |
|---|---:|---:|---|
| OR-Tools accept（主档） | — | **+287.7** | A-05e 重基线（`results/a1_ext_a05e_ortools/`，非 ordering 入口，不受时限修复影响） |
| PyVRP accept（主档） | — | **+878.4** | A-05e 重基线（`results/a1_ext_a05e_pyvrp/`，同上） |
| RRNCO accept（p_c=0，诊断） | 824.8 [759.5, 899.4] | −659.8 [−729.0, −579.9] | fallback 6058/7761（78.1%），hard_feasible 1.0，timeout 日 {0,10,20,30}×1 |
| RRNCO accept（主档，诊断） | −689.5 [−754.3, −625.9] | −998.1 [−1102.9, −876.5] | 同上 |
| myopic 规则（p_c=0） | 1305.2 [1284.1, 1325.7] | −179.3 [−202.2, −156.6] | 无学习规则臂（run_a1_downstream_ablation，非 ordering 入口） |
| myopic 规则（主档） | 30.5 [−94.4, 149.7] | −278.1 [−308.2, −245.8] | C1 SAA 胜 myopic +278.1 |
| value-e λ=2（p_c=0） | 1403.3 [1314.0, 1498.0] | −81.3 [−175.8, 15.6] | CI 过 0 |
| value-e λ=2（主档） | 180.6 [118.1, 243.2] | −127.9 [−268.5, 17.4] | CI 过 0；能耗 72% 预算 |
| value-e λ=3 | 0.0（p_c=0）/ −1943.1（主档） | −2251.7（主档） | 退化（全拒） |

### 12.2 运行中臂（全部=诊断批次，时限修复前代码）

| 方法 | 状态 | 位置 |
|---|---|---|
| RouteFinder（vrptw 100 真模型，GPU 1） | 40 天 × 2 档运行中（tmux xrf），诊断批次 | `results/a1_ext_extra/routefinder_{0,51015}/` |
| CaDA / MVMoE（P2，CPU） | provider+烟测+臂，子代理执行中，诊断批次 | `results/a1_ext_extra/{cada,mvmoe}_{0,51015}/` |
| POMO / Sym-NCO / SGBS（P3，CPU） | 同上 | `results/a1_ext_extra/{pomo,symnco,sgbs}_{0,51015}/` |
| AttentionModel / DeepACO / Omni-VRP（P3，CPU） | 同上 | `results/a1_ext_extra/{attention,deepaco,omnivrp}_{0,51015}/` |
| Learn-Improvement-Heuristics（P4，CPU） | 同上 | `results/a1_ext_extra/lih_{0,51015}/` |
| MAPT / CO-enriched-ML / L2D | 评估完成、实施排后（见各 A1_ASSESSMENT.md） | — |
| PIP-constraint | 判死复认维持 | `CC_Compare/PIP-constraint/A1_ASSESSMENT.md` |

### 12.3 时限语义修复（2026-09-30，用户复核报告第 1/2 点对比方法侧落实）

- `OrderingAcceptReplanner._decide`：provider 推理后 + certify 后各查一次
  `elapsed ≤ time_limit`；越限 = 保旧计划 + 拒单（reason=timeout）——闭合「先提交接单、
  再记录超时」缺口（用户反例：限额 0.01s、认证 0.06s 仍接单）。
- provider 在决策预算外预载（ordering 分支 `provider._load()`；rrnco 分支
  `preference_provider._load()`），消除「首决策含模型加载」的伪超时（RRNCO 主档
  timeout 日 {0,10,20,30}×1 即此来源）。
- 启动版本封存：ordering/rrnco 臂的 `rrnco_accept_replanner.py`、`ordering_providers.py`、
  方法 provider（rrnco 另含 dcc_rh_v4 adapter+backend）纳入 source_seal 启动/结束 hash；
  每日 `max_decide_s` 入 meta（逐事件耗时审计）。
- 正式批次 = 修复后代码对全部 ordering 臂统一重跑（RRNCO/RouteFinder/P2/P3/P4 ×
  p_c=0 + 主档），在跑臂完成后启动。

**裁定原则（工作包 2026-09-29）**：是否进入论文主表按「可比性与适配验收」判定，
不按输赢筛选；CVRP-only 方法按原生变体求解排序（忽略 TW），真实 TW/C0 由下游
dcc_rh_v4 协调器 + certify_plan 硬认证兜底——此为零样本跨任务口径，各方法 RESULTS.md
写明。全表数字完成后再行汇总。

### 12.4 正式批次（修复版代码，进行中）

**已完成的正式臂**（sha 2c4dee57 replanner + 修复版 driver，source_stable=True，
provider 文件入启动封存）：

| 臂 | utility | vs cond_hist | hfr | fallback% | max_decide | 备注 |
|---|---:|---:|---:|---:|---:|---|
| RouteFinder accept（p_c=0，正式） | 1266.2 [1244.1, 1287.5] | −218.3 [−241.8, −194.3] | 1.0 | 0% | 3.89s | duplicate 修复后 40 天全可裁决；timeouts 0 |
| RouteFinder accept（主档，正式） | −26.1 [−152.1, 93.4] | −334.6 [−365.0, −302.2] | 1.0 | 0% | 3.91s | 同上；reject=certify_budget 5045 + deferred 13 |
| RRNCO accept（p_c=0，正式） | 1235.4 [1202.4, 1265.2] | −249.1 [−282.2, −220.6] | 1.0 | **0%** | 7.42s | 诊断臂 78.1% fallback → 修复版（预算外预载 + 池排除在途）0 fallback；timeouts 0 |
| RRNCO accept（主档，正式） | −73.2 [−203.1, 54.4] | −381.8 [−432.3, −338.3] | 1.0 | **0%** | 7.44s | 同上；reject=certify_budget 5131 + deferred 2 |
| CaDA accept（p_c=0，正式） | 1157.0 [1120.6, 1190.8] | −327.5 [−365.4, −293.0] | 1.0 | 0% | 4.59s | CVRP-only 零样本口径；timeouts 0 |
| CaDA accept（主档，正式） | −190.3 [−327.8, −54.5] | −498.8 [−554.6, −445.5] | 1.0 | 0% | 4.44s | 同上；CPU 诊断臂 67% 决策超时 → GPU 正式臂 timeouts 0 |
| MVMoE accept（p_c=0，正式） | 1257.7 [1231.3, 1284.0] | −226.9 [−256.6, −196.7] | 1.0 | 0% | 3.17s | VRPTW 真实解算（vrptw_n100）；timeouts 0 |
| MVMoE accept（主档，正式） | −38.6 [−165.6, 83.4] | −347.1 [−389.1, −304.9] | 1.0 | 0% | 2.06s | 同上 |
| AttentionModel accept（p_c=0，正式） | 1236.2 [1207.3, 1263.2] | −248.3 [−274.7, −222.9] | 1.0 | 0% | 0.22s | 官方 greedy 档；CPU 低负载 |
| AttentionModel accept（主档，正式） | −72.9 [−196.8, 46.6] | −381.5 [−418.4, −345.6] | 1.0 | 0% | 0.30s | 同上 |
| DeepACO accept（p_c=0，正式） | 1221.2 [1188.2, 1254.6] | −263.3 [−297.5, −229.5] | 1.0 | 0% | 1.04s | 官方空间最小档 T=30；CPU 低负载 |
| DeepACO accept（主档，正式） | −113.4 [−244.9, 15.5] | −422.0 [−472.4, −372.8] | 1.0 | 0% | 1.01s | 同上 |
| Omni-VRP accept（p_c=0，正式） | 1226.9 [1195.8, 1254.5] | −257.6 [−289.0, −226.4] | 1.0 | 0% | 4.12s | 官方 aug8 档；官方权重无 TW 档 → CVRP 零样本 |
| Omni-VRP accept（主档，正式） | −85.7 [−209.6, 33.8] | −394.3 [−440.1, −347.6] | 1.0 | 0% | 2.64s | 同上 |
| （LIH / SGBS / Sym-NCO / POMO 正式：xformal_cpu 队列运行中） | | | | | | |

**正式臂现状（11 方法中 7 个完成，全部 hfr=1.0、failures={}、fallback 0%、timeouts 0）**：
p_c=0 档排序 RouteFinder 1266.2 > MVMoE 1257.7 > AttentionModel 1236.2 > RRNCO 1235.4 >
Omni-VRP 1226.9 > DeepACO 1221.2 > CaDA 1157.0；主档排序 RouteFinder −26.1 >
MVMoE −38.6 > AttentionModel −72.9 > RRNCO −73.2 > Omni-VRP −85.7 > DeepACO −113.4 >
CaDA −190.3。**全部低于 cond_hist 基线**（vs cond 差 −218 ~ −499），与「路由强度换不来
决策收益」的诊断口径一致。配对身份限制见工作包 §6①（`formal_adjudication=False`）。

诊断→正式对照（RouteFinder p_c=0）：诊断（旧代码）util 1184.8、hfr 0.1（36/40 天
duplicate_service）→ 正式 util 1266.2、hfr 1.0、failures={}——duplicate_service 修复
（pool 减在途 committed_next）在 40 天尺度验证成立。其余方法正式臂按 GPU→CPU 队列推进。

