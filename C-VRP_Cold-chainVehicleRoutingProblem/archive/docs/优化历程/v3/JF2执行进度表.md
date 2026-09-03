# JF2 执行进度表（重构版）

> **依据**：[`科研方法创新主控文档.md`](./科研方法创新主控文档.md)（最高优先级）+
> [`工程实现与实验执行手册.md`](./工程实现与实验执行手册.md) +
> [`各阶段问题与解决预案.md`](./各阶段问题与解决预案.md)
> **更新日期**：2026-09-01
> **当前状态**：**Phase 5（HFR-M0）实现完成 + Gate A FAIL（F3）→ 方向定：Cost-Aware Joint Fleet–Route Preference Learning（冻结 structural imitation）**
> **图例**：✅ 完成 · 🔄 进行中 · ⏳ 未开始
>
> ⚠️ 旧 `JF2至最终阶段_方法优化唯一执行主文档_2026-08-31.md` 已过时，以本表 + 三份新主控文档为准。

---

## 0. 一句话状态 + 关键数字

**科学状态**：exact-vehicle CE 有害（Gate A FAIL）→ 2×2 oracle 证明 partition×route 交互 → HFR-M0 已实现但 **Gate A FAIL（F3）**：G 对 partition 有正价值（g_only 24.97 < min_travel 25.92），但 route 结构监督无 online utility（structural imitation ≠ online utility）。→ 方向定为 **Cost-Aware Joint Fleet–Route Preference Learning**（保留 G 作 coarse partition representation）。

| 项目 | 值 |
|---|---:|
| JF1-H（最强 heuristic joint） | **24.50 / 100%** |
| OR-joint（teacher / reference） | 22.79 / 100% |
| B0 fleet gap / sequencing gap | 17.9% / −1.25% |
| α=0 regression | JF2 == JF1-H，128 例 exact |
| JF1-H rollout states / OR-joint labels | 8165 / 56735 |
| 对称分歧（OR≠min-travel 里等价车 ID 歧义） | 37779（95.3%） |
| **真实非等价分歧** | **1866（4.7%，占全标签 3.3%）** |
| M0-v1 raw exact-id CE disagreement top-1 | 3.9%（≈随机） |
| M0-v2 canonicalized disagreement top-1 | 56.7% |
| Gate A（JF2-M0-v2 vs JF1-H，VAL128） | 25.91 vs 24.50（FAIL，110 恶化/18 改善） |
| Regret mass | 硬 infeasible 0.5%；median +0.13；Top20% = 52.8%（温和非重尾）；27.3% negative |
| **2×2 oracle interaction** | **−1.6151**（partition 与 route 强负交互） |
| **HFR-M0 Gate A（7 变体，VAL128，JF1-H=24.50）** | g_only **24.97** / min_travel 25.92 / a_only 26.63 / full 29.95 / group_only 30.19 / group_logit 30.07 —— **F3** |
| HFR-M0 G partition 价值 | g_only(24.97) < min_travel(25.92)，Δ−0.95（**正**） |
| HFR-M0 route 信号 | 全部 > 25.92（a_only/full/group_only/group_logit），**无 online utility** |

---

## 1. 阶段总览

| 阶段 | 内容 | 状态 |
|------|------|:---:|
| **Phase 0** | Protocol / Regression Closure（JF1-H faithful 24.5019/100%，sound candidate set，pair_feasible audit，no-silent-drop 原语） | ✅ |
| **Phase 1** | Fleet residual architecture（FleetAssignmentHead + `S=S_H+α·r_θ`，α=0 exact regression） | ✅ |
| **Phase 2** | Expert data（8165 states + 56735 OR-joint labels + candidate QC PASS） | ✅ |
| **Phase 3A** | Raw exact-ID M0（CE）→ disagreement top-1 3.9% ≈ random，**negative diagnostic** | ✅ |
| — | Symmetry Audit（95.3% 分歧 = 等价车辆 ID 歧义） | ✅ |
| **Phase 3B** | Symmetry-corrected M0（canonicalization）→ true-disagreement top-1 **56.7%** | ✅ |
| **Phase 3C** | **Downstream Utility Gate：`JF2 < 24.50` @ VAL128 + 2×2 oracle factorial** | ✅（FAIL → interaction −1.6151） |
| Phase 4 | Attribution（Real / Shuffle / Feature / OldEdge / Generic）—— **在 partition/route 新 formulation 下做** | ⏳ |
| Phase 5 | **Hierarchical Fleet–Route Masked Reconstruction**（pairwise grouping `G_ij` + route edge `A_ij` 联合学 + cross-level consistency；`VehicleEquivalenceClass` + `FleetSlot` + anonymous route slot） | ✅（实现完成，Gate A FAIL/F3） |
| Phase 6 | **Cost-Aware Joint Fleet–Route Preference Learning**（pairwise preference + joint action `(slot, insertion_pos)` + `Q_θ ≈ −ΔJ(a)`）—— **HFR-M0 F3 后的新主方向** | 🔄 |
| Phase 7 | Best-insertion / route-aware features（M1） | ⏳ |
| Phase 8 | JF3 non-separable joint search（真正资源耦合 beam） | ⏳ |
| 后续 | stochastic / cold-chain / generalization（沿用 v4.0） | ⏳ |

---

## 2. 当前 Phase 3C：VAL128 Gate A（唯一主 Gate）

```
VAL128：Completion=100%，TW/Cap/Return=100%，JF2_{M0-v2} < 24.5019
```

**同环境比较**（唯一变量 = assignment score）：JF1-H vs JF2-M0-v2，同 VAL128、同 strict_online_env、同 candidate builder、同 route sequencing、同 guard、同 decode seed、同 evaluator。

**输出**：complete / TW/Cap/Return / distance / instance paired delta / action divergence / event-level true-disagreement hit rate。

**Gate A 前禁止**（否则丢 M0-v2 因果归因）：
解冻 encoder · 上 preference · 改 partition 架构 · 加 insertion features · 上 JF3 · 加层/加 seed · 改 candidate semantics · 碰 test set。

---

## 2.5 关键结论：partition × route interaction（2×2 oracle factorial，2026-09-01）

四路 factorial（VAL32，同 skeleton，唯一变量 = partition 来源 × sequencing 来源）：

| | greedy sequencing | TSPTW sequencing |
|---|---|---|
| **JF1-H partition** | C_HG = 24.4133 | C_HT = 24.6369 |
| **OR partition** | C_OG = 24.5533 | C_OT = 23.1618 |

**一阶主效应**：
- partition（OR vs JF1-H，都 greedy）`Δ = C_OG − C_HG = +0.1400`（单独换 OR partition 反而更差）
- sequence（TSPTW vs greedy，JF1-H partition）`Δ = C_HT − C_HG = +0.2236`（单独换 TSPTW 也变差）
- sequence（TSPTW vs greedy，OR partition）`Δ = C_OT − C_OG = −1.3915`（在 OR partition 下 TSPTW 突然巨大收益）

**二阶交互项**（决定性）：
$$Δ_{\text{interaction}} = C_{OT} − C_{OG} − C_{HT} + C_{HG} = \boxed{−1.6151}$$

**instance-level interaction（导师 §11 严谨口径）**：`I_n = C_OT−C_OG−C_HT+C_HG` 逐实例 → **mean −1.79、median −2.06、95% bootstrap CI [−2.40, −1.18]（不跨 0）、P(I_n<0)=84.4%**。→ 负交互是**一致**的（84.4% 实例为负），不是 aggregate mean 假象，可写成强机制结论。

**方向定论（导师确认）**：
> **fleet recourse 的质量由 partition × route 的交互决定，不能由独立的 vehicle assignment 或独立的 sequencing 解释。** 因此：
> 1. exact-vehicle assignment → **正式 No-Go**（OR partition 配 greedy 反而更差）。
> 2. partition-only → **也不是最终方案**（partition 的价值依赖 route reconstruction）。
> 3. **主方向 = Hierarchical Fleet–Route Masked Reconstruction**：`P(O,A) = P(O)·P(A|O)`（factorized architecture ≠ factorized objective；不能独立优化两层，要 joint supervision 共同塑造梯度）。
> 4. 新表示 `X = (O, A)`，`O[s,j]` = customer→route slot，`A^(s)[i,j]` = slot 内 route edge；用 **pairwise grouping `G_ij = 1[i,j same route]`** 作 permutation-invariant 监督（天然吸收 vehicle symmetry）。

> ⚠️ **表述收紧**：这是「强烈表明存在显著 partition × sequencing interaction」，不是「已严格证明 non-separable」；「TSPTW 把车送到对未来 reveal 不利位置」是机制**假设**，四路实验只证明 trajectory 更差，未直接证明 anchor 机制。后续补 **instance-level interaction**（`I_n = C_OT−C_OG−C_HT+C_HG` 的 mean/median/95% CI + `P(I_n<0)`）才可写成强结论。

---

## 3. 关键新发现：vehicle symmetry → label noise

raw exact vehicle CE 失败（3.9%≈随机）的根因不是「模型学不会」，而是**监督目标违反车辆置换对称性**：

- idle@depot 同质车（同 location/time/load/capacity/status/route history）互相可置换，OR-joint 在它们之间**任意挑一个** ID → teacher label 是噪声。
- 95.3% 的「OR≠min-travel」分歧只是等价车 ID 歧义；真正有信息量的**非等价分歧只有 1866（3.3%）**。

**含义**：fleet allocation 的 17.9% gap 大头是 **PARTITION（哪几个客户进同一条 route）**，不是 exact vehicle identity——idle@depot 车之间选哪辆无所谓。

**canonicalization（min-ID）是临时修复，不是最终创新**：它人为偏好小 ID，破坏 permutation equivariance。下一版升级为 **set-valued / equivalence-class CE**：

$$L_{sym}(j)=-\log\frac{\sum_{k\in E_j}e^{S_{kj}}}{\sum_{k\in V_j^F}e^{S_{kj}}}$$

**必须新增的 blocking test**：vehicle permutation test（随机置换 vehicle ID → 模型输出只跟着 permutation 改编号，cost 不变）。

---

## 4. 方法定位升级（科研 framing）

从「MaskCO 后面加一个 MLP Fleet Head」升级为：

> **Permutation-Aware Fleet Ownership / Partition** —— 对等价 idle 车辆只学等价类/route partition，对非等价（ready/committed）车辆学 state-conditioned ownership。

长期统一表达（Hierarchical Fleet–Route Masked Reconstruction）：

$$Mask(O,A)\rightarrow Reconstruct(O,A\mid\mathcal F_e,FleetState_e)$$

其中 $O$ = fleet ownership/partition，$A$ = route adjacency。这条主线与 B0「fleet allocation 主导、sequencing 饱和」一致。

**三层贡献（重构）**：
1. **Causal Dynamic Masked Reconstruction**（future gating + visible-only normalization + visibility attention）
2. **Permutation-Aware Fleet-State Masked Recourse**（决策空间 edge→fleet partition + 显式处理 permutation symmetry）
3. **Feasibility-Preserving Joint Fleet Reconstruction**（frozen prefix + committed invariant + sound candidate + resource search + service-first guard）

---

## 5. 当前 P0–P3 优先级（72h，按顺序）

- **P0**：冻结 M0-v2 artifact（checkpoint + train_config + dataset manifest + symmetry_audit + hash）。
- **P1**：跑 VAL128 Gate A（JF1-H vs JF2-M0-v2）。
- **P2**：同步记录 **DecisionTrace**（jf1h_vehicle / jf2_vehicle / teacher_vehicle / is_symmetry / is_true_disagreement / residual / rank_flip / event_progress / n_pending / feasible_count）。
- **P3**：Gate A 后分叉（见 §6）。

**代码优化优先级**（不是更深 MLP / 更大 hidden / 更多 beam / 更多 seed）：
1. Gate A evaluation path → 2. DecisionTrace → 3. symmetry/permutation audit → 4. equivalence-class label abstraction（`VehicleLabelAdapter` / `VehicleEquivalenceClass`）→ 5. regret-mass analysis → 6. set-valued CE → 7. preference → 8. route-slot partition → 9. encoder fine-tune → 10. JF3。

---

## 6. Gate A 后分叉

| 结果 | 含义 | 下一步 |
|------|------|------|
| **PASS**（<24.50, 100% service） | learned fleet correction 有实际价值 | Gate B/C/D attribution → 再决定 encoder fine-tune |
| **FAIL but service PASS**（≈24.50） | classifier 学到 label 但没进 decision / 杠杆不足 | 查 residual calibration + rank-flip rate → regret-mass diagnosis |
| **FAIL cost 变差**（>24.50, service 100%） | 正确 decision 收益小、错误 override regret 大 | regret-weighted preference（Phase 6） |
| **service < 100%** | 立即停模型优化 | 查 candidate semantics / hard mask / unresolved / guard / sequencing |

**成功分级**（M0 只求 proof-of-usefulness，不要求一次逼近 22.79）：
Gate A `<24.50` · minimum `<24.25` · strong `≤24.0` · very strong `≤23.75` · near-OR `23.2~23.5`。

---

## 7. 历史错误清单（实现时逐条对照）

**旧清单（继承）**：未来坐标参与 normalization · future edge 无 visible mask · idle 从 t=0 dispatch · service time 重复加 · committed demand 未计入 anchor load · reveal 后 stale-tail duplicate · mutable 全锁死/全释放 · `if no candidate: continue` silent drop · cost 低但 completion 不完整仍排名 · solver 自己 feasible flag 代替 evaluator · candidate set 被 learned score 改变 · Real/Shuffle 用不同随机流程 · 旧 edge logits 直接当 fleet score · α≠0 时 Gate-0 失去意义 · test 反复调参 · instance 当独立 replicate。

**新增（symmetry 相关）**：
- 把 exact vehicle ID 当 one-hot 真值监督（违反 permutation symmetry → label noise）。
- 长期依赖 min-ID canonicalization（人为偏好小 ID，破坏 permutation equivariance）。
- 用 disagreement count 而非 **regret mass** 判断信号量（3.3% 决策可承担大部分 regret）。
- 把 56.7% 写成「方法有效」的最终结论（它只证明「学到了」，未证明「cost 改善」）。
- set-valued CE 的 `target_equiv_mask` 与 candidate_mask 无交集时 silent ignore（应 fail-hard）。
- 在 symmetry 未解决前用 focal loss（会把等价 label 歧义当难样本放大）。

---

## 8. 结果表必须增加的列（Phase 3C 起）

`Method / LabelMode / EncoderMode / LossMode / Cost / Completion / TrueDisagreeTop1 / EquivClassAcc / RankFlip / TrueDisagreeOverride / p95 latency`

其中 **EquivClassAcc**（预测车与 teacher 同等价类即对）比 raw top-1 更适合 symmetry-aware 方法；主科学指标优先看 true-disagreement accuracy + EquivClassAcc，不只看 raw top-1。

---

## 9. 实验命名规范

```
JF2-M0-v1-rawid     （raw exact vehicle CE，3.9% null）
JF2-M0-v2-canon     （canonicalization，56.7%，当前 Gate A 用）
JF2-M0-v3-setce     （set-valued / equivalence-class CE）
JF2-M0-v4-regret    （regret-weighted preference）
JF2-M1-insertion    （route-aware best-insertion features）
JF2-PART-slot       （route-slot partition 架构）
```
