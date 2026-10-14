# 取货主门：`cond_hist` 前置缺陷的处置与协议修订 v2（2026-10-10，交接续办）

> 机器可读版本：同目录 [`P_LIVE_DEV_PANEL_V2_REGISTRATION.json`](P_LIVE_DEV_PANEL_V2_REGISTRATION.json)。
> 本文件为**协议修订记录**，不含任何新测量、不含任何正向结论。

## 1. 决定了什么

用户裁定采用 `COND_HIST_PREMISE_DEFECT.md` 的两个诚实选项中的 **(a)**：

> **显式采用已记录的等价性，把主门重新表述为针对真正有区别的臂，并把该重述登记为协议修订。**

- **(b)（放宽能耗预算 B 让条件性产生差异）已被测量否定**，不是判断否定：在 B×3 下两 SAA 臂**仍逐位相同**（见 §2 证据 E3）。
- **(C)（换向）** 本次不采用；它仍是可用的未来假设，但需另行登记。

## 2. 证据（四条，均已独立复核）

| ID | 事实 | 工件 | 关键值 |
|---|---|---|---|
| **E1** | `cond_hist` 与 `uncond_hist` 在登记 A-v1 运行点逐位相同 | `research/pickup_20round_audit/pickup_wiring_smoke_n5h60_v1/result.json` | 两者均 **−376.40726526174694**，58 单 / 142 拒；`strong_rolling` −361.1270；`uncond_dynamic` +552.8176 |
| **E2** | 源码早已把该等价性记为实测结论 | `scripts/evaluation/scenario_saa.py:1012` | 逐字：「…使投票退化为「p_c>0 全接 / p_c=0 全拒」的空转（**40 天实测 cond≡uncond≡myopic**）」 |
| **E3** | 加宽预算改变区制但**仍不分离两臂** ⇒ (b) 否定 | `research/pickup_20round_audit/pickup_wiring_smoke_B3_v1/result.json` | B=710.6548152249378 → 两臂 −376.4073；**B×3=2131.9644** → 两臂 **4480.7185**；`strong_rolling` 4541.5080（199/200 单）；`uncond_dynamic` 4390.5643 |
| **E4** | 「场景为空」候选解释已被否定 | `COND_HIST_PREMISE_DEFECT.md` | 8/8 天非空，clock 12.0 时仍有 441 单 |

E1/E3 的扫描维度：历史 **1 / 8 / 60** 天 × 时限 **2s / 10s** × `n_neighbors` **10 / 5**。

> ⚠️ E1/E3 都是 **1 实例接线烟测**，**不得**当作性能结果引用。它们只证明"对照可构建、且两臂不分离"。

## 3. 修订后的主门（v2）

| 角色 | 对照 | 判据 |
|---|---|---|
| **主确认** | `maskco − strong_rolling` | 均值 **≥ 20 utility/day** 且**配对 CI 下界 > 0** |
| **归因** | `maskco − recon_only` | 均值 > 0 且配对 CI 下界 > 0（同初值、同解码调用、丢弃 logits 的纯重构对照） |
| **次要报告** | `maskco − discard_control` | 仅报告，用于分离信息价值与算力消耗 |
| **降级为报告臂** | `maskco − cond_hist` | **不再作为门**；保留在每张结果表中，作为"该配置下不具判别力"的长期记录 |

**阈值 20 不变、方向不变（新方法必须更好）。**

**为什么这不是降低门槛**：在两个已测运行点上 `strong_rolling` 都**不弱于** `cond_hist`（B 下 −361.13 vs −376.41；B×3 下 4541.51 vs 4480.72），因此换上更强的对照只会**更难**。且 PROGRAM 本就要求"先与合格同合同强滚动控制器比较"。

**诚实边界**：20 utility/day 这个阈值原本是对 `cond_hist` 对照校准的；把它**移用到更强的对照**是**项目决定**，不是文献推出的常数，本文件如实记录，不主张它对新对照已校准。

**禁止跨运行点比较**：已封存的 I3 四臂 40 日裁决（`cond_hist` 均值 600.997）属于**不同流水线、不同运行点**，不得与本文的烟测/面板数字相减或并表。

## 4. 方法学依据（为什么必须重述，而不是"就这样报"）

i-DMVRP（整合需求管理与车辆路径）的主流解法是「近似机会成本 (OC) → 据此接单/拒单」，而 OC 可分解为**边际服务成本 MCTS** 与**位移成本 DPC**；该文献的关键论断是：**OC 近似误差只有在"改变了决策"且"发生在实际会访问到的状态"时才会影响解质量**。本题 `cond ≡ uncond ≡ myopic` 正意味着该近似**不改变决策** ⇒ 以它为对照的判据在结构上无检验力。文献中 anticipatory 方法的标准标尺是 **myopic / rolling-horizon**，而不是 conditional vs unconditional 采样器。

- Fleckenstein, Klein, Klein & Steinhardt, *From approximation error to optimality gap* — <https://ar5iv.labs.arxiv.org/html/2412.13851>
- Bent & Van Hentenryck, *Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers*, Operations Research 2004 — <https://pubsonline.informs.org/doi/10.1287/opre.1040.0124>；IJCAI 2003 — <https://www.ijcai.org/Proceedings/03/Papers/199.pdf>
- 动态 VRP 的 event-driven + rolling-horizon 基准实践 — <https://papers.cool/arxiv/2608.13799>

**边界**：文献只论证"对照必须重建"，**不**提供 20 utility/day 这个阈值。

## 5. 本次修订**没有**做的事

1. **未改动 v1 注册的任何字节**（v1 SHA `f9d9396ca8144e3c67d6099ea919d4182b589b8bc750c219d02d7ac7a2695d4f` 仍有效）；本文件为**追加式**修订并 pin 住 v1。
2. **未改动已封存的 I3 裁决、协议、权重、结果**。
3. **未产生任何新测量**：面板仍为 `amended_not_implemented_not_run`。
4. **未解除**四项禁止声称：稳定在线增益、MaskCO 归因、取货不可能性、等价性。CI 跨零仍只表示"未建立差异"。
5. **未删除任何臂**：`cond_hist` 从门降级为报告臂，逐字保留在面板中。

## 6. 下一屏障（执行前必须满足）

面板仍受 v1 的 `qualification_required_before_the_panel_runs` 约束，v1 的 `required_artifacts` 不变。本轮已就地解决两项、并发现一项**必须在数据前修掉的种子完整性问题**：

| # | 事项 | 状态 |
|---|---|---|
| 1 | pin 唯一的能耗预算 B | ✅ **已 pin：`710.6548152249378`**。权威来源：`configs/i3_20261008_v1/protocol.json`（`"budget": 710.6548152249378`）与 `configs/i3_20261008_v2/protocol.json`；亦是 A-v1 SAA 复算器构造值与精确枚举见证所用值。`709.1017637699684` **仅**是 P-BUDGET-v1 结果里的**计算量**，不是登记预算；`2131.9644456748134`（B×3）是仅用于检验选项 (b) 的诊断运行点 |
| 2 | 固定实例数与新种子 | ⚠️ **发现缺陷并已修正**：v1 登记的 `mask_seed_dev=2026120202` 与 `bootstrap_seed_dev=2026120212` 在 v1 登记时确为全新，但**其后被 n8–n14 的开发 pilot 消耗**（它们是 `pickup_maskco_arm.py --qual-seed`、`pickup_arm_pilot.py --pilot-seed/--bootstrap-seed` 的默认值，并出现在 `pickup_arm_pilot_v1/_n45_v1/_mf{50,70,80,90,100}_v1`、`pickup_arm_qualify_v1`、`pickup_headroom_*`、`pickup_cust_headroom_v1`、`pickup_learnability_v1`）。沿用将违反 v1 自己的"不得复用任何已观测种子"规则 ⇒ **v2 改为：`mask_seed_dev=2026120231`、`bootstrap_seed_dev=2026120232`**；`dev_dataset_seed=2026120201` 与 `arm_order_seed_dev=2026120211` 经全库检索确认仍全新，保持不变。备用全新种子：`2026120213 / 2026120221 / 2026120233` |
| 3 | 补完 **whole-fleet full-J replay** | ✅ **已完成**（`research/pickup_20round_audit/live_qualify_fleet_v3/result.json`，协议 P-LIVE-QUALIFY-FLEET-v2）：在**派遣宽度 3–4 / 15**（此前 1）的 12 个捕获上，整日 full A-v1 J、accepted 集合与 served 数**精确一致 12/12**，重放确定性 12/12，选点可复现 12/12。附带发现：能量账本通道**活跃但仅 5/12 捕获决策相关**（+50 与 +10000 kWh 击中同一组捕获），4/7 惰性捕获的剩余决策点数恰为 0。详见 `PICKUP_FLEET_FULLJ_QUALIFICATION.md` |
| 4 | 断言各臂**实际解码调用次数相等** | ⬜ 执行时从运行自身计数器断言，不得假设 |

**关于第 2 项的规则化教训**：v1 的"种子全新"是在**登记时**成立的陈述，但**种子会在登记之后被消耗**。因此"全新"必须在**数据生成前**重新检索确认；本次修订的 JSON 内含该规则（`seed_integrity.inference_rule`），后续运行注册必须重复此检查。

