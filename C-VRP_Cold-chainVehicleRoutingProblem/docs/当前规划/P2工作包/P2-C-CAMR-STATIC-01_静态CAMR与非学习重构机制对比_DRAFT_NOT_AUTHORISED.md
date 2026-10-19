# P2-C-CAMR-STATIC-01：静态 CAMR 与非学习重构的机制对比

**状态：`DRAFT_NOT_AUTHORISED`（R2.1）。** 本文件只是可授权的运行前草案，**没有任何执行、没有 `registration.json`、没有枚举、没有模型加载**。
启动它需要单独的书面授权；且在授权之后、第一次评分调用之前，仍须先冻结 §9 的常量表。未经授权时，本文件不得被引用为"已批准的工作"。

- 上位裁定：2026-10-19 第二轮复核 → 第三轮授权前复核 → 第四轮 R2 复核。R1（`e638f320`）→ R2（`bd759020`）→ 本 R2.1，修订记录见 §附录 A。
- 前置包：`P2-C-ANCHOR-01`（`a2d74c955`）。其结论已按 [ERRATA_01.json](../../../research/p2_c_anchor_01_20261019/ERRATA_01.json) 收紧为**结构可行性已建立、执行事务资格尚未建立**。
- 本轮**不做**：改引擎、真实续程、标签采集、价值头训练、GPU、正式 L2/L3、触碰最终种子、更换 SAA、新增算法模块。

## 0. 唯一科学假设（本轮只回答这一条）

> **H-CAMR-STATIC-01**：在相同 strict-online 公开状态、冻结锚点、**固定双订单 mask**、**共享合法动作空间**与**受控计算预算**下，冻结 MaskCO 的重构评分是否会产生**不同于 distance-greedy 的合法完整计划**？

**对应层级：L1（机制证据）。** 本轮**不**检验质量优劣，**不**检验在线效用，**不**检验可安装性。

### 0.1 正式命名（R2.1 修订）

| 比较 | 研究职责 | 层级 |
|---|---|---|
| **`MaskCO-greedy` vs `distance-greedy`** | **唯一主比较**：学习评分是否改变合法重构选择 | **L1-A** |
| `MaskCO-SGBS` vs `distance-SGBS` | **次级搜索分析**：同搜索配置下的候选覆盖关系 | **L1-B**（不参与主判据） |

R2 把主对照写成 "distance-SGBS"，但 `sgbs_search()` 的**第一个**补全就是任何 beam 分支之前记录的无分支贪心补全（`run_sgbs_fixed_state.py:110` `record(greedy_complete(partial, list(mask_set)))`）。因此"主比较"实质上是 **greedy vs greedy**，R2.1 更正命名，**不改设计**：固定搜索规则、只替换动作评分，反而更利于因果归因。**不为保留 "SGBS" 之名而把多补全择优放回主实验。**

### 0.2 本轮明确不回答（各自另立命题）

| 不回答的问题 | 为什么不在本轮 | 何时才谈 |
|---|---|---|
| 学习到的 **mask 选择**是否优于非学习 mask 规则 | 需要已训练的 mask-value 头；当前头是**零初始化**（恒 KEEP），用它选 mask 等于把未训练策略当证据 | 另立 mask-selection 命题，且必须先有可用的训练信号来源 |
| MaskCO 重构是否**质量更优** | 本轮只比"是否产生不同的合法计划"，不产生任何 J / 距离 / 能耗比较 | L2，且必须先解决 §8 阻断 |
| 重构结果能否**可靠传播到在线执行** | 执行事务资格未建立 | L3，且必须先有隔离事务实现 |

## 1. 分析单元与冻结身份键（R2.1 修订）

| 项 | 值 | 依据 |
|---|---|---|
| 相位 | **15**（SAA 9 / MYOPIC 6） | `anchor_scope.csv` 判决 `ANCHOR_LEGAL_MULTI` |
| 队列长度分布 | `queue_len = 2`：13 个；`queue_len = 3`：2 个 | 同上 |
| 每相位双订单 mask 数 | 13 个相位各 **1** 个；2 个相位各 **3** 个 | `n_masks = C(queue_len, 2)` |
| **相位–mask 比较单元** | **19** = 13×1 + 2×3 | 本轮全部主结论的分母 |

**身份键（R2.1 修正，替代 R2 的 `(day, ordinal, mask)`）：**

```
unit_id = (arm, day_id, ordinal, public_identity, mask)
  arm             ∈ {'SAA', 'MYOPIC'}          # 控制器标识，必填
  day_id          = 开发者实例编号（0/1）
  ordinal         = 该日相位序号
  public_identity = 原始相位记录中的 64-hex 公开状态哈希（不是时钟、不是锚点）
  mask            = tuple(sorted(...)) 的双订单掩码
```

- R2 的键不足以区分两个控制器产生的相位。`arm` 必须显式入键。
- **已核验**（本轮只读计算）：该键在 15 个相位上产生 **19 个全不相同的 unit**；15 个相位的 `public_identity` **互不相同**（0 个重复）。
- **同钟同锚点 ≠ 同一实例**：SAA `d1 ord6` 与 MYOPIC `d1 ord7` 的 clock（1.6605）与锚点订单（21）相同、锚点后首动作集合也相同，但 `public_identity` 不同、锚点 finish 不同（3.807 vs 6.222）⇒ **不是同一个物理决策实例**，不得合并、不得互相替代。
- 15 个相位来自 **2 个**先前已检查的开发者实例。**19 个单元不是 19 个独立样本**，一律按"相位–mask 暴露量"报告，不做独立同分布假设、不算显著性检验。
- mask 生成规则固定为 `tuple(sorted(p) for p in combinations(queue, 2))`，与 ANCHOR-01 探针逐字一致；`registration.json` 必须显式存下这 19 个 `unit_id` 及其 sha256，运行不得增删。

## 2. 冻结输入与共享接口（公平性的硬要求）

四个"相同"缺一不可：**相同公开状态、相同冻结锚点、相同合法动作集合、相同计算预算**。任何一项不成立，该单元记为无效（`PAIR_INVALID_*`，见 §2.4），**保留在 19 个单元的分母内**。

### 2.1 共享合法动作接口（唯一一套，双侧同用）

代码里两侧**目前用的是两套不同谓词**，必须在运行前统一为一套薄封装（**不新增算法、不新增评分**，只把既有谓词按同一顺序组合一次）：

| 现有路径 | 谓词 | 问题 |
|---|---|---|
| `p2_pos01_reference.legal_actions` / `probe_anchor.legal_actions_with` | `cand.feasible` **且** 全车队 `route_check` | ANCHOR-01 的口径 |
| `mpre_policy.enumerate_legal_actions` | **只有** `cand.feasible`（**无** `route_check`） | `run_sgbs_fixed_state.py` 用它 |

容量口径也不同：`certify_route` 用 `load > capacity + 1e-6` 拒绝（`action_contract.py:190`），`route_check` 用 `load <= capacity - 1e-4` 接受（`p2_pos01_reference.py:43,226-227`）。**同一包内两套容量容差**，若两侧各用一套，"差异"无法归因。

**规定**：本轮唯一合法动作集合 =

```
legal_actions_shared(view, plans, customer, allowed)
  = { (c, action) | cand.feasible                       # certify_route 的证书
                    AND 对 trial 全车队逐车 route_check  # capacity - 1e-4 口径
                    AND allowed 显式给出（含在途车） }
```

- `allowed` 必须**两侧同一个对象**：`mutable_ids(view) ∪ {committed_vid}`。
- **不得**直接复用 `enumerate_legal_actions(..., mutable_ids)`（它把在途车排除在外，且不带 `route_check`）；只允许在共享封装内部调用其下层的同一个 `enumerate_actions_from_plans`。
- 容量口径固定为**更严的** `capacity - 1e-4`（15 个相位本身就在该口径下筛出）。

### 2.2 锚点冻结

锚点 = `(committed_next, committed_arrive, committed_finish)` 与取货后载重，**逐字不变**；`committed_next` 不得出现在任何后缀里。
（代码事实：`apply_action` 在每个分支逐字复制 `anchor_node/time/load`，所以"锚点未变"对该枚举器恒真——它只能当一致性检查，不能当"冻结无代价"的证据，见 ERRATA_01 E2。）

### 2.3 确定性 tie-break（R2.1 新增）

**两侧的动作排序必须都是 `(score 降序, action_id 升序)`**，`action_id` 为稳定全序。

- 代码事实：`_distance_propose` 已经是 `legal.sort(key=lambda x: (x[2].incremental_distance, x[1].action_id()))`，**有** tie-break；而 `_model_propose` 用 `np.argsort(-scores)`（`run_sgbs_fixed_state.py:54`），**没有** `action_id` tie-break。两侧不对称 ⇒ 分数并列时可能各自产生不同计划，从而**伪造**出"模型差异"。
- 因此在 §9 的退化自检（把模型排序换成 `-incremental_distance`）之前，必须先给模型侧补上同一 tie-break；否则该自检不成立。

### 2.4 预算与失败账（R2.1 修订）

`sgbs_search` **没有墙钟截止**，预算全部是调用计数（`beam` / `max_per_parent` / `max_completions`）；`wallclock_total_s` 只是事后**测量**（`run_sgbs_fixed_state.py:288`），不是上限。因此预算分两本账：

| 账本 | 内容 |
|---|---|
| **Pass A（主比较，greedy only）** | 每侧 1 个完整补全；记录**动作评分调用次数**、**完整补全次数**、**实际墙钟**、**超时状态** |
| **Pass B（次级，SGBS both sides）** | `beam=2`、`max_per_parent=2`、`max_completions=8`；记录评分调用次数、候选补全次数、失败次数（`n_fail`）、实际墙钟、超时状态 |

- `beam` 配置**不是**主比较的预算，不得用来解释或辩护 Pass A 的差异。
- 墙钟/内存/单元超时上限对**两侧同一数值**；**不得只作废一侧**。
- **失败账纪律**：若某单元超时，**保留其在 19 个预定单元中的身份**，记为 `PAIR_INVALID_TIMEOUT`，**不得从分母中静默删除**；报告必须同时给出有效单元数与无效单元清单。任何其它作废原因同样记 `PAIR_INVALID_<reason>` 并保留槽位。
- 事后调参禁令：`beam` / `max_per_parent` / `max_completions` / 墙钟 / 内存 / 单元超时一经冻结，出现不利结果也不得修改。

## 3. 模型身份（MaskCO 侧的唯一评分入口）

R1 写的"冻结 CVRP decoder + 适配层 + 定向 insertion 头"**作废**——本线不存在训练过的 insertion 头。以源码为准：

- **唯一评分入口**：`scripts/models/p2_pos01_mask_value.py` 的 **`FrozenBackbone.insertion_scores(view, base, partial, actions)`**（`:111-117`）。返回 `edge_insertion_scores(L, cust, pred, succ)`，`L = model.feature2logit(Z)`；**没有任何额外可训练参数**，也不含 mask-value 头。
- 文件头权威声明："Frozen real M-pre features and the **ONLY trainable CA-UGMR mask-value head**. CPU only. **No decoder fine-tuning, insertion head**, …"；`MaskValueModel.head.l3` 被显式置零（`:141-142`）⇒ **零初始化 mask-value 头恒为 KEEP，本轮不得用它产生或挑选 mask**。
- checkpoint 身份已在本轮**实测核验**（只读哈希）：
  - `MASKCO_code/ckpts/cvrp100.ckpt`，305,061,150 bytes
  - `sha256 = 03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db`，与 `EXPECTED_CHECKPOINT`（`:32`）**逐字节一致**
  - `FrozenBackbone` 另断言 `encoder_input_dim == 3`、`(d_encoder, d_decoder) == (512, 512)`，并记录 `parameter_sha256`
- 运行时另须固定：`JAX_PLATFORMS=cpu`、`PUBLIC_NODE_BUCKET=51`、`coord_normalize_visible`（严禁 `coord_normalize`）。
- `registration.json` 必须写入 `checkpoint_sha256` + `parameter_sha256` + 上述常量；运行后逐字比对，不一致即整包作废。

## 4. 输出契约（对称化，不得混用两种对象）

**Pass A 主对象 = 单个贪心决策输出（每单元每侧 1 个计划）。**

- `distance-greedy` = `_distance_propose` 排序下的 `greedy_complete(partial, mask_set)`；它是 `sgbs_search` 记录的**第一个**补全，与 beam 宽度无关、可复现、**不是事后择优**。
- `MaskCO-greedy` = `insertion_scores` 在同一共享动作集合上逐层取 top-1 的确定性补全，与上一行同构。
- 两侧补全都必须先通过 §2.1 共享接口与 §2.2 锚点检查，否则记为**该侧无输出**（不是"无差异"）。
- **严禁**用 `min(J_vis)` 式择优替代贪心输出。

**Pass B 次对象 = 完整合法补全集合（单独报告，不参与主判据）。**

- 分别报 `MaskCO-SGBS` 与 `distance-SGBS` 的 `n_unique` / `n_total` / `n_fail`、集合大小、以及**集合归属**（MaskCO-greedy 输出是否落在 distance-SGBS 集合内）。
- 集合层**不得改写主判据**：Pass A 输出相同即"无差异"，集合层面的差异只能作描述性补充。
- 旧 `_eval_best` 的接受规则**禁止带入**：它按 `min(J_vis)` 选优、无认证补全时静默返回 `J0`（`run_sgbs_fixed_state.py:133-148`），把"没产出"与"产出但没更好"混为一谈——而区分这两者正是本轮全部意义。

## 5. 主观察量（三个，分开报）

对 19 个单元逐单元登记：

1. **合法性**：`MaskCO-greedy` 输出是否保持锚点、并通过 §2.1 共享参考检查（`plan_hash` 级）。
2. **集合归属**：该输出是否属于 `distance-SGBS` 的合法补全集合（Pass B，`plan_hash` 比较）。
3. **计划身份与锚点后首动作**：两侧 `plan_hash` 是否相同；若不同，`suffix[0]`（0 = 返仓）是否相同。

判据层级：**不同且都合法**才是阳性机制证据。**锚点后首动作相同而更深后缀不同**，必须如实登记为"差异存在但不在紧邻动作上"，不得因"首动作相同"就判无差异（ANCHOR-01 口径）。

## 6. 三层判定（R2.1：L1 拆成两条独立判据，消除 R2 的 §4/§6 冲突）

### 第一层 L1-A：学习评分是否产生可复现的合法行为差异

只比较两侧 **greedy** 输出（Pass A）。**Go** 需同时满足：

- 至少一个单元 `MaskCO-greedy` 的 `plan_hash` **≠** `distance-greedy`；
- 两侧输出**都**合法（保持锚点 + 通过共享参考检查），差异不是靠违规换来的；
- 两侧在**同一** `allowed` 集合、**同一** tie-break、**同一**侧预算、**同一**共享接口下产生；
- **可复现**：整包连跑两次，19 个单元的 `plan_hash` 逐字相同。

**L1-A No-Go** 的正当理由**只有**：

1. 19 个单元两侧 `plan_hash` **全部相同**；
2. 差异**不可复现**；3. 差异**不合法**；4. 差异**可完全由预算/动作集合/tie-break 不对称解释**（消除不对称后消失）。

> R2 曾把"差异可由非学习 proposer 直接复现"列为 L1-A 的 No-Go 理由之一，与"集合层不得改写主判据"冲突。**R2.1 撤销该条，移入 L1-B**：`distance-SGBS` 覆盖了 `MaskCO-greedy` 的输出，只能说明"非学习搜索也能到达该计划"，**不能撤销已经观察到的 L1-A 行为事实**，也不能把两件事合并成一次 L1 失败。

### 第二层 L1-B：强非学习搜索是否覆盖这些差异

用 Pass B 的 `distance-SGBS` 集合检查 Pass A 的 `MaskCO-greedy` 输出：

| 结果 | 结论 | 允许说 / 不允许说 |
|---|---|---|
| **全部被覆盖** | 学习评分的**独立搜索价值仍缺乏支持** | 允许说"评分改变了贪心选择"（L1-A 事实保留）；**不允许**主张"学习重构具有不可替代价值"，**不允许**据此扩张训练或事务工程 |
| **存在未被覆盖的合法计划** | 仅说明**值得继续检查** | **不允许**宣称质量优势；只允许说"存在非学习 SGBS 在同等预算下未到达的合法计划" |

### 第三层：是否有可认证的质量收益

**必须另立实验**，且先解决 §8 的 C0 预算与冷链物理检查资格。本次静态 L1 不回答，也不自动授权。

## 7. 止损线

- 仅仅发现两侧产生**不同路线**，**不足以**授权训练价值头，**也不足以**授权开发完整事务引擎。
- 后续仍须证明这些差异具有**可信、可认证的质量价值**（能过物理认证，且不是靠违规或预算不对称换来）。
- 若 **L1-A No-Go**：CAMR 静态重构方向 **NO-GO**，停止训练路线，不再投入持久化工程、未来标签生成与价值头训练；结论写成负结果，不改写成"方法还需调参"。
- 若 **L1-A Go 而 L1-B 全部覆盖**：不扩张；把结论记为"评分改变了贪心选择，但非学习搜索可覆盖"，并据此关闭"不可替代价值"主张。

## 8. 三个独立阻断（不合并，且**绕开 ≠ 解决**）

| 阻断 | 阻止当前静态 L1 对比？ | 何时必须解决 |
|---|---|---|
| C0 正式预算 `B` 未裁决 | **不阻止**（L1 只做路线级机制诊断） | 完整冷链可行性认证、任何 L2 之前 |
| HIST 未来支持域不足 | **不阻止**（本轮不使用未来信息的静态对比） | 真实反事实续程标签之前 |
| 执行事务资格未建立 | **不阻止**（本轮只比较静态计划） | 任何真实在线安装、L3 之前 |

配套代码事实：`certify_route` 无条件返回 `projected_energy_kwh / projected_quality_loss / projected_thermal_margin = None`（`action_contract.py:171-173`）⇒ 能耗与品质在本管线里是"跑不了"，不是"没跑"。任何物理认证主张都必须先新增这条投影路径；`B_ref` 不能自动替代正式批准的硬预算。

**明确声明**：静态工作能绕开这三个阻断，**不等于**它们已被解决；三者继续各自登记为 open。

## 9. 运行前必做（顺序不可交换）

1. **冻结 `registration.json`**：唯一假设与 L1-A/L1-B 拆分、19 个 `unit_id`（含 `arm` 与 `public_identity`）及 sha256、共享接口定义、容量口径、锚点断言、**tie-break 规则**、Pass A / Pass B 各自预算常量、**墙钟 / 内存 / 单元超时上限**、`PAIR_INVALID_*` 记账规则、主/次观察量、三层判定、止损线、禁止项、模型身份三元组。
2. 逐一 sha256 记录：`p2_pos01_mask_value.py`、`action_contract.py`、`p2_pos01_reference.py`、`p2_pos01c_gap.py`、`mpre_policy.py`、`strict_online_env.py`、`run_sgbs_fixed_state.py`、两个相位文件。
3. **零假设自检（三条，全过才可正式跑）**
   - 主对照确定性：`distance-greedy` 在同一单元重复运行，`plan_hash` 逐字相同；
   - MaskCO 侧确定性：`MaskCO-greedy` 在同一单元重复运行，`plan_hash` 逐字相同；
   - **退化闸门**：把模型侧排序换成 `-incremental_distance`（并使用 §2.3 的统一 tie-break）后，结果必须**逐字退化**为 `distance-greedy` 的 `plan_hash`。不退化 ⇒ 两侧不是同一条搜索路径，先修接口，**不得**进入正式对比。
4. 再跑 Pass A → Pass B；整包连跑两次做可复现性核对。

**待冻结常量（不在本草案内定值，必须在授权后、首次评分调用前写入 `registration.json`）**：每单元墙钟上限、整包墙钟上限、内存上限、每单元评分调用上限、单元超时判定方式、`PUBLIC_NODE_BUCKET`、JAX 平台变量、19 个 `unit_id` 清单哈希、模型身份三元组。这些数值**不得留到执行过程中自由决定**，且报告必须原样回显。

## 10. 禁止项

修改执行引擎；修改控制器或阈值；重跑源日；采集标签；训练；GPU；触碰最终种子。
声称在线效用；声称 CAMR 新颖性；把 `candidate` 说成"部署"。
在 15 个相位之外挑选相位；出现不利结果后更换 mask 规则、容量口径、共享接口、tie-break、预算或超时上限；把 `completions[0]` 与 SGBS 补全集合混为同一种决策输出；用集合覆盖改写或撤销 L1-A 结论；用 `min(J_vis)` 事后择优替代贪心输出；把超时单元从 19 个分母中静默删除；把 `beam` 预算当作主比较预算的辩解。

## 11. 授权状态

**`DRAFT_NOT_AUTHORISED`（R2.1）。** 本文件可被审阅、可被要求再次修订；在书面授权之前，不得创建输出目录、不得冻结 `registration.json`、不得运行任何枚举或模型推理。本轮亦未新增任何算法模块、未启动任何实验。

## 附录 A：修订记录

### R1 → R2（`e638f320` → `bd759020`）

| # | 复核意见 | R2 的处理 |
|---|---|---|
| 1 | 科学问题混杂 | 唯一假设收窄为固定 mask 下的 repair 差异；mask 选择移出本轮 |
| 2 | 主对照输出不对称 | 主对象 = 单决策输出，SGBS 单输出预冻结为 `completions[0]`；集合覆盖为次对象 |
| 3 | Go/No-Go 过严 | 撤销"差异仅来自单一通道即 No-Go"；改为五条正当理由 |
| 4 | MaskCO 身份歧义 | 指定 `FrozenBackbone.insertion_scores()`；声明无训练过 insertion 头；checkpoint 实测核验 |
| 5 | 历史 SGBS 不能直接当已合格对照 | 要求共享合法动作接口、计数预算、禁用旧 `_eval_best` |
| 6 | 分析单元与样本性质 | 明确 19 个单元、非独立样本 |
| 7 | 阻断分阶段 | 改为三行表并加"绕开 ≠ 解决" |
| 8 | 新增止损线 | 写入停止条件 |

### R2 → R2.1（`bd759020` → 本次）

| # | 复核意见 | 问题 | R2.1 的处理 |
|---|---|---|---|
| 1 | 主对照实际是 distance-greedy | R2 把主对照命名为 "distance-SGBS"，与 `sgbs_search` 第一步记录贪心补全的事实不符 | §0.1 正式命名 `MaskCO-greedy vs distance-greedy` 为唯一主比较；`*-SGBS` 降为次级搜索分析；**设计不变** |
| 2 | 第五条 No-Go 与 §4 冲突 | "非学习 proposer 可复现即 No-Go" 会用一个集合层事实撤销主判据 | §6 拆分 L1-A（评分行为差异）与 L1-B（SGBS 覆盖）为两条独立判据；该条从 L1-A 的 No-Go 移入 L1-B 的结论表 |
| 3 | 身份键缺控制器标识 | `(day, ordinal, mask)` 无法区分 SAA / MYOPIC | §1 改为 `(arm, day_id, ordinal, public_identity, mask)`；已核验产生 19 个互不相同的 unit、15 个相位 `public_identity` 互不相同；并写明"同钟同锚点 ≠ 同一实例" |
| 4 | 预算不能只写 beam | 主比较是单次贪心，`beam` 不应充当主比较预算 | §2.4 拆 Pass A / Pass B 两本账，分别记录评分调用、补全次数、失败数、墙钟、超时 |
| 5 | 超时不得静默出分母 | R2 未规定失败记账 | §2.4 引入 `PAIR_INVALID_TIMEOUT` 等，保留槽位、报告有效单元数与无效清单 |
| 6 | 退化自检需统一平分规则 | 两侧 tie-break 不对称（`_model_propose` 无 `action_id`） | §2.3 规定两侧统一为 `(score 降序, action_id 升序)`；§9 要求先补齐再跑退化闸门 |
| 7 | 资源与身份须启动前冻结 | R2 把这些留给执行阶段 | §9 新增"待冻结常量"清单与"不得执行中自由决定"的要求 |
