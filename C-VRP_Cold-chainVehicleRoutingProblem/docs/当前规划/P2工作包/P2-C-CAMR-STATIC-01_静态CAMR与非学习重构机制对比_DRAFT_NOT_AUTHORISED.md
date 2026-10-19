# P2-C-CAMR-STATIC-01：静态 CAMR 与非学习重构的机制对比

**状态：`DRAFT_NOT_AUTHORISED`（R2）。** 本文件只是可授权的运行前草案，**没有任何执行、没有 `registration.json`、没有枚举、没有模型加载**。
启动它需要单独的书面授权。未经授权时，本文件不得被引用为"已批准的工作"。

- 上位裁定：2026-10-19 第二轮复核 + 第三轮授权前复核。R1（`e638f320`）之后按复核意见整体修订，见 §附录 A。
- 前置包：`P2-C-ANCHOR-01`（`a2d74c955`）。其结论已按 [ERRATA_01.json](../../../research/p2_c_anchor_01_20261019/ERRATA_01.json) 收紧为**结构可行性已建立、执行事务资格尚未建立**。
- 本轮**不做**：改引擎、真实续程、标签采集、价值头训练、GPU、正式 L2/L3、触碰最终种子、更换 SAA。

## 0. 唯一科学假设（本轮只回答这一条）

> **H-CAMR-STATIC-01**：在相同 strict-online 公开状态、冻结锚点、**固定双订单 mask**、**共享合法动作空间**与**受控计算预算**下，冻结 MaskCO 的重构评分是否会产生**不同于 distance-SGBS 的合法完整计划**？

**对应层级：L1（机制证据）。** 本轮**不**检验质量优劣，**不**检验在线效用，**不**检验可安装性。

### 本轮明确不回答（各自另立命题）

| 不回答的问题 | 为什么不在本轮 | 何时才谈 |
|---|---|---|
| 学习到的 **mask 选择**是否优于非学习 mask 规则 | 需要已训练的 mask-value 头；当前头是**零初始化**（恒 KEEP），用它选 mask 等于把未训练策略当证据 | 另立 mask-selection 命题，且必须先有可用的训练信号来源 |
| MaskCO 重构是否**质量更优** | 本轮只比"是否产生不同的合法计划"，不产生任何 J / 距离 / 能耗比较 | L2，且必须先解决 §8 阻断 |
| 重构结果能否**可靠传播到在线执行** | 执行事务资格未建立（ANCHOR-01 合成臂只证明接口有缺陷） | L3，且必须先有隔离事务实现 |

R1 草案把"固定 mask 下的 repair 差异"与"mask 选择差异"混在同一个假设里，已撤销；这是本轮最实质的设计修订。

## 1. 分析单元（不补抽、不扩分母）

| 项 | 值 | 依据 |
|---|---|---|
| 相位 | **15**（SAA 9 / MYOPIC 6） | `anchor_scope.csv` 判决 `ANCHOR_LEGAL_MULTI` |
| 队列长度分布 | `queue_len = 2`：13 个；`queue_len = 3`：2 个 | 同上 |
| 每相位双订单 mask 数 | 13 个相位各 **1** 个；2 个相位各 **3** 个 | `n_masks = C(queue_len, 2)` |
| **相位–mask 比较单元** | **19** = 13×1 + 2×3 | 本轮全部主结论的分母 |

- 分析对象**仅**这 15 个相位，**不补抽、不新跑源日、不换相位**。
- 15 个相位来自 **2 个**先前已检查的开发者实例（day 0 / day 1）。**19 个单元不是 19 个独立样本**，一律按"相位–mask 暴露量"报告，不做独立同分布假设、不算显著性检验。
- mask 生成规则固定为 `tuple(sorted(p) for p in combinations(queue, 2))`，与 ANCHOR-01 探针逐字一致；`registration.json` 必须显式存下这 19 个 `(day, ordinal, mask)` 三元组及其 sha256，运行不得增删。

## 2. 冻结输入与共享接口（公平性的硬要求）

四个"相同"缺一不可：**相同公开状态、相同冻结锚点、相同合法动作集合、相同计算预算**。任何一项不成立，该单元的比较结果**作废**，不得进入主结论。

### 2.1 共享合法动作接口（唯一一套，双侧同用）

R1 草案只写了"相同合法插入空间"，不够。实际代码里两侧**目前用的是两套不同谓词**，必须在运行前统一为一套薄封装（**不新增算法、不新增评分**，只是把既有谓词按同一顺序组合一次）：

| 现有路径 | 谓词 | 问题 |
|---|---|---|
| `p2_pos01_reference.legal_actions` / `probe_anchor.legal_actions_with` | `cand.feasible` **且** 全车队 `route_check` | ANCHOR-01 的口径 |
| `mpre_policy.enumerate_legal_actions` | **只有** `cand.feasible`（**无** `route_check`） | `run_sgbs_fixed_state.py` 用它 |

两者不是同一集合，且容量口径不同：`certify_route` 用 `load > capacity + 1e-6` 拒绝（`action_contract.py:190`），`route_check` 用 `load <= capacity - 1e-4` 接受（`p2_pos01_reference.py:43,226-227`）。**同一个包里存在两套容量容差**，若两侧各用一套，"差异"就无法归因。

**规定**：本轮唯一合法动作集合 =

```
legal_actions_shared(view, plans, customer, allowed)
  = { (c, action) | cand.feasible                       # certify_route 的证书
                    AND 对 trial 全车队逐车 route_check  # capacity - 1e-4 口径
                    AND allowed 显式给出（含在途车） }
```

- `allowed` 必须**两侧同一个对象**：`mutable_ids(view) ∪ {committed_vid}`。
- **不得**直接复用 `enumerate_legal_actions(..., mutable_ids)`（它把在途车排除在外，且不带 `route_check`）；只允许在共享封装内部调用它下层的同一个 `enumerate_actions_from_plans`。
- 容量口径固定为**更严的那一个**（`capacity - 1e-4`），因为 15 个相位本身就是在该口径下筛选出来的。

### 2.2 锚点冻结

锚点 = `(committed_next, committed_arrive, committed_finish)` 与取货后载重，**逐字不变**；`committed_next` 不得出现在任何后缀里。
（代码事实：`apply_action` 在每个分支逐字复制 `anchor_node/time/load`，所以"锚点未变"对该枚举器恒真——它只能当一致性检查，不能当"冻结无代价"的证据，见 ERRATA_01 E2。）

### 2.3 计算预算

- `sgbs_search` **没有墙钟截止**，预算全部是调用计数（`beam`、`max_per_parent`、`max_completions`），`wallclock_total_s` 只是事后**测量**（`run_sgbs_fixed_state.py:288`），不是上限。
- 因此本轮预算以**计数**声明并冻结：`beam=2`、`max_per_parent=2`、`max_completions=8`；另设一个**双侧相同**的墙钟上限作为硬失败条件（超时 = 该单元作废，**不得只作废一侧**）。
- **不得**事后调参：`beam` / `max_per_parent` / `max_completions` / 墙钟上限一经冻结，出现不利结果也不得修改。

## 3. 模型身份（MaskCO 侧的唯一评分入口）

R1 写的"冻结 CVRP decoder + 适配层 + 定向 insertion 头"**作废**——本线里不存在训练过的 insertion 头。以源码为准：

- **唯一评分入口**：`scripts/models/p2_pos01_mask_value.py` 的 **`FrozenBackbone.insertion_scores(view, base, partial, actions)`**（`:111-117`）。它返回 `edge_insertion_scores(L, cust, pred, succ)`，其中 `L = model.feature2logit(Z)`；**没有任何额外可训练参数**，也不含 mask-value 头。
- 该文件头两句是权威声明："Frozen real M-pre features and the **ONLY trainable CA-UGMR mask-value head**. CPU only. **No decoder fine-tuning, insertion head**, …"；且 `MaskValueModel.head.l3` 被显式置零（`:141-142`）⇒ **零初始化的 mask-value 头恒为 KEEP，本轮不得用它产生或挑选 mask**。
- checkpoint 身份已在本轮**实测核验**（只读哈希）：
  - `MASKCO_code/ckpts/cvrp100.ckpt`，305,061,150 bytes
  - `sha256 = 03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db`
  - 与 `EXPECTED_CHECKPOINT`（`p2_pos01_mask_value.py:32`）**逐字节一致**；`FrozenBackbone` 另外断言 `encoder_input_dim == 3` 且 `(d_encoder, d_decoder) == (512, 512)`，并记录 `parameter_sha256`。
- 运行时另须固定：`JAX_PLATFORMS=cpu`、`PUBLIC_NODE_BUCKET=51`、`coord_normalize_visible`（严禁 `coord_normalize`）。
- `registration.json` 必须写入 `checkpoint_sha256` + `parameter_sha256` + 上述常量；运行后逐字比对，不一致即整包作废。

## 4. 输出契约（对称化，不得混用两种对象）

R1 的不对称在于：MaskCO 天然给 1 个确定性输出，distance-SGBS 给 ≤8 个补全。**单个决策输出与候选集合是两种不同的对象，不能直接相比。**

**主对象 = 单个决策输出（每单元每侧 1 个计划）。**

- distance-SGBS 的单输出选择规则**预先冻结**为 `completions[0]`：即"无分支的贪心补全"，它是搜索在**任何 beam 分支发生之前**产生的第一个补全，因而与 beam 宽度无关、可复现、且不是事后择优（**严禁**用 `min(J_vis)` 式择优）。
- MaskCO 侧单输出 = `insertion_scores` 逐层 argmax 的确定性贪心补全，与 `completions[0]` 同构。
- 两侧补全都必须先通过 §2.1 共享接口与 §2.2 锚点检查，否则记为**该侧无输出**。

**次对象 = 完整合法补全集合（单独报告，不参与主判据）。**

- 分别报 MaskCO 集合与 distance-SGBS 集合的 `n_unique` / `n_total` / `n_fail`、集合大小、以及**集合归属**（MaskCO 单输出是否落在 SGBS 集合内）。
- 集合覆盖**不得**用来挽救主对象为空的结果：主对象相同即"无差异"，集合层面的差异只能作为描述性补充，不得改写第一层判定。
- 旧 `_eval_best` 的接受规则**禁止带入**：它用 `validate_repair_plan` + `min(J_vis)` 选优，且在无认证补全时静默返回 `J0`（`run_sgbs_fixed_state.py:133-148`），把"没产出"与"产出但没更好"混为一谈——而这两者的区分正是本轮的全部意义。

## 5. 主观察量（三个，分开报）

对 19 个单元，逐单元登记：

1. **合法性**：MaskCO 单输出是否保持锚点、并通过 §2.1 共享参考检查（`plan_hash` 级）。
2. **是否属于 distance-SGBS 的合法补全集合**（集合归属，`plan_hash` 比较）。
3. **两者最终被比较计划的身份与锚点后首动作**：`plan_hash` 是否相同；若不同，`suffix[0]`（0 = 返仓）是否相同。

判据层级：**不同且都合法**才是本轮的阳性机制证据。**锚点后首动作相同而更深后缀不同**，必须如实登记为"差异存在但不在紧邻动作上"，不得因为"首动作相同"就判无差异（ANCHOR-01 已明确此口径）。

## 6. 两层 Go / No-Go（因果解释已修正）

### 第一层：MaskCO 独立机制是否成立

**Go（通过）** 需要同时满足：

- 至少一个单元出现 MaskCO 单输出 **≠** 主对照单输出（`plan_hash` 不同）；
- 两侧输出**都**通过共享参考检查且保持锚点（差异不得靠违规换来）；
- 两侧在**同一** `allowed` 集合、**同一**侧预算、**同一**共享接口下产生（预算或集合不对称 ⇒ 证据无效，不算 Go）；
- 差异**可复现**：整包连跑两次，19 个单元的 `plan_hash` 逐字相同。

**No-Go（不通过）** 的正当理由**只有**以下五种：

1. 19 个单元里 MaskCO 与主对照的 `plan_hash` **全部相同**（"没有差异"）；
2. 差异**不可复现**（两次运行不一致）；
3. 差异**不合法**（锚点被改，或参考检查不过）；
4. 差异**可完全由预算/动作集合不对称解释**（把不对称消除后差异消失）；
5. 差异**可由非学习 proposer 直接复现**（distance-SGBS 在同等预算下也能产出同一计划 ⇒ 不是模型特有行为）。

> **撤销 R1 的旧判据**："差异仅来自 mask 或 insertion 任一通道 ⇒ No-Go" 作废。
> 本轮 mask 固定，唯一可能的差异通道就是 repair/insertion 选择——**这正是本轮要检验的对象**，把它当失败理由等于把假设本身判死。真正决定成败的是"差异是否独立、合法、可复现、可归因"。
> 反方向的约束同样成立：**行为不同也只是 L1 机制证据，永远不能升级为 L2 质量优势。**

### 第二层：只在第一层通过后开启

若通过 → 才值得申请：(a) 冷链物理认证检查（**必须先新增**能耗/品质投影路径，见 §8）；(b) **隔离的**事务实现。
即便第二层通过，**不自动**获得完整认证、真实续程或训练授权。

## 7. 止损线（新增，本轮写入）

- 仅仅发现 MaskCO 与 distance-SGBS 产生**不同路线**，**不足以**授权训练价值头，**也不足以**授权开发完整事务引擎。
- 后续仍必须证明：这些差异具有**可信、可认证的质量价值**（能过物理认证，且不是靠违规或预算不对称换来）。
- 若第一层 No-Go：**停止训练路线**，不再投入持久化工程、未来标签生成与价值头训练；把结论写成负结果，不改成"方法还需调参"。

## 8. 三个独立阻断（不合并，且**绕开 ≠ 解决**）

| 阻断 | 阻止当前静态 L1 对比？ | 何时必须解决 |
|---|---|---|
| C0 正式预算 `B` 未裁决 | **不阻止**（L1 只做路线级机制诊断） | 完整冷链可行性认证、任何 L2 之前 |
| HIST 未来支持域不足 | **不阻止**（本轮不使用未来信息的静态对比） | 真实反事实续程标签之前 |
| 执行事务资格未建立 | **不阻止**（本轮只比较静态计划） | 任何真实在线安装、L3 之前 |

配套代码事实（本轮复核确认）：`certify_route` 无条件返回 `projected_energy_kwh / projected_quality_loss / projected_thermal_margin = None`（`action_contract.py:171-173`），**能耗与品质在本管线里是"跑不了"，不是"没跑"**。任何物理认证主张都必须先新增这条投影路径；`B_ref` 不能自动替代正式批准的硬预算。

**明确声明**：静态工作能绕开这三个阻断，**不等于**它们已被解决。三者继续各自登记为 open。

## 9. 运行前必做（顺序不可交换）

1. 冻结 `registration.json`：唯一假设、19 个 `(day, ordinal, mask)` 三元组、共享接口定义、容量口径、锚点断言、主/次观察量、`completions[0]` 选择规则、预算常量与墙钟上限、两层 Go/No-Go、止损线、禁止项、模型身份三元组（`checkpoint_sha256` / `parameter_sha256` / 常量）。
2. 逐一 sha256 记录：`p2_pos01_mask_value.py`、`action_contract.py`、`p2_pos01_reference.py`、`p2_pos01c_gap.py`、`strict_online_env.py`、`run_sgbs_fixed_state.py`、两个相位文件。
3. **零假设自检（三条，全过才可正式跑）**
   - 主对照对自身确定性：distance-SGBS 在同一单元重复运行，`completions[0]` 逐字相同；
   - MaskCO 侧确定性：`insertion_scores` argmax 贪心在同一单元重复运行，`plan_hash` 逐字相同；
   - 接口一致性：把 MaskCO 的 `insertion_scores` 排序换成 `-incremental_distance` 排序后，必须**逐字退化**为 distance-SGBS 的 `completions[0]`——若不能退化，说明两侧走的不是同一条搜索路径，先修接口再谈模型差异。
4. 再跑正式对比；整包连跑两次做可复现性核对。

第 3 条最后一项是本轮识别"差异来自模型还是来自搜索路径"的关键闸门；它不过，后面所有差异都不可归因。

## 10. 禁止项

修改执行引擎；修改控制器或阈值；重跑源日；采集标签；训练；GPU；触碰最终种子。
声称在线效用；声称 CAMR 新颖性；把 `candidate` 说成"部署"。
在 15 个相位之外挑选相位；在出现不利结果后更换 mask 规则、容量口径、共享接口、预算或 `completions[0]` 规则；用集合覆盖改写主对象判据；用 `min(J_vis)` 事后择优替代 `completions[0]`。

## 11. 授权状态

**`DRAFT_NOT_AUTHORISED`（R2）。** 本文件可被审阅、可被要求再次修订；在书面授权之前，不得创建输出目录、不得冻结 `registration.json`、不得运行任何枚举或模型推理。本轮亦未新增任何算法模块。

## 附录 A：R1 → R2 修订记录

| # | 复核意见 | R1 的问题 | R2 的处理 |
|---|---|---|---|
| 1 | 科学问题混杂 | 同一 mask 下比较，又要求解释 mask 选择差异 | §0 唯一假设固定为"固定 mask 下的 repair 差异"；mask 选择移出本轮，列入 §0 不回答表 |
| 2 | 主对照输出不对称 | 未定义 MaskCO 一个输出 vs SGBS ≤8 补全如何对等 | §4 主对象 = 单决策输出，SGBS 单输出预先冻结为 `completions[0]`；集合覆盖单列为次对象且不得改写判据 |
| 3 | Go/No-Go 过严 | "差异仅来自 mask 或 insertion 任一通道 ⇒ No-Go" | §6 撤销该条；改为五种正当 No-Go 理由（无差异／不可复现／不合法／预算不对称可解释／非学习 proposer 可复现），并写明 insertion 通道差异**正是**本轮检验对象 |
| 4 | MaskCO 身份歧义 | 写成"decoder + 适配层 + 定向 insertion 头" | §3 指定 `FrozenBackbone.insertion_scores()` 为唯一入口；声明**无**训练过的 insertion 头；checkpoint sha256 已实测核验；零初始化 mask-value 头禁用 |
| 5 | 历史 SGBS 不能直接当已合格对照 | 只说"复用 `sgbs_search`" | §2.1 要求共享合法动作接口（现已查明两侧谓词与容量口径都不同）；§2.3 声明预算是计数制、无墙钟截止；§4 禁用旧 `_eval_best` 接受规则 |
| 6 | 分析单元与样本性质 | 只写"15 个相位" | §1 明确 **19** 个相位–mask 单元（13×1+2×3），并声明非独立样本、不做显著性检验 |
| 7 | 阻断需区分阶段 | 只写"两个独立阻断" | §8 改为三行表（C0 预算 / HIST / 事务资格）分列"是否阻止本轮"与"何时必须解决"，并加"绕开 ≠ 解决" |
| 8 | 新增止损线 | 未写 | §7 写入：仅有不同路线不足以授权训练价值头或完整事务引擎 |
