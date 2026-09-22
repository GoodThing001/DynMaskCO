# MaskCO 直接目标训练工作包（有限学习原型）

> 2026-09-21 后续决定：原配方扩张仍停止；补充[状态身份与固定状态诊断](../决策与审计/直接目标训练后续决策_状态身份与固定状态诊断.md)。当前本地采集器的引用保存与实例计数存在具体协议风险，先核对运行身份，不把新发现自动等同于服务器缺陷。该后续包单独规定最小诊断、条件性唯一纠正训练和停止边界。

> 最新状态：2026-09-21 · seed42 直接目标训练与 Step 5 CAL/DEV 闭环已运行，当前配置未建立相对 M-pre 或 R 的学习增量，按有限投入规则收束。不自动加步数、seed43/44、换奖励或未 padding 重跑。详见 [Step 5 收束结果](../结果与进度/MaskCO直接目标训练Step5收束结果.md)。以下步骤、验收和历史回传保留为设计与实施记录，不代表本轮全部独立复验；最新证据边界以收束结果为准。

## 1. 研究问题（固定）

> 在相同可见信息、可行性规则和在线预算下，预训练 MaskCO 能否通过完整冷链目标的直接训练，生成比未微调 MaskCO 和 regret-2 更有效的局部重构？

## 2. 与已收束配方的区别

| 项目 | 已收束配方 | 新工作包 |
|---|---|---|
| 学习目标 | 复制 R 的下一步插入（CE） | 提高完整修复的冷链效用（奖励） |
| 预训练利用 | 冻结 encoder，新建 decoder | 复用可兼容的预训练 encoder、decoder 和输出路径 |
| 冷链条件 | 客户属性及少量约束余量 | 显式加入车辆热状态、在舱货物及当前部分计划 |
| 训练结果判据 | 插入 CE、闭环表现 | 完整候选质量、生成分布变化、最终闭环增量 |
| teacher | 必需（regret-2） | 不依赖新的强 teacher |

「RL＋修复」已有研究先例（含 VRPTW，如 [Neural improvement](https://arxiv.org/abs/2205.00772)），是合理候选训练方式，但本身不是创新；仍需证明**动态冷链条件与 MaskCO 机制**的具体贡献。

## 3. 六步实施顺序

### 第一步：整理研究入口和运行身份（不跑新实验）

1. 把交换结果加入实验总表，标为「当前搜索配方负结果」；保留旧摘要，同时补候选预算（R 16 重构 / S ~12 重构 + ~33 交换）与统计口径（聚类 bootstrap CI）说明。**已完成**（见 [总表 §1.5j](../结果与进度/实验结果总表.md)、[交换结果](../结果与进度/cc_lns_swap增强修复验证结果.md)）。
2. 更新状态入口：旧 N1/N2、交换包均结束；下一项 = 「MaskCO 直接目标训练的有限原型」。
3. 新包选定一套数据、合同、profile、源码和 checkpoint 身份，记录完整 SHA256、实例列表及各类 seed。本地与服务器差异只能写「原因未核实」，不能直接认定一定来自数据或 seed。不要把两套历史 R 数字合成同一张排名表。

### 第二步：真实预训练 MaskCO 重构对照（M-pre）

**⚠️ 2026-09-20 checkpoint 维度核对（本地 + 服务器）**：预训练 decoder 与扩展 encoder 维度不匹配，且服务器上**没有**适配的中间 checkpoint：

| 来源 | 类型 | embed_dim | 输入 | 层数 |
|---|---|---|---|---|
| `MASKCO_code/ckpts/cvrp{100,500,1000}.ckpt` | CVRP（含容量） | **(512, 512)** | 3D（coord+demand） | 16+6 |
| `MASKCO_code/ckpts/tsp{100,500,1000}.ckpt` | TSP | (256, 256) | 2D（coord） | 16+6 |
| 扩展 `ckpts/*/step*.ckpt`（base_r1/c1/rc1、p0_*、cross_edod、r1_5_baseline…） | DynamicColdChainModel（encoder-only） | **(256, 256)** | 7D | 16+6 |

服务器全部扩展 checkpoint 都是 256 维、7D、encoder-only；无 512 维扩展 encoder，也无 256 维 CVRP decoder。

**选定方案（M-pre）**：**直接用完整预训练 CVRP 模型 `cvrp100.ckpt`（512 维 encoder+decoder，3D 输入 coord+demand）**。理由：① 唯一容量感知的预训练模型，提供**容量感知的静态路由先验**（并不原生表达动态取货和冷链过程）；② 无新增层，`权重真实复用` 最干净（避免 256→512 投影这种非预训练、非原则的桥接层）；③ encoder 缺 TW/温区/reveal 正是 M-trained 要补的增量，使 M-pre→M-trained 增量可解释。M-trained 与 M-pre 同源（都用 `cvrp100.ckpt` 骨干，见第二步补），故 M-pre→M-trained 是「同一骨干 + 冷链条件适配 + 直接目标训练」的整体增量，不单独归因 encoder 与 decoder。

**实现要点**：`load_ckpt` 加载 `cvrp100.ckpt`（`models.CVRPModel`，需 `tensorboardX` stub 或服务端已装）→ `decode(target='logit')` 出**对称**边 logits `features@features.T`（经 final_norm + final_logit_scale=16/512 + softcap，对角线 mask）→ 固定映射转插入偏好（见下）→ 整体移除 mask、贪心重插 → 完整认证 + `J_vis` 接受（沿用当前版本）。预训练边输出对称的局限：方向/车辆锚点/冷链状态由后续适配层补，不声称原模型原生具备。

- 加载现有 checkpoint 中实际保存的 encoder、decoder 和边输出相关参数；逐项记录哪些成功加载、哪些需新增（本步记录：cvrp100.ckpt 的 encoder+decoder+`feature2logit` 全加载；需新增的仅有冷链接入/方向适配，属 M-trained）。
- 输入真实部分计划 + 掩码信息，timestep 用**保留边比例**（原计划与部分计划共有边 / 原计划边，对齐 CVRP `keep_prob=timestep` 语义；这是双方共同的适配定义，不声称完全复现预训练掩码分布）。
- 预训练边分数经**预先固定**的映射转合法插入偏好：`score(c at pred→succ) = L[pred][c] + L[c][succ]`，仅当 pred≠succ 时再减 `L[pred][succ]`（空车开路线 pred=succ=depot 无自环可删，对角哨兵不进算术）。映射是适配规则、要记录，不接随机头后称「预训练直接重构」。
- 整体移除 mask 再完成修复；完整认证和 `J_vis` 接受规则沿用当前版本。

验收（不要求立刻超过 R）：**权重真实复用、输出确实参与重构、候选合法、调用口径一致。** 依据 [MaskCO 原论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d85816bf41e54d9b0847e02249f4fcc6-Abstract-Conference.html)。

**实施状态（2026-09-20）**：M-pre 已实现并本地 smoke 通过——`scripts/models/mpre.py`（`load_cvrp_model`/`encode_cvrp`/`decode_edge_logits`，cvrp100.ckpt 加载出 512 维 CVRPModel，边 logits 对称、对角掩码、有限值）+ `scripts/simulation/mpre_replanner.py`（`mpre_reconstruct` 增删边评分贪心重插 + `cc_lns_mpre_search` + `MPreReplanner`）+ `scripts/evaluation/run_mpre_fixed_state.py`（同状态 R vs M-pre 对照）。2 实例 smoke：固定状态 mean Δ(J_M−J_R)=+0.042（D/Q/E 全更差）——**暂标「旧映射 smoke 结果」**，因下述两处接线问题未修，**现在还不能把变差解释为「CVRP 预训练与冷链任务不同、所以符合预期」**；正式 TRAIN-64 对照属第五步比较包，暂不跑。

**第二步两处接线问题（已并入第三步并修复，单测通过）**：

1. **空车插入不能减对角掩码值**：已修——`if pred != succ` 才减 `L[pred][succ]`，空车开路线 pred=succ=depot 无自环可删，对角哨兵不进算术（单测：普通插入 5、空车 10，不再是 1.7e38 伪收益）。
2. **timestep 对齐预训练保留语义**：已修——`retained_edge_ratio(A0, A_partial)`（保留边比例，对齐 `keep_prob=timestep`；单测 0.5）。

### 第二步补：M-pre→M-trained 架构决策（2026-09-20 定）

**M-trained 使用与 M-pre 同源的完整 CVRP 512 模型**：冻结 3D encoder，微调预训练 decoder，新增冷链状态适配层 + 有向插入残差头。本轮不接回旧 256 维 encoder、不做 256→512 桥接。「冻结 encoder」指 **`cvrp100.ckpt` 的 encoder**。

| 模块 | 来源与维度 | 本轮训练 |
|---|---|---|
| 坐标+需求输入投影、depot embedding、encoder | `cvrp100.ckpt`，512 维 | 冻结 |
| 冷链状态适配层 `C` | 新增，输出 512 维残差 | 训练 |
| `mid_proj`、timestep embedding、decoder、`final_proj`、输出归一化 | 同一 `cvrp100.ckpt` | 训练 |
| 原边 logits 路径 | 保留 `feature2logit` | 随 decoder 更新 |
| 有向插入残差头 `δ` | 新增，每个合法动作标量修正 | 训练 |

$$H_0=E_{\mathrm{CVRP}}(X_{\mathrm{coord,demand}}),\qquad
Z=D_\theta\bigl(H_0+C_\phi(S_{\mathrm{visible}},P_{\mathrm{partial}},M),\,t,\,A_{\mathrm{keep}}\bigr)$$
$$s_\theta(a)=s_{\mathrm{edge}}(a;Z)+\delta_\psi(a,Z,\text{目标车辆状态},P_{\mathrm{partial}})$$

- `C` 给 decoder 补动态冷链条件；`δ` 按客户/目标车辆/前驱/后继的**有序关系**修正插入分数；原对称边 logits 继续提供结构偏好，不承担其表达不了的方向/车辆状态信息。
- **适配层与残差头最后一层零初始化，内部层正常初始化**：训练前、共享解码规则下，M-trained 初始动作分数 == 修正后 M-pre，避免随机新增模块破坏预训练行为。零初始化时上游梯度可能暂为零属正常；验收先确认输出层能更新，再确认更新后梯度进内部层，不把所有层都零初始化。

### 第三步：新学习器输入 + 训练—部署链路一致性（一次做好）

**输入 schema `repair_state_v1`（五张表，统一由公开状态生成；不改预训练 3D 输入层）：**

| 表 | 必需字段 |
|---|---|
| `nodes` | 可见坐标、需求、TW、服务时间、温区、初始品质；served/committed/mutable/masked 标记 |
| `vehicles` | 当前状态、anchor、ready time、载重/剩余容量、各温区当前温度、保护标记 |
| `cargo` | 每笔在舱订单的所属车辆、需求、温区、当前品质、取货后暴露时间 |
| `partial_plan` | 每车有序 suffix、客户归属、位置、前驱/后继、待修复集合、不可修改部分 |
| `context` | 当前时钟、公开合同/profile 标识、归一化尺度、各表有效位与索引映射 |

固定原则：① 512 维 encoder **仍只吃坐标与需求**，TW/冷链/车队从适配层进入；② cargo 保留订单级记录，经带有效位的聚合得到车辆表示，不先压成「平均品质」丢掉明细；③ 车辆 ID/订单 ID 只用于关联与回放，**不作可学习身份 embedding**；④ 每步插入后更新部分计划、合法动作、计划资源摘要，不把「计划将来取货」写成「当前已入舱」；⑤ 尺度来自公开合同或 TRAIN 固定统计，不随隐藏未来变化；⑥ padding 必须在**归一化、encoder 注意力、decoder 注意力、状态聚合**中一致屏蔽。

**统一动作分布与可微路径**：对全部合法「客户—车辆槽位—插入位置」动作做 softmax。训练从此分布采样、记录被选动作 `log_prob`；确定性评价取 argmax；M-pre 与 M-trained 用相同动作枚举、温度、解码规则、候选预算。原 greedy-regret 版本保留为 `M-pre-regret` 历史实现，正式训练增量比较用共同策略下的 M-pre，不把换解码规则的效果归给训练。`mpre.py` 当前把 decoder 输出转 NumPy 只用于推理；训练评分与 `log_prob` 必须留在 JAX 图内，合法动作枚举/计划应用/C0 评价可在图外。

**验收（一次交付，不增加正式性能实验）：**

| 验收 | 通过条件 |
|---|---|
| 初始继承 | 零残差 M-trained 与 M-pre 在相同策略下，合法动作分数及确定性选择一致 |
| 边界评分 | 空车/单客户/普通插入均不用对角哨兵产生伪收益 |
| 因果 + padding | 扰动隐藏未来、改变填充长度不改变有效分数与动作 |
| 计划一致性 | 逐车有序路线、exact-once、保护集合、committed 全查，不用客户集合代替 |
| 拒绝隔离 | 丢弃候选后公开状态与原计划无残留变化 |
| 梯度链路 | decoder 与新增模块按预期更新；冷链条件能影响输出；冻结 encoder 参数不变 |
| 训练/部署共用 | 相同状态/参数/动作前缀得到一致合法动作分数 |

完成后只做这些边界的小 smoke，再按 seed42、1000 次更新预算衔接第四步。暂不跑正式 TRAIN-64 M-pre 对照、不扩网络/预算。M-pre→M-trained 的可解释结论 = **「同一 CVRP 预训练骨干 + 冷链条件适配 + 直接目标训练的整体增量」**；冷链输入与 decoder 微调各自贡献留到有信号后的消融。

**实施状态（2026-09-20，已按复核补强）**：`scripts/models/mpre_trained.py` 已实现 `MpreTrainedModel`；继承 smoke `scripts/evaluation/run_mpre_inheritance_smoke.py` **9 项全过、失败返回非零退出码**：① 参数继承（train 83 / frozen 163 / leak 0，backbone 与 M-pre 逐元素一致）；② 零残差（C=0、δ=0、输出有限）；③ 原输出继承（非对角 logits 差 0.0）；④ 动作继承（分数/log-prob 差 0.0、argmax 一致，logp_diff 入通过条件）+ ④b 空车动作只含两条新增边（err 0.0）；⑤ 梯度与冻结（decoder 60/c_adapter 2/delta_head 1 非零梯度；冻结参数从更新后模型重提取逐元素不变；train 模块 74 个实际更新；独立 M-pre 不变）；⑥ 附加特征通路（出口更新后扰动改变 log-prob 0.027，第二次反向传播梯度进入适配内部层 4 参数）；⑦ padding（B=2 不同有效长度 + 不同有限填充，init 与更新后模型，分数差 <8e-5、argmax 一致）。

**Step 3 后半（①②③ 已完成）**：`scripts/data/repair_state.py`（`extract_repair_state_v1`，F_NODE=10 / F_ACTION_EXPLICIT=17，含目标车辆 X_state 13 维 + 增量/松弛 4 维；车辆/订单 ID 只作关联索引）+ `scripts/simulation/mpre_policy.py`（`enumerate_legal_actions`/`stepwise_reconstruct` 统一逐步策略 + `validate_repair_plan` 完整计划对账）+ `scripts/evaluation/run_mpre_policy_smoke.py`（真实决策点端到端验收）。**3 事件全过**：零残差 M-trained 与 M-pre 逐步动作序列一致（seq_match）、log-prob 差 0.0、计划对账（exact-once/逐车有序/committed/protected 不变）通过、评分确定性通过。拒绝隔离由 stepwise 副本语义保证（不 mutate 输入 plans）；在线写回后无副作用的完整验收待 Step 4 的 M-trained 在线 replanner。

**Step 4 门槛**：预训练继承 + 初步梯度 + 真实冷链条件 + 完整策略 + 部署一致性均已通过 smoke。下一步按已定预算（seed42、1000 次更新、batch 8 状态×4 采样、≤32000 候选评价）接直接目标训练（REINFORCE，r=(J_vis(P0)−J_vis(P))/s_TRAIN）。训练前锁定 schema（repair_state_v1）、动作策略（统一 softmax）、checkpoint 身份与验收报告。

**训练接线修正 + 训练器（2026-09-20）**：按复核修正 `repair_state.py`（`resolve_target_vid` 传 `allowed_vehicle_ids` 与 `apply_action` 一致；mutable 通道填充；松弛量加有效位；clock 相对时间窗；**订单级 cargo 表 + 有效位**）+ `mpre_policy.py`（`stepwise_sample`/`replay_log_prob` 采样与可微回放分离；动作回放记录解析后真实 vehicle_id；`validate_repair_plan` 全查车辆集合/anchor/exact-once/非 mask 归属与顺序/保护车，含反例）。已实现 `scripts/training/train_mpre_reinforce.py`（REINFORCE，采样/回放分离、只更新可训练分区、失败奖励预固定、正确 1000 步训练循环、`s_TRAIN` 由 TRAIN 决策点 J0 标准差固定、**功能式 JIT + 固定容量 padding** 提速 ~16×）。**smoke 全过**：继承（9 项，冷链接口用真实 initial_quality 扰动 logp_diff 0.18）+ 策略端到端（完整动作记录/对账/隔离/三反例）+ REINFORCE 可微回放（梯度+frozen）。

**Step 4 训练完成（2026-09-21，服务器 CPU，seed42）**：1000 更新跑完（~5.5 小时），`s_TRAIN=0.35`、256 决策点、`Nv_max=51`/`M_max=40`。loss 终值 −0.0104，全程 mean −0.0037 / min −0.0617 / max 0.0466（幅度小、批次间波动、无发散）；reward_mean 全程为负（−0.098 ~ −0.320，预训练 CVRP 采样的修复平均差于 P0，符合预期）。产物：`results/m0_scale/mpre_reinforce_s42/model.ckpt` + `summary.json`（已下载本地）。**训练本身不回答 M-trained 是否优于 M-pre/R**——须接第五步闭环比较（B/R/M-pre/M-trained，固定状态 + CAL/DEV 闭环）才能判读学习增量。

### 第四步：小规模「完整修复奖励」训练（不生成新 teacher 数据）

第一版用简单策略梯度（不先上 actor–critic/PPO）。对同一公开状态+部分计划，从合法动作分布采样完整修复 P；认证通过的候选用：

$$r(P)=\frac{J_{\mathrm{vis}}(P_0)-J_{\mathrm{vis}}(P)}{s_{\mathrm{TRAIN}}}$$

- P₀ = 共同事件初始完整计划；s_TRAIN = 只在 TRAIN 固定的正尺度。
- 奖励来自完整修复，不能用某一步距离下降替代；可行但变差的候选保留负奖励（不截零）；不完整/不可行用预先规定的失败处理，不参与「少服务更便宜」比较；不读该实例真实未来。
- 基线 = 同状态其他采样结果的平均奖励。训练保留候选动作概率、认证结果、奖励，可追溯到整份计划。

**只优化公开场景目标，不保证动态终局改善**——目标错位风险仍在，须在后面闭环检验。

| 项目 | 建议上限 |
|---|---|
| 场景 | 50-node、R1、EDoD=0.5 |
| 训练状态 | TRAIN-64，每实例最多 4 个固定规则抽取的 eligible 状态 |
| 首轮训练 | seed42，1000 次更新 |
| 每批 | 8 状态 × 4 份完整修复/状态 |
| 完整候选评价 | 训练最多 ~32,000 次（验收/评估成本另列） |
| checkpoint | 预先固定最终 checkpoint，不在 CAL/DEV 挑步数 |
| 变量 | 固定一种模型、一种奖励、一种采样配置 |

原型预算，不是「足以训练成功」的保证；不抽已知有改善的状态，不把同实例多事件当独立实例。

### 第五步：连续比较包决定是否继续

固定状态 + CAL/DEV 闭环，四种方法：

| 方法 | 作用 |
|---|---|
| B：JF1-H-F | 判断实际改善 |
| R：regret-2 + 保护 | 判断学习独立价值 |
| M-pre：真实预训练重构 | 判断新训练增量 |
| M-trained：直接目标训练后 | 新方法 |

固定状态比较相同起点、mask、候选上限；闭环允许各方法形成自己的轨迹，主比较用统一整链时间预算（冷启动编译单列，在线额外重编译不能从实际用时消失）。

必报：完整服务/硬约束；逐实例 J、D/Q/E；**直接相对 R、相对 M-pre 的配对差异**；候选可行率/独特率/接受率；实际重构次数/评价次数/p50/p95 用时/超时。固定状态优势不代替闭环优势；加候选数的收益不全归学习。

**seed42 同时出现对 M-pre 的改善 + 对 R 的有用信号**，才用相同配方预算跑 seed43/44（seed42 保留、不挑 best）。无信号则停止，不自动加步数或换奖励。

### 第六步：只有学习信号成立才扩论文矩阵

继续条件：服务/硬约束通过；三 seed 总体支持相对 M-pre 的训练增量；相对 R 至少建立一种优势（同预算更好真实 J，或达到预定质量用时更少）；结果不依赖单异常实例/单 best seed。探索阶段不强求每个区间排除零，但要看效应大小、跨 seed 一致性、成本。

通过后加 PyVRP-RH-D / OR-Tools-RH-D / RRNCO-Ordering-RH-D，并做对应消融：预训练 vs 同容量随机初始化；冷链状态条件有/无；事件 mask vs 同大小随机 mask；单轮 vs 多轮重构（匹配总预算）。再扩分布/动态程度/规模/参数敏感性，最后锁独立测试。CAL/DEV 始终是开发数据。

## 4. 立即交付（前三步）

1. **修正归档解释**（已完成，见上）。
2. **建立真实预训练重构对照（M-pre）**。
3. **完成新接口一致性**。

然后按固定预算衔接第四、五步（直接目标训练 + 闭环比较）。若这个边界明确的原型仍失败，带着证据与导师讨论缩小问题范围或调整研究要求，而不是继续换配方承诺成功。
