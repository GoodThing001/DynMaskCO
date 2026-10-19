# P2-C-ANCHOR-01：独立复核处置（第二轮）

**处置状态：`ACCEPTED_WITH_SCOPE_TIGHTENING`。** 本轮**不修改**已封存的 `REPORT.md` 与全部数值产物；收紧以 [ERRATA_01.json](ERRATA_01.json) 登记，今后引用一律使用收紧后的措辞。
本轮**不启动** CAMR-STATIC-01，不改引擎，不采标签，不训练。

基线提交：`a2d74c95509061ed60d7d23bab1e5758a296112b`（`main` = `codex/o0cc-cal-phys`）。

## 0. 处置总表

| 裁决中的判断 | 我的处置 | 依据 |
|---|---|---|
| 当前原型是否排除了本可考虑的在途后缀 | 接受 | `P2-POS-01:59` 原文 + `_commit()` 单格推进 |
| 冻结锚点后是否存在不同路线候选（15 相位） | 接受，但降级为"当前枚举规则下" | `action_contract.py:377-427` 是插入式枚举 |
| 候选是否通过路线级 TW / 容量 / 返仓检查 | 接受该范围 | `p2_pos01_reference.py:341-359` |
| 候选是否满足完整冷链品质与能耗认证 | **接受"未证明"，并追加"结构上不可能由本管线证明"** | `action_contract.py:171-173` 三个投影字段恒为 `None` |
| 原执行机制是否可能丢失候选后缀 | 接受 | `synthetic_tx.json` 三个臂 |
| CAMR 是否比强非学习搜索产生更高效用 | 接受"未证明" | 本轮未做任何效用计算 |
| "合法层面可以放开且不付合法性代价" | **撤销**，替换为收紧句 | 见 §1、§2.2 |
| `ANCHOR_SCOPE_ESTABLISHED` 的效力 | **仅作描述性机制发现**，不构成相对保守域的正式比较优势 | 主对照未按原规定执行（§2.3） |
| CAMR-STATIC-01 | 已就位草案，`DRAFT_NOT_AUTHORISED` | 见 §5 |

## 1. 措辞收紧（今后唯一可引用的表述）

`REPORT.md:97` 的"**可以放开，且不付合法性代价**"作废。替代句：

> **在当前枚举规则和路线级检查范围内，存在保持在途锚点不变的不同候选计划，尚未建立完整冷链物理可行性和实际安装资格。**

> Within the current enumeration rules and the route-level check, distinct candidate plans exist that keep the in-transit anchor unchanged. Full cold-chain physical feasibility and actual installability are **not** established.

`REPORT.md` 原件保留原文不改；引用方以本文件与 `ERRATA_01.json` 为准。这是科学证据范围的限定，不是结果回改。

## 2. 三个科研问题的逐条处置

### 2.1 「152 个候选不等于 152 个 MaskCO 学习选择」——接受，并按核到的代码补强

`probe_anchor.py` 里**没有任何模型、checkpoint 或 decoder**。候选的唯一来源是 `action_contract.enumerate_actions_from_plans`（`:377-427`）：对每个 anchored slot 把**一个**客户插到 `base = p.suffix` 的每个位置，外加一个匿名 `NEW_ROUTE` slot；`probe_anchor.py:72-85` 只在此基础上对 mask 内的订单做**插入顺序**递归，并用 `plan_hash` 去重。

所以 152 这个数的构成是"插入位置 × 车辆归属 × 插入顺序"，不是"不同重构策略"。相位侧的结构也已被核实：15 个可测相位中 **13 个只有 1 个双订单 mask**，仅 2 个（`queue_len = 3`）有 3 个 mask。

**处置**：本轮证据只支持"锚点后的重构动作空间存在"，**不支持**"CA-UGMR 学习掩码选择具有丰富监督信号"。两者在后续工作中必须分开记账（对应 CAMR-STATIC-01 的关键消融）。审稿人会问"贡献来自学习 mask 还是普通插入搜索"——这正是当前证据无法回答的问题，我同意把它写在包的显式限制里。

### 2.2 「`n_legal == n_raw` 不是独立证明冻结没有代价」——接受，且理由比裁决更硬

裁决给出的理由是"预过滤后再检查，自然容易全部通过"。核到代码后可以更准确地讲：

1. `legal_actions_with()`（`probe_anchor.py:52-69`，镜像 `p2_pos01_reference.py:362-374`）**在候选被计数之前**就已经丢弃 `not cand.feasible` 与任一 `route_check` 失败的 trial。因此 `n_raw_candidates` 本身就是过滤后的集合，`n_legal/n_raw` 只说明"预过滤结果内部一致"。
2. 更强的一点：**登记的 5 条锚点冻结断言里有 2 条对该枚举器恒真**。`apply_action()` 在每个分支都逐字复制 `p.anchor_node / p.anchor_time / p.anchor_load`（`action_contract.py:226, 241, 249, 256`），枚举又只做插入（`:393-396`），所以"锚点未被改动"和"`committed_next` 不在后缀里"**不可能**被这个枚举器违反。真正被行使的只有 exact-once 那一条，记录为 0 失败——这是一条真实的（虽然很弱的）一致性检查。

**处置**：`REPORT.md:41` 那句作废；"152 个候选内部自洽"保留可引用，"冻结没有合法性代价"不作任何形式的引用。同时记下：要真正给冻结定价，必须在不冻结的可行域上枚举，那是 CAMR-STATIC-01 之外的另一个设计问题，本轮不假装能回答。

### 2.3 「预声明的主对照没有按原规定执行」——接受，不申辩

[amendment_01.json](amendment_01.json) 已记录：`primary_control` 未执行、属运行后登记、替代对照与原因已披露、原始产物未改。我补充一点代码事实支持"它按字面写不可执行"：把在途车排除出 `allowed_vehicle_ids` 后，mask 里的队列订单既留在它的冻结后缀里、又可以被插入别的车，得到的是**重复服务计划**；而登记的对照只用 `route_check`（按车、看不见跨车重复），根本判别不了这种情况。

**处置**：本轮的正确定名是**描述性机制发现**，不是"完整按预声明执行的比较实验"，更不能写成"相对原保守域的正式比较优势"。`ANCHOR_SCOPE_ESTABLISHED` 一词今后只描述"作用域成立"。

## 3. 合成事务实验：保留什么、撤销什么

`REPORT.md` §3 的三条机制现象保留；但它的**资格上限**必须写明，这是本轮降级的主要理由：

- 运行时是 `StrictOnlineEnv(..., coldchain_contract=None, booking_horizon=None)`；
- 控制器是手写 `ScriptedCAMR`，逐字回写存储的尾，**不做任何重优化**；
- **没有**接拒承诺账（无 `on_reveal` 接拒记录）、**没有** C0 能耗预算、**没有** `certify_plan` 式硬认证。

| 它**支持** | 它**不支持** |
|---|---|
| 当前**引擎接口**存在事务缺陷：合法后缀写不进在途车；揭示静默清空队列；无前缀对账会重复服务 | CAMR 在真实冷链合同下可安全安装；回写的顺序有任何质量；现在应当去改引擎 |

**处置**：结论改写为"执行事务资格**未建立**"，而非"已验证需要持久化工程"。也正因为如此，本轮明确**不建议**现在改正式引擎。

## 4. 复核过程中新确认的三条事实（对下一包直接相关）

1. **保存的公开视图是带完整 `ColdChainContractV2` 的**（`p2_pos01_reference.py:253`），因此 `certify_route` 的温度等级门（`action_contract.py:178-183`）在探针里**是开着的**；但 `supported_temp_classes=(0,1,2)` 覆盖全部等级，这道门在本数据上**恒真**。
2. **能耗与品质不是"没跑"，而是"跑不了"**：`certify_route` 无条件返回 `projected_energy_kwh=None` / `projected_quality_loss=None` / `projected_thermal_margin=None`（`action_contract.py:171-173`）。任何要声称冷链物理可行性的包，都必须**新增**这条投影路径，不能靠重跑现有函数。
3. `route_check` 只判容量（`view.capacity - CAPACITY_MARGIN`）、`arrival <= tw_end`、返仓；**不判 tw_start**（早到等待合法）。仓库既有的三套容量容差口径原样继承。

## 5. 下一包：P2-C-CAMR-STATIC-01

草案已按裁定的七要素写成，状态 **`DRAFT_NOT_AUTHORISED`**：
[`docs/当前规划/P2工作包/P2-C-CAMR-STATIC-01_静态CAMR与非学习重构机制对比_DRAFT_NOT_AUTHORISED.md`](../../docs/当前规划/P2工作包/P2-C-CAMR-STATIC-01_静态CAMR与非学习重构机制对比_DRAFT_NOT_AUTHORISED.md)

> **该草案已按授权前复核修订为 R2**（唯一假设收窄为固定 mask 下的 repair 差异；SGBS 单输出预冻结为 `completions[0]`；撤销"单一通道即 No-Go"；模型身份改指 `FrozenBackbone.insertion_scores()`；补齐共享合法动作接口与预算口径；分析单元明确为 19 个相位–mask；阻断分阶段；新增止损线）。修订记录见草案 §附录 A。**仍未授权、仍未执行。**

两个**独立阻断**（我不把它们合并成一个风险条目）：

1. **C0 预算 `B` 尚未正式裁决**。已登记的 `B_ref` **不能**自动替代正式批准的硬预算；只要 `B` 未落定，"通过冷链物理认证"就没有可判定的判据。
2. **HIST 未来支持域问题未解决**。它决定"用哪些未来信息构成对照"是否合法；不解决就会把未来信息泄漏成方法收益。

授权边界：即使 CAMR-STATIC-01 通过第一层 Go，也**不自动**获得完整认证、真实续程或训练授权；它只买到"值得申请更严格的参考物理检查与隔离事务实现"这一步。

## 6. 不变项（本轮未触，继续封存）

`Gate A FAIL`、`Q3-S3 NO-GO`、`GTMIN FAIL`（永久）、`V-10 FAIL`、`L1 BLOCKED`、`L2/L3 NOT_AUTHORISED`、`FULL_INSTALLABILITY_NOT_AVAILABLE`、`stable_online_gain=False`、`t_min=0`、正式测试种子 `20260927`/`20260928` **SEALED**、`PRODUCTION_NOT_QUALIFIED`。

路线表维持：价值头训练 `NO-GO`；SAA 替换 `暂停`；`COMMIT-01` `暂缓`；`GRAN-01` 仅次级消融；`ANCHOR-01` 结论保留；`CAMR-STATIC-01` **待单独授权**。

## 7. 本轮产物

| 文件 | 角色 |
|---|---|
| [ERRATA_01.json](ERRATA_01.json) | 机器可读的范围更正：4 条更正、3 条新确认事实、可引用/不可引用清单、源码 sha256 |
| `DISPOSITION.md`（本文件） | 对复核裁决的逐条处置 |
| `research/p2_c_anchor_01_20261019/*`（其余） | **未改动**，含 `REPORT.md` 原文 |

引用纪律：`REPORT.md` 里的数字照旧可用（16/16 复核、289/216/58/15、152 候选、3 个锚点后首动作、合成 4 臂），**但不得**再用"不付合法性代价""锚点冻结失败 0 证明冻结无代价""152 个候选 = 152 个学习选择"这三类表述，改用 `ERRATA_01.json` 给出的收紧句。
