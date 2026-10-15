# A1_D3_L1_MECHANISM_DIAGNOSTIC_PREREGISTRATION.md

- 轮次：`A1-D3-L1-MECHANISM-PREREG`（**预登记 + 执行前资格检查**，不含真实 F−D-Geo 运行）
- 状态：`PREREG_DRAFT_FOR_USER_SIGNATURE`；其中**状态来源一项为 `STATE_ASSET_GAP`（阻断）**
- 授权边界（用户裁定）：允许只读盘点、F 加载闭包核对、共同 legal 接口设计、七要素预声明、
  运行上限/停止规则/身份证据；**不允许**启动真实 F−D-Geo 比较、看分歧挑状态、改权重、E2 训练、L2/L3、正式在线批次
- 不改变：D3 G0、D8、E2 原有资格；I3 正式负结果；`stable_online_gain=False`

---

## 0. L1 的唯一研究问题

> 当 F 用真正训练后的 MaskCO 权重评分、D-Geo 用 `−Δd` 评分时，在**完全相同的合法动作集合**中，
> 它们是否产生**不同的可执行修复选择**？

L1 只回答"是否产生**可归因的动作分歧**"。L2 才回答"分歧是否改善可认证决策质量"，L3 再回答
"能否增加真实 strict-online 效用"。三层不可跳级；L1 的分歧**不得**被追认为增益或显著性证据。

---

## 1. 七要素预声明

### 要素① 状态来源 —— **`STATE_ASSET_GAP`：L1 暂不可执行**

只读盘点结果（证据：`A1_D3_L1_STATE_AND_F_ASSET_AUDIT.json`）：

| 盘点对象 | 结论 |
|---|---|
| `results/m0_scale/*/per_state.json`（14 个资产；13 个 32 条 + `swap_fixed_train_4s` 4500 条） | **索引 + 标量**：`inst`、`event`、`J0`、各臂 `J` 标量、（`swap` 另有 `clock`、`P0_feasible`） |
| 28 个已登记快照字段的覆盖率 | 历史资产中**仅 `clock` 出现**，其余 27 个字段 **MISSING** |
| `mpre_reinforce_s42_clean/state_manifest.json` | 仅 `{inst_dist, inst_event(256 对索引), s_TRAIN}`，**不含状态内容** |
| 快照机制 | `scripts/simulation/recourse_snapshot.py` **存在**（sha `0af3535a…`），提供 `capture_recourse_snapshot` / `restore_recourse_snapshot` / `resume_from_snapshot` / `snapshot_state_hash` / `validate_snapshot_against_env`，schema `recourse_snapshot_v2` / `recourse_snapshot_v3_cc` |
| 已登记判断（`D3_STATE_RECOVERABILITY.json`） | `index_only_recovery_requires_a_new_execution`；`can_the_32_states_be_recovered_from_the_sealed_asset_alone=false`；`is_that_route_read_only=false`；`authorised_this_round=false` |

**判定**：不存在任何完整状态快照；`per_state.json` 的效用标量不能重建合法动作集合。
按用户规则，本轮**停在 `STATE_ASSET_GAP`**：不擅自重放或重跑历史事件，不凭结果反推状态，
不临时挑选容易分歧的事件。L1 执行前必须先另行授权一份**确定性重采集预登记**，其最低要求：

1. 在登记的 split/数据/代码身份下重跑采集，每个决策点**捕获 `recourse_snapshot`（v2 或 v3_cc）**并落盘；
2. 要求 `(inst, event, J0)` 三元组**逐位复现**（`J0` 仅作校验，不用于反推状态）；
3. 冻结状态清单、顺序、空池处理与恢复失败处理；
4. 关闭契约/画像身份缺口；
5. 全部失败状态保留在分母中，单列原因。

**一旦重采集被授权完成**，L1 状态要求为：可由明确来源身份恢复；恢复过程不读取决策时刻尚未揭示的订单信息；
同一状态由 F 与 D-Geo 共用；不得把已用于开发的状态说成独立科学验证数据（L2 必须另用事前登记的新鲜状态）。

### 要素② 唯一比较（单因素）

```
mask_scope      = B                      # 与 mask_candidate_pool 的既定作用域一致
eps_rel         = 0.0
eps_abs         = 0.0
tie_break       = legal_order_first      # 主规则；canonical_key 仅作审计列

F:      s_F(a) = restored MaskCO logits        a_F = first argmax_{a in legal} s_F(a)
D-Geo:  s_D(a) = -incremental_distance(a)      a_D = first argmax_{a in legal} s_D(a)
```

两臂**必须共享**：初始状态、B 掩码、有序候选枚举器、候选顺序、可变车辆集合、合同与信息边界。
`canonical_key`、规范排序位置、精确同分集合**允许记录**，但**不得**用于改写主对照的选择结果，
也不得替换 F 的历史 `first argmax` 语义。

**分叉语义**（用户裁定）：两臂一旦选择不同动作，后续部分计划不同，下一步 `legal` 可能自然不同——
这是策略作用的结果，**不属输入身份不匹配**。机械可比性只在**共同初始状态与相同部分计划**处检查，
记录**首次分叉位置**；分叉之后各自记录自己的 legal 指纹，**不得**强制继续一致。

### 要素③ 机制判定（只输出可复核的分歧清单）

必须区分并分别记录：

- `first_step_divergence`：首步动作不同（六字段动作 + canonical_key + 分叉位置）；
- `full_repair_completed`：该臂是否完成完整修复（**不含**认证）；
- `certified_executable_plan`：是否存在**通过认证的可执行完整计划**。

典型反例（必须如实记录，不得合并）：两臂首步不同，但 F 在第二步 `score_len` 回退 →
只能记为**首步分歧诊断**，**不能**算作"F 成功生成了不同的可执行完整计划"。
L1 **不要求**统计 CI 或效用胜出门槛；L1 结果**不得**被追认为 L2 显著性证据。

### 要素④ 日志（至少冻结以下记录组）

| 记录组 | 必需字段 |
|---|---|
| 状态身份 | `day_id`、`inst`、`event`、`state_hash`、`reveal_time`、可见信息指纹 |
| 模型身份 | F checkpoint SHA、**backbone SHA**、训练配置、源码 SHA |
| 共同条件 | `mask_hash`、`mutable_ids`、`contract_sha256`、`legal_order_hash` |
| 逐步动作 | 两臂 logits、完整**六字段**动作、`canonical_key`、分叉点 |
| 执行结果 | `failure_reason`、`n_steps`、候选计划身份、认证状态 |
| 资源 | 实际墙钟、枚举/评分/认证/求解次数、超时与回退原因 |

JAX 首次编译与模型初始化**单独计时并保留**；L1 可区分初始化与评分阶段耗时，
但**不得**据此声称满足 L3 的完整 8+2 秒在线发布门。

### 要素⑤ 失败语义与停止规则（fail-closed）

超时、权重恢复失败、状态不可恢复、评分不合法（长度/非有限）、身份不匹配 —— **全部保留在完整分母中**并单列原因：

- 不得把它们当成"F 与 D-Geo 选择一致"；
- 不得把未取得的结果记作"F 落后"；
- 超时以 `timeout` / `censored` 表示，**不剔除状态、不追加运行直到出现有利结果**；
- 认证失败、空池（`no_legal`）一律如实保留。

### 要素⑥ 预算与资源门（执行前冻结）

- 状态数量、每状态调用次数、总评分次数、墙钟上限：**首次运行前冻结**；
- REV3 的 `PENDING` 值**不得**被理解为无限预算；
- 既有 D3 的 10.0 s 单决策口径**不得**在结果产生后临时放宽；
- 若只做离线加载诊断，**不得**宣称通过完整在线时限。

### 要素⑦ 源码与资产身份

| 类别 | 冻结值 |
|---|---|
| D3 源码身份 | 冻结树 `freeze_tree_5273739d`（24 文件，逐文件 sha256；`git archive HEAD` 复现 24/24） |
| F 权重 | `mpre_reinforce_s42_clean/model.ckpt` sha256 `f5acf83f…903a7f5`（实测一致） |
| F backbone | `MASKCO_code/ckpts/cvrp100.ckpt` sha256 `03615b08a55125a8…`（**本轮新登记**；训练期身份未证，见 F 契约 §3） |
| 训练身份 | `training_identity.json`（`src_sha256` 含 `train_mpre_reinforce.py`、`mpre_trained.py`、`repair_state.py`、`strict_online_env.py`、`action_contract.py`、`coldchain_contract.py` 等）、`summary.json`、`state_manifest.json` |
| 数据 | `data/m0_scale/dcc_50_r1_edod05_{train,cal,dev_check}_teacher.npz` + `M0_TEACHER_MANIFEST.json`（sha256 见审计 JSON） |
| 快照机制 | `scripts/simulation/recourse_snapshot.py` sha256 `0af3535a…` |
| 明确排除 | **并行 `CC_Compare` 工作不属于 D3 源码身份**（`9d4e851` 等提交不得计入本轮身份） |

---

## 2. 本轮已完成 / 未完成

**已完成（只读）**
1. 状态资产只读盘点 → `STATE_ASSET_GAP` + 重采集最低要求（要素①）；
2. F 加载闭包审计 → 正确加载路径、pickle 结构、任务参数非零证据、**backbone 身份缺口**（`A1_D3_L1_F_LOADING_CONTRACT.md`）；
3. 七要素预声明草案、失败/停止规则、身份清单（本文档）；
4. 机器可读审计：`A1_D3_L1_STATE_AND_F_ASSET_AUDIT.json`。

**未完成 / 未授权**
1. 真实 F−D-Geo 比较（**本轮明确不授权**）；
2. `score_actions` 真实前向与重复前向一致性（L1 执行前置资格门，待授权）；
3. 确定性重采集（需另行授权独立预登记）；
4. E2 训练、L2/L3、AT2B-2b、D8 安装通道。

## 3. 执行前置门（全部满足才可申请 L1 正式运行）

- [ ] 用户签署本预登记（含要素①的 `STATE_ASSET_GAP` 处置方式：先授权重采集 or 先授权基于重放恢复）
- [ ] 状态清单冻结且每个状态带 `state_hash` / 可见信息指纹
- [ ] `F_LOADED_AND_SCORED` 四条判定门全部通过（含 backbone 身份裁定）
- [ ] 契约/画像身份缺口关闭
- [ ] 预算、调用次数、墙钟上限与超时处理已冻结
- [ ] 独立只读复核：规格可机械执行

## 4. Claim limits

1. 本文档是**规格**，不是结果；`STATE_ASSET_GAP` 未解除前 L1 不可执行。
2. L1 只输出机制差异清单；不得表述为算法优势、学习增益或在线效用。
3. F 的 backbone 训练期身份未证；即使 L1 通过，也只能声明"工程验证 + 行为差异"两类结论。
4. 复用已用于开发的状态只支持"在这些状态上两评分器机制有何差异"，**不是**独立科学验证；
   L2 必须另用事前登记的新鲜状态。
5. I3 正式负结果（配对差 −144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、
   D3 G0 未通过、D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain=False` **全部不变**。

---

## 5. 追加记录（2026-10-14，用户裁决后；**不改写上文任何判据**）

本追加记录不删除、不改写 §1 要素①中"`(inst, event, J0)` 逐位复现"这一判据原文。该判据**原样保留**，
但其适用范围被明确限定：

### 5.1 要素①的两条分支（用户裁决：选择 A2 前瞻性确定性采集）

| 分支 | 主判据 | 状态 |
|---|---|---|
| **A1 历史状态复现** | 历史执行策略/数据/随机性/合同身份一致 **且 `(inst,event,J0)` 逐位复现** | **未选**；§1 要素①中该判据仅对本分支有效 |
| **A2 前瞻性确定性采集** | 新快照的 `capture→restore` 一致性、`state_hash` 稳定、`resume_parity`、strict-online 因果校验 | **已选**；规格见 `A1_D3_L1_STATE_RECOLLECTION_PREREGISTRATION.md` |

要点：**"确定性重采集" ≠ "精确恢复历史状态"**。A2 分支**不要求**新状态的 `J0` 等于旧结果，
也**不**声称旧的 32 条 (inst,event) 已可恢复；`J0` 在 A2 中仅作审计列。

### 5.2 要素①的处置（由 `STATE_ASSET_GAP` 变为"待重采集解除"）

用户批准**设计**独立的确定性重采集协议（不选直接恢复历史结果的路线）：本轮只交付可签字规格，
**未**执行采集；实际采集须先完成事前冻结（状态生成策略、开发日来源、采集位置、快照 schema、
恢复与因果验证、失败分母、运行预算、身份清单）。**仍不授权** F−D-Geo L1 运行、E2 重训、L2/L3 与最终测试。

### 5.3 要素⑦的 backbone 身份（用户裁决②）

- 当前文件登记为 **`CURRENT_BACKBONE_CANDIDATE`**：`MASKCO_code/ckpts/cvrp100.ckpt`
  sha256 `03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db`；
- **不接受**"当前文件即训练期文件"的无证据假设；溯源结果与残差假设见
  `A1_D3_L1_BACKBONE_PROVENANCE.md` / `.json`（本轮新增命令记录证据，标签仍待用户批准）；
- 标签未获批准前，任何以该文件重建的评分器必须标注
  `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED`，
  且**不得**用于正式学习贡献归因。

### 5.4 执行前置门的更新

原 §3 前置门新增两项：**（7）** 状态重采集规格获签字且采集完成并通过 §5.1 的 A2 判据；
**（8）** backbone 身份标签获得用户裁定。其余前置门不变。

### 5.5 研究主线（不变）

可恢复的共同状态 → 真实 F 评分 → 同掩码 F−D-Geo 动作归因 → L2 合法决策质量 → L3 在线效用。

### 5.6 追加记录之二（2026-10-14，用户复核后）

- **裁定标签**：任务 A 方法路线 `COLLECTION_METHOD_APPROVED`，**执行版预登记 `EXECUTION_PREREG_NOT_FROZEN`**
  （待 R1/K、开发日、预算冻结）。执行参数补全件：`A1_D3_L1_RECOLLECTION_EXEC_PARAMS.md`。
- **backbone 标签不批准提升**：原建议的 `TRAINING_BACKBONE_VERIFIED_BY_COMMAND_RECORD` **撤回**；
  改标 `BACKBONE_PATH_DOCUMENTED_IN_SCRIPT` / `TRAINING_EXECUTION_BINDING_UNVERIFIED`，
  保留 `HISTORICAL_BACKBONE_SHA_UNVERIFIED`。理由：`_run_mpre_seed.sh`（09-23 08:41）**晚于**
  s42-clean 运行（09-22 11:11），见 `A1_D3_L1_BACKBONE_PROVENANCE_REVIEW_OPINION.md`。
  因此 §5.3 中"标签仍待用户批准"的表述以本条为准：**该建议标签已被否决**。
- §1 要素① 中"运行前可冻结 `day_id` × **事件位置序号**，实际 `event_id` 采集后按预冻结规则物化登记"这一措辞修正，
  已写入 `A1_D3_L1_RECOLLECTION_EXEC_PARAMS.md` §1；**不得**声称完整 `event_id` 清单在运行前生成。
