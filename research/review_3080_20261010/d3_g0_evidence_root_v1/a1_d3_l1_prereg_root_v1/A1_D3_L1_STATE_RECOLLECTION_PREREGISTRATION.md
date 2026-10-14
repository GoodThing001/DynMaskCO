# A1-D3-L1 确定性状态重采集预登记（前瞻性采集分支）

- 轮次：`A1-D3-L1-MECHANISM-PREREG` / 任务 A
- 状态：`RECOLLECTION_PREREG_DRAFT_FOR_USER_SIGNATURE`（**只交付可签字规格；本文件不授权、也未执行任何重采集**）
- 用户裁决：选择**前瞻性确定性采集**，不选"恢复历史结果"路线；实际重采集须先完成事前冻结
- 上位文档：`A1_D3_L1_MECHANISM_DIAGNOSTIC_PREREGISTRATION.md`（七要素；本次追加记录见其 §5）
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；`stable_online_gain=False`

## 0. 两条分支必须分开（本轮协议澄清，追加记录，不删除旧判据）

| 分支 | 判据 | 用途 / 本轮取向 |
|---|---|---|
| **A1 历史状态复现** | 历史执行策略、数据、随机性、合同身份一致，且 `(inst, event, J0)` **逐位复现** | **未选**。旧预登记把它列为"最低要求"，该判据**原样保留**，仅适用于本分支 |
| **A2 前瞻性确定性采集** | 同一新快照的 **capture → restore 一致性**、`state_hash` 稳定、恢复后行程与后缀一致、strict-online 合法性 | **本文件所选**。不假装旧的 32 条状态可恢复，**不要求新状态的 J0 等于旧结果** |

理由：历史资产只有 `(inst,event)` 与结果标量（`STATE_ASSET_GAP`，见
`A1_D3_L1_STATE_AND_F_ASSET_AUDIT.json`），要恢复一个决策时刻至少还需要车辆位置、剩余路线、
接单承诺、服务历史、可见订单、冷链状态与规划器状态。与其从标量反推，不如按
`scripts/simulation/recourse_snapshot.py`（sha `0af3535a…`）的既有结构**新建一批身份完整的诊断资产**。

## 1. 状态生成策略（与 F/D-Geo 结果无关，事前固定）

1. **预先指定、共同的状态生成策略**：状态集合由登记的基线策略（JF1-H-F 严格在线 rollout）在
   **事前指定的开发日 × 事件位置**上产生；不运行 F 或 D-Geo 之后挑状态。
2. **禁止**：先跑两臂、再挑选分歧明显的状态进入集合；禁止按分歧/增益/损失筛选；
   禁止临时替换难产状态；禁止依据结果追加运行。
3. 采集位置（事件位置的选取规则，事前写死，二选一并冻结）：
   - 规则 R1：每个开发日的**前 K 个揭示事件**（K 冻结）；
   - 规则 R2：每个开发日按 `reveal_idx` 的**固定步长/固定分位**取 M 个事件（步长/分位冻结）。
   若某日事件数不足，记录 `insufficient_events` 并保留该日在分母中，不替换、不补采。
4. 状态清单（日 × 事件）必须在运行前**完整列出并冻结**（写进预登记的清单文件，含 sha256）。

## 2. 开发日来源

- 数据来源：登记的 `data/m0_scale/dcc_50_r1_edod05_{train,cal,dev_check}_teacher.npz`
  （各文件 sha256 见审计 JSON 的 `data_identity`）。**不使用 TEST 数据**。
- L1 允许使用**已参与开发**的状态（用户裁定①），但必须在资产中标注
  `previously_used_for_development = true`，且**不得**被表述为独立科学验证数据；
  L2 必须另用事前登记的新鲜开发日。
- 开发日选取规则事前冻结（例如"`dev_check` 全部 16 个实例 × 规则 R1"），不得事后增删。

## 3. 快照 schema（完整状态身份）

使用 `scripts/simulation/recourse_snapshot.py` 已登记的结构（`recourse_snapshot_v3_cc` 冷链版优先）：

- 捕获接口：`capture_recourse_snapshot(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask, all_customers=None)`
- 必含字段（与既有 38 字段清单一致）：`clock`、`served_mask`、`served_route(_len)`、`mutable_suffix(_len)`、
  `needs_replan`、`replan_reason`、`replan_count`、`committed_next/arrive/finish`、`dispatch_time`、
  `vehicle_node/ready/load/status`、`services_*`、`visible_mask`、`customer_universe`、`num_nodes`、`num_vehicles`、
  `force_suffix`、`replanner_state`、`schema_version`
- 附加速率元数据：`inst`、`event`、`reveal_time`、可见信息指纹、`state_hash = snapshot_state_hash(snapshot)`
- **审计用完整环境信息可与决策可见信息分离存放**：快照可为复核保存完整环境；但见 §4。

## 4. 恢复与因果验证（A2 分支的主判据）

每个状态必须通过以下全部检查，否则按 §6 记失败，**不替换状态**：

| 检查 | 判据 |
|---|---|
| `capture_restore_identity` | `restore_recourse_snapshot(snapshot)` 后再 `capture` 得到 `state_hash` 相同 |
| `state_hash_stable` | 同一快照重复计算 `state_hash` 逐位一致 |
| `resume_parity` | 从快照恢复后继续 rollout，得到的后缀与原始 rollout 后缀**逐位一致** |
| `validate_snapshot_against_env` | 既有校验器通过 |
| `strict_online_causal` | 传给 F/D-Geo 的特征、节点归一化（`coord_normalize_visible`）与候选枚举**不读取未揭示订单**；未揭示订单在特征层被屏蔽 |
| `legal_nonempty_or_recorded` | 合法动作集非空；若为空则记录 `empty_pool` 并保留在分母 |

`J0` 在本分支**只作为审计列**（可记录、可供交叉参考），**不作为准入判据**。

## 5. 身份清单（运行前逐项核对，任一不符即 `not_qualified`）

- 数据：§2 的 NPZ sha256 + `M0_TEACHER_MANIFEST.json` sha256
- 合同 / profile：`results/o0cc/scale_v2/objective_profile.json` 与所用冷链合同（**须显式指定并记录 sha256**；
  上游审计记录的"契约/画像身份缺口"必须先关闭）
- 代码：`scripts/simulation/recourse_snapshot.py`、`strict_online_env.py`、`jf1h_repair.py`、
  `dynmaskco_cc_context.py`、`action_contract.py`、`repair_state.py`、采集器自身（全部 sha256）
- 运行环境：python / jax / flax 版本（评分接口与快照结构对版本敏感）
- F 侧（仅登记，不参与本任务）：`model.ckpt` sha `f5acf83f…903a7f5`；
  backbone 身份标注见 `A1_D3_L1_BACKBONE_PROVENANCE.md`（**当前为候选 + 命令记录绑定，未经用户批准前不得写作已闭合**）
- **明确排除**：并行 `CC_Compare` 提交不属于 D3 身份

## 6. 失败分母与失败语义（fail-closed）

- 事前冻结的**全部状态**都留在分母中；捕获/恢复/校验失败的状态**不得**被剔除或替换；
- 失败原因分类固定：`capture_failed`、`restore_mismatch`、`state_hash_unstable`、`resume_divergence`、
  `causal_violation`、`empty_pool`、`insufficient_events`、`timeout`、`identity_mismatch`；
- 禁止"追加运行直到通过"；禁止把失败状态记作"F 与 D-Geo 一致"。

## 7. 运行预算（执行前按实测冻结）

- 状态数量上限、每状态墙钟上限、采集/恢复/验证各自的调用次数上限，全部**首次运行前冻结**；
- 上游 `PENDING` 值**不得**被解释为无限预算；
- 先做一次**计时资格运行**（少量状态）以取得真实单位成本，再冻结总预算；计时运行的状态**不进入**正式集合；
- 超时按 `timeout` 记账并保留在分母，**不**放宽上限。

## 8. 本任务的交付物与边界

**交付（本文件）**：可签字规格。**未交付、未授权**：实际重采集、快照资产生成、任何 F/D-Geo 评分。

后续（需另行授权）应产出：快照存储目录 + 清单（含逐文件 sha256）+ 每状态验证记录 +
`A1_D3_L1_STATE_ASSET_MANIFEST.json`，并据此更新 L1 七要素的要素①。

## 9. Claim limits

1. 本文件是规格，不是结果；未产生任何状态资产。
2. 前瞻性采集产生的是**新的 L1 开发诊断资产**，不是历史状态恢复；其 J0 与旧结果无关。
3. 即使采集与恢复全部通过，也只支持 L1 的机制差异诊断；L2/L3 仍需独立资格与授权。
4. 复用已用于开发的数据只支持"在这些状态上两评分器机制有何差异"，不是独立科学验证。
