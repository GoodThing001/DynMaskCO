# A1-D3-L1：F 推理加载契约审计（只读，工程准备）

- 轮次：`A1-D3-L1-MECHANISM-PREREG`
- 性质：**加载闭包审计 + 契约冻结**，不是 F−D-Geo 比较，不产生任何增益结论
- 依据源码：`scripts/training/train_mpre_reinforce.py:309-314`（保存）、
  `scripts/models/mpre_trained.py:104-140`（`MpreTrainedModel` / `score_actions`）、
  `scripts/simulation/mtrained_replanner.py` 的 `load_trained_model`
- 机器可读证据：`A1_D3_L1_STATE_AND_F_ASSET_AUDIT.json` 的 `F_identity`

## 1. 用户指出的风险：**成立**

`mpre_trained.py` 的 `MpreTrainedModel` = `backbone(CVRP512) + c_adapter + delta_head`，两个残差模块
最后一层**零初始化**（`zero_last=True`）。`load_mpre_trained(cvrp_ckpt, F_NODE, F_ACTION_EXPLICIT, seed)`
只按 CVRP backbone 构造模型，**不含任何冷链任务训练参数**。因此：

> 只调用 `load_mpre_trained()` = 得到一个"未训练的冷链适配器"模型，其评分 ≠ 训练后的 F。

实测佐证（本轮只读探针，本地 flax 0.10.4 / jax 0.5.0）：

| 组 | 叶子数 | 参数量 | 全零叶子 | max\|值\| |
|---|---|---|---|---|
| `backbone`（含 decoder / feature2logit） | 71 | 21,657,600 | 0 | 1.4156 |
| `c_adapter` | 6 | 530,944 | 0 | **0.7194** |
| `delta_head` | 6 | 207,617 | 0 | **0.2422** |

`c_adapter` / `delta_head` 的绝对值非零 → 任务训练确实改变了这两个残差模块 → 不做 `train_state`
恢复的加载路径会得到一个**不同的评分器**（delta ≡ 0）。

## 2. 正确加载闭包（L1 唯一允许路径）

`model.ckpt` 是 **pickle**，内容为
`{'train_state', 's_TRAIN', 'F_NODE', 'F_ACTION_EXPLICIT', 'num_steps', 'batch_states', 'K', 'seed'}`，
其中 `train_state` 类型为 `flax.nnx.statelib.State`（83 叶子）。
**冻结的 encoder 部分不在该 pickle 内**（`partition()` 把它放进 `frozen_state`，未落盘），
所以必须由 CVRP backbone checkpoint 重建后合并：

```
backbone, cfg, _ = load_cvrp_model(CVRP_CKPT)                       # 仅用于身份核对
model, cfg, _    = load_mpre_trained(CVRP_CKPT, F_NODE=10, F_ACTION_EXPLICIT=20, seed=<训练期 seed>)
gd, _, frozen_state, _ = partition(model)                           # 结构 + 冻结编码器
data = pickle.load(open(MODEL_CKPT,'rb'))
F = nnx.merge(gd, data['train_state'], frozen_state)                # ← 唯一合法的 F
```

L1 冻结契约（全部必须在运行前核对）：

| 项 | 冻结值 / 要求 |
|---|---|
| `MODEL_CKPT` | `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt`，sha256 `f5acf83f…903a7f5`，89,592,616 B（**实测一致**） |
| `CVRP_CKPT` | `MASKCO_code/ckpts/cvrp100.ckpt`，sha256 `03615b08a55125a8…`（**本轮新登记**，见 §3 缺口） |
| 训练元数据 | `num_steps=1000, batch_states=8, K=4, seed=42, s_TRAIN=0.1334134797781528, F_NODE=10, F_ACTION_EXPLICIT=20`（pickle 与 `summary.json` 一致） |
| 结构重建 | `partition()` 分支与训练期一致；`train_state` 叶子 83（backbone 71 / c_adapter 6 / delta_head 6） |
| 特征构造 | `extract_repair_state_v1`（`F_NODE=10`, `F_ACTION_EXPLICIT=20`），训练期同款可见节点索引 / padding / timestep |
| 评分接口 | `MpreTrainedModel.score_actions(raw_3d, node_valid, node_feats, timestep, adjmat, cust, pred, succ, action_feats) -> (scores [B,M], internals)`；缓存路径 `score_from_H` |
| 选择语义 | 历史语义：`first argmax`（首个最大值），**不替换模型、不重训、不改 tie 规则** |

## 3. 本轮发现的**身份缺口**（必须写进 L1 预登记）

1. **backbone 身份未登记**：`training_identity.json` 顶层只有
   `data_sha256 / model_ckpt_sha256 / src_sha256 / state_manifest / summary`，
   `summary.json` 也只有 `K/batch_states/final_loss/loss_hist/n_states/num_steps/s_TRAIN/seed`
   —— **训练时使用的 `--cvrp-ckpt` 路径与 SHA 均未记录**。
   由于冻结 encoder 必须由该文件重建，backbone 的 SHA 是 F 身份的必要组成。
   本轮登记 `03615b08a55125a8…`（当前本地文件），但**只能声明"与当前文件一致"，
   不能声明"等于训练时的那一个"**——除非补充训练期证据。L1 执行前必须裁定这一点。
2. **契约/画像身份缺口**：先前 `D3_STATE_RECOVERABILITY.json` 已记录
   "the contract and profile identity gap must be closed first, else the run cannot claim identity"。
   本轮未关闭该缺口。

## 4. `F_LOADED_AND_SCORED` 判定门（未执行，属 L1 执行前置）

用户给定的四条必须同时满足，执行时逐条落记录：

- [ ] `MODEL_CKPT` SHA 与登记值一致（**已满足**），结构与 backbone 身份可追溯（**backbone 缺口待裁定**）
- [ ] 训练状态恢复成功（`nnx.merge` 后参数结构与特征维度匹配；叶子数/形状与训练期一致）
- [ ] 在**同一可见状态 + 同一有序 legal 输入**下，评分向量的长度、有限性、重复前向逐位一致，符合冻结判据
- [ ] 动作选择沿用历史 `first argmax` 语义

**本轮已完成**：SHA/结构/元数据/pickle 组统计（只读，未构造模型、未前向、未比较）。
**本轮未执行**：`score_actions` 的真实前向、重复前向一致性、以及任何 F−D-Geo 评分比较
（等待 L1 正式授权）。

## 5. 两类结果必须分开（用户裁定）

| 结果 | 允许的结论 |
|---|---|
| checkpoint 加载成功、评分稳定可复现 | F 的模型接口获得**工程验证** |
| F 与 D-Geo 在共同 legal 上产生分歧 | 当前训练权重具有**可辨识的评分行为差异** |
| （均不） | 都不能单独证明学习使效用提高；任何"任务训练/预训练带来独立增益"的主张仍需合格 E2 |

## 6. Claim limits

1. 本轮只证明"pickle 内存在非零任务参数、且正确加载路径需要 `train_state` 恢复"；**未证明** F 的在线/离线效用。
2. 探针在**本地**（jax 0.5.0 / flax 0.10.4）读取 checkpoint 结构与元数据；未构造模型、未前向，
   因此不构成"评分接口已验证"。
3. backbone SHA 只与当前本地文件一致，训练期身份未证（§3.1）。
4. E2 不存在，不阻止 L1 机制诊断，但阻止任何学习增益主张；I3 正式负结果与 `stable_online_gain=False` 不变。

---

## 7. 追加记录（2026-10-14，用户裁决②之后；不改写上文）

- §2 表中 `CVRP_CKPT` 一行原记为"本轮新登记"；现按用户裁决**改标为
  `CURRENT_BACKBONE_CANDIDATE`**，sha256 `03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db`。
  **不登记为 `TRAINING_BACKBONE_VERIFIED`**：用户不接受"当前文件就是训练期文件"的无证据假设。
- 本轮新增的溯源证据（服务器训练命令记录 `_run_mpre_seed.sh` 显式指定
  `--cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt`，且该机器仅此一份 `cvrp*.ckpt`）见
  `A1_D3_L1_BACKBONE_PROVENANCE.md` / `.json`；建议标签
  `TRAINING_BACKBONE_VERIFIED_BY_COMMAND_RECORD` 仍需用户批准。§3 的两条身份缺口结论**不变**。
- 在标签获批前，§4 的 `F_LOADED_AND_SCORED` 即使通过，也只能表述为
  `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED`，
  不得称为"历史训练 F 的精确复现"，不得用于正式学习贡献归因。
