# E2 同任务随机初始化对照：训练预登记草案（**不执行**）

- 轮次：`A1-D3-L1` / E2 backbone-pretraining attribution control
- 状态：`PREREG_DRAFT_FOR_USER_SIGNATURE` / **`NOT_AUTHORISED`**
- 本文件**只交付可签署规格**。它**不授权、也未执行**任何训练、任何前向、任何比较。
- 不改变：D3 G0、D8、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**
- 上位于：`D3_PREREGISTRATION_REV2.md`（D3 资格分层）、
  `D3_GEO_IMPLEMENTATION_SPEC.md`（D-Geo 主对照）、
  `D2_CLOSURE.md` / `D2_UNBLOCK_SUMMARY.md`（E2 = `missing` 的原始判定）

---

## 0. E2 要回答的唯一问题

> 在**完全相同的任务、数据、优化器、更新日程与模型选择规则**下，MaskCO 的
> **backbone 是否经过 CVRP 预训练**这一个因素，是否改变了同掩码修复决策的质量？

E2 **不**回答"MaskCO 是否有效"，也**不**替代 F−D-Geo 主对照。E2 只是把 F 臂的
**backbone 来源**从两个候选值中固定下来，从而让"学习贡献"与"预训练贡献"可分。

## 1. 已确认的事实（决定了本规格的形状，**不得**用猜测覆盖）

| # | 事实 | 来源 |
|---|---|---|
| E2-1 | E2 产物在检索范围内**不存在**（`missing`） | `D2_UNBLOCK_SUMMARY.md` |
| E2-2 | `--seed` **不等于** E2：`train_mpre_reinforce.py` 的 `--seed` 只改**新增可训练层**的初始化，backbone 仍由 `load_cvrp_model` 载入**预训练**权重，与 F 共享 | 源码核对（同上） |
| E2-3 | 因此"换 seed 当随机初始化对照"是**无效**做法，已被明确否决 | 同上 |
| E2-4 | F 的训练身份为 `training_identity.json` `src_sha256`（含 `train_mpre_reinforce.py`、`mpre_trained.py`、`repair_state.py`、`strict_online_env.py`、`action_contract.py`、`coldchain_contract.py` 等） | `A1_D3_L1_MECHANISM_DIAGNOSTIC_PREREGISTRATION.md` 要素⑦ |
| E2-5 | F checkpoint `mpre_reinforce_s42_clean/model.ckpt` sha `f5acf83f…903a7f5`（实测一致） | V1.18 运行身份 |
| E2-6 | backbone 训练期身份**未证**，标签保持 `HISTORICAL_BACKBONE_SHA_UNVERIFIED` | `A1_D3_L1_BACKBONE_PROVENANCE_REVIEW_OPINION.md` |

**E2-6 的直接后果**：即使 E2 训练完成，**也不得**据此断言"训练期 backbone 就是某个具体文件"。
E2 只能声明"在**当前** backbone 文件身份下，预训练 vs 随机的差异是什么"。

## 2. 单因素设计（本规格的核心）

```
因素      : backbone 初始权重来源
  E2-pre  : 载入 MASKCO_code/ckpts/cvrp100.ckpt（预训练）        = 与 F 相同的那一支
  E2-rand : 同一架构、同一形状，按登记的初始化器随机初始化       = 唯一被替换的对象
其余一切必须逐项相同（并逐项落盘身份）：
  数据 / split / 实例集 / context-sampling 规则
  优化器（类型 + 全部超参） / 更新步数 / batch 形状
  reward 定义（(J_vis(P0) − J_vis(P))/s_TRAIN 与 s_TRAIN 的**估计方式**）
  K 与采样数 / 训练--评价分离规则 / 模型选择规则（**同一规则、同一留出集**）
  新增可训练层（`c_adapter` / `delta_head`）的初始化：**同一个初始化器与同一个 seed**
  部署侧 adapter/界面版本身份
```

**第二个混杂必须被消除（E2-7，新登记）**：`MpreTrainedModel` 的 `c_adapter` 与 `delta_head`
是 **zero-init last layer**（初始化时得分恰等于 M-pre）。若 E2-rand 只改 backbone 而随机化了
这两层，则"预训练 vs 随机"会与"零初始化 vs 随机初始化"**同时**变化 ⇒ 违反单因素。
因此规格**强制**：两层沿用与 F **完全相同的初始化器与 seed**，只有 backbone 变。

**E2-8（新登记）**：两侧的**参数量、张量形状、前向图**必须逐张量相等（形状 + dtype + 参数名
逐项比对并落盘）。形状不同即 `not_qualified`，不得"调参让它跑起来"。

## 3. 训练身份冻结（七要素要素⑦ 的 E2 展开）

运行前必须冻结并落盘：

| 类别 | 冻结内容 |
|---|---|
| 数据 | 训练/校准/留出 NPZ 逐文件 sha256 + `M0_TEACHER_MANIFEST.json` sha256；**不使用 TEST 数据** |
| 代码 | `train_mpre_reinforce.py`、`mpre_trained.py`、`mpre.py`、`repair_state.py`、`mtrained_replanner.py`、`strict_online_env.py`、`action_contract.py`、`coldchain_contract.py`、`coldchain_state.py` 的**逐文件 sha256**（起止各一次） |
| backbone 预训练支 | `MASKCO_code/ckpts/cvrp100.ckpt` sha256（实测 `03615b08…`），标签 `HISTORICAL_BACKBONE_SHA_UNVERIFIED` **强制保留** |
| backbone 随机支 | **初始化器身份**：`jax.random` PRNG 实现 + seed + 逐层 shape 列表 + 初始化后的**参数摘要 sha256** |
| 环境 | python / jax / jaxlib / flax / numpy / optax 版本；`jax.default_backend()`；设备 |
| 新增可训练层 | 初始化器 + seed（**两侧相同**）+ 初始化后逐张量 sha256 |
| reward 标度 | `s_TRAIN` 的值与**其估计所用的样本**（TRAIN only） |
| 产物 | 每步 checkpoint 的落盘策略、最终 checkpoint sha256、`training_identity.json` |

## 4. 失败语义与停止规则（fail-closed）

- 任一身份不符、任一形状不等、任一侧 NaN/Inf 梯度、任一 checkpoint 无法重载 ⇒
  **该支 FAILED**，原件保留，**不重跑、不调参、不换 seed 再试、不挑结果**；
- 两支出**非对称**的步数或数据顺序即 `not_qualified`（不得事后对齐）；
- 训练崩溃/超时按 `crashed` / `timeout` 记账，保留在分母；
- **不得**用 E2 的任何中间结果选模型、改判据或调整 F。

## 5. 预算（执行前按实测冻结）

- 更新步数、batch 形状、墙钟上限、checkpoint 频率：**首次运行前冻结**；
- `PENDING` / `None` / 无限预算**不得**被解释为"无上限"；
- 两侧预算**逐项相同**；实际消耗（步数、墙钟、显存峰值）分别记录；
- 若只做有限步数的工程验证，**不得**把该结果表述为 E2 的科学结论。

## 6. E2 的判定与主张边界

E2 完成后**允许**声明的：
1. 在登记身份下，`E2-pre` 与 `E2-rand` 的**固定状态**修复质量差异（含逐状态配对与不确定度）；
2. 该差异**是否**可由 backbone 来源解释（单因素设计 + 形状/初始化等价的机械证据）。

E2 完成后**不允许**声明的：
1. 任何"MaskCO 有效/无效"的总体结论；
2. 任何在线效用、显著性、稳定增益结论（那是 L2/L3，须各自独立授权）；
3. "训练期 backbone 就是 ckpt `cvrp100.ckpt`"（E2-6 未证）；
4. 用 E2 替换 F−D-Geo 主对照，或把 E2 与 F−D-Geo 合并成一次比较。

## 7. 执行前置门（全部满足才可申请 E2 训练）

- [ ] 用户签署本预登记（**含 E2-7 的零初始化强制项**）
- [ ] 两侧形状/参数名/新增层初始化的**机械等价证据**已产出（可先于训练完成）
- [ ] 训练身份清单冻结（§3 全表）
- [ ] 预算按 §5 冻结
- [ ] 独立只读复核：规格可机械执行
- [ ] 平台与设备登记（服务器/本地、GPU 型号、jax/flax 版本组合）

## 8. Claim limits

1. 本文件是**规格**，不是结果；产出时 E2 训练次数 = **0**。
2. E2 的存在**不**提升 D3 G0/D8 的资格，**不**改变 I3 正式负结果与 D2 `not_qualified`。
3. backbone 训练期身份未证；E2 只能在**当前文件身份**下给出条件性结论。
4. 复用已用于开发的数据**不是**独立科学验证。
5. `stable_online_gain = False` **不变**。
