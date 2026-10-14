# D2 五项资产解阻（只读阶段）汇总 — 2026-10-14

**基准提交**：`4dfbc8e…`（用户核对）· **权威登记**：`CC_Compare/P1A_D2_同掩码MaskCO归因筛查_登记_2026-10-14.md`
**性质**：只读资产清查 + 身份核验 + 能力证明。**未做任何 F−D / F−E 数值比较；未启动训练；未触碰最终测试种子。**
**原裁决**：`not_qualified` —— **保持不动**（`D2_QUALIFICATION_AUDIT.json` 未被改写）。

## 0. 结论速览

| 项 | 判定 | 依据 |
|---|---|---|
| **F：已训练 MaskCO 权重** | **`ready`** | 服务器上按记录路径找到，**SHA 与历史记录逐字节一致** |
| **D：非学习几何修复** | **`incompatible`** | 基线**存在**（regret-2 在 `cc_lns_replanner.py`）；但①登记未固定 D 语义、文档所述 swap 变体两树皆无；②`cc_lns_replanner.py` 与 `dynmaskco_cc_context.py` **相对 F 的训练身份已漂移** |
| **统一掩码池** | **`ready`** | 定义 + 调用路径齐全（F/D/训练/评价），**且找到实际执行通过证据** |
| **E2：同任务随机初始化对照** | **`missing`** | 搜索范围内无任何 E2 产物；源码证明 `--seed` **不**等于随机 backbone |
| **逐候选动作 / 真实提交字段** | **`not_verifiable`** | 按**真实 schema** 检查：两个字段在所有已定位资产中都不存在 |
| **总体** | **`not_qualified`（原裁决维持）** | 5 项中 3 项 ready，但 E2 缺失 + 两字段缺失即已阻断 |

## 1. 我上一轮 D2 审计的三处搜索偏差（已修正）

| # | v1 的问题 | v2 的修法 |
|---|---|---|
| 1 | checkpoint 搜索用**文件名关键字**过滤，历史训练器写的通用名 `model.ckpt` 会被漏掉 | 改为**按授权目录遍历**（`find results -maxdepth 6`，剪掉 `_bverify_pinned`），**不做名字过滤** |
| 2 | 掩码证据指向**不存在**的 `scripts/models/mpre_policy.py` | 改为真实模块 `scripts/simulation/dynmaskco_cc_context.py`，并列出调用点 |
| 3 | 封存 JSON 用**子串计数**（只读前 20,000 字符）判字段 | 改为**读实际 schema**（从封存文件本身探测键集合） |

> 直接后果：v1 说"磁盘上 0 个已训练 checkpoint"，**是搜索范围造成的假阴性**。用户对这一点的判断正确。

## 2. F：找回（`ready`）

```
results/m0_scale/mpre_reinforce_s42_clean/model.ckpt
  89,592,616 bytes   mtime Sep 22 11:11
  sha256 = f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5   ← 与历史记录一致
```

* 另有独立身份记录 `results/m0_scale/mpre_reinforce_s42_clean/training_identity.json`
  （`model_ckpt_sha256` 同上，并含 `data_sha256` 与 12 个 `src_sha256`）；
* 冻结的 `results/correction_validation_1009_v2/SOURCE_START.json` / `SOURCE_END.json` 也逐字记录了同一 SHA
  —— 但路径写作 `B_REPRO_CLEAN/results/...`（**镜像树**，同 SHA）。登记时须写明是哪一份。
* 同级还有 `s42_defective`（**不得作为 F**）、`s42_corrected`、`s43_clean`、`s44_clean`、`repair_s43`。

## 3. D：基线存在，但**语义未定 + 身份漂移**（`incompatible`）

存在（服务器 SHA）：
```
ae634f53…  scripts/simulation/cc_lns_replanner.py        ← 非学习 regret-2 几何重建
4eebb2a5…  scripts/simulation/dynmaskco_cc_context.py    ← 统一掩码池定义
fd38cb3c…  scripts/evaluation/run_fixed_state_quality.py
MISS       scripts/simulation/cc_lns_swap.py
MISS       scripts/evaluation/run_cc_lns_swap_fixed_state.py
```

⚠️ **新发现的阻塞（超出用户五项清单，由 schema+身份核对得到）**：把 F 的
`training_identity.json` 里的 `src_sha256` 与**当前**服务器源码对拍：

| 文件 | F 训练时 | 当前服务器 | 一致 |
|---|---|---|---|
| `scripts/simulation/dynmaskco_cc_context.py`（**定义统一掩码池**） | `9de8ceb5…` | `4eebb2a5…` | **否** |
| `scripts/simulation/cc_lns_replanner.py`（D 臂） | `7ea14c07…` | `ae634f53…` | **否** |
| `scripts/models/mpre.py` | `585f0296…` | `585f0296…` | 是 |

⇒ **"所有臂共享同一掩码池"这一前置条件，对着当前源码并不成立**。
必须先裁定：登记指的是哪种 D 语义，并**恢复/冻结训练期源码**，或**证明当前实现与训练期在固定状态上行为等价**。

## 4. 掩码池：能力齐备 **且** 有执行通过证据（`ready`）

* 定义：`dynmaskco_cc_context.py:124 mask_candidate_pool` / `:155 validate_mask_scope`；
* 调用点：`mtrained_replanner.py`（**F 臂** 206/235）、`cc_lns_replanner.py`（**D 臂** 114/151）、
  `train_mpre_reinforce.py`（**训练** 159/165）、`run_fixed_state_quality.py`（101/107）、
  `run_sgbs_fixed_state.py`、`run_headroom_census.py`；
* **执行通过证据**：`results/correction_validation_1009_v3/fixed_state/per_state.json` 逐状态记录
  `mask_audit` —— `per_customer[o]{in_decision_pool,is_committed_next,in_non_mutable_tail,in_protected_tail,in_mutable_vehicle}`、
  `decision_pool_size`、`mask_size`；其 `summary.json` 记录
  `n_states_with_out_of_pool = 0`、`n_out_of_pool_customers = 0`。

**范围限制（写进结论）**：该证据**只覆盖那一批封存资产**，不能据此声称"历史每批都通过"。

## 5. E2：`missing`，且**不能用 `--seed` 冒充**（源码已核）

`scripts/training/train_mpre_reinforce.py`：
```
line 213  backbone, cfg, _step = load_cvrp_model(args.cvrp_ckpt)          ← 预训练 backbone 从 checkpoint 载入
line 214  load_mpre_trained(args.cvrp_ckpt, F_NODE, F_ACTION_EXPLICIT, seed=args.seed)
line 259  opt_state = tx.init(train_state)                                ← 只训练 train_state，backbone 冻结
```
⇒ `--seed` 只改**新增可训练层**的初始化，backbone 与 F **共享** ⇒ 换 seed **不构成**随机初始化对照。

**搜索范围**（服务器 + 本地）：`results/m0_scale/*mpre*/*mtrained*/*reinforce*/*random*`、
`results/**/manifest.json`（深度 ≤4/5，剪掉 `_bverify_pinned`）⇒ **零 E2 产物**。
⇒ E2 **无法靠资产整理解决**，需要**另行获准的训练预声明**；在此之前任何预训练归因主张都不成立。

## 6. 逐候选动作 / 真实提交字段：`not_verifiable`（按真实 schema）

最丰富的封存资产 `results/correction_validation_1009_v3/fixed_state/per_state.json` 的**真实结构**是：

```
{inst, event, J0, mask_audit{...},
 R|Mpre|Mtrained: {det: {J,D,Q,E,fail,fail_reason,time,hash,cert_ok,
                         cert{dup,missing,extra,order_changed,protected_vehicle_changed,vehicle_changed},diag},
                   samples:[同结构...]}}
```

| 必需字段 | 现状 | 已有替代物 | 为何不可替代 |
|---|---|---|---|
| `per_candidate_executable_actions` | **不存在** | 每候选一个**计划 hash**（如 `5c28be5a88095753`）+ C0 认证对账表 | hash 标识的是计划，不是登记要求的**可执行动作身份** |
| `true_committed_differences` | **不存在** | **提交前**候选评价（哪个候选分数最好） | 分数与最终路线都是候选；只有落盘的**安装后计划**或接/拒决定才是提交证据 |

源码级独立确认：`scripts/simulation/mtrained_replanner.py` 中**没有** `installed` / `write_back` /
`committed` / `P_after` 任何持久化（不是输出子串探测，是源码检查）。
另一份 `results/m0_scale/sgbs_fixed_cal/per_state.json`（32 条）**只有标量 J**（`R/Mpre_greedy/
Mtrained_greedy/Mpre_sgbs/Mtrained_sgbs/dist_sgbs/Mtrained_greedy7`），连计划 hash 都没有。

⇒ 两个字段**不可追溯**：**不得重建、不得近似补齐**；需要新的前瞻性、预声明采集。

## 7. 范围诚实性

* 服务器清查是**有界**的：`find results -maxdepth 4/6`，剪掉 `*/_bverify_pinned`；
  **未覆盖**：扩展根以外的树、`B_REPRO_CLEAN`、登录用户无读权限的路径、未挂载的归档。
* 因此所有"未发现"只写作**"在所检索范围内未发现"**，**不写作"服务器不存在"**。
* 本地环境、服务器环境（`host=user`、`2026-10-14T15:47:54+08:00`、Python 3.10.19）、
  检索目录清单与原件 SHA 均写入 `D2_ASSET_INVENTORY.json`。
* 原始探测输出原样归档于 `raw/`（`d2_server_inventory.txt`、`d2_schema_probe.txt`、`d2_schema_probe.sh`）。

## 8. 退出条件与禁止事项（均已遵守）

已完成五项资产清查、可追溯身份核对、缺口登记 ⇒ **到此结束**。
未做 F−D / F−E 比较；未启动训练；未访问最终测试种子；未把 E1 当 E2、候选分数当动作、候选计划当真实提交；
未改写原 `D2_QUALIFICATION_AUDIT.json` 与原 `not_qualified` 裁决；未改任何冻结目录 / gate / checkpoint / 封存结果。

## 9. 下一步（需用户裁定）

1. **D 语义**：登记指的是 regret-2、纯距离逐步插入，还是 swap 增强版？若为 swap ⇒ 该实现两树皆无。
2. **身份漂移处置**：恢复/冻结训练期 `dynmaskco_cc_context.py`（`9de8ceb5…`）与
   `cc_lns_replanner.py`（`7ea14c07…`），**或**先做"当前实现 vs 训练期实现"在固定状态上的行为等价证明。
3. **E2**：单独立项、另行获准的随机 backbone 训练预声明（本轮**不启动**）。
4. **两字段**：若确需 `per_candidate_executable_actions` 与 `true_committed_differences`，
   作为**新的前瞻性预声明采集**，不得补写进旧 D2。
