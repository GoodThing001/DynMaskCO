# P2-C-CAMR-STATIC-01：冻结 MaskCO 评分 vs 距离评分的静态 L1 机制对比

**状态：`COMPLETE`（限定执行，一次预声明批次 + 双次确定性复核）。资格 `REFERENCE_ONLY / L1`。**
未修改执行引擎或 `MASKCO_code/`，未重跑源日，未采集标签，未训练，未用 GPU，未触碰最终种子。

## 0. 一句话结论

**L1-A：GO —— 19/19 个单元上，冻结 MaskCO 评分在完全相同的公开状态、冻结锚点、固定 mask、共享合法动作空间与受控预算下，产生了与 distance-greedy 不同的合法完整计划；两次独立进程运行逐字一致，退化闸门 19/19 通过，因此差异可归因到学习评分而不是接口。**
**L1-B：全部被覆盖 —— 同一预算下的 distance-SGBS 在 19/19 个单元里都能到达 MaskCO-greedy 选中的那个计划。按预声明规则：行为差异成立，但"学习重构具有不可替代搜索价值"未获支持，停止工程与训练扩张。**

一句话读法：**评分确实改变了选择，但它选中的路线并没有跑出非学习搜索的可达范围。**

## 1. 授权与身份链

| 环节 | 内容 | sha256（前 16） |
|---|---|---|
| 授权 | GO — 限定执行，仅对应 `f4b7e65` 上的 R2.1 | — |
| **T1** 预声明冻结 | `registration.json`：唯一假设、19 个 `unit_id`、共享接口、容量口径、锚点断言、tie-break、预算、资源上限、异常分类、禁止项、停止条件 | `fb3f84948fbd257a` |
| 实现 + 合成回归 | `camr_shared.py`、`synthetic_regression.py`（9/9 通过，未加载模型） | `0eebcd0a8b4099fa` |
| **T2** 运行身份冻结 | `run_identity.json`：绑定 T1、19 个源文件哈希、合成回归回执、**加载后**的模型参数身份 | `0e4a057b0b3b58db` |
| **preflight** 退化闸门 | `preflight.json`：绑定 `run_identity_sha256`，**不含**闸门结果以外的内容；闸门本身在 T2 之后运行 | `d6dfb29fee1fb565` |
| 正式比较 | `pass_a/b_run1.json`、`pass_a/b_run2.json` | 见 §6 |
| 独立复核 | `verification.json`：12/12 通过 | — |

- 模型身份（加载后实测，非仅文件哈希）：`cvrp100.ckpt` sha256 `03615b08a55125a8…`、305,061,150 B、encoder/decoder = 512/512、input_dim = 3、`platform=cpu`、**`parameter_sha256 = 7b43ece321619b7c…`**。
- 运行时：Python 3.12.7 / numpy 1.26.4 / jax 0.5.0 / flax 0.10.4，`jax_devices=['TFRT_CPU_0']`，单 worker，无 GPU。
- T1 冻结的**文件清单**与 T2 冻结的**文件哈希**是两个节点；`run_identity.json` 明确记录 `gate_result_included: false`，闸门回执独立成 `preflight.json`，两者以 sha256 互绑（`A_identity_chain` PASS）。

## 2. 方法：唯一的差别只能是评分

| 项 | 冻结内容 |
|---|---|
| 分析单元 | **19** = 13 个相位 ×1 mask + 2 个相位 ×3 mask，全部来自 15 个 `ANCHOR_LEGAL_MULTI` 相位（SAA 9 / MYOPIC 6），身份键 `(arm, day_id, ordinal, public_identity, mask)` |
| 锚点 | `(committed_next, committed_arrive, committed_finish)` 与取货后载重逐字不变；`committed_next` 不得出现在任何后缀 |
| 共享合法动作集合 | `cand.feasible` **且** trial 全车队逐车 `route_check`；`allowed = mutable_ids ∪ {committed_vid}`，两侧同一对象 |
| 容量口径 | `route_check` 的 `capacity − 1e-4`（仓库内更严的那一个） |
| tie-break | `(primary 升序, FleetAction.action_id() 升序)`，两侧同一实现 |
| 搜索骨架 | greedy 先行，然后逐层 beam（`beam=2`、每父≤2、≤8 completions、`plan_hash` 去重）；两侧同一实现 |
| 评分 | 唯一差别：distance 侧 `-incremental_distance`，模型侧 `FrozenBackbone.insertion_scores()` |
| 预算 | Pass A 每侧 1 个贪心补全；Pass B 上述 beam 参数；单元墙钟 300 s、整包 3600 s、内存 6 GiB、单 worker |
| 输出契约 | Pass A 单决策输出是唯一主判据；Pass B 集合只作次级描述，**不得**改写 L1-A |

**退化闸门**（`preflight.json`，`all_ok=true`，19/19 equal）：在真实模型路径上执行 `insertion_scores`，然后**丢弃其分数**、改用 `-incremental_distance` 排序，结果必须逐字等于 distance-greedy。19/19 通过 ⇒ 两侧的批量、索引映射、分号号志、tie-break 与搜索骨架都不产生差异，**Pass A 的任何差异只能来自学习评分**。

## 3. L1-A：评分行为差异（主判据）

| 指标 | 值 |
|---|---|
| 可评估单元 / 注册单元 | **19 / 19**（`n_invalid = 0`，分母未缩水） |
| 两侧 `plan_hash` 不同 | **19** |
| 两侧 `plan_hash` 相同 | **0** |
| 锚点后首动作也不同的单元 | **17** |
| 计划不同但锚点后首动作相同的单元 | **2** |
| 两侧合法（保持锚点 + 通过共享参考检查） | 19 / 19 |
| 两次独立进程运行逐字一致 | 是（`E1`/`E2` PASS） |

| arm | day | ord | mask | MaskCO plan | distance plan | MaskCO 锚点后首动作 | distance 锚点后首动作 |
|---|---:|---:|---|---|---|---:|---:|
| MYOPIC | 0 | 3 | `[13,24]` | `21df49d6` | `feaa6799` | **0（返仓）** | 24 |
| MYOPIC | 0 | 10 | `[12,40]` | `8732c416` | `06d6bc5e` | **0** | 12 |
| MYOPIC | 0 | 11 | `[12,37]` | `51ac4e59` | `a81557f3` | 40 | 12 |
| MYOPIC | 0 | 11 | `[12,40]` | `711d0426` | `a81557f3` | 37 | 12 |
| MYOPIC | 0 | 11 | `[37,40]` | `ec586de7` | `a81557f3` | 12 | 12 |
| MYOPIC | 1 | 7 | `[8,16]` | `799e9894` | `d308f77e` | **0** | 8 |
| MYOPIC | 1 | 9 | `[16,35]` | `ebae675b` | `87b58ff4` | **0** | 35 |
| MYOPIC | 1 | 11 | `[14,16]` | `76de8a58` | `18dc6f24` | **0** | 16 |
| SAA | 0 | 18 | `[5,27]` | `0fd40884` | `045025d9` | **0** | 5 |
| SAA | 0 | 40 | `[26,38]` | `61b5cb42` | `e9f54a1c` | **0** | 38 |
| SAA | 0 | 44 | `[11,44]` | `b0f47e6a` | `bc081348` | **0** | 11 |
| SAA | 0 | 45 | `[11,21]` | `610d78c0` | `f43a5718` | 44 | 21 |
| SAA | 0 | 45 | `[11,44]` | `3ce8f833` | `f43a5718` | 21 | 21 |
| SAA | 0 | 45 | `[21,44]` | `18310239` | `f43a5718` | 11 | 21 |
| SAA | 0 | 62 | `[4,47]` | `97776d41` | `0557ea53` | **0** | 47 |
| SAA | 1 | 6 | `[8,16]` | `799e9894` | `d308f77e` | **0** | 8 |
| SAA | 1 | 41 | `[15,48]` | `ead8cab8` | `da19af71` | **0** | 15 |
| SAA | 1 | 48 | `[5,42]` | `a46b5331` | `11d4fa13` | **0** | 5 |
| SAA | 1 | 51 | `[27,42]` | `54fb369c` | `387d8e93` | **0** | 27 |

三点值得单独记下：

1. **差异的方向有结构**：MaskCO 在 **13/19** 个单元里让在途车就地返仓（首动作 `0`），distance 侧 **0/19** 这样做。也就是说学习评分倾向于"把队列订单交给别的车、自己空车回仓"，而距离评分总是继续前往最近的客户。
2. **2 个单元只有更深的后缀不同**（MYOPIC d0 ord11 `[37,40]`、SAA d0 ord45 `[11,44]`），锚点后首动作相同。按 ANCHOR-01 已登记的口径，这类差异**必须**记为"差异存在但不在紧邻动作上"，不得因首动作相同就判无差异——本轮正是如此记录的。
3. **两对相位的 `plan_hash` 相同**（SAA d1 ord6 与 MYOPIC d1 ord7；两侧都相同）。它们的公开状态不同（`public_identity` 不同），只是重构出的车队计划恰好一致；这不影响任何单元内比较。

## 4. L1-B：非学习搜索的覆盖（次级判据）

| 指标 | 值 |
|---|---|
| MaskCO-greedy 计划落在 distance-SGBS 合法补全集合内 | **19 / 19** |
| 落在集合外 | **0** |
| MaskCO 自身 SGBS 唯一补全数 | 19 个单元**全部 = 4** |
| distance SGBS 唯一补全数 | 6 个单元 2 个、8 个单元 3 个、5 个单元 4 个（2–4） |
| 两个集合的交 / 并（19 单元合计） | **9 / 123** |
| 模型集合中不被距离集合覆盖的计划（合计） | **67**（19/19 单元都存在） |
| 补全失败数（两侧） | 0 |
| 合法补全数（Pass B） | distance 56 / model 76 |

读法必须两层：

- **覆盖的结论只针对冻结预算**：`beam=2`、每父≤2、≤8 completions。它说明"这个预算下的距离 beam 能到达同一计划"，**不是**"任何非学习搜索都能到达"。
- **"模型集合比距离集合大" 并不反证覆盖结论**：模型 SGBS 有 67 个计划不在距离集合里，但主判据只比**贪心单输出**，而那一个输出 19/19 都在距离集合内。这正是 R2.1 把主对象与集合对象分开的目的。

## 5. 三层裁决（按预注册规则执行，未事后调整）

| 层 | 判据 | 结果 | 允许的结论 |
|---|---|---|---|
| **L1-A** | 存在合法、可复现的贪心行为差异 | **GO**（19/19） | 仅建立**行为差异事实**，不代表更优 |
| **L1-B** | 同一预算 distance-SGBS 是否覆盖 | **全部覆盖**（19/19） | 行为差异成立；**独立搜索价值未获支持**；**停止工程与训练扩张** |
| **L2/L3** | 可认证质量收益 | **未回答，未授权** | 需另立实验并先解决 C0 预算与冷链物理检查资格 |

**按预注册止损线执行**：L1-A Go 而 L1-B 全部覆盖 ⇒ 不扩张。具体地，本轮**不**启动价值头训练、**不**改动执行引擎、**不**开发事务引擎、**不**进入正式 L2/L3、**不**做生产资格认证。`C0 预算 B`、`HIST 未来支持域`、完整物理认证、执行事务安装资格四项阻断各自继续 open，未因本轮结果获得任何豁免。

## 6. 复现

```bash
cd C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_camr_static_01_20261019
python register.py              # T1 -> registration.json        (refuses overwrite)
python synthetic_regression.py  # -> synthetic_regression.json    (no model, no SOURCE day)
python freeze_identity.py       # T2 -> run_identity.json         (loads the frozen model)
python degeneration_gate.py     # -> preflight.json (binds run_identity_sha256)
python run_pass.py --run 1      # -> pass_a_run1.json, pass_b_run1.json
python run_pass.py --run 2      # -> pass_a_run2.json, pass_b_run2.json
python verify_camr_static.py    # -> verification.json (12/12)
```

全部脚本 `refuse overwrite`；`run_pass.py` 在 `preflight.all_ok != true`、或身份链任一环节哈希不匹配、或源文件与 T2 记录不符时**拒绝启动**。两次运行耗时 23.8 s / 24.1 s（单元上限 300 s、整包上限 3600 s）。

## 7. 本轮没有建立的结论

1. **没有**质量优势：全程没有任何 J / 距离 / 能耗 / 品质比较。
2. **没有**"不可替代搜索价值"：L1-B 全部覆盖，恰恰否证了这一主张在本预算内的成立。
3. **没有**一般化的"非学习搜索可达"结论：覆盖只在 `beam=2 / ≤2 per parent / ≤8 completions` 下成立。
4. **没有**代表性：19 个单元是**相位–mask 暴露量**，来自 **2 个**先前已检查的开发者实例，不是 19 个独立样本，未做显著性检验。
5. **没有**物理可行性：`certify_route` 的 `projected_energy_kwh / quality_loss / thermal_margin` 恒为 `None`，能耗与品质在本管线里仍是"跑不了"。
6. **没有**安装资格：合成回归的两个夹具都没有在途车（多订单在途队列只存在于这 19 个单元），真实的"在途 + 多车"组合首次被执行是在退化闸门；执行事务缺陷（ANCHOR-01 合成臂）依旧未修复。
7. **没有** mask 选择结论：mask 固定，`MaskValueModel` 的零初始化头未使用。

## 8. 轮内偏差与更正（已登记）

首次 T2 链（`superseded_v1/`）被作废：`verify_camr_static.py` 的 `F1` 用了 `plan_hash` 长度 = 64 位的错误假设（`action_contract.plan_hash` 返回 `sha256(...)[:16]`）。该校验器列在 T1 的源文件清单内，因此按裁定"源码修复后必须重新冻结 T2 并重跑全部前置自检"：

1. 首次链全部产物原样保留在 `superseded_v1/`，**其数字不得引用**；
2. `registration.json`（T1）**未改动**；
3. 全部前置自检重跑：合成回归（输出逐字相同 `0eebcd0a…`）→ T2 → 退化闸门 19/19 → Pass A/B 两次 → 独立复核 12/12；
4. 修正只影响校验器判据，不影响任何分析口径；首次链与最终链的 19/19 结果一致，但只有最终链产物作为证据。

## 9. 封存不变项

`Gate A FAIL`、`Q3-S3 NO-GO`、`GTMIN FAIL`（永久）、`V-10 FAIL`、`L1 BLOCKED`、`L2/L3 NOT_AUTHORISED`、`FULL_INSTALLABILITY_NOT_AVAILABLE`、`stable_online_gain=False`、`t_min=0`、正式测试种子 `20260927`/`20260928` **SEALED**、`PRODUCTION_NOT_QUALIFIED`。路线表：价值头训练 `NO-GO`、SAA 替换 `暂停`、`COMMIT-01` `暂缓`、`GRAN-01` 次级消融、`CAMR-STATIC-01` **本轮已完成，不自动续期**。

## 10. 产物

| 文件 | 角色 |
|---|---|
| `registration.json` | T1 预声明冻结（19 单元、契约、口径、资源、禁止项、停止条件） |
| `camr_shared.py` | 隔离实现：共享合法动作接口、统一 tie-break、greedy/SGBS 骨架、评分插件 |
| `synthetic_regression.json` | 9/9：共享集合过滤、`allowed` 门、tie-break、确定性、`completions[0]` ≡ greedy、检查器负例、评分无关性 |
| `run_identity.json` | T2：绑定 T1 + 19 个源文件哈希 + 合成回归回执 + 模型参数身份（`gate_result_included: false`） |
| `preflight.json` | 退化闸门 19/19，绑定 `run_identity_sha256` |
| `pass_a_run1/2.json`、`pass_b_run1/2.json` | 19 单元 × 两侧的逐单元原始记录（含失败账字段） |
| `verification.json` | 12/12 独立复核 + 全部产物 sha256 |
| `superseded_v1/` | 首次 T2 链，作废，含作废说明 |
