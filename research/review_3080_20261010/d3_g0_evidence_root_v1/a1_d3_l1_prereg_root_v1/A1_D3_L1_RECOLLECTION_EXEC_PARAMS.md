# A2 重采集执行参数补全（P0 提交件：R1 / 开发日 / K / 状态顺序 / 资源门）

- 轮次：`A1-D3-L1-MECHANISM-PREREG` / P0（用户裁定：`COLLECTION_METHOD_APPROVED / EXECUTION_PREREG_NOT_FROZEN`）
- 本文件是 `A1_D3_L1_STATE_RECOLLECTION_PREREGISTRATION.md` 的**执行参数补全**，不替换其任何规则
- **未执行**任何采集、未运行任何 rollout、未做任何 F/D-Geo 评分；全部为只读检查与规模计算
- K 值**不由我指定**：本文件提交候选规模与资源测算方案，**待用户确认后**才写入冻结版

## 1. 事件选择规则：采用 R1（用户倾向，我同意）

**R1 = 每个开发日的「前 K 个揭示事件」**，按揭示顺序，天然因果：取样位置只依赖揭示事件序列，
不需要知道整天未来事件的分布来反推"分位数"，因此不引入未来信息依赖。

冻结写法（修正用户指出的措辞问题）：

- **运行前冻结**：`day_id`（实例）× **事件位置序号**（1..K）、顺序、以及"位置 → 事件"的物化规则；
- **采集后物化登记**：实际的 `event_id`（及 `reveal_idx`、`reveal_time`）按上述规则物化写入清单；
- **不得**声称"完整 event_id 清单已在运行前生成"。清单中必须标注每个字段是
  `PRE_FROZEN`（day_id、位置序号、K、顺序规则）还是 `MATERIALISED_AT_COLLECTION`（event_id、reveal_time、state_hash）。
  物化后若某实例事件数不足 K，记 `insufficient_events` 并保留在分母。

## 2. 开发日候选规模（**精确计数，未执行任何 rollout**）

来源：`data/m0_scale/M0_TEACHER_MANIFEST.json`（`purpose="M0 教师模型（TRAIN-64 / CAL-16 / DEV-CHECK-16）"`，
`train_seed=9701 / cal_seed=9702 / dev_check_seed=9703`，三 cell 的 `file`+`sha256` 已登记在
`A1_D3_L1_STATE_AND_F_ASSET_AUDIT.json.data_identity`）。

状态上限 = 实例数 × K（若某实例揭示事件数 < K，则被 `insufficient_events` 截断）：

| split | 实例数 | K=1 | K=2 | K=3 | K=4 | K=6 | K=8 |
|---|---|---|---|---|---|---|---|
| `dev_check` | 16 | 16 | 32 | 48 | **64** | 96 | 128 |
| `cal` | 16 | 16 | 32 | 48 | 64 | 96 | 128 |
| `train` | 64 | 64 | 128 | 192 | 256 | 384 | 512 |

**建议**（待用户确认）：
- 首选 **`dev_check` × K=4 → ≤64 个候选状态**（若按 K=2 取则为 32，与既有 32 条资产的量级一致）；
- `cal` 可作为备用集合，但**两集合不得在同一轮混用后择优**；
- `train` 不用于 L1（保留给 L2/L3 的新鲜状态需求）。

## 3. 资源测算：现状与方案（**诚实报告：无现成合格计时记录**）

- 检索结论：仓库中**没有**"仅基线 rollout + 快照捕获"的现成墙钟记录。
  可比记录都混入了别的重负载（例如 `sgbs_fixed_cal/summary.json` 含固定状态评分与搜索；
  `mpre_reinforce_s42_clean` 的采集是**带模型的**采集，日志显示 256 点 / 64 实例，**不能**用作本路线成本）；
- 因此**不得**凭猜测填 K 与预算。方案是：先做**计时资格运行**（需另行授权执行），再冻结正式资源门：

| 项 | 冻结方案 |
|---|---|
| 计时资格运行 | 1 个实例 × R1 前 2 个事件；仅基线 rollout + `capture_recourse_snapshot` + `restore` + `snapshot_state_hash` + `resume_parity`；记录各段墙钟 |
| 外推规则 | 按状态数线性外推（注明外推系数与假设）；不引入 F/D-Geo 任何调用 |
| 采集上限 | 状态数上限（建议 ≤64）、总墙钟上限、每状态墙钟上限；三者独立计 |
| 恢复验证调用次数 | 每状态 1 次 capture、1 次 restore、1 次 hash 重算、1 次 resume-parity；重复次数上限固定 |
| 失败处理 | 任一状态失败即按 §6 原因分类保留在分母；**不追加运行、不放宽上限** |
| 计时运行的状态 | **不进入**正式集合（仅用于计时） |
| 既有 PENDING | 不得被解释为无限预算；既有 10.0 s 单决策口径不得事后放宽 |

## 4. 采集器合规性（P0 只读检查的两个结论）

**结论 1（关键，负向）**：既有先例采集器 `scripts/evaluation/run_fixed_state_quality.py`
**不是**合格的 L1 采集器：它在**同一进程**内先 `load_trained_model(...)`（:248）并构造
`mpre_score_fn`/`mtrained_score_fn`（:249-250），再创建 `Probe`（:253）生成状态（:116 存 `'env': env`）。
虽然模型实际只在事后评分（:319）被使用，但**同进程存在 F** 使"状态生成独立于 F"无法自证。
⇒ L1 必须使用**独立的、不 import 任何 F/D-Geo 模块**的采集器（新模块，需另行授权编写与运行），
且采集过程必须在**无 F 的进程**中完成，采集产物不含任何 F 评分。

**结论 2（正向）**：特征归一化走**可见**变体——`scripts/data/repair_state.py:34` 导入
`from cvrptw_utils import coord_normalize_visible`，且在 `:106` 使用它；
在本次检查的采集/快照/环境模块中**未发现**带泄漏性质的 `coord_normalize(` 调用。
该结论**仅覆盖本次检查的文件**；完整因果门见下。

**因果门（采集时逐状态强制）**：传给 F/D-Geo 的特征、`coord_normalize_visible` 结果与候选枚举
**不得**读取未揭示订单；快照可保存完整环境供复核，但**必须与决策可见信息分离存放**，
且记录"决策时刻可见集合指纹"。

## 5. 身份清单（与重采集预登记 §5 一致，K/规模确认后一并冻结）

数据三 cell 的 `file`+`sha256`；`M0_TEACHER_MANIFEST.json` sha256；profile 与冷链合同（**契约/画像身份缺口须先关闭**）；
`recourse_snapshot.py`、`strict_online_env.py`、`jf1h_repair.py`、`cc_lns_replanner.py`、
`dynmaskco_cc_context.py`、`action_contract.py`、`repair_state.py`、`cvrptw_utils.py` 与**新采集器自身**的 sha256；
python/jax/flax 版本；F 侧仅登记（`model.ckpt` sha `f5acf83f…`；backbone 标签见独立审查意见）。
**排除**并行 `CC_Compare` 提交。

## 6. 失败分母（引用预登记 §6，不新增例外）

`capture_failed` / `restore_mismatch` / `state_hash_unstable` / `resume_divergence` / `causal_violation` /
`empty_pool` / `insufficient_events` / `timeout` / `identity_mismatch`；全部保留在分母，禁止替换、筛除与追加。

## 7. 本文件的状态与待办

- 状态：`EXECUTION_PARAMS_SUBMITTED / AWAITING_USER_CONFIRMATION`（R1 已采纳并写入；K、开发日、预算待确认）；
- 待用户确认项：**(a)** 采纳 R1；**(b)** 开发日集合（建议 `dev_check`）；**(c)** K（建议 4，规模 ≤64）；
  **(d)** 授权计时资格运行（仅此一项需要执行）；**(e)** 授权编写 F-free 采集器；
- **仍未授权**：实际采集、F−D-Geo 比较、E2 重训、L2/L3、最终测试。
