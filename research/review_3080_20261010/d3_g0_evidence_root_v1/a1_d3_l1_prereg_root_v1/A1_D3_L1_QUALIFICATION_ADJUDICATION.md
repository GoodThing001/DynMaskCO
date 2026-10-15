# A1-D3-L1 只读资格裁决（CAUSAL_GATE / COLLECTION_TIMING / F_SCORING_INTERFACE）

- 轮次：`A1-D3-L1-MECHANISM-PREREG`（用户授权批次的机械执行顺序第 1、3、4 项）
- 性质：**只读资格裁决**。未采集状态、未写快照、未做 F−D-Geo 比较、未训练、未改冻结源码
- 证据：`l1_gates/A1_D3_L1_CAUSAL_GATE.json`(+stdout/stderr)、`l1_gates/A1_D3_L1_F_SCORING_INTERFACE.json`(+stdout/stderr)
- 三项状态**分别报告**，任一未通过**不得**自动升级 L1 执行资格

## 1. `CAUSAL_GATE` = **PASS_WITH_MANDATORY_HORIZON**（9/9 PASS）

**用户指出的污染源成立**，并已在代码级与行为级同时确认：

| 证据 | 内容 |
|---|---|
| 源码 | `StrictOnlineEnv.__init__(..., booking_horizon=None)` 默认 None；`has_future_reveal`(248-258) 在 None 时扫描**真实** `reveal_time`；`jf1h_repair.py:194` 正是它的消费者 |
| 行为 | 在决策时刻 `clock=5` 固定已揭示前缀，**删除一个当时不可见的未来订单** → horizon=None 时答案由 `True` **翻转为 `False`**（G2 PASS）；horizon 设定时答案**不变**（G3 PASS） |
| 精细刻画 | 仅扰动未来订单的**时间/特征**（仍不可见）不触发翻转（G3b PASS）→ 泄漏来自"**未来是否还有揭示**"这一存在性信息，horizon 正是消除它的手段 |
| 边界 | horizon 分支是干净的公开判据：`clock<H` → True，`clock≥H` → False（G4 PASS） |

**强制条件（写入 L1 采集规格，未通过不得标 `STRICT_ONLINE_QUALIFIED`）**：

1. 采集器构造 `StrictOnlineEnv` 时**必须显式传 `booking_horizon`**；
2. 取值 **24.0**，来源是数据生成器的**公开声明**：`scripts/data/generate_coldchain_data.py:119`
   （`if instance_type.endswith('1'): tw_width, horizon = 0.2, 24.0`，`tw_end[0] = horizon`）；
   **不得**按每天最后真实揭示时间倒推；
3. **既有 pilot 常数 16.0 不可用**：dev_check 的 `reveal_time` 最大值 = **18.7394 > 16.0**，
   用 16.0 会丢弃晚期揭示、改变问题本身（G7a PASS 记录该否决，G7b 记录 24.0 覆盖全部揭示）；
4. **R1 槽位必须按 `reveal_idx` 计，不能按 `event_id`**：`event_id` 是**物理事件**计数器
   （服务完成/返仓/预约截止都会 +1，见 `strict_online_env.py:488-511`），而揭示事件是
   `sorted(set(reveal_time))` → **同一时刻的多个揭示算一次事件**（G5 PASS）；
5. 快照钩子同时收到 `event_id` 与 `reveal_idx`（`:553`），采集器必须用后者选槽。

**本项未覆盖、留待采集资格的部分**：完整的 **rollout 级**未揭示订单扰动测试
（本文的扰动在决策接口层完成）；采集时需在真实 rollout 上复做并逐状态记录。

## 2. `COLLECTION_TIMING` = **NOT_RUN**（诚实报告：本轮未执行）

- **F-free 采集器尚未编写**，因此计时资格运行未开始；**未**自动启动 64 槽位正式采集；
- 已就绪的可冻结项：R1 / `dev_check` / K=4 / 槽位按 `reveal_idx` / `booking_horizon=24.0`；
- dev_check 的逐实例揭示事件数、`K=4` 下的槽位数与 `insufficient_events` 实例数已按实测记入
  `A1_D3_L1_CAUSAL_GATE.json` 的 `dev_check_census`（**槽位数是预登记预算，不是合格快照保证**）；
- 计时运行的硬上限（总墙钟、单状态上限、调用次数、环境与身份）仍需在运行前登记；
  分段成本（baseline rollout / capture / restore / hash / resume-parity）必须分别记录；
- **若因果门或契约/profile 身份不成立则停止，不用不合格样本估算正式成本**（本轮因果门已通过，
  但契约/profile 身份缺口仍**未关闭**）。

## 3. `F_SCORING_INTERFACE` = **VERIFIED_WITH_CURRENT_BACKBONE**（8/8 PASS）

永久标签：`F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED`
（**不能**推出 `F_HISTORICAL_IDENTITY_VERIFIED`）。

| 检查 | 结果 |
|---|---|
| I1 checkpoint = 登记的 F | sha256 `f5acf83f…903a7f5` 匹配 |
| I2 经审计路径重建（`load_mpre_trained → partition → nnx.merge`） | 成功，**2.43 s** |
| I3 恢复的确是**训练后**任务残差 | `c_adapter` 6 叶子 absmax 0.7194、`delta_head` 6 叶子 absmax 0.2422（非零） |
| I4 固定输入前向 | 成功；`raw_3d` 维数 **3**；单次前向 **1.80 s** |
| I5 长度与有限性 | 形状 `[1,3]` = 共同动作数；全部有限 |
| I6 重复前向 | **逐位一致** |
| I7 历史 `first argmax` 语义 | `argmax=2` = 首个最大值索引 2 |
| I8 恢复与否确实不同 | 与未恢复（fresh）模型的最大绝对差 **26.16**（fresh 自身 absmax 4.52） |

评分值（声明为**固定合成输入**，非决策状态、非任何性能证据）：
`[17.3728, 10.2915, 30.6826]`。**未调用 D-Geo、未做输入搜索、无任何增益主张**。
`I2/I3` 也是"只调 `load_mpre_trained` 会得到另一个评分器"这一论断的实证。

## 4. 本轮我自己的三处工具缺陷（已修正并留痕）

1. 因果门 stand-in 传入 1-D `reveal_time`（真实方法按 `[inst, i]` 取 2-D）→ `IndexError`；
2. F 检查传 0-d `timestep` → backbone 内部索引失败；改为形状 `[B]`；
3. F 检查 `np.asarray(s).block_until_ready()` 误用（ndarray 无该方法）→ 改 `jax.block_until_ready`；
   且初次未做 try/except，异常直接中断进程而非记 FAIL——已改为**记录失败并继续裁决**。

前两次失败运行原文保留在 `l1_gates/*_stderr.txt` 与 `*_stdout.txt` 的对应版本中（未改写为通过）。

## 5. 下一步（需用户授权才执行）

1. 编写 **F-free 采集器**（不 import 任何 F/D-Geo 模块；用 `booking_horizon=24.0`；按 `reveal_idx` 选槽；
   因果断言内建）并对 1 个 `cal` 实例 × 前 2 个揭示事件做**一次**计时资格运行（不纳入正式 64 槽）；
2. 关闭**契约/profile 身份缺口**（`booking_horizon` 不在 profile/contract 中，需在 L1 身份清单中显式冻结）；
3. 依据实测成本冻结正式资源门后**再次提交审查**；
4. 正式 64 槽位采集与 F−D-Geo L1 比较**均未授权**。

## 6. Claim limits

1. 本轮只证明：因果污染源存在且 horizon 可消除；F 的**评分接口**在**当前 backbone** 下可用且稳定。
2. 未证明：F 优于 D-Geo、任何学习增益、任何在线效用；I3 正式负结果、D2、D3 G0、D8、E2、
   `stable_online_gain=False` 全部不变。
3. 扰动测试在决策接口层完成，rollout 级因果门留待采集时执行。
4. 历史 backbone 精确身份仍 `UNVERIFIED`；本轮未继续投入其溯源（按用户裁定暂缓）。
