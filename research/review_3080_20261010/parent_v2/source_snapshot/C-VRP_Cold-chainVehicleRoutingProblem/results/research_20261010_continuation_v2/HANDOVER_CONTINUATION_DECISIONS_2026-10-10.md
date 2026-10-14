# 交接续办：裁定与执行记录（2026-10-10 17:37 起）

> 依据 [`HANDOVER_2026-10-10.md`](HANDOVER_2026-10-10.md)。本文件记录**用户裁定**与**本轮自行执行的技术裁定**，每项带证据。
> 本文件不产生任何正向结论；未建立项仍为：稳定在线增益、MaskCO 归因、取货不可能性、等价性。

## 1. 用户裁定（4 项）

| # | 事项 | 裁定 |
|---|---|---|
| 1 | 取货主门前置缺陷 | **(a)** 重述主门 + 记为协议修订（(b) 已由 B×3 诊断否定） |
| 2 | 5 条文档不一致 | **只在新文件记录，不改封存字节** |
| 3 | `program_support_audit.py` 改造 | **保留 v2 并记录行为变更** |
| 4 | 3080 长路径 | 用户明确「我的账号就是管理员，你可以决定」⇒ 由本方裁定（见 §2.3） |

## 2. 本轮技术裁定（含证据）

### 2.1 主门修订已登记
- 新增 `pickup/P_LIVE_DEV_PANEL_V2_REGISTRATION.json`（追加式，pin 住 v1 SHA `f9d9396c…`，**v1 字节未改**）与人类可读的 `pickup/P_LIVE_DEV_PANEL_V2_AMENDMENT.md`。
- 主确认对照改为 `maskco − strong_rolling`（≥20 utility/day 且配对 CI 下界>0）；归因对照 `maskco − recon_only`；`cond_hist` **从门降级为报告臂**。
- 已论证该替换**不是降低门槛**：两个运行点上 `strong_rolling` 都不弱于 `cond_hist`（B：−361.13 vs −376.41；B×3：4541.51 vs 4480.72）。

### 2.2 运行点已 pin（原为待定项）
- **B = `710.6548152249378`**，权威来源 `configs/i3_20261008_v1/protocol.json`（`"budget": 710.6548152249378`）与 `configs/i3_20261008_v2/protocol.json`；同值亦用于精确枚举见证与 B×3 诊断（×3）。
- `709.1017637699684` 仅出现在 P-BUDGET-v1 结果中，是**计算量**而非登记预算 —— 此前把它与登记预算并列是**口径混淆**，已在修订中显式排除。

### 2.3 3080：**不启用长路径**（有证据的裁定，非偏好）
- 裁定依据：**本管线不使用 orbax**。全库检索 `orbax|nnx.save|nnx.load|CheckpointManager|flax.training.checkpoints` 仅命中两处**注释**，且都说明刻意不用：
  - `scripts/training/train_fleet_head.py:230`：「保存 head params（Flax 0.10.4 无 nnx.save，用 pickle 存 nnx.State + 配置）」
  - `scripts/training/train_coldchain_utility_probe.py:290`：「Flax 0.10.4 无 nnx.save：pickle 存 nnx.State + 配置」
- ⇒ 启用长路径**不会解锁任何本管线需要的能力**，却要对**共享机做全局注册表改动**。故**不改**；若将来确需 orbax，再单独提案。
- 3080 实探（`tools/_probe.py .vscode/sftp.3080.json windows`）与交接一致：`desktop-9v1opdi\zh`、RTX 3080 Ti 12288MiB（驱动 595.79）、16 核 / 31.8GB、C:138.4GB D:434.8GB、`project\MASKCO-Main` 已存在、`long_paths=0`。
- **新发现的坑（交接未记）**：3080 上 `python` 解析到 `D:\anaconda\python.exe` = **Python 3.9.7（base）**，**不是** `MASKCO_env`(3.10.20)。远端命令若直接写 `python` 会跑在 base 环境。**必须显式使用 `MASKCO_env` 的解释器**或先 `conda activate`。另 `git` 探测为空，远端不要依赖 git。

### 2.4 审计器保留 v2（已复核实现）
- 独立复核 `program_support_audit.py`：`exit_code_for()` **只返回 0 或 2**；`--allow-tool-output` 的路径校验只放行本工具 `reports/`。
- 复核 docstring 的准确表述：**通过/失败契约不变（0/2）**；变的是**硬错误**由"未捕获异常→退出 1"改为"结构化 `audit_error`→退出 2"。
- 实跑 `--selftest`：**exit 0**，三夹具 `clean→0 / unsupported_claim→2 / missing_input→2` 全对。
- 已把上述契约写入新增的 `scripts/research_claim_gate/README.md`。

### 2.5 文档不一致只记录不改字节
- 新增 `reproduction/DOC_INCONSISTENCY_CORRECTIONS_2026-10-10_HANDOVER.md`，逐项登记 5 条并标注复核状态（含独立复核了 `_bverify_pinned/fsq_s42_{train,cal}_v1/per_state.json` 不在盘、`delivery_any50_confirm_v1/result.json` 自相矛盾）。

### 2.6 发现并修掉一个**种子完整性缺陷**（本轮最重要的一处）
- v1 把 `mask_seed_dev=2026120202`、`bootstrap_seed_dev=2026120212` 登记为"全新"，这在**登记时**是成立的；但**其后被 n8–n14 的开发 pilot 消耗**（二者是 `pickup_maskco_arm.py --qual-seed`、`pickup_arm_pilot.py --pilot-seed/--bootstrap-seed` 的默认值，并出现在 `pickup_arm_pilot_v1/_n45_v1/_mf{50,70,80,90,100}_v1`、`pickup_arm_qualify_v1`、`pickup_headroom_*`、`pickup_cust_headroom_v1`、`pickup_learnability_v1`）。
- 沿用即违反 v1 自己的"不得复用任何已观测种子"规则 ⇒ **v2 改为 `mask_seed_dev=2026120231`、`bootstrap_seed_dev=2026120232`**；`dev_dataset_seed=2026120201`、`arm_order_seed_dev=2026120211` 经**全库检索确认仍无任何匹配**，保持不变；备用全新种子 `2026120213 / 2026120221 / 2026120233`。
- **规则化**：「全新」是**时点陈述**，种子会在登记后被消耗 ⇒ 运行注册**在数据生成前必须重复全库检索**。该规则已写入修订 JSON 的 `seed_integrity.inference_rule`。

### 2.7 面板最后一项未闭合资格：整队全 J 重放（本轮闭合）
- 新工具 `research/pickup_20round_audit/live_qualify_fleet.py`；协议 **P-LIVE-QUALIFY-FLEET-v2**；产物 `live_qualify_fleet_v3/result.json`（通过），v1 协议失败运行 `live_qualify_fleet_v2/` 原样保留。
- 关键：**按"派遣车队宽度"选点**取代盲目时钟步长。原设计取 clock 2.0–5.8h（车队还空着 ⇒ 宽度 1）；新扫描显示宽度可达 **4/15**（宽度≥3 的决策点有 459 个），故此前"只有 1 辆车被派遣"纯属采样缺陷。
- 结果：**12/12 捕获精确一致**（终局 full A-v1 J、accepted 集合、served 数；`max_abs_delta = 0.0`），捕获宽度 **3–4 / 15**，重放确定性 12/12，选点可复现 12/12。
- 附带发现（如实记录）：能量账本通道**活跃但仅 5/12 捕获决策相关**；C1（+50 kWh）与 C4（+10000 kWh）**击中完全相同的捕获集合** ⇒ 排除"探针太小"；7 个惰性捕获中 **4 个剩余决策点数恰为 0**（已测量），另 **3 个仍有 9/14/43 个剩余决策点**，解释未闭合（与已登记的 threshold 92.71% / budget 7.29% 一致，但本工件未直接测量）。
- **判据修订已披露**：v1 要求 C1 在每个捕获都检出——该判据把"决策相关性"误当"仪器属性"；更正后要求"通道须至少活跃一次 + 必须报告检出率与实测剩余决策数"，**提高**而非降低报告要求。
- 承诺不可改在**快照路径**上不敏感（1 个可行使捕获 0/1）；强制点仍是 packet 校验器（其负向对照在 P-LIVE-QUALIFY-FULL-v1 q3 确实触发）。记为范围边界。

### 2.8 跨主机验证与一个新发现的工具缺陷
- **3080 路径已打通**：推送 `research/pickup_20round_audit`（226 文件 / 15.8 MB）并在 3080 实跑。新增 `_remote_run.py` 包装器（在导入 numpy 前把 BLAS/OMP 线程钉为 1，满足 PROGRAM 的单线程隔离要求——`tools/_srv_exec.py` 本身无处注入环境变量）。
- **跨主机一致性的准确表述**：1158 个决策点的 `event_id` / 派遣宽度 / `has_commit` 序列 **0 处差异**；`clock` 浮点有 **1e-9~2e-8** 差异。⇒ 正确表述是"**离散决策结构逐位一致，时钟浮点有 1e-8 级差异**"，**不得**写成"逐位完全相同"。
- **基础设施缺陷（已修）**：`tools/_srv_exec.py` 末尾重复了 `if __name__ == '__main__': main()` ⇒ **每条远端命令执行两遍**（首次成功、第二次 `FileExistsError`）。已删除重复块。不影响任何已归档数值。



1. **未实现、未运行**修订后的面板（状态仍 `amended_not_implemented_not_run`）。
2. **未改动**任何封存字节：I3 裁决、v1 注册、183 文件包、`delivery_any50_confirm_v1/result.json` 均原样。
### 2.9 运行注册已写（第 2 轮，pre-data）
- `pickup/P_LIVE_DEV_PANEL_V2_RUN_REGISTRATION.json`，状态 `registered_not_run`；**7 项 pin 的 SHA 已用脚本自校验全部匹配**。
- pin 了：运行点（B=710.6548152249378，来源 `configs/i3_20261008_v1/protocol.json`）、K=10、10s 屏障、p_c、车队 15、容量 50、cooling_share。
- **开关声明**（逐个查过源码默认值）：S3-3 `overage_price=None`、S3-4 `incr_eval=False`、S3-4-D3 `anytime_vote=False` **全部关闭**；`soft_knn` 采样器**未使用**（`self.sampler = sampler` 是显式构造参数）；`standby_orders=None`、`future_policy='reveal'`。
- **基线 pin**：三个重构臂共享 **`uncond_dynamic`** 基底（资格与两个 pilot 已用它，属延续而非新选择）；`strong_rolling` 独立成臂；`cond_hist` 仅报告。并明确 `maskco−strong_rolling` 是**部署级**对照（同信息/同预算、不同基底），**不是**单变量消融。
- **种子重新检索确认全新**（本轮当场重跑）：`2026120201/2026120211/2026120231/2026120232` 全库**无匹配**；并写明"全新"是时点陈述，**生成数据前必须再查一次**。
- **分阶段执行**：Stage A（4 实例便宜门，含局部获胜计数与"对不做任何事报告"）→ 通过才 Stage B（8 实例五臂）。诚实声明 8 实例是资源受限的探索选择、**无功效声明**。

### 2.10 Stage A 局部获胜普查：**修正了一条此前的直觉**（第 2 轮）
- 工具 `pickup_guide_local_win_census.py`（**只读**，仅重读归档 pilot 行，零新计算/零新数据/零种子）；产物 `stage_a_local_win_census_v1/result.json`。
- 结果：200 次配对中引导项**赢 74 / 输 86 / 平 40** ⇒ **37% 局部获胜**，**Stage A 停止规则（零获胜即停）不触发**。
- ⚠️ **与交付线机制不迁移**：交付线的引导项是"结构上 0/400 从不更便宜"，从两个 pilot 的负均值 + 交付线机制很容易推断取货也一样——**普查证明不是**。取货引导项**活跃**，只是**登记掩码比例附近净为负**（mf50：赢6输10；mf70：赢4输24）。
- 0.8 及以上局部转正（16/16、20/12）但 **0.9 与 1.0 结果完全相同、非独立点**；扣重后有效 168 对：**赢 54 / 输 74 / 平 40，仍净为负**。⇒ **不得**因为高掩码格子的局部数字更有利就改面板比例（那是选择）。
- 空对照 `recon_only vs discard` **200/200 全平局**，独立复核了臂资格断言。

### 2.11 Stage A 迭代实验：得到一条**阻断性**结论（第 3 轮）
- 工具 `pickup_arm_iterative.py`（4 个**全新**实例，数据种子 `2026120201`、掩码种子 `2026120231`、3 轮、三模式、解码调用各 21 次**匹配**）；产物 `stage_a_iter_v1/`（本地）与 `stage_a_iter_3080_v1/`（**3080**）。
- **可行性关键**：`propose` 的子问题**完全由传入的 `order_ids` 构造**，`env` 只给坐标/需求 ⇒ **加一轮 = 把上一轮置换再喂回去**，无需写回活动计划，风险低。
- **结果一（结构阻断）**：4 个完整日的单车可变后缀规模普查为 0×10921、1×11、2×12、3×9、4×6、5×4、6×2，**最大 6，规模 ≥12 的机会数 = 0**。而 n13 把 oracle 奖赏定位在 **n≥12** ⇒ **按单车后缀构造的注册面板结构上够不到奖赏区制**（独立复现 `P-CENSUS-v1` 的"平均 2.09 单/车、71.5% 路线 ≤2 单"）。
- **结果二（迭代无用）**：引导臂 **0/7 状态有任何一轮被接受**，最好结果**恰为 0.0**；第 1 轮均值已更差（−0.0703）且从未挽回。唯一非零的 +0.003431 来自**纯距离对照**（非引导项），相对 ~700 的日效用为 **0.0017%**，比 ≥20 门低三个数量级。空对照 `recon_only−discard` = 0.0（7/7 全平）。
- **停止规则如实处理**：字面规则（"零局部获胜"或"两臂差于不做任何事"）**均未触发**（局部获胜 74/200；两臂 0 例更差，是**惰性**而非有害）。因此**不把别的理由洗成"规则触发"**——Stage A 给出的是一条**更强**的理由：**结构够不到奖赏区制 + 处理效应恰为 0**，故**不授权 Stage B**。
- **跨主机**：规模普查、最大规模/≥12 计数、每模式 delta、迭代后配对 **全部逐位一致**（本地 68.0s / 3080 40.5s）；3080 本次只执行一遍（`_srv_exec.py` 修复生效）。
- **建议**：**不要按现规格建 Stage B**；有证据的下一步是**改子问题构造 → 多车联合重指派**（交接 §8 第 7 项，即主门处置时被列为选项 (C) 的方向，本项目早前测得 `G_fleet=17.891%`）。



## 3. 未做（边界）

1. **未实现、未运行**修订后的面板（状态仍 `amended_not_implemented_not_run` / `registered_not_run`）。Stage A 已给出**不授权 Stage B** 的理由（见 §2.11）。
2. **未改动**任何封存字节：I3 裁决、v1 注册、183 文件包、`delivery_any50_confirm_v1/result.json` 均原样。
3. 未跑**在线全事件**迭代策略（Stage A 只在每个状态的**单点**迭代 3 轮），也未跑多车联合子问题。
4. **未**解除四项禁止声称。

## 4. 下一步（按序）

1. ~~补完面板资格中唯一未闭合一项：whole-fleet full-J replay~~ ✅ 第 1 轮。
2. ~~写运行注册~~ ✅ 第 2 轮（7 项 pin 自校验通过）。
3. ~~实现并跑 Stage A（迭代 vs 一次性）+ 局部获胜普查~~ ✅ 第 2–3 轮；**结论：不授权 Stage B**（结构够不到奖赏区制 + 处理效应为 0）。
4. **改子问题构造 → 多车联合重指派**：把子问题从"单车 ≤6 单"扩到跨车规模（本项目早前测得 `G_fleet=17.891%` 的联合车队-路线动作空间）。需**先登记新假设与新协议**，再按同一套资格与对照纪律执行（Stage A 式便宜门优先）。
5. 全程与本地做离散结构一致性对照（已建立：离散结构一致、时钟浮点 ~1e-8 差异）。

## 5. 目标收尾核验（第 4 轮，独立复核）

本会话结束时按项目惯例做**"新工作之后必须由原硬门校验器复核负结果保全"**：

| 核验 | 结果 |
|---|---|
| `verify_negative_archive.py`（183 文件封存） | **exit 0**；`engineering_consistency_pass=true`；`original_package_files_unchanged=183`；`formal_json_count=14`；四项否定标志（`scientific_positive_gain` / `stable_maskco_core_online_gain` / `full_from_zero_reproduction` / `equivalence_established`）**全部仍为 false** |
| I3 封存 gate | ✅ `results/i3_online_recovery_1009_v1/online_20260926/gate.json` = `48306155…` 与裁决记录一致 |
| I3 封存 protocol | ✅ `configs/i3_20261008_v2/protocol.json` = `4825392b…` 与裁决记录一致（**更正**：该 `protocol_sha256` 指向 **v2**，不是 v1；v1 为 `345604e3…`） |
| 面板 v1 注册字节 | ✅ `f9d9396c…` 未变 |

**结论**：本会话**未改动任何封存字节**，四项禁止声称均未被触碰；第 1–3 轮的产出全部是**新增**文件。

**唯一待用户决定的事项**：是否把取货线的下一步换向到**多车联合重指派**（Stage A 已给出支持证据：单车后缀最大 6 单、n≥12 机会数为 0），或先推交付线算子改造。二者都需**新的假设与新协议登记**，本会话**未**擅自启动。




