# 负结果 / 方法学论文 · 独立复现文件夹

> **当前阅读顺序（2026-09-26）**：先看 [投稿前端到端审计](docs/01_论文/方向B投稿前端到端审计_2026-09-26.md)，再看 [完整初稿](docs/01_论文/论文完整初稿_2026-09-25.md) 与 [C1 预注册](docs/01_论文/独立确认实验预注册_C1_2026-09-25.md)。本 README 与 2026-09-23 的短稿/主表含历史数字和旧判读，不能作为投稿数值来源；统一 v1 主表以 `results/09_unified_v1/`、C1 以 `results/10_confirm_c1/` 为准。C1 预注册的三项联合成功判据**未通过**；未来采样回退路径已核验（零触发，见审计执行端收口记录）；平行 A-v1 研究的修正后负结果与 B 的一致性见 [一致性证据](docs/04_证据链/方向A修正下游负结果_对B的一致性证据_2026-09-26.md)。

本文件夹是 **B-ii（方法学/负结果论文）** 方案的独立归档。目标：如果换任务/合同的路径走不通，可以从这一个文件夹**完整复现**整条负证据链，作为论文的实证核心。

> 定位（已收窄，2026-09-22）：这不是"性能创新论文"，也不是"非学习论文"——它是**"MaskCO 在动态冷链路径优化中的迁移与失效边界"的系统性实证研究**（从局部修复质量到在线决策效用）。核心方法仍是 MaskCO（满足导师"基于 MaskCO 的动态冷链物流方法优化"），但贡献是负结果 + 机制诊断，不是超越基线。
>
> ⚠️ 早期版本主张"未来信息主导 / 信息论上无信号 / 不可学习"，该结论**尚未被实验证明**，已按证据边界收窄。本 README 以下文表述为准；完整审计见 [论文设计与证据审计](docs/01_论文/论文设计与证据审计_2026-09-22.md)。

> 投稿前新增源码核查见 [投稿前核查（2026-09-24）](docs/01_论文/投稿前核查_2026-09-24.md)：未来采样器的需求范围与原始生成器不一致、动作间未共享场景；主性能表不受直接影响，但机制论证需要据此收窄或复核。

## 1. 核心主张（一条可发表的负证据链）

在当前合同/规模（50 节点 R1 EDoD=0.5，pickup-to-depot，objective = distance + quality + energy）下：

> **在所研究的动态冷链协议、适配架构和计算预算下，MaskCO 微调改善了局部重构质量，但未建立相对距离启发式搜索的稳定在线增量。分层诊断表明，候选互补性有限，且局部目标改善与终局收益存在明显脱节；未来信息与后续重规划的作用仍需通过受控实验进一步区分。**

八步证据链（每步有脚本 + 结果 + 数字；②b 未来信息 vs 评价时域对照见下）：

| 步 | 诊断 | 脚本 | 关键结果 | 结论 |
|---|---|---|---|---|
| 1 | 搜索增强 | `run_sgbs_fixed_state.py` | dist_sgbs 1.0617/1.1447 < R 1.0697/1.1515 < Mtrained 1.0789/1.1528 | 学习输给距离搜索 |
| 2 | 互补性 | `analyze_complementarity.py` | gain_union 0.0015/0.0005（<0.15%） | 已生成候选集的平均互补收益很小 |
| 3 | myopic 头腔（D1/D2） | `run_headroom_census.py` | gap_exh_vs_dist_sgbs 0.003–0.005 | myopic 头腔存在但稀疏 |
| 4 | 终局前瞻（D3） | `run_terminal_headroom.py` | 终局均值未见正收益，CI 跨零（~2/3 状态=0） | myopic 头腔未稳定转成终局收益 |
| 5 | 因果头腔 | `run_causal_headroom.py` | myopic 顺序 oracle +0.09/+0.11（有害）；clairvoyant −0.21/−0.27 | myopic 评分有害；未来信息 vs 评价时域未分离 |
| 6 | 价值可预测性 | `run_value_predictability.py` | 修正后线性探针 out-of-sample Spearman ≈ −0.073，R²<0 | 所测 16 维特征+线性探针未检出样本外预测力（探索性） |
| 7 | 易腐排序头腔（D7-a） | `run_perishability_headroom.py` | 重排净收益 −0.13~−0.15（品质 +0.08~0.11 盖不住距离 0.20） | 品质项太弱，撑不起偏离距离最优 |
| 8 | 品质重标定终局（D7-b/b'/c） | `run_terminal_perishability.py` 等 | ×5 时 myopic 头腔转正（+0.141），但终局头腔仍 ≈0（CI 跨零） | 品质量级不改变终局头腔 |

参照（clairvoyant 顺序 oracle，提供 −0.21/−0.27 上界）：`run_action_oracle.py` → `o0cc_seq_{cal,devcheck}/summary.json`。

> **证据边界**：以上八步共享实例/状态/评价器/后续策略，是**相互关联的诊断**，不是八次独立确认。各步能/不能主张的结论见 [论文设计与证据审计](docs/01_论文/论文设计与证据审计_2026-09-22.md) §2 的 CAN/CANNOT 表。要点：步骤 5 的 "≈100% 未来信息" 未控制"评价时域"与"未来服务过滤"两个混杂（clairvoyant 还做终局服务过滤，myopic 没有）；步骤 6 的线性探针不能证明"不可学习"；步骤 4 测的是"局部最优方案的终局效果"而非"终局最优方案"。
>
> **②b 受控对照（2026-09-24 修复采样器后重跑）**：`run_future_sampling_oracle.py` 在相同状态/候选集/终局时域下比较当前可见目标、所用重采样器的有限样本未来评分、真实未来评分。修复后（demand 1–10 与生成器一致、候选间共享未来场景、重试兜底自洽）N=8 时，full −0.005 [−0.047,+0.033]、attr +0.017 [−0.011,+0.046]，clairvoyant +0.045 [+0.019,+0.074]；**配对差异** clairvoyant − dist_full = +0.050 [+0.022,+0.082]、− dist_attr = +0.028 [+0.014,+0.043]（CI 均不含零）。**当前采样评分未恢复预知未来评分的平均收益**；clairvoyant 是同候选集特权上界，差异非负是构造结果，不证明价值不可预测。full 的重采样规则尚未证明等于条件于公开状态的真实分布；attr 保留真实揭示时间，含部分未来信息。详见 [机制实验结果](docs/02_机制实验/机制实验结果_未来信息vs评价时域_2026-09-22.md)。
>
> **②a 动作追踪（2026-09-22）**：`run_execution_trace.py` 比对候选分配与最终服务车辆，两名掩码客户均按候选车辆服务的状态为 dg 0/30、exh 1/30。该探针未核对完整路线执行；后续重规划覆盖是与观察一致的解释。详见 [机制实验结果](docs/02_机制实验/机制实验结果_局部改善为何消失_2026-09-22.md)。
>
> **③ 种子稳健性 + 留出实例 + 闭环（2026-09-23，服务器）**：seed 43/44 训练 + 固定状态评估 + 留出闭环。固定状态 M-trained 不超 R（3 seed + 留出），**闭环里 M-trained 与 R 差异不显著**（seed42 +0.0002 / seed43 +0.017，CI 跨零），显著优于 M-pre（−0.104/−0.087）。"优于 M-pre" 对 seed 敏感（seed44 塌缩）。详见 [种子稳健性与留出实例](docs/03_稳健性/种子稳健性与留出实例_2026-09-23.md) 与 [结果主表](docs/01_论文/结果主表_锁定_2026-09-23.md)。

## 2. 文件结构

```
negative_result_paper/
  README.md
  scripts/                      # 诊断入口脚本（副本）
  docs/
    01_论文/                    # 初稿 / 证据审计 / 结果主表 / 复现包 / 导师总结
    02_机制实验/                # ②a 执行追踪 + ②b 未来信息vs时域（设计+结果）
    03_稳健性/                  # seed 稳健性 / 阈值等效性 / EDoD 条件轴
    04_证据链/                  # 八步证据链各步结果文档
    05_历史分支/                # 更早的负结果（event_plan / cc_lns / n1n2 / swap / M-pre 直接训练）
    06_研究设计/                # 研究设计 / 决策 / 工作包
  results/
    01_oracle/                  # clairvoyant 顺序 oracle
    02_搜索互补/                # sgbs_fixed + complementarity
    03_头腔普查/                # headroom_census + terminal_headroom (D1/D2/D3)
    04_因果可预测/              # causal_headroom + value_predictability
    05_易腐/                    # perishability (D7-a) + perish_terminal/commit/oracle (D7-b/b'/c)
    06_机制实验/                # execution_trace + future_sampling_oracle (②a/②b)
    07_揭示度轴/                # reveal_axis_edod*
```

`results/` 内 JSON 命名约定：`<原目录>__<文件名>`（如 `02_搜索互补/sgbs_fixed_train_v2__summary.json`）。

## 3. 复现步骤（从零）

依赖（不在本文件夹内，需在仓库根目录提供）：
- **数据**：`data/m0_scale/dcc_50_r1_edod05_{train,cal,dev_check}_teacher.npz`（50 节点 R1 EDoD=0.5）。
- **objective profile**：`results/o0cc/scale_v2/objective_profile.json`（`o0cc-pilot-devmean-equal-v2`）。
- **checkpoint**：`MASKCO_code/ckpts/cvrp100.ckpt`（CVRP 预训练）+ `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt`（M-trained 直接目标训练产物，SHA256 `f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5`，89.6MB 不入库；`mpre_reinforce_s42_defective/`、`mpre_reinforce_s42_corrected/` 是已废弃旧缺陷 checkpoint，勿用于论文结论）。
- **代码**：本脚本依赖仓库的 `scripts/{simulation,evaluation,coldchain,data,models,training,baselines}` 模块（`strict_online_env`, `jf1h_repair`, `action_contract`, `cc_lns_replanner`, `visible_state`, `counterfactual_teacher`, `sequential_oracle`, `recourse_snapshot`, `coldchain_contract`, `coldchain_state`, `mpre_policy`, `mpre`, `mtrained_replanner` 等）。本文件夹的 `scripts/` 只存**入口脚本**，不重复模拟层代码。
- **环境**：Python 3.10，NumPy（D1–D6 纯 NumPy，无 GPU）；step 1/2 需要 JAX（加载模型）。

命令（在仓库扩展根 `C-VRP_Cold-chainVehicleRoutingProblem/` 下执行，`--data` 用对应 split 的 npz）：

```bash
# 步骤 1：两层搜索（dist_sgbs / R / Mtrained）
python scripts/evaluation/run_sgbs_fixed_state.py \
  --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
  --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --model-ckpt results/m0_scale/mpre_reinforce_s42_clean/model.ckpt \
  --objective-profile results/o0cc/scale_v2/objective_profile.json \
  --split train --out results/m0_scale/sgbs_fixed_train_v2

# 步骤 2：互补性（读步骤 1 的 per_state.json）
python scripts/evaluation/analyze_complementarity.py \
  --per-state results/m0_scale/sgbs_fixed_train_v2/per_state.json \
  results/m0_scale/sgbs_fixed_cal_v2/per_state.json \
  --out results/m0_scale/complementarity_v2

# 步骤 3：D1/D2 myopic 头腔普查（无模型）
python scripts/evaluation/run_headroom_census.py \
  --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
  --objective-profile results/o0cc/scale_v2/objective_profile.json \
  --split train --out results/m0_scale/headroom_census_train \
  --ref-per-state results/m0_scale/sgbs_fixed_train_v2/per_state.json

# 步骤 4：D3 终局前瞻重评（无模型）
python scripts/evaluation/run_terminal_headroom.py \
  --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
  --objective-profile results/o0cc/scale_v2/objective_profile.json \
  --split train --out results/m0_scale/terminal_headroom_train

# 步骤 5：因果头腔（myopic 顺序 oracle，无模型）
python scripts/evaluation/run_causal_headroom.py \
  --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
  --objective-profile results/o0cc/scale_v2/objective_profile.json \
  --max-instances 16 --out results/m0_scale/causal_headroom_cal

# 步骤 6：价值可预测性探针（无模型）
python scripts/evaluation/run_value_predictability.py \
  --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
  --objective-profile results/o0cc/scale_v2/objective_profile.json \
  --max-instances 8 --out results/m0_scale/value_predictability_cal

# 参照：clairvoyant 顺序 oracle（~45 分钟/实例，用 --workers 并行）
python scripts/evaluation/run_action_oracle.py \
  --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz --capacity 50 --num_vehicles 25 \
  --objective coldchain --objective-profile results/o0cc/scale_v2/objective_profile.json \
  --role regression --workers 8 --out results/m0_scale/o0cc_seq_cal
```

## 4. 各步结果文件与关键数字

| 结果文件（本文件夹 results/ 内） | 关键字段 / 数字 |
|---|---|
| `sgbs_fixed_train_v2__summary.json` | `accepted_mean.dist_sgbs=1.0617`, `R=1.0697`, `Mtrained_sgbs=1.0789` |
| `complementarity_v2__complementarity.json` | `gain_union.mean=0.0015(train)/0.0005(cal)` |
| `headroom_census_train__summary.json` | `D1.gap_exh_vs_dist_sgbs=0.0032`, `D2.correction_signal=0.0121` |
| `terminal_headroom_train__summary.json` | `terminal_headroom_exh=-0.0055`(CI 跨零), `terminal_gap_exh_vs_dg=0.0039` |
| `causal_headroom_cal__summary.json` | `mean_delta=+0.0914`(myopic 有害), 对照 clairvoyant −0.2096 |
| `value_predictability_cal__summary.json` | `linear_probe.structural_only.spearman_test=-0.073`（修正并列秩/截距/train-only 后） |
| `o0cc_seq_cal__summary.json` | `mean_delta=-0.2096`（clairvoyant 上界，70% quality + 24% energy） |

## 5. 关键文档（本文件夹 docs/ 内）

- `01_论文/论文完整初稿_2026-09-25.md` — **完整论文初稿**（问题设定/方法/三列表格/图注/22 篇文献；顶部含定稿前必办清单），取代 2026-09-23 短稿骨架
- `01_论文/端到端复现审计.md` — 服务器资产 SHA、源码身份、物理口径混用发现、逐项复现结果与缺口清单（2026-09-25）
- `01_论文/论文初稿_2026-09-23.md` — 旧短稿骨架（历史）
- `04_证据链/方向A修正下游负结果_对B的一致性证据_2026-09-26.md` — 平行 A-v1 研究的修正后负结果（条件前瞻增量恰为 0）与 B 的一致性和边界
- `价值可预测性探针结果_2026-09-22.md` — 步骤 6 + 六步证据链最终闭合
- `因果头腔诊断结果_2026-09-22.md` — 步骤 5
- `头腔普查D1D2D3结果_2026-09-22.md` — 步骤 3/4
- `搜索路线互补性分析_2026-09-22.md` — 步骤 2
- `头腔定位与下一步决策_2026-09-22.md` — 整条路线的方法学决策
- `对比方法可迁移机制审读_2026-09-22.md` — 方法审读（贡献定位依据）
- 其余（`n1n2…`、`cc_lns_swap…`、`event_plan_v2…`、`MaskCO直接目标训练…`）— 更早的负结果（Huber 配方、蒸馏、交换修复等），证明"已排除多条配方路线"

## 6. 论文主张落点（若走 B-ii）

可主张的贡献（诚实、可复现）：
1. **严格的动态迁移与评估协议**：公开信息、不可撤销动作、订单级冷链评价及训练-部署一致性。
2. **局部质量与在线效用脱节的实证刻画**：区分"学到了什么""搜索带来了什么""最后实际执行获得了什么"——即迁移改善为何没有稳定成为在线收益。
3. **可复用的诊断流程**：从候选互补、局部改善空间到终局重评，避免把预知未来的参照误当成可学习的在线目标（穷举 oracle / 终局前瞻重评 / 因果 oracle / 可预测性探针）。

注意：本文件夹当前更准确的称呼是**"复现材料集合"**（依赖外部数据/checkpoint/模块，`scripts/` 存的是入口脚本副本，非独立完整复现包）。③ 的独立确认已补齐（多 seed 43/44、留出实例、预设收益阈值等效性、EDoD 条件轴、留出闭环），完整身份见 [复现包](docs/01_论文/复现包_2026-09-23.md)；投稿时需确保审稿人能取得对应数据/结果/checkpoint。论文主张口径见 [论文初稿](docs/01_论文/论文初稿_2026-09-23.md) 与 [论文设计与证据审计](docs/01_论文/论文设计与证据审计_2026-09-22.md)。
