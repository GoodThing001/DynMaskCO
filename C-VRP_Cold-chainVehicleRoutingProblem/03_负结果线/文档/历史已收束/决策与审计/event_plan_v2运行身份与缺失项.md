# event_plan_v2 运行身份封存与缺失项清单

> 2026-09-19 · 实施端本地 CPU 运行后回传。本文件只记录**运行身份（代码/数据/模型/配置）与已知缺失项**，不重跑、不补算、不作出方法决策。方法决策见 [event_plan 两轮收束与后续决策](event_plan两轮收束与后续决策.md)，结果与判读见 [event_plan_v2 表示对照结果](../结果与进度/event_plan_v2表示对照结果.md)。

## 1. 代码身份（event_plan_v2 相关源文件 SHA-256）

| 文件 | SHA-256（前 16） |
|---|---|
| scripts/simulation/event_plan_candidate.py | 367bfeae51801d27 |
| scripts/simulation/event_plan_projection.py | fc45ff73bcd83954 |
| scripts/simulation/event_plan_features.py | a33e8efd2a43a4e5 |
| scripts/simulation/event_plan_replanner.py | bfcc73900f977ec6 |
| scripts/evaluation/run_event_plan_export.py | f3fbd4514f93069b |
| scripts/training/train_event_plan.py | f5b14324dac828bf |
| scripts/evaluation/run_event_plan_gate.py | 571caa4262bfd54d |
| scripts/evaluation/run_event_plan_select.py | fa25efba977e51bb |
| scripts/evaluation/run_event_plan_closed_loop.py | d72d5df168e8f028 |

## 2. 数据身份（本地重新生成 data/m0_scale/）

| 文件 | n | seed | sha256（前 16） |
|---|---|---|---|
| dcc_50_r1_edod05_train_teacher.npz | 64 | 9701 | 26fbf1201b09377b |
| dcc_50_r1_edod05_cal_teacher.npz | 16 | 9702 | 3dfbf8844d66cad2 |
| dcc_50_r1_edod05_dev_check_teacher.npz | 16 | 9703 | 9c74d1dda7f44fec |

> 复现口径：执行人报告 inst0–2 的 18 个事件 event/clock/context 与 g_B 列表到 6 位小数一致、keep_J 差 ~2e-8；**未做服务器文件哈希或数组逐项比对**，故只算「数值/事件一致」，不声称逐字节一致。

## 3. 模型身份（6 个 checkpoint，共用 s=0.0520316）

| 模型 | 表示 | seed | in_dim | checkpoint sha256（前 16） |
|---|---|---|---|---|
| event_plan_v2_A_s42 | A | 42 | 125 | 3d8f97c8a9c63fca |
| event_plan_v2_A_s43 | A | 43 | 125 | f52b215ad8b5d298 |
| event_plan_v2_A_s44 | A | 44 | 125 | 7679ec82782a7af9 |
| event_plan_v2_B_s42 | B | 42 | 554 | f9c75e07cbda257f |
| event_plan_v2_B_s43 | B | 43 | 554 | 0fbd0146ead00967 |
| event_plan_v2_B_s44 | B | 44 | 554 | 3e13caf9474731fb |

配置：`PlanEvalMLP(128,64)` GELU；Huber δ=1.0；adamw(1e-3, wd=1e-2)；2000 步；batch 8；实例→事件→候选平衡抽样。合同 hash（effective，应用 profile 后）与 profile hash 见各 export `data.json` summary。

## 4. 结果位置

- 导出：`results/m0_scale/event_plan_v2_export/data.json`（TRAIN）、`event_plan_v2_export_cal/data.json`（CAL）。
- 模型：`results/m0_scale/event_plan_v2_{A,B}_s{42,43,44}/`。
- 同状态选择：`results/m0_scale/select_{train,cal}_s{42,43,44}/selection.json`。
- 闭环：`results/m0_scale/closed_loop_v2/closed_loop.json`。

## 5. 已知源码限制（复用投影/B 特征前须先处理；本轮不修复、不重跑）

1. **WAIT 分支未推进到公开 horizon H**：`project_vehicle` 对空 suffix + has_future 只置 `open_plan=True`、`return_time=anchor_time`，没有按工作包约定「推进已知等待至 H」的那段 C0 转移——该等待段的品质/能耗增量未计入代理。影响 `d_quality`/`d_energy` 的绝对量（delta 上对 frozen 车抵消，但 mutable 车的 WAIT 代理不完整）。
2. **committed 载重摘要重复加需求**：`build_vehicle_plans` 对 committed 车已把 `anchor_load = current_load + demand[committed_next]`，`project_vehicle` 的 committed-leg 块又 `load += demand[node]` 一次——`max_load` 等字段被高估（环境实际 cargo 不重复入舱）。
3. **训练 seed 兼作生成 seed**：`run_event_plan_closed_loop.py` 把 learned 方法的 `seed=sd`（42/43/44）同时传给候选生成器 `generate_simple_candidates_r1`，而导出与 distance/proxy 对照用 `seed=0`。A/B 同 seed 仍可配对，但跨 seed 与对照的差异混入候选生成变化，不能全归训练随机性。

## 6. 缺失项（如实记录，不自动补齐重跑）

- 闭环 runner 只写汇总 `closed_loop.json`，**未保存方法逐实例 rows**；`CAL runtime=0` 是占位（`_summarize` 对 CAL 传 `runtime=0.0`）。
- 未做服务器文件哈希 / 数组逐项核对（见 §2）。
- 未提交仓库；未生成正式 seal/archive（本轮为开发运行身份记录，非 `freeze_source_archive.py` 正式封存）。

## 7. 复用前置（仅当未来决定复用这些资产时）

投影引擎/B 特征/闭环链路在复用前需：补 §5 三项修复并做针对性回归；保存逐实例产物与真实计时；在服务器上核对数据字节身份；重定「学习模块职责 + 同信息同预算可部署对照」后再决定是否重训。
