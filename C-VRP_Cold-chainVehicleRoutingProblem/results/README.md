# results/ 实验结果索引

> 最后更新：2026-09-02

项目当前状态以 [`../项目当前状态.md`](../项目当前状态.md) 为准。结果是否可比较，以 complete/service-first Gate 和论文优先版 evaluation contract 为准。

## 当前执行位置

P0-R、P0-S、P0-A、P0-U 已完成。下一阶段是 **O0 on-trajectory action-space oracle**。O0 尚未产生正式目录；在 O0 Go 之前不训练 CostAware-M0。

## 状态总表

| 目录 | 状态 | 用途 |
|---|---|---|
| `p0r/` | **CURRENT / 已复核** | comparator、service-first、B0 与 VAL32 oracle ceiling 复核 |
| `p0s/` | **CURRENT / 已完成** | RecourseSnapshotV2 round-trip 与 resume parity |
| `p0a/` | **CURRENT / 已完成** | full-fleet Action Contract v1 测试 |
| `p0u/` | **CURRENT / 已完成** | common-continuation utility teacher 测试 |
| `jf1_three_way/` | **VALID BASELINE** | JF1-H/JF1-R/JF1-S 联合分配基线 |
| `r1_5_model_utility/` | **EVIDENCE** | H0-H3、guard 和 learned-preference utility 诊断 |
| `r1_7/` | **EVIDENCE** | proposal、state leverage 与 guard 归因 |
| `hfr/` | **FROZEN NEGATIVE** | HFR-M0 训练信号与 Gate A/F3 负结果 |
| `jf2/`, `jf2_closure/` | **FROZEN NEGATIVE** | exact-vehicle symmetry、utility 与 closure 诊断 |
| `jf15_b1/`, `jf15_b4/`, `jf15_b8/`, `jf15_b16/` | **FROZEN DIAGNOSTIC** | assignment-beam null result |
| `b0_gap_decomposition/` | **SUPERSEDED** | comparator 修复前结果；由 `p0r/b0_gap_decomposition/` 取代 |
| `b0_gap_decomposition_smoke/` | **SMOKE** | 调试结果，不用于科研结论 |
| `r1_baseline/` | **INCOMPLETE/HISTORICAL** | 旧 strict-online matrix，不是当前主表 |
| `phase0_baseline_freeze/` | **INVALIDATED** | P0 审计前旧协议结果 |

## 当前可引用的 P0 产物

### P0-R

- `p0r/evaluation_protocol_tests.json`
- `p0r/b0_gap_decomposition/summary.csv`
- `p0r/b0_gap_decomposition/service_matrix.csv`
- `p0r/oracle_ceiling/oracle_ceiling.csv`
- `p0r/oracle_ceiling/interaction_stats.json`
- `p0r/oracle_ceiling/service_matrix.csv`

复核结果：B0 `G_fleet=17.891%`、`G_seq=-1.227%`；VAL32 interaction `-1.7430`，四路 episode complete 100%。

### P0-S / P0-A / P0-U

- `p0s/snapshot_roundtrip_tests.json`
- `p0a/action_contract_tests.json`
- `p0u/counterfactual_teacher_tests.json`

这些文件证明基础设施测试通过，不等价于 O0 或 CostAware 的性能结果。

## 冻结负结果

### JF2

JF2 exact physical-vehicle classification 受车辆身份对称性影响；canonical exact-ID 虽提高监督准确率，但 Gate A 仍为 `25.91`，差于 JF1-H `24.50`。该方向冻结。

### HFR-M0

HFR-M0 的 G/A 训练信号成立，但部署端 Gate A 失败：

| 变体 | cost | complete |
|---|---:|---:|
| JF1-H | 24.50 | 100% |
| g_only | 24.97 | 100% |
| min_travel | 25.92 | 98.4% |
| a_only | 26.63 | 100% |
| full | 29.95 | 99.2% |
| group_only | 30.19 | 100% |
| group_logit | 30.07 | 99.2% |

因此 HFR 作为 structural diagnostic 保留，不能宣称为优于 JF1-H 的方法。

## 结果使用规则

1. cost 始终是纯 travel distance，不混入惩罚项。
2. complete、TW、capacity、duplicate、prefix violation 等 blocking 指标必须先通过，才比较 cost。
3. `smoke` 目录只验证代码路径，不支持科研结论。
4. `phase0_baseline_freeze/` 及 `14.77`、`15.45`、`14.08` 等旧协议数字不得作为当前主结果。
5. 新正式实验应保存 `manifest/config`、`summary`、`per_instance` 和必要的 solver/service audit。

## 新结果命名规范

从 O0 开始使用：

```text
results/<stage>/<experiment>/<split-or-seed>/
```

阶段名采用 `o0`、`m0`、`m1`、`a1`、`e1`、`t1`。不要把 smoke、临时修复和正式结果写入同一目录。

