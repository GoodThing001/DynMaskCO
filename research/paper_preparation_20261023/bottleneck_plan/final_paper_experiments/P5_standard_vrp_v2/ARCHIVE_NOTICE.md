# 冻结通知（P5 / P5-V2 实验终止迭代）

> 本目录及其并列的 `../P5_standard_vrp/` 已于 **2026-11-26** 冻结归档，**不再迭代**。

## 状态

- 终止迭代：不再新增端点、不补跑、不调参、不换绑定、不删例、不重训。
- 两棵树的完整快照、逐文件 SHA256 清单与归档包见仓库外：
  `D:\PyCharm_\MASKCO-Main\archive\p5_frozen_20261126\`
  - `P5_P5V2_FROZEN_20261126.tar.gz`（11.7 MB，含 `P5_standard_vrp/**` 与 `P5_standard_vrp_v2/**`）
  - `MANIFEST.json`（26,276 文件逐文件 SHA256）、`INDEX.csv`、`ARCHIVE_RECEIPT.json`、
    `VERIFY_RESULT.json`（解包自校验：0 缺失、0 不符）、`ARCHIVE.md`（完整归档说明）
- 对应 git 提交：`350e18f0`（分支 `codex/o0cc-cal-phys`，远端同 SHA）。

## 冻结前最后一次有效结果

| 面板 | 端点 | 独立复算不一致 |
|---|---:|---:|
| 开发面板（5 版本） | 240 | 0 |
| 正式 CVRP（6 组合） | 1,800 | 0 |
| 正式 VRPTW（6 组合） | 1,008 | 0 |
| STD_NORM | 468 | 0 |
| 外部 PyVRP VRPTW | 168 | 0 |
| 固定候选诊断 | 48 | 不适用 |

结论与限制见 `FINAL_REPORT_V2.md`；缺陷全链条（含 V2 自身 D6–D8）见 `BUGFIX_REPORT.md`。

## 后续

如需继续研究，必须新建目录与新身份（例如 `P5_standard_vrp_v3`），重新登记协议与冻结清单；
**不得**复用本轮身份、覆盖本轮结果、或把本轮作废面板（`dev01_prefix_D6`、`dev01_pre_D8`）
混入新的汇总。
