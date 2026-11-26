# P5 标准任务求解器源码纠错、机制恢复与受控复验（P5-V2）

> **状态：已完成并冻结归档（2026-11-26，实验终止迭代）。** 起始 commit
> `736414664773d83ef55007a541b8bfb6ca3aaae0`，交付 commit `350e18f0`（分支 `codex/o0cc-cal-phys`）。
> 归档包与逐文件 SHA256 清单见 `ARCHIVE_NOTICE.md`。P0—P4 冷链实验、锁版机制、`f0_r0` 检查点
> 与 P5-V1 全部冻结，本目录未修改其中任何文件。

## 结论速览（全部引用本目录真实统计文件）

| 问题 | 结果 | 出处 |
|---|---|---|
| V1 的"神经无正贡献"能否归因于网络？ | 不能：V1 有 D1–D5 行为缺陷 + F1–F5 编码偏离 | `audit/V2_缺陷复现记录.md` |
| 修复后机制恢复是否有增益？ | 有：STD_FIX 相对仅修缺陷 **−4.14%**（43/48 单元） | `development_results/dev01_final/` |
| 神经引导在正确实现下是否有正贡献？ | 仍为负：CVRP +1.09%、VRPTW +0.98%（p<0.01） | `tables/V2-7_neural_contribution.csv` |
| 编码对齐（STD_NORM）是否有效？ | 无可测效果（CVRP p=0.20、VRPTW p=0.87） | `tables/V2-6_norm_effect.csv` |
| 与外部基线差距？ | CVRP Gap 6.3%–11.9%；PyVRP VRPTW 几乎贴住 BKS | `tables/V2-2`、`V2-ext` |
| 全流程可靠性？ | 3684 端点（240 开发 + 3276 正式 + 168 外部）零失败、零复算不一致 | `*/audit*.json` |

完整叙事与限制见 `FINAL_REPORT_V2.md`。

## 本目录结构

| 路径 | 内容 |
|---|---|
| `FINAL_REPORT_V2.md` | 最终报告（结论、方法、结果、审计、限制） |
| `SOURCE_AUDIT.md` | 锁版机制与 V1 差异的源码级审计 |
| `BUGFIX_REPORT.md` | D1–D5（V1）、D6–D8（V2 自身）逐条缺陷、修复与证据 |
| `MECHANISM_PARITY.md` | 机制对照表（M1–M9）、候选语义、已知差异 |
| `EXPERIMENT_PROTOCOL_V2.md` | 先行登记协议（5 版本、面板、计时、口径、诚实性条款） |
| `audit/` | 缺陷复现证据、参考口径核验、外部基线矩阵、冻结身份核验 |
| `standard_vrp/` | V2 求解器（core/adapters/solver）+ 31 项测试 |
| `standard_vrp/core/calibrated.json` | 锁版 `calibrated.json` 的逐字节副本（包内自足） |
| `development/` | 24 个开发实例 + manifest（含构造参考 K、车队上限对齐记录） |
| `configs/` | 全部 phase（开发/正式/外部/诊断） |
| `development_results/` | 开发面板端点（`dev01_final` 有效；`dev01_prefix_D6`、`dev01_pre_D8` 为作废证据） |
| `formal_results/` | 正式面板端点（cvrp / vrptw / norm）与复算审计 |
| `external_results/` | PyVRP VRPTW 补齐（168 端点）与复算审计 |
| `diagnostics/` | 固定候选数量诊断（48 单元） |
| `tables/` | V2-1…V2-10 + 诊断 + 外部表（CSV / MD / LaTeX / `P5V2_tables.xlsx`） |
| `figures/` | F01–F12（PNG 300dpi + PDF） |
| `manifests/` | 冻结身份、部署包 SHA 回执 |
| `scripts/` | 复现入口（见下） |

## 复现入口（`python -B scripts/<name>.py`）

| 脚本 | 作用 |
|---|---|
| `p5v2_frozen_check.py` | 冻结源 89 文件 SHA 核验 |
| `p5v2_dev_deploy.py` / `p5v2_formal_deploy.py` / `p5v2_external_deploy.py` / `p5v2_diag_deploy.py` | 各面板部署与启动 |
| `p5v2_worker.py` | 远端执行器（5 版本 + 6 内部组合，单端点 60 s） |
| `p5v2_audit.py` | **独立**复算审计（不导入求解器包） |
| `p5v2_dev_analysis.py` / `p5v2_formal_analysis.py` / `p5v2_ext_analysis.py` / `p5v2_diag_analysis.py` | 统计与表 |
| `p5v2_figures.py` | 12 张图 |
| `p5v2_tables_export.py` | CSV → MD/LaTeX/Excel |
| `p5v2_report_finalize.py` / `fill_report_dev.py` | 报告数值生成（数字全部来自统计文件） |
| `p5v2_receipt.py` | 远端部署包与本地逐文件 SHA 对照 |

测试：`python -B -m pytest standard_vrp/tests -q`（31 项）。

## 运行身份（远端）

- 解释器：`/home/hzeng/envs/MASKCO_env/bin/python`（torch 2.6.0+cpu、JAX CPU）
- 数据：`_maskco_final_paper_p5/{cvrplib,solomon}`（只读）
- 结果目录：开发 `/tmp/p5v2_dev/dev01`、正式 `/tmp/p5v2_formal/{cvrp,vrptw,norm}`（`/home` 仅余 ~100MB）
- 单物理核（taskset）+ `JAX_PLATFORMS=cpu`、`CUDA_VISIBLE_DEVICES=`；内部截止
  `seconds − max(0.8, 1.0+0.0015n)`，硬截止再留 0.3 s，槽位粒度守卫保证 `online_s ≤ 60`。

## 冻结边界

- 未修改：`maskco_perishable_external_20261119/reserved_validation/solver/*`、
  `f0_r0` 检查点、`P5_standard_vrp/*`（P5-V1 源、结果、日志、统计、图）、P0–P4 全部结果。
- 未开启：32 个 FLEET_PRICE 封存实例。
- 未做：重新训练、微调、按正式结果调参、删除负结果。
