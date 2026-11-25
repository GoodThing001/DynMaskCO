# P4/P5 交付清单（DELIVERY）

> 生成时间：2026-11-26。全部实验完成、独立审计通过后交付。
> 推送方式：`python tools/git_deliver.py --apply` → 显式提交 → 普通快进推送（已获用户授权，不得 force push）。

## 1. 实验规模与状态

| 批次 | 规模 | 状态 |
|---|---|---|
| P4 Solomon 冷链统一 18 法 | 9 例 × 3 种子 × 18 法 = 486 端点（含 5/15/30/60s 轨迹） | 完成，独立复算通过 |
| P5-A CVRP 内部 8 法 | 100 例 × 3 种子 × 8 法 = 2400 格（stage01/02/03） | 完成，审计 0 问题 |
| P5-A PyVRP | 100 例 × 3 种子 = 300（CVRP）+ 168（VRPTW）= 468 格 | 完成（含 8 例截断登记） |
| P5-A 作者模型（CVRP n=100） | X-n101-k25 × 3 种子 × 7 法 = 21 格 | 完成；RRNCO 无 CVRP 权重（UNSUPPORTED） |
| P5-B VRPTW 内部 8 法 | 56 例 × 3 种子 × 8 法 = 1344 格 | 完成，审计 0 问题 |
| P5-B 作者模型（VRPTW） | 56 例 × 3 种子 × 4 法 = 672 格 | 完成（CaDA 等不可行格逐格登记） |
| 注册矩阵 | P5-A 5400 / P5-B 3024 | UNSUPPORTED 按预登记 §9 登记 |

## 2. 主交付物

- `FINAL_REPORT_P4_P5.md`：总报告（协议、P4/P5-A/P5-B 结果、泄漏审计、结论与诚实声明、注册勘误）
- `tables/表12–表16`（MD/CSV/LaTeX + 导师版 Excel 页签）
- `figures/P5A_图1–4`、`figures/P5B_图1–4`（PDF/SVG/300dpi PNG）
- `P4_solomon_coldchain/`：预登记、调度器、轨迹工作器、486 端点收据、RESULTS.md
- `P5_standard_vrp/`：
  - `P5_预登记.md`（协议与支持矩阵）
  - `standard_vrp/`（评价器/特征/掩膜搜索/Split/经典基线/单元测试）
  - `p5_worker.py`、`p5_external_worker.py`、`methods/`（8 个作者模型适配）
  - `phase_*.json`（各批次注册）、`analysis_p5a.json`、`analysis_p5b.json`
  - `results_*/endpoints/*.json`（逐端点收据：路线、距离/K、初解、在线耗时）
- `scripts/`：审计（p5a_audit）、分析（p5a_analysis/p5b_analysis）、表/图生成、
  回收（p5a_recover*/p5a_recover_direct）、续跑（p5_resume2）、收据打包（p5_package_receipts）

## 3. 审计证据

- 每端点：独立评价器全量重算路线（CVRP 逐边整数舍入 / VRPTW 双精度，容差 1e-6）、
  身份/预算/共同初解（搜索族）核验；
- 终点三态：AUDITED / TRUNCATED_OK（预登记 §6 的 8 个截断实例，超窗 ≤0.11）/ INFEASIBLE
  （仅构造式方法，逐格登记并计数）；
- 冒烟：P5-A 36 格、P5-B 24 格，独立审计 0 问题；
- 正式批次：stage01 528、stage02 1104、stage03 781、vrptw_full 1344、PyVRP 504（含重复键）、
  作者批 ~700，审计 0 问题（合并目录的多期重复以 (case,seed,method) 去重计）。

## 4. 已知边界与诚实声明

- 神经热图（ORDER_SPLIT vs ORDER_SPLIT_D）在 P4/P5-A/P5-B 均为**负贡献**，如实保留；
- 冷链 f0_r0 训练集 = 38 例 R/RC（rc205 除外），验证 C2、折外 C1——P5-B 中 38 例训练几何
  不得称未见数据（表16/图3 逐例标注）；
- CaDA 为"约束感知双注意力"模型（非扩散），适配按作者默认 POMO 式解码；
- RRNCO 作者默认 num_starts=100 单核 ~490–530s，按构造式预算档（≤1200s）如实记录；
- 车队口径：CVRP 距离目标不把 Uchoa k 作为硬约束（注册勘误，FINAL_REPORT §2）；
- 远端磁盘曾满：收据以 sftp 逐文件回收、截断文件剔除并重跑补齐，最终缺口为 0。
