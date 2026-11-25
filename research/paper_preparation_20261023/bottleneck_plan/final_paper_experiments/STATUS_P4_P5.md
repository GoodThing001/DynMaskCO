# P4/P5 轮次执行状态（供继续执行参考）

## 本轮已完成
- 文档统计修复：FINAL_REPORT/README 改为 3,744 累计端点、3,348 统一18法主实验端点口径。
- P4 预登记（P4_预登记.md）：C1=fold0 折外几何（历史 confirm01 已访问）、C2=fold0 验证几何；
  9 例冷链投影注册（SHA 由远端 stage 从工作区 load_original 导出，几何哈希用 fold0 定义）；
  种子 67001–67003；486 端点；轨迹 5/15/30/60。
- P4 执行包：p4_dispatch.py（MASK/SLOT→compare链；RECENT5→recent_worker；其余→external_common）、
  p4_trajectory_worker.py（直接时钟补丁，构造式无时钟记 no_search_clock）、p4_stage.py（导出/复制链/初解准入/核心准入）、
  build_p4_matrix.py、campaign_transport p4 spec。
- P4 冒烟迭代修复：①admission 需 restricted_scope；②install 只能调用一次（ProfitTask 被适配器补丁）；
  ③run_dev72_clock 在 RUN10 workspace tools/（worker 需显式加路径）；④RECENT 方法必须走 recent_worker
  （external_common 无 CaDA 绑定）。当前第 4 轮冒烟运行中（后台 job pwsh-7 完成后自动回收）。
- P5 数据：本地 Solomon 56 原件（data/classic_static_sources_20261020/solomon_100/，FILES_SHA256 清单）；
  远端 CVRPLIB Uchoa X 100 例已下载（tmux cvrplib_download，BKS 修复后重跑中）；
  远端 CVRPLIB Solomon VRPTW 集（id 380–435）实例+BKS 下载中（tmux solomon_bks）；
  全部 SHA 登记于远端 _maskco_final_paper_p5/*/source_manifest.json，稍后回收本地。
- P5 预登记（P5_预登记.md）：种子 68001–68003 / 69001–69003；16通道中性编码；n>100 窗口规则
  （仓库+40遮+60近邻）；CVRP 距离口径待 BKS 复核锁定；VRPTW 字典序 (K,D)。
- standard_vrp/core/standard_evaluator.py：CVRP/VRPTW 实例解析、评价、Split DP（CVRP 距离 / VRPTW 字典序）；
  tests/test_enumeration.py 全部通过（5客CVRP枚举1920车队、5客VRPTW字典序枚举、可行性/等待/字典序单测）。

## 关键命令（Windows PowerShell）
- P4 冒烟状态：job_output pwsh-7（完成后自动 follow/recover）
- P4 完整提交：python -B scripts\campaign_transport.py p4 submit（冒烟通过后）
- P4 审计/分析/表11：待写（p4_analysis.py：C1/C2 分报告、18法表、配对增量、轨迹汇总）
- 数据回收：SFTP 拉取 _maskco_final_paper_p5/{cvrplib,solomon} 到本地 P5_standard_vrp/data/
- BKS 验证：用 standard_evaluator 复算 X 集 BKS 路线（double vs 取整两种口径锁定官方规则）；
  Solomon BKS 从 CVRPLIB 下载的 .sol 复算，并与本地 SINTEF 原件坐标比对。

## 下一步（按序）
1. P4 冒烟核验 18 方法全过 → 提交 P4 完整 486 批次（monitor）→ 独立复算审计 → 表11 + P4 图。
2. P5 数据回收 + BKS 复算锁定距离口径（9.2）。
3. 方法支持矩阵（8.2 身份审计）初版登记。
4. standard_vrp 求解适配：标准 masked repair（Python 先行验证，n=100）→ C++ 动态尺寸内核；
   Split 用 standard_evaluator 的 DP；中性特征构建 + n>100 窗口。
5. P5-A 冒烟（n=100 若干例）→ 三阶段正式批次；P5-B 冒烟 → 56 例批次；泄漏审计。
6. 表12–16、8 类图、FINAL_REPORT_P4_P5.md、状态更新、git_deliver 清单。

## 关键事实
- 远端：/home/hzeng/project/MASKCO-Main；P4 目标 _maskco_final_paper_p4/p4（smoke01 验证中）；
- run_dev72_clock 路径：…/comparison_stage3_20261022/run10_six_methods_dev72/recovery_preparation/workspace/C-VRP_Cold-chainVehicleRoutingProblem/tools/run_dev72_clock.py
- CVRPLIB X：set-17，实例 id 158–257；Solomon VRPTW：set-22，id 380–435（C101=380）。
- 统一 runner 在任一 cell 失败时全批停止（官方语义）——完整批次前必须冒烟全绿。
- 冷链回归义务：P5 开发不改动冻结链与 P0–P3；standard_vrp 目录与冷链完全隔离。
