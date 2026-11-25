# P4/P5 轮次执行状态（第 19 轮更新）

## 本轮完成
- **磁盘满事故处理**：远端 /home 100% 满导致三批进程终止（1.7T 已满，占满者为既有数据；
  本次清理自身可重算产物释放 21MB）；改用 sftp 逐文件回收（tar 需额外空间）并清理
  已回收工作区；修复 pkill 自匹配、p5_resume 的 worker 选择（内部 vs 外部）、
  runner 超时（RRNCO 需 1500s）、合并时文件名冲突（前缀 r_/r2_ 回收）。
- **P5-A 全量完成（2716 格，0 失败，审计通过）**：
  PyVRP 2.18%（100/100 第 1）＞ ORDER_SPLIT_D 9.79% ＞ DIST/DIST_SLOT 10.14% ＞
  ALNS 10.56% ＞ ORDER_SPLIT 12.23% ＞ MASK 12.31% ＞ SLOT 12.42% ＞ VNS 14.10%。
  **神经热图在标准 CVRP 无正贡献**（与 P4 一致，如实保留）；框架价值来自距离重插+Split。
  P5-A 章节已定稿写入 FINAL_REPORT §4/§4b（含作者模型 CVRP 表）。
- **P5-B（1344 格）与 PyVRP（468 格）均已完成并审计**；作者批次续跑 author5 进行中
  （398 格，6 核，ETA ~21:45 UTC）。
- 交付清单与收据打包脚本就绪（endpoints 目录、PyVRP 去重合并）；git_deliver 工具修复。

## 下一步
1. author5 完成 → 回收（前缀）→ 审计 → P5-A（含作者）/P5-B（含作者）全量分析。
2. 表12–16、八图 → FINAL_REPORT 定稿 → 项目状态/交付清单 → git_deliver --apply + 提交 + 推送。
