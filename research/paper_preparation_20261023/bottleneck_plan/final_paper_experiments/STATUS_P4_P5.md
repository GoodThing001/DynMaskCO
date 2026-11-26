# P4/P5 轮次执行状态（第 20 轮 · 收口）

## 全部完成并已推送 GitHub
- **P4**（Solomon 冷链，486 端点）：ORDER_SPLIT 面板 1/13（2/18），对 ALNS +83.58/+59.63 U；
  神经热图净贡献 −5.32 U（9/27 正）。
- **P5-A**（Uchoa X 100 例，2737 格）：PyVRP 2.18% 全面第 1；ORDER_SPLIT_D 9.79% ＞ DIST 10.14%
  ＞ ALNS 10.56% ＞ ORDER_SPLIT 12.23% ＞ VNS 14.10%；作者模型（CVRP n=100，21 格）CaR 3.61%、
  RouteFinder 4.89%、POMO 5.47%、CaDA 5.82%、MVMoE 6.42%、AM 10.39%、DeepACO 14.76%，
  RRNCO UNSUPPORTED（无 CVRP 权重）。
- **P5-B**（Solomon 56 例，2023 格）：ALNS 最优 9.3/1039.9；ORDER_SPLIT_D 9.5/1048.8；
  DIST 9.6/1049.7；ORDER_SPLIT 10.2/1091.6；VNS 12.9/1177.2；作者 VRPTW
  RouteFinder 12.5/1366.9、MVMoE 13.3/1504.6、CaDA 14.3/1475.0、RRNCO 14.5/1493.4。
- **审计**：全部批次独立评价器重算 0 问题（AUDITED / TRUNCATED_OK×3 / INFEASIBLE×6 逐格登记；
  重复键按 (case,seed,method) 去重）。
- **泄漏审计**：f0_r0 训练 38 例 R/RC（rc205 除外）、验证 C2、折外 C1，逐例标注（表16/图3）。
- **交付物**：FINAL_REPORT_P4_P5.md、DELIVERY.md、表12–16（MD/CSV/LaTeX/Excel）、
  P5A_图1–4 + P5B_图1–4（PDF/SVG/PNG）、全部端点收据（endpoints）。
- **Git**：提交 fa08a47d，快进推送 origin/codex/o0cc-cal-phys（6421 文件，无 force push）。

## 关键负结果（如实保留）
冻结冷链神经热图在标准 CVRP 与标准 VRPTW 上均无正贡献；框架可迁移价值在距离重插 +
跨车 Split 结构。未通过调参或删例掩盖。
