# A-v1 动态冷链接单协议 · 对比方法总表

> 生成器：`CC_Compare/tools/collect_compare_table.py` + `render_compare_table.py`。
> 协议：动态 pickup-to-depot；揭示即接/拒不可撤销；同容量/车辆数/预算/温区/时间窗；
> 决策时限 10s；**同一天 40 天开发集（seed 20260926）**；效用 = 收入 − 燃油 − 拒绝损失。
> 统计 = 按天聚类 bootstrap（seed+777，2000 次重采样，2.5/97.5 分位），逐位复刻项目 `_stat`。
> 基线 = `a1_c1rr2_reveal_40`（身份块：有，source_stable=False）。
> **正式裁决：已达成（臂侧落盘）**。主行取自 v2 正式批次 `a1_ext_formal2` 中**需要有配对身份的 34 个臂**（全部 `identity_verified=true`），各臂在**自身 gate 内**以基线 `a1_c1fresh_reveal_40`
> 完成配对并落盘 `formal_adjudication=true`（3 档）—— 这是本批次相对上一版的关键升级。
> 另有 **6 个纯规则臂**（myopic、valuee20、valuee30）**不参与上述裁决声明**：它们没有学习权重，
> 产出脚本 `run_a1_downstream_ablation.py` **从不调用** `pairing_identity_check`（源码已核：0 处出现），
> 故其 gate 结构性没有 `_pairing_identity` 块 —— 这是设计如此，**不是缺陷**；它们的 Δ 列同样由本表以身份合格基线的 per-day 重配对给出。
> ⚠️ **但本表的 Δ 列是另行重配对的，不等于上述裁决**：为与上一版逐位可比，Δ 列统一改用
> `a1_c1rr2_reveal_40` 的 per-day 重算，而该基线在自身记录里 `source_stable=False`
> （见上一行「基线 = …」）。⇒ **Δ 列本身仍属「重配对证据」**，并因新基线更优
> （超时更少）而系统性偏移：相对新基线的差比 Δ 列小约 2–7。引用 Δ 时必须一并说明；
> 臂侧的 `formal_adjudication=true` 不受影响。详见 `v2_baseline_verdict.md`。

## 0. Ours（主模型）摘要

> ⚠️ **以下主模型臂未纳入本节的 Ours 块**（`filter_main_model` 只认 `MAIN_MODEL_ARMS`，未登记即静默剔除）。附实测数字与 `超时/天`，请自行判断是「有意排除」还是「新臂忘了登记」：
>
> | 臂 | 档位 | utility | 超时/天 | 来源 gate |
> |---|---|---:|---:|---|
> | `a1_consensus` | p_c=(10,20,30) | -464.5 | 42.35 ← **超时污染，只能作诊断** | `results/a1_a1_formal_40_1004_171907/gate.json` |
> | `a1_consensus` | p_c=(5,10,15) | 605.5 | 45.92 ← **超时污染，只能作诊断** | `results/a1_a1_formal_40_1004_171907/gate.json` |
> | `a1_consensus` | p_c=0 | 1687.6 | 46.40 ← **超时污染，只能作诊断** | `results/a1_a1_formal_40_1004_171907/gate.json` |
>
> 若这是你**新改的**主模型臂，请把臂名加入 `CC_Compare/tools/render_compare_table.py`
> 的 `MAIN_MODEL_ARMS`（必要时也加进其下方 `order` 字典控制排序）后重渲染 ——
> 否则总表会**只显示旧臂**，看起来像「改进没生效」。若属有意排除（如上表超时污染），
> 忽略本告警即可。

| 档位 | 行 | 臂 / 方法 | utility | 相对 C1 cond_hist | 来源 gate |
|---|---|---|---:|---:|---|
| p_c=0 | 主模型/内部臂 | a1_rollout | 1312.6 | -370.2 | `results/a1_a1_formal_40_1004_171907/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1682.8 | 0.0 | `results/a1_a1_formal_40_1004_171907/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1577.9 | -104.9 | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1694.7 | 11.9 | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1582.0 | -100.8 | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1680.0 | -2.8 | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1569.4 | -113.4 | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1689.9 | 7.2 | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1162.0 | -520.8 | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1694.7 | 11.9 | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1144.0 | -538.7 | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1680.0 | -2.8 | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1148.5 | -534.2 | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1689.9 | 7.2 | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1449.3 | -233.5 | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1694.7 | 11.9 | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1467.7 | -215.1 | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1680.0 | -2.8 | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | ★ maskco | 1440.4 | -242.3 | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1689.9 | 7.2 | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1683.3 | 0.5 | `results/a1_c1r_reveal_40/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1703.7 | 21.0 | `results/a1_c1rr2_reveal_40/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1694.7 | 11.9 | `results/a1_rr3_density_40_1001_234935/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1708.2 | 25.4 | `results/a1_rr3_reveal_40_1001_234935/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 450.9 | -1231.9 | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1643.3 | -39.5 | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1484.5 | -198.2 | `results/a1_step2_gate_c1_20260926/gate.json` |
| p_c=0 | 主模型/内部臂 | cond_hist | 1677.2 | -5.6 | `results/a3_baseline_full-cond_20260926_1007_230001/gate.json` |
| p_c=0 | 最强外部方法 | rule: value/energy λ=2 | 1403.3 | — | 本表 §1 |
| p_c=(5,10,15) | 主模型/内部臂 | a1_rollout | 40.2 | -558.3 | `results/a1_a1_formal_40_1004_171907/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 598.5 | 0.0 | `results/a1_a1_formal_40_1004_171907/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | 458.8 | -139.7 | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 617.3 | 18.8 | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | 511.9 | -86.6 | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 666.3 | 67.7 | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | 422.4 | -176.2 | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 598.2 | -0.3 | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | -178.8 | -777.3 | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 617.3 | 18.8 | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | -184.5 | -783.0 | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 666.3 | 67.7 | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | -205.0 | -803.5 | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 598.2 | -0.3 | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | 238.9 | -359.6 | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 617.3 | 18.8 | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | 339.4 | -259.1 | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 666.3 | 67.7 | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | 216.3 | -382.2 | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 598.2 | -0.3 | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 581.8 | -16.8 | `results/a1_c1r_reveal_40/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 631.4 | 32.9 | `results/a1_c1rr2_reveal_40/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 617.3 | 18.8 | `results/a1_rr3_density_40_1001_234935/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 635.6 | 37.1 | `results/a1_rr3_reveal_40_1001_234935/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | -1238.9 | -1837.5 | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 504.5 | -94.0 | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 308.6 | -290.0 | `results/a1_step2_gate_c1_20260926/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 596.5 | -2.0 | `results/a3_baseline_full-cond_20260926_1007_230001/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | ★ maskco | 456.3 | -142.2 | `results/i3_online_recovery_1009_v1/online_20260926/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | warmstart | 447.0 | -151.6 | `results/i3_online_recovery_1009_v1/online_20260926/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | reconstruction_control | 473.4 | -125.2 | `results/i3_online_recovery_1009_v1/online_20260926/gate.json` |
| p_c=(5,10,15) | 主模型/内部臂 | cond_hist | 601.0 | 2.5 | `results/i3_online_recovery_1009_v1/online_20260926/gate.json` |
| p_c=(5,10,15) | 最强外部方法 | rule: value/energy λ=2 | 180.6 | — | 本表 §1 |

## 1. 主表（C1 基线 + 全部对比方法 + Ours，同一张表按 p_c=0 效用降序）

| # | 方法 | 比较状态 | Q_core | 依赖闭包 | 配对来源身份 | protocol_id | 类别 | 期刊/会议 | 期刊/会议等级 (CCF) | Backbone / 权重 | 参数量(M) | 原生约束 | Utility p_c=0 ↑ | Utility 主档 p_c=(5,10,15) | Utility p_c=(10,20,30) | Δ vs cond_hist (p_c=0) | Δ vs cond_hist (主档) | 超时/天 p_c=0 | 超时/天 主档 | 硬可行率 p_c=0 | 服务单/天 p_c=0 | 能耗 kWh/天 p_c=0 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ◆ C1 cond_hist（SAA 强基线） | — | — | — | — | — | 内部基线 | 本工作 | — | a1_c1rr2_reveal_40 | — | 场景采样 SAA | 1703.7 [1680.9, 1726.9] | 631.4 [522.7, 727.6] | -435.6 [-651.0, -239.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | — | — | — | — | — | Δ 相对自身 ⇒ 0 |
| 2 | ★ Ours: MaskCO A3（掩码重构 · 预训练权重） | DIAGNOSTIC / CROSS-SOURCE | **否** | **否** | 跨源（shared_source_sha256_mismatch） | a1_a3_online_density_B_20260926_40_1003_040454 | 主模型/内部臂 | 本工作 | — | a1_a3_online_density_B_20260926_40_1003_040454 | — | 接/拒 + 掩码重构 | 1577.9 [1558.5, 1597.0] | 458.8 [356.1, 557.8] | -704.1 [-904.6, -510.5] | -116.7 [-138.6, -96.0] | -158.5 [-203.4, -114.7] | 19.82 | 18.85 | 1.000 | 89.2 | 709.8 | gate `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` |
| 3 | ★ Ours: MaskCO I3（决策感知） | QUALIFIED DEVELOPMENT / NEGATIVE | 是 | **否** | 同源 | i3_online_recovery_1009_v1/online_20260926 | 主模型/内部臂 | 本工作 | — | i3_online_recovery_1009_v1 | — | 接/拒 + 掩码重构 | — | 456.3 [369.8, 538.0] | — | — | -144.7 [-191.1, -100.1] | — | 37.33 | — | — | — | gate `results/i3_online_recovery_1009_v1/online_20260926/gate.json` |
| 4 | rule: value/energy λ=2 | QUALIFIED / NEGATIVE | 是 | — | 不适用（纯规则臂） | a1_c1fresh_reveal_40 | 规则基线 | — | — | — | — | 预算感知门槛 | 1403.3 [1314.0, 1498.0] | 180.6 [118.1, 243.2] | — | -300.5 [-379.8, -214.5] | -450.8 [-570.5, -319.9] | 0.00 | 0.00 | 1.000 | 63.5 | 508.9 | 固定价值/能耗门槛 |
| 5 | rule: myopic | QUALIFIED / NEGATIVE | 是 | — | 不适用（纯规则臂） | a1_c1fresh_reveal_40 | 规则基线 | — | — | — | — | 完整覆盖可行性 | 1305.2 [1284.1, 1325.7] | 30.5 [-94.4, 149.7] | — | -398.5 [-423.9, -370.8] | -600.9 [-638.5, -560.6] | 0.00 | 0.00 | 1.000 | 69.0 | 710.1 | 无前瞻即收 |
| 6 | OR-Tools-RH accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 经典求解器 | Google OR-Tools（软件，非论文） | — | CP-SAT/GLS | — | CVRPTW（完整覆盖） | 1300.7 [1281.2, 1320.6] | 20.9 [-102.8, 139.7] | -1258.8 [-1492.3, -1037.5] | -403.1 [-428.4, -375.3] | -610.5 [-647.4, -569.4] | 0.00 | 0.00 | 1.000 | 68.5 | 710.2 | myopic 接单 + 强路由 |
| 7 | SGBS accept | QUALIFIED-TIMEOUT / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（搜索） | NeurIPS 2022 | A | CVRP100 ckpt-30500 | 1.254 | CVRP（零样本） | 1266.5 [1244.2, 1289.9] | -52.8 [-182.8, 69.5] | — | -437.3 [-463.8, -409.0] | -684.2 [-732.1, -640.5] | 12.03 | 9.07 | 1.000 | 67.1 | 710.2 | beam=10 计入时限；与 POMO 同权重 |
| 8 | RouteFinder accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（基础模型） | TMLR 2025 / ICML 2026 (J2C) | A（ICML 2026 J2C；TMLR 未列入） | rf-transformer 100 | 1.678 | CVRP（零样本，TW 由外层认证） | 1266.2 [1244.1, 1287.5] | -26.1 [-152.1, 93.4] | — | -437.6 [-466.3, -408.7] | -657.5 [-699.9, -611.9] | 0.00 | 0.00 | 1.000 | 67.6 | 710.1 |  |
| 9 | MVMoE accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（多任务 MoE） | ICML 2024 | A | pomo_vrptw_n100 epoch-5000 | 1.254 | VRPTW（TW 感知） | 1257.7 [1231.3, 1284.0] | -38.6 [-165.6, 83.4] | — | -446.1 [-478.4, -414.7] | -670.0 [-722.6, -619.0] | 0.00 | 0.00 | 1.000 | 66.9 | 710.1 | POMO 架构 + VRPTW 权重 |
| 10 | POMO accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（构造） | NeurIPS 2020 | A | CVRP100 ckpt-30500 | 1.254 | CVRP（零样本） | 1238.8 [1214.8, 1261.5] | -68.1 [-190.0, 48.6] | — | -464.9 [-498.3, -432.8] | -699.5 [-752.3, -650.5] | 0.00 | 0.00 | 1.000 | 66.0 | 710.3 | 与 SGBS/Omni-VRP **同一权重文件**（sha256 d7a08f61…） |
| 11 | AttentionModel accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（构造） | ICLR 2019 | A（2022版：未列入） | cvrp_100 | 0.693 | CVRP（零样本） | 1236.2 [1207.3, 1263.2] | -72.9 [-196.8, 46.6] | — | -467.5 [-501.3, -433.7] | -704.3 [-756.6, -652.9] | 0.00 | 0.00 | 1.000 | 65.4 | 710.3 | 官方 greedy 档 |
| 12 | RRNCO-Ordering accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（排序） | ICLR 2026 | A（2022版：未列入） | RRNCO rcvrptw epoch_199 | 4.290 | RCVRPTW（有 TW） | 1235.4 [1202.4, 1265.2] | -73.2 [-203.1, 54.4] | — | -468.3 [-511.9, -428.0] | -704.6 [-768.5, -646.1] | 0.00 | 0.00 | 1.000 | 65.7 | 710.3 | 冻结 R1 管线逐车排序 |
| 13 | CO-enriched-ML accept（纯学习排序） | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（预测+优化） | Transportation Science 2024 | 未列入 | 官方 MLP 20→10×4→1（NumPy 前向） | 0.001 | 原生动态 VRPTW（波次语义 → 逐揭示改造） | 1234.5 [1204.9, 1263.3] | -74.3 [-210.8, 53.0] | — | -469.3 [-505.9, -432.4] | -705.7 [-760.7, -651.3] | 0.00 | 0.00 | 1.000 | 65.8 | 710.3 | 不用 PC-HGS；0.02 s/决策 |
| 14 | CO-enriched-ML accept（+官方 PC-HGS） | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（预测+优化） | Transportation Science 2024 | 未列入 | 官方 MLP + PC-HGS（C++20 原二进制） | 0.001 | 原生动态 VRPTW（波次语义 → 逐揭示改造） | 1230.5 [1197.6, 1259.7] | -80.5 [-218.0, 48.3] | — | -473.2 [-516.7, -434.1] | -711.9 [-778.9, -655.2] | 0.00 | 0.00 | 1.000 | 65.6 | 710.3 | PC-HGS 作优先级生成器，按池缓存 ⇒ 1 次/决策 ≈2.1 s |
| 15 | Sym-NCO accept | QUALIFIED-TIMEOUT / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（构造） | NeurIPS 2022 | A | CVRP ckpt-8000 | 1.303 | CVRP（零样本） | 1229.3 [1196.0, 1261.3] | -76.1 [-217.8, 55.3] | — | -474.5 [-515.9, -435.0] | -707.5 [-772.9, -644.8] | 0.07 | 0.23 | 1.000 | 65.3 | 710.1 | 对称变换 |
| 16 | Omni-VRP accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（构造） | ICML 2023 | A | POMO-CVRP 档 | 1.254 | CVRP（零样本） | 1226.9 [1195.8, 1254.5] | -85.7 [-209.6, 33.8] | — | -476.8 [-512.8, -442.0] | -717.1 [-776.5, -662.0] | 0.00 | 0.00 | 1.000 | 65.0 | 710.3 | 官方 aug8 档；与 POMO 同权重 |
| 17 | MAPT accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（多智能体指针 Transformer） | AAAI 2026 | A | MAPT 自训 144 updates（1.44 M 参数） | 1.437 | MVDPDPSR（无 TW；TW/冷链由外层认证） | 1215.4 [1189.0, 1240.8] | -102.2 [-224.6, 18.5] | — | -488.3 [-526.0, -451.5] | -733.6 [-789.0, -679.8] | 0.00 | 0.00 | 1.000 | 64.7 | 710.3 | **弱介入档**：模型只决定前 ~3 位、尾部 EDD；dev 效用未超弱贪心（871 vs 951），服务率超（36.4% vs 26.9%） |
| 18 | Learn-Improvement-Heuristics accept | QUALIFIED-TIMEOUT / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（改进） | IEEE TNNLS 2022 | B | CVRP50 epoch-199 | 0.710 | CVRP（零样本） | 1215.0 [1149.2, 1269.0] | -335.1 [-536.5, -144.0] | — | -488.7 [-560.8, -428.3] | -966.6 [-1143.1, -807.0] | 25.35 | 54.23 | 1.000 | 64.2 | 685.1 | steps=100 局部改进；跨次不可复现 |
| 19 | DeepACO accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（ACO 增强） | NeurIPS 2023 | A | cvrp100 | 0.067 | CVRP（零样本） | 1213.6 [1175.9, 1250.2] | -100.0 [-231.2, 30.8] | — | -490.2 [-534.8, -446.3] | -731.4 [-797.1, -665.8] | 0.00 | 0.00 | 1.000 | 64.8 | 710.1 | 官方最轻档 T=30 |
| 20 | CaDA accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 学习型（约束感知） | ICML 2025 | A | CaDA 100 checkpoint-300 | 3.421 | CVRP（零样本） | 1157.0 [1120.6, 1190.8] | -190.3 [-327.8, -54.5] | — | -546.7 [-592.1, -502.5] | -821.7 [-892.2, -755.2] | 0.00 | 0.00 | 1.000 | 62.0 | 710.3 |  |
| 21 | PyVRP-RH accept | QUALIFIED / NEGATIVE | 是 | **否** | 同源 | a1_c1fresh_reveal_40 | 经典求解器 | INFORMS JOC 2024 | B | HGS (pyvrp 0.14.0) | — | CVRPTW（完整覆盖） | 909.2 [856.8, 960.4] | -569.8 [-660.7, -482.9] | -2048.8 [-2223.5, -1876.5] | -794.5 [-852.0, -739.4] | -1201.2 [-1286.6, -1117.5] | 0.00 | 0.00 | 1.000 | 47.9 | 710.3 | 第二经典锚点 |
| 22 | rule: value/energy λ=3 | DEGENERATE / NEGATIVE | **否** | — | 不适用（纯规则臂） | a1_c1fresh_reveal_40 | 规则基线 | — | — | — | — | 预算感知门槛 | 0.0 [0.0, 0.0] | -1943.1 [-2051.4, -1841.0] | — | -1703.7 [-1726.9, -1680.9] | -2574.5 [-2611.9, -2538.0] | 0.00 | 0.00 | 1.000 | 0.0 | 0.0 | 退化（0 接单） |

- ※ **`期刊/会议等级 (CCF)` 列**＝**CCF 推荐国际会议/期刊目录**等级（A / B / C）；口径为**第 7 版（2026 年，现行）**。CCF 目录**分「会议」与「期刊」两册**：会议行取会议册（ICLR / ICML / NeurIPS / AAAI），期刊行取期刊册（IEEE TNNLS=B 人工智能；INFORMS JOC=B 计算机科学理论；Transportation Science 与 TMLR **未列入**）。⚠️ **ICLR 有版本变化**：2019（第 5 版）/ 2022（第 6 版）均**未列入**，**第 7 版（2026）新增为 A** —— 本表 ICLR 两行（`RRNCO` / `AttentionModel`）即属此情况。换口径只需改 `collect_compare_table.CCF_EDITION` 一个常量。`—` ＝**非论文**（软件 `Google OR-Tools`、本项目规则基线、本工作）；`未列入` ＝该出处在 CCF 目录中**查无此项**；`未确证` ＝**未能在权威来源核实**，引用前请自行核对。⚠️ 该列描述的是**出处（venue）的等级**，**不代表**本表的复现质量或适配保真度 —— 表内全部方法都运行在**共用的在线接拒框架**下，不等于原生论文实现的性能。
- ※ `rule: value/energy λ=3`：该规则**全拒单**（选择不服务任何订单），**不是「性能很差的规则」**。依据见 `读表注意与已知限定.md` §2.1。
- ※ `MAPT`：**弱介入**——模型只决定排序**前 ~3 位**、其余尾部由 EDD 补全，不得当作「MAPT 方法能力」的证据。见 `读表注意与已知限定.md` §2.2。
- ※ `LIH` / `DeepACO`：**跨次不可复现**（清缓存后两次调用不一致），不得与确定性方法并列宣称「同 seed 逐位复现」。见 §2.3。
- ※ `POMO` / `SGBS` / `Omni-VRP`：**共用同一权重文件**（三者 checkpoint 的 sha256 完全相同），是同一 checkpoint 的三种解码/搜索用法，**不是三个独立训练的模型**。见 §2.4。
- ※ 引用任何数字时**必须同时报告** `mean_timeouts`、`hard_feasible_rate` 与设备/解释器（同一方法在 cpu/cuda 下速度差异可达数量级，直接影响超时）。
- ※ **主档列是「同一策略的惩罚重评分」，不是「策略对惩罚的响应」**：接拒 replanner 的构造**不接收 `p_c`**（`run_a1_external_accept.py:127-130`、`rrnco_accept_replanner.py:48-49`），故档间差异**不可能**来自策略调整，只能来自评分函数。（本批各档是**分别运行**的两次，因此 `LIH`/`DeepACO` 这类跨次不可复现的方法还可能混入运行噪声。）若要真正的惩罚敏感性，需让决策规则使用 `p_c` 并逐档重跑（属新协议，不得追认）。依据：`CC_Compare/audit/INDEPENDENT_READONLY_ADJUDICATION_2026-10-14.md` §二-问题 2。
- ※ **Ours 行同样受 CPU 争用影响，且其中一行配对身份未通过 —— 引用主模型结果时必须与外部臂一样带限定**：**Ours: MaskCO A3（掩码重构 · 预训练权重）**（超时/天 = 19.82，配对身份 = False，问题=shared_source_sha256_mismatch）；**Ours: MaskCO I3（决策感知）**（超时/天 = 37.33，配对身份 = True）。作为对比，对比方法臂的正式批次中**除上表 † 标出的 6 个之外全部 `超时/天 = 0.0`**。（注意「方法数」与「臂数」不同：方法按行计，臂按 (方法 × 档位) 计。）依据：`CC_Compare/audit/BASELINE_QUALIFICATION_MATRIX_2026-10-14.json`。
- ※ **结论口径：正式论文的主结论以主档 `p_c=(5,10,15)` 为准**；`p_c=0` 是**敏感性结果**，不得因为它的绝对效用更高就充当头条贡献证据。本表按 p_c=0 排序仅为呈现习惯，不代表主结论口径。**主档同批三元组**（同一 gate `results/i3_online_recovery_1009_v1/online_20260926/gate.json`）：Ours: MaskCO I3（决策感知） = **456.317**、同批 `cond_hist` = **600.997**、配对差 = **-144.680** [-191.109, -100.125]，恒等式校验 **通过**。⇒ **主档上主模型未超过 C1 强基线**。⚠️ 解释该配对差时**必须用同批的 600.997**；`631.4` 属**另一套协议**（外部臂配对基线 `a1_c1fresh_reveal_40|cond_hist`），**不得**与本行并列。
- ※ **本表各行的 `future_policy` 并不一致**：Ours 两行取自 `density` 配置（gate 名含 `density` / `i3_online_recovery`），而 19 个对比方法的配对基线由`--future-policy reveal` 生成（`scenario_saa.py:268` 的默认值）。**A2 证据已表明该杠杆会显著改变 `cond_hist−uncond_hist` 的相对差** ⇒ 本表的效用排序只可读作**开发集能力对照**，**不构成**同协议下的方法归因。依据：同上 §二-问题 7。

## 2. 身份合格的 C1 基线臂（配对参照）

| 基线臂 | 档位 | Utility | 服务单/天 | 拒绝单/天 | 收入/天 | 燃油/天 | 拒绝损失/天 | 能耗 kWh/天 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cond_hist | p_c=(10,20,30) | -435.6 [-651.0, -239.6] | 99.3 | 94.7 | 1749.2 | 47.83 | 2137.0 | 709.4 |
| explicit_feat | p_c=(10,20,30) | -373.4 [-551.3, -208.4] | 103.2 | 90.8 | 1781.2 | 49.64 | 2105.0 | 702.0 |
| uncond_hist | p_c=(10,20,30) | -420.4 [-610.4, -239.8] | 100.9 | 93.2 | 1756.8 | 47.60 | 2129.5 | 707.8 |
| cond_hist | p_c=(5,10,15) | 631.4 [522.7, 727.6] | 99.2 | 94.8 | 1748.2 | 47.84 | 1069.0 | 709.5 |
| explicit_feat | p_c=(5,10,15) | 676.3 [594.8, 747.7] | 102.9 | 91.2 | 1779.5 | 49.80 | 1053.4 | 700.5 |
| uncond_hist | p_c=(5,10,15) | 642.7 [551.6, 732.7] | 100.5 | 93.5 | 1755.2 | 47.07 | 1065.5 | 708.9 |
| cond_hist | p_c=0 | 1703.7 [1680.9, 1726.9] | 99.0 | 95.0 | 1751.8 | 48.01 | 0.0 | 709.2 |
| explicit_feat | p_c=0 | 1742.8 [1711.3, 1771.4] | 103.4 | 90.6 | 1792.5 | 49.67 | 0.0 | 702.1 |
| uncond_hist | p_c=0 | 1712.1 [1691.6, 1733.2] | 100.5 | 93.5 | 1759.8 | 47.70 | 0.0 | 709.1 |

## 3. 主模型 / 内部臂（**不同 future_policy / standby / 基线，不与上表同尺度比较**）

| 交付行？ | gate | 档位 | 臂 | utility | 超时/天 | 身份块 | source_stable | future_policy | standby | workers | 配对身份 |
|---|---|---|---|---:|---:|---|---|---|---|---|---|
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=(10,20,30) | a1_rollout | -1231.1 [-1467.4, -1004.2] | 0.00 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=(10,20,30) | cond_hist | -485.3 [-652.9, -324.4] | 45.02 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=(10,20,30) | explicit_feat | -451.6 [-647.7, -268.2] | 47.58 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=(10,20,30) | uncond_hist | -574.8 [-773.0, -387.1] | 48.10 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=(5,10,15) | a1_rollout | 40.2 [-85.1, 160.9] | 0.00 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=(5,10,15) | cond_hist | 598.5 [527.3, 670.4] | 46.05 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=(5,10,15) | explicit_feat | 612.1 [517.4, 698.7] | 51.60 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=(5,10,15) | uncond_hist | 527.2 [421.9, 624.6] | 48.27 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=0 | a1_rollout | 1312.6 [1292.0, 1332.2] | 0.00 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=0 | cond_hist | 1682.8 [1652.5, 1715.2] | 45.25 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=0 | explicit_feat | 1702.4 [1682.4, 1720.9] | 51.92 | 有 | True | density | None | 10 | — |
| 否（消融/重复/内部臂） | `results/a1_a1_formal_40_1004_171907/gate.json` | p_c=0 | uncond_hist | 1631.2 [1610.4, 1651.7] | 47.92 | 有 | True | density | None | 10 | — |
| ★ 是 | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=(10,20,30) | maskco | -704.1 [-904.6, -510.5] | 4.95 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=(10,20,30) | cond_hist | -459.9 [-628.9, -298.5] | 34.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=(10,20,30) | explicit_feat | -440.9 [-633.6, -263.2] | 38.77 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=(10,20,30) | uncond_hist | -539.6 [-743.1, -345.6] | 33.08 | 有 | True | density | None | 2 | False |
| ★ 是 | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=(5,10,15) | maskco | 458.8 [356.1, 557.8] | 18.85 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=(5,10,15) | cond_hist | 617.3 [544.8, 685.9] | 34.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=(5,10,15) | explicit_feat | 618.0 [522.4, 705.1] | 34.17 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=(5,10,15) | uncond_hist | 551.6 [453.5, 645.2] | 32.30 | 有 | True | density | None | 2 | False |
| ★ 是 | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=0 | maskco | 1577.9 [1558.5, 1597.0] | 19.82 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=0 | cond_hist | 1694.7 [1664.7, 1724.0] | 35.45 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=0 | explicit_feat | 1702.1 [1677.2, 1724.0] | 38.40 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json` | p_c=0 | uncond_hist | 1653.5 [1635.6, 1671.1] | 33.73 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=(10,20,30) | maskco | -592.1 [-753.6, -421.5] | 4.30 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=(10,20,30) | cond_hist | -356.1 [-480.9, -233.9] | 48.65 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=(10,20,30) | explicit_feat | -309.7 [-462.7, -148.1] | 46.65 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=(10,20,30) | uncond_hist | -425.9 [-599.6, -247.4] | 44.38 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=(5,10,15) | maskco | 511.9 [425.2, 603.4] | 17.43 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=(5,10,15) | cond_hist | 666.3 [609.9, 722.7] | 48.60 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=(5,10,15) | explicit_feat | 679.4 [602.3, 756.6] | 48.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=(5,10,15) | uncond_hist | 595.1 [504.5, 687.5] | 45.05 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=0 | maskco | 1582.0 [1563.7, 1600.3] | 17.25 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=0 | cond_hist | 1680.0 [1650.7, 1708.3] | 50.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=0 | explicit_feat | 1700.1 [1676.3, 1721.9] | 49.58 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20260930_40_1003_040454/gate.json` | p_c=0 | uncond_hist | 1641.5 [1621.1, 1661.3] | 42.55 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=(10,20,30) | maskco | -767.8 [-917.8, -608.9] | 4.95 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=(10,20,30) | cond_hist | -506.6 [-637.3, -371.1] | 47.62 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=(10,20,30) | explicit_feat | -474.9 [-629.3, -313.9] | 48.12 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=(10,20,30) | uncond_hist | -598.5 [-751.5, -440.6] | 42.95 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=(5,10,15) | maskco | 422.4 [345.1, 508.1] | 18.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=(5,10,15) | cond_hist | 598.2 [543.0, 655.9] | 48.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=(5,10,15) | explicit_feat | 625.7 [544.9, 704.7] | 48.12 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=(5,10,15) | uncond_hist | 520.8 [440.5, 603.7] | 42.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=0 | maskco | 1569.4 [1544.5, 1593.5] | 18.20 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=0 | cond_hist | 1689.9 [1658.6, 1720.0] | 47.60 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=0 | explicit_feat | 1701.5 [1684.5, 1720.3] | 48.62 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_B_20261001_40_1003_040454/gate.json` | p_c=0 | uncond_hist | 1638.3 [1620.7, 1656.5] | 42.80 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=(10,20,30) | maskco | -1014.2 [-1170.7, -850.1] | 98.83 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=(10,20,30) | cond_hist | -459.9 [-628.9, -298.5] | 34.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=(10,20,30) | explicit_feat | -440.9 [-633.6, -263.2] | 38.77 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=(10,20,30) | uncond_hist | -539.6 [-743.1, -345.6] | 33.08 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=(5,10,15) | maskco | -178.8 [-268.3, -93.2] | 126.47 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=(5,10,15) | cond_hist | 617.3 [544.8, 685.9] | 34.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=(5,10,15) | explicit_feat | 618.0 [522.4, 705.1] | 34.17 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=(5,10,15) | uncond_hist | 551.6 [453.5, 645.2] | 32.30 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=0 | maskco | 1162.0 [1064.0, 1266.5] | 126.85 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=0 | cond_hist | 1694.7 [1664.7, 1724.0] | 35.45 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=0 | explicit_feat | 1702.1 [1677.2, 1724.0] | 38.40 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260926_40_1004_171907/gate.json` | p_c=0 | uncond_hist | 1653.5 [1635.6, 1671.1] | 33.73 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=(10,20,30) | maskco | -915.9 [-1069.8, -763.2] | 99.40 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=(10,20,30) | cond_hist | -356.1 [-480.9, -233.9] | 48.65 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=(10,20,30) | explicit_feat | -309.7 [-462.7, -148.1] | 46.65 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=(10,20,30) | uncond_hist | -425.9 [-599.6, -247.4] | 44.38 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=(5,10,15) | maskco | -184.5 [-265.6, -104.6] | 127.55 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=(5,10,15) | cond_hist | 666.3 [609.9, 722.7] | 48.60 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=(5,10,15) | explicit_feat | 679.4 [602.3, 756.6] | 48.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=(5,10,15) | uncond_hist | 595.1 [504.5, 687.5] | 45.05 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=0 | maskco | 1144.0 [1039.4, 1247.2] | 126.70 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=0 | cond_hist | 1680.0 [1650.7, 1708.3] | 50.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=0 | explicit_feat | 1700.1 [1676.3, 1721.9] | 49.58 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20260930_40_1004_171907/gate.json` | p_c=0 | uncond_hist | 1641.5 [1621.1, 1661.3] | 42.55 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=(10,20,30) | maskco | -1061.5 [-1213.8, -898.5] | 101.47 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=(10,20,30) | cond_hist | -506.6 [-637.3, -371.1] | 47.62 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=(10,20,30) | explicit_feat | -474.9 [-629.3, -313.9] | 48.12 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=(10,20,30) | uncond_hist | -598.5 [-751.5, -440.6] | 42.95 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=(5,10,15) | maskco | -205.0 [-309.7, -102.2] | 128.85 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=(5,10,15) | cond_hist | 598.2 [543.0, 655.9] | 48.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=(5,10,15) | explicit_feat | 625.7 [544.9, 704.7] | 48.12 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=(5,10,15) | uncond_hist | 520.8 [440.5, 603.7] | 42.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=0 | maskco | 1148.5 [1047.7, 1239.8] | 128.90 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=0 | cond_hist | 1689.9 [1658.6, 1720.0] | 47.60 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=0 | explicit_feat | 1701.5 [1684.5, 1720.3] | 48.62 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_C_20261001_40_1004_171907/gate.json` | p_c=0 | uncond_hist | 1638.3 [1620.7, 1656.5] | 42.80 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=(10,20,30) | maskco | -976.6 [-1212.3, -746.1] | 3.23 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=(10,20,30) | cond_hist | -459.9 [-628.9, -298.5] | 34.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=(10,20,30) | explicit_feat | -440.9 [-633.6, -263.2] | 38.77 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=(10,20,30) | uncond_hist | -539.6 [-743.1, -345.6] | 33.08 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=(5,10,15) | maskco | 238.9 [113.5, 360.4] | 7.30 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=(5,10,15) | cond_hist | 617.3 [544.8, 685.9] | 34.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=(5,10,15) | explicit_feat | 618.0 [522.4, 705.1] | 34.17 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=(5,10,15) | uncond_hist | 551.6 [453.5, 645.2] | 32.30 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=0 | maskco | 1449.3 [1424.5, 1474.6] | 7.60 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=0 | cond_hist | 1694.7 [1664.7, 1724.0] | 35.45 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=0 | explicit_feat | 1702.1 [1677.2, 1724.0] | 38.40 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260926_40_1003_040454/gate.json` | p_c=0 | uncond_hist | 1653.5 [1635.6, 1671.1] | 33.73 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=(10,20,30) | maskco | -823.6 [-1012.9, -619.8] | 3.38 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=(10,20,30) | cond_hist | -356.1 [-480.9, -233.9] | 48.65 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=(10,20,30) | explicit_feat | -309.7 [-462.7, -148.1] | 46.65 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=(10,20,30) | uncond_hist | -425.9 [-599.6, -247.4] | 44.38 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=(5,10,15) | maskco | 339.4 [233.2, 451.4] | 7.78 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=(5,10,15) | cond_hist | 666.3 [609.9, 722.7] | 48.60 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=(5,10,15) | explicit_feat | 679.4 [602.3, 756.6] | 48.50 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=(5,10,15) | uncond_hist | 595.1 [504.5, 687.5] | 45.05 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=0 | maskco | 1467.7 [1441.2, 1492.9] | 8.35 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=0 | cond_hist | 1680.0 [1650.7, 1708.3] | 50.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=0 | explicit_feat | 1700.1 [1676.3, 1721.9] | 49.58 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20260930_40_1003_040454/gate.json` | p_c=0 | uncond_hist | 1641.5 [1621.1, 1661.3] | 42.55 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=(10,20,30) | maskco | -1011.4 [-1174.8, -840.1] | 4.15 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=(10,20,30) | cond_hist | -506.6 [-637.3, -371.1] | 47.62 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=(10,20,30) | explicit_feat | -474.9 [-629.3, -313.9] | 48.12 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=(10,20,30) | uncond_hist | -598.5 [-751.5, -440.6] | 42.95 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=(5,10,15) | maskco | 216.3 [128.6, 308.6] | 8.15 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=(5,10,15) | cond_hist | 598.2 [543.0, 655.9] | 48.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=(5,10,15) | explicit_feat | 625.7 [544.9, 704.7] | 48.12 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=(5,10,15) | uncond_hist | 520.8 [440.5, 603.7] | 42.88 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=0 | maskco | 1440.4 [1420.4, 1461.4] | 8.12 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=0 | cond_hist | 1689.9 [1658.6, 1720.0] | 47.60 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=0 | explicit_feat | 1701.5 [1684.5, 1720.3] | 48.62 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_a3_online_density_D_20261001_40_1003_040454/gate.json` | p_c=0 | uncond_hist | 1638.3 [1620.7, 1656.5] | 42.80 | 有 | True | density | None | 2 | False |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=(10,20,30) | cond_hist | -507.8 [-708.8, -321.5] | 26.35 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=(10,20,30) | explicit_feat | -381.1 [-567.2, -211.9] | 5.92 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=(10,20,30) | uncond_hist | -442.4 [-636.3, -258.8] | 11.35 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=(5,10,15) | cond_hist | 581.8 [475.7, 678.5] | 25.57 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=(5,10,15) | explicit_feat | 674.0 [591.6, 750.8] | 5.12 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=(5,10,15) | uncond_hist | 638.1 [543.3, 731.1] | 9.53 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=0 | cond_hist | 1683.3 [1656.5, 1710.9] | 23.88 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=0 | explicit_feat | 1732.8 [1701.6, 1761.3] | 4.45 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1r_reveal_40/gate.json` | p_c=0 | uncond_hist | 1706.1 [1684.2, 1727.5] | 8.62 | 有 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=(10,20,30) | cond_hist | -435.6 [-651.0, -239.6] | 2.80 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=(10,20,30) | explicit_feat | -373.4 [-551.3, -208.4] | 0.47 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=(10,20,30) | uncond_hist | -420.4 [-610.4, -239.8] | 1.75 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=(5,10,15) | cond_hist | 631.4 [522.7, 727.6] | 3.23 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=(5,10,15) | explicit_feat | 676.3 [594.8, 747.7] | 0.50 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=(5,10,15) | uncond_hist | 642.7 [551.6, 732.7] | 2.02 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=0 | cond_hist | 1703.7 [1680.9, 1726.9] | 3.52 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=0 | explicit_feat | 1742.8 [1711.3, 1771.4] | 0.45 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_c1rr2_reveal_40/gate.json` | p_c=0 | uncond_hist | 1712.1 [1691.6, 1733.2] | 0.82 | 有 | False | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=(10,20,30) | cond_hist | -459.9 [-628.9, -298.5] | 34.50 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=(10,20,30) | explicit_feat | -440.9 [-633.6, -263.2] | 38.77 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=(10,20,30) | uncond_hist | -539.6 [-743.1, -345.6] | 33.08 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=(5,10,15) | cond_hist | 617.3 [544.8, 685.9] | 34.50 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=(5,10,15) | explicit_feat | 618.0 [522.4, 705.1] | 34.17 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=(5,10,15) | uncond_hist | 551.6 [453.5, 645.2] | 32.30 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=0 | cond_hist | 1694.7 [1664.7, 1724.0] | 35.45 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=0 | explicit_feat | 1702.1 [1677.2, 1724.0] | 38.40 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_density_40_1001_234935/gate.json` | p_c=0 | uncond_hist | 1653.5 [1635.6, 1671.1] | 33.73 | 有 | True | density | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=(10,20,30) | cond_hist | -439.7 [-654.1, -245.4] | 0.03 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=(10,20,30) | explicit_feat | -374.7 [-554.3, -209.1] | 0.00 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=(10,20,30) | uncond_hist | -414.5 [-605.7, -231.9] | 0.00 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=(5,10,15) | cond_hist | 635.6 [526.7, 733.3] | 0.05 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=(5,10,15) | explicit_feat | 673.6 [595.6, 745.0] | 0.00 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=(5,10,15) | uncond_hist | 643.9 [551.6, 735.9] | 0.00 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=0 | cond_hist | 1708.2 [1685.7, 1731.4] | 0.10 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=0 | explicit_feat | 1737.0 [1705.9, 1765.9] | 0.00 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_rr3_reveal_40_1001_234935/gate.json` | p_c=0 | uncond_hist | 1710.0 [1689.8, 1730.2] | 0.00 | 有 | True | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=(10,20,30) | cond_hist | -2947.2 [-3135.2, -2768.6] | 169.35 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=(10,20,30) | explicit_feat | -2985.5 [-3109.5, -2862.4] | 171.95 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=(10,20,30) | uncond_hist | -2952.8 [-3079.8, -2825.0] | 170.72 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=(5,10,15) | cond_hist | -1238.9 [-1332.1, -1147.6] | 169.43 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=(5,10,15) | explicit_feat | -1278.8 [-1341.1, -1215.9] | 172.15 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=(5,10,15) | uncond_hist | -1229.7 [-1301.1, -1159.1] | 170.45 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=0 | cond_hist | 450.9 [420.6, 480.8] | 169.55 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=0 | explicit_feat | 439.2 [373.7, 509.8] | 172.05 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_off_40_1007_125522_att1/gate.json` | p_c=0 | uncond_hist | 474.5 [400.7, 551.5] | 170.10 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=(10,20,30) | cond_hist | -601.1 [-774.3, -430.1] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=(10,20,30) | explicit_feat | -617.4 [-805.7, -431.2] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=(10,20,30) | uncond_hist | -656.0 [-861.2, -463.2] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=(5,10,15) | cond_hist | 504.5 [424.9, 583.6] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=(5,10,15) | explicit_feat | 494.8 [405.2, 584.2] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=(5,10,15) | uncond_hist | 469.3 [368.7, 565.2] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=0 | cond_hist | 1643.3 [1614.4, 1670.3] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=0 | explicit_feat | 1634.0 [1616.5, 1650.7] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_s3_4_d3_on_40_1007_125522_att1/gate.json` | p_c=0 | uncond_hist | 1624.8 [1608.2, 1641.5] | 0.00 | 有 | True | density | None | 16 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=(10,20,30) | cond_hist | -868.2 [-1094.4, -650.2] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=(10,20,30) | explicit_feat | -961.2 [-1204.5, -737.0] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=(10,20,30) | uncond_hist | -959.7 [-1191.8, -743.4] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=(5,10,15) | cond_hist | 308.6 [191.3, 418.3] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=(5,10,15) | explicit_feat | 244.8 [115.7, 364.0] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=(5,10,15) | uncond_hist | 237.1 [115.7, 355.5] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=0 | cond_hist | 1484.5 [1465.0, 1504.0] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=0 | explicit_feat | 1447.5 [1404.1, 1490.2] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a1_step2_gate_c1_20260926/gate.json` | p_c=0 | uncond_hist | 1444.7 [1403.2, 1486.5] | 0.00 | 无 | None | reveal | None | 9 | — |
| 否（消融/重复/内部臂） | `results/a3_baseline_full-cond_20260926_1007_230001/gate.json` | p_c=(10,20,30) | cond_hist | -507.3 [-682.2, -342.9] | 57.60 | 有 | True | density | None | 3 | — |
| 否（消融/重复/内部臂） | `results/a3_baseline_full-cond_20260926_1007_230001/gate.json` | p_c=(5,10,15) | cond_hist | 596.5 [519.4, 671.4] | 57.62 | 有 | True | density | None | 3 | — |
| 否（消融/重复/内部臂） | `results/a3_baseline_full-cond_20260926_1007_230001/gate.json` | p_c=0 | cond_hist | 1677.2 [1649.8, 1705.6] | 57.83 | 有 | True | density | None | 3 | — |
| ★ 是 | `results/i3_online_recovery_1009_v1/online_20260926/gate.json` | p_c=(5,10,15) | maskco | 456.3 [369.8, 538.0] | 37.33 | 有 | True | density | 0.0 | 4 | True |
| 否（消融/重复/内部臂） | `results/i3_online_recovery_1009_v1/online_20260926/gate.json` | p_c=(5,10,15) | warmstart | 447.0 [347.3, 538.8] | 36.83 | 有 | True | density | 0.0 | 4 | True |
| 否（消融/重复/内部臂） | `results/i3_online_recovery_1009_v1/online_20260926/gate.json` | p_c=(5,10,15) | reconstruction_control | 473.4 [384.4, 563.6] | 44.17 | 有 | True | density | 0.0 | 4 | True |
| 否（消融/重复/内部臂） | `results/i3_online_recovery_1009_v1/online_20260926/gate.json` | p_c=(5,10,15) | cond_hist | 601.0 [525.6, 673.7] | 42.70 | 有 | True | density | 0.0 | 4 | True |

## 4. 臂明细与身份

| 臂目录 | 方法 | 档位 | 正式/诊断 | 天数 | 可裁决 | 身份块 | source_stable | 记录配对身份 | gate 路径 |
|---|---|---|---|---:|---|---|---|---|---|
| attention_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/attention_51015/gate.json` |
| attention_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/attention_51015/gate.json` |
| attention_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/attention_51015_f/gate.json` |
| attention_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/attention_0/gate.json` |
| attention_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/attention_0/gate.json` |
| attention_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/attention_0_f/gate.json` |
| cada_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/cada_51015/gate.json` |
| cada_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | False | False | `results/a1_ext_extra/cada_51015/gate.json` |
| cada_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/cada_51015_f/gate.json` |
| cada_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/cada_0/gate.json` |
| cada_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/cada_0/gate.json` |
| cada_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/cada_0_f/gate.json` |
| coenrich_nopchgs_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/coenrich_nopchgs_51015/gate.json` |
| coenrich_nopchgs_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/coenrich_nopchgs_0/gate.json` |
| coenrich_pchgs_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/coenrich_pchgs_51015/gate.json` |
| coenrich_pchgs_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/coenrich_pchgs_0/gate.json` |
| deepaco_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/deepaco_51015/gate.json` |
| deepaco_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/deepaco_51015/gate.json` |
| deepaco_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/deepaco_51015_f/gate.json` |
| deepaco_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/deepaco_0/gate.json` |
| deepaco_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/deepaco_0/gate.json` |
| deepaco_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/deepaco_0_f/gate.json` |
| lih_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/lih_51015/gate.json` |
| lih_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/lih_51015/gate.json` |
| lih_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/lih_51015_f/gate.json` |
| lih_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/lih_0/gate.json` |
| lih_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | False | False | `results/a1_ext_extra/lih_0/gate.json` |
| lih_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | False | False | `results/a1_ext_extra/lih_0_f/gate.json` |
| mapt_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/mapt_51015/gate.json` |
| mapt_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/mapt_0/gate.json` |
| mvmoe_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/mvmoe_51015/gate.json` |
| mvmoe_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/mvmoe_51015/gate.json` |
| mvmoe_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/mvmoe_51015_f/gate.json` |
| mvmoe_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/mvmoe_0/gate.json` |
| mvmoe_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | False | False | `results/a1_ext_extra/mvmoe_0/gate.json` |
| mvmoe_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/mvmoe_0_f/gate.json` |
| rule_myopic_51015 | rule_myopic | p_c=(5,10,15) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_formal2/rule_myopic_51015/gate.json` |
| rule_myopic_51015 | rule_myopic | p_c=(5,10,15) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_extra/rule_myopic_51015/gate.json` |
| rule_myopic_0 | rule_myopic | p_c=0 | 正式 | 40 | True | 无 | None | — | `results/a1_ext_formal2/rule_myopic_0/gate.json` |
| rule_myopic_0 | rule_myopic | p_c=0 | 正式 | 40 | True | 无 | None | — | `results/a1_ext_extra/rule_myopic_0/gate.json` |
| omnivrp_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/omnivrp_51015/gate.json` |
| omnivrp_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/omnivrp_51015/gate.json` |
| omnivrp_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/omnivrp_51015_f/gate.json` |
| omnivrp_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/omnivrp_0/gate.json` |
| omnivrp_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/omnivrp_0/gate.json` |
| omnivrp_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/omnivrp_0_f/gate.json` |
| a1_ext_a05e_ortools | ortools_accept | p_c=(10,20,30) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_a05e_ortools/gate.json` |
| ortools | ortools_accept | p_c=(10,20,30) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/ortools/gate.json` |
| a1_ext_a05e_ortools | ortools_accept | p_c=(5,10,15) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_a05e_ortools/gate.json` |
| ortools | ortools_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/ortools/gate.json` |
| a1_ext_a05e_ortools | ortools_accept | p_c=0 | 正式 | 40 | True | 无 | None | — | `results/a1_ext_a05e_ortools/gate.json` |
| ortools | ortools_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/ortools/gate.json` |
| pomo_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/pomo_51015/gate.json` |
| pomo_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/pomo_51015_f/gate.json` |
| pomo_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/pomo_0/gate.json` |
| pomo_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | False | False | `results/a1_ext_extra/pomo_0/gate.json` |
| pomo_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/pomo_0_f/gate.json` |
| a1_ext_a05e_pyvrp | pyvrp_accept | p_c=(10,20,30) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_a05e_pyvrp/gate.json` |
| pyvrp | pyvrp_accept | p_c=(10,20,30) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/pyvrp/gate.json` |
| a1_ext_a05e_pyvrp | pyvrp_accept | p_c=(5,10,15) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_a05e_pyvrp/gate.json` |
| pyvrp | pyvrp_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/pyvrp/gate.json` |
| a1_ext_a05e_pyvrp | pyvrp_accept | p_c=0 | 正式 | 40 | True | 无 | None | — | `results/a1_ext_a05e_pyvrp/gate.json` |
| pyvrp | pyvrp_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/pyvrp/gate.json` |
| routefinder_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/routefinder_51015/gate.json` |
| routefinder_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | False | False | `results/a1_ext_extra/routefinder_51015/gate.json` |
| routefinder_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/routefinder_51015_f/gate.json` |
| routefinder_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/routefinder_0/gate.json` |
| routefinder_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | False | False | `results/a1_ext_extra/routefinder_0/gate.json` |
| routefinder_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/routefinder_0_f/gate.json` |
| rrnco_51015 | rrnco_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/rrnco_51015/gate.json` |
| rrnco_51015 | rrnco_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/rrnco_51015/gate.json` |
| rrnco_51015_f | rrnco_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/rrnco_51015_f/gate.json` |
| rrnco_0 | rrnco_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/rrnco_0/gate.json` |
| rrnco_0 | rrnco_accept | p_c=0 | 诊断 | 40 | True | 有 | None | False | `results/a1_ext_extra/rrnco_0/gate.json` |
| rrnco_0_f | rrnco_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/rrnco_0_f/gate.json` |
| sgbs_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/sgbs_51015/gate.json` |
| sgbs_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/sgbs_51015/gate.json` |
| sgbs_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/sgbs_51015_f/gate.json` |
| sgbs_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/sgbs_0/gate.json` |
| sgbs_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/sgbs_0/gate.json` |
| sgbs_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/sgbs_0_f/gate.json` |
| symnco_51015 | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/symnco_51015/gate.json` |
| symnco_51015 | ordering_accept | p_c=(5,10,15) | 诊断 | 40 | True | 有 | False | False | `results/a1_ext_extra/symnco_51015/gate.json` |
| symnco_51015_f | ordering_accept | p_c=(5,10,15) | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/symnco_51015_f/gate.json` |
| symnco_0 | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | True | `results/a1_ext_formal2/symnco_0/gate.json` |
| symnco_0 | ordering_accept | p_c=0 | 诊断 | 40 | True | 有 | True | False | `results/a1_ext_extra/symnco_0/gate.json` |
| symnco_0_f | ordering_accept | p_c=0 | 正式 | 40 | True | 有 | True | False | `results/a1_ext_extra/symnco_0_f/gate.json` |
| rule_valuee20_51015 | rule_valuee_lam2.00 | p_c=(5,10,15) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_formal2/rule_valuee20_51015/gate.json` |
| rule_valuee20_51015 | rule_valuee_lam2.00 | p_c=(5,10,15) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_extra/rule_valuee20_51015/gate.json` |
| rule_valuee20_0 | rule_valuee_lam2.00 | p_c=0 | 正式 | 40 | True | 无 | None | — | `results/a1_ext_formal2/rule_valuee20_0/gate.json` |
| rule_valuee20_0 | rule_valuee_lam2.00 | p_c=0 | 正式 | 40 | True | 无 | None | — | `results/a1_ext_extra/rule_valuee20_0/gate.json` |
| rule_valuee30_51015 | rule_valuee_lam3.00 | p_c=(5,10,15) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_formal2/rule_valuee30_51015/gate.json` |
| rule_valuee30_51015 | rule_valuee_lam3.00 | p_c=(5,10,15) | 正式 | 40 | True | 无 | None | — | `results/a1_ext_extra/rule_valuee30_51015/gate.json` |
| rule_valuee30_0 | rule_valuee_lam3.00 | p_c=0 | 正式 | 40 | True | 无 | None | — | `results/a1_ext_formal2/rule_valuee30_0/gate.json` |
| rule_valuee30_0 | rule_valuee_lam3.00 | p_c=0 | 正式 | 40 | True | 无 | None | — | `results/a1_ext_extra/rule_valuee30_0/gate.json` |

## 5. 不可适配 / 未适配方法

> 只列**机制层面**接不上的方法；其资产（权重/代码）本身是可加载的，缺的是可迁移到 A-v1
> 「接单 + 车队路由」决策的语义。MAPT 与 CO-enriched-ML **不在此列**：二者已适配并产出
> v2 正式臂（见 §1 主表；MAPT 的弱介入代价见 README §8）。

| 方法 | 不可适配的具体原因 | 依据文档 |
|---|---|---|
| Learning-to-Delegate | 机制不匹配：其回归目标是 HGS 子问题距离，规模 n≥500，而 A-v1 单决策可见客户 ≤90；资产本身可加载（3/3 strict，MAE 0.071–0.138），缺的是可迁移到「接单」决策的语义。 | `CC_Compare/Learning-to-Delegate/A1_ASSESSMENT.md` |
| PIP-constraint | 机制不匹配：模型是单车辆 TSPTW/TSPDL，其 env 定义里 num_vehicles / vehicle_id / capacity 出现 0 次，没有车队维度可对接 A-v1 的 15 车接单+路由决策；43/48 个 .pt 权重可加载，但「单车辆 → 多车辆」这一步接不上。 | `CC_Compare/PIP-constraint/A1_ASSESSMENT.md` |
