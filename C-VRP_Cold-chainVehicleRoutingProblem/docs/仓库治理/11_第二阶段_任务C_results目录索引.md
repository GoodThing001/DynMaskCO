# 11 · P0 第二阶段 · 任务 C：`results/` 目录索引与复现资格

> 只读枚举。所有计数用 `os.walk(followlinks=False)` + 逐目录 `lstat().st_file_attributes & 0x400`
> 显式拒绝 reparse point（junction/symlink）后统计；**未使用任何 `Get-ChildItem -Recurse`**。
> 本轮未改名、未移动、未删除任一目录。

## 0. 先更正第一阶段的计数

| 第一阶段写的 | 实测 |
|---|---|
| `results/` 有 "151 个顶层目录" | **150 个目录 + 11 个文件 = 161 项** |
| "~4,295 个文件 / 2.46 GB" | **3,823 文件 / 2,573,417,451 B ≈ 2.40 GiB**（junction 内部真实内容另计 461 文件 / 7.31 MiB） |
| "13 个跟踪文件" | ✅ 正确（`README.md` + `_m1a_synth` 4 + `_m1b_q2` 4 + `_m1b_q2fix` 4） |

第一阶段文档按"保留原文"原则不改，以本表为准。

## 1. junction 环：已识别并排除

`results/m0_scale/_bverify_pinned/` 下有 **4 个 reparse point**：

```
_bverify_pinned/MASKCO_code          -> <repo>/MASKCO_code
_bverify_pinned/shadow/MASKCO_code   -> <repo>/MASKCO_code
_bverify_pinned/shadow/data          -> <ext>/data
_bverify_pinned/shadow/results       -> <ext>/results     ← 回指祖先，递归即死循环
```

- 4 个 junction **全部**在 `m0_scale/_bverify_pinned/` 内，**最深位于第 3 层**（`results/m0_scale/_bverify_pinned/…`），
  顶层与第 2 层没有任何 junction。
- 拒绝规则：`os.walk(followlinks=False)` 之外，**额外**用 `os.lstat()` 的 `st_file_attributes` 判 reparse 位
  （Windows 上 `os.path.islink()` 对 JUNCTION 返回 False，**只靠它不够**——这正是 `tools/git_deliver.py`
  里 `_is_reparse_point()` 存在的原因）。
- 结果：**没有一次越界、没有一次超时、没有不可读目录、没有 0 文件目录误判**。
- 副产物：`m0_scale/results/m0_scale/` 是一个**真实存在的空嵌套目录**（不是 junction），不影响统计。

## 2. 表列语义（读表前必看）

| 列 | 含义 |
|---|---|
| **文件 / 大小** | junction 排除后的真实文件数与字节数 |
| **git** | `git ls-files` 跟踪数；`—` = 0 |
| **关键裁决** | 在该目录**深度 ≤3** 的小型 JSON 里找到的**第一个**判定键值，键限于 `verdict / adjudication / conclusion / numeric_passed / formal_adjudicable / primary_pilot_pass / registered_primary_passed / all_pass / status / gate`。**只取顶层键，不抓嵌套行级 `verdict`**（这些文件里有大量行级 `"verdict": "REFUSED"`，正则会把它们误报成文件裁决） |
| **证据来源** | 上述判定所在的文件路径 |
| **读取到的身份** | 该目录内能读到的第一个 markdown 标题（深度 ≤2） |
| **`—`** | **未取得机器可读身份**。这不是"不重要"，而是"该目录的判定在更深层或不带 `verdict` 字段"。**此列留空的行，其状态只能按 §4 的族级分类理解，本文件不为它们编造逐行身份。** |

机器可读覆盖率：**21/150 有判定键、16/150 有 md 标题、117/150 两者都没有**。
这是 `results/` 的**真实可索引程度**，也是本任务最重要的结论之一：
**绝大多数结果目录无法仅靠文件自身说明"这是什么、结论是什么"。**

## 3. 全量索引表（150 行）

| # | 路径（`results/…`） | 文件 | 大小 | git | 关键裁决（解析所得） | 证据来源 | 读取到的身份 |
|---:|---|---:|---:|:--:|---|---|---|
| 1 | `_d3_l1_leakscan` | 14 | 5.67 MiB | — | `verdict=STAGED_LEAK_PREVENTION_ACCEPTED` | C-VRP_Cold-chainVehicleRoutingProblem\results\_d3_l1_leakscan\acceptance_after_fix.json | — |
| 2 | `_diag_i1_smoke` | 1 | 75.2 KiB | — | — | — | — |
| 3 | `_diag_i1_smoke2` | 1 | 75.1 KiB | — | — | — | — |
| 4 | `_diag_i1_smoke3` | 1 | 77.1 KiB | — | — | — | — |
| 5 | `_diag_i2dep_smoke` | 1 | 1.3 KiB | — | — | — | — |
| 6 | `_m1a_synth` | 4 | 31.4 KiB | 4 | `status=M1A_SYNTHETIC_ENGINEERING_ACCEPTED` | C-VRP_Cold-chainVehicleRoutingProblem\results\_m1a_synth\acceptance.json | — |
| 7 | `_m1b_q2` | 4 | 59.2 KiB | 4 | `status=T_Q2_ALL_PASS` | C-VRP_Cold-chainVehicleRoutingProblem\results\_m1b_q2\regression.json | — |
| 8 | `_m1b_q2fix` | 4 | 38.2 KiB | 4 | `status=Q2FIX_ALL_PASS` | C-VRP_Cold-chainVehicleRoutingProblem\results\_m1b_q2fix\regression.json | — |
| 9 | `_m1b_q3` | 1 | 7.5 KiB | — | — | — | — |
| 10 | `_maskco_gen_check` | 4 | 228.4 KiB | — | — | — | — |
| 11 | `_maskco_p0_smoke` | 2 | 220.6 KiB | — | — | — | — |
| 12 | `_maskco_p0_smoke_clock` | 2 | 224.7 KiB | — | — | — | — |
| 13 | `_maskco_partial_smoke` | 2 | 224.8 KiB | — | — | — | — |
| 14 | `_newrev_staging` | 9 | 221.1 KiB | — | — | — | — |
| 15 | `_q3_next_20261017` | 63 | 1.16 MiB | — | `verdict=G2_QUALIFICATION_PASS` | C-VRP_Cold-chainVehicleRoutingProblem\results\_q3_next_20261017\g2\controls.json | — |
| 16 | `_steps_reuse_smoke` | 1 | 3.4 KiB | — | — | — | — |
| 17 | `_t31_saved_weights` | 2 | 224.9 KiB | — | — | — | — |
| 18 | `a1_a1_formal_40_1004_171907` | 1 | 6.2 KiB | — | `formal_adjudicable=True` | C-VRP_Cold-chainVehicleRoutingProblem\results\a1_a1_formal_40_1004_171907\adjudication_a1_ind… | — |
| 19 | `a1_a3` | 3 | 76.7 KiB | — | `formal_adjudicable=False` | C-VRP_Cold-chainVehicleRoutingProblem\results\a1_a3\A3_INDEPENDENT_RECHECK_ORIGINAL_1007_v2.json | — |
| 20 | `a1_a3_online_density_B_20260926_40_1003_040454` | 2 | 539.5 KiB | — | — | — | — |
| 21 | `a1_a3_online_density_B_20260930_40_1003_040454` | 2 | 542.1 KiB | — | — | — | — |
| 22 | `a1_a3_online_density_B_20261001_40_1003_040454` | 2 | 540.6 KiB | — | — | — | — |
| 23 | `a1_a3_online_density_C_20260926_40_1004_171907` | 2 | 540.4 KiB | — | — | — | — |
| 24 | `a1_a3_online_density_C_20260930_40_1004_171907` | 2 | 542.7 KiB | — | — | — | — |
| 25 | `a1_a3_online_density_C_20261001_40_1004_171907` | 2 | 540.9 KiB | — | — | — | — |
| 26 | `a1_a3_online_density_D_20260926_40_1003_040454` | 2 | 539.9 KiB | — | — | — | — |
| 27 | `a1_a3_online_density_D_20260930_40_1003_040454` | 2 | 542.1 KiB | — | — | — | — |
| 28 | `a1_a3_online_density_D_20261001_40_1003_040454` | 2 | 540.4 KiB | — | — | — | — |
| 29 | `a1_c1fresh_reveal_40` | 1 | 378.9 KiB | — | — | — | — |
| 30 | `a1_c1r_reveal_40` | 1 | 1.5 KiB | — | — | — | 修订版同版配对 —— 首批（13:45 启动）高负载超时伪影归档（2026-09-30） |
| 31 | `a1_c1rr2_reveal_40` | 2 | 382.8 KiB | — | `formal_adjudicable=False` | C-VRP_Cold-chainVehicleRoutingProblem\results\a1_c1rr2_reveal_40\adjudication.json | — |
| 32 | `a1_c1smoke2_reveal` | 1 | 32.6 KiB | — | — | — | — |
| 33 | `a1_compare_20260924` | 1 | 6.8 KiB | — | — | — | — |
| 34 | `a1_diag_null` | 1 | 445.5 KiB | — | — | — | — |
| 35 | `a1_ext_R1_ortools` | 2 | 52.6 KiB | — | — | — | a1_ext_R1_ortools —— A-05 前诊断（不作公平比较证据） |
| 36 | `a1_ext_a05e_ortools` | 2 | 60.4 KiB | — | — | — | OR-Tools 公平重基线（A-05e 修正后，2026-09-29） |
| 37 | `a1_ext_a05e_pyvrp` | 2 | 59.3 KiB | — | — | — | PyVRP 公平重基线（A-05e 修正 + 0.11.3 API 端口，2026-09-29） |
| 38 | `a1_ext_c1_ortools` | 1 | 96.7 KiB | — | — | — | — |
| 39 | `a1_ext_c1_pyvrp` | 1 | 53.6 KiB | — | — | — | — |
| 40 | `a1_ext_extra` | 12 | 257.5 KiB | — | — | — | — |
| 41 | `a1_ext_fixed_smoke` | 1 | 4.7 KiB | — | — | — | — |
| 42 | `a1_ext_formal2` | 13 | 336.7 KiB | — | — | — | — |
| 43 | `a1_ext_formal2_3080` | 8 | 209.7 KiB | — | — | — | — |
| 44 | `a1_ext_rrnco_smoke` | 1 | 8.7 KiB | — | — | — | — |
| 45 | `a1_ext_rrnco_smoke2` | 1 | 8.8 KiB | — | — | — | — |
| 46 | `a1_ext_v2_ortools` | 1 | 52.0 KiB | — | — | — | — |
| 47 | `a1_external_dev` | 41 | 847.5 KiB | — | — | — | — |
| 48 | `a1_formal_10s` | 1 | 6.6 KiB | — | — | — | — |
| 49 | `a1_formal_1s` | 1 | 6.6 KiB | — | — | — | — |
| 50 | `a1_l2b_smoke_pc0` | 1 | 8.9 KiB | — | — | — | — |
| 51 | `a1_l2c_smoke2_pc0` | 1 | 9.0 KiB | — | — | — | — |
| 52 | `a1_l2c_smoke_pc0` | 1 | 8.9 KiB | — | — | — | — |
| 53 | `a1_optim_campaign` | 2 | 3.1 KiB | — | — | — | — |
| 54 | `a1_q04_identity_audit_20260929` | 2 | 5.1 KiB | — | — | — | — |
| 55 | `a1_s3_2_l2c_full` | 1 | 1.2 KiB | — | — | — | a1_s3_2_l2c_full —— 原实现诊断归档（2026-09-29） |
| 56 | `a1_s3_2rr2_density_40` | 2 | 383.5 KiB | — | `formal_adjudicable=False` | C-VRP_Cold-chainVehicleRoutingProblem\results\a1_s3_2rr2_density_40\adjudication.json | — |
| 57 | `a1_smoke` | 1 | 6.9 KiB | — | — | — | — |
| 58 | `a1_smoke_unified` | 1 | 6.9 KiB | — | — | — | — |
| 59 | `a1_step2_fixed_smoke` | 1 | 4.0 KiB | — | — | — | — |
| 60 | `a1_step2_gate` | 2 | 135.4 KiB | — | — | — | ⚠️ 本目录为「旧实现」证据（2026-09-26 审计后标注） |
| 61 | `a1_step2_gate_c1_20260926` | 2 | 377.8 KiB | — | — | — | C1 修正基线 + 两验证种子复验（2026-09-28）—— 结果记录 |
| 62 | `a1_step2_gate_c1_verify_20260930` | 1 | 376.4 KiB | — | — | — | — |
| 63 | `a1_step2_gate_c1_verify_20261001` | 1 | 375.2 KiB | — | — | — | — |
| 64 | `a1_step2_gate_fixed_20260926` | 2 | 132.7 KiB | — | — | — | 修正后 SAA 的步骤 2 门调试复跑（20260926，开发证据） |
| 65 | `a1_step2_gate_l1_20260926` | 1 | 0.0 KiB | — | — | — | — |
| 66 | `a1_step2_gate_l2b2_20260926` | 1 | 371.2 KiB | — | — | — | — |
| 67 | `a1_step2_gate_verify_20260930` | 2 | 372.7 KiB | — | — | — | A-v1 步骤 2 门复验结果（20260930 / 20261001，修正后 SAA） |
| 68 | `a1_step2_gate_verify_20261001` | 1 | 370.2 KiB | — | — | — | — |
| 69 | `a1_step2_gate_vote2_20260926` | 1 | 369.5 KiB | — | — | — | — |
| 70 | `a1_step2_smoke_local` | 1 | 4.1 KiB | — | — | — | — |
| 71 | `a1_step2_smoke_srv` | 1 | 9.7 KiB | — | — | — | — |
| 72 | `a1_step3` | 18 | 489.9 KiB | — | — | — | results/a1_step3 — MaskCO 原型训练世代清单（诊断资产，非 A3 正式证据） |
| 73 | `a1_step3_dev` | 2 | 66.6 KiB | — | — | — | ⚠️ 本目录仅为「旧实现」诊断证据（2026-09-26 审计后标注） |
| 74 | `a1_step3_smoke_local` | 1 | 3.7 KiB | — | — | — | — |
| 75 | `a3_baseline_full-cond_20260926_1007_230001` | 2 | 130.0 KiB | — | — | — | — |
| 76 | `a3_baseline_full-cond_20260930_1007_230001` | 2 | 130.4 KiB | — | — | — | — |
| 77 | `a3_baseline_full-cond_20261001_1007_230001` | 2 | 130.2 KiB | — | — | — | — |
| 78 | `a3_baseline_smoke-cond_20260926_1007_223507` | 2 | 7.6 KiB | — | — | — | — |
| 79 | `a3_baseline_smoke_20260926_1007_220517` | 2 | 11.7 KiB | — | — | — | — |
| 80 | `a3_repair_chain_1007_2250` | 29 | 167.8 KiB | — | `formal_adjudicable=True` | C-VRP_Cold-chainVehicleRoutingProblem\results\a3_repair_chain_1007_2250\chain_state.json | — |
| 81 | `b0_gap_decomposition` | 2 | 10.7 KiB | — | — | — | — |
| 82 | `b0_gap_decomposition_smoke` | 2 | 1.3 KiB | — | — | — | — |
| 83 | `c0` | 3 | 11.2 KiB | — | — | — | — |
| 84 | `delivery_20_round_audit_20261010_v2` | 1 | 14.2 KiB | — | — | — | — |
| 85 | `delivery_engineering_1008_v1` | 15 | 12.8 KiB | — | — | — | — |
| 86 | `delivery_pilot_1008_v1` | 12 | 22.04 MiB | — | — | — | — |
| 87 | `delivery_pilot_1008_v2` | 23 | 22.30 MiB | — | `primary_pilot_pass=False` | C-VRP_Cold-chainVehicleRoutingProblem\results\delivery_pilot_1008_v2\INDEPENDENT_FINAL_ADJUDI… | — |
| 88 | `delivery_provider_1008_v1` | 4 | 1.4 KiB | — | — | — | — |
| 89 | `delivery_provider_audit_1008_v2` | 2 | 9.0 KiB | — | — | — | — |
| 90 | `heartbeat_checks` | 32 | 190.0 KiB | — | — | — | — |
| 91 | `hfr` | 17 | 30.87 MiB | — | — | — | — |
| 92 | `i3_collection_engineering_local_1008_v1` | 11 | 296.3 KiB | — | — | — | — |
| 93 | `i3_day12_collection_failure_diag_1008_v1` | 6 | 6.03 MiB | — | — | — | — |
| 94 | `i3_online_adapter_engineering_local_1008_v1` | 21 | 907.8 KiB | — | — | — | — |
| 95 | `i3_online_recovery_1009_v1` | 30 | 1.12 MiB | — | `status=PAUSED` | C-VRP_Cold-chainVehicleRoutingProblem\results\i3_online_recovery_1009_v1\AUTOMATION_PAUSE_REC… | — |
| 96 | `i3_optimizer_pretrained_probe_1008_0915` | 5 | 140.3 KiB | — | — | — | — |
| 97 | `i3_optimizer_pretrained_probe_1008_v2` | 6 | 150.4 KiB | — | — | — | — |
| 98 | `i3_optimizer_pretrained_probe_1008_v3` | 6 | 152.6 KiB | — | — | — | — |
| 99 | `i3_registered_1008_v1` | 1 | 0.3 KiB | — | — | — | — |
| 100 | `i3_registered_1008_v2` | 9 | 224.0 KiB | — | — | — | — |
| 101 | `i3_registered_1008_v3` | 14 | 292.93 MiB | — | `formal_adjudicable=True` | C-VRP_Cold-chainVehicleRoutingProblem\results\i3_registered_1008_v3\A3_ACTIVATION.json | — |
| 102 | `i3_registered_1008_v4` | 20 | 292.88 MiB | — | — | — | — |
| 103 | `i3_registered_1008_v5` | 42 | 307.96 MiB | — | — | — | — |
| 104 | `i3_runner_engineering_local_1008_1110` | 4 | 192.4 KiB | — | — | — | — |
| 105 | `i3_runner_engineering_local_1008_1110_v2` | 7 | 378.4 KiB | — | — | — | — |
| 106 | `i3_runner_engineering_local_1008_1110_v3` | 7 | 378.4 KiB | — | — | — | — |
| 107 | `i3_runner_engineering_server_1008_1110` | 4 | 248.7 KiB | — | — | — | — |
| 108 | `i3_teacher_200_archive_1008_1055` | 4 | 7.34 MiB | — | — | — | — |
| 109 | `i3_teacher_200_archive_replay_1008` | 3 | 2.0 KiB | — | — | — | — |
| 110 | `i3_teacher_200_probe_1008_0905` | 2 | 0.7 KiB | — | — | — | — |
| 111 | `i3_teacher_200_probe_1008_0908` | 8 | 7.2 KiB | — | — | — | — |
| 112 | `i3_teacher_archive_small_local_1008` | 3 | 6.96 MiB | — | — | — | — |
| 113 | `i3_teacher_archive_small_server_1008` | 3 | 6.95 MiB | — | — | — | — |
| 114 | `i3_teacher_data_audit_1008_v1` | 1 | 5.1 KiB | — | — | — | — |
| 115 | `i3_teacher_data_audit_1008_v2` | 1 | 1.9 KiB | — | — | — | — |
| 116 | `i3_teacher_data_audit_1008_v3` | 3 | 196.5 KiB | — | — | — | — |
| 117 | `i3_teacher_risk_1008_v1` | 8 | 89.2 KiB | — | — | — | — |
| 118 | `i3_teacher_risk_1008_v2` | 2 | 2.0 KiB | — | — | — | — |
| 119 | `i3_teacher_risk_1008_v3` | 9 | 170.0 KiB | — | — | — | — |
| 120 | `i3_training_runner_probe_1008_1110` | 2 | 0.5 KiB | — | — | — | — |
| 121 | `jf15_b1` | 4 | 6.3 KiB | — | — | — | — |
| 122 | `jf15_b16` | 4 | 6.3 KiB | — | — | — | — |
| 123 | `jf15_b4` | 4 | 6.3 KiB | — | — | — | — |
| 124 | `jf15_b8` | 4 | 6.3 KiB | — | — | — | — |
| 125 | `jf1_smoke` | 4 | 10.8 KiB | — | — | — | — |
| 126 | `jf1_three_way` | 4 | 21.1 KiB | — | — | — | — |
| 127 | `jf2` | 33 | 14.87 MiB | — | — | — | — |
| 128 | `jf2_closure` | 8 | 13.2 KiB | — | — | — | — |
| 129 | `m0_scale` | 777 | 628.50 MiB | — | `verdict=PROTOCOL_PASS` | C-VRP_Cold-chainVehicleRoutingProblem\results\m0_scale\o0cc_seq_cal\summary.json | O0-CC sequential/local oracle 运行说明（m0_scale 补跑，2026-09-17） |
| 130 | `maskco_fitcheck` | 2 | 471.1 KiB | — | — | — | — |
| 131 | `o0cc` | 1516 | 274.37 MiB | — | `verdict=PROTOCOL_PASS` | C-VRP_Cold-chainVehicleRoutingProblem\results\o0cc\agg_rev.json | 控制链验收报告（第 4 步，不跑 oracle rollout） |
| 132 | `o0d` | 15 | 765.9 KiB | — | `verdict=GO` | C-VRP_Cold-chainVehicleRoutingProblem\results\o0d\parity_w1\summary.json | — |
| 133 | `p0a` | 1 | 1.1 KiB | — | — | — | — |
| 134 | `p0m` | 1 | 7.4 KiB | — | — | — | — |
| 135 | `p0r` | 10 | 78.1 KiB | — | — | — | — |
| 136 | `p0s` | 1 | 1.0 KiB | — | — | — | — |
| 137 | `p0u` | 1 | 0.5 KiB | — | — | — | — |
| 138 | `phase0_baseline_freeze` | 2 | 2.4 KiB | — | — | — | — |
| 139 | `q3_s1_route_bracket` | 1 | 23.4 KiB | — | `verdict=S1_CONTROLS_PASS` | C-VRP_Cold-chainVehicleRoutingProblem\results\q3_s1_route_bracket\controls.json | — |
| 140 | `q3_s2_online_cost` | 1 | 12.5 KiB | — | `verdict=S2_CONTROLS_PASS` | C-VRP_Cold-chainVehicleRoutingProblem\results\q3_s2_online_cost\controls.json | — |
| 141 | `q3_s3_float_outward_bound` | 1 | 35.8 KiB | — | `verdict=S3_CONTROLS_PASS` | C-VRP_Cold-chainVehicleRoutingProblem\results\q3_s3_float_outward_bound\controls.json | — |
| 142 | `r1_5_model_utility` | 37 | 62.30 MiB | — | — | — | — |
| 143 | `r1_7` | 33 | 2.98 MiB | — | — | — | — |
| 144 | `r1_baseline` | 1 | 0.5 KiB | — | — | — | — |
| 145 | `reproduction_20_round_audit_20261010` | 2 | 40.9 KiB | — | — | — | 三线 20 轮结题记录：独立支持等级核查 |
| 146 | `research_20261010` | 531 | 273.95 MiB | — | `all_pass=True` | C-VRP_Cold-chainVehicleRoutingProblem\results\research_20261010\PARENT_CLAIM_GATE_HOLDOUT.json | 三线二十轮结题陈述：独立核查更正版 |
| 147 | `research_20261010_continuation_v2` | 517 | 178.99 MiB | — | `verdict=The 3080 Windows host is fastest on every single metric (1.3-1.9x faster than local, …` | C-VRP_Cold-chainVehicleRoutingProblem\results\research_20261010_continuation_v2\MACHINE_SELEC… | MASKCO 三线研究：交接文档（2026-10-10，第 21 轮） |
| 148 | `rule_control_smoke_unified` | 1 | 7.3 KiB | — | — | — | — |
| 149 | `three_line_corrections_20261009` | 80 | 1.94 MiB | — | — | — | — |
| 150 | `three_line_parallel_20261009` | 11 | 124.4 KiB | — | — | — | — |
| | **合计 150 个目录** | **4284** | **2461.53 MiB** | **12** | | | |

## 4. 状态分类（按族，附证据）

> 族级判定依据：目录内文件内容 + 已入库文档的登记。**不按目录名推断**。
> 判定词汇沿用 `results/README.md` 已声明的分类（CURRENT/已复核 · VALID BASELINE · EVIDENCE ·
> FROZEN NEGATIVE · FROZEN DIAGNOSTIC · SUPERSEDED · SMOKE · INCOMPLETE/HISTORICAL · INVALIDATED），
> 该 README 最后更新 2026-09-02，**只覆盖 P0 期目录**。

| 族 | 代表目录 | 规模 | 状态 | 证据/依据 |
|---|---|---|---|---|
| **P0 期** | `p0a` `p0m` `p0r` `p0s` `p0u` `c0` `o0d` | 7 目录 | `CURRENT/已复核`（P0-M 7/7、C0 20/20） | `results/README.md` 状态表；本轮重跑 `test_coldchain_contract.py` 20/20 PASS |
| **官方门控** | `o0cc/dev_gate_v4_official` | 1,209 文件 | **唯一封印官方裁决** `verdict=GO` | `dev_gate_summary.json`；9 个 `canonical.COMPLETE` |
| **官方门控（非官方兄弟）** | `o0cc/dev_gate_drift_diag_v1` | 含 `r1_02` 与 `parity_r1_02` | **DIAGNOSTIC_NOT_OFFICIAL** | 该目录 `DIAGNOSTIC_NOT_OFFICIAL.md` 逐字："these results MUST NOT be used for the official DEV-GATE lookahead decision" |
| **门控回归/parity** | `o0cc/r4_* r5_* r6_wire v4_archive_wire_regression` + `agg_rev.json` `cross_cell_summary_v2.json` | 27 子目录 | `PROTOCOL_PASS`，但 **`mode="regression"` / 18 实例**，**不是** 1152 官方门 | 各 `summary.json` 的 `verdict` + `mode` 字段 |
| **A-v1 战役** | `a1_*`（57 目录） | 最大族 | 混合：多数 `无 verdict 字段`；判定在 `gate.json` 的数值与 `formal_adjudicable` 上 | ⚠️ 见 §5.2 |
| **A3 / I3** | `a3_*`（6）、`i3_*`（29） | — | `a3_repair_chain_1007_2250` = `formal_adjudicable=true` 且 **9 个臂×种子主档全 `numeric_passed=false`**（干净正式 FAIL）；I3 首开发 40 日 `registered_primary_passed=false` | 各 `gate.json`/`adjudication` |
| **配送线** | `delivery_*`（7） | — | `delivery_pilot_1008_v2` `primary_pilot_pass=false`；v1 是**工程激活故障**（96 次运行 0 重构），不可裁作方法失败 | `results/delivery_pilot_1008_v2/`、`02_配送线/文档/结果与进度/…2026-10-08.md` |
| **冻结负结果** | `hfr` `jf2` `jf2_closure` | — | `FROZEN NEGATIVE`：`SERVICE GATE` hfr=FAIL / jf2=PASS，`COST GATE A`=FAIL | 各 `gate.json`；`results/README.md` |
| **冻结诊断** | `jf15_b1 b4 b8 b16`、`jf1_smoke` | 5 | `FROZEN DIAGNOSTIC` | `results/README.md` |
| **作废/替代** | `phase0_baseline_freeze`（INVALIDATED）、`b0_gap_decomposition`（SUPERSEDED） | 2 | 明确标注 | `results/README.md` |
| **历史不完整** | `r1_baseline`、`a1_step3`（早期代） | — | `INCOMPLETE/HISTORICAL`；`a1_step3/CALIBRATION_I1.md` 自述"全表作废"（旧 init 权重） | 文件内自述 |
| **Q3** | `q3_s1_route_bracket` `q3_s2_online_cost` `q3_s3_float_outward_bound` `_q3_next_20261017` | 66 文件 / 1.23 MiB | `S1/S2/S3_CONTROLS_PASS` + `"kind":"DIAGNOSTIC"` + `"not_for_production":true`；s3 另有 `kappa_verdict=EXP_ERROR_BOUND_NOT_ESTABLISHED`、`f_level_established=false` | 各 `controls.json` |
| **事后研究树** | `research_20261010`（531）`research_20261010_continuation_v2`（517） | 1,048 文件 / ≈453 MiB | `WORK_STATUS.json` 在 r1–r20 全记 `stable_online_gain:false` | 该 JSON |
| **三线** | `three_line_parallel_20261009`、`three_line_corrections_20261009`、`delivery_20_round_audit_20261010_v2`、`reproduction_20_round_audit_20261010` | 4 | 轮次审计记录 | 目录内 JSON |
| **冒烟 / 临时（下划线前缀）** | `_diag_*_smoke`（4）、`_maskco_*_smoke*`（3）、`_steps_reuse_smoke`、`_newrev_staging`、`_t31_saved_weights`、`maskco_fitcheck`、`_probe_roundtrip.bin` 等 | ~15 | `SMOKE` / staging | 内容形态（日志/权重/无判定） |
| **强制跟踪的验收证据（下划线但**不是**草稿）** | `_m1a_synth` `_m1b_q2` `_m1b_q2fix` | 各 4 文件，**全部跟踪** | `EVIDENCE`（`PASS` / `OMEGA10_INITIAL_VALUE_MISMATCH` / `OK`） | 根 `.gitignore` 为它们写了否定规则 |
| **`m0_scale`** | 1 顶层目录 | 最大单目录 | 混合：M0/M1/M-pre/event_plan/⑤ 归因 全部落在这里 | 目录内子目录；`_bverify_pinned` 是 scratch |

## 5. 官方 vs 非官方：三个必须说清的坑

### 5.1 全树只有一个封印官方裁决

`results/o0cc/dev_gate_v4_official/dev_gate_summary.json` → `verdict = "GO"`，
九 cell 等权 Δ = **−0.27713774115465817**，grouped-bootstrap CI **[−0.28480, −0.26959]**，
1152 实例 / 128 seed 组 / `bootstrap {n_boot:10000, seed:42}`。
**它是 pilot 离线 oracle 证据，不是 learned-method 声称。**

### 5.2 最大的误引用陷阱：`a1_s3_2rr2_density_40`

主档 **+63.44 [23.57, 103.77]，`numeric_passed = true`**，
但同一文件里 **`formal_adjudicable = false`**，原因是 `identity_source_hash_mismatch` /
`identity_source_not_stable`（源码启动/结束哈希不一致）。
`a1_c1rr2_reveal_40` 同样是 `formal_adjudicable=false`。
**一个 CI 全为正的"数字通过"，被它自己的证据拒绝授予正式资格。**
反向的例子也存在：`a3_repair_chain_1007_2250` 是 `formal_adjudicable=true` 而 9 个主档全 `numeric_passed=false`
——这是一个**干净的正式 FAIL**。

→ 引用 `results/` 里任何数字前，**必须同时读 `formal_adjudicable` / `numeric_passed` / `identity_ok` / `zero_timeout_ok`
/ `not_for_production` / `baseline_override_used`**，不能只看 CI。

### 5.3 `q3_c2` / `q3_g3` 不是顶层目录

它们只存在于 `_q3_next_20261017/c2/` 与 `_q3_next_20261017/g3/` 之下。
按顶层目录名找 Q3 的 C2/G-3 证据**会找不到**。

### 5.4 "首命中 ≠ 权威"：表里至少两行的判定已被后续文件取代

§3 的表取的是**深度 ≤3 的第一个判定键**。这是机械规则，会命中**已作废的旧名**。两处实测例子：

| 行 | 表里显示的 | 权威状态（**以这个为准**） | 出处 |
|---|---|---|---|
| `_q3_next_20261017` | `verdict=G2_QUALIFICATION_PASS` | **`G2_DIAGNOSTIC_COMPLETE_WITH_ERRATA`**。命名链：`G2_QUALIFICATION_PASS` → `G2_QUALIFICATION_PASS_WITH_ERRATA` → `G2_DIAGNOSTIC_COMPLETE_WITH_ERRATA`；原文写「本轮**不再**使用任何"PASS"措辞描述整体 G-2」 | `docs/当前规划/决策与审计/G2_勘误裁决与G1隔离候选_2026-10-17.md:11` |
| `o0cc` | `verdict=PROTOCOL_PASS`（来自 `agg_rev.json`） | **该文件是 `mode="regression"` 的 18 实例回归**，不是 1152 实例官方门。官方门只在 `o0cc/dev_gate_v4_official/dev_gate_summary.json`（`GO`） | 各 `summary.json` 的 `mode` 字段 |

**规则**：表里的判定只是"该目录里第一个机器可读的判定"，**不是最终裁决**。
引用前必须按 §5.2 同时读限定词，并回到 `04` 的台账核对。

## 6. 复现阻断（按类聚合）

| # | 阻断类别 | 影响面 | 说明 |
|---|---|---:|---|
| 1 | **结果不在 git** | 147/150 目录、3,810/3,823 文件 | 干净 clone 只有 13 个文件 |
| 2 | **依赖被忽略的 `data/`** | ≈110 目录 | `data/` 本地存在（2,217 MB），但 clone 后不存在；且其**哈希清单本身也未跟踪**（见 `10` §5） |
| 3 | **依赖本机不存在的 checkpoint** | ≈18 目录 | `C-VRP_…/ckpts/` **目录不存在**，`ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt` **全树搜索缺失**。受影响：`m0_scale` 的 mpre/fsq/step5/event_plan/sgbs、`hfr`、`jf2`、`jf2_closure`、`r1_5_model_utility`、`r1_7`、`a1_step3`、`jf1_three_way`、`jf15_*`。注：`MASKCO_code/ckpts/cvrp100.ckpt`（305 MB）**在**。 |
| 4 | **悬空引用：被文档/裁决点名但目录不存在** | 5 个 | `a1_rr3_density_40_1001_234935`、`a1_rr3_robust_20260930_density_40_1002_145359`、`a1_rr3_robust_20261001_density_40_1002_145359`（被 `A2_LOCK.json` 与 A3 裁决点名）、`a1_s3_2r_density_40`、`delivery_pilot_1008_v1/src`（被 `delivery_provider_audit_1008_v2` 点名）。**A2/A3 链的裁决依赖的基线目录缺失。** |
| 5 | 外部基线权重不在库 | ~78 个 `gate.json`（`a1_ext_*`） | CaDA `checkpoint-300.pt` 等 |
| 6 | 只有聚合、无逐日行 | 10 目录 | 无法重算 CI |
| 7 | 身份被工件自身拒绝 | 2 + 9 + 9 个 a1 目录 | 见 §5.2 |
| 8 | 封印文件非 UTF-8 | 1（`o0cc/dev_gate_v4_official/EVIDENCE_SEAL.json`，GBK） | 见 `09` §6.2 |
| 9 | **本机现在就能重跑** | `c0` `_m1a_synth` `_m1b_q2` `_m1b_q2fix` `_m1b_q3` `_q3_next_20261017` `q3_s1/s2/s3` `_steps_reuse_smoke` `_d3_l1_leakscan` | 纯 NumPy |

## 7. 增量缺口：2026-10-13 之后、还没进 `EVIDENCE_MANIFEST.md` 的证据

### 7.1 manifest 的实际边界（实测，未改动它）

- 表行 **474 行**，**474/474 全部以 `research/review_3080_20261010/` 开头**；**`results/` 下的行数为 0**。
- 全文出现 `q3` **0 次**、`20261014…20261018` **各 0 次**；Release tag 固定 `evidence-2026-10-13`。
- ⇒ **`results/` 从来不在它的收录范围内**（它的判据是 `research/` 下 ≥1 MiB 的 `*.json/*.jsonl`）。

### 7.2 缺口清单（我实测，未声称完整）

**A. `research/` 下 ≥1 MiB 且晚于 2026-10-13 的 json/jsonl：13 个文件，合计约 27.2 MiB**

| MiB | mtime | 路径（`research/review_3080_20261010/…`） |
|---:|---|---|
| 3.08 | 10-14 00:46 | `pickup/strong_independent_owner_next_root_v2/OWNER_RESULT_0.json` |
| 2.95 | 10-14 01:15 | `pickup/strong_independent_owner_fault3_root_v1/CANDIDATE_RESULT_0.json` |
| 2.94 | 10-14 00:46 | `pickup/strong_independent_owner_next_root_v2/CANDIDATE_RESULT_0.json` |
| 2.85 | 10-14 00:30 | `pickup/strong_independent_owner_root_v3/OWNER_RESULT_0.json` |
| 2.72 | 10-14 00:30 | `pickup/strong_independent_owner_root_v3/CANDIDATE_RESULT_0.json` |
| 2.58 | 10-14 05:18 | `pickup/learned_after_strong_recovery_execution_root_v2/RESULT.json` |
| 2.18 | 10-14 01:15 | `pickup/strong_independent_owner_fault3_root_v1/RESPONSE_ATTEMPT_0.json` |
| 2.17 | 10-14 00:46 | `pickup/strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json` |
| 2.02 | 10-14 00:30 | `pickup/strong_independent_owner_root_v3/RESPONSE_ATTEMPT_0.json` |
| 1.58 ×2 | 10-14 05:52 | `delivery/split_final_timed_execution_root_v1/downloaded/TIMED_AUDIT_SOURCE_{START,END}.json` |
| 1.55 ×2 | 10-14 05:52 | `delivery/split_final_timed_execution_root_v1/downloaded/{pretrained_initial_fault0_worker,random888_initial_fault0_worker}/SOURCE_START.json` |

⇒ **这 13 个文件若丢失，manifest 无法证明它们存在过。**

**B. 整个 `research/` 里晚于 2026-10-13 的文件：8,571 个 / 157,522,003 B（≈150.2 MiB），全部在 `review_3080_20261010/` 下。**

**C. 扩展 `results/` 里晚于 2026-10-13 的目录：15 个**

| 目录 | 新文件 | 新字节 | 最近 mtime | 说明 |
|---|---:|---:|---|---|
| `_q3_next_20261017` | 63 | 1,191.3 KiB | 10-17 18:08 | Q3 各轮（含 5 版 SHA256 manifest、`gtmin/*.PRE_FIX`）**未被任何 manifest 覆盖** |
| `_d3_l1_leakscan` | 14 | 5,803.0 KiB | 10-17 00:08 | 密钥扫描验收 |
| `a1_ext_formal2` | 5 | 127.1 KiB | 10-14 03:11 | 外部基线 |
| `_m1a_synth` `_m1b_q2` `_m1b_q2fix` | 4+4+4 | 128.8 KiB | 10-16 22:15 | **已跟踪** |
| `_m1b_q3` | 1 | 7.5 KiB | 10-17 01:37 | — |
| `q3_s1_route_bracket` `q3_s2_online_cost` `q3_s3_float_outward_bound` | 1+1+1 | 71.7 KiB | 10-17 19:22–19:23 | S1/S2/S3 控制行 |
| `m0_scale` | 1 | 87,492.8 KiB | 10-15 22:32 | 单个 85 MiB 文件 |
| `research_20261010_continuation_v2` | 1 | 276.9 KiB | 10-14 11:35 | — |
| `o0cc` `c0` `o0d` | 4+1+1 | 19.9 KiB | **10-18 00:59–01:00** | ⚠️ **是本届 P0 回归测试写入的**，不是科研证据（见 §8） |

**D. 判据层面的结论**：Q3 这一族（66 文件 / 合计约 1.23 MiB）**没有任何单个文件 ≥1 MiB**，
所以 `EVIDENCE_MANIFEST` 的"≥1 MiB"判据**在结构上永远不可能收录 Q3**。
这是**判据缺口**，不是遗漏——再补册也补不进去，必须另立一条索引（本文件就是那条索引）。

**我不声称现有证据包是完整的。**

## 8. 必须披露：本轮回归测试改写了 `results/` 下 6 个测试记录文件

第一阶段与第二阶段为验证控制链，运行了 `scripts/tests/*`，这些测试**按设计**会写自己的记录：

| 目录 | 被重写的文件 | 性质 |
|---|---|---|
| `results/c0/` | `coldchain_contract_tests.json` | 测试输出（非裁决） |
| `results/o0d/` | `sequential_oracle_tests.json` | 同上 |
| `results/o0cc/` | `aggregate_cross_cell_tests.json`、`control_chain_tests.json`、`dev_gate_control_tests.json`、`oracle_*_tests.json`、`protocol_integrity_tests.json` 等 | 同上 |

- 这些文件**均未被 git 跟踪**，且是"同一测试的复跑结果"，不是科研裁决文件。
- 但它们的 **mtime 与内容确实被更新了**。按"不得覆盖历史证据"的谨慎口径，**此处如实登记**。
- 若你要求严格冻结，正确做法是让测试写入临时目录（需改测试源码）——**本轮未做，也不建议为治理目的去改测试**。

## 9. 怎么读这份索引（给新研究员）

1. **先看状态，别看名字。** `_` 前缀 ≠ 草稿（`_m1a_synth` 是正式验收证据）；`GO` 字面 ≠ 官方（`C2_BINDING_GO` 是分析绑定）。
2. **先开证据文件，再看数字。** 150 个目录里 117 个没有机器可读身份——不要凭目录名判断它是什么。
3. **数字必须与限定词一起读**：`formal_adjudicable` / `numeric_passed` / `not_for_production` / `identity_ok`。
4. **想重跑之前先看 §6 第 3 条**：≈18 个目录在本机**不可复现**（缺 256-dim encoder checkpoint）。
5. **`results/` 的任何内容都不在 git 里**（除 13 个文件）。引用之前先确认本机还在。
