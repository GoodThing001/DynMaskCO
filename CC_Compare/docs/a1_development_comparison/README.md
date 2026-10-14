# A-v1 对比方法总表（2026-10-13，对比方法线）

本目录是对比方法线在**统一 A-v1 接单协议**下的结果汇总交付物。

| 文件 | 内容 |
|---|---|
| `A-v1对比方法总表.md` | 人类可读总表：Ours 摘要 / 对比方法主表 / C1 基线臂 / 主模型与内部臂 / 臂明细与身份 / 未适配方法 |
| `A-v1对比方法总表.xlsx` | 同内容的 Excel（5 sheet：性能总表 / C1基线臂 / 臂明细与身份 / 主模型与内部臂 / 未适配方法） |
| `A-v1对比方法效用图.png` | 论文/汇报配图：左=各方法效用（p_c=0 与主档）+ C1 基线参考线；右=相对 C1 的配对差（95% CI） |
| `dataset.json` | 机器可读数据集（55 个臂的逐臂摘要 + 重配对结果 + 身份状态） |

生成链（服务器执行采集，本地渲染）：

```bash
# 服务器（C-VRP_Cold-chainVehicleRoutingProblem 下）
/home/hzeng/envs/a1_ext/bin/python ../CC_Compare/tools/collect_compare_table.py \
    --json ../CC_Compare/results/a1_compare_table/dataset.json
# 本地
python CC_Compare/tools/render_compare_table.py \
    --data CC_Compare/results/a1_compare_table/dataset.json \
    --outdir CC_Compare/results/a1_compare_table
```

统计口径逐位复刻项目 `run_exp_reserve._stat`（按天聚类 bootstrap，seed+777，
2000 次重采样，2.5/97.5 分位）与 `run_a1_external_accept.paired_all_days`
（按 day_id 显式配对，以预期 40 个唯一 day id 验收；缺日 = 不可裁决，不静默过滤）。

---

## 1. 覆盖范围

**已适配并跑满 40 天（seed 20260926 开发集）**：

| 类别 | 方法 | 臂 |
|---|---|---|
| 经典求解器 | OR-Tools-RH accept、PyVRP-RH accept | 各 3 档 p_c |
| 学习型（11 个） | RouteFinder / CaDA / MVMoE / POMO / AttentionModel / Sym-NCO / SGBS / DeepACO / Omni-VRP / Learn-Improvement-Heuristics / RRNCO-Ordering | 各 2 档（p_c=0、主档） |
| 规则基线 | myopic、value/energy λ=2、λ=3 | 各 2 档 |
| 内部 C1 臂 | cond_hist / uncond_hist / explicit_feat | 各 3 档 |

**未适配（如实记录，不按输赢筛选，现仅 2 个）**：Learning-to-Delegate（机制为 n≥500 子问题委托，
需 LKH/HGS 子求解器，与 A-v1 每决策 ≤90 池规模方向相反）、
PIP-constraint（单车辆 TSPTW，缺容量/多车，结构性不可比）。依据见各自 `CC_Compare/*/A1_ASSESSMENT.md`。
⚠️ **原先也列在此处的 MAPT 与 CO-enriched-ML 已完成适配**，故**不再属于「未适配」**：
MAPT 已按桥接语义自训、并找到官方参数空间内的可行档（`config=fast`）→ 40 天臂已获人工放行并进入总表
（见 §8，须带「弱介入」限定）；CO-enriched-ML 已完成 accept/reject 改造 + 重训 + PC-HGS 编译
（2 模式 × 2 档）。**这一点容易写错**：两者既产出结果行，就**不能**再出现在「未适配」清单里。

---

## 2. 主结果（p_c=0 无拒绝罚 / 主档 p_c=(5,10,15)）

C1 基线（身份合格 gate `a1_c1rr2_reveal_40`）：**p_c=0 = 1703.7**，**主档 = 631.4**。

| 排序 | 方法 | p_c=0 | 主档 |
|---|---|---:|---:|
| — | **C1 cond_hist（SAA 强基线）** | **1703.7** | **631.4** |
| — | ★ **主模型 MaskCO（A3 pretrained 臂）** | **1577.9** | 458.8 |
| — | ★ **主模型 MaskCO（I3 决策感知臂）** | — | **456.3** |
| 1 | rule: value/energy λ=2 | 1403.3 | 180.6 |
| 2 | rule: myopic | 1305.2 | 30.5 |
| 3 | OR-Tools-RH accept | 1300.7 | 20.9 |
| 4 | SGBS accept † | 1266.5 | −52.8 |
| 5 | RouteFinder accept | 1266.2 | −26.1 |
| 6 | MVMoE accept | 1257.7 | −38.6 |
| 7 | POMO accept | 1238.8 | −68.1 |
| 8 | AttentionModel accept | 1236.2 | −72.9 |
| 9 | RRNCO-Ordering accept | 1235.4 | −73.2 |
| 10 | **CO-enriched-ML accept（纯学习排序）** | 1234.5 | −74.3 |
| 11 | **CO-enriched-ML accept（+官方 PC-HGS）** | 1230.5 | −80.5 |
| 12 | Sym-NCO accept † | 1229.3 | −76.1 |
| 13 | Omni-VRP accept | 1226.9 | −85.7 |
| 14 | **MAPT accept** | 1215.4 | −102.2 |
| 15 | Learn-Improvement-Heuristics accept † | 1215.0 | −335.1 |
| 16 | DeepACO accept | 1213.6 | −100.0 |
| 17 | CaDA accept | 1157.0 | −190.3 |
| 18 | PyVRP-RH accept | 909.2 | −569.8 |
| 19 | rule: value/energy λ=3 | 0.0（0 接单，判负留档） | −1943.1 |

> **† = 本行数字受本轮 CPU 争用影响（`超时/天 > 0`），不是方法能力差异**：
> `sgbs` **12.0 / 9.1**、`symnco` 0.08 / 0.23、`lih` 25.4 / **54.2**（最长单决策 **50.79 s**，
> 超 10 s 协议预算 **5.1×**）。**引用这些行必须同时报 `超时/天` 与 `最大单决策`**
> —— 详见 §12 R10 与 `读表注意与已知限定.md` §2.8（`lih` 主档因此偏离 v1 **+219%**）。
> 其余 **24 个正式臂 `超时/天 = 0.0`、`最大单决策 ≤ 6.76 s`**。
>
> ⚠️ **2026-10-14 独立复核补充（对 Ours 行同样适用，且更严重）**：上面 † 原先只打给 3 个
> **外部**方法，但实测**主模型自己那两行才是超时最重的** —— **A3 19.82 / I3 37.33 次超时每天**，
> 而对比方法臂的正式批次中**除上表 † 标出的 6 个之外全部 `超时/天 = 0.0`**。且 **A3 的配对身份未通过**
> （`_pairing_identity.problems = ['shared_source_sha256_mismatch']`，即它与 v2 基线**不是同一版
> 共享计算文件**；它自身的 `source_stable=True` 只证明"启动==结束"，**不能**替代跨运行同源）。
> ⇒ **引用主模型结果时必须与外部臂一样带超时与身份限定**。
> 依据：`CC_Compare/audit/BASELINE_QUALIFICATION_MATRIX_2026-10-14.json` 的 `role='ours'` 行。
>
> **本表为 v2 正式批次**（新 C1 基线 `a1_c1fresh_reveal_40`，`formal_adjudication=true` 落盘）。
> 新增两行进表：**CO-enriched-ML**（两种模式各占一行；`+PC-HGS` 未优于纯学习排序，主档与主信号档
> 分别差 5.2 / 4.0，**远小于 CI 半宽 ⇒ 应读作「无显著差异」**）与 **MAPT**（单档实测 ≈57 min、
> `超时/天 = 0.0` ⇒ **推翻了早先「预算不可行、无法产出有效臂」的结论**）。

**读点**（**口径已于 2026-10-14 按用户要求调整：正式论文的主结论一律以主档 `p_c=(5,10,15)` 为准；
`p_c=0` 是敏感性结果，不得因绝对效用更高而充当头条证据**）：

0. **三个层级必须分开讲，前两层合格不等于第三层成立**（这是本轮 P0 审计的核心结论）：
   | 层级 | 含义 | 本批状态 |
   |---|---|---|
   | ① 可执行 | 硬可行、超时回退、异常处理、模型真实参与 | 38/62 臂 `Q_core` 通过（v2 正式批次 **38/40**） |
   | ② 可正式裁决 | 源码/权重/数据/合同/主对照身份完整一致 | **A3 未通过**（跨源）；I3 通过；外部 38/40 通过（2 个例外是 λ=3 全拒单） |
   | ③ 已建立优势 | 按预声明判据，配对差均值与 CI 过门 | **未建立**：主档上 I3−cond_hist = **−144.7**，CI 不含 0 |
1. **主结论（主档）：主模型未超过 C1 强基线**。主档下 Ours(I3 maskco) = **456.317**，
   **同批** `cond_hist` = **600.997**，`maskco − cond_hist` = **−144.680，95% CI [−191.109, −100.125]**
   （40 日完整配对，CI 不含 0 ⇒ **有效的负向结果**；恒等式 456.317 − 600.997 = −144.680 成立）。
   ⚠️ **本行三个数必须同批同源**：解释 `−144.680` 的对照是 **600.997**（I3 gate 内 `cond_hist`），
   **不是** 631.4 —— 631.4 属于**另一套协议**（外部方法配对基线 `a1_c1fresh_reveal_40|cond_hist` 主档，
   `future_policy=reveal`），把它与本行的配对差并列即为**跨批次混用**
   （456.317 − 631.410 = **−175.093** ≠ −144.680）。已加机器检查见 §12 R14。
   同批内部对照同样未过门：
   `− warmstart = +9.348 [−33.396, +51.057]`、`− reconstruction_control = −17.044 [−47.283, +14.581]`
   —— **两条 CI 都跨 0**。⇒ 主模型相对强对照**尚未建立优势**。
   ⚠️ **不要把该负向结果归因于超时**：同一批里 `cond_hist` 自己**平均每天超时 42.70 次**
   （比 maskco 的 37.325 **更多**）却取得**更高**效用（**601.0 vs 456.3**）
   ⇒ **超时次数本身不足以解释效用差异**。「零超时」是更严格的工程指标，
   但**不能取代预声明主门**，也不能事后据此剔除样本。
2. **A3 的 `p_c=0` 1577.9 只能作诊断**：其配对身份未通过
   （`shared_source_sha256_mismatch`，见上表「配对来源身份」列 = 跨源），
   且**超时 19.8 次/天**。⇒ 降级为 `DIAGNOSTIC / CROSS-SOURCE`，
   **不得**用于跨基线的合格优势主张或同协议排名。它相对最好学习型外部方法的 +311.4 同样受此限制。
3. **「换更强的路由器」换不来决策收益**：全部 **16 个外部方法**（含 OR-Tools/PyVRP）
   都低于 C1 基线 **−403 ~ −795**（p_c=0 重配对差；其中最好的是 OR-Tools **−403.1**），
   差距主要由**接/拒决策**而非路径质量造成。
4. **零样本 CVRP-only 方法（CaDA/POMO/…）并不比有 TW 的方法差**：
   MVMoE（原生 VRPTW 权重）1257.7 ≈ POMO（CVRP 权重）1238.8，说明本任务上
   预训练任务的约束匹配度不是决定因素。
5. **规则臂 value/energy λ=2 是所有外部/规则方法中最好的**（1403.3），且能耗仅约 72% 预算
   —— 预算感知的接单门槛比任何 learned 排序都更值钱。
   ⚠️ **这条是最需要防守的审稿质疑**：它意味着"收益可能只是预算选择效应"。
6. **p_c=(5,10,15) 档全部塌陷到 ≤ 631**：拒绝 95 单/天 × 平均 p_c 使拒绝损失 ≈1070/天，
   各臂差异被压缩；因此**决策质量的主信号在 p_c=0 档**，主档回答的是「有合同损失时的净效用」。

---

## 3. 身份与可裁决性（重要）

| 项 | 状态 |
|---|---|
| 共享计算身份（`strict_online_env` / `coldchain_*` / `coldchain_evaluator_a1` / `run_exp_reserve`） | 全部 22 个正式 learned 臂与 `a1_c1rr2_reveal_40` **逐文件相同** |
| contract_hash | 现在重跑 = `e25a12104b…` = 臂记录值 → **SAME（已复验）** |
| data_sha256 / data_meta_sha256 | 现在重跑 = `8767a03d…` / `72a805fd…` = 臂记录值 → **SAME（已复验）** |
| 臂自报 `_pairing_identity` | `identity_verified=false`（原因 `baseline_identity_missing`：基线取自无身份块的旧 gate） |
| 本表配对 | 改用身份合格的 `a1_c1rr2_reveal_40` per-day 重算 |
| `formal_adjudication` | **仍为 false**：`run_identity.pairing_identity_check` 要求双侧 `source_stable=True` 且 `shared_source_sha256` 全等；`a1_c1rr2_reveal_40` 记录 `source_stable=False`（其运行期间 `run_a1_step2_gate.py` / `scenario_saa.py` 变化） |
| 09-30 之后漂移的文件 | `scenario_saa.py`（10-03）、`run_a1_external_accept.py`（09-30 12:46）、`rrnco_accept_replanner.py`（09-30 09:10）——**恰好是臂运行时加载的版本没有留存**（git 只有当前版本；臂只留下 hash） |
| OR-Tools / PyVRP 臂 | 无 `_identity` 块（09-29 旧驱动），身份不可核验 |

**结论**：本表数字是**同共享计算身份下的重配对结果**，可作为开发证据与论文对比表的骨架；
但**不等于已获正式裁决**。要取得正式裁决，必须用**同一版源码**重跑 C1 基线与全部臂
（见 §5）。

---

## 4. 主模型侧的独立发现

- I3 gate（`i3_online_recovery_1009_v1/online_20260926`）四臂 `source_stable=True`、
  身份块完整、`formal_adjudication=true`，但**超时严重**：
  cond_hist 42.7 次/天、maskco 37.3 次/天、reconstruction_control 44.2 次/天
  （对比：外部臂全部 0 次/天）。该 gate 的 `workers=4` + 高负载 → 超时污染，
  虽绝对效用与 A3 同档一致（cond_hist 601.0 vs 617.3，CI 重叠），
  但**主模型侧建议在低负载下重测**。
- A3 三个编码器臂（B=`a3_train200_pretrained_s42` / C=`a3_train200_partial_pretrained_s42` /
  D=`a3_train200_random_cvrp_s42`）在 p_c=0 上：1577.9 / 1162.0 / 1449.3，均低于同 gate 的
  cond_hist 1694.7（→ 与既有「条件场景生成器未过门」结论一致）；且 A3-C 超时 126.9 次/天，
  其数字不可用。
- **Ours 行的时效性已核（`scan_main_model_gates.py`，since 2026-10-09）**：主模型线在 10-09
  之后**没有**产生新的 40 天 `maskco` 臂（另外两个 40 天 gate 只有 `cond_hist`，无 maskco），
  因此总表 Ours 行仍取 `i3_online_recovery_1009_v1`（10-09）——是当前最新的正式主模型结果。
- **干净基线交叉核对**：`a1_s3_4_d3_on_40_1007_125522_att1`（10-09）的 cond_hist
  = **1643.25**（p_c=0）/ **504.54**（主档），**超时 0.0 次/天**；与 c1rr2 的 1703.74 / 631.41
  同量级。对照 `d3_off_..._att1` 同臂超时 169.4 次/天、p_c=0 只有 450.90 —— 再次印证
  **超时/负载会大幅压低基线**，故主模型侧结论必须与超时数同报。

---

## 5. 要「全部改好」还差什么（按成本排序）

| 项 | 内容 | 状态（2026-10-13 更新） | 说明 |
|---|---|---|---|
| F1 | **同版源码正式重跑**：C1 基线（reveal，3 臂）+ 全部外部方法 × 2 档 | ▶ **运行中** | 新基线 ✅ 已产出（9 组 40/40、身份合格）；四条腿运行中（3080 上 GPU 腿四臂 + cpuA/B/C），随后 CO-enriched → MAPT。唯一能拿到 `formal_adjudication=true` 的路径；**已在首个落盘臂 `attention_0` 上实测 `formal_adjudication=true`×3 档、`source_stable=true`** |
| F2 | 补 OR-Tools/PyVRP 的身份块 | ✅ **由 F1 覆盖**（cpuC 用新基线重跑 ortools+pyvrp） | 查实原 `a1_ext_a05e` 求解器臂的 `baseline_from` 是**无 identity 块**的旧基线 ⇒ 现有 OR-Tools/PyVRP 数字本就不是正式可裁决的，必须随 F1 重跑（见 §13.3 缺陷 B） |
| F3 | MAPT 适配 | ✅ **已完成** | 已按桥接语义自训（144 updates = 21% 论文预算，`mapt_a1.pt`）；实测官方参数空间内**可行档** `config=fast`（0.50 s/次 × 15 辆 = **7.6 s ≤ 10 s**）⇒ 40 天臂已人工放行、运行中。**属弱介入，须带限定解读**（见 §8） |
| F4 | CO-enriched-ML 适配 | ✅ **已完成** | accept/reject 改造 + 按 A-v1 特征/目标重训 + PC-HGS(C++20) 编译；2 模式（纯排序 / +PC-HGS）× 2 档运行中 |
| F5 | Learning-to-Delegate | ❌ **不可适配（机制）** | 回归目标为 HGS 子问题距离、规模 n≥500，与 A-v1 每决策 ≤90 池规模方向相反；资产可加载（3/3 strict，MAE 0.071–0.138）但语义不可迁移 |
| F6 | PIP-constraint | ❌ **不可适配（机制）** | 单车辆 TSPTW/TSPDL，其 env 里 `num_vehicles`/`vehicle_id`/`capacity` 出现 **0 次**；43/48 个 `.pt` 可加载但「单车辆→多车辆」接不上 |

---

## 6. 复验脚本

```bash
# 臂源码身份 vs 当前磁盘（判定漂移面）
python ../CC_Compare/tools/verify_arm_source_identity.py

# 合同/数据身份是否可从当前代码复现（应全部 SAME）
python ../CC_Compare/tools/verify_contract_identity.py

# 收口前排演：真实数据跑 collect -> render -> 4 道门禁，全程写临时位置、不碰交付物
pwsh -NoProfile -File CC_Compare/tools/_rehearsal_prefinalize.ps1

# 单点门禁（收口时由 _local_finalize.ps1 自动跑；也可手跑，均支持 --data 指向排练产物）
python CC_Compare/tools/check_table_consistency.py --md <MD> --xlsx <XLSX>
python CC_Compare/tools/check_arm_provenance.py [--data <dataset.json>]
python CC_Compare/tools/check_method_coverage.py [--data <dataset.json>]
python CC_Compare/tools/check_gpu_device.py [--root <results/a1_ext_formal2>]
python CC_Compare/tools/_doc_prose_check.py           # 散文个数/极值 vs 权威生成器（收口后必跑）
```

**为什么要「排演」而不是只在收口时跑一次**（2026-10-13 教训，见 §13.3 缺陷 J）：
「判据永远不可能通过」这类缺陷只有用**真实数据**完整跑一遍才会暴露；只在收口时跑第一次
等于拿交付物做实验。排演全程只写 `_rehearsal_dataset.json`（临时，脚本结束即删）与 `$env:TEMP`。

**排演实测（2026-10-13 20:17 首轮 / 20:39 扩展到全链）**：排演覆盖收口链的**全部**渲染与门禁步骤
（`collect → render_compare_table → render_compare_figure → render_v2_summary → 4 道门禁`）——
因为收口是链式执行，**任一步崩溃都会在门禁之前中断**、交付物里既没表也没图；
「只跑 table 就放心」是不成立的。

| 步骤 | 结果 | 说明 |
|---|---|---|
| 表一致性（MD vs Excel 逐列） | ✅ `不一致字段数: 0`（22 行 × 22 行） | 新增「参数量(M)」列在 80+ 臂规模下跨格式一致（此前只在 13/22 行合成数据上验过） |
| 配图 `render_compare_figure.py` | ✅ rc=0，**PNG 132,074 B 落盘** | 覆盖「新数据集形状把配图写崩」这一收口中断风险 |
| v2 摘要 `render_v2_summary.py` | ✅ rc=0（27 个 v2 臂） | §4 自动判定表可用 |
| 方法覆盖 | ⚠️ 4 条 = **真实待办** | 缺 `coenrich_*`/`mapt` 结果行（对应腿尚未产出）；`未适配 2 项`（Learning-to-Delegate / PIP-constraint，均带具体原因）；**E 项：正式臂 21 个 → 缺 identity_verified 0 个，纯规则臂豁免 6 个** ⇒ 缺陷 J 的修复在真实数据上成立 |
| 臂来源批次 | ⚠️ 预期 | 如实报出 7 档仍回落 v1（`lih`/`sgbs`/`symnco` + `rrnco_51015`），正是该门禁要抓的；收口后应转 OK |
| GPU 设备口径 | ⚠️ 仅缺 `rrnco_51015` | 其余 7 档均为 cuda（`provider_device` 或 rrnco 的 `rrnco_device`） |

**最新一次排演（2026-10-13 21:10，**85 臂**，GPU 腿已全部回传后）**：`table_consistency` **0 不一致**、
`figure`/`v2_summary` **rc=0**（PNG 132,077 B、摘要 30 个 v2 臂）、
**`gpu_device` → `PASS: 8 个臂齐全，且各自的实际 device 键都是 cuda`（验收项 10 由 7/8 补齐为 8/8）**、
`method_coverage` 的 E 项 **`正式臂(需配对身份) 24 个 → 缺 identity_verified 0 个；纯规则臂豁免 6 个`**、
未适配仍为 **2 项**；`arm_provenance` 只剩 **4 档**落在 v1
（`lih_51015` / `sgbs`×2 / `symnco_51015`，正是在跑的两条腿）—— 即**除在跑的臂外，全链无红灯**。
排演结束时本地与远端临时文件**均已清理**（残留检查为 False）。

**已完成的独立复算核对（2026-10-13）**：`dataset.json` 里每个臂的效用是由本工具
从**逐日 `per_day` 行重新计算**的（`_summ_rows`，与臂自报值相互独立）。逐臂比对
「重算值 vs 臂落盘的自报 `utility_all_days`」：**55/55 臂 mean 与 CI 上下界全部吻合
（差 < 1e-6）**；基线臂重算值与 `a1_c1rr2_reveal_40` 记录一致
（cond_hist p_c=0 = 1703.7439 / 主档 631.4096）。即总表数字不是抄摘要，而是从逐日记录复算所得。

---

## 7. 正式批次 v2（2026-10-13 用户批准，运行中）

用户裁定「A. 现在就做全量正式重跑」。批次脚本与调度：

| 脚本 | 作用 |
|---|---|
| `CC_Compare/_formal_v2_start.sh` | 起 tmux `xf2_launch`，启动整批 |
| `CC_Compare/_formal_v2_launch.sh` | 先跑新 C1 基线（reveal，3 臂 × 3 档，workers=9）→ 再起四条腿 |
| `CC_Compare/_formal_v2_gpu.sh` | GPU1：RouteFinder→CaDA→MVMoE（顺序）；GPU0：RRNCO |
| `CC_Compare/_formal_v2_cpuA.sh` | attention / deepaco / omnivrp / pomo / symnco / sgbs（workers=6，顺序） |
| `CC_Compare/_formal_v2_cpuB.sh` | lih（单跑，最慢） |
| `CC_Compare/_formal_v2_cpuC.sh` | OR-Tools / PyVRP（3 档）+ 规则臂 myopic / valuee-λ2 / λ3 |
| `CC_Compare/_formal_v2_finalize.sh` | 等四条腿 ALLDONE → 源码冻结对账 → 重采集数据集 → 打印各臂 identity/裁决 |

- 新基线输出：`results/a1_c1fresh_reveal_40/gate.json`（身份块 + `shared_source_sha256` + `source_stable`）。
- 各臂输出：`results/a1_ext_formal2/<method>_{0,51015}`、`.../ortools|pyvrp/gate.json`、`.../rule_*`。
- 启动前源码冻结基线：`freeze_v2_start.json`（29 个文件）；收口时与 `freeze_v2_end.json` 逐文件对账，
  任何变化都会打印出来——**配对要求双侧 `source_stable=True` 且 `shared_source_sha256` 全等**，
  因此批次期间不得修改 `scenario_saa.py` / `strict_online_env.py` / `coldchain_*.py` /
  `run_exp_*.py` / `coldchain_evaluator_a1.py` 等共享计算文件与各 provider。
- **负载口径**：批次运行期间机器 load ≈30–38（另一用户 20 进程占 ~22 核、GPU 0/1 各 ~93%），
  实测 C1 基线每决策日 **9–14 分钟**（原探针 145–166 s/日）→ 基线（9 个 arm×p_c 任务并行）约
  **6–9 h**，整批预计 **20–35 h**。各臂 `mean_timeouts` 必须与结果一起报告；超时升高属资源混淆，
  不得当作方法效果。
  > **负载事故（如实登记，2026-10-13 13:41–13:44 UTC）**：对比方法线自己的 P0 参与度补证脚本
  > 误在基线运行期跑了 MAPT provider，占 **≈24 核**、把 load 从 32 推到 **61**（已即时叫停，
  > 90 s 内回落）。**这段窗口可能给基线少数决策带来超时**；基线 gate 落盘后必须核对
  > `mean_timeouts`——若明显高于 c1rr2 的 3.5 次/天，则该批需在低负载下重跑 C1 基线。
  > 事后诊断（`_progress_dump.sh`）：基线运行到第 14–17 天时各 combo 的**当日耗时**
  > min/avg/max = 378–479 / **538–668** / 942–980 秒，**未出现异常日**（事故窗口只占一个
  > ~9 分钟日的 1/3）。整体比 c1rr2 的 ≈189 秒/日**慢约 3 倍**，属负载差异而非单点尖峰。
  > **污染判据（阈值取自 c1rr2 实测，由 `compare_two_gates.py` 自检输出）**：
  > cond_hist **3.52**（p_c=0）/ 3.23（主档）/ 2.80；uncond_hist 0.82 / 2.02 / 1.75；
  > explicit_feat 0.45 / 0.50 / 0.47 次/天。**若新基线任一臂超过该档位 2 倍以上，判为受污染、
  > 需在低负载下重跑 C1 基线**；判定由守候任务 `tools/_baseline_watch.ps1` 在基线落盘瞬间自动执行
  > （含逐日 `max|Δday|` 对账）。
- **追加腿**：`CC_Compare/_formal_v2_coenrich.sh`（CO-enriched-ML，2 模式 × 2 档）
  ——必须等四条腿 ALLDONE 后再点火（改 `ordering_providers.py` 会让在跑臂 `source_stable=False`；
  注意该文件**不在** `SHARED_SOURCE_FILES` 里，所以事后接线不影响与基线的配对）。
  接线 = 运行 `CC_Compare/tools/wire_extra_providers.py`（幂等，自动备份 + py_compile 自检）。
  **两模式实测（1 天探针，workers=6）**：纯排序 `util=1191.3`、+PC-HGS `util=1175.3`，
  **两者 `timeouts=0.0`、`hfr=1.0`** ⇒ 均在 10 s 协议内可行；但 +PC-HGS 的缓存键含
  per-vehicle 几何（`_feature_hash` 含 coords/dist_mat）⇒ **每车都可能 miss**，实测
  **≈13.6 min/天 ⇒ 约 9 h/臂**（纯排序仅 1–2 min/天）。4 臂合计约 18–20 h，
  故收口编排对 CO-enriched 的等待上限设为 40 h。
- **一键收口（已挂起，幂等）**：`CC_Compare/_formal_v2_after.sh`（tmux `xf2_after`）
  自动「等四条腿 → 接线 → 跑 CO-enriched → 若 MAPT ckpt 就位则跑 MAPT → 重采集 + 打印各臂
  `identity_verified/formal_adjudication`」。MAPT ckpt 迟到时单独执行：
  `MAPT_CKPT=<路径> bash CC_Compare/_formal_v2_mapt.sh`。
- **预算等价性已核**（`check_budget_equiv.py`）：新基线口径重算
  `B = 710.6548152249378`、`cooling_share = 3.7208946347769882`，与 `a1_c1rr2_reveal_40`
  **逐位相同** → 追加臂的决策与基线文件无关，故事后用新基线重配对合法
  （重配对只需共享源码哈希一致，而两者都是当前代码）。
- 采集器已支持新批次（`a1_ext_formal2` 优先于旧批次），收口后本地执行：
  `python CC_Compare/tools/render_compare_table.py --data CC_Compare/results/a1_compare_table/dataset.json --outdir CC_Compare/results/a1_compare_table`。

### 7.1 配对可行性已用小样**实测通过**（2026-10-13）

正式批次要在 ~30 h 后才出结果，为免整批白跑，先用 2 天小样端到端验证「新基线 + 当前代码」
能否真正通过 `pairing_identity_check`（脚本 `CC_Compare/_formal_v2_smoke.sh`，输出
`results/a1_c1smoke2_reveal/` 与 `results/a1_ext_formal2_smoke/`）：

| 检查项 | 结果 |
|---|---|
| 基线 identity 块 | 12 个键齐全，`source_stable=True`，`shared_source_sha256` 7 文件 |
| 臂 identity | `source_stable=True`，`shared_source_sha256` 7 文件 |
| `shared_source_sha256` / `contract_sha256` / `data_sha256` / `data_meta_sha256` / `budget_B` / `cooling_share` / `shared_config` | **7/7 match=True** |
| `_pairing_identity` | **`identity_verified=True`，`problems=[]`** |
| 配对差 | `adjudicable=True`、**`formal_adjudication=True`**、`identity_verified=True` |

→ 结论：v2 批次的设计成立，正式臂将拿到**身份可裁决**的配对（与旧批次
`identity_verified=false` 形成对照）。

### 7.2 独立复核记录（助手自己重跑，不采信子代理自述）

| 复核对象 | 方法 | 结果 |
|---|---|---|
| 55 个臂的效用数字 | 从逐日 `per_day` 重算，与臂自报 `utility_all_days` 比对 | **55/55 match**（mean+CI 差 <1e-6） |
| 合同/数据身份 | `verify_contract_identity.py` 当前代码重算 | `contract`/`data`/`meta` 与旧臂记录**逐位相同** |
| 臂源码漂移面 | `verify_arm_source_identity.py` | 仅 `scenario_saa.py`/`run_a1_external_accept.py`/`rrnco_accept_replanner.py` 漂移 |
| 配对可行性 | 2 天小样端到端（§7.1） | `identity_verified=True`、`formal_adjudication=True` |
| 新基线预算等价 | `check_budget_equiv.py` | `B`/`cooling_share` 与旧基线**逐位相同** |
| CO-enriched-ML 适配 | 重跑 `smoke_coenrich.py` | **SMOKE_OK**（覆盖/确定性/偏好改变 58-60/60/未来信息不变性 hash_eq=True/PC-HGS 合法） |
| Learning-to-Delegate 资产 | 重跑 `probe_l2d_weights.py` | **3/3 权重 strict 加载成功**（miss=0/unexp=0），MAE 0.071–0.138（`prev_dists` 0.183、常数 2.918），ρ 0.991–0.999，确定性 max\|Δ\|=0 → 资产可用，**不可比原因是机制+规模**（回归头=估 HGS 子问题里程、官方 n≥500 vs A-v1 池 ≤90） |
| PIP-constraint 判死 | 重跑 `probe_pip_assets.py` | `TSPTWEnv.py`/`TSPDLEnv.py` 中 `num_vehicles`/`vehicle_id`/`capacity` 出现次数**均为 0** → 单车辆 TSP 变体；`AM+PIP` 三个 ckpt 确为损坏（zip central directory 错误） |
| MD 与 Excel 是否数字一致 | `check_table_consistency.py`（两套渲染路径交叉比对） | **22 行 / 0 不一致字段**（MD 显示 1 位小数，容差 0.06） |
| 四条腿能否正常起飞 | `_preflight_v2.sh`（语法 + 11 个 ckpt 路径 + provider 注册表） | 全部 OK（脚本 8/8 语法通过、ckpt 11/11 存在、provider 10/10 已注册） |
| 批次中途是否有源码漂移 | `record_freeze_v2.py` 生成 `freeze_v2_mid.json` 与启动快照对账（批次进行中实测） | **29 个冻结文件中仅 1 个变化**：`scripts/evaluation/ordering_providers.py` —— 即本轮**有意接线**（加 `coenrich`/`mapt`）。该文件不在 `SHARED_SOURCE_FILES`、也不在基线的 `SOURCE_FILES_STEP2` 里，且接线在四条腿起飞前完成 ⇒ **配对与基线 `source_stable` 均不受影响**；其余 28 个文件（7 个共享计算文件 + 两个驱动 + RRNCO 适配/后端 + 11 个 provider + 判死相关）**零漂移** |
| README 数字 vs 数据 | `check_readme_numbers.py` | **34/34 一致**（cond_hist 1703.7/631.4、RouteFinder 1266.2/−26.1、OR-Tools 1300.7/20.9、PyVRP 909.2、λ=2 1403.3/180.6、CaDA 1157.0、MVMoE 1257.7、Ours 1577.9，**并含 v2 新增的 SGBS 1266.5/−52.8、Sym-NCO 1229.3/−76.1、DeepACO 1213.6/−100.0、LIH 1215.0/−335.1**）。⚠️ 该门禁在**文档同步前**会报 FAIL：README §2 表体仍是 v1 数字（SGBS/DeepACO/LIH/Sym-NCO 四行）⇒ **收口后必须把 `README_SECTION2_paste.md` 的表体贴进 §2 才会转绿** |

---

## 8. 模型参与证据与使用限制（重要，读表前必读）

台账：`模型参与证据.md`（逐方法：权重 sha256 / 是否真实加载前向 / 偏好改变排序的证据 /
未来扰动不变性 / 确定性 / 单决策耗时 / **证据强度**）。

**证据强度分布（2026-10-13 补证后实测）：强 10 / 中 2 / 弱 0。**
（11 个 learned provider 中 **9 强 / 2 中**：中档是 DeepACO 与 LIH，二者**清缓存两次调用不一致**
⇒ 确定性不成立；另有 CO-enriched-ML 维持「强」。**MAPT 的权重替换 C2 已于 2026-10-13 补齐**，
`order_changed_by_weights=True`，故由「中」升为「强」——详见 `模型参与证据.md` §1/§2.1。）


- **强（10）**：RouteFinder、CaDA、SGBS、POMO、Sym-NCO、AttentionModel、Omni-VRP、MVMoE
  （以上 8 个由补证升为强：**权重替换后排序确实改变** + 未来扰动不变 + **清缓存后**确定性）、
  RRNCO（`decision_changed_vs_edd=true`）、CO-enriched-ML（与 EDD 不同 58/60、Spearman 0.419，
  且有官方 vs 重训权重的对照）。
- **中（3）**：
  - **LIH / DeepACO —— 实测本身不确定**：清缓存后两次 `order()` **结果不一致**
    （与其 RESULTS.md 自述一致：LIH 每步 multinomial 采样、DeepACO 的 ACO 未固定采样 seed；
    其未来扰动 `ordering_same=False` 与 B3 同源，`subproblem_hash_same=True` 排除泄漏）。
    ⇒ **这两臂的决策跨次不可复现**，论文中不可与确定性方法并列宣称「同 seed 逐位复现」。
  - **MAPT —— 已适配；40 天臂已获人工放行并进入总表**（`config=fast` 实测 0.50 s/次 × 15 辆 =
    **7.6 s ≤ 10 s** 预算），但它是**弱介入**（只决定排序前 ~3 位，其余由 EDD 补全），
    **不得**当作「MAPT 方法能力」的证据，见 §8。
- **没有一行是「权重不敏感」**：9 个 torch provider 全部 `order_changed_by_weights=True`，
  与随机权重的 first-12 重合仅 1–9/12（Spearman −0.44…+0.68）。⇒ 原先「可能模型没起作用」
  的担忧**已被实测排除**。

⇒ **本表的新读法**：这些行不仅证明「provider 被真实调用、输出合法、40 天跑满」，
**还证明了权重对排序有因果作用**（LIH/DeepACO 需加不确定性的限定）。

**MAPT：已找到「官方参数空间内最重可行档」，其 40 天臂已获批准并将进入总表（**弱介入**，须带限定解读）**

- 原阻塞在**单决策耗时**：`rrnco_guided_adapter` 的不变量是「**每辆 replan 车调一次 provider**」
  （无跨车缓存）× replan 车 ≤15。
- 可行档定为 provider 默认 `config='fast'`：`n_frame_eval=8, n_phantom_vehicles=0,
  use_tsp=1, deterministic=True, cpu_threads=4`。1003 实测（`verify_provider_defaults_1003.py`）：

  | pool | 单次 `order()` | ×15 车 = 单决策 | ≤8 s? |
  |---:|---:|---:|---|
  | 30 | 0.19–0.25 s | **3.8 s** | ✅ |
  | 60 | 0.30 s | **4.6 s** | ✅ |
  | 90 | **0.50 s** | **7.6 s** | ✅ |

- **三条反直觉但可复现的实测**：(a) **CPU 4 线程比 cuda 快**（pool=90：0.50 vs 0.91 s，
  子问题太小、GPU kernel 启动主导）；(b) **torch 默认 48 线程是陷阱**（同配置慢 2.8×，
  故 CPU 路径显式 `set_num_threads(4)`）；(c) `n_phantom_vehicles=0` 是最强杠杆
  （帧内车辆解码 15→1，5.99 s → 0.45 s，≈13×）。
- **代价（必须与结果同报）**：模型只决定**前 3 位（pool=90）/ 前 4 位（pool=30）**，尾部走 EDD
  ⇒ 该臂实质是「**EDD + 模型决定前 ~3 位**」，属**弱介入**，**不是**「MAPT 方法能力」的证据。
  研究档 `MAPT_PROVIDER_CONFIG=full`（≈90 s/决策，超时限）只用于离线取证。
- **同一 40 天上的配对对照**（`greedy_dev.py`，纯 CPU 无模型）：弱贪心 reward
  **951.13±118.78**、served **26.91%**；MAPT 最终 ckpt(144) reward **871.04**（**低于弱贪心 8.4%**）、
  served **36.44%**。⇒ **按 reward 未超过弱贪心；按服务率明显超过**。原因已查明：模型服务的
  订单**价值中性**（均值 20.0 = 全体均值 20.03），弱贪心服务的订单均值 **27.7/单**，两侧燃油接近
  （≈545 vs ≈538）⇒ **多服务 ≠ 多赚钱**。
- **只能主张「参与」，不能主张「能力」**：同输入只换权重 → 排序不同（first-12 重合 5–8/12、
  Spearman 0.06–0.41），但**随机初始化统计同量级**（8–11/30）；update-7 在两种动作规则下都是
  **纯 EDD**（ranked=0），update-32 才开始接单。
- **训练 vs dev 不可混读**：训练曲线 update 15 后平台化（3050–3200）= **训练信号已收敛**；
  dev 40 天曲线（官方 runner 评测，9 个点）−118→1144→696→…→871 **非单调、未见提升**。
- `mapt_a1.pt` = 官方 runner `run4/models/transformer_best.pt`（**update 32**），
  sha256 `ebdaff0a69e08b63…`（1003 已核对）；另存 `mapt_a1_final.pt`（update 143）备切换。
- **未做到（逐条见 `MAPT_code/a1_accept_v1/RESULTS.md` §6）**：未接 TW/冷链；未训 p_c≠0 档；
  provider 上线时 `value` 特征被中和、车辆维/节点空间分布外；**独立 dev 重放失败**
  （同 40 天同 env，官方 runner 24 s，本适配层独立重放 >25 min 未完成一回合，原因未定位，
  故 dev 数字全部取自官方 runner 自身评测）。

**两条必须进论文脚注的发现**：

1. **POMO / SGBS / Omni-VRP 用的是同一份权重文件**（sha256 全等 `d7a08f6115623eba`，
   16,358,236 B）→ 三者效用差（POMO **1238.8** / **SGBS 1266.5 †** / Omni-VRP **1226.9** @ p_c=0，
   已更新为 v2 值；† 见表注）**只能归因于解码/搜索过程**，
   不能读成「三个独立预训练模型」。⚠️ 但三者**差值的大小**不宜当作精确的「解码/搜索增益」：
   **SGBS 带 12.0 次超时/天**而 POMO / Omni-VRP 为 **0** ⇒ 它这一格本身受争用影响；
   **「同一份权重」这条结论不受影响**。MVMoE 亦为 POMO 架构 + VRPTW 权重。
2. **确定性证据存在缓存陷阱**：8 个 provider 的内容寻址缓存使「同池两次 order() 相等」
   可能只是缓存命中（MVMoE 已记录该陷阱）⇒ 未清缓存的相等**不算**确定性证据。

**已排除的一处数据质量风险**：台账提到 MVMoE 曾因 provider 的 id/index 覆盖 bug 导致
7761/7761 走 EDD fallback、臂塌缩为 1 单/天 —— 该坏批次是 `mvmoe_*_providerbug_v1`，
**被本采集器按目录名显式排除**；当前 provider 的 EDD 兜底有 `_EDD_FALLBACK_POOL = 120`
守卫（A-v1 池上限 ≈90，结构上不触发），故正式臂不受影响。

---

## 9. 主模型出新臂后，如何并入本总表（给主模型线用）

总表的 **Ours 行**取自主模型侧自己的 gate（不是本线跑的）。主模型出新版本时，按下面三步刷新：

1. **把新 gate 加进采集范围**：编辑 `CC_Compare/tools/collect_compare_table.py` 的
   `MAIN_MODEL_PATTERNS`，加一行 glob（例如 `results/<新目录>/online_*/gate.json`）。
   采集器会自动识别两种 schema（`results[year][p_c]` 与 `gate[p_c]`），并同时读取
   **block 顶层**与**嵌套 `main_comparison`** 里的 `<arm>_minus_cond_hist` 配对差。
2. **重采集 + 重渲染**（本地或服务器）：

   ```bash
   # 服务器（C-VRP_Cold-chainVehicleRoutingProblem 下）
   /home/hzeng/envs/a1_ext/bin/python ../CC_Compare/tools/collect_compare_table.py \
       --json ../CC_Compare/results/a1_compare_table/dataset.json
   # 本地
   python CC_Compare/tools/render_compare_table.py \
       --data CC_Compare/results/a1_compare_table/dataset.json \
       --outdir CC_Compare/results/a1_compare_table
   python CC_Compare/tools/render_compare_figure.py --data <...>/dataset.json --outdir <...>
   python CC_Compare/tools/check_table_consistency.py --md <...>/A-v1对比方法总表.md \
       --xlsx <...>/A-v1对比方法总表.xlsx
   ```
3. **看结果是否可用**：`canonical_ours()` 会把主模型侧**最终结果**渲染成交付主表里的
   ★ Ours 行（当前是 A3 / I3 两行；档位并成列）。**必须同时看 `超时/天` 列** ——
   已知资源竞争会把基线压低（例：`d3_off` 同臂 169.4 次超时/天 vs `d3_on` 0 次，
   p_c=0 效用 450.9 vs 1643.3）。超时显著高于外部臂（外部臂正式批次为 0）时，
   该 gate 的数字应标注为受污染、只作诊断。

**控制点（2026-10-14 重构后只剩两个，别照旧文档改第三个）**：

| 常量（`CC_Compare/tools/render_compare_table.py`） | 控制哪个展示位 | 适用场景 |
|---|---|---|
| `MAIN_MODEL_ARMS` | **§0 Ours 摘要**、**§3 主模型 / 内部臂**、Excel「主模型与内部臂」sheet | 主模型侧的新臂 / 对照臂 / 内部基线（含 `cond_hist` 等） |
| `OURS_TABLE_SPECS` | **交付主表（§1 / Excel 性能总表）里出现的 1–2 行 Ours** | 只放代表本工作的最终配置（当前 A3 + I3） |

**旧的第三个开关 `OURS_BLOCK_ARMS` 已删除**，连带废弃 `build_ours_rows()`（改成显式抛错）。
原因：它会把**每一条** `(gate, 档位)` 记录铺成一行 —— 3 配置 × 3 次重复运行 × 档位 = 27 行、
再加内部三臂 = **30 行堆在表尾**。用户 2026-10-14 反馈的三点（Ours 不该单独成块、
主模型该只有一个最终结果、空格太多）都由此而来。现在主表里 Ours 与基线**同表**，
其余记录留在「主模型与内部臂」sheet 并逐行标注 `★ 是（主表 Ours 行）` /
`否（消融 / 重复运行 / 内部臂）`。
（另注：两处都会受**开发集判定**影响 —— 见 §13.3 缺陷 K：判定已改为读 gate 里记录的真实
`seeds.dev`/`seeds.gate`，不再靠目录名。）

**未登记的臂不会再静默消失**：`unlisted_main_model_arms()` 会把「开发集里有、但没进 Ours/内部臂节」
的臂连同 `utility` 与 `超时/天` 渲染成 **§0 顶部告警表**（超时>0 标「超时污染，只能作诊断」），
并给出「若是新臂请登记」的指引。当前已知的常驻条目是 `a1_consensus`（42–46 次超时/天，
有意排除 —— 按 §9 的规矩它只能作诊断，不能进 Ours 行）。

---

## 10. 结论与论文表述建议（哪些能主张、哪些不能）

> 以下基于 v1 数字（v2 正式批次落盘后数值可能小幅变动，**结论方向预计不变**；
> 变动后本节的数字需同步更新）。

**可以主张**

1. **在统一 A-v1 接单协议下（同 40 天开发集 seed 20260926、同 C0 硬认证、同 10s 决策时限、
   同容量/车辆/温区/时间窗），主模型 MaskCO 的效用高于全部 16 个外部方法与 3 条规则对照**：
   p_c=0 档 **1577.9** vs 最好**学习型**外部 **1266.5**（`sgbs`；与 `RouteFinder` 1266.2
   **统计并列**，差 0.3 ≪ CI 半宽 ≈28，且 `sgbs` 带 12.0 次超时/天 ⇒ **不构成优势**）
   （**+311.4**）； 主档 **456.3**
   vs 最好外部 RouteFinder **−26.1**。且 16 个外部方法**无一**超过 C1 基线。
2. **结论对「挑哪个零样本模型」不敏感**：9 个零样本 CVRP 方法在 p_c=0 档集中在
   1213–1266（极差 ≤53），远小于它们与主模型的差距（≥312）→ 不是「选对模型」带来的差异。
3. **差距来自接/拒决策，不是路径质量**：外部方法 `reject_reasons` 以 `certify_budget` 为主
   （约 5000–5800 次/40 天），说明它们把预算花在了低价值订单上；这正是「路由强度换不来
   决策收益」的直接证据。
4. **规则基线最强的是预算感知门槛**：`value/energy λ=2` = **1403.3**（p_c=0）/ 180.6（主档），
   高于所有 learned 外部方法 → 论文必须把它列为强基线之一，否则会**高估** learned 方法的
   相对价值。

**不能主张（或需加限定）**

5. **不能说「learned 表征有增量」——但"模型是否真的在起作用"这一点已实测通过**：
   9 个 torch provider 权重替换后排序确实改变（§8），所以可以主张「这些臂的差异**确实来自
   模型的输出**」；但**没有做「去掉 learned 排序、换 EDD/随机排序」的同臂消融**，因此
   不能把效用差归因于「表征质量」而非「解码/搜索过程」。
6. **不能把 POMO / SGBS / Omni-VRP 当作三个独立预训练模型**（三者权重文件 sha256 相同）。
7. **不能把 C1（cond_hist）与外部方法混为一谈**：C1 是**使用未来信息**的内部场景采样基线
   （SAA 采样未来订单），主模型主档 **−144.7 [−191.1, −100.1]** 输给它。论文应把 C1 单列为
   「信息上界型基线」，并明确：**主模型赢的是"可部署/零未来信息"这一类对照**。
8. **LIH / DeepACO 两臂不可复现**（清缓存后两次调用不一致）→ 引用其数字时不得与确定性方法
   并列宣称「同 seed 逐位复现」。
9. **MAPT 作为可裁决行，但必须带「弱介入」限定**（**2026-10-13 更正**：旧文写「不能作为可裁决行、
   只能作机制参考」，那是按单次 `order()` ≈40.6 s 的**旧估计**；实测 `config=fast` 下
   0.50 s/次 × 15 辆 = **7.6 s ≤ 10 s** 预算，故 40 天臂已获人工放行并进入总表）。
   限定项：只决定排序**前 ~3 位**（其余尾部 EDD 补全）、`n_phantom_vehicles=0` 与训练分布（15 辆）
   不一致、桥接不感知 TW/冷链预算/品质（由外层 certify + C0 evaluator 强制）、value 特征上线被中和
   （常数 20）⇒ **不得**表述为「MAPT 方法能力」。研究档 `MAPT_PROVIDER_CONFIG=full`（≈90 s/决策）
   超时限，仅用于离线取证（§8）。

**必须与数字同时报告的量**

8. `mean_timeouts`（本批外部臂为 0；主模型 I3 gate 曾出现 37–43 次/天，属资源混淆）、
   `hard_feasible_rate`（正式臂全 1.0）、`fallback%`（全 0）、单决策最大耗时。
9. 未适配方法与理由（**现仅 2 个**）：Learning-to-Delegate（机制+规模不匹配：回归目标为 HGS 子问题
   距离、n≥500 vs A-v1 ≤90）、PIP-constraint（单车辆，无容量/多车字段）。
   **MAPT 与 CO-enriched-ML 已完成适配，不再属于「未适配」**：MAPT 已自训（21% 论文预算，
   训练曲线收敛但 dev 40 天曲线非单调）并作为**弱介入行**进入总表；CO-enriched-ML 已重训 + 接线
   （2 模式 × 2 档）。

---

## 11. v2 批次最终验收清单（每次重跑后逐条核对）

数据落盘后按顺序核对；**任一条不过，不得直接把数字写进论文**。

| # | 检查 | 通过标准 | 工具 |
|---|---|---|---|
| 1 | 四条腿是否跑完 | `gpu/cpuA/cpuB/cpuC.done` 均含 `ALLDONE`，无 `DONE_* <非零>` | `_batch_status.sh` |
| 2 | 天数完整性 | 每臂 `n_days=40`、`day_ids_complete=true`、`missing_day_ids=[]` | `dataset.json` / collector |
| 3 | 可裁决性 | 每臂 `adjudicable=true`（缺日不静默过滤） | 同上 |
| 4 | 身份与配对 | 臂 `source_stable=true`；7 个配对字段与基线全等；落盘 `formal_adjudication=true` | `_formal_v2_after.sh` 输出 |
| 5 | 源码冻结未漂移 | `freeze_v2_start.json` vs `freeze_v2_end.json` 差异文件数 = **0**，**唯一允许的例外是 `ordering_providers.py`**（追加 provider 的有意接线；它**不在** `SHARED_SOURCE_FILES` 也不在基线的 `SOURCE_FILES_STEP2` 封存清单里，且在四腿起飞前完成 ⇒ 配对与基线 `source_stable` 均不受影响） | `pwsh-7` / `record_freeze_v2.py` |
| 6 | 负载污染 | 各臂 `mean_timeouts` 不超 c1rr2 同档位的 **2 倍**（阈值见 §7 负载口径）；超了就在低负载重跑基线 | `_baseline_watch.ps1` |
| 7 | 同协议复现 | 新基线 vs c1rr2 逐日 `max\|Δday\|` 量级合理（无系统性偏移） | `compare_two_gates.py` |
| 8 | 可行性守恒 | 每臂 `hard_feasible_rate=1.0`、`fail_counts={}`、fallback 0% | `dataset.json` |
| 9 | 数字一致性 | MD 与 Excel 逐字段一致；55+ 臂重算值与臂自报值一致；**README 正文引用的头条数字与 `dataset.json` 一致** | `check_table_consistency.py`、`check_readme_numbers.py` |
| 10 | 新臂并入 | CO-enriched 两模式（纯排序/+PC-HGS）与 **MAPT 两档**、其余方法同表（19 方法行 + ★Ours）；**MAPT 行必须带「弱介入」限定注**（只定排序前 ~3 位） | `render_compare_table.py` |
| 11 | 文档同步 | §2 主结果、§8 证据强度、§10 论文表述随新数字更新；工作包 §5/§7 同步 | 人工 |
| 12 | 未适配记录 | **仅** Learning-to-Delegate（机制+规模）与 PIP-constraint（单车辆）两项，其**具体机制原因**在表内**直写**可追溯（Excel sheet「未适配方法」= 方法 / 不可适配的具体原因 / 依据文档 三列；MD §5 同）；**MAPT 与 CO-enriched-ML 已适配，不得出现在此清单** | `render_compare_table.py` §5 + **`check_method_coverage.py`（机器判）** |

### 11.1 机器可判部分 = 收口末尾的「6 道门禁汇总」

第 2–5、8–10、12 项中可自动化的部分已收敛为 6 道门禁，由 `_local_finalize.ps1` 依次执行
**并汇总退出码**（末尾打印 `GATES_ALL_PASS` 或 `GATES_FAIL: <名字>`；见 §13.3 缺陷 L 同轮新增）：

| 门禁 | 覆盖的验收项 | 命令 |
|---|---|---|
| `table_consistency` | 9（MD↔Excel 逐字段） | `check_table_consistency.py --md … --xlsx …` |
| `readme_numbers` | 9（正文头条数字） | `check_readme_numbers.py` |
| `arm_provenance` | 10（每档确实命中 `a1_ext_formal2`，防静默回落 v1） | `check_arm_provenance.py [--data …]` |
| `gpu_device` | 10（8 个 GPU 臂按**实际启动键**记为 cuda） | `check_gpu_device.py [--root …]` |
| `method_coverage` | 12 + 10（无静默丢弃 / 结果行∩未适配=∅ / 19 key 齐 / **E: 非规则正式臂必须 `identity_verified`** / **F: 13 个学习型方法必须有 `participation/*.json`，且纯 NumPy 实现必须另有专用 C2** / 依据文档在位） | `check_method_coverage.py [--data …]` |
| `v1_v2_agreement` | 4（`rc=0`+`timeouts=0`+身份合格 **仍可能是空臂**，缺陷 D 的教训） | `check_v1_v2_agreement.py`（在 1003 跑） |

> ⚠️ **远端那道门禁的退出码曾经「永远是 0」（2026-10-13 修复，否则汇总会给假绿灯）**：
> `tools/_srv.py` 原先**从不传递远端退出码**（CLI 里没有 `sys.exit`）⇒
> `_local_finalize.ps1` 里 `$GATES['v1_v2_agreement'] = $LASTEXITCODE` 拿到的是
> `_srv.py` 自己的 0，**即使 1003 上 `check_v1_v2_agreement.py` FAIL，汇总也会写 PASS**。
> 已新增 `run_rc()`（返回 `(out, rc)`，用 `recv_exit_status()`）并让 CLI 以远端 rc 退出；
> **`run()` 的返回类型保持字符串不变**（它被 ~30 个工具 `from _srv import run` 依赖）。
> 实测：远端 `exit 0` → rc 0；远端 `exit 5` → rc 5；`run()` 仍返回 `str` 且含输出。
> 另注：`_rehearsal_prefinalize.ps1` 记录的都是**本地** python 的 rc，不受此影响。

第 5 项的冻结对账已改为按**键的并集**分类（`changed` / `added=只在 end` / `removed=只在 start`）：
只有 `changed` 里不在白名单的才算意外漂移；`added/removed` 会**显式列出**而不静默（README §12 R9）。
第 11 项（文档同步）仍是**人工**，不可由脚本替代。

---

## 12. 剩余风险与处置（v2 批次运行期间）

| # | 风险 | 侦测方式 | 处置 |
|---|---|---|---|
| R1 | **GPU 竞争**（**已迁走，限制在 1003 侧**）：1003 两卡被他人占 ~89%，原 GPU 腿（RouteFinder/CaDA/MVMoE/RRNCO）会变慢并可能逐决策超时 | 各臂 `mean_timeouts`、`solver_meta.max_decide_s` | **已按用户指示把 GPU 腿整体搬到空闲的 3080**（`nvidia-smi` 实测 106 MiB / 1% → 四臂起飞后 ~5.0 GB / 94%，四臂 `errmatches=0`）；见 §13.1–§13.3。若 3080 侧仍出现超时 ⇒ 如实标注为**资源混淆**并逐臂报告 |
| R1′ | **实测占用（2026-10-13 09:27 UTC，迁移前）**：GPU0 12279/32607 MiB、**89% 利用**；GPU1 11716/32607 MiB、**89% 利用**（他人 20 个 python 进程）；CPU load 35→**24** | `nvidia-smi` | 该占用**已不再影响本批次**：GPU 腿在 3080 上跑（1003 的 `_formal_v2_gpu.sh` 改为只写 `ALLDONE` 的 no-op）。保留本条作为「当时为何要迁移」的证据 |
| R2 | **基线受 13:41–13:44 负载事故污染** | `pwsh-6` 自动跑的超时对账（阈值：c1rr2 同档 2 倍） | 超阈值 ⇒ 低负载重跑 C1 基线，再重跑四腿（脚本幂等，可整体重跑） |
| R3 | **CO-enriched +PC-HGS 模式慢**（单次 2.06 s，按池缓存 ⇒ 1 次/决策） | 该臂 `max_decide_s` 与 `mean_timeouts` | 若超时限，则该模式记「超预算」，只保留纯排序模式 |
| R4 | ~~MAPT 未创建 marker~~ **已解除**：marker 已于 2026-10-13 创建 ⇒ MAPT 两档会跑 | 编排第 4 步**不再**打印 `SKIP MAPT`；`mapt.done` 应含 `ALLDONE` | 已静态核验「放行档位 = 实跑档位」（`config=fast`，`n_frame_eval=8/n_phantom=0/cpu_threads=4`，与 marker 逐项一致；`_formal_v2_mapt.sh` 不设 `MAPT_PROVIDER_CONFIG`）见 NEXT_STEPS §3 第 6 项。若 10 h 上限内未 `ALLDONE` ⇒ 采集时会缺 MAPT 行，**须如实标注为「部分/未完成」而不得静默跳过**。⚠️ **2026-10-13 更正**：`_formal_v2_mapt.sh` 头注原先据**已被取代**的 `order()`=40.6 s（未限线程+`full` 档）写「默认不要跑 / 无法产出有效臂」—— 与「marker 已放行、`config=fast` 实测 0.536 s、单决策 7.6 s ≤ 10 s」**直接矛盾**，可能诱使人工取消该臂。已按实测重写头注（只改注释，执行段未动，1003 上 `bash -n` 复核通过） |
| R5 | **臂启动后再改 `ordering_providers.py`** | 臂 `source_stable=false` | 受影响臂重跑（该文件在臂封存清单内） |
| R6 | **任何人编辑 7 个共享计算文件**（`scenario_saa` / `strict_online_env` / `coldchain_*` / `run_exp_*` / `coldchain_evaluator_a1`） | 冻结对账（`freeze_v2_end` vs `start`）会列出文件 | 双侧哈希不等 ⇒ **配对失效**，受影响的臂与基线必须同版重跑 |
| R7 | 磁盘/内存（结果目录增长、`workers=6`×3 CPU 腿 + 9 基线 worker） | `df -h` / `free -g` / `uptime` | 降低并发或分腿重跑 |
| R8 | 轮次内未完成的收口 | `pwsh-7` 只在服务端打印「收口完成」时触发 | 三个守候任务与服务器编排彼此独立，任一存活即可推进；产物落盘后新的一轮可直接读取 |
| **R9** | **批次级冻结清单漏了两个 v2-only provider** | `record_freeze_v2.py::FILES` 早于 `mapt_provider.py` / `coenrich_provider.py`，只有 29 个文件 ⇒ **批次级**起止漂移不会发现这两个文件中途被改 | **逐臂身份不受影响**（已实测：各臂 `_identity.source_sha256` 13 个文件、含自己的 provider，`source_stable=True` 且 start==end 逐键相等）。**已把对账改成按「键的并集」分类**（`changed` / `added=只在 end` / `removed=只在 start`，见 `render_v2_summary.py` §3 与 `_local_finalize.ps1`；已用合成快照实测：新加的文件会**显式列出**且**不计入意外漂移**）⇒ 下次启动前把两个 provider 加进 `FILES` 是**安全**的（不会再「补了也看不见」）。本次**仍不补**，以保持 start/end 键集一致 |
| **R10** | **本轮 CPU 争用影响了 6 个臂的耗时**（2026-10-14 01:5x 按收口实测定稿）：`lih_51015` **54.225 次/天、`max_decide_s` 50.79 s**（最严重，超 10 s 预算 **5.1×**）；`lih_0` 25.35/天、54.35 s；`sgbs_0` 12.025/天、16.04 s；`sgbs_51015` 9.075/天、15.81 s；`symnco_51015` 0.225/天、21.50 s；`symnco_0` 0.075/天、17.40 s | `tools/_scan_timeouts.py`（收口链已产出 `TIMEOUT_SCAN.md`）+ `check_v1_v2_agreement.py`（**26/30 一致；4 个偏差恰好全在这 6 臂内**：`lih_51015` −335.15 vs −105.01 = **+219.2%**、`sgbs_51015` +27.2%、`symnco_51015` +7.9%、`sgbs_0` +2.4%） | **已定位根因（修正过一次）**：**机器被过度订阅**，而非方法慢 —— 实测 `nproc=96` 但 **load average 峰值 151–152**（7 用户），且 CPU 前列**全是我们自己**的 worker：一个腿的 6 个 worker 各 **~830–892% CPU**、另一个腿各 ~528–532%，合计 ≈**70 核**。⚠️ **关键：每个 worker ≠ 1 核**（provider 的 torch 多线程）⇒ `--workers 6` 实际吃 ≈40 核，两腿并行 ≈70–80 核，再叠加他人负载即到 151。**下一批的正确杠杆是给 worker 限线程（`OMP_NUM_THREADS=1` / `torch.set_num_threads(1)`）或下调 `--workers`** —— 只减 worker 数而不限线程仍会按核数放大。配置层面与 v1 完全相同。⚠️ **重要方法论（此处二次验证）**：**「与 v1 数值一致」不证明耗时干净** —— `lih_0` 数值通过却仍有 **25.35** 次超时/天、单决策最长 54.35 s；反过来 `lih_51015` 的 **219%** 偏差则与它 54.2 次/天完全吻合 ⇒ **偏差可被超时解释**，属时间混淆而非模型差异。**注意 `symnco_51015` 只有 0.225 次/天却偏 7.9%**，说明耗时抖动也会改变搜索步数（同 `v2_baseline_verdict.md` 的逐日不可复现）。⇒ 引用这 6 行**必须同时报 `超时/天` 与 `最大单决策`** 并注明受争用影响（`读表注意与已知限定.md` §2.8）。6 臂均 `hfr=1.0`/`fail_counts={}` ⇒ **仍可裁决**。**决策已更新（2026-10-14）**：原文写「不在收口前重跑」，理由是当时 `lih` 未结束且会与 CO-enriched 争用；负载随后降至 ~23 ⇒ 改为**收口后由非破坏性腿 `xf2_tfix` 重跑到独立目录 `results/a1_ext_formal2_timingfix/`**（采集器不扫该目录 ⇒ **默认不进表**），是否替换需先比对超时表再显式决定。其余 24 臂 `超时=0`、`最大单决策 ≤6.76 s` |

| **R11** | **`p_c`（拒单惩罚）不进入任何决策规则** —— 三档只是「同一策略的惩罚重评分」 | 源码实测：`run_a1_external_accept.py:143-167` 每天只 `env.run(i)` 一次，再对每个罚分档重新评分；`run_arm` 构造 replanner 时不传罚分（`:127-130`）；`OrderingAcceptReplanner.__init__`（`rrnco_accept_replanner.py:48-49`）**没有 `p_c` 形参** | **不得**声称「外部方法在不同拒单惩罚档位下调整了动态接单行为」。`p_c=(5,10,15)` 与 `p_c=(10,20,30)` 两列只表示**同一策略在不同合同损失下的净效用**。若要真正的策略敏感性，必须让**决策规则**接收 `p_c` 并**逐档重跑** —— 属**新比较协议**，不得对既有结果事后追认。已写进总表 §1 脚注与 `读表注意与已知限定.md` §2.9。依据：`CC_Compare/audit/INDEPENDENT_READONLY_ADJUDICATION_2026-10-14.md` §二-问题 2 |
| **R12** | **本表各行的 `future_policy` 并不一致** —— Ours 两行是 `density`，19 个对比方法的配对基线是 `reveal` | `scenario_saa.py:268` 默认 `future_policy='reveal'`；`_formal_v2_launch.sh:43` 以 `--future-policy reveal` 生成 C1 基线；`run_i3_online_compare.py:116` 硬编码 `'density'`；`run_a1_external_accept.py` **全仓无 `future_policy`**（故外部臂 gate 根本不记录该字段）；`dataset.json` 实测 `a1_c1r/c1rr2_reveal_40` = reveal、**全部** `a1_a3_online_density_*` 与 `i3_online_recovery_1009_v1` = density | **本表的效用排序只可读作「开发集能力对照」，不构成同协议下的方法归因**（A2 证据已表明该杠杆会显著改变 `cond_hist−uncond_hist` 相对差）。正式合并主表前应**预登记新的同源对照**（统一 `density`/`marginal`/`greedy`、K、时限、合同、决策状态），而不是把两种运行设置追认成同一协议。已写进总表 §1 脚注与限定 §2.9。依据：同上 §二-问题 7 |
| **R13** | **`formal_adjudication` 判据过宽 + 身份不含 checkpoint/依赖闭包** | `run_a1_external_accept.py:404`：`formal_adjudication = bool(pairing['identity_verified'])`（**只等价于身份**，未与 days/finite/hard/method 合取）；`_run_identity()` 的 15 个字段**无 checkpoint 哈希**，封存清单也不含 provider 依赖闭包（RRNCO 适配器 import 的 `coordinator`/`pickup_certificate`/`preference`/`subproblem` 均不在内）。独立矩阵对 **62 个方法臂逐一**确认 `identity_has_checkpoint_sha256 = False` | **本轮不改写任何既有 gate 字段**（用户明确要求）。**新版本**改用合取门 `Q_formal = Q_identity ∧ Q_days ∧ Q_finite ∧ Q_hard ∧ Q_nocrash ∧ Q_nondegen ∧ Q_method`，实现于 `CC_Compare/tools/qualification_gate.py`，反例测试 `test_qualification_gate.py`（**51/51 通过**，含用户要求的四类反例：39/40 天、某日 hard=False、求解器全抛异常、checkpoint 身份缺失；另加㉒–㉕「合法全拒 vs 故障全拒」对齐、**方法身份类型 `nonlearning_solver`**（Ⅰ Role C 无 checkpoint ⇒ 不强制、也不得伪造；Ⅱ 求解器体检**不豁免**；Ⅲ 防绕过 `Q_class_consistency`）与 gate·`q_days` 非整数日号反例）。⇒ 引用现有 38 个 v2 合格臂时须知：其「身份合格」**强于**运行器的 `formal_adjudication`，**弱于**协议要求的完整身份 |

| **R14** | **我把两套协议的基线值混进同一句** —— 正文曾写「Ours(I3) 456.3 vs C1 cond_hist **631.4**，`−144.680`」 | `tools/check_pairing_identity.py`（新）；`check_table_consistency.py` **抓不到这一类**（它只保证表↔表一致） | **错误确证**：`456.317 − 600.997 = −144.680` ✅ 同批成立；`456.317 − 631.410 = −175.093` ✗ 与 −144.680 不符。**631.410 属另一套协议**（外部臂配对基线 `a1_c1fresh_reveal_40|cond_hist`，主档，`future_policy=reveal`），**不能**与 I3 gate 内的配对差并列。**已修 7 处**：README §2 读点、渲染器表注（MD+Excel，且改成**从同一 gate 现算**并自证恒等式）、限定 §2.9(c)、P1-A 预声明、`项目当前状态.md`、`执行进度表.md`。**永久护栏**：`check_pairing_identity.py` 四段 —— A 数据层恒等式（52 个配对块，要求 A/B/D 同 gate 同档同天数）；B **隐含对照值** `A−D` 必须等于该 gate 自己的 `cond_hist`（I3 实测 = **600.997**）；C 文档层混用扫描（带"另一套协议/不得/≠"等声明标记的行豁免）。反例测试 `test_pairing_identity.py` **4/4 通过**，其中「把 I3 配对差换成用 631.410 算的值」被 A、B 双双拒绝。**2026-10-14 补第 D 段**（日级证据四项）：①相同 `day_id` 配对（**在 dict 化之前**校验原始记录数/唯一性/集合恰为 `0..39`）②逐日差均值重算 ③CI 排零方向一致 ④**CI 数值精确复现**（`ci_exact_tol=1e-6`）；④ 单列报告，不通过与笼统 PASS 混同 |

| **R15** | **D 段 ④ 一度报 18 个配对块 CI 未逐位复现**（偏差 5.06–9.96，但"是否排除 0"一致）⇒ 曾据此**误判**「主模型 driver 与采集器复刻的 `_stat` 不是同一个 bootstrap 生产者」 | `tools/check_pairing_identity.py::_gate_seed`（新）；生产者 `run_a1_step3_compare.py:405` / `run_a1_external_accept.py:411`（gate 顶层 `seeds` **只写 `ran`，不写 `gate`**）；官方裁决器 `diag_a3_adjudicate.py:165`、`diag_i3_online_adjudicate.py:45`（**正是读 `seeds['ran']`**） | **根因是取错种子，不是生产者不同**（原推断**作废**）：`a1_a3_online_density_*` 有**三个不同开发日集**（20260926/20260930/20261001），每块必须用该 gate 自己的种子；旧实现只找 `seeds['gate']`，找不到就**回落到 `generated_from.seed`(20260926)** ⇒ 0930/10001 的块 CI 必然对不上。**修复按确定性优先级**：`seeds.ran`（规范字段）▸ `seeds.gate` ▸ 目录名 `_YYYYMMDD_` ▸ `seeds.dev`；**冲突/歧义 fail-closed 记 SKIP**，解析不出的块**不回落任何默认种子**。⚠️ **明确不做**「试遍候选种子取能对上的那个」（那是事后择优，会使 ④ 失去证据价值）；`part_d` 现**披露每块实际用了哪一路证据**。**1003 复跑**：`A` 52/52、`B` I3=**600.997**、`C` 0 条、`D` 硬失败 0 / CI 未复现 **0** / SKIP 0 ⇒ **`PAIRING_IDENTITY_PASS`（rc=0）**，④ `CI_exact_reproduction=True`，种子来源 `results_key=28 / seeds.gate=24`（无一块需弱兜底）。新增种子解析反例 **10 条**（含「按 `pc` 选块」——取错档会一次跑出 148 条假失败），`test_pairing_identity.py` 现 **14/14 通过**。**另按用户要求只读核对生产者 bootstrap 并并列记录两种预先定义**（`tools/_probe_bootstrap_defs.py`）：`_stat` = 单 `default_rng(seed+777)` + 2000 次**顺序** `r.integers(0,n,n)` + 朴素百分位（`method='linear'`）；定义 **P**（生产者顺序，`paired_all_days` 的 `for r in a_rows`）为规范口径，定义 **S**（`day_id` 升序，`collect_compare_table._pair` / `diag_a3_adjudicate._stat_exact`）为消费侧口径。**实测：52/52 个配对块 P == S，CI 上下界最大差 0.000000，落盘值同时命中 P 与 S、两者皆不命中 0** ⇒ 本次**不存在**"事后挑更有利区间"的余地（也未这样做）；但顺序敏感性是结构性性质，护栏保留，**未替换任何已发布 CI** |

| **R16** | **资格检查工具自身有两处确定性缺陷**（2026-10-14 用户复核在源码里指出） | ① `audit_qualification_matrix.py` 的 `ids = [int(d.get('i')) …]` **静默强转** ⇒ `0.5→0`、`'5'→5`、`True→1`，40 条 `i=0.5..39.5` 被截断成 `0..39` ⇒ **非整数日号伪装成"日号完整"**；② 同文件 `fin = [u for u in utils if isinstance(u,(int,float))]` ⇒ `nan`/`±inf` **都是 `float`**，被算作"有限"，判据形同虚设；③ `qualification_gate.py::q_validity` 用 `not n_solves` 把「字段**缺失**」当故障证据 ⇒ 纯规则臂结构性 `solver_meta={}`，**合法的规则全拒臂被误判为有效性失败** | **三处均已修复**：①② 改为「原始日号必须是**真整数**（`bool` 不算）」+ `math.isfinite`（并排除 `None`/字符串/`bool`）；③ 改为**只认正面故障证据**（`execution_crash` / `n_fail>0` / **显式** `n_solves==0`），"方法是否真被执行"归 `Q_method`（非规则臂 `n_solves` 缺失**依然**否决 ⇒ **不留漏洞**）。⚠️ **判据只收紧、不放松**：修复后在 **264 个真实臂**上逐臂复算，`Q_formal_core`/`Q_formal_strict`/`Q_vetoes` **逐字段零差异**（core **38/62**、strict **32/62** 不变；λ=3 仍由 `Q_nondegen` 否决）⇒ **历史裁决未被改写**；矩阵 JSON 已由修复后工具重新生成（`criteria.revision=2`）。新增反例 `test_audit_qualification_matrix.py` **22/22**（含 `naive_*` 断言**故意复刻旧实现**以证明缺陷真实存在）、`test_qualification_gate.py` 现 **36/36**（新增 ㉒–㉕：规则臂+合法全拒+`solver_meta={}` ⇒ `qualified=True`/`informative=False`） |

| **R17** | **归档清单把两件不同的 commit 挤在一个字段**（`target_commit = git rev-parse HEAD`），存在**循环自指** | `tools/make_artifact_manifest.py`（2026-10-14 第二版） | 已拆分为：**`source_commit`**（快照来源，入库清单；附 `source_commit_dirty_n/_files` 与 `self_reference_note`）+ **`release_commit`/`release_tag`**（发布靶点，写进**独立** `results/.../RELEASE_METADATA.json`，**不入 Git**，以 `manifest_sha256` **指回**清单 ⇒ **单向无环**）。`--verify`：`source_commit != HEAD` ⇒ **告警**（清单自身所在提交必是其**子**提交，属**预期**）；**不是** HEAD 祖先 ⇒ **判失败**（真漂移）；清单里仍有 `target_commit` ⇒ **判失败**（负例实测 rc=1）。**`source_commit_dirty_*` 的范围只看入库快照的目的地目录（`docs/a1_development_comparison/`，已排除清单自身）**：原因是对 `results/**`（被 gitignore）做 `git status` **恒为空 ⇒ 会假报"干净"**，毫无意义；而目的地是 git 跟踪的，`git status` 在那里**确实**回答「本次归档的快照是否已提交」。语义：`=0` ⇒ 快照内容**恰等于** `source_commit` 的树；`>0` ⇒ 内容 = `source_commit` **加上**所列改动（此时**不得**把 `source_commit` 当作内容定位）。本次修复后该字段与 `source_commit` **语义自洽**：清单记录的来源提交**必是** HEAD 的祖先（`--verify` 实测 `merge-base --is-ancestor` rc=0），而 `dirty` 表示「快照内容是否已提交」——归档后先提交快照、再重建一次清单即可到达 `dirty=0` 的不动点（此后重建不再产生 diff，因清单自身被排除在该判据之外）。三份 Markdown 快照重建后与已提交版本**逐字节一致** |

**交接说明**：即使无人值守，链路上任一环节完成后都会把产物写到
`CC_Compare/results/a1_compare_table/`（表格、图、`dataset.json`、`V2批次结果.md`、
冻结快照）；重新执行只需 `collect_compare_table.py` + `render_compare_table.py` 两条命令。

---

## 13. 3080 机器可用性评估（2026-10-13 实测）

**身份兼容性：✅ 已验证通过**。把 1003 的共享计算文件同步到 3080 后，3080 用当前代码重算：

| 项 | 3080 结果 | 与 1003 臂记录 |
|---|---|---|
| `contract_hash` | `e25a12104b20c3ea…` | **SAME** |
| `data_sha256` | `8767a03d787c0e0f…` | **SAME** |
| `data_meta_sha256` | `72a805fd5bc0b2b7…` | **SAME** |

⇒ **跨平台（Windows py3.9 + numpy）与 1003 逐字节一致**，因此「把臂放到 3080 跑、再与新基线配对」
在身份上是合法的（工具 `CC_Compare/tools/verify_contract_identity.py`，在 3080 上可直接复验）。

**但依赖可行性限制了可迁移的方法**（3080 现状：`pytorch` env = py3.9 + torch 2.0.1 + cu117；base = py3.9 + torch 2.5.1）：

| 方法 | 依赖 | 可否迁到 3080 |
|---|---|---|
| CO-enriched-ML（纯排序） | **仅 NumPy**（MLP 为 NumPy 复刻） | ✅ 可直接跑 |
| MAPT | torch + 官方 MAPT 代码 + `_deps/tree.py`（该 env 已装齐，训练即在此完成） | ✅ 可直接跑 |
| MVMoE / POMO / AttentionModel / Sym-NCO / SGBS / DeepACO / Omni-VRP / LIH | torch + 各自包（+ 权重需单独上传） | ⚠️ 依赖较易，但**要迁 8 个包 + 权重**，且 torch 版本与 1003 不同（数值会有差异） |
| **RouteFinder** | torch + 自身包 | ⚠️ 同上 |
| **CaDA** | torch + **torchrl**（官方 pin 0.1.1，需兼容补丁） | ❌ 成本高 |
| **RRNCO** | torch≥2.7 + **rl4co 0.6.0 + tensordict**（`rrnco_backend` 用 `RMTVRPEnv`） | ❌ 3080 的 py3.9/torch2.0.1 装不上，需新建 py3.10+ 环境并重装 torch（~2.5 GB） |

**结论与已采取的替代优化**：
1. **不迁 RRNCO/CaDA**（依赖代价 > 收益，且有污染 3080 共享环境的风险）；
2. **临界路径分析**：四腿的瓶颈是 GPU0 上的 RRNCO（~12 h × 竞争）；把 LIH/cpuB 迁走**不改变**
   总时长（瓶颈仍在 GPU），只有迁 RRNCO 才有净收益 —— 而它恰是依赖最重的一个；
3. **真正省时间的动作已落地**：把 CO-enriched 的**同模式 2 档并行**（`_formal_v2_coenrich.sh`
   `run_pair()`），+PC-HGS 段从 ≈18 h → **≈9–10 h**（依据：探针单决策 ≈4.2 s / 10 s 预算，
   且与 9-worker 基线同跑仍 `timeouts=0`；机器 ~70 空闲核，12 workers 不构成瓶颈）。
4. 若将来需要把 GPU 腿整体搬到 3080：前置工作是**新建 py3.10 + torch≥2.7 + rl4co 0.6.0 的
   隔离环境**（不要动 `base`/`pytorch`），再上传 4 个 checkpoint（≈128 MB）与相关包。

### 13.1 用户指示「把 GPU 腿整体搬到 3080」——已执行（2026-10-13）

| 步骤 | 状态 |
|---|---|
| 身份兼容性实测 | ✅ `contract`/`data`/`data_meta` 三哈希与 1003 **逐字节相同** |
| 源码推送（4 方法包 + provider + `dcc_rh_v4` + `common`） | ✅ 14 个目录树 ≈300 文件 |
| 权重推送（4 个 checkpoint，128 MB） | ✅ |
| 3080 上链路烟测（MVMoE 1 天） | ✅ `rc=0`，gate 生成（用现有 `pytorch` env，纯 torch） |
| 1003 GPU 腿关闭（防双写） | ✅ `_formal_v2_gpu.sh` 改为只写 `ALLDONE`（编排等待逻辑不变） |
| `cc_gpu` 隔离环境（py3.10 + torch2.4.1 cu121 + rl4co 0.6.0） | ❌ **失败**（见 §13.2：rl4co 把 torch 换成了 CPU build）→ 改用 `cc_compare` |
| 4 个 per-method 臂脚本（各跑两档，4 方法并行） | ✅ 已生成并上传（`tools/_gpu3080/`），device 已修正为 `cuda`（见 §13.2） |
| 端到端编排（等基线→等 env→上传→WMI 并行启动→等 4 done→8 个 gate 回传 1003） | ✅ 已挂起（`tools/_gpu3080_orchestrate.ps1`，本地后台任务） |

**输出与回传**：3080 侧 `results/a1_ext_formal2_3080/<tag>_{0,51015}/`，完成后由编排回传到
1003 的 `results/a1_ext_formal2/<tag>_{0,51015}/gate.json`（在收口采集之前），因此最终总表
不区分机器、只按身份与协议口径合并。**已知风险**：rl4co 在 py3.10/torch2.4.1 上的兼容性；
CaDA 可能仍需 1003 上那份 torchrl 兼容补丁 —— 若某方法在 3080 起不来，按 R1 处置
（该腿回落到 1003 低负载时段重跑，或如实标注为未完成）。

### 13.2 设备口径缺陷与修正：CPU→CUDA（2026-10-13，启动前拦截）

**发现**：首批生成的 4 个 3080 臂脚本把**全部 4 个 provider 都写成 `--provider-device cpu`**，
而 v1 这条腿的**终版（`_f`）臂全部是 `cuda`**。逐臂实测（工具 `CC_Compare/tools/check_gpu_device.py`，
读落盘 gate 里**与实际启动命令行对应的那个键**）：

| v1 终版臂 | 实际 device 键 | `mean_timeouts` |
|---|---|---|
| `mvmoe_0_f` / `mvmoe_51015_f` | `provider_device=cuda` | 0.00 / 0.00 |
| `routefinder_0_f` / `routefinder_51015_f` | `provider_device=cuda` | 0.00 / 0.00 |
| `cada_0_f` / `cada_51015_f` | `provider_device=cuda` | 0.00 / 0.00 |
| `rrnco_0_f` / `rrnco_51015_f` | `rrnco_device=cuda` | 0.00 / 0.00 |

**为什么必须修**：
1. `CC_Compare/setup_envs.sh:18-22` 明确写「CaDA/RRNCO 都是 PyTorch-GPU 模型，**必须 CUDA torch**」；
2. 1003 上这条腿的权威解释器是 **`/home/hzeng/envs/cc_compare/bin/python`**
   （`CC_Compare/_gpu_leg_1003_fallback.sh:13`）= **torch 2.11.0+cu128 + rl4co 0.6.0**；
3. v1 终版**四臂八档全部 cuda 且 `mean_timeouts=0.00`**。若 v2 按生成脚本用 cpu，就是
   **无理由偏离 v1 口径**；而 10 s 预算是硬约束，provider 若因设备退化成 cpu 而超时，
   等于**主动削弱 RouteFinder（v1 最强外部臂）与 CaDA**，总表会失真。
   3080 的 GPU 当时完全空闲（106 MiB / 12288 MiB、util 1%、driver 595.79 / CUDA 13.2），
   没有理由不用。

**一个真实的读数陷阱（已固化成工具，避免我自己或后人再踩）**：
`results/a1_ext_extra/` 下同名方法有**两套**臂——非 `_f`（更早、已被取代）与 `_f`（终版、
总表实际使用）。**只有 `_f` 是正式口径**：非 `_f` 的 `cada_0`/`cada_51015` 记的是
`provider_device=cpu`、`mean_timeouts=7.60/32.74`（本地副本；1003 上那两份更早数据为 30.4/130.95），
**不**代表 v1 终版。此外 **cada 的 gate 里 `provider_device=cpu` 与另一处 `cuda`（别处的默认值）
同时存在**，所以判定必须**按实际使用的键**取值（routefinder/cada/mvmoe 看 `provider_device`，
rrnco 看 `rrnco_device`），**不能只搜「出现过 cuda」**——宽松判定会把 cpu 臂误判为通过。
`tools/check_gpu_device.py` 默认只核 v2 的 8 个臂，用 `--arms` 可核 v1 `_f` 臂（即上表）。

**旧 `cc_gpu` 环境为何是 CPU 版（根因，已留日志证据）**：`_setup_cc_gpu_3080.cmd` 先
`pip install torch==2.4.1 --index-url .../cu121`（第 12 行），随后
`pip install rl4co==0.6.0`（第 14 行）——rl4co 0.6.0 要求的 torch 更高，**pip 直接把 CUDA torch
替换成 PyPI 的 CPU build**。`C:\Users\zh\cc_gpu_setup.log` 显示
`Successfully installed … torch-2.14.1 …`，校验行 `torch 2.14.1+cpu cuda False`。
⇒ 「装完 rl4co 再校验」这一步本来能发现，但当时没校验 `cuda.is_available()`。

**修正动作（4 项，均已落地）**：
1. 在 3080 新建 **`cc_compare`**（py3.11 + `torch==2.11.0+cu128` + `torchvision` +
   `rl4co==0.6.0` + einops/pyyaml/tqdm），对齐 1003 的同名环境；工具
   `tools/_setup_cc_compare_3080.cmd`（**先装 CUDA torch 再装 rl4co**，末尾强制校验
   `torch.cuda.is_available()` 并打印 `VERIFY torch 2.11.0+cu128 cuda 12.8 True`）；
2. 重生成 4 个臂脚本（`tools/gen_gpu3080_scripts.py`）：env 改为
   routefinder/cada/rrnco→`cc_compare`、mvmoe→`pytorch`（MVMoE provider 只 import torch），
   **4 个 provider 全部 `--provider-device cuda`**；
3. 编排器（`tools/_gpu3080_orchestrate.ps1`）增加**功能性 env 门禁**：轮询
   `tools/_check_cc_compare.cmd`（`import torch,rl4co,torchrl,tensordict` 且
   `cuda.is_available()` 才打印 `ENVOK`），**不再依赖 `.done` 标记**——该标记由建环境脚本
   无条件写出，即使安装失败也会出现（这一点已实测踩到）；
4. 启动前清掉 3080 上旧的 `<tag>.done`，避免编排器误判「已完成」。

**第二个缺陷（同类，同样是启动前拦截）：`--provider rrnco` 臂根本起不来。**
RRNCO **不在** `ordering_providers.py::_REGISTRY` 里（该 registry 只有 10 项：
routefinder/cada/mvmoe/pomo/attention/symnco/omnivrp/sgbs/deepaco/lih），
`CC_Compare/RRNCO/a1_accept_v1/` 下只有一个**原生 driver** `run_formal_rr.sh`，
真实口径是：

```bash
run_a1_external_accept.py --solver rrnco --rrnco-ckpt .../epoch_199.ckpt \
    --rrnco-device cuda --baseline-from <BASE> --dev-instances 40 --workers 4 \
    --penalty "p_c=0" --out results/.../rrnco_0
```

写成 `--solver ordering --provider rrnco` 会在 `build_ordering_provider` 处直接
`SystemExit: unknown --provider: 'rrnco'`（`run_a1_external_accept.py:271` 的 `--solver`
choices 是 `ortools|pyvrp|rrnco|ordering`，rrnco 是独立 solver 面，不是 provider）。
**修正**：`tools/gen_gpu3080_scripts.py` 拆成两个模板（provider 面 3 个方法 / RRNCO 原生面），
重生成并重上传 4 个脚本。RRNCO 路径依赖的 `rrnco_accept_replanner.py`、
`dcc_rh_v4/rrnco_guided_adapter.py`、`dcc_rh_v4/rrnco_backend.py`、`rrnco` 包
与全部 4 个 checkpoint（51.6 / 20.1 / 41.3 / 15.3 MB）已在 3080 上预检存在
（`tools/_preflight_3080_files*.cmd`；`dcc_vrp/run_dcc.py` 缺失但**不影响**——该文件
是另一条独立 driver，`--solver rrnco` 不 import 它）。

**对总表的口径声明**：
- v2 GPU 腿四臂统一在**空闲 RTX 3080 Ti** 上以 cuda 运行（routefinder/cada/mvmoe 用
  `--provider-device cuda`，rrnco 用 `--rrnco-device cuda`），与 v1 终版的设备口径**一致**，
  因此 v1↔v2 可按同口径比较；
- 逐臂照旧报告 `mean_timeouts`：v1 终版八档均为 `0.00`，若 v2 出现非零超时应单独说明原因；
- **事后核验命令**：`python CC_Compare/tools/check_gpu_device.py`（要求 8 个臂各自的实际
  device 键都是 `cuda`；缺 gate 或非 cuda 即 FAIL）。

### 13.3 v2 批次的缺陷全台账（A–L：启动缺陷 → 身份缺陷 → 收口链缺陷，均逐条定位/修复/验证）

基线 `a1_c1fresh_reveal_40` 正常产出（9 组 40/40，`gate.json` 388 KB，`base.done`）后，
四条腿的启动暴露了三个缺陷。**它们都会「看起来已启动、实则为空/坏」**，因此逐条记录：

**缺陷 A：四条腿全部秒退、零日志（`_formal_v2_launch.sh` 相对路径）**
`_formal_v2_launch.sh:25` 先 `cd` 到**扩展根**，而第 68–71 行却写
`tmux new -d -s xf2_cpuA "bash CC_Compare/_formal_v2_cpuA.sh"` —— 扩展根下**没有** `CC_Compare`
目录（它在仓库根）。于是 4 个 tmux 会话立刻 `No such file or directory` 退出：
会话消失、无任何日志，而 `launch.log` 仍打印「已启动 tmux: xf2_gpu / …」、`base.done` 也正常
——**从表面看像已启动**。回看 `tmux ls` 才发现 4 个会话都不存在、且无 `run_a1_external_accept` 进程。
**修正**：改用绝对路径 `bash /home/hzeng/project/MASKCO-Main/CC_Compare/_formal_v2_<leg>.sh`
并给每条腿加独立日志 `/tmp/xformal_v2/<leg>.log`；另加幂等补救脚本
`CC_Compare/_formal_v2_legs_only.sh`（只起腿、不重跑基线，已有 `<leg>.done` 则跳过）。

**缺陷 B：cpuC 用错解释器 → 产出「全失败」的 OR-Tools/PyVRP gate，并会顶掉好结果**
`_formal_v2_cpuC.sh` 用 `$PY=…/envs/cc_compare/bin/python` 跑 `--solver ortools/pyvrp`，但
**`cc_compare` 里没有 ortools / pyvrp** → 每个实例 `ModuleNotFoundError`。更危险的是 **driver 仍然
写出了 `results/a1_ext_formal2/{ortools,pyvrp}/gate.json`（各 ~60 KB 的失败记录）**；而
`render_compare_table.build_rows` 的 `GROUP_RANK` 把 `a1_ext_formal2` 排在 `a1_ext_a05e` 之前，
**坏结果会静默顶掉 a05e 里本来正确的 OR-Tools/PyVRP**。
**修正**：(1) 已 `rm -rf` 那两个污染目录；(2) 求解器改用实测含 `ortools 9.11.4210` + `pyvrp` 的
`/home/hzeng/envs/a1_ext/bin/python`（规则臂仍用 `cc_compare`，纯 NumPy）；(3) 参数全部走 driver
默认值，与冻结的 a05e 配置**逐项一致**（`solution_limit=30` / `pyvrp_max_iterations=1000` /
`time_limit=10` / `workers=8` / `n_orders=200` / `capacity=50` / `num_vehicles=15` / `penalty=null`
即三档全跑），只把 `baseline_from` 换成身份合格的新基线。
**顺带查实**：现有 `a1_ext_a05e_ortools_run.log` 显示 a05e 的 `baseline_from` 是
`results/a1_step2_gate_c1_20260926/gate.json`（**无 identity 块**）⇒ 现在总表里的 OR-Tools/PyVRP
**本来就不是身份可裁决的**，正是 v2 必须重跑它们的原因。

**缺陷 C：3080 的 `cc_compare` 缺 `entmax` → CaDA 秒退**
CaDA 的 provider 需要 `entmax`（entmax15 注意力）。首轮 4 臂里 mvmoe/routefinder/rrnco 起得来，
**cada 在 ~16 s 内 `ModuleNotFoundError: No module named 'entmax'`、rc=1，并写下了 `cada.done`**
（因为 `.cmd` 末尾无条件写 `.done`）——即「失败也会被编排当成完成」。
**修正**：`pip install entmax`（1.3，纯 Python 13 KB）→ 删掉陈旧的 `cada.done` → 重跑该臂。
修正后四臂同时在跑：`nvidia-smi` 显示 **~94% 利用率、5.0/12.3 GB**（3802 MiB → 5045 MiB，
cada 启动后上升），证明四臂**确实在 GPU 上**。

**缺陷 D：PyVRP 臂「静默全拒单」——一条 rc=0、timeouts=0、身份全合格却是空的臂**
（2026-10-13 实测，**本批最危险的一类缺陷**）

`_formal_v2_cpuC.sh` 让 ortools 与 pyvrp 共用 `a1_ext`（**pyvrp 0.14.0**）。但 pyvrp 0.14.0 的签名已改：

```
a1_ext (0.14.0) : add_depot(self, location: Location, tw_early=0, tw_late=…, service_duration=0, *, name='')
MASKCO_env (0.11.3): add_depot(self, x: int, y: int, tw_early=0, tw_late=…, *, name='')
```

而 `solver_accept_replanner._solve_pyvrp:307`（代码注释明写「0.11.3」）按后者调用
`m.add_depot(x, y, tw_early=0, tw_late=…)` ⇒
**`TypeError: Model.add_depot() got multiple values for argument 'tw_early'`**，每个实例都抛
（实测 **40 天 × 3 档 = 120 次** `SOLVER_EXC`）。

危险之处在于 driver 对 `SOLVER_EXC` 是**静默降级**：该决策退化为「全部拒单」。于是该臂表现为

| 指标 | 空臂实测 | 判读 |
|---|---|---|
| `p_c=0` util | **0.00** | 恰好 0 —— 因为 p_c=0 时拒单损失为 0，**全拒 = 0** |
| 主档 util | **−1943.12** | 全拒 200 单/天 × p_c=(5,10,15) 的罚金 |
| `hard_feasible_rate` | 1.0 | 空计划天然可行 |
| `mean_timeouts` | 0.0 | 没超时（根本没在解） |
| rc / 身份 / 配对 | 0 / `source_stable=true` / `formal_adjudication=true` | 全合格 |

⇒ **「rc=0 + timeouts=0 + 身份合格」不足以判定一条臂有效。** 对照 v1 同臂 **909.2 / −569.8**，
差异量级一眼可见——**这就是为什么必须做 v1↔v2 逐臂对账**（同臂同代码应逐位复现，
`attention` 实测 1236.21 = 1236.21）。

**修正**：(1) 删除空臂目录 `results/a1_ext_formal2/pyvrp/`；
(2) `_formal_v2_cpuC.sh` 改为**按 solver 选解释器**——ortools 用 `a1_ext`（其 ortools 9.11.4210
实测 `p_c=0` util=**1300.65**，与 v1 的 1300.7 一致，无需改），pyvrp 用
`/home/hzeng/envs/MASKCO_env`（pyvrp **0.11.3**，已实测 `CALL_OK`）；
(3) 一次性修复脚本 `_formal_v2_pyvrp_fix.sh`：等 cpuC 的规则臂 ALLDONE（让机器回到安静状态，
避免 10 s 预算被负载吃掉）→ 删空臂 → 用 MASKCO_env 重跑 pyvrp → 写
`/tmp/xformal_v2/pyvrp_fix.done`。
**为什么两个 solver 用不同 env 是合法的**：臂的身份链记录的是**源码哈希**与合同/数据哈希，
不含解释器；环境只影响 API 兼容性与速度，不影响身份。

**✅ 缺陷 D 已修复并验证（2026-10-13 19:2x）**：`_formal_v2_pyvrp_fix.sh` 用
`MASKCO_env`（pyvrp **0.11.3**）重跑后，`results/a1_ext_formal2/pyvrp/gate.json` 实测

| 档位 | 修复前（空臂） | **修复后** | v1 `a1_ext_a05e_pyvrp` |
|---|---|---|---|
| `p_c=0` | **0.00** | **909.22** | **909.2** ✅ |
| 主档 | **−1943.12** | **−569.78** | **−569.8** ✅ |
| 高罚档 | — | −2048.78 | — |
| `mean_timeouts` / `hfr` | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |

且 `identity_verified=true`、`problems=[]`。⇒ (a) 解释器修正确实恢复了正确行为；
(b) v2 **逐位复现** v1 的 PyVRP 数字；(c) 该臂身份合格、可进入正式表。
（对照：全拒单空臂在高罚档应约为 −3886，而实测 −2048.78 说明它确实在求解。）
**这条也再次印证缺陷 D 的教训**：`rc=0 + timeouts=0 + 身份合格`并不能证明臂有效，
只有与 v1 同臂对账（或看是否命中「全拒单」签名）才能发现空臂。

**缺陷 E（不是缺陷，但总表必须加脚注）：`rule_valuee λ=3` 臂「全拒单」**
v2 实测该规则 **p_c=0 util = 0.00**、主档 = **−1943.1** —— 即**每天都把所有订单拒掉**
（p_c=0 时拒单损失为 0，所以恰好等于 0；主档等于全拒 200 单/天 × p_c=(5,10,15) 的罚金）。
**这是该规则真实、可复现的行为，不是空臂**：v1 同臂逐位相同（`check_v1_v2_agreement.py`
判 OK），且 λ 越大越倾向拒单符合该规则定义。同一家族的 λ=2 是正常值 **1403.27**。
⇒ **总表里 λ=3 那一行必须加注「该规则全拒单」**，否则读者会把它误读成「一个性能极差的规则」，
而它其实是「一个**不服务**的规则」。这条也正是我第一版空臂判据的**假阳性来源**——
修好后 `check_v1_v2_agreement.py` 只在**无 v1 对照**的臂（coenrich_*/mapt_*）上启用
「p_c=0 util 恰好 0」签名，见该脚本 `degenerate()` 的文档字符串。
**顺带确认**：规则臂（`run_a1_downstream_ablation.py` 产出）**确实**带
`_pairing_identity`、`formal_adjudication`×3 档、`source_stable`，且
`baseline_from = results/a1_c1fresh_reveal_40/gate.json`（新基线）✅。

**缺陷 F（本批最严重的身份缺陷）：3080 整个 GPU 腿的 `_pairing_identity.identity_verified=False`**
（2026-10-13 实测）

3080 首个完成的臂 `mvmoe_51015` 报 `identity_verified=False`、
`problems=['shared_source_sha256_mismatch']`、`formal_adjudication=False` —— 即**整条 GPU 腿都
无法正式裁决**，v2 的核心目的对该腿失效。

**定位**：逐文件比对两侧 `shared_source_sha256` 映射（工具 `tools/dump_shared_source.py`），
**7 个共享计算文件里只有 1 个不同**：

| 文件 | 1003（基线） | 3080（臂） |
|---|---|---|
| `scripts/evaluation/scenario_saa.py` | **36e08e06…** | **4e47b99f…** ← 09-30 修复前的旧版 |
| 其余 6 个 | — | **全部逐字节相同** ✅ |

**根因（比缺陷本身更重要）**：**3080 的源码是从「本地工作区」推的，而本地工作区对
`scenario_saa.py` 落后于 1003**——即 09-30 的修复只落在 1003，从未回同步到本地。
⇒ **「本地副本」不等于「权威副本」**。本轮的 `dump_tree_hashes.py` 全树比对（315 vs 421 文件、
21 个内容不同、106 个仅存在于 3080）也印证：本地/1003/3080 三者**各自漂移**，
不能以任一为默认基准。
**就本腿而言只有 `scenario_saa.py` 是实质问题**：另外 20 个不同的文件（`run_a1_step2_gate.py`、
`run_a1_step3_compare.py`、`event_plan_*`、`i3_*`、`tests/*` 等）**不参与这 4 条臂**；
`run_a1_external_accept.py`、`solver_accept_replanner.py`、`rrnco_accept_replanner.py`、
`coldchain_*`、`strict_online_env.py`、`run_exp_*` 两侧**均一致**。

**修正**：(1) **从 1003 拉**正确的 `scenario_saa.py`（**不是从本地上传**——本地就是旧版），
再推到 3080；(2) 核实 3080 侧哈希 = `36E08E06…` ✅；
(3) **杀掉并清空已跑的 4 条臂**（它们封存的是错误的共享哈希，结果不可用），
删除 `results/a1_ext_formal2_3080/*` 与 4 个 `.done`，重新起 4 条臂
（`tools/_restart_gpu_leg_3080.cmd`）；(4) 重启后实测 GPU **4851 MiB / 93%**、
8 个 `run_a1_external_accept` 进程、三条臂日志在增长 ⇒ 腿在跑。
**待办**：新臂落盘后必须复核 `identity_verified=True`（`check_arm_gate.py`），
并让 `check_gpu_device.py`/`check_v1_v2_agreement.py` 一并通过。

**缺陷 G：3080 被跑到 commit-limit 崩溃（PowerShell 起不来 + SSH 被 reset）**
（2026-10-13 实测，缺陷 F 的处置过程中触发）

重启 GPU 腿后不久，3080 出现：`powershell` 无法启动
（`无法加载 System.Management.Automation … HRESULT 0x800705AF`，即
**ERROR_COMMITMENT_LIMIT**）、SSH `WinError 10054 远程主机强迫关闭连接`。
`wmic` 实测 **TotalVisibleMemorySize ≈ 31.8 GB 而 FreePhysicalMemory ≈ 418 MB**。

**根因不是「3080 内存小」，而是「重启没清干净」**：我重起前那次 kill 的进程匹配写得太窄
（只按 `*_gpu3080_*` / `*run_a1_external_accept*` 匹配 `python.exe`/`cmd.exe`），
实测**只杀掉了 1 个进程**，于是**旧臂与重起的新臂同时运行** ⇒
4 臂 × 4 workers 实际变成 **~24–32 个 torch 进程** ⇒ 内存耗尽。

**恢复与纠正**：
1. 退避重试 SSH（第 2 次恢复）→ `taskkill /F /IM python.exe` + `/IM conda.exe` →
   实测 **python 进程数 = 0**、**FreePhysicalMemory 回升到 25.0 GB**；
2. 改为**串行**驱动 `tools/_gpu3080_sequential.cmd`：一次只跑一个方法（方法内部仍
   `--workers 4`），每个方法各自写自己的 `<tag>.done`，**编排的等待逻辑不用改**；
3. 重启后实测 **python 进程 6 个、空闲 18.7 GB、GPU 1296 MiB / 64%** ⇒ 内存安全。
   理由：GPU 本来就是 94% 利用率、瓶颈在 GPU，串行对总时长的代价有限，但把
   「4 臂并发 × 4 workers」的峰值内存从 ~24 GB 压到 ~6–8 GB。

**教训（两条）**：
- **重启前必须证明「旧进程全部消失」**（本例：先 `Get-Process python | Measure` 确认为 0 再重起），
  否则新臂会与旧臂叠加；进程匹配要按**可执行名**整体清理，而不是猜命令行子串。
- 任何**守护/重启脚本**都要自带「清理—验证—启动」三步，并把验证结果打印出来。

**缺陷 H：`_formal_v2_after.sh` 的同类相对路径 bug —— 会让 CO-enriched 白等 40 h、并静默跳过 MAPT**
（2026-10-13 在 CO-enriched 起飞前拦截）

与缺陷 A 完全同源（`cd` 到扩展根后却用仓库根相对路径），但后果更隐蔽：

| 行 | 原写法 | 实际解析 | 后果 |
|---|---|---|---|
| 50 | `bash CC_Compare/_formal_v2_coenrich.sh` | `<扩展根>/CC_Compare/…`（不存在） | `xf2_coe` 会话**秒退**；`coenrich.done` **永不出现**；`wait_for` 空转 **2400×60s = 40 h** 才继续，而 `after.log` 全程只显示「等 CO-enriched」——**看起来像在跑** |
| 62-63 | `MAPT_MARK=CC_Compare/MAPT_code/…/RUN_ARM_APPROVED` | 同上 | `[ -f … ]` 恒假 ⇒ **即使 marker 与 ckpt 都在，MAPT 也被静默 `SKIP`** |
| 66 | `bash CC_Compare/_formal_v2_mapt.sh` | 同上 | 同 50 |

**修正**：第 54 行改为 `$ROOT/CC_Compare/_formal_v2_coenrich.sh`、第 69–70 行改用
`$ROOT/CC_Compare/MAPT_code/…`、第 73 行改为 `$ROOT/CC_Compare/_formal_v2_mapt.sh`
（`$ROOT` 在脚本第 12 行已定义）；并逐条 grep 确认其余 `CC_Compare/` 引用都是
`../CC_Compare/…`（相对扩展根，正确）或绝对路径。
**处置**：因为 `xf2_after` 当时**正在运行**（bash 会边读边执行，改文件不一定生效），
故 **kill 掉 `xf2_after` 会话后用 `_formal_v2_after_start.sh` 重启**（该脚本先 `cd` 仓库根，
再用 `CC_Compare/…` 启动，相对路径正确）；重启后 `tmux ls` 见 `xf2_after` 已重建并于
18:57:08 通过 `F1:gpu`，继续等 cpuA/cpuB。**幂等**：它只是重新等腿标记，不重复任何计算。
**注意**：重启会**截断 `after.log`** —— 这正是我为收口 watcher 加「新鲜度守卫」的原因
（否则会拿上一轮遗留的「收口完成」立刻误判，见 `NEXT_STEPS.md` §5 第三个运维坑）。

**系统性排查（缺陷 A/H 之后做的全家族扫描，结论：已无同类隐患）**：对 `CC_Compare/*.sh` 全量
grep 所有 `CC_Compare/` 引用，并逐个核对「脚本先 `cd` 到哪里」：
- **先 `cd` 到仓库根**的 3 个 `*_start.sh`（`_formal_v2_start` / `_formal_v2_after_start` /
  `_formal_v2_smoke_start`）用 `bash CC_Compare/…` ✅ 正确；
- **先 `cd` 到扩展根**的脚本一律用 `../CC_Compare/…`（`_formal_v2_coenrich` / `_formal_v2_mapt` /
  `_formal_v2_finalize` / `_gpu_leg_1003_fallback` / `_formal_v2_after` 的第 40/85-90 行）或
  绝对路径（`$ROOT/CC_Compare/…`、`/home/hzeng/.../CC_Compare/…`）✅ 正确；
- 其余命中全是注释/用法示例，或 `PY=/home/hzeng/envs/cc_compare/bin/python` 里的
  `cc_compare` 子串（非路径）。
⇒ 同类缺陷只有 A 与 H 两处，均已修复；启动/重启脚本family 现已自洽。

> ⚠️ **2026-10-13 20:2x 更正上面这条结论的适用范围**：该扫描只覆盖 `CC_Compare/*.sh`（1003 侧），
> **没有覆盖本地 PowerShell 编排链**。补扫后立刻发现**缺陷 I**（见下），正是同一族
> 「相对/绝对路径假设」问题，只是发生在本地 `_gpu3080_orchestrate.ps1` 的 PowerShell 回传步骤里。
> 教训：家族扫描必须按**执行面**（1003 bash / 3080 cmd / 本地 pwsh）分别做，不能只扫一种后缀。

**缺陷 I：GPU 腿 8 个 gate 永远到不了 1003 —— `Copy-Item` 不建父目录 + 失败被静默吞掉**
（2026-10-13 20:2x，**在该步骤首次真正执行之前**由代码审查发现，尚未造成后果）

- 原 `_gpu3080_orchestrate.ps1` 第 4 步回传逻辑：
  `Copy-Item -Force <本地暂存> <本地 $EXT/results/a1_ext_formal2/<臂>/gate.json>`。
- 四重叠加导致**必然静默失败**：① 本地 `C-VRP_.../results/a1_ext_formal2/` 目录**根本不存在**
  （实测 `Test-Path` 为 MISSING）；② `Copy-Item` **不会创建缺失的父目录**，直接报
  "Could not find a part of the path"；③ 随后的 `_sftp_put.py` 因**本地文件不存在**同样失败；
  ④ 脚本 `$ErrorActionPreference='Continue'` 把这两次失败**都吞掉**（且过滤只匹配
  `uploaded|error` 字样，异常文本不含该词）。
- **后果链（若未拦截）**：8 个 gate 全部到不了 1003 → `_local_finalize.ps1` 的
  「等 8 个 GPU gate」**白等 24 h 后超时** → 采集照跑 → `mvmoe/routefinder/cada/rrnco`
  4 个方法按 `GROUP_RANK` **静默回落到 v1 臂** → 总表在这 4 行混入旧批次来源，
  且只有 `check_arm_provenance.py` 会在**一天之后**才报出来。这与 `_local_finalize.ps1`
  自己注释里担心的风险完全一致，属于「已识别风险但防线本身有洞」。
- **修复**：新增 `CC_Compare/tools/_gpu3080_copyback.ps1`，并把编排第 4 步**委托**给它。
  新实现：① **增量**回传（每轮把 3080 上已存在的 gate 先搬走 —— 6 个非 rrnco gate 不再干等
  最后一个方法）；② `New-Item -ItemType Directory -Force` 先建本地目标父目录；
  ③ 上传后**从 1003 读回文件大小并要求逐字节相等**（"uploaded" ≠ "landed"）；
  ④ 打印明确的 `COPYBACK_PASS` / `COPYBACK_FAIL` 而不再吞异常。
- **验证**：两个 `.ps1` 均通过 PowerShell AST 解析（`PARSE_OK`，避免自动化单点因语法错误静默失效）；
  等待条件用真实命令实测（已完成的 `mvmoe_51015`/`routefinder_51015` → `True`，未完成的
  `rrnco_0` → `False`）；本轮已以 `pwsh-38` 启动增量回传（先搬 6 个，rrnco 的 2 个随后自动补）。
- 注：正在运行的 `pwsh-26` 已把旧脚本读入内存，其第 4 步仍会按旧逻辑失败（无害：它到不了 1003），
  实际回传由 `pwsh-39` 完成；修正后的编排脚本供后续重跑使用。
- **我自己在新脚本里又犯了一次同族错误（如实记录，已被首轮实测抓到）**：第一版
  `_gpu3080_copyback.ps1` 把 `$STAGE` 定义成**已含** `$EXT` 前缀（`"$EXT/results/a1_ext_formal2_3080"`），
  构造 3080 侧远端路径时又拼了一次 `$EXT` ⇒ 路径变成
  `…\C-VRP_…/C-VRP_…/results/…`，`Test-Path` 对 8 个臂全为 `False`，首轮打印
  `已回传 0/8`。**Fail-closed（什么都没搬错），但零产出** —— 说明「路径前缀重复」是这条链上的
  高频错误类型（缺陷 D/F/I 皆与路径/身份拼接有关）。已拆成 `$STAGE_REL`（相对扩展根，给 3080 远端用）
  与 `$STAGE`（相对仓库根，给本地/SFTP 用），并**静态演算**核对拼出的两条路径后再重启（`pwsh-39`）。

**缺陷 I 的后续（2026-10-13 21:0x 实测）—— 它不只是「冗余」，而是「破坏性」的，且我先前判断错了**

- **更正我先前的判断**：我曾在 `NEXT_STEPS` 里写「`pwsh-26` 内存里是旧版，它重跑第 4 步只会
  **重复上传**同样字节、**无害**」。**这是错的。** 旧代码里 `_sftp_get.py` 的远端路径是按
  **仓库根**解析的，而 `$REMOTE_OUT='results/a1_ext_formal2_3080'` 是相对**扩展根**的
  ⇒ 它去下载一个**不存在的远端路径**，于是先**创建了 0 字节的本地文件**（落在
  `<仓库根>/results/a1_ext_formal2_3080/…` 这棵**游离目录**里），再把这些 **0 字节**文件
  `Copy-Item` 到目标、并 `_sftp_put.py` **上传覆盖了 1003 上原本正确的 7 个 gate**（20:57）。
- **为何没被立刻发现**：`pwsh-39` 在更早的轮次里已把那 7 个臂标记为 verified（内存里的
  `$verified` 表），因此**不会再复检**它们；直到 `check_gpu_device.py` 读到 0 字节 JSON
  **抛异常崩溃**才暴露。⇒ 教训：**「已验证」是内存态、不等于磁盘态**；跨进程/跨时间的
  「已验证」必须落盘或用内容校验复核。
- **影响面（如实）**：1003 上 8 个 GPU gate 里的 **7 个**变成 0 字节，持续约 **7 分钟**
  （20:57–21:04）。若收口恰在此窗口运行，采集器会读空文件 → 该 4 个方法的行会失败/丢失。
  **未造成最终损失**（已修复并复核）。
- **修复**：重跑带逐字节校验的 `_gpu3080_copyback.ps1` ⇒ **`COPYBACK_PASS` 8/8**（21:04）；
  并**独立复核**：1003 端 `ls -l` 大小 26,765–27,016 B、`check_gpu_device.py`
  `PASS: 8 个臂齐全，且各自的实际 device 键都是 cuda`；游离目录 `<仓库根>/results/` 已删除
  （删除前记录到其中 8 个文件**全是 0 字节**，与上述根因吻合）。
- **顺带加固**：`check_gpu_device.py` 现在把「空文件/坏 JSON」归类为**明确 FAIL 并继续跑完**，
  不再抛 traceback 中断整条收口链（当时它会**在该门禁处直接崩掉**，后面的门禁连跑都跑不到）。
- **根因层加固（真正堵住这一类）**：事故的**使能条件**是「失败的下载会留下一个 0 字节本地文件」。
  `tools/_sftp_get.py` 现已：①下载**前**先 `stat` 远端，不存在就 `FAIL` 并**完全不碰本地**；
  ②下载后校验**本地大小 == 远端大小**；③任何异常都**清掉残留的本地文件**；
  ④成功时把大小打进日志（`downloaded … (N B)`）。⇒ 「0 字节假文件被当成真文件上传」
  这条路径**从工具层被封死**。已实测两条路径：正常文件 → 打印 `4569 B`；
  远端不存在 → `FAIL …（本地未改动）` 且**未创建任何本地文件**（连父目录也不建）。
- **全家族扫描（这次覆盖全部脚本类型，修正了 §13.3 开头那次只扫 `*.sh` 的不完整扫描）**：
  对所有 `*.ps1` / `*.sh` / `*.cmd|*.bat` grep `_sftp_get.py|_sftp_put.py`，逐个人工核对
  「脚本当前工作目录 / 路径基准」与「文件真实所在位置」是否一致。结论：**共 5 处真实调用**
  （`_rehearsal_prefinalize:41`、`_local_finalize:69`、`_gpu3080_orchestrate:42-43`、
  `_gpu3080_copyback:68/77`、`_baseline_watch:25`；`sh/cmd` 类脚本**不用**这两个助手），
  **全部为「仓库根相对」且与文件实际位置一致** ✅。原先唯一违规的那处（旧编排内联第 4 步的
  `_sftp_get.py $rel`）**已从文件里删除**（改为委托 `_gpu3080_copyback.ps1`）；
  加上 `_sftp_get.py` 自身的加固，**这一类「路径基准错 → 0 字节假文件 → 覆盖好数据」
  已由构造上封死**：即使再写错，也只会**响亮失败**并保持本地不动。
- 另：这再次印证 `pwsh-35 → pwsh-46` 那条教训（**改脚本不影响已在运行的实例**），
  且这次证明后果可能是**破坏性**而非仅仅冗余 —— 因此对任何仍在运行的旧实例，
  应当**主动终止**而不是假设它无害。

**缺陷 J：v2 批次的「正式裁决」判据结构性不可满足 —— 两个判定器都会永久报错**（2026-10-13 20:3x，**收口前的真实数据排练**中发现）

- **根因（源码级，已核实）**：6 个纯规则臂（`rule_myopic` / `rule_valuee20` / `rule_valuee30`）
  由 `run_a1_downstream_ablation.py` 产出，而该脚本**从不 import / 调用
  `pairing_identity_check`**（grep 该文件：0 处出现）⇒ 它们的 gate 里**既没有
  `_pairing_identity`，也没有 `identity`**。这是设计如此，不是缺陷。
- **但两处判据都写成「全部 formal2 臂都必须合格」**，于是**永远不可能成立**：
  1. `render_compare_table.py` 的标题 ⇒ 永远落在「重配对证据／正在补齐 N/N」分支，
     **v2 批次花十几小时换来的「落盘正式裁决」永远显示不出来**；
  2. `render_v2_summary.py` §4 的 #3「臂 source_stable=True」与 #4「落盘配对
     identity_verified」⇒ **永久 FAIL** ⇒ **总判定永久 FAIL**，哪怕批次完美无缺。
- **发现方式**：**收口前用真实数据把 collect→render 全链排演一遍**（服务端采集写
  `/tmp`、本地渲染进临时目录，**不碰交付物**），拿到「19/25 个正式臂已合格」这个数字后
  才顺着那 6 个异常臂摸到根因。若只在收口时跑一次，最可能的结果是：用户拿到一份写满 FAIL
  的 v2 摘要，并**误以为批次没做对** —— 而这恰恰是本批最贵的一次误判。
- **修复**：按「是否需要配对身份」把臂分两类核验，并把**豁免理由写进渲染出的标题里**
  （不静默豁免）：`_NO_PAIR_ID = {myopic, valuee20, valuee30}`；标题现在写的是
  「需要有配对身份的 **19** 个臂（全部 `identity_verified=true`）……另有 **6** 个纯规则臂
  **不参与上述裁决声明**，产出脚本从不调用 `pairing_identity_check`，故结构性无身份块」。
- **永久守卫**：`check_method_coverage.py` 新增 **E 项** —— 正式批次里**非纯规则**的臂
  一旦缺 `identity_verified` 立即 FAIL（**这正是缺陷 F 那一类**：学习型臂静默丢配对身份），
  纯规则臂按设计豁免。E 的 FAIL / PASS 两条路径均已用合成数据实测。
- **同一守卫再补 F 项（2026-10-14）**：**每个学习型方法都必须有 `participation/<name>.json`**，
  且凡通用 C2 只能报 `NOT_SUPPORTED` 的方法（纯 NumPy 实现）必须另有**专用 C2** 产物。
  动因是实测发现 `CO-enriched-ML` 是 13 个学习型方法里**唯一没有落盘参与度产物**的
  —— 台账写着「强」，但那是**散文断言**（根因：`COEnrichedMLProvider` 无 `_load()`，
  通用 harness 直接 `AttributeError`；已在 harness 侧改为「有 `_load` 才调」）。
  F 的**双向**均已实测：真实数据 `13/13 份在位；专用 C2 齐备` PASS；
  临时移除 `symnco.json` → 报 `F: … 缺 participation/<name>.json`；
  临时移除 `coenrich_c2_swap.json` → 报 `F: … 缺**专用** C2 产物`；两次测试后文件均已还原。
  ⇒ **「台账里写强」不再能替代产物**，这类缺口以后由机器发现，不靠人读。
- **顺带修正一处自相矛盾的标题**：原「正式裁决：已达成」分支同时写着
  「（双侧 `source_stable=True`）」和上一行的「基线 = `a1_c1rr2_reveal_40`（`source_stable=False`）」。
  现按事实拆成两件事：**臂侧**裁决（配对对象 = 各臂 `config.baseline_from`，即同源新基线
  `a1_c1fresh_reveal_40`，`source_stable=True`，落盘 `formal_adjudication=true` ×3）与
  **本表 Δ 列**（为与上一版逐位可比，另以 `a1_c1rr2_reveal_40` 重配对；该基线自身
  `source_stable=False` ⇒ Δ 列仍属「重配对证据」，且相对新基线系统性偏移 2–7）。
- **同时新增**逐臂 `baseline_from` 采集（读 gate `config.baseline_from`），使「每个臂到底
  跟谁配对的」可逐行核验，而不是只能靠叙述。

**缺陷 K：「20260926 开发集」的判定靠目录名 → 整个合法主模型 gate 被静默丢掉**
（2026-10-13 20:5x，为「用户改进主模型后新臂会不会进表」做排查时发现，**本次实际丢了一个臂**）

- **症状**：`render_compare_table.filter_main_model` 用
  `if '20260926' not in gate_path and 'online_20260926' not in gate_path: continue`
  判「是不是开发集」。而 `results/a1_a1_formal_40_1004_171907/gate.json` 的**目录名不含该串**，
  于是整 gate 被剔除 —— 尽管实测该文件里 `seeds={'train':20260925, **'gate':20260926**, ...}`、
  `source_stable=True`、contract 正是新基线的 `e25a12104b20c3ea`（用
  `tools/_probe_gate_meta.py` 打出来的）。
- **丢了什么**：该 gate 里 15 个主模型侧条目，含
  - **`a1_rollout`：p_c=0 util = 1312.6、超时 = 0.00/天** —— 比**任何**外部方法都高
    （最好外部 RouteFinder 1266.2），却从 §0 Ours 摘要与 §3 内部臂section里**完全消失**；
  - `a1_consensus`：42.35–46.40 次超时/天（远超 10 s 预算）。
- **为什么必须修，而不只是"补一刀"**：① 用户此刻正在改主模型，**新臂很可能落在新目录名下**，
  这正是同一类沉默失败；② 它和 `README §9` 自己写的规矩冲突 —— §9 说超时污染 gate 的数字
  「应标注为受污染、只作诊断」，也就是**要显示但带警告**，而实际做法是**直接丢**。
- **修复（三层）**：
  1. **按文件里记录的真实 seed 判**，不再靠目录名：`collect_compare_table.py` 为每个
     `main_model` 条目采 `dev_seed`（`seeds.dev` 或 `seeds.gate`）；
     `render_compare_table._is_dev_gate()` 优先用它，旧 dataset 无该字段时**回退**到目录名判定
     （行为兼容）。
  2. **把丢掉的东西说出来**：新增 `unlisted_main_model_arms()`，把「开发集里存在、
     但没进 Ours/内部臂节」的臂**连同 utility 与 `超时/天`** 渲染成 §0 顶部告警表，
     并对超时>0 的臂直接标 `← 超时污染，只能作诊断`；同时给出「若是新臂，请加入
     `MAIN_MODEL_ARMS`」的可执行指引。
  3. **`a1_rollout` 加进 `MAIN_MODEL_ARMS`**（0 超时、合法 ⇒ 应当可见）；
     **`a1_consensus` 有意不加**（超时污染），但它出现在上面那张告警表里 ⇒ **不是静默排除**。
- **验证**：
  - 合成数据：给 `a1_a1_formal_40_*` 条目补 `dev_seed` 后重渲染 → 告警表如期出现，
    6 行数字正确、`a1_consensus` 三行带「超时污染」标记；`a1_rollout` 三档进入 §0/§3
    （p_c=0 **1312.6 [1292.0, 1332.2]、超时 0.00、source_stable=True**）。
  - 真实数据：重新采集后有 **199/199** 个 `main_model` 条目带 `dev_seed`，
    `a1_a1_formal_40_1004_171907` → `20260926`（15 条），`a1_rollout` → `20260926`（3 条）。
  ⇒ 收口重采集后，这个臂会真正进入交付表，而不是继续悄悄缺席。

**缺陷 L（报告口径；底层证据一直是对的）：收口清单里「正式裁决」那一列**恒为 `None`****
（2026-10-13 20:5x，为核收口链最后一步时发现）

- 收口脚本 `_local_finalize.ps1` 末尾打印 `repaired_vs_baseline.cond_hist.formal_adjudication`，
  而 `repaired_vs_baseline` 由采集器 `_pair()` 产出，**从不包含 `formal_adjudication` 键**
  ⇒ 该列**恒为 `None`**。人读这张清单会得出「这些臂没有正式裁决」的**相反**结论 ——
  而 v2 批次的目的恰恰是拿到 `formal_adjudication=true`。
- 修复：①采集器逐臂新增 `gate_adjudication`（读 gate 内 `<arm>_minus_cond_hist` 的
  `formal_adjudication` / `identity_verified` / `n_paired`；它与「本表另以 `--baseline` 重算的差」
  **是两个口径**）；②收口脚本改打印该字段，并把重配对那侧的 `re_paired_adjudicable` **分列**。
- 实测（真实数据，27 个正式臂）：`gate_formal_adjudication=True` **21 个**、`None` **6 个**
  —— 那 6 个正是纯规则臂（与缺陷 J 同源，结构性没有该块）；学习型臂全部
  `identity_verified=True`、`n_paired=40`。
- 性质说明：**底层证据一直是对的**（gate 里本来就写着 `formal_adjudication=true`），
  属于「输出层把对的证据说反了」，故单列为缺陷 L 而非并入 J。

**缺陷 M：`_sftp_put.py` 把 Windows 反斜杠当成远端文件名的一部分 —— 上传「成功」但目标文件没变**
（2026-10-13 22:0x，上传收口守护脚本时当场踩到并当场修好）

- 触发：在 PowerShell 里用很自然的方式传相对路径
  `python tools/_sftp_put.py CC_Compare\tools\_xf2_guard.sh`。
  `_sftp_put.py` 原本把 `rel` **原样**拼到远端根（`remote = remote_base + rel`），
  于是远端**新建了一个名字里含反斜杠的单文件** `CC_Compare\tools\_xf2_guard.sh`，
  而真正的 `CC_Compare/tools/_xf2_guard.sh` **根本没被写**。
  实测 `ls CC_Compare/tools/_xf2_guard.sh` → `No such file or directory`，
  而 `ls -l` 在**仓库根**列出了那个怪名字文件（3025 B）。
- 危害等级：**静默失效**（比报错更坏）。工具打印 `uploaded ...`，退出码 0；
  若目标文件已存在，会让人以为「新版本已生效」，实际跑的还是旧版本 ——
  这正是缺陷 A/H/I 那一族「路径拼接」错误的又一个入口，只不过这次在**本地↔远端通道**上。
  对本次的具体后果只是多了一个垃圾文件（新脚本，没有旧版本可被顶掉），但同类调用一旦
  用在**已存在的关键文件**（如 `ordering_providers.py`、批次脚本）上就会静默回滚。
- 修复：`tools/_sftp_put.py` 与 `tools/_sftp_get.py` 各加 `_rel()`，
  把 `\` 统一成 `/` 并去掉前导 `/`，**local 与 remote 两侧都用归一化后的路径**。
  远端残留的怪名字文件已 `rm`（实测 `STRAY_REMOVED`，随后 `ls` 确认真路径 3025 B 在位）。
- 教训（与「平台无关」那几条并列）：**凡是从 Windows shell 传相对路径给远端工具，
  都必须在工具内部归一化，而不是指望调用方写对分隔符**。

**同一轮的残余风险加固（缺陷 H 的未尽之处）：`_xf2_guard.sh`**

- 缺陷 H 修好了「相对路径导致子腿秒退」，但**没修「秒退之后编排怎么办」**：
  `wait_for` 只认 MARK 里的 `ALLDONE`，而 `_formal_v2_coenrich.sh` / `_formal_v2_mapt.sh`
  都在开头 `: > "$MARK"`、**只有跑到最后一行才追加 `ALLDONE`**
  ⇒ 子腿在任何一条 FATAL 早退（缺基线/缺权重/provider 未接线/缺 `PCHGS` 可执行），
  MARK 就永远没有 `ALLDONE`，`wait_for` 会白等满配额：**CO-enriched 40 h、MAPT 10 h**，
  而且日志看起来像「还在跑」。
- 加固：新增 `CC_Compare/tools/_xf2_guard.sh`（tmux 会话 `xf2_guard`，5 min 一次），
  只在**三项同时成立**时才补 `ALLDONE`（tmux 会话已消失 **且** `pgrep -f "[p]rovider <name>"`
  无进程 **且** MARK 无 `ALLDONE`），并写入 `GUARD_FORWARDED_NO_ALLDONE` 留痕；
  另外若 `xf2_after` 会话消失而 after.log 无「收口完成」则重新拉起 `after.sh`（其本身幂等，
  已产出 `gate.json` 的臂会 SKIP）。**宁可产出「缺该腿」的表并大声记录，也不要静默等 40 h。**

**缺陷 N：本地 `_srv.py` 命令里的 `$` 会被本地 shell 吃掉 —— 远端脚本「跑了但变量全是空」**
（2026-10-13 23:5x，盯进度时当场发现）

- 现象：`python tools/_srv.py 'bash -lc "for d in A B; do echo $d; done"'` 打印出**两个空标签**
  （`: 0 files`），而不是 `A:` / `B:`。**命令看起来正常执行，输出也在，只是变量被替换成空**。
- 对照实测（同一轮，两行结论）：

  | 写法 | 远端 `echo val=$X` 的结果 |
  |---|---|
  | 外层**单引号**：`'bash -lc "X=abc; echo val=$X"'` | `val=`（**空**，仍被吃掉） |
  | 外层**双引号 + 反引号转义**：`"bash -lc 'X=def; echo val=\`$X'"` | `val=def` ✅ |

  ⇒ 关键事实：**即使本地写成单引号，`$` 也到不了远端**（本地这一层会先做一次展开/重解析）。
  唯一可靠写法是**把 `$` 反引号转义**（`` `$X ``），或者**根本不写远端 shell 变量**。
- 危害等级：**静默出错**。`for d in ...; do echo $d; done` 只是丢标签；
  但同样的机制会把 `F=$src; cp $F $dst` 变成 **`cp ""`**、把
  `for f in *.done; do cat $f; done` 变成 **`cat`（无参数 → 读 stdin 永久挂起）**。
  后者已真实发生：一次状态探测因此**挂住整个前台命令**直到超时，看起来像「服务器很慢」。
- 修复/规避（本批之后的硬规矩）：
  1. **远端 shell 变量一律不用**；要循环就显式展开每一项，或
  2. 需要 `$` 时**反引号转义**；或
  3. **>2 行或含变量的远端逻辑，一律写成 `.sh`/`.ps1` 上传后执行**（`xf2_guard.sh` 就是这么做的），
     既躲开三层引号，也留下可复算的脚本本体。
- 与缺陷 A/H/I/M 同族：**全部是「跨层拼接/转义」在中间层被吃掉**，
  且**都以「看起来成功」的形式出现**。凡跨层传字符串，都要有一个「断言实际到达了什么」的检查。

**缺陷 O：我新写的收口后置门禁把**通过**的检查报成 FAIL —— 纯粹是控制台编码**
（2026-10-14 04:1x，收口后第一次跑 `_post_finalize_check.ps1` 时）

- 现象：汇总打印 `POST_FINALIZE_CHECKS_FAIL: docprose, docrefs`，但**同一批命令我手动跑时全绿**。
  明细里其实是两个**编码异常**，不是判定失败：
  - `_doc_prose_check.py` → 子进程 `print_readme_section2.py` 的输出按 utf-8 解码时报
    `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbb`（子进程 stdout 实际是 **GBK**）；
  - `_check_doc_refs.py` → 自己打印 `✅` 时报
    `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'`。
- 根因：**中文 Windows 的控制台默认编码是 GBK**。我手动跑时习惯性先设了
  `$env:PYTHONIOENCODING="utf-8"`，所以一直没暴露；而后置门禁脚本**没有设**，
  于是两个**本来通过**的门禁被记成 rc=1。
- 危害：**假 FAIL 与假 PASS 同样危险** —— 它会让人去"修"根本没坏的东西，
  或者干脆不再相信这套门禁。这属于「输出层把对的结论说反了」（同类：缺陷 L）。
- 修复（三处，都已实测）：
  1. `_post_finalize_check.ps1` 顶部设 `$env:PYTHONIOENCODING='utf-8'`（覆盖它启动的所有子进程）；
  2. `_doc_prose_check.py` 给 `subprocess.run` 显式传 `env`（`PYTHONIOENCODING=utf-8`）**并**
     重绑自己的 stdout；
  3. `_check_doc_refs.py` 重绑自己的 stdout。
- **验证方式（关键）**：不是"设好环境再跑一次就完"，而是**在故障条件下**验证 ——
  先 `Remove-Item Env:\PYTHONIOENCODING` 清掉变量再跑，两条都 rc=0；
  最后 `_post_finalize_check.ps1 -SkipWait`（新加的开关，等价 `_local_finalize.ps1 -SkipWait`
  的约定）打印 **`POST_FINALIZE_CHECKS_ALL_PASS`**。
- 教训：**凡是要产出"通过/失败"判定的脚本，必须在**没有环境变量加持**的条件下也自洽**；
  只在自己习惯的环境里跑通，等于没测。

### 13.4 追加腿（CO-enriched / MAPT）的前置预检与一条**不可逆操作禁令**

**（a）基线污染判定 = PASS**：新基线 9/9 档全部通过（超时全面低于 c1rr2：cond_hist `p_c=0`
0.80 vs 阈值 7.04），身份与 c1rr2 逐字节同源。完整对账见 **`v2_baseline_verdict.md`**，其中记录
两条必须随总表读的事实：均值稳定但**逐日不可复现**（`max|Δday|` 达 233.6、`|Δmean| ≤ 6.81`，
根因是 10 s 硬预算下的耗时抖动），以及新基线略优（基线平移 +2.6~+6.8）。

**（b）追加腿 provider 预检 = PASS**（工具 `tools/_preflight_extra_providers.py`，在 1003 上用
**将要运行它们的 `cc_compare`** 实跑）：

| provider | 目录 | 权重 | import | 构造 | `.order()` | 结论 |
|---|---|---|---|---|---|---|
| `coenrich` | ✅ | ✅ `artifacts_a1` | ✅ | ✅ | ✅ | 可跑 |
| `mapt` | ✅ | ✅ `mapt_a1.pt` | ✅ | ✅ | ✅ | 可跑 |

⇒ 不会重演 cpuC（缺 ortools/pyvrp）与 3080 cada（缺 entmax）那类「跑到一半才发现缺依赖」。
该预检**故意绕开 `ordering_providers._REGISTRY`** 直接按路径 import，因为**在 F1 四条腿还在跑时
改 `ordering_providers.py` 会让在跑臂的 `source_stable` 翻成 False**
（臂在启动与结束各做一次 provider 源码 hash）。`_formal_v2_coenrich.sh` 把这个前提写在第 6 行，
`xf2_after` 也是在四腿 ALLDONE 之后才接线 —— 这个先后顺序是**协议要求，不是优化**。

**（c）⚠️ 不可逆操作禁令：绝不要把本地的 `ordering_providers.py` 上传到 1003。**
1003 上的该文件**已经**接好线（`coenrich` / `mapt` 各出现 2 次，registry 13 行，
md5 `7f96960ed774dc5d05c0cfe5b003620a`），而**本地副本仍是未接线的 10 项版本**。
上传本地副本会 (1) 抹掉接线、(2) 在臂运行期改变该文件 →
**把在跑/已跑臂的 `source_stable` 翻成 False、毁掉整批的身份链**。
`wire_extra_providers.py` 是幂等的（`already wired; nothing to do`，不写文件），所以
`xf2_after` 的接线步骤是安全的 no-op。

