# 独立只读复核：CC_Compare 对比方法线的资格与公平性
**日期**：2026-10-14 · **性质**：只读审计，**未修改任何 gate、冻结树或已封存结果**
**配套产物**：`BASELINE_QUALIFICATION_MATRIX_2026-10-14.json`（逐臂独立资格矩阵）、
`METHOD_ADAPTATION_BOUNDARY_2026-10-14.md`（逐方法适配边界）

> **方法说明（为什么叫"独立"）**：本复核**不采信**运行器自报的 `formal_adjudication`。
> 每一条完整性判据都从 gate 内的**原始逐日记录 `per_day`** 重新计算，再与运行器汇总并排，
> 分歧显式列出。复核脚本：`CC_Compare/tools/audit_qualification_matrix.py`（只读）。
>
> **审计边界（如实声明）**：本轮**读取了源码、gate 原始记录与交付物**；
> **没有**重放任何外部方法的 checkpoint，**没有**逐个核对 3080 的运行日志原始字节，
> **没有**重新执行 PyVRP 的历史故障批次。凡属这三类的结论，下面都标注为"未独立复核"。

---

## 一、结论摘要

| # | 审查方提出的问题 | 本复核判定 | 依据 |
|---|---|---|---|
| 1 | 11 个 provider ≠ 11 个完整原生算法 | **成立**（且非缺陷，是受控实验设计） | `OrderingAcceptReplanner` 文档串 + `run_a1_external_accept.py:120-130` |
| 2 | 三档惩罚是同一轨迹重评分，策略不随 `p_c` 改变 | **成立**（最严重的一条） | `run_a1_external_accept.py:143-167`；`rrnco_accept_replanner.py:48-49` |
| 3 | 汇总器混用两个基线身份 | **部分成立**（Δ 列确为另一基线重配对；但交付物已分开标注，未继承正式裁决） | `collect_compare_table.py`；`render_compare_table.py` 表头 |
| 4 | PyVRP 曾"全拒单但形式检查通过" | **成立**（历史故障），**但当前已修复**；证据治理有缺口 | `_formal_v2_pyvrp_fix.sh:49`；缺陷 D |
| 5 | `formal_adjudication` 判据过宽 | **成立**（源码逐字确认） | `run_a1_external_accept.py:404` |
| 6 | 身份清单未覆盖 checkpoint SHA 与依赖闭包 | **成立**（矩阵逐臂确认 `False`） | `run_a1_external_accept.py:57-104`；`run_identity.py:22-33` |
| 7 | 外部臂与 MaskCO 线 `future_policy` 不同 | **成立**（源码 + 数据双证） | `scenario_saa.py:268`、`run_i3_online_compare.py:116`、`dataset.json` |
| 8 | 总表按目录名选臂 | **成立**（设计如此） | `render_compare_table.py` `GROUP_RANK` |
| 附 | 仓库 `_REGISTRY` 无 `coenrich` | **成立**（仓库与 1003 **不一致**） | `ordering_providers.py:15-33`（本地）vs 1003 实测第 33–36 行 |

**一处需要更正的引用**：审查把「`p_c=0` 效用 0 / 主档 −1943.12」记为 PyVRP 故障批次的数值。
该数值是**全拒单指纹**，且它**恰好等于**交付表里 `rule: value/energy λ=3` 的数值
（λ=3 全拒单是已登记的预期行为）。两者指纹相同、无法从数值上区分。
当前交付的 PyVRP 值是 **909.22 / −569.78**（已修复批次）。
原故障目录已被 `_formal_v2_pyvrp_fix.sh:49` 的 `rm -rf` 删除 ⇒ **本轮无法独立复核故障批次的原始数值**。

---

## 二、逐条证据

### 问题 1：适配语义 —— 成立

`OrderingAcceptReplanner` 的类文档串（`rrnco_accept_replanner.py:42-45`）自述为
「通用『学习排序 + 全覆盖接单』replanner：任意 `provider.order(sp)` → 经冻结 dcc_rh_v4 协调器
打包成多车 suffix，全覆盖 + 真实 C0 `certify ≤ B` 才接单」。

⇒ 被**替换**的是：车辆分配、路线拼装、接/拒判定、C0 认证（全部由共享协调器承担）；
被**保留**的是：各方法自己的排序/打分机制。
**因此正确命名是「model-guided ordering（模型引导的共同在线接拒框架）」**，
不是各原论文完整原生机制的直接竞争。这是**受控设计**，但论文必须写明保留/替换边界
（详见 `METHOD_ADAPTATION_BOUNDARY_2026-10-14.md`）。

### 问题 2：三档惩罚是重评分，不是重决策 —— 成立（最重要）

```python
# run_a1_external_accept.py:143-167
for i in range(days):
    traces, _ = env.run(i)                              # ← 每天只跑一次
    acc, rej = set(rp._accepted), set(rp._rejected)     # ← 只有一组接/拒决定
    for pname in penalty_names:
        ev = evaluate_trace_a1(..., acc, rej, B, PENALTIES[pname])   # ← 只换评分函数
```

且 `OrderingAcceptReplanner.__init__` 的签名（`rrnco_accept_replanner.py:48-49`）是
`(budget, capacity, booking_horizon, contract, cooling_share, provider, time_limit)`
—— **根本没有 `p_c` 形参**；`run_a1_external_accept.py:127-130` 的构造调用也没有传。

**⇒ 策略从未见过 `p_c`。** 精确表述（避免反向夸大）：

- **代码能力**：单次运行内，`for pname in penalty_names` 会用**同一组 `acc/rej`** 对每个罚分档
  重新评分 —— 若一次跑多档，多档必然共享同一条轨迹。
- **本批实际做法**：各腿**每档单独调用一次**运行器（`--penalty "p_c=0"` / `--penalty "p_c=(5,10,15)"`
  各一次，输出到不同目录）⇒ 交付表的两列来自**两次独立运行**，不是同一条轨迹。
- **但两次运行用的是同一个策略**：`run_arm` 构造 replanner 时不传任何罚分
  （`run_a1_external_accept.py:127-130`），`OrderingAcceptReplanner.__init__` 也没有该形参
  （`rrnco_accept_replanner.py:48-49`）。
- ⇒ 档间差异**不可能**是"策略对惩罚的响应"；它是**不同评分函数作用在同一个策略上**。
  两次运行的决策是否逐日一致，取决于 provider 的确定性 —— 而 `LIH` / `DeepACO` 已被登记为
  **跨次不可复现**（`读表注意与已知限定.md` §2.3），因此这两臂的档间差异还可能混入运行噪声。

因此：

- **不能**声称「外部方法在不同拒单惩罚档位下调整了动态接单行为」；
- `p_c=(10,20,30)` 档同理（交付表里只有 OR-Tools/PyVRP 有该档，也是同一性质）；
- 主档列的**唯一**信息是「同一决策在合同损失下的净效用」，**不含**任何策略响应；
- 若要真正的策略敏感性，必须让**决策规则**接收 `p_c` 并**逐档重跑** —— 这属于**新比较协议**，
  不得对既有结果事后追认。

（注：README §2 读点现有措辞「主档回答的是『有合同损失时的净效用』」**本身没有过度声称**；
但两档并排陈列会诱导误读，故已在交付表 §1 加显式脚注，见 §四。）

### 问题 3：汇总器的两个基线身份 —— 部分成立

- 臂侧配对基线：各臂 gate 的 `config.baseline_from`（v2 = 同源新基线 `a1_c1fresh_reveal_40`，
  `source_stable=True`，落盘 `formal_adjudication=true`）。
- 交付表 Δ 列：另以 `a1_c1rr2_reveal_40` 做 per-day 重配对（该基线 `source_stable=False`）。

**判定"部分成立"的理由**：交付物**确实**把两者分开了 —— 表头与脚注明确写着
Δ 列是「相对 c1rr2 的重配对」，且**没有**把另一基线算出的差值继承为 `formal_adjudication`
（`formal_adjudication` 只出现在 gate 内的 `<arm>_minus_cond_hist` 块，由臂侧配对对象决定）。
⇒ 审查担心的「继承为正式结论」**在交付物层面没有发生**；但「同一张表里存在两种配对口径」
这一事实**成立**，且必须在读表时始终带着脚注。这属于**已登记限定**
（`读表注意与已知限定.md` §2.8、README §3 表头），不是新缺口。

### 问题 4：PyVRP 历史故障 —— 成立（且证据治理确有缺口）

- 故障机制已由项目登记（缺陷 D）：PyVRP 0.14.0 与 `_solve_pyvrp()` 的 0.11.3 API 不兼容，
  `TypeError` 使轨迹退化为**全拒单**，而 `hard_feasible_rate=1.0`、`mean_timeouts=0`、
  退出码 0、身份检查全部通过 —— **形式检查完全无法发现**。
- 修复腿用 `MASKCO_env`（pyvrp 0.11.3）重跑，实测 `p_c=0` **909.22** / 主档 **−569.78**，
  与 v1 逐位一致；本次独立矩阵确认 `a1_ext_a05e` 的 pyvrp/ortools 臂 **`Q_formal_core` 通过**
  （非退化、身份合格）。
- **证据治理缺口（审查方正确）**：`_formal_v2_pyvrp_fix.sh:49` 用 `rm -rf "$OUT/pyvrp"`
  就地重建 ⇒ **原故障目录已不存在**，无法再独立复核故障批次的原始 gate 与退出日志。
  ⇒ 建议（未来）：**新目录恢复 + 保留旧失败目录**，并把失败批次标 `solver_invalid`。

### 问题 5：`formal_adjudication` 判据过宽 —— 成立（源码逐字确认）

```python
# run_a1_external_accept.py:400-405
for pname in penalty_names:
    blk = results[gate_seed].get(pname, {})
    for key in list(blk):
        if key.startswith('%s_accept_minus_' % args.solver):
            blk[key]['formal_adjudication'] = bool(pairing['identity_verified'])
```

即 `formal_adjudication ≡ identity_verified`。协议要求的是
`Q_formal = Q_identity ∧ Q_days ∧ Q_finite ∧ Q_hard ∧ Q_method`。
运行器**另行**算了 `adjudicable` / `hard_feasible_rate` / `finite_n` / `missing_day_ids`，
但**没有**与 `formal_adjudication` 合取 ⇒ 「身份合法」被读成「实验正式合格」。
**本轮未改写任何既有字段**（审查方也明确要求不得事后改写）；新门见 §五。

### 问题 6：身份不含 checkpoint SHA、不含 provider 依赖闭包 —— 成立

`_run_identity()`（`run_a1_external_accept.py:57-82`）的全部字段为：
`solver / time_limit / solution_limit / pyvrp_max_iterations / shared_config /
budget_B / cooling_share / source_sha256 / source_end_sha256 / source_stable /
shared_source_sha256 / contract_sha256 / data_sha256 / data_meta_sha256 / seeds`。

独立矩阵对 **62 个方法臂逐一确认**：`identity_has_checkpoint_sha256 = False`（**全部为假**）。

封存文件集 = `SOURCE_FILES_EXTERNAL`（10 个，`run_identity.py:22-33`，**无任何 checkpoint**）
`+ _ordering_seal_files(args)`（`run_a1_external_accept.py:89-104`）＝
`rrnco_accept_replanner.py`、`ordering_providers.py`，再加
（rrnco）`rrnco_guided_adapter.py`/`rrnco_backend.py` 或（ordering）单个 provider 文件。

而 `rrnco_guided_adapter.py:21-24` 实际 import 了
`coordinator`、`pickup_certificate`、`preference`、`subproblem`
—— **这四个都不在封存清单里**（审查方点名正确）。
⇒ 结论：**`source_stable=True` 不能证明模型权重、完整外部库及其依赖未变。**

**未独立复核**：审查提到的 `RouteFinder load_from_checkpoint(..., strict=False)` 的缺失/多余键
清单，本轮**没有**实际加载检查，因此不对其影响下结论。

### 问题 7：`future_policy` 不同 —— 成立（源码 + 数据双证）

| 证据 | 内容 |
|---|---|
| `scenario_saa.py:268` | `future_policy='reveal'` 是**默认值** |
| `_formal_v2_launch.sh:43` | 外部臂的 C1 基线以 `--future-policy reveal` 生成 |
| `run_a1_external_accept.py` | **全仓 grep 无 `future_policy`** ⇒ 外部臂的 gate **根本不记录**该字段 |
| `run_i3_online_compare.py:116` | 硬编码 `'future_policy': 'density'` |
| `dataset.json`（本轮实测） | `a1_c1r_reveal_40`/`a1_c1rr2_reveal_40` = **reveal**；**全部** `a1_a3_online_density_*` 与 `i3_online_recovery_1009_v1` = **density** |

⇒ 交付表 §1 里，**Ours 两行运行在 density 配置下，而 19 个对比方法的配对基线是 reveal 配置**。
这与 A2 证据（`future_policy` 会显著改变 `cond_hist−uncond_hist` 相对差）一致。
**因此 §1 的效用排序不能直接读成"同协议方法优势"**；必须按审查建议分成
「系统能力对照」与「方法归因对照」两类。**本轮已把该限定写进交付表脚注**（见 §四）。

### 问题 8：按目录名选臂 —— 成立（设计如此）

`render_compare_table.py` 的 `GROUP_RANK = {'a1_ext_formal2': 0, 'a1_ext_a05e': 1,
'a1_ext_extra': 2}`，按目录组优先级选臂；`collect_compare_table.py` 也按目录组名/`_f` 后缀
标记 `formal`。⇒ 只要某结果落在更高优先级目录，就可能压过更早但更健康的结果。
**本轮未改该逻辑**（改它会影响已交付表格）；建议的未来形态见 §五。

### 附：仓库 `_REGISTRY` 与 1003 不一致 —— 成立

- **本仓库** `scripts/evaluation/ordering_providers.py:15-33` 的 `_REGISTRY` 只有 **10 项**
  （routefinder / cada / mvmoe / pomo / attention / symnco / omnivrp / sgbs / deepaco / lih）
  —— **没有 `coenrich`，没有 `mapt`**。
- **1003** 上同一文件第 **33–36 行**确有 `coenrich` 与 `mapt`（本会话早前 grep 实测）。

⇒ **仓库的默认入口无法启动 CO-enriched-ML / MAPT**，而它们的结果**确实**在交付表里。
⇒ 接线只存在于服务器、未回写版本库 ⇒ 「文档说已接入」与「仓库代码可启动」目前**不能互相证明**。
**处理建议（不可逆操作禁令仍然有效）**：**绝不**把本地 `ordering_providers.py` 上传覆盖 1003
（会破坏在跑臂的 `source_stable`）；应反向操作 —— 把 **1003 上的实际文件**取回仓库并记录其
SHA-256，作为「该批次实际使用的注册表」的证据。

---

## 三、独立资格矩阵结果（本轮实测）

**扫描**：58 个 gate / **62 个方法臂** + **186 个内部基线臂**。

| 批次 | `Q_formal_core` 通过 | 说明 |
|---|---|---|
| `a1_ext_formal2`（v2 正式批次） | **38 / 40** | 未过的 **2** 个是 `rule_valuee30` 两档 —— **λ=3 全拒单（已登记的预期行为）**，非故障 |
| `a1_ext_a05e`（旧求解器臂） | 通过 | 含 PyVRP 修复后批次 |
| `a1_ext_extra`（旧 `_f` 批次） | **0 / 22** | 全部 veto 于 `Q_identity`（`baseline_identity_missing`）—— 这正是当初重跑 v2 的原因 |
| **合计** | **38 / 62** | |

`Q_formal_strict`（额外要求 `超时/天 == 0`）= **32 / 62**。差额 **6** 个臂正是已登记的受 CPU 争用臂：

| 臂 | 档 | `超时/天` | `最大单决策(s)` |
|---|---|---:|---:|
| `lih_0` | p_c=0 | 25.35 | 54.35 |
| `lih_51015` | 主档 | **54.225** | 50.79 |
| `sgbs_0` | p_c=0 | 12.025 | 16.04 |
| `sgbs_51015` | 主档 | 9.075 | 15.81 |
| `symnco_0` | p_c=0 | 0.075 | 17.40 |
| `symnco_51015` | 主档 | 0.225 | 21.50 |

**判读**：v2 正式批次的资格状况比审查方担心的要好 —— 除 6 个**已登记**的争用臂与 2 个
**设计如此**的全拒单规则臂外，**38/40 通过独立复算**。
但 `Q_identity` 只覆盖源码/合同/数据，**不覆盖 checkpoint 与依赖闭包**（问题 6），
所以这 38 个臂的"身份合格"**强于**运行器的 `formal_adjudication`，**弱于**协议要求的完整身份。

### 3.1 ⚠️ 交付表里的 **Ours 两行**（2026-10-14 补充扫描，本复核最严重的发现之一）

矩阵原先只覆盖 gate 结构的对比臂，**Ours 侧是空白**；补扫后（14 个交付 Ours 记录）结果如下：

| Ours 行 | 档 | 臂 | 配对身份 | `超时/天` | `Q_formal_core` |
|---|---|---|---|---:|---|
| **A3（掩码重构·预训练权重）** | p_c=0 | maskco | **False**（`shared_source_sha256_mismatch`） | **19.825** | **False** |
| A3 | 主档 | maskco | **False** | **18.85** | **False** |
| A3 | p_c=(10,20,30) | maskco | **False** | 4.95 | False |
| **I3（决策感知）** | 主档 | maskco | True | **37.325** | True |
| I3 | 主档 | warmstart | True | 36.825 | True |
| I3 | 主档 | reconstruction_control | True | 44.175 | True |

**两条必须随主模型结果一起披露的事实**：

1. **A3 的配对身份未通过**，原因是 `shared_source_sha256_mismatch` ——
   即 A3 gate 与 v2 基线 `a1_c1fresh_reveal_40` **不是同一版共享计算文件**。
   注意：它自己的 `source_stable=True`（启动==结束）**不能**替代跨运行的同源性。
   ⚠️ **A3 正是交付表 §1 里 p_c=0 最强的那一行（1577.9）**。
2. **两行 Ours 的 `超时/天` 远高于全部外部臂**：A3 19.8 / I3 37.3，
   而 **19 个外部方法正式臂全部为 `0.0`**。⇒ 主模型结果行**同样**受 CPU 争用影响，
   引用时必须与外部臂一样带超时限定。**这一点此前在表里没有披露**（† 只打给了 3 个外部方法）。

⇒ 交付表 §1 已**数据驱动地**加了一条脚注，每次渲染自动带上这两个事实
（`render_compare_table.py::md_table_notes` 第 3 条 + Excel 表下第 1 条）。

⚠️ 主模型 gate **不在该块产出 `solver_meta`**，故 `Q_method_health` 对 Ours 记 `None`（不适用）
而非 False —— 避免又一次"结构性不适用被算成失败"。其可信度**不由**该方法项背书。

**本轮未纳入矩阵**：主模型（Ours）臂。它们来自 `main_model` schema（非 gate 结构），
需要单独一轮抽样，故**尚未**给出 Ours 侧的独立资格判定。

---

## 四、本轮对交付物做的限定（只加限定，不改数字）

1. **交付表 §1 加脚注**：主档列 = 同一策略的惩罚**重评分**，**不是**策略对惩罚的响应（问题 2）。
2. **交付表 §1 加脚注**：Ours 两行运行于 `future_policy=density`，19 个对比方法的配对基线为
   `reveal` ⇒ 效用排序**不构成同协议方法归因**（问题 7）。
3. **交付表 §1 保留**逐臂 `超时/天` 列与 † 标记（6 个争用臂）。

## 五、建议的未来资格门（**只用于新版本，不回改既有 gate**）

```
Q_formal = Q_identity ∧ Q_days ∧ Q_finite ∧ Q_hard ∧ Q_nocrash ∧ Q_nondegen ∧ Q_method
Q_identity  ← 现字段 + checkpoint_sha256 + provider_dependency_sha256（闭包）
Q_method    ← 外部求解器健康（真实调用核心入口、非全拒单、非异常退化）+ 模型参与证据
```
并要求**反例测试**：源码匹配但 39/40 天 / 某日 hard=False / 求解器全抛异常 / checkpoint 身份缺失
—— 四类都必须被拒绝。**保留**原 `source_stable` / `formal_adjudication` 字段语义不变，
另增 `dependency_closure_verified` 等新字段，**不悄悄改旧字段含义**。

## 六、本复核未做的事（如实声明）

- **未**重放任何外部方法 checkpoint；**未**重新执行 PyVRP 故障批次（目录已被删除）。
- **未**逐个核对 3080 的原始运行日志字节（本会话早前已由 `_gpu3080_copyback.ps1` 做逐字节校验、
  `check_gpu_device.py` 确认 8 臂 `device=cuda`；但那是**编排侧**证据，不是独立审计）。
- **未**给出 Ours 侧的独立资格判定（见 §三末）。
- **未**修改任何 gate、冻结树或已交付数字。
