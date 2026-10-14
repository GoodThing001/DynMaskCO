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

## 〇、**2026-10-14 更正附注（统一口径；不覆盖、不删除原文）**

用户在远端核查中指出本文件内部有三处口径矛盾。**原文一律保留**，此处给出**唯一有效口径**；
引用本报告时以本节 + §三 + `BASELINE_QUALIFICATION_MATRIX_2026-10-14.json` 为准。

| 位置 | 原文（保留） | **正确口径** |
|---|---|---|
| 正文 L239 | 「而 **19 个外部方法正式臂全部为 `0.0`**」 | **错误**。正确说法：**除已登记的 6 个受 CPU 争用臂（`lih`×2 / `sgbs`×2 / `symnco`×2）之外**，其余对比方法臂 `超时/天 = 0.0`。逐臂数据见本文件 §三与资格矩阵。 |
| 正文 L248 | 「**本轮未纳入矩阵**：主模型（Ours）臂…」 | **已过时**。§3.1 已完成 Ours 补扫（14 条交付 Ours 记录）。该句只描述**当时**的范围。 |
| 正文 L203 | 「`Q_formal_strict` … 差额 **6** 个臂正是已登记的受 CPU 争用臂」 | **与 §3.1 一致，正确**；与 L239 冲突的是 L239，不是 L203。 |

**另附一条本轮新测（不属于原文错误，属新增证据）**：D 段日级复算（见下）在 52 个配对块上
**均值与 day_id 集合全部逐位一致**，但 **4 个主模型块**的 bootstrap CI 边界偏差 **5.06–9.96**，
而「CI 是否排除 0」结论一致 ⇒ 主模型 driver 与采集器复刻的 `_stat` **不是同一个 bootstrap 生产者**，
引用主模型 CI 时须知这一点（结论未受影响）。

> ⚠️ **上述推断已在 §〇.1 第 1 条中被推翻（2026-10-14 第二轮）**：CI 对不上的真实原因**不是**
> 生产者不同，而是**取错了 gate 的 bootstrap 种子**（回落到了 `generated_from.seed`）。
> 「不是同一个生产者」这一说法**作废**；`_stat` 实为同一生产者。**数值结论不变**，
> 且修复后 52 个块的 CI **全部逐位复现**（④ `CI_exact_reproduction=True`）。

> D 段（2026-10-14 新增，由 `CC_Compare/tools/check_pairing_identity.py` 提供）在**持有原始 gate** 的
> 1003 上实测 **PASS**：`A` 52/52 恒等式、`B` 隐含对照（I3 = **600.997**）、`C` 0 条跨协议混用、
> `D` 52 个块 day_id 集合相同且均值逐位复算一致。

---

## 〇.1 第二轮复核（2026-10-14 晚）：两项**确定性**缺陷修复 + CI 复现边界已闭合

> **性质与边界（如实声明）**：本轮**只改 `CC_Compare/tools/` 下的检查工具**，
> **未修改任何 gate、冻结树、已封存结果或已交付数字**；**未重跑** A3；
> **未接触**保留的最终测试种子（`20260927`/`20260928`）；**未**发起任何新的正式科学实验。
> 用户 2026-10-14 复核在源码里指出两处**确定性**逻辑缺陷与一处未闭合的 CI 复现边界，本节逐条闭合。

### 〇.1.1 CI 精确复现边界：**已解决**（根因是**取错种子**，不是生产者不同）

**症状**：D 段 ④ 一度报告 18 个配对块落盘 CI 与独立重算不符（偏差 5.06–9.96，但"是否排除 0"一致）。
（该边界在口径修正过程中先后报过 **4 个**与 **18 个**块未通过 —— 数量随参与比对的块数变化，
根因**同一个**；修复后为 **0**。）

**根因（只读核对两个生产者源码后确定）**：这些块属于 `a1_a3_online_density_*`，它们有**三个不同的
开发日集**（`20260926` / `20260930` / `20261001`），**每块必须用该 gate 自己的种子**。但：

| 事实 | 位置 |
|---|---|
| `run_a1_step3_compare.py` 的 gate 顶层 `seeds` **只写 `ran`，不写 `gate`** | `run_a1_step3_compare.py:405`（`"ran": list(seeds)`） |
| 外部臂 gate 同样只写 `ran` | `run_a1_external_accept.py:411` |
| 官方裁决器**正是读 `ran`** 作为 gate 种子 | `diag_a3_adjudicate.py:165`（断言 `seeds['ran'] == [int(seed)]`）、`diag_i3_online_adjudicate.py:45`（`doc['seeds']['ran']`） |
| 旧 `part_d` 只找 `seeds['gate']`，找不到就**回落到 `generated_from.seed`(=20260926)** | 已修复 |

⇒ 0930/20261001 两个 gate 的块被**用错了种子**算 bootstrap —— 这是**种子**差异，**不是**生产者差异。
**§〇 附注中"主模型 driver 与采集器复刻的 `_stat` 不是同一个 bootstrap 生产者"的推断据此作废。**

**修复（确定性优先级，禁止事后择优）**：

```
seeds['ran']（单一 int，规范字段） ▸ seeds['gate'] ▸ 目录名 _YYYYMMDD_ ▸ seeds['dev']
冲突（ran≠gate）或歧义（ran 非单一 int） ⇒ **fail-closed**，该块记 SKIP
解析不出的块**不回落任何默认种子**（否则 ④ 会静默变成假象）
```

⚠️ 明确**不做**的事：**不**"试遍候选种子取能对上 CI 的那一个" —— 那是事后择优，会使 ④ 失去证据价值。
`part_d` 现在**披露每个块实际用了哪一路证据**（`src_tally`）。

**复跑结果（在持有原始 gate 的 1003 上，只读）**：

```
A 52/52 恒等式 → 0 失败 ；B 隐含对照 I3 = 600.997 ；C 文档混用 0 条
D 核对 52 个配对块 → 硬失败 0、CI 数值未复现 0、SKIP 0
  每块种子来源：results_key=28, seeds.gate=24     ← 无一块需要弱兜底
  ✅ ④ CI_exact_reproduction=True
PAIRING_IDENTITY_PASS（①②③④ 全部通过，CI 逐位复现）   rc=0
```

新增反例测试 10 条（`tools/test_pairing_identity.py::run_gate_seed`）：`ran` 单一 / `gate`-only /
`ran` 歧义 / `ran≠gate` 冲突 / 目录日期兜底 / `seeds.dev` 兜底 / 全无 ⇒ SKIP / 嵌套 `results[seed]`
取外层键 / **按 `pc` 选块**（取错档会一次跑出 148 条假失败）/ gate 缺失。**14/14 通过**（含原 4 条 A/B 例）。

#### 〇.1.1.1 原生产者 bootstrap 的**只读**核对（随机数初始化 / 采样顺序 / 分位数算法）

用户要求「**只读**检查原生产者的 bootstrap 实现，若确有两种预先固定的 bootstrap 定义则**分别记录**，
**不得事后选择更有利的区间**」。逐字核对结果如下（`run_exp_reserve.py:353-362`，未改动任何源码）：

```python
def _stat(x, seed):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]      # ① 先剔除非有限
    n = len(x)
    r = np.random.default_rng(seed + 777)                       # ② 初始化：单一 PCG64，种子 seed+777
    mm = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])   # ③ 2000 次**顺序**有放回抽样
    return {"mean": float(np.mean(x)), "ci_lo": float(np.percentile(mm, 2.5)),
            "ci_hi": float(np.percentile(mm, 97.5)), "n": int(n)}          # ④ 朴素百分位（method='linear'）
```

| 要素 | 事实 | 含义 |
|---|---|---|
| 随机数初始化 | `np.random.default_rng(seed + 777)`，**单** rng、**无**并行/分块 | CI 完全由 `seed` 决定（**与机器无关**）⇒ 可逐位复现 |
| 采样顺序 | `r.integers(0, n, n)` 共 2000 次，**顺序消费**同一 rng（非 `size=(2000,n)` 向量化） | **按位置**抽样 ⇒ **输入向量顺序会改变 CI**，均值不变 |
| 分位数算法 | `np.percentile(mm, 2.5/97.5)`，默认 `method='linear'` | 确定性；无随机插值 |
| 剔除非有限 | `x[np.isfinite(x)]` | 非有限值在**统计前**被丢弃（与矩阵 `Q_finite` 的**门**是两回事） |
| 样本量 | `n = len(x)`（此处 40） | 日聚类（每天一票），非逐事件 |

**两种预先固定的定义**（共用同一个 `_stat`，只差输入顺序）：

| 定义 | 取对顺序 | 代码位置 | 角色 |
|---|---|---|---|
| **P（生产者顺序）** | `for r in a_rows`（原始顺序） | `run_a1_step2_gate.py:207` | **规范**：落盘 `mean/ci_*` 的真实来源 |
| **S（day_id 升序）** | 按 `day_id` 排序 | `collect_compare_table._pair`；`diag_a3_adjudicate._stat_exact`（`range(40)`） | 消费侧复刻口径 |

**并列实测（`tools/_probe_bootstrap_defs.py`，只读，不改任何落盘值）**：

```
全部配对块 = 52
  顺序相同（P == S）          = 52        ← 本批 per_day 列表本来就是 day_id 升序
  顺序不同（P != S）          = 0
  CI 上下界最大差（P vs S）    = 0.000000
  落盘值命中 P                = 52
  落盘值命中 S                = 52
  两者皆不命中                = 0
```

⇒ **两种定义在本次全部 52 个块上数值完全等同**，因此**不存在**"事后挑更有利区间"的余地
（也**没有**这样做）。⚠️ 但顺序敏感性是 `_stat` 的**结构性**性质，**换一批数据仍可能出现 P≠S**；
故 `part_d` 仍按**生产者顺序**复刻并在输出中披露每块的种子来源，**不**以本次等同为由删掉该护栏。
**落盘 CI 一律保留原值，本次未替换任何已发布区间。**

### 〇.1.2 `q_validity` 的确定性逻辑缺陷：**已修复**（合法全拒曾被误判为失败）

旧实现把「`n_solves` **缺失**」当作故障证据（`not n_solves`）。纯规则臂**结构性**没有求解器
（`solver_meta={}`），于是一个**完全合法**的规则全拒臂会被判为有效性失败 —— **缺失字段不是失败证据**。

**修复**：只认**正面**故障证据 —— `execution_crash`、`n_fail > 0`、或 `n_solves` **显式为 0**。
「方法究竟有没有被执行」由 `Q_method` 负责（对**非规则**臂，`n_solves` 缺失依然否决）；
两条判据职责分开后**不留漏洞**（非规则臂全拒 + `solver_meta={}` 仍由 `Q_method` 否决）。

**新增反例 ㉒–㉕ + `validity_checks()` 3 条**（`tools/test_qualification_gate.py`，**36/36 通过**）：
① 规则臂 + 合法全拒 + `solver_meta={}` ⇒ `qualified=True`、`informative=False`（用户点名的精确反例）；
② 规则臂 + 真崩溃 ⇒ **仍**判失败（修正未放宽故障判定）；
③ 非规则臂 + 全拒 + `solver_meta={}` ⇒ `Q_validity` 通过但 `Q_method` 否决；
④ 显式 `n_solves=0` / `execution_crash` ⇒ 否决。**λ=3 的历史判定（由 `Q_nondegen` 否决）不改写。**

### 〇.1.3 资格矩阵 `Q_days` / `Q_finite` 的确定性缺陷：**已修复（只收紧，零判决漂移）**

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| ① | `ids = [int(d.get('i')) …]` **静默强转** | `0.5→0`、`'5'→5`、`True→1` ⇒ 40 条 `i=0.5..39.5` 被截断成 `0..39`，**非整数日号伪装成"日号完整"** | 原始日号必须是**真整数**（`bool` 不算），否则直接判不完整 |
| ② | `fin = [u for u in utils if isinstance(u,(int,float))]` | `float('nan')`/`±inf` **都是** `float` ⇒ 被算作"有限"，判据形同虚设 | `math.isfinite`，并显式排除 `None`/字符串/`bool` |

**关键复核（防"事后改判据"）**：修复后在 **264 个真实臂**上逐臂复算，
`Q_formal_core` / `Q_formal_strict` / `Q_vetoes` **逐字段零差异**（core **38/62**、strict **32/62** 不变；
λ=3 全拒臂仍由 `Q_nondegen` 否决；A3 仍由 `Q_identity` 否决）。⇒ **判据只收紧，历史裁决未被改写。**
矩阵 JSON 已由修复后的工具重新生成（`criteria.revision=2`，并写入本节所述修订说明）。

新增反例测试 22 条（`tools/test_audit_qualification_matrix.py`），其中 `naive_*` 断言**故意复刻旧实现**
以证明"旧实现确实会放过这些输入"：非整数/字符串/bool/缺失/越界/重复日号、NaN/±Inf/None/字符串/bool 效用，
以及"缺陷必须传导到 `Q_formal_core`"的端到端 4 条。**22/22 通过。**

### 〇.1.4 归档清单的 commit 语义：**已拆分**（消除循环自指）

旧实现只写一个 `target_commit = git rev-parse HEAD`，把两件事挤在一个字段里。现改为：

| 字段 | 含义 | 落在哪 |
|---|---|---|
| `source_commit`（+ `source_commit_dirty_n/_files`） | **快照来源**：归档取树的那个提交 | 入库清单 `ARTIFACT_MANIFEST.json` |
| `release_commit` / `release_tag` | **发布靶点**：Release 切出的提交 | **独立** `results/.../RELEASE_METADATA.json`（**不入 Git**），以 `manifest_sha256` **指回**清单 ⇒ **单向无环** |

`--verify`：`source_commit != HEAD` ⇒ **告警**（清单自身所在提交必然是其**子**提交，属**预期**）；
`source_commit` **不是** HEAD 的祖先 ⇒ **判失败**（那才是真漂移）；清单里若仍有 `target_commit` ⇒ **判失败**。
负例实测：注入 `target_commit` 后 `ARTIFACT_MANIFEST_VERIFY_FAIL: 1`（rc=1）。
三份 Markdown 快照重建后与已提交版本**逐字节一致**。

### 〇.1.5 P1-A 正式效果门 δ = 20：**已登记（待用户最终锁定）**

`CC_Compare/P1A_强动态对照_预声明_2026-10-14.md` 新增 **§0.1**：判据为
主档 `p_c=(5,10,15)` 上、**以天为单位**按 `day_id` 配对，`mean(Δ) ≥ 20 ∧ CI_95%,lo(Δ) > 0`。
δ=20 **沿用 A3 的预注册值**（`A3在线比较预声明_2026-10-01.md`；`A3在线比较正式裁决_2026-10-06.md`
L5–6/L10「δ₃=+20」；`run_a1_step3_compare.py:49` `DELTA_3 = 20.0`、`:340/:343` 判定式），
**不是**为新实验量身定阈值。同时登记 bootstrap 规格（2000 次、2.5/97.5、种子跑前冻结、
④ 必须逐位复现否则**不算过门**）。
⚠️ **状态为"已登记 · 待用户确认"**，锁定前**不放行**任何正式科学批次；且只有一个开发种子 ⇒
过门**不得**声称跨种子稳健性，也**不替代** MaskCO 在 A3 上的贡献。

### 〇.1.6 本轮**仍未**做的事

- **未**重跑 A3 或任何已完成批次；**未**碰保留最终测试种子；**未**修改任何 gate / 冻结树 / 已封存结果。
- **未**修改任何**数字**：交付表、`dataset.json`、`Q_formal_core` 38/62 等一律不变。
- 仍**未**独立复核：3080 原始日志字节、PyVRP 历史故障批次（目录已删）、
  `RouteFinder strict=False` 缺失键清单、1003 `ordering_providers.py` 反向取回与 SHA。

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
