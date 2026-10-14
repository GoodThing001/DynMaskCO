# RouteFinder A-v1 接单臂结果（正式批次）

> 说明：本文件是补写的**最小 RESULTS.md**（此前该目录只有 `routefinder_provider.py` /
> `smoke_routefinder.py` / `run_formal_gpu.sh`，缺 RESULTS.md）。数字取自正式臂落盘
> `results/a1_ext_extra/routefinder_{0,51015}_f/gate.json`，参与度对照取自
> `results/a1_compare_table/participation/routefinder.json`（2026-10-13 实测）。

## 1. 方法与档位

| 项 | 值 |
|---|---|
| 方法 | RouteFinder（`ai4co/routefinder`，TMLR 2025 / ICML 2026 J2C），VRP 基础模型 |
| provider | `CC_Compare/RouteFinder/a1_accept_v1/routefinder_provider.py`（类 `RouteFinderProvider`） |
| checkpoint | `CC_Compare/RouteFinder/checkpoints/100/rf-transformer.ckpt`（sha256 前 16 `eb09cee46108bac0`，20,147,300 B） |
| 档位 | 官方 rf-transformer ×50/100 档中的 100 档（`num_loc=100`），无 fallback 参与 |
| 设备 | 正式臂 GPU（`--provider-device cuda`）；参与度对照用 CPU（单线程，n=40） |
| 输入规范化 | 官方 MTVRPEnv 口径；provider 按子问题**精确规模**构造 td（无 phantom 行，见下 §5.1） |
| 排序来源 | 解码序列删 depot 后客户首次出现序；未覆盖客户按 EDD 补齐（本次 40 天未触发） |
| 运行命令 | `run_a1_external_accept.py --solver ordering --provider routefinder --provider-ckpt <ckpt> --provider-device cuda --baseline-from <gate> --dev-instances 40 --workers 4 --penalty <p_c> --out results/a1_ext_extra/routefinder_<tag>` |

## 2. 正式臂结果（seed 20260926 开发集 40 天）

| 指标 | p_c=0（`routefinder_0_f`） | 主档 p_c=(5,10,15)（`routefinder_51015_f`） |
|---|---|---|
| utility_all_days mean [95% CI] | **1266.19** [1244.15, 1287.45] | **−26.10** [−152.14, 93.40] |
| n_days / adjudicable | 40 / True | 40 / True |
| hard_feasible_rate / fail_counts | 1.0 / `{}` | 1.0 / `{}` |
| mean_timeouts | 0.0 | 0.0 |
| reject_reasons | `{certify_budget: 5045, deferred: 13}`（n_solves=7761, n_fail=0） | 同构（certify_budget 主导） |
| mean_solve_time_s / max_decide_s | 0.985 / **3.89** | 1.003 / **3.91** |
| fallback 率 | 0% | 0% |

### 与基线的配对差

| 配对 | p_c=0 | 主档 |
|---|---|---|
| 落盘（对旧基线 `a1_step2_gate_c1_20260926`，`identity_verified=false`） | −218.35 [−241.82, −194.33] | −334.60 [−365.03, −302.20] |
| 重配对（对身份合格 `a1_c1rr2_reveal_40`，见对比总表 README §7） | **−437.60** [−466.30, −408.71] | **−657.48** |

**读数**：RouteFinder 是全部外部方法中 **p_c=0 档最强**（1266.19，全表第 4），但仍低于
C1 cond_hist 基线约 −437.6（重配对口径）；主档 −26.1 亦低于基线 −657.5。
即「换更强的路由基础模型」未换来接单决策收益，与全表结论一致。

## 3. 模型参与度对照（`verify_provider_participation.py`，2026-10-13，CPU 单线程）

| 对照 | 结果 |
|---|---|
| C1 真实权重 | 覆盖全池 ✔，单次 `order()`（清缓存）**0.426 s**（n_pool=40） |
| **C2 随机权重替换**（`_policy`，替换 1,677,824 参数） | 与真实权重排序 **39/40 位置不同**、Spearman = **−0.307**、first-12 重合 **1/12** → `order_changed_by_weights = True` |
| C3 EDD | 与真实权重 **38/40 位置不同**，Spearman = 0.416 |
| B6 未来扰动（只改不可见节点） | `ordering_same=True`、`subproblem_hash_same=True` |
| B3 确定性（**清缓存**后两次） | `True` |

## 4. 已知局限（诚实口径）

- **CVRP 零样本跨任务**：RouteFinder 原生无 TW/冷链/温区/在途锚点；本臂按其 VRP 口径求解排序、
  忽略 TW，真实可行性由下游冻结 `dcc_rh_v4` 协调器 + `certify_plan` 硬认证兜底。
- 子问题规模 = 可见 pool（≤ ~90），模型为 100 规模档；本参与度对照用 n_pool=40。
- 参与度对照的「不同位置数」基于**单个** mock 子问题（n=40）；不是 40 天轨迹上的统计量。

## 5. 适配日志

1. **TD 必须按子问题精确规模构造，不得含 phantom 行**：mtvrp 解码以 `done=全节点访问` 为终止条件，
   phantom 行会因原生 depot 位移导致 TW 永久不可达而死锁（旧适配曾因此挂死）。
2. 未覆盖客户按 EDD 补齐（防御；A-v1 pool ≤ ~90，40 天未触发）。
3. 参与度对照在 CPU 单线程下运行（避免抢 F1 正式批次 CPU）；正式臂在 GPU 上 `max_decide` 3.9 s，
   远低于 10 s 预算。
