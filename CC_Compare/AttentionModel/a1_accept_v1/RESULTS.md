# AttentionModel A-v1 对比臂结果（诊断批次）

> **⚠️ 诊断批次声明（2026-09-30 主线协议更新）**：本批四条臂运行于**时限修复前**的
> OrderingAcceptReplanner（旧语义：越限决策仍先提交接单、再计超时），且 pool 含
> 在途 `committed_next` 客户（主线已修复为 pool 减在途）。若 40 天中出现
> `duplicate_service` 硬失败（POMO 诊断臂 40/40 天复现），即该机制的实锤证据；
> 本批数字**仅诊断，不构成正式裁决**。正式重跑由主线在修复后统一执行，本批产物保留。

## 1. 方法与档位

| 项 | 值 |
|---|---|
| 方法 | AttentionModel（wouterkool/attention-learn-to-route，ICLR 2019） |
| provider | `CC_Compare/AttentionModel/a1_accept_v1/am_provider.py`（类 `AMProvider`） |
| checkpoint | `pretrained/cvrp_100/epoch-99.pt`（官方 CVRP100；args: embedding/hidden 128, 3 层, batch-norm） |
| 档位 | **官方 greedy 档**（`eval.py --decode_strategy greedy`，width=0）。官方论文主结果 beam1280 在 CPU 10s/决策预算内不可行；greedy 是官方参数空间内的轻档。 |
| 设备 | CPU（`--provider-device cpu`，禁止 GPU） |
| 输入规范化 | 官方 `make_instance` 口径：坐标 [0,1]（A-v1 原生即 [0,1]）、demand/capacity |
| 桥接语义 | 子问题 = depot + pool（官方模型无 anchor/ready_time/load/TW 输入，忽略之）；每决策同 pool 对 15 辆 replan 车 memoize 只解码一次 |

## 2. 烟测（服务器，CPU）

mock sp 用**非连续真实风格 id**（101..299），断言 `sorted(order) == sorted(pool_customer_ids)`：

| pool | 解码耗时 | 覆盖 |
|---|---|---|
| 5 | 0.10s | ✓ |
| 25 | 0.08s | ✓ |
| 60 | 0.15s | ✓ |

## 3. n=90 无竞争最大池探针（2026-09-30，服务器 load ≈200/96 共享窗口）

- 3 次单决策 `provider.order`（n=90）中位数 **0.17s**（0.16–0.21s）→ 远低于 7s 探针线，**正式档 = greedy 不动**。

## 4. 结果（seed 20260926 开发集 40 天；`results/a1_ext_extra/attention_{0,51015}/gate.json`）

### p_c=0

| 指标 | 值 |
|---|---|
| utility_all_days mean [ci_lo, ci_hi] | **1236.21** [1207.35, 1263.20]（n=40） |
| n_days / adjudicable | 40 / True |
| hard_feasible_rate / failures | 1.0 / `{}`（无 duplicate_service） |
| mean_timeouts | 0.0 |
| reject_reasons (solver_meta) | `{'certify_budget': 5135, 'deferred': 10}`（n_solves=7761, n_fail=0, mean_solve 0.034s） |
| ordering_accept − cond_hist | **−248.33** [−274.73, −222.91]（n_paired=40, adjudicable=True） |
| ordering_accept − uncond_hist | −208.52 [−254.36, −158.39] |
| ordering_accept − explicit_feat | −211.31 [−257.68, −163.11] |

### p_c=(5,10,15)

| 指标 | 值 |
|---|---|
| utility_all_days mean [ci_lo, ci_hi] | **−72.92** [−196.81, 46.63]（n=40） |
| n_days / adjudicable | 40 / True |
| hard_feasible_rate / failures | 1.0 / `{}` |
| mean_timeouts | 0.0 |
| reject_reasons (solver_meta) | `{'certify_budget': 5135, 'deferred': 10}`（n_solves=7761, n_fail=0） |
| ordering_accept − cond_hist | **−381.49** [−418.38, −345.61] |
| ordering_accept − uncond_hist | −310.00 [−379.77, −231.30] |
| ordering_accept − explicit_feat | −317.72 [−388.22, −241.88] |

（配对块 `formal_adjudication=False`、`identity_verified=False`（Q-04 门控：
诊断批次驱动源码与步骤 2 基线身份不匹配，仅诊断口径；正式重跑由主线修复后复核。））

## 5. 适配日志（错误与修复）

1. **首测解码极慢（n=60 7.7s）** → 96 核服务器上小算子线程调度开销；provider `_load()` 内 `torch.set_num_threads(8)` + `set_num_interop_threads(1)`（运行期设置，不改算法）→ 0.15s。
2. **烟测 mock 升级**（MVMoE 教训）：id==index mock 升级为非连续真实风格 id + 真实 id 集合覆盖断言；provider 全程经 `sp.node_index(c)` 取真实 id 行列、输出 `pool[t-1]` 真实 id，无 index/id 混淆。

## 6. 已知局限（诚实口径）

- **CVRP-only 零样本跨任务**：原生无 TW/冷链/多温区/在途锚点；本臂按 CVRP 求解排序、忽略 TW，真实 TW/C0 可行性由下游冻结 dcc_rh_v4 协调器 + certify_plan 硬认证兜底（接单决策的可行性不依赖本 provider）。
- 子问题规模为可见 pool（≤ ~90），模型训练于 100 客户规模；greedy 档为官方轻档（非论文 beam1280 主档）。
- 诊断批次时限与 pool 语义见文件头声明。
