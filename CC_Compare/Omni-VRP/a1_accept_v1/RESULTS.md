# Omni-VRP A-v1 对比臂结果（诊断批次）

> **⚠️ 诊断批次声明（2026-09-30 主线协议更新）**：本批四条臂运行于**时限修复前**的
> OrderingAcceptReplanner（旧语义：越限决策仍先提交接单、再计超时），且 pool 含
> 在途 `committed_next` 客户（主线已修复为 pool 减在途）。若 40 天中出现
> `duplicate_service` 硬失败（POMO 诊断臂 40/40 天复现），即该机制的实锤证据；
> 本批数字**仅诊断，不构成正式裁决**。正式重跑由主线在修复后统一执行，本批产物保留。

## 1. 方法与档位

| 项 | 值 |
|---|---|
| 方法 | Omni-VRP（mktabak/Omni-VRP，ICML 2023）POMO-CVRP 基础模型（instance-norm 零样本档） |
| provider | `CC_Compare/Omni-VRP/a1_accept_v1/omnivrp_provider.py`（类 `OmniVRPProvider`） |
| checkpoint | `pretrained/POMO-CVRP/uniform/checkpoint-30500-cvrp100-instance-norm.pt`（官方 uniform CVRP100） |
| 模型参数 | 官方 test.py 默认（embedding 128 / encoder 6 层 / head 8 / qkv 16 / logit_clipping 10 / eval_type 'argmax'）+ norm='instance'（与 instance-norm ckpt 匹配） |
| 档位 | 官方评测档：pomo_size=problem_size、aug_factor=8（x8 几何增强，官方主档） |
| 设备 | CPU（`--provider-device cpu`，禁止 GPU） |
| 输入规范化 | 官方口径：坐标 [0,1]（A-v1 原生即 [0,1]）、demand/capacity |
| 排序来源 | POMO 全部 rollout（aug × pomo）中官方 reward（负路程）最小者的路线，删 depot 后首次出现序；防御性 EDD 补尾 |

## 2. 烟测（服务器，CPU）

mock sp 用**非连续真实风格 id**（101..299），断言 `sorted(order) == sorted(pool_customer_ids)`：

| pool | 解码耗时 (aug×8) | 覆盖 |
|---|---|---|
| 5 | 5.0s（首调 MKL 预热；后续 <0.2s） | ✓ |
| 25 | 0.19s | ✓ |
| 60 | 0.40s | ✓ |

## 3. n=90 无竞争最大池探针（2026-09-30，服务器 load ≈200/96 共享窗口）

- 3 次单决策 `provider.order`（n=90，aug×8）中位数 **1.61s**（0.98–7.43s；首调含
  MKL 预热 7.4s）→ ≤7s 探针线，**正式档 = 官方评测档（aug×8）不动**。

## 4. 结果（seed 20260926 开发集 40 天；`results/a1_ext_extra/omnivrp_{0,51015}/gate.json`）

### p_c=0

| 指标 | 值 |
|---|---|
| utility_all_days mean [ci_lo, ci_hi] | **1226.91** [1195.78, 1254.53]（n=40） |
| n_days / adjudicable | 40 / True |
| hard_feasible_rate / failures | 1.0 / `{}`（无 duplicate_service） |
| mean_timeouts | 0.0 |
| reject_reasons (solver_meta) | `{'certify_budget': 5160, 'deferred': 1}`（n_solves=7761, n_fail=0, mean_solve 0.358s） |
| ordering_accept − cond_hist | **−257.63** [−289.01, −226.42]（n_paired=40, adjudicable=True） |
| ordering_accept − uncond_hist | −217.82 [−266.66, −166.61] |
| ordering_accept − explicit_feat | −220.61 [−269.46, −171.13] |

### p_c=(5,10,15)

| 指标 | 值 |
|---|---|
| utility_all_days mean [ci_lo, ci_hi] | **−85.72** [−209.56, 33.83]（n=40） |
| n_days / adjudicable | 40 / True |
| hard_feasible_rate / failures | 1.0 / `{}` |
| mean_timeouts | 0.0 |
| reject_reasons (solver_meta) | `{'certify_budget': 5160, 'deferred': 1}`（n_solves=7761, n_fail=0） |
| ordering_accept − cond_hist | **−394.29** [−440.08, −347.63] |
| ordering_accept − uncond_hist | −322.80 [−400.02, −239.40] |
| ordering_accept − explicit_feat | −330.52 [−404.94, −254.40] |

（配对块 `formal_adjudication=False`、`identity_verified=False`（Q-04 门控：
诊断批次驱动源码与步骤 2 基线身份不匹配，仅诊断口径；正式重跑由主线修复后复核。））

## 5. 适配日志（错误与修复）

1. **depot_xy 形状错误**（`augment_xy_data_by_8_fold` 需 (batch,1,2)，初版给了 (1,2)）→ 修正为 (1,1,2)。
2. **烟测 mock 升级**（MVMoE 教训）：非连续真实风格 id + 真实 id 集合覆盖断言；provider 无 index/id 混淆（输出 `pool[t-1]` 真实 id）。
3. **首调 5-7s 预热**：instance-norm POMO 首解触发 MKL 线程池预热（进程内一次性）；40 天臂中按决策均值口径如实记录。

## 6. 已知局限（诚实口径）

- **官方权重无 TW 训练档**：仓库放出的 POMO 权重只有 CVRP/TSP 档（CVRP 为
  x,y,demand 3 维输入）；本臂按 **CVRP 零样本**求解排序、忽略 TW，真实 TW/C0
  可行性由下游冻结 dcc_rh_v4 协调器 + certify_plan 硬认证兜底。
- 子问题规模 = 可见 pool（≤ ~90），pomo_size 随规模变化（官方 tester 对 .pkl 数据同样按实例规模设 pomo_size）。
- 诊断批次时限与 pool 语义见文件头声明。
