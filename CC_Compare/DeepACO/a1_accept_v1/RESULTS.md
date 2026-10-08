# DeepACO A-v1 对比臂结果（诊断批次）

> **⚠️ 诊断批次声明（2026-09-30 主线协议更新）**：本批四条臂运行于**时限修复前**的
> OrderingAcceptReplanner（旧语义：越限决策仍先提交接单、再计超时），且 pool 含
> 在途 `committed_next` 客户（主线已修复为 pool 减在途）。若 40 天中出现
> `duplicate_service` 硬失败（POMO 诊断臂 40/40 天复现），即该机制的实锤证据；
> 本批数字**仅诊断，不构成正式裁决**。正式重跑由主线在修复后统一执行，本批产物保留。

## 1. 方法与档位

| 项 | 值 |
|---|---|
| 方法 | DeepACO（henry-yeh/DeepACO，NeurIPS 2023）vanilla CVRP 档（非 cvrp_nls） |
| provider | `CC_Compare/DeepACO/a1_accept_v1/deepaco_provider.py`（类 `DeepACOProvider`） |
| checkpoint | `pretrained/cvrp/cvrp100.pt`（官方 CVRP100；任务指令：先用 cvrp100.pt） |
| 依赖 | torch_geometric 2.8.0.post1（本批新装进 cc_compare env，CPU） |
| 档位 | n_ants=20（官方）。官方主档 T=100 在 CPU 超 10s/决策预算 → **官方 t_aco 网格 [1,10,20,30,40,50,100] 内最大 ≤7s 探针线的档 T=30**（定档依据见 §3） |
| 设备 | CPU（`--provider-device cpu`，禁止 GPU） |
| 输入规范化 | 官方 `utils.gen_instance` 口径：坐标 [0,1]（A-v1 原生即 [0,1]）、demand 原始值（官方 1..9 vs 容量 50；A-v1 1..3 vs 容量 50，同口径不缩放） |
| 排序来源 | `ACO.run(T)` 后 `shortest_path` 删 depot 后首次出现序；防御性 EDD 补尾与超容量兜底（A-v1 不触发） |

## 2. 烟测（服务器，CPU）

mock sp 用**非连续真实风格 id**（101..299），断言 `sorted(order) == sorted(pool_customer_ids)`：

| pool | 求解耗时 (T=30) | 覆盖 |
|---|---|---|
| 5 | 0.96s | ✓ |
| 25 | 2.41s | ✓ |
| 60 | 5.71s | ✓ |

（T=100 时 n=60 实测 11.3s —— 超预算证据）

## 3. n=90 无竞争最大池探针（2026-09-30，服务器 load ≈200/96 共享窗口）

3 次单决策 `provider.order`（n=90，n_ants=20）中位数（秒）：

| T | 100 | 50 | 40 | 30 | 20 | 10 |
|---|---|---|---|---|---|---|
| median | 17.54 | 9.35 | 7.50 | **6.11** | 5.86 | 2.90 |

→ 官方主档 T=100 与 T=50、T=40 均超 7s 探针线；**定档 T=30**（官方 t_aco 网格内最大 ≤7s 档）。正式批次由主线按此定档复核。

## 4. 结果（seed 20260926 开发集 40 天；`results/a1_ext_extra/deepaco_{0,51015}/gate.json`）

### p_c=0

| 指标 | 值 |
|---|---|
| utility_all_days mean [ci_lo, ci_hi] | **1239.47** [1208.40, 1267.33]（n=40） |
| n_days / adjudicable | 40 / True |
| hard_feasible_rate / failures | 1.0 / `{}`（无 duplicate_service） |
| mean_timeouts | 0.0 |
| reject_reasons (solver_meta) | `{'certify_budget': 5105, 'deferred': 27}`（n_solves=7761, n_fail=0, mean_solve 0.507s） |
| ordering_accept − cond_hist | **−245.07** [−273.52, −216.83]（n_paired=40, adjudicable=True） |
| ordering_accept − uncond_hist | −205.26 [−250.45, −158.67] |
| ordering_accept − explicit_feat | −208.05 [−253.87, −160.88] |

### p_c=(5,10,15)

| 指标 | 值 |
|---|---|
| utility_all_days mean [ci_lo, ci_hi] | **−85.43** [−210.02, 33.92]（n=40） |
| n_days / adjudicable | 40 / True |
| hard_feasible_rate / failures | 1.0 / `{}` |
| mean_timeouts | 0.0 |
| reject_reasons (solver_meta) | `{'certify_budget': 5139, 'deferred': 14}`（n_solves=7761, n_fail=0） |
| ordering_accept − cond_hist | **−394.00** [−435.39, −352.42] |
| ordering_accept − uncond_hist | −322.51 [−388.17, −251.55] |
| ordering_accept − explicit_feat | −330.24 [−399.91, −255.96] |

（配对块 `formal_adjudication=False`、`identity_verified=False`（Q-04 门控：
诊断批次驱动源码与步骤 2 基线身份不匹配，仅诊断口径；正式重跑由主线修复后复核。））

## 5. 适配日志（错误与修复）

1. **torch_geometric 缺失** → `pip install torch-geometric`（2.8.0.post1，CPU）入 cc_compare env；torch 保持 2.11.0+cu128。
2. **官方 T=100 超预算** → 定档 T=30（§3 探针）。
3. **防御**：官方 `aco.gen_path` 对 demand > capacity 的客户会死循环（构造无防护）→ provider 加超容量 EDD 兜底（A-v1 demand ≤3 << 50，不触发）。
4. **烟测 mock 升级**（MVMoE 教训）：非连续真实风格 id + 真实 id 集合覆盖断言；provider 无 index/id 混淆（输出 `pool[t-1]` 真实 id）。

## 6. 已知局限（诚实口径）

- **CVRP-only 零样本跨任务**：原生无 TW/冷链/多温区/在途锚点；本臂按 CVRP 求解排序、忽略 TW，真实 TW/C0 可行性由下游冻结 dcc_rh_v4 协调器 + certify_plan 硬认证兜底。
- T=30 为官方 t_aco 网格档（非论文 T=100 主档）；ACO 构造随机性（test.py 不固定采样 seed），同决策内 15 辆车共享同一 pool memoize 保证一致性。
- 诊断批次时限与 pool 语义见文件头声明。
