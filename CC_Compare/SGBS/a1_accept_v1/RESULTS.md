# SGBS — A-v1 接单臂结果（诊断批次）

> **诊断批次（时限修复前，正式重跑由主线统一执行）**：本批次运行于
> 2026-09-29 的 OrderingAcceptReplanner（旧版「先提交接单、再记录超时」语义，
> 越限决策仍生效），且运行期服务器负载 150–200+/96 核（多批次并行争抢）。
> 下方数字**仅诊断**：超时与求解耗时被争抢显著放大，不作为正式裁决证据；
> 正式批次（`sgbs_*_f`）由主线在时限修复 + 修复 pool 口径后统一执行。
> 注：SGBS 两臂 `hard_feasible_rate=1.0`（duplicate_service 未触发——该缺陷
> 依赖排序，本方法排序未把在途客户分给他车），但仍按诊断批次标注。

- 方法：SGBS（Simulation-Guided Beam Search，ICLR 2022），CVRP100 官方 checkpoint
  `1_pre_trained_model/Saved_CVRP100_Model/checkpoint-30500.pt`（2_SGBS 官方代码）。
- provider：`CC_Compare/SGBS/a1_accept_v1/sgbs_provider.py`（类 `SGBSProvider`，
  registry 条目 `sgbs`）；烟测：`smoke_sgbs.py`。
- 运行命令（在 `C-VRP_Cold-chainVehicleRoutingProblem/` 下）：
  `run_a1_external_accept.py --solver ordering --provider sgbs --provider-ckpt <ckpt> --provider-device cpu --baseline-from results/a1_step2_gate_c1_20260926/gate.json --dev-instances 40 --workers 6 --penalty "<p_c>" --out results/a1_ext_extra/sgbs_<tag>`

## 1. 主数字（开发集 seed 20260926，40 天）

| 臂 | utility_all_days mean [95% CI] | n_days | adjudicable | hard_feasible_rate | fail_counts | mean_timeouts（timeout_days） |
|---|---|---|---|---|---|---|
| sgbs_0（p_c=0） | **1213.42 [1133.99, 1286.59]** | 40 | True | 1.0 | `{}` | 93.8（38/40） |
| sgbs_51015（p_c=(5,10,15)） | **−16.92 [−146.08, 100.61]** | 40 | True | 1.0 | `{}` | 19.2（39/40） |

### 与步骤 2 三基线臂配对差（按天配对 day-clustered bootstrap）

| 配对 | sgbs_0 | sgbs_51015 |
|---|---|---|
| ordering_accept − cond_hist | **−271.12 [−351.06, −198.46]** | −325.49 [−363.43, −289.15] |
| ordering_accept − uncond_hist | −231.31 [−326.25, −144.44] | −254.00 [−318.53, −185.94] |
| ordering_accept − explicit_feat | −234.10 [−327.24, −148.75] | −261.72 [−325.02, −193.67] |

（全部 n_paired=40、adjudicable=True、identity_verified=False
（baseline_identity_missing，旧批次系统性条件）。）

### 决策/求解侧（solver_meta）

| 臂 | n_solves | n_fail | mean_solve_time_s | reject_reasons | n_served 均值/天 | elapsed_total_s（40 天求和） |
|---|---|---|---|---|---|---|
| sgbs_0 | 7761 | 0 | 10.88（争抢下） | `{timeout: 3753, deferred: 2, certify_budget: 1433}` | 64.3 | 86079 |
| sgbs_51015 | 7761 | 0 | 3.67（争抢下） | `{timeout: 766, certify_budget: 4261, deferred: 11}` | 68.1 | 29776 |

- sgbs_0 的 3753 次超时（≈48% 决策）与 max_decide ≈220s 是「beta=10 + aug=8 +
  负载 150–200 争抢 + 旧版越限仍生效」的叠加（单进程 n=60 基准 ≈7.8s，见 §2）；
  sgbs_51015 运行时段负载较低（766 次超时、mean_solve 3.67s），同配置差异主要
  来自外部争抢。

## 2. 超参数与官方口径

- 官方算法（2_SGBS/CVRPTester `_test_one_batch_simulation_guided_beam_search`）：
  模型贪心 POMO 多起点 rollout 选 num_starting_points 个最优起点 → 仿真引导束
  搜索（每步对每个 beam 取 top-gamma 展开分支做贪心 rollout 至完成，按 rollout
  回报保留 top-beta beam）。官方论文 CVRP100 用 **beam=1280**（CPU 上不可行）。
- **定档**：官方增广（augmentation_enable=True 的 x8 8-fold）下，服务器 CPU 单
  进程 n=60 mock 池实测：beam=10 ≈7.8s、beam=50 ≈10.9s（beam=100 ≈53s）。
  → 取 10s 预算内最大 beam：**beam=10**（官方参数空间内取值：`--beta` 为官方
  test.py 已有参数；gamma=4 = 官方 sgbs_gamma_minus1=3 默认；num_starting_points
  = beam_width = 官方 tester 语义）。RESULTS 口径：
  **官方参数空间内 beam=10（10s 预算约束；官方默认 1280 不可行）**。
- torch 线程数 = 16（基建旋钮，非方法配置）。

## 3. 零样本跨任务口径（诚实标注）

- 三个方法原生均为 CVRP（无 TW）：provider 只取子问题 depot + pool_customer_ids
  的（坐标、需求、容量）做 CVRP 求解，**忽略 TW / 锚点 / ready_time / 当前载重**；
  真实 TW / 容量 / C0 能耗可行性由下游 dcc_rh_v4 协调器（certify_append）与
  `certify_plan` 硬认证兜底。SGBS 不读取任何未来或隐藏信息。
- 排序 = 最优轨迹（aug × beam 中总距离最短）的客户访问顺序（去 depot），
  精确覆盖全池；空池返回 `()`；pool>100 → EDD fallback（本次 40 天未触发）。
- 内容寻址 memo：同决策内各车共享同一池，CVRP 求解不依赖 vehicle_id/锚点 →
  同一内容必然同一排序，按池内容缓存（每车 `order(sp)` 仍由 adapter 调用，
  缓存不改变任何决策，仅省重复计算）。

## 4. 适配日志（错误与修复）

1. **provider 路径错误（已修）**：初版 `_SGBS = dirname(dirname(_A1))` 多退
   一层 → `ModuleNotFoundError: CVRPEnv`。改为 `_SGBS = dirname(_A1)` 后通过。
2. **小池切片越界（已修）**：官方代码 `ordered_prob[:, :, :expansion-1]` 假设
   n+1 ≥ expansion（官方 n=100 恒成立）；池 n=1 时切片变短导致布尔赋值形状
   不匹配。按官方「冗余替换」机制补齐 prob=0 + greedy 节点（无效分支 → -inf），
   官方 n+1 ≥ expansion 时行为逐行不变。
3. **beam 与池大小的 min 截断**：beam_eff = min(beam, n)（官方 beta ≤ n 恒成立，
   小池时防止 `step(starting_points)` 形状不匹配）。
4. **烟测 mock 升级（已修）**：按主线要求将 mock 池升级为非连续真实风格 id
   （101/105/120/133/141、200+3i、300+i、400+i），断言
   `sorted(order) == sorted(pool_customer_ids)`。provider 映射全程
   `sp.node_index(c)` + `pool[k-1]` 回映，无 index/id 混淆，全部通过。
   烟测池 5/25/60 = 0.33s / 1.66s / 9.09s（争抢时段）。
5. **torch 弃用告警**：`torch.set_default_tensor_type` 在 torch 2.11 下产生
   deprecation 告警（官方 tester 原语，保留；不影响行为）。

## 5. 已知局限

- **时限与争抢**：诊断批次在负载 150–200 下运行，sgbs_0 mean_solve 10.9s、
  3753 次超时（max_decide ≈220s）；旧版时限语义下越限决策仍生效。n=90 无竞争
  探针**未执行**：正式批次 `*_f` 已在服务器运行，补跑探针会与正式臂争抢 CPU，
  故以 n=60 单进程基准 + 诊断批次实测超时作为计时证据。
- duplicate_service 未复现（两臂 hard_feasible_rate=1.0），但该缺陷存在且
  排序依赖，正式批次仍以主线修复后的 replanner 为准。
- p_c=(5,10,15) 下拒绝损失主导（utility 均值 −16.9），与 A-v1「p_c>0 使
  reject-loss 主导」既有结论一致；主信号看 p_c=0。
- pairing identity_verified=False：基线 gate.json 为 Q-04 身份前旧批次。
