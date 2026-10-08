# POMO — A-v1 接单臂结果（诊断批次）

> **诊断批次（时限修复前，正式重跑由主线统一执行）**：本批次运行于
> 2026-09-29 的 OrderingAcceptReplanner（旧版「先提交接单、再记录超时」语义），
> 且含「pool 未扣除在途 committed_next」的 duplicate_service 协议缺陷。
> 下方数字**仅诊断**：utility 主数字在 `hard_feasible_rate=0`（duplicate_service 40/40）
> 下不可作为正式裁决证据；正式批次待主线修复时限语义与 pool 口径后统一重跑。

- 方法：POMO（yd-kwon/POMO，NeurIPS 2021），CVRP100 官方 checkpoint
  `POMO/result/saved_CVRP100_model/checkpoint-30500.pt`。
- provider：`CC_Compare/POMO/a1_accept_v1/pomo_provider.py`（类 `POMOProvider`，
  registry 条目 `pomo`）；烟测：`smoke_pomo.py`。
- 运行命令（在 `C-VRP_Cold-chainVehicleRoutingProblem/` 下）：
  `run_a1_external_accept.py --solver ordering --provider pomo --provider-ckpt <ckpt> --provider-device cpu --baseline-from results/a1_step2_gate_c1_20260926/gate.json --dev-instances 40 --workers 6 --penalty "<p_c>" --out results/a1_ext_extra/pomo_<tag>`

## 1. 主数字（开发集 seed 20260926，40 天）

| 臂 | utility_all_days mean [95% CI] | n_days | adjudicable | hard_feasible_rate | fail_counts | mean_timeouts（timeout_days） |
|---|---|---|---|---|---|---|
| pomo_0（p_c=0） | **1132.78 [1096.10, 1169.48]** | 40 | True | **0.0** | `{duplicate_service: 40}` | 1.95（5/40） |
| pomo_51015（p_c=(5,10,15)） | **诊断腿缺失**（被外部 SIGTERM 误杀，exit 143；正式批次 `pomo_*_f` 已由主线执行，正式腿为准） | — | — | — | — | — |

### pomo_0 与步骤 2 三基线臂配对差（按天配对 day-clustered bootstrap）

| 配对 | mean [95% CI] | n_paired | adjudicable | identity_verified |
|---|---|---|---|---|
| ordering_accept − cond_hist | **−351.76 [−393.81, −312.75]** | 40 | True | False（baseline_identity_missing，旧批次） |
| ordering_accept − uncond_hist | −311.95 [−370.26, −256.24] | 40 | True | False（同上） |
| ordering_accept − explicit_feat | −314.74 [−370.12, −259.03] | 40 | True | False（同上） |

### pomo_0 决策/求解侧（solver_meta）

- n_solves = 7761（≈194 决策/天）；n_fail = 0（无 provider 异常/fallback）；
  mean_solve_time_s = 0.476（provider 推理，6 worker 并行，运行期负载 ~57）。
- reject_reasons = `{certify_budget: 5243, timeout: 78, deferred: 24}`；
  n_served 均值 60.4/天；elapsed_total_s（40 天求和）= 4593。
- max_decide_s ≈ 40.5s（大池 + aug=8 越 10s 预算；旧版语义越限仍生效、仅记超时）。

## 2. 超参数与官方口径

- 官方评测口径（test_n100.py + CVRPTester）：CVRP100 checkpoint；pomo_size=
  problem_size 多起点 rollout（首步 depot、次步每客户一起点、其后 argmax）；
  augmentation_enable=True 的 **x8 8-fold 增广**。
- 10s/决策基准（服务器 CPU，无争抢，n=60 mock 池实测）：**aug=8 ≈ 0.27s ≪ 10s
  → 保留官方评测档 x8**（官方更轻档 num_aug=1 亦存在，未启用）。
- torch 线程数 = 16（基建旋钮，非方法配置；影响速度不影响决策）。

## 3. 零样本跨任务口径（诚实标注）

- 三个方法原生均为 CVRP（无 TW）：provider 只取子问题 depot + pool_customer_ids
  的（坐标、需求、容量）做 CVRP 求解，**忽略 TW / 锚点 / ready_time / 当前载重**；
  真实 TW / 容量 / C0 能耗可行性由下游 dcc_rh_v4 协调器（certify_append）与
  `certify_plan` 硬认证兜底。这不是借换目标：POMO 不读取任何未来或隐藏信息。
- 排序 = 最优轨迹（aug × pomo 中总距离最短）的客户访问顺序（去 depot），
  精确覆盖全池；空池返回 `()`；pool>100（超出官方 CVRP100 域）→ EDD fallback
  （防解码超时，本次 40 天未触发）。
- 内容寻址 memo：同决策内各车共享同一池，CVRP 求解不依赖 vehicle_id/锚点 →
  同一内容必然同一排序，按池内容缓存（每车 `order(sp)` 仍由 adapter 调用，
  缓存不改变任何决策，仅省重复计算）。

## 4. 适配日志（错误与修复）

1. **provider 路径错误（已修）**：初版 `_POMO = dirname(dirname(_A1))` 多退一层
   （指向 CC_Compare），导致 `ModuleNotFoundError: CVRPEnv`。改为
   `_POMO = dirname(_A1)` 后服务器烟测通过。
2. **烟测 mock 升级（已修）**：按主线要求将 mock 池从连续 id 升级为非连续真实
   风格 id（101/105/120/133/141、200+3i、300+i、400+i），断言
   `sorted(order) == sorted(pool_customer_ids)`。provider 映射全程
   `sp.node_index(c)` + `pool[k-1]` 回映，无 index/id 混淆，全部通过。
3. **torch 弃用告警**：`torch.set_default_tensor_type` 在 torch 2.11 下产生
   deprecation 告警（官方 tester 原语，保留不动；不影响行为）。
4. **pomo_51015 被外部 SIGTERM 误杀（诊断腿缺失）**：exit 143（非代码错误；
   OOM 应为 137）。发生时段服务器负载 190+/96 核（cada/mvmoe/lih/attention/
   omnivrp 等并行批次）。**不再重跑**：正式批次 `pomo_*_f` 已由主线统一执行，
   本诊断批次缺一腿不影响结论；pomo_0 不受影响。

## 5. 已知局限

- **duplicate_service 40/40**：本批次 replanner 的 pool 含在途 committed_next
  客户，POMO 的排序使协调器将其分给另一辆车 → 同客户服务两次（主线已知修复 =
  pool 减去在途 committed_next）。utility 数字因此**协议无效，仅诊断**。
- 大池（≳80 客户）时 aug=8 单决策会越 10s（本批次 max 40.5s、78 次超时）；
  旧版时限语义下越限决策仍生效。n=90 无竞争探针未执行：正式批次 `*_f` 已在
  服务器运行，补跑探针会与正式臂争抢 CPU；以 n=60 单进程基准（aug=8 ≈0.27s）
  + 本批次实测超时作为计时证据（smoke --probe90 已备好，供正式批次复核用）。
- pairing identity_verified=False：基线 gate.json 为 Q-04 身份前旧批次
  （baseline_identity_missing），属系统性条件，不升级正式裁决。
