# Sym-NCO — A-v1 接单臂结果（诊断批次）

> **诊断批次（时限修复前，正式重跑由主线统一执行）**：本批次运行于
> 2026-09-29 的 OrderingAcceptReplanner（旧版「先提交接单、再记录超时」语义），
> 且含「pool 未扣除在途 committed_next」的 duplicate_service 协议缺陷。
> 下方数字**仅诊断**：utility 主数字在旧时限语义（越限决策仍生效）下不可作为
> 正式裁决证据；正式批次待主线修复后统一重跑。注：Sym-NCO 两臂
> `hard_feasible_rate=1.0`（duplicate_service 未触发——该缺陷依赖排序，
> 本方法排序恰好未把在途客户分给他车），但仍按诊断批次标注。

- 方法：Sym-NCO（ICLR 2022），CVRP100 官方 checkpoint
  `pretrained_model/Sym-NCO/checkpoint-8000.pt`（CVRPModel_ours）。
- provider：`CC_Compare/Sym-NCO/a1_accept_v1/symnco_provider.py`（类 `SymNCOProvider`，
  registry 条目 `symnco`）；烟测：`smoke_symnco.py`。
- 运行命令（在 `C-VRP_Cold-chainVehicleRoutingProblem/` 下）：
  `run_a1_external_accept.py --solver ordering --provider symnco --provider-ckpt <ckpt> --provider-device cpu --baseline-from results/a1_step2_gate_c1_20260926/gate.json --dev-instances 40 --workers 6 --penalty "<p_c>" --out results/a1_ext_extra/symnco_<tag>`

## 1. 主数字（开发集 seed 20260926，40 天）

| 臂 | utility_all_days mean [95% CI] | n_days | adjudicable | hard_feasible_rate | fail_counts | mean_timeouts（timeout_days） |
|---|---|---|---|---|---|---|
| symnco_0（p_c=0） | **1246.36 [1212.41, 1280.19]** | 40 | True | 1.0 | `{}` | 11.6（39/40） |
| symnco_51015（p_c=(5,10,15)） | **−25.20 [−150.08, 93.28]** | 40 | True | 1.0 | `{}` | 10.8（40/40） |

### 与步骤 2 三基线臂配对差（按天配对 day-clustered bootstrap）

| 配对 | symnco_0 | symnco_51015 |
|---|---|---|
| ordering_accept − cond_hist | **−238.18 [−271.40, −203.81]** | −333.77 [−367.98, −298.87] |
| ordering_accept − uncond_hist | −198.37 [−245.56, −147.94] | −262.28 [−324.54, −192.59] |
| ordering_accept − explicit_feat | −201.16 [−246.46, −152.18] | −270.01 [−333.62, −202.41] |

（全部 n_paired=40、adjudicable=True、identity_verified=False
（baseline_identity_missing，旧批次系统性条件）。）

### 决策/求解侧（solver_meta）

| 臂 | n_solves | n_fail | mean_solve_time_s | reject_reasons | n_served 均值/天 |
|---|---|---|---|---|---|
| symnco_0 | 7761 | 0 | 4.33（争抢下） | `{certify_budget: 4619, timeout: 464, deferred: 3}` | 66.9 |
| symnco_51015 | 7761 | 0 | 4.12（争抢下） | `{timeout: 432, certify_budget: 4631, deferred: 3}` | 67.4 |

- 运行期服务器负载 180–200/96 核（多批次并行争抢），solve 时间与超时计数
  显著被放大（单进程 n=60 基准仅 ≈0.47s，见下）；elapsed_total_s（40 天求和）
  ≈ 35.1k / 33.5k。

## 2. 超参数与官方口径

- 官方评测口径（test_symnco.py + CVRPTester，is_pomo=False → CVRPModel_ours）：
  首步 depot；次步按「depot 出发贪心概率」排序取 pomo_size 个不同起点
  （second_beam=1）；第 3 步每轨迹 top-1；其后 argmax。评测增广 = 官方等价
  **8 增广**（augment_xy_data_by_N_fold 随机旋转/翻转，N=8；本适配固定
  torch.manual_seed(0) 保证跨进程/跨次可复现）。
- 10s/决策基准（服务器 CPU，无争抢，n=60 mock 池实测）：**aug=8 ≈ 0.47s ≪ 10s
  → 保留官方评测档 x8**（官方更轻档 num_aug=1 亦存在，未启用）。
- torch 线程数 = 16（基建旋钮，非方法配置）。

## 3. 零样本跨任务口径（诚实标注）

- 三个方法原生均为 CVRP（无 TW）：provider 只取子问题 depot + pool_customer_ids
  的（坐标、需求、容量）做 CVRP 求解，**忽略 TW / 锚点 / ready_time / 当前载重**；
  真实 TW / 容量 / C0 能耗可行性由下游 dcc_rh_v4 协调器（certify_append）与
  `certify_plan` 硬认证兜底。Sym-NCO 不读取任何未来或隐藏信息。
- 排序 = 最优轨迹（aug × pomo 中总距离最短）的客户访问顺序（去 depot），
  精确覆盖全池；空池返回 `()`；pool>100 → EDD fallback（本次 40 天未触发）。
- 内容寻址 memo：同决策内各车共享同一池，CVRP 求解不依赖 vehicle_id/锚点 →
  同一内容必然同一排序，按池内容缓存（每车 `order(sp)` 仍由 adapter 调用，
  缓存不改变任何决策，仅省重复计算）。

## 4. 适配日志（错误与修复）

1. **provider 路径错误（已修）**：初版 `_SYMNCO = dirname(dirname(_A1))` 多退
   一层 → `ModuleNotFoundError: CVRPEnv`。改为 `_SYMNCO = dirname(_A1)` 后通过。
2. **烟测 mock 升级（已修）**：按主线要求将 mock 池升级为非连续真实风格 id
   （101/105/120/133/141、200+3i、300+i、400+i），断言
   `sorted(order) == sorted(pool_customer_ids)`。provider 映射全程
   `sp.node_index(c)` + `pool[k-1]` 回映，无 index/id 混淆，全部通过。
3. **N-fold 随机增广的确定性**：官方 `augment_xy_data_by_N_fold` 用 torch.rand，
   已在 `_load` 固定 `torch.manual_seed(0)`（跨 worker/跨次运行可复现）。
4. **torch 弃用告警**：`torch.set_default_tensor_type` 在 torch 2.11 下产生
   deprecation 告警（官方 tester 原语，保留；不影响行为）。

## 5. 已知局限

- **旧时限语义 + 高争抢 → 超时偏多**：mean_timeouts ≈ 10.8–11.6、40/40 天有
  超时（合计 464/432 次 ≈ 6% 决策），是「负载 180–200 争抢 + 旧版越限仍生效
  （先接单后记超时）」的叠加，不代表官方档在干净 CPU 上的预算表现；n=90 无竞争
  探针未执行（正式批次 `*_f` 已在服务器运行，补跑会争抢正式臂 CPU），以 n=60
  单进程基准（aug=8 ≈0.47s）+ 本批次实测超时作为计时证据（smoke --probe90
  已备好，供正式批次复核用）。
- duplicate_service 未复现（两臂 hard_feasible_rate=1.0），但该缺陷存在且
  排序依赖，正式批次仍以主线修复后的 replanner 为准。
- p_c=(5,10,15) 下拒绝损失主导（≈133 单/天被拒）：utility 均值 −25.2，
  与 A-v1「p_c>0 使 reject-loss 主导」的既有结论一致；主信号看 p_c=0。
- pairing identity_verified=False：基线 gate.json 为 Q-04 身份前旧批次。
