# MVMoE（Routing-MVMoE, POMO-VRPTW）A-v1 接单臂结果

> **⚠️ 诊断批次（时限修复前）**：本批次接单臂按主线指令标记为诊断批次，
> 正式重跑由主线统一执行，本目录产物保留不删。数字照实记录，不作正式裁决依据。
> 版本细节（两臂跑在**不同** replanner 版本上，如实记录）：
> - `mvmoe_0`（p_c=0）：加载旧版 `rrnco_accept_replanner.py` sha `e3094bd9…`
>   （「先提交接单、再记录超时」；运行期间主线恰好上传修复版 → 驱动封存
>   `source_stable=False`）。34 次越限决策（max_decide 15.18s）在旧语义下
>   **仍被接单**，仅记录 timeout → p_c=0 数字相对正式口径略偏高（≤34 单）。
> - `mvmoe_51015`（p_c=(5,10,15)）：主线误杀后 17:07 重跑，加载**修复版** sha
>   `2c4dee57…`（越限=保旧计划+拒单；`source_stable=True`）；全程无越限决策
>   （max_decide 5.88s、mean_timeouts=0）→ 该臂轨迹与正式口径一致。
> - 两臂均未复现 duplicate_service：hard_feasible_rate=1.0、failures=[]（主线
>   通报该机制在 POMO 诊断臂 40/40 复现；本臂未触发，如实报告）。

## 方法配置（provider = CC_Compare/MVMoE/a1_accept_v1/mvmoe_provider.py）

- checkpoint：`pretrained/pomo_vrptw_n100/epoch-5000.pt`（md5 `977ba94728081bb1d86bb9a00c986c3b`，
  本地/服务器一致；problem=VRPTW，epoch=5000）
- 模型：官方 `SINGLEModel`（POMO），官方超参（embedding_dim=128 / encoder_layer_num=6 /
  qkv_dim=16 / head_num=8 / logit_clipping=10 / ff_hidden_dim=512 / norm=instance /
  norm_loc=norm_last），`load_state_dict(strict=True)`
- 实例构造：官方 `_solve_cvrptwlib` 缩放约定（scaler = max(max_coord, depot_tw_end/3)；
  坐标/TW/服务同除 scaler；需求 = raw/capacity；speed=1；depot TW=[0,3]）
- 解码：官方 POMO rollout（reset → pre_forward → pre_step → argmax 贪心），
  **变长直接喂子问题规模（depot + pool，无 phantom）**；步数上限 3×(m+2) 防死锁，
  未访问客户按 EDD 补尾保证精确覆盖全池
- 超参档：`aug_factor=1`（官方 CLI 档位 {1,8}）、`multistart`（pomo_size = 池规模，
  官方 VRPTW 评测口径）。选档依据：官方默认 aug=8×pomo=n 在池 60 时约 4.6s/决策
  （单进程烟测），池 90 + 6 worker 争抢下估计超 10s/决策 → 按协议「超 10s/决策取官方空间
  内轻档」取 aug=1×pomo=n；烟测池 60 解码 0.57s、池 90 无竞争实测见 §无竞争探针
- 推理：CPU（`--provider-device cpu`），`torch.set_num_threads(8)`；
  跨车同池缓存（同决策多车子问题池相同 → 每决策一次解码）
- 已知局限：anchor / ready_time / current_load 不注入（POMO 解码器无原生起点状态
  注入；真实 anchor 可行性由 dcc_rh_v4 协调器 + certify_suffix + C0 certify_plan 兜底）

## 服务器烟测（smoke_mvmoe.py，全 PASS；mock 用真实风格客户 id 101+ 回归）

| 用例 | 结果 |
|---|---|
| 加载 checkpoint（problem/epoch） | VRPTW / 5000，4.9–7.2s（预算外预加载） |
| pool=5 / 25 / 60 多起点 aug1 解码 | 0.39 / 0.24 / 1.05s，排序精确覆盖全池 |
| 空池 | 返回 `()` |
| 紧 TW 恶意客户（官方 mask 永久不可达） | 步数上限内终止，EDD 补尾后仍覆盖全池 |
| 跨车同池缓存 | 命中，两次排序逐元素一致 |
| single-aug8 / single-aug1（池 60） | 0.43 / 0.12s |
| 1 天真实数据探针（200 决策） | n_fail=0，reject 仅 certify_budget(146)+deferred，max_decide 0.36s |

## 无竞争 n=90 单决策耗时探针（probe_n90.py，正式定档依据）

- 无竞争窗口：**17:23–19:53 观察 2.5h 未出现**（load1 全程 170–250，xpomo/xcada/xam/
  a1c1r/a1s3r2 + 主线正式 GPU 批次持续占满 96 核）→ 以 19:54 实测（load1=184.3，
  争抢上界口径）记录，正式定档已由 GPU 烟测接管（见下节）。
- 当前选档 aug1+multistart，n=90，3 次墙钟 / 中位数：**1.64 / 1.67 / 1.68s，中位 1.67s**
  （争抢下仍 ≪ 7s 阈值与 10s 决策预算 → 维持当前选档，无需降档）
- 更轻档实测（官方空间内，同 load）：aug1-single 中位 0.34s；aug8-single 中位 0.75s
- 探针实现备注：跨车同池缓存会在重复调用同 sp 时命中（首版探针计时 0.00s）→
  已改为每轮 `_cache.clear()` 强制真解码。

## GPU 正式批次前置烟测（mvmoe_gpusmoke，2026-09-29 18:53）

主线正式两臂候选 = GPU 1。GPU 1 空闲后（/tmp/xformal_gpu.done ALLDONE 18:51；
nvidia-smi GPU1 21MiB/0%）跑 1 天 GPU 烟测：

```
CUDA_VISIBLE_DEVICES=1 /home/hzeng/envs/cc_compare/bin/python \
  scripts/evaluation/run_a1_external_accept.py --solver ordering --provider mvmoe \
  --provider-ckpt <ckpt> --provider-device cuda \
  --baseline-from results/a1_step2_gate_c1_20260926/gate.json \
  --dev-instances 1 --workers 1 --penalty "p_c=0" \
  --out results/a1_ext_extra/mvmoe_gpusmoke
```

| 指标 | 结果 |
|---|---|
| n_solves / n_fail | 200 / **0** |
| max_decide_s | **2.51**（含 coordinate+certify；< 10s 预算） |
| mean_solve_time_s | 0.083（GPU 解码 ~83ms/决策，CPU 为 ~0.7s） |
| hard_feasible_rate / failures | 1.0 / {} |
| mean_timeouts / timeout_days | 0.0 / 0 |
| served / rejected | 54 / 146（reject_reasons={certify_budget:146}） |
| source_stable | True |

provider CUDA 兼容确认：`_build_data` 全部张量显式 `device=self.device` 构造（修复了
初版 torch.tensor CPU 默认构造——官方 test.py 靠 set_default_tensor_type 兜底、
provider 未复制该机制）；checkpoint map_location='cpu' 加载后 .to(device)；
SINGLEModel/VRPTWEnv device 传递一致；无 .cpu() 硬编码。CPU 路径修复后重跑烟测仍
SMOKE OK。结论：**MVMoEProvider 支持 device='cuda'，GPU 正式两臂（mvmoe_0_f /
mvmoe_51015_f）可点**。

## 两臂结果（开发集 seed 20260926，40 天，--workers 6，CPU）

命令（C-VRP_Cold-chainVehicleRoutingProblem 目录，tmux `xmv`，脚本
`CC_Compare/MVMoE/a1_accept_v1/run_mvmoe_arms.sh` + `run_mvmoe_arm2.sh` 重跑）：

```
/home/hzeng/envs/cc_compare/bin/python scripts/evaluation/run_a1_external_accept.py \
  --solver ordering --provider mvmoe --provider-ckpt <ckpt> --provider-device cpu \
  --baseline-from results/a1_step2_gate_c1_20260926/gate.json --dev-instances 40 \
  --workers 6 --penalty "p_c=0" --out results/a1_ext_extra/mvmoe_0
（第二条 --penalty "p_c=(5,10,15)" --out results/a1_ext_extra/mvmoe_51015）
```

### 主表（arms["ordering_accept"]，utility_all_days 为 day-clustered bootstrap）

| 指标 | p_c=0（mvmoe_0） | p_c=(5,10,15)（mvmoe_51015） |
|---|---|---|
| utility_all_days mean | **1267.888** | **−38.576** |
| utility_all_days 95% CI | [1242.319, 1292.385] | [−165.560, 83.446] |
| n_days / finite_n | 40 / 40 | 40 / 40 |
| adjudicable | True | True |
| hard_feasible_rate | 1.0（failures=[]，无 duplicate_service） | 1.0（failures=[]，无 duplicate_service） |
| mean_timeouts / timeout_days | 0.85 / 6 | 0.0 / 0 |
| served / rejected（40 天合计） | 2694 / 5067 | 2675 / 5086 |
| elapsed_total_s | 6754.4 | 4428.0 |

### 配对差（ordering_accept − 基线臂，按天配对，n_paired=40，adjudicable=True）

| 配对 | p_c=0 | p_c=(5,10,15) |
|---|---|---|
| minus cond_hist | −216.649 [−244.463, −189.448] | −347.146 [−389.148, −304.881] |
| minus uncond_hist | −176.838 [−218.821, −131.015] | −275.656 [−343.062, −201.652] |
| minus explicit_feat | −179.627 [−220.091, −137.279] | −283.380 [−350.917, −213.984] |

> 配对 identity_verified=false（reason=baseline_identity_missing：步骤 2 基线
> gate.json 早于 Q-04 严格身份字段）→ 正式裁决留待主线正式批次；与诊断批次标注一致。

### solver_meta

| 字段 | p_c=0 | p_c=(5,10,15) |
|---|---|---|
| n_solves / n_fail | 7761 / 0 | 7761 / 0 |
| solve_time_s / mean_solve_time_s | 5404.4 / 0.696 | 3351.6 / 0.432 |
| max_decide_s | 15.18（越限，旧语义仍接单） | 5.88 |
| reject_reasons | {timeout: 34, certify_budget: 5030, deferred: 3} | {certify_budget: 5083, deferred: 3} |

### 坏批次证据（provider v1 覆盖 bug，只作诊断，不引用；产物保留）

`mvmoe_0_providerbug_v1` / `mvmoe_51015_providerbug_v1`：n_solves=7761、
n_fail=7721、reject_reasons={'fallback': 7721}、n_served 合计 40（每日 1 单）、
utility 18.8 / −1914、hfr=1.0、failures=[]。

## 适配日志（错误与修复）

1. 任务 tar 清单写 `utils/` 目录，实际本地为 `utils.py` 单文件 → tar 改列 `utils.py`；
   provider 不 import `utils`（会拖入 scipy/matplotlib/tqdm），直接
   `from models import SINGLEModel` / `from envs import VRPTWEnv`（均为 torch-only）。
2. `_sftp_put.py` 要求本地文件先在仓库相对路径（`CC_Compare/_upload/` 本地目录
   不存在 → 先建目录再放 tgz；远端 `CC_Compare/MVMoE/` 不存在 → 先 mkdir 再解包）。
3. 服务器烟测一次通过（解码计时远低于预算，无需降档）。
4. **provider v1 覆盖 bug（已修复）**：`order()` 的 EDD 补尾用 local-index 集合比对
   真实客户 id——A-v1 真实客户 id（如 108）≠ 数组下标（初版烟测 mock 恰好 id==index，
   未暴露）→ 每个决策抛 `ValueError: ordering 未精确覆盖全池` → 适配器 EDD fallback →
   臂退化为「每日仅 1 单」拒单臂（见坏批次证据）。修复 = 补尾用真实 id 集合
   （`covered_ids`）；烟测升级为真实风格 id（101+）回归后全 PASS；1 天真实数据
   探针复验 n_fail=0。修复后重跑两臂。
5. **主线误杀重跑**：主线排查正式 GPU 臂时 pkill 模式过宽，误杀 tmux xmv
   （DONE_51015 143=SIGTERM）。核对产物：mvmoe_0 完整（40 天，保留），仅
   mvmoe_51015 缺失 → 用 `run_mvmoe_arm2.sh` 于 17:07 重跑 arm 2（17:21 完成，
   DONE_51015_RETRY 0，此时磁盘上已是修复版 replanner `2c4dee57…`）。
6. 批次身份：arm 1 加载 `e3094bd9…`（旧语义，source_stable=False——运行窗口内
   主线恰好上传修复版）；arm 2 加载 `2c4dee57…`（修复语义，source_stable=True）。
   见文件头标注。
7. GPU 定档：provider CUDA 审计发现 `_build_data` 张量默认 CPU 构造（官方 test.py
   靠 set_default_tensor_type 兜底、provider 未复制）→ 改为显式 device= 构造；
   GPU 1 空闲后 1 天烟测 PASS（max_decide 2.51s / n_fail 0 / hfr 1.0），见
   §GPU 正式批次前置烟测。
8. n=90 探针首版计时 0.00s = 跨车同池缓存命中（同 sp 重复调用）→ 每轮清缓存后
   重测；且观察窗口内无低负载时刻，按争抢上界口径记录（见 §无竞争探针）。
