# CaDA A-v1 接单臂结果 —— 诊断 CPU 批次 + 主线 GPU 正式批（合并记录）

> **批次定性**：本代理执行的是**诊断 CPU 批次**（数值仅诊断参考）；主线已在 GPU1
> 统一执行**正式两臂**（cada_0_f / cada_51015_f），其结果一并记录在 §4.2，正式结论
> 以主线批为准。
> - 诊断批次执行时服务器 `rrnco_accept_replanner.py` sha = `2c4dee571477…`（主线
>   部署的**时限修复**：越限 = 保旧计划 + 拒单，推理后与认证后各查一次；
>   **duplicate_service 修复**：pool 排除在途 committed_next）。即诊断批次运行于
>   「双修复后」语义，如实记录 vintage。
> - 四臂均按要求报告 `hard_feasible_rate` 与 `failures` 计数：全部 hfr=1.0、
>   fail_counts={}，**未复现 duplicate_service**。

## 1. 方法口径（零样本跨任务，诚实声明）

- **CaDA（CIAM-Group/CaDA, NeurIPS 2024）原生只支持 TSP/CVRP 等无时间窗问题**。
  本臂把 CaDA 官方 CVRP100 checkpoint 接到 A-v1 排序 provider 接口，**按 CVRP 求解
  排序、忽略 TW**：`tw_start/tw_end/service_time` 不进入模型输入与 action mask；
  温度区 / C0 能耗不在 CaDA 输入空间，不参与排序。
- 真实 TW / C0 可行性由下游冻结 dcc_rh_v4 协调器（`pickup_certificate` 逐车 TW/容量
  认证）+ `OrderingAcceptReplanner.certify_plan`（真实 C0 硬预算）兜底：CaDA 排序
  不满足真实约束时该单被拒（`deferred` / `certify_budget` / `fallback` / `timeout`）。
- 排序返回 `sp.pool_customer_ids` 全池**真实客户 id** 的偏好顺序（精确覆盖全池、
  无 depot/anchor）；空池返回 `()`。

## 2. 官方评测配置（超参数口径）

| 项 | 取值 | 出处 |
|---|---|---|
| checkpoint | `CC_Compare/CaDA/100/result/2024-1121-1355/checkpoint-300.pt`（官方 CVRP100，sha `03eb7ef8…`） | 官方下载 |
| 解码 | **贪心 multi-start**（每个可见池客户一个起点），无采样、**无增广**（官方 a8gap 8× 增广为外包 best-of-8，单次解码即官方基础档） | `100/run.py --test` / `model.VRPModel.forward` |
| best-start 选择 | 官方 `get_reward` 口径：depot→tour→depot 总长最小 | `envs/env.py` |
| 数据约定 | demand/capacity 归一（vehicle_capacity=1）、depot 行 0、coords∈[0,1]²、distance_limit=+inf、time_windows=[0,inf]（encoder nan_to_num 后特征=0）、service=0、open_route=0、speed=1、p_s_tag=[1,0,0,0,0,(n-1)/2000]（官方 dataset()/generator 两套约定下 CVRP 任务码同为 [1,0,0,0,0]） | 官方 generator/dataset |
| 规模 | 变长直接前向（attention 结构），**不补 phantom**；解码步数硬上限 6(n+1)+32，超限抛错 → adapter fallback → 拒单（不静默降级） | 官方 select_start_nodes 按 n 自适应 |
| 设备 | **CPU**（诊断批 `--provider-device cpu`，GPU0/GPU1 均未使用）；正式批 GPU1（§4.2/§6） | 任务指定 |
| 线程 | `OMP_NUM_THREADS=16` + `torch.set_num_threads(16)`（6 worker × 16 = 96 核；运行时设置，非模型配置） | 本 provider |

**与官方解码的两处忠实差异（记录在案，均非自创模型配置）**：
1. multi-start 起点 = **可见池客户**（官方起点数为全部客户；池外 anchor 已预访问、
   不作起点——与冻结 dcc_rh_v4 `PoolStartNodes` 口径一致）；
2. **anchor 车辆注入**：anchor 行预标记 visited、初始 used_capacity =
   min(current_load, capacity)/capacity（回 depot 清零=卸载，官方 env._step 语义；
   RRNCO 后端同款注入）。

解码引擎：torchrl 无关——官方 `MTVRPEnv._step` / `get_action_mask` 的 CVRP 子集数学
在 provider 内逐式复刻（`envs/env.py` 语义对应），encoder/decoder 走官方 `model.py`
原码。**耗时实测**（服务器烟测，CPU，16 线程）：
pool 1/2/5/25/60/90/120 → order() ≈ 0.00(单客户捷径)/0.17/0.06–0.27/0.24–0.37/
0.46–0.65/0.68–1.01/1.67 s。

**无竞争最大池探针（主线定档依据）**：pool≈90、真实风格 id、3 次 wall-clock，
**中位数 0.814s**（reps 0.681/0.814/0.964）。⚠️ 测量时服务器 loadavg≈192
（RouteFinder GPU 臂 + POMO CPU 臂 + 本臂并行），非严格无竞争——正式定档可再测。
按每决策 ~15 辆 replan 车各调一次 order() 折算：**pool≈90 决策级 ≈ 12.2s > 10s
预算**（pool ≲70 时 ≈ 10s 内）。官方参数空间内**不存在更轻的解码档**（贪心 +
全客户 multi-start 已是官方最轻；增广 8× / 采样均更重；multi-start 起点数官方
固定为全客户，削减起点数属非官方自创配置）——故本臂保持官方档、超时如实记录，
轻档替代方案（如起点数削减）由主线定夺。
每决策需对 ~15 辆 replan 车各调一次 order() → **大池（≳70）决策会超过 10s 预算**，
在时限修复语义下按 `timeout` 拒单（如实计入 `reject_reasons['timeout']`）。

## 3. 运行命令与身份

```bash
# tmux xcada（服务器，C-VRP_Cold-chainVehicleRoutingProblem 目录下）
CC_Compare/CaDA/a1_accept_v1/run_cada_arms.sh
# 内含两条（顺序执行）：
/home/hzeng/envs/cc_compare/bin/python scripts/evaluation/run_a1_external_accept.py \
  --solver ordering --provider cada \
  --provider-ckpt ../CC_Compare/CaDA/100/result/2024-1121-1355/checkpoint-300.pt \
  --provider-device cpu --baseline-from results/a1_step2_gate_c1_20260926/gate.json \
  --dev-instances 40 --workers 6 --penalty "p_c=0" --out results/a1_ext_extra/cada_0
# 第二条同参数，--penalty "p_c=(5,10,15)" --out results/a1_ext_extra/cada_51015
```

执行时代码 vintage（服务器 sha256）：
| 文件 | sha256（前 16 位） | 说明 |
|---|---|---|
| `scripts/evaluation/rrnco_accept_replanner.py` | `2c4dee57147766db` | 时限修复 + duplicate_service 修复（主线部署） |
| `scripts/evaluation/run_a1_external_accept.py` | `681905ff5f229fbf` | 冻结 driver |
| `scripts/evaluation/ordering_providers.py` | `5d0da99848495a27` | registry 含 `cada` |
| `CC_Compare/RRNCO/dcc_rh_v4/rrnco_guided_adapter.py` | `f1ef810e46344d55` | 冻结协调适配器 |
| `CC_Compare/CaDA/a1_accept_v1/cada_provider.py` | `9b1a99c8b6df8694` | 本 provider（device 无关版；前版 `cc379311e65aad6f` 为 CPU-only——51015 臂用前版、p_c=0 重跑与 GPU 正式批用新版，CPU 数值路径完全一致） |
| checkpoint-300.pt (CVRP100) | `03eb7ef827587fde` | 官方权重 |

**device 支持**（主线正式批 GPU1 前置确认）：`_load()` 模型 `.to(device)`；`_build_td`/
`_decode` 全部张量 `device=self.device` 构造（无 `.cpu()` 硬编码，checkpoint 仅
`map_location='cpu'` 加载后整体 `.to(device)`）；step/mask/reward 均为 device 无关
算子。CPU 烟测复验无回归（同 seed 同排序输出）。

烟测（服务器，`smoke_cada.py` v3，**非连续真实风格 id** 101/105/120…，断言
`sorted(order) == sorted(pool_customer_ids)`）：加载 9–10.9s（决策预算外）、空池→()、
pool 1/2/5/25/60 + anchor/载重变体全池精确覆盖、解码终止、无 id/index 混淆 → **PASS**。

## 4. 主数字（40 天开发集 seed 20260926；utility_all_days = mean [ci_lo, ci_hi]，n_days=40）

四臂均：adjudicable=True、hard_feasible_rate=1.0、fail_counts={}、
**无 duplicate_service**（执行 vintage 均为 sha `2c4dee57` 双修复，pool 排除在途
committed_next——与 POMO 诊断臂 40/40 复现形成对照，CaDA 四臂均未复现）。

### 4.1 诊断 CPU 臂（本代理执行；诊断批次）

| 臂 | utility mean [CI] | vs cond_hist [CI] | mean_timeouts | reject_reasons | n_served/7761 | mean_solve_time_s | max_decide_s | source_stable |
|---|---|---|---|---|---|---|---|---|
| cada_0 (p_c=0，重跑) | **1279.074** [1254.460, 1303.347] | **−205.464** [−227.671, −182.351] | 30.4/天（39/40 天） | timeout 1216 + certify_budget 3825 + deferred 1 | 2719 | 4.091 | 27.945 | True |
| cada_51015 (p_c=(5,10,15)，主档) | **−387.326** [−450.991, −323.195] | **−695.896** [−804.704, −577.652] | 130.95/天（40/40 天） | timeout 5238 + certify_budget 393 | 2130 | 14.740 | 73.274 | **False**（运行期间磁盘源码被主线后续编辑变更；行为仍为启动时加载的 2c4dee57） |

其余基线配对：cada_0 vs uncond_hist −165.653 [−202.498, −124.112]、vs explicit_feat
−168.442 [−210.184, −123.095]；cada_51015 vs uncond_hist −624.406 [−758.238, −489.343]、
vs explicit_feat −632.130 [−764.792, −494.916]。
（n_paired=40、adjudicable=True；formal_adjudication=False——基线缺身份块。）

**诊断口径**：CPU 官方 multi-start 决策时延在并发竞争（load≈190–300，与 POMO/RouteFinder
/正式 GPU 臂并行）下大幅抬高 → 两臂 timeout 拒单占比 15.7% / 67.5%，效用被超时拒单
主导（51015 主档尤甚）——即「CPU 放不下 10s 预算」的直接诊断证据（第 2 节探针已预告
pool≥70 超限；据此正式批定档 GPU1）。cada_0 与 cada_51015 两次运行并发负载不同 →
超时分布不同、接单轨迹不同，两臂如实并列表述，不跨臂混合口径。

### 4.2 正式 GPU 臂（主线统一点火 GPU1；本代理未执行、未触碰）

| 臂 | utility mean [CI] | vs cond_hist [CI] | timeouts | reject_reasons | n_served/7761 | mean_solve_time_s | max_decide_s | source_stable |
|---|---|---|---|---|---|---|---|---|
| cada_0_f (p_c=0) | **1156.998** [1120.596, 1190.804] | **−327.539** [−365.406, −292.959] | 0（0/40 天） | certify_budget 5278 + deferred 3 | 2480 | 0.567 | **4.595** | True |
| cada_51015_f (p_c=(5,10,15)) | **−190.252** [−327.754, −54.535] | **−498.822** [−554.590, −445.529] | 0（0/40 天） | certify_budget 5278 + deferred 3 | 2480 | 0.541 | **4.442** | True |

其余基线配对：cada_0_f vs uncond_hist −287.728 [−337.499, −235.564]、vs explicit_feat
−290.518 [−340.171, −238.929]；cada_51015_f vs uncond_hist −427.332 [−504.944, −342.135]、
vs explicit_feat −435.056 [−510.417, −353.664]。
（n_paired=40、adjudicable=True；formal_adjudication=False——基线缺身份块，
pairing_identity_verified=False。）

**正式臂要点**：GPU 下 max_decide 4.595/4.442s < 10s → 零 timeout、零 fallback；
两档 p_c 接单轨迹一致（penalty 不影响决策，reject_reasons 同构）；
mean_solve_time_s ≈ 0.54–0.57s/决策。两个正式臂对 cond_hist 的配对差均为负
（−327.5 / −498.8）——CaDA（CVRP 排序 + 全覆盖接单 + 真实 C0 认证）在此口径下
未超过条件历史基线。

## 5. 适配日志（错误与修复，按时间序）

1. **cc_compare env 缺 `entmax`**（CaDA 官方 requirements 含 `entmax==1.2`）→
   `pip install entmax`（装到 1.3，py3-none-any 纯 Python 轮子）。config
   `use_sparse='topk'` 下 entmax 激活函数不被调用，仅 import 需要。
2. **torch 2.11 默认 `weights_only=True` 无法解包 checkpoint**（内含 pickled
   env/rng 状态）→ provider 进程内临时 patch `torch.load(weights_only=False)`，
   加载后恢复（RouteFinder provider 同款）。
3. **checkpoint 的 `env_state_dict` pickle 了 torchrl 0.1 旧 spec 类名**
   （`CompositeSpec` 等在 torchrl 0.13 已改名）→ provider 进程内别名桥接
   `torchrl.data.tensor_specs`（幂等，仅加载期使用，不 import CaDA 的 env 模块）。
4. **`_get_reward` 静态方法缺 `import torch`**（首次烟测 NameError）→ 修复后烟测通过。
5. **MVMoE 教训落地**：provider 输出映射经审计为真实 id（行号→`sp.node_ids[a]`），
   但初版烟测 mock id==index 无法证明 → 烟测升级 v2（非连续真实风格 id），断言
   真实 id 集合相等 → PASS。provider 内 EDD 补尾也用真实 id（`sp.tw_end[sp.node_index(c)]`）。
6. **启动卫生事故**（本地诊断探针导致重复臂）：`bash -x` 前台探针被 SSH 超时杀死后
   遗留孤儿 worker 进程与 tmux 臂并发同写输出目录 → 已按 PID 清理（tmux 树保留、
   孤儿全杀、marker 重截断），最终仅一条 tmux 臂在跑。输出目录无中间文件
   （driver 仅在结束时写 gate.json），无污染。
7. **首臂全 fallback 事故（pool=1 形状崩溃，MVMoE 同型教训）**：第一版 provider
   上线后 40 天 7761/7761 决策全部 `fallback`（utility=0、0 served）。单日复现
   （`repro_one.py`，真实 day 0 + traceback）定位：**pool=1 → num_starts=1 → 官方
   解码器 `unbatchify` 把 current_node 压成 2-D，与 3-D state_embedding cat 报
   `RuntimeError: Tensors must have same number of dimensions`**（官方评测
   num_loc≥50 永不触发该形状；A-v1 每决策首个 reveal pool=1 必触发）。
   修复：pool==1 走单客户捷径直接返回 `(c,)`（单客户排序被强制，官方无法解码
   n=1 无对照口径；记录在本日志）。烟测升级含 pool 1/2 后重跑：day 0 真实复现
   **accepted=54 / rejected=146（全部 `certify_budget`，C0 预算正常口径）/
   n_fail=0 / provider 异常=0 / timeouts=0** → 修复生效后重放双臂。
   （首个 cada_0/gate.json 为事故产物，已被重跑覆盖。）
8. **服务器并发负载与外部进程清理误伤**：首跑 p_c=0 臂于 17:44 被外部 SIGTERM
   中断（exit 143；同窗口 POMO 臂也被杀重启、xrf 会话关闭——疑似他臂/主线进程
   清理按宽 pattern 误伤）。p_c=(5,10,15) 臂 17:45 起正常跑完；p_c=0 臂改用
   独立脚本 `run_cada_arm0.sh` + 独立标记 `/tmp/xcada0.done` 单独重跑（tmux
   xcada0）。已向主线协调：清理进程时避开 `provider cada` 进程。
9. **device 无关重构（主线 GPU 正式批前置）**：provider 张量构造全部改为
   `device=self.device`（前版 CPU-only，device='cuda' 会设备不匹配）。CPU 数值
   路径不变（烟测复验同 seed 同排序输出）；sha `cc379311…` → `9b1a99c8…`。
10. **GPU 正式批**：主线在 xformal_rr（GPU1）完成后统一点火正式两臂
    （run_formal_gpu2.sh；cada_0_f / cada_51015_f，GPU1、--workers 4），本代理
    未执行、未触碰。结果见 §4.2（正式结论以该批为准）。1 天 GPU 烟测亦由主线统一
    执行，本代理未跑。

## 6. GPU 正式批（主线统一执行）

- 正式两臂 `cada_0_f` / `cada_51015_f`（GPU1、`--workers 4`、`--provider-device
  cuda`）由主线统一点火并跑完（输出 `results/a1_ext_extra/cada_0_f` /
  `cada_51015_f`；本代理未执行、未触碰）。数字见 §4.2。
- GPU 实测要点：**max_decide_s 4.595 / 4.442s < 10s 预算**、timeouts 0（0/40 天）、
  fallback 0、mean_solve_time_s 0.567 / 0.541s、hfr=1.0、fail_counts={}、
  source_stable=True；两档 p_c 接单轨迹一致（penalty 不影响决策）。
- provider device='cuda' 审查结论（2026-09-30，sha `9b1a99c8…`）：map_location='cpu'
  加载 → strict load → model.to(device)；全部输入张量显式 device 构造；无 .cpu()
  硬编码；无 set_default_tensor_type 依赖；CUDA 初始化在 fork 后 worker 内
  （与 RouteFinder GPU 臂同款流程）；CPU 烟测复验同 seed 同排序无回归。

## 7. 已知局限

- **零样本跨任务**：CVRP 排序忽略 TW/温度区/C0；排序质量 ≠ 真实可行性，由下游
  认证兜底。跨分布：A-v1 坐标（3 簇中心 + N(0,0.12)，[0,1]²）与 CaDA 训练分布
  （uniform [0,1]²）不同；A-v1 需求 U(1,3)/50 ∈ [0.02,0.06] 位于训练范围
  [0.02,0.18] 的轻尾。
- **CPU 官方 multi-start 在大池超预算**：pool ≳ 70 时 15 车 × order() > 10s，
  时限修复语义下按 `timeout` 拒单（诊断两臂 timeout 占比 15.7% / 67.5%，第 2 节
  实测表）；官方参数空间无更轻档，未降档——正式批据此放 GPU1（max_decide 4.6s 内）。
- **anchor/载重注入为忠实差异**（第 2 节）；起点池化与冻结协调器口径一致。
- 诊断 CPU 批次：数字仅供诊断（含 51015 臂 source_stable=False 的运行期源码变更
  记录）；正式结论以主线 GPU 批（§4.2）为准。
