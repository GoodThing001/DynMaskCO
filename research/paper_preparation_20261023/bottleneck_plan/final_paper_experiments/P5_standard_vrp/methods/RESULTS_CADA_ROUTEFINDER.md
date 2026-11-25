# CaDA / RouteFinder 接入 standard_vrp 统一外部工作器 —— 结果报告

日期：2026-11-20（远端验证 UTC）。执行者：P5 外部方法工作器子代理。

## 1. 每方法 × 每任务状态

| 方法 | 任务 | 状态 | 验证 (ok, K/D) | 耗时 | 备注 |
|------|------|------|----------------|------|------|
| CaDA | CVRP X-n101-k25 | **OK** | ok=True, K=28, D=29197.0（逐边取整）/ 29203.033（原始） | 17.1 s | BKS=27591，gap≈5.8%，在预期 28k–34k 内 |
| CaDA | VRPTW C101 | **OK** | ok=True, K=10, D=828.9368669428341 | 15.5 s | 达到已知最优 K=10 / D=828.9369 |
| RouteFinder | CVRP X-n101-k25 | **OK** | ok=True, K=27, D=28940.0（取整）/ 28938.907（原始） | 43.9 s | gap≈4.9%，与作者官方 notebook 的 29035 同量级 |
| RouteFinder | VRPTW C101 | **OK** | ok=True, K=10, D=828.9368669428341 | 38.0 s | 达到已知最优 |

确定性：seed=1 复跑 4/4 与 seed=0 逐位一致（贪心解码 + 固定 8 增广 + 固定 POMO 起点）。
所有评价均用 standard_vrp 独立评价器复算（evaluate_cvrp rounded=True/False、evaluate_vrptw），
容量/时窗/覆盖全部可行；affinity=[30] 或 [31]（taskset 单核证据）。
无 UNSUPPORTED / FAILED 组合（支持矩阵内组合全部 OK；不支持组合由 solve() 抛
NotImplementedError 登记 UNSUPPORTED）。

## 2. 模块路径

远端（已部署、经终态复跑验证）：
- `/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/methods/cada.py`
- `/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/methods/routefinder.py`
- 配套：同目录 `_mtvrp_common.py`（共享 MTVRP 环境/输入构造/路线切分）、
  `_vendor/`（tensordict 兼容层、entmax 桩、rl4co 0.6.0 MIT 推理子集）
- 验证证据：`/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01/verify_cada_rf/`
  （verify_runner.py、smoke_test.py、job 脚本、res_*.json）

本地镜像（与本目录 cada.py/routefinder.py 同源）：
- `research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/P5_standard_vrp/methods/cada.py`
- 同目录 `routefinder.py`、`_mtvrp_common.py`、`_vendor/`
（远端 37 个文件经 SHA256 核验与本地逐字节一致）

## 3. 关键口径

- 作者资产身份：CaDA 上游 CIAM-Group/CaDA commit b9868e1（policy.pt=162 张量，
  100/model.py+config.yaml+utils/functions.py 与本地归档 SHA 一致）；RouteFinder 上游
  ai4co/routefinder commit cc3ab07（policy.pt=61 张量，与 rf-transformer-100.yaml 结构对应）。
- **口径勘误**：任务描述 "CaDA=Collaborative Diffusion Model" 与作者仓库不符——CaDA 为
  Cross-Problem Routing Solver with Constraint-Aware Dual-Attention（arXiv:2412.00346），
  官方推理是 POMO 式多起点 + 贪心解码，仓库无扩散代码；本适配因此没有（也不需要）
  扩散采样步数与噪声种子。
- 输入构造（两模型同源 MTVRP 语义，逐式复刻作者 env）：
  - 坐标：仓库+客户整体各向同性 min-shift / max-span 归一化（CaDA 作者 normalize_coord）；
    RouteFinder 作者 notebook 用逐轴 min-max，本适配改用各向同性口径以保证
    VRPTW 时窗约束与真实问题严格等价（报告注明此差异）；
  - 需求 demand/capacity（车辆容量=1，作者 scale_demand）；
  - VRPTW 时间轴 s = 4.6 / depot_window_end（作者 max_time=4.6）、speed=1/(s·scale)，
    模型内时间=s×真实时间，TW/服务/回仓约束严格等价；CVRP 取 TW[0,inf]、service=0；
  - CaDA p_s_tag=[1,0,TW,0,0,(n+1)/2000]；RouteFinder 特征走作者 init/context 嵌入。
- 解码：CaDA = 作者 VRPModel.forward 复刻（eval 贪心、POMO 全 100 起点、dihedral8×8 增广、
  best-by-get_reward）；RouteFinder = 作者 evaluate() 口径
  policy(phase='test', decode_type='greedy', num_starts=100) + 8 二面体增广。
- 种子：seed 仅 torch.manual_seed（接口统一）；greedy+固定增广+固定起点 → 结果确定
  （seed=0/1 复跑逐位一致已证）。
- 环境依赖：MASKCO_env（torch 2.6.0+cpu）缺 tensordict/rl4co/torchrl，本适配在方法目录
  内嵌纯 torch 垫片（tensordict 最小兼容层 + rl4co 0.6.0 MIT 裁剪子集，逐字取自服务器
  cc_compare 环境同版本源码，只裁剪未用模块），不改 MASKCO_env、不改任何全局状态。

## 4. 合规声明

- 未修改作者仓库：两个 assets 目录（neural_assets/CaDA、neural_assets/RouteFinder）全程只读，
  仅导入其 model/routefinder 包与 policy.pt；作者 model.py/config.yaml 与本地归档 SHA256 一致。
- 只用空闲核：所有验证经 taskset -c 30（CaDA）与 31（RouteFinder）单核执行，
  torch.set_num_threads(1) + OMP/OpenBLAS/MKL/NUMEXPR=1；0–25 核未使用。
- 未启动批次：仅 2 个单核 tmux 作业（每方法顺序跑 2 任务）与 3 次短冒烟，全部已退出，
  tmux has-session 确认无残留会话/进程。
- 未触碰父任务文件：远端 methods 目录中 am.py/pomo.py/car.py/deepaco.py 等既有文件经
  SHA256 核验未被改动；误上传的 mvmoe.py/rrnco.py 已删除恢复原状。

## 5. 备注（供父 agent 决策）

- 统一工作器在线预算 ≤60s：CaDA（15–17s）与 RouteFinder（38–44s）均在预算内。
- 若需进一步逼近 BKS（CVRP gap 4.9–5.8%），可考虑作者 CVRPLIB 口径的 32 任务提示
  集成（CaDA test_lib）或 RouteFinder 的逐轴归一化口径；当前口径以正确性优先。
- 验证脚本与结果 JSON 保留在远端 verify_cada_rf/ 与本地 methods/../_probe/results/。
