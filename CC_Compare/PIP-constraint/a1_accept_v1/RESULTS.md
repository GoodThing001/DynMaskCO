# PIP-constraint A-v1 可比性复核（2026-10-13，判死复认）

> 本目录只新增探针与文档；`CC_Compare/PIP-constraint/**` 官方源码**未改动**。
> 原始日志：`/tmp/pip_probe.log`。

## 0. 结论摘要

| 问题 | 旧记录 | 本次复核 |
|---|---|---|
| 本地资产 | 「README + LICENSE，完整代码未 clone」 | **更正**：官方三个实现完整存在（POMO+PIP / AM+PIP / GFACS+PIP + baselines + data） |
| 训练产物/权重 | 「本地无 PIP 训练产物」 | **部分更正**：48 个 `.pt` 中 **43 个是合法 PyTorch 归档且可加载**；2 个 0 字节；3 个损坏 |
| 任务形态（车辆数/容量） | 「单车辆 TSPTW，缺容量/多车」 | **复认成立**：两个 env 的 `num_vehicles/vehicle_id` 与 `capacity` 出现次数均为 **0** |
| 是否进入 A-v1 40 天正式臂 | 判死 | **维持不进入主表**，但标签从「无权重/未 clone」更正为「**单路线任务形态不匹配（资产可用）**」 |

## 1. 资产清单（更正旧评估）

```
CC_Compare/PIP-constraint/
├── POMO+PIP/    envs{TSPTWEnv.py, TSPDLEnv.py}, models/SINGLEModel.py,
│                Trainer.py / Tester.py / train.py / test.py / utils.py,
│                pretrained/TSPTW/{tsptw50,tsptw100}_{easy,medium,hard}/POMO_star[_PIP|_PIP-D]/
│                pretrained/TSPDL/...                       ← 权重文件存在
├── AM+PIP/      nets/ (attention_model, graph_encoder, pointer_network),
│                problems/tsptw/ (problem/state/baseline/gurobi/reader),
│                utils/ (含 load_model), eval.py / train.py / run.py / options.py,
│                pretrained/tsptw*.pt                       ← 部分权重损坏（见 §3）
├── GFACS+PIP/   TSPTW/ (C++ 局部搜索) + aco/net/train/test + lkh.py
├── baselines/   LKH_baseline.py（支持 CVRPTW/DCVRP 等，但只是**基线脚本**）、OR-Tools_baseline.py
├── data/        TSPTW/, TSPDL/
└── dcc_vrp/     ★ 本项目**早前**留下的适配说明（见 §5），非官方代码
```

## 2. 任务形态证据（车辆数 / 容量字段检查，**复认判死依据成立**）

`probe_pip_assets.py` 对本地实现做字段级扫描：

| 文件 | `num_vehicles` / `vehicle_id` 出现次数 | `capacity` / `vehicle_capacity` 出现次数 | POMO 多起点 |
|---|---|---|---|
| `POMO+PIP/envs/TSPTWEnv.py` | **0** | **0** | 是（`pomo_size`） |
| `POMO+PIP/envs/TSPDLEnv.py` | **0** | **0** | 是（`pomo_size`） |

- `POMO+PIP/envs/` 只有 `TSPTWEnv.py`、`TSPDLEnv.py`（**没有** CVRPEnv / VRPTWEnv / OVRPEnv）；
- `AM+PIP/problems/` 只有 `tsptw`；
- 解码形式 = 「单车辆访问全部节点一次」的 TSP 式自回归（POMO 多起点），
  约束由**时间窗**（TSPTW）或**吃水/载重上限**（TSPDL：`current_load + demand > draft_limit`）
  表达，**没有车队、没有 depot 回程分隔、没有每车容量字段**。
- A-v1 需要：15 辆车、容量 50、200 单 → 总需求 ≫ 单车容量，**必须多车**；
  PIP 的掩码目标结构（单路线可行性）不覆盖 A-v1 的决策空间。

（`baselines/OR-Tools_baseline.py`、`baselines/LKH_baseline.py` 里确实有
`num_vehicles/vehicle_capacity/CVRPTW` 字样，但那是**第三方基线脚本**，不是 PIP 的
学习方法；PIP 的学习机制本身仍只覆盖单路线。）

## 3. 权重可用性（真实加载实测；旧记录「无训练产物」不成立）

服务器侧全量完整性扫描（`zipfile` 归档校验）：

```
PIP .pt 总数=48  0字节=2  可解压(zip)=43  损坏(非zip)=3
```

**可加载（真权重）**，以 `torch.load` 实测：

| 权重 | 结果 |
|---|---|
| `POMO+PIP/pretrained/TSPTW/tsptw100_hard/POMO_star_PIP-D/epoch-10000.pt` | `LOAD OK keys=['model_state_dict']` |
| `POMO+PIP/pretrained/TSPTW/tsptw50_hard/POMO_star_PIP-D/epoch-10000.pt` | `LOAD OK keys=['model_state_dict']` |
| `POMO+PIP/pretrained/TSPDL/tspdl100_hard/POMO_star_PIP-D/epoch-10000.pt` | `LOAD OK keys=['model_state_dict']` |

进一步用官方 `models/SINGLEModel.py` 做**严格架构对账**（`tsptw100_hard/POMO_star_PIP-D`）：

```
official SINGLEModel 参数张量=86  ckpt 张量=90
missing=0  unexpected=4  shape_mismatch=0
unexpected = ['decoder.Wk_sl.weight','decoder.Wv_sl.weight',
              'decoder.multi_head_combine_sl.weight','decoder.multi_head_combine_sl.bias']
```

→ 86/86 重叠张量**逐形状吻合**；4 个「多余」键是 PIP-D 的**辅助双头解码器**参数
（本探针未开启官方 `is_train_double_head_decoder`/`W_*_sl` 开关），开启后即为 strict 干净加载。
**即：PIP 的真实权重可用，不是「无权重」。**

**损坏清单（3 个 + 2 个空 args.json）**：

| 路径 | 状态 |
|---|---|
| `AM+PIP/pretrained/tsptw100_hard/AM_star_PIP-D/epoch-99.pt` | 2,162,688 B，`PK\x03\x04` 开头但**非合法 zip**（`zipfile`→BadZipFile；torch→`failed finding central directory`）；本地/服务器 sha256 相同（`e9401e96…`）→ 资产本身损坏，非传输问题 |
| `AM+PIP/pretrained/tsptw100_easy/AM_star_PIP-D/epoch-99.pt` | 同上（2,031,616 B） |
| `AM+PIP/pretrained/tsptw100_hard/AM_star_PIP/epoch-99.pt` | 同上（1,703,936 B） |
| `AM+PIP/pretrained/**/AM_star_PIP/args.json`（部分） | 0 字节 → 官方 `load_model` 需同目录 `args.json`，会 `JSONDecodeError` |

（POMO+PIP 的权重目录未发现损坏样本。）

## 4. 最终判定

- **维持「不进入 A-v1 40 天正式臂 / 不进主表」**；
- **判死理由更正**为：任务形态不匹配——PIP 的机制是**单车辆单路线**的约束感知掩码
  （TSPTW / TSPDL），缺多车与容量字段（§2 字段级证据），其可保留机制在 A-v1 的
  多车 + 容量 50 + 揭示即承诺决策空间里没有对应物；**不是**「无代码/无权重」（§1、§3）。
- 用户提出的备选口径「只在『单车单路线』子问题上跑 PIP 并与共享认证拼接」：
  **技术上可行**（权重可用 + env 完整，本轮已验证 `SINGLEModel` 与权重逐形状吻合），
  但这会把 PIP 的掩码框架限定为「单条 TW 路线的可行性掩码」，而 A-v1 的容量/多车/承诺
  全部交给共享 `certify_plan`——此时 PIP 的**核心机制（学掩码防不可行）在多车容量维度上
  没有作用对象**，产物将是「另一个构造型 TW 排序 provider」，与原文贡献不对应。
  **本轮不实施**，记录为「资产可用、机制范围不覆盖，单路线参考版可按需另立项」。
- 论文引用：按「约束感知掩码框架（单路线；多车容量不覆盖，未适配）」文献锚点处理。

## 5. 与既有 `PIP-constraint/dcc_vrp/README.md` 的一致性

仓库里已有一份早前的适配说明 `dcc_vrp/README.md`（结论「无法直接对比，不参与主协议」，
理由 = 单车辆 TSPTW/TSPDL、无容量、需重写解码器）。**本次复核与该结论一致**，
仅更正两处事实：(a) 代码并非「未 clone」；(b) 并非「无权重」（43/48 权重合法可加载）。
