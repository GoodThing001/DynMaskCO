# RRNCO-RH R0 可适配性判定报告

> 方法：RRNCO（`ai4co/real-routing-nco`，ICLR 2026）接入 strict-online DCC-VRP 的可适配性判定。
> 判定方式：源码分析 + **离线可执行骨架（Stage A.1）**；真实模型尚未运行。
> 边界：不改 `rrnco/` 上游源码、不改 `common/`、不改 `MASKCO_code/`；只读。

## 结论（verdict，PROVISIONAL）

```
PROVISIONAL_GO_RRNCO_ORDERING_RH_D
```

**证据等级：offline-skeleton-executed; real-model-pending。** 本结论批准进入一个小规模 R0.5 可执行验证，**暂不进入 R1 协议冻结**。Stage A 离线骨架 + Stage A.1 收尾修复（provider 单次调用 / SubProblem 无 view 通道 / 禁止默认 EDD / open-new-vehicle 优先 + 精确边界测试 + late-reveal 6/6 complete）已实现并通过；真实模型验证（epoch_199.ckpt 注入、未来扰动、可变规模、贡献对照）待 Stage B 上服务器执行。只有 R0.5 全部通过后，才升为 `GO_R1_RRNCO_ORDERING_RH_D`。

RRNCO 的 `RMTVRPEnv` 是**单一 active-route 状态、隐式同构车队的顺序解码器**：当前位置/时间/载荷都是单组状态，回到 depot 后重置，多路线靠 depot 分隔顺序生成，POMO 起点不是并行多车辆。因此它不能原生承接 strict-online DCC 中多辆车各自不同的 anchor、当前载荷和冻结前缀。排除 `GO_RRNCO_RH_D`、保留 ordering 路线（模型只输出客户排序，外层负责 fleet packing）是合理的。

## 执行状态（Stage A.1 后）

| 验证项 | 状态 |
|---|---|
| mock 三类快照（初始部分揭示 / 行程中间 / 多车冻结前缀） | **EXECUTED_PASS**（`r0_5_snapshots.py` 断言） |
| 公共 Bridge late-reveal（t=0 4 单 + t=5 2 单 → 6/6 complete） | **EXECUTED_PASS**（`tests/test_late_reveal_integration.py`，0 fallback，hard vector 全 true） |
| 离线模块单元测试（subproblem/preference/certificate/coordinator/adapter） | **EXECUTED_PASS**（27 项，两遍） |
| 真 checkpoint 三类快照 | **NOT_YET_EXECUTED**（Stage B） |
| 真模型状态继承 / 未来泄漏 / 可变规模 / 贡献对照 | **NOT_YET_EXECUTED**（Stage B） |

最终候选名：**RRNCO_ORDERING_RH_D**。

## 源码已支持（可写入报告的）

- RRNCO 只有一个 active-route 状态（`env.py:155` `_step`、`env.py:308` `_reset` 的 `current_node`/`current_time`/`used_capacity` 均为单组标量）。
- 多路线依赖 depot 分隔顺序生成；`select_start_nodes` 是 POMO 多起点采样，非并行多车辆。
- 因此无法原生承接异构多车 anchor/load/frozen-prefix。
- 现有 legacy `DCCRMTVRPEnv` 屏蔽输入特征的同时仍用 `_true_*` 参与环境掩码与奖励，不是严格 rolling-horizon（`dcc_vrp/dcc_env.py`）。

## 真模型尚未验证（NOT_YET_EXECUTED，Stage B）

- 真 checkpoint 三类快照 **实际**可行；
- 真模型七项泄漏检查 **实际** PASS；
- anchor/time/load 注入后真模型 **确实**从该状态开始——注入不进静态 encoder（encoder 只读 locs/demand/TW/service/distance/duration），而是经 POMO 强制首步进入 decoder context 与 action mask，未被 reset 覆盖；
- `<100` 节点子问题能被 `num_loc=100` checkpoint **稳定**处理（N = depot + 可选非 depot anchor + pool；N<25 走 `visible_prob_sampling_v1` replacement=True 抽样）；
- pickup 容量语义映射正确（**用原生 `demand_backhaul` 映射**：`demand_linehaul=0`、`demand_backhaul=demand/capacity`、`used_capacity_backhaul=load/capacity`、`backhaul_class=1`、`vehicle_capacity=1`，不把 pickup 强塞进 linehaul）；
- 公共 contract 能「形成并认证动作」（离线已通过 mock，真模型待验证）。

**特别强调（contract 边界）**：公共 `PlanProposal`/bridge 只验证节点集合、重复、车辆键与写回一致性（`common/method_adapter.py:494`），它**不会**替 RRNCO 完成跨车辆分配、冲突消解和 suffix 构造。这部分必须由新适配器明确实现；否则最终测到的可能主要是外层启发式，而不是 RRNCO。

**`has_future_reveal` 布尔泄漏边界**：`DecisionView` 暴露 `has_future_reveal`（`method_adapter.py:80`）。「未来数量任意改变不影响决策」只有在**该布尔值保持不变**时成立；「0 个未来订单」与「至少 1 个未来订单」之间是否允许产生差异，需协议明确规定。

## 三类人工快照（mock EXECUTED_PASS / 真模型 NOT_YET_EXECUTED）

| 快照 | 需求 | mock 结果 / 真模型状态 |
|---|---|---|
| 初始部分揭示 | 未来订单不进模型 | mock 通过（可见子问题结构排除未来）；真模型待 Stage B |
| 行程中间 | 继承 anchor/time/load | mock 通过（注入单组状态）；真模型待 Stage B |
| 多车混合 + 冻结前缀 | 异构多车 | mock 通过（逐车子问题 + 协调器）；真模型待 Stage B |

## 七项泄漏检查（离线 mock 已执行 / 真模型待 Stage B）

| 泄漏点 | legacy（不合规） | ORDERING（离线已执行） |
|---|---|---|
| 模型输入 | feature 屏蔽但节点在 `locs` | 未来节点不进 `locs`（结构缺席） |
| 节点集合 | 全部 50 客户 | depot+anchor+可见 |
| 归一化 | `normalize=False` | 同左 |
| action mask | `_true_*` 真值泄漏 | 只对可见客户 |
| multi-start 数量 | 可见客户数 | 按可见客户数 |
| reward/route | `_true_*` 真值 | 只对可见子问题 |
| evaluator 对齐 | 无（自报 cost） | 由 common runner 重放 |

## 后续（R0.5 可执行验证，见下）

---

## R0.5 可执行验证清单（下一步，不跑正式 DEV 数据）

1. 实现逐车 visible-only 子问题构造器（depot + 当前 anchor + 当前已揭示候选）。
2. 实现 anchor/current_time/current_load 注入，并验证模型第一步确实从注入状态开始，未被 reset/decode strategy 覆盖。
3. 明确定义「RRNCO 偏好」：首步 logits / 完整解码次序 / 候选排名 / 边际分数，**只能固定一种**。
4. 实现确定性车队协调器：多车竞争裁决、tie-break、每车 suffix 生成、未分配订单处理、无安全动作 fallback。
5. 用真实 `epoch_199.ckpt` 运行三类人工快照。
6. 未来信息变形测试：改坐标/需求/TW/温区/揭示时间；增减未来订单但保持 `has_future_reveal=true`；单独记录 `false↔true` 是否属允许暴露信息。
7. 覆盖可变规模：pool 客户 1/2/5/10/25/50（模型节点数 N = depot + 可选非 depot anchor + pool；N<25 时 `visible_prob_sampling_v1` replacement=True）+ anchor-only/无可服务订单，检查 shape/NaN/死循环/非法动作。
8. 独立计算器验证 pickup 容量（服务后载荷增加、回 depot 卸载），不依赖 RRNCO 自身判断。
9. 最终 suffix 经公共 evaluator 重放，执行轨迹/完成订单/距离/TW/容量完全一致。
10. 模型贡献对照：真实 ckpt vs shuffled/uniform 偏好 vs 固定启发式——至少证明改变模型偏好会改变实际决策。
11. 前后复验 PyVRP/OR-Tools 冻结状态；确认未改 `rrnco/`、`common/`、`MASKCO_code/`。

### R0.5 放行标准

- 全部通过 → `GO_R1_RRNCO_ORDERING_RH_D`；
- 若模型实际只输出订单排序 → 改名 `RRNCO-Ordering-RH-D`；
- 任一以下情况停止：注入状态被重置/忽略、可变规模不可靠运行、当前动作受未揭示订单影响、pickup 容量映射不成立、协调器决定几乎全部动作而真实模型偏好不起作用。

### R1 并行整理（只读，不冻结三层身份）

可并行整理上游 commit、许可证、`epoch_199.ckpt` 哈希、Torch/RL4CO 版本清单；但**不要先冻结三层身份**——子问题编码、偏好提取、车队协调规则都会进入 control/analysis 身份，必须等 R0.5 定型后再冻结。
