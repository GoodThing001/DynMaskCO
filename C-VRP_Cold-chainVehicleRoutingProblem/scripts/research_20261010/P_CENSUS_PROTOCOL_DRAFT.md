# P-CENSUS-v1：取货「同车可修改尾部重排」headroom 普查（协议草案，登记待执行）

2026-10-10 登记。属于取货树 n3 的前置门，不是新的学习实验。旧 I3、训练、教师归档、D2、最终 20260927/28 均不改写。

## 为什么先做这个

n3 登记的干预是「对已接受订单的可修改尾部做完整 encoder+decoder 掩码重构」。在投入该原型之前，必须先回答一个更便宜、更基础的问题：**在只允许同车重排、不跨车改派、不改动已接受集合的动作集下，是否存在任何可捕获的终局效用 headroom。**

若该动作集本身没有 headroom，则任何学习型后缀重构都无法产生正增益，n3 应以「动作集无 headroom」被剪枝并留档；若有 headroom，则普查给出的逐状态最优改进 `g*` 同时充当学习型臂的**上界**与可行性判据。这是本项目既有的 D1–D7 普查方法论的直接沿用。

## 干预与边界

- 固定：已接受订单集合、每车归属、执行前缀（已服务节点、当前 committed 节点 `committed_next`）、车辆数与容量。
- 可变：每辆车 `mutable_suffix` 中、位于已承诺前缀之后的客户顺序（含是否提前返仓 0 的位置）。
- 显式禁止：跨车改派、改动已接受集合、从未来未揭示节点补位、修改温度或合同参数以制造收益。
- 状态来源：`scripts/simulation/strict_online_env.py` 的 `StrictOnlineEnv`；
  - `Replanner.plan(...)` 写入 `v.mutable_suffix`；
  - `env.get_reserved_customers(vehicles)` = `committed_next` ∪ 所有 `mutable_suffix`；
  - env 每个事件只 commit 每车 suffix 的第一个元素；
  - `env.force_suffix[(event_id, vehicle_id)]` 可安装候选 suffix（干预入口）；
  - `env.snapshot_hook` 提供决策点状态。

## 待固化的接口确认点（执行前必须逐条核对并写入结果）

1. 轨迹结构：`env.run(i)` 返回的 `traces` 需确认能恢复每车的服务序列与各段时间，用于重建候选 suffix 的可行性。
2. 评价器口径：取货线现役口径为 A-v1 效用（收入 − 燃油 − 拒绝损失，品质不货币化），对应 `scripts/evaluation/coldchain_evaluator_a1.py`；旧 `evaluate_coldchain_trace` 是 C0 口径。**必须明确用哪一个并说明理由**，不得两者混用。
3. 硬约束：TW、容量、C0 能耗预算、承诺兑现、鱼可售阈值；候选不可行按 g*=0 处理并单独计数，不静默丢弃。
4. 枚举范围：先做单车内 ≤2 次交换（relocate/swap）的完全枚举作为下界普查；若已有 headroom 即停止，不扩大搜索。
5. 数据角色：使用 `data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz` 做开发普查；确认集必须另注册种子，不得复用被烧过的实例。

## 注册判定与产物

- 主统计量：逐（实例，决定点）的 `g* = max(0, U_keep − U_best_reorder)`，按实例配对聚合，2000 次实例级配对 bootstrap。
- 判定：
  - 若实例级平均 `g*` 的 95% 区间下界 > 0 且均值 ≥ 1%（相对 `U_keep`），则动作集有 headroom，n3 继续，`g*` 作为上界写入 n3 证据；
  - 否则 n3 剪枝，理由写为「同车尾部重排动作集在开发集上无可捕获 headroom」，并保留全部逐状态记录。
- 产物目录：新目录 `results/research_20261010/pickup_tail_census_v1/`，含协议、逐状态 JSONL、结果 JSON。
- 恒为 false：`online_gain_established`、`stable_gain_established`。普查不产生任何方法收益主张。

## 本轮为何未执行

D-ANYTIME-v1 正在本机以**墙钟预算**运行（每个臂 10 秒）。墙钟预算对 CPU 争用敏感，任何并行 CPU 占用都会压低 LNS/ANY 的迭代次数并污染判定。因此普查在下一轮单独执行，且必须同样在无竞争条件下运行。

## 来源

- 动作集定义与禁用项：`research/pickup_resource_probe/N2_RUNTIME_QUALIFICATION.md` 第 35–37 段（分支 `arbor-pickup-resource-20261010@6b53eb5`）。
- 前置阻断项：同文第 28–31 段；`NEXT_INTEGRATION.md` 第 19 段。
- 普查方法论先例：`negative_result_paper/` 的 D1–D7（头腔普查 D1/D2、终局 headroom D3、因果 headroom D5）。
