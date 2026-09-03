# DynMaskCO 论文总规划 v4

> 修订日期：2026-09-03  
> 本版覆盖 v3 中“主论文先降为 dynamic CVRPTW”和“以独立 CostAware scorer 为最终主线”的选择。

## 一句话主线

> **DynMaskCO-CC 将 MaskCO 的掩码—重构—迭代改进范式，从静态组合优化扩展到严格非预知的动态冷链物流：在订单事件发生时掩码受影响的可变车队—路线决策，以当前温度、品质、能耗和车队状态为条件，生成经约束认证的补救动作，并用反事实终局冷链效用训练。**

## 四条不可更改的定位

1. **研究对象是动态冷链物流优化**，CVRPTW 只是其时间窗与容量约束底座。
2. **方法来源是 MaskCO**，最终方法必须保留掩码、重构、迭代改进以及对应归因；只用 frozen encoder 加 MLP 不足以构成最终贡献。
3. **cost-aware preference 是训练机制**，用于把 masked generation 从“模仿路线结构”改为“对齐可部署动作的终局冷链效用”，不是论文标题级目标。
4. **负结果必须保留**：JF2/HFR 说明旧监督与部署效用错位，不得改写成 route signal 本质无用，也不得伪装成正结果。
5. **主实验是动态冷链取货回仓**：车辆空载出发，`service_finish` 取货入舱，`return_arrival` 卸货并关闭；不把该协议描述成 depot-to-customer 配送。

## 当前状态

P0-R、P0-S、P0-A、P0-U 已完成。pickup-to-depot 运营语义已冻结；当前进入 **C0 冷链评价闭环**，实现 cargo manifest，并统一物理单位、温度状态转移、品质衰减、能耗累计和 execution-trace evaluator。C0 通过后依次运行 O0-D 与 O0-CC；只有动作空间同时具备可行性和冷链 headroom，才开始学习模型。

## 权威文档

| 文档 | 唯一职责 |
|---|---|
| [科研方法创新主控文档.md](科研方法创新主控文档.md) | 研究问题、最终方法、贡献和 Go/No-Go |
| [代码实现蓝图.md](代码实现蓝图.md) | 后续写代码时的文件、接口、顺序和验收标准 |
| [实验执行手册.md](实验执行手册.md) | 实验 DAG、产物、阶段 Gate |
| [评估口径.md](评估口径.md) | 服务、距离、冷链效用和统计的硬契约 |
| [理论形式化.md](理论形式化.md) | 在线状态、掩码、动作、热状态与效用定义 |
| [实验细节.md](实验细节.md) | 特征、损失、数据、消融和矩阵 |
| [风险与解决预案.md](风险与解决预案.md) | 故障定位、回退和停止规则 |
| [执行进度表.md](执行进度表.md) | 唯一实时进度与下一项任务 |
| [实验结果总表.md](实验结果总表.md) | 已确认结果和待填主表 |
| [决策记录.md](决策记录.md) | 方向变化及不可逆决策 |
| [基线对比.md](基线对比.md) | 公平基线与对比轴 |

若发生冲突：科学定位服从主控文档，代码接口服从实现蓝图，评价服从评估口径，当前阶段服从执行进度表。

## 唯一执行顺序

```text
已完成：P0-R → P0-S → P0-A → P0-U

当前：C0 冷链状态与评价闭环
  ↓
O0-D 纯距离动作空间 oracle
  ↓
O0-CC 冷链效用动作空间 oracle
  ├─ No-Go：先扩 action/lookahead，不训练网络
  └─ Go
      ↓
M0 frozen-MaskCO 效用表征探针
      ↓
M1 DynMaskCO-CC 效用对齐事件掩码重构
      ↓
M2 最多 1–2 轮 on-policy masked data aggregation（按需）
      ↓
A1 MaskCO 身份 + 冷链机制 + 动作耦合归因
      ↓
E1 多场景/EDoD/温度压力/规模/5 seeds
      ↓
T1 VAL 冻结后的 one-shot TEST
```

## Paper-ready 最低条件

- C0 的 pickup/manifest、单位、单调性与 trace parity 共 15 项测试通过；
- O0-CC 在 100% service 下证明现有动作空间对冷链目标有显著 headroom；
- M1 在严格在线完整轨迹上优于 JF1-H-CC，而非只提高离线标签准确率；
- `real mask > no-mask/shuffle mask`，`MaskCO pretrained > random/generic`；
- `cold-chain utility > distance-only utility` 在冷链指标上成立，且距离退化受控；
- 9-cell 动态矩阵、温度压力矩阵、100-node、5 seeds 和一次性 TEST 完成；
- 论文中始终分开报告 `distance_cost` 与 `coldchain_cost`。

## 证据等级

- **A**：源码、测试与 raw artifact 均可复核；
- **B**：来源记录一致，但缺完整 raw artifact；
- **C**：已定义、尚未实现；
- **P**：探索性结果，协议或复核尚未完整。

当前 P0-R/S/A/U 为 A；JF2/HFR 主数字主要为 B；C0/O0/M0/M1 为 C。任何 C 项不得写成已经实现或有效。
