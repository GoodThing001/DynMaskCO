# 外部基线口径矩阵（V2 公平性审计）

> 只读核对，不修改任何外部方法源码或权重。结论用于决定 V2 哪些外部基线**复用**
> P5-V1 冻结结果、哪些**补齐重跑**、哪些登记为 **UNSUPPORTED**（附具体理由）。
> 复核时间：2026-11-26；复核依据：`CC_Compare/configs/methods.json`（活动树）、
> `P5_standard_vrp/P5_预登记.md` §5 表、`P5_standard_vrp/p5_external_worker.py`（冻结）、
> `P5_standard_vrp/results_external_*`（冻结结果与 status.json）。

## 1. 统一协议前提（A 组，全部外部方法必须满足才可比）

单物理核、在线 60 秒、共同初解（构造式方法除外）、全部候选生成/前向/评价计入预算、
截止后不接纳；CVRP 目标 = 逐边整数舍入口径总距离；VRPTW 目标 = 字典序 (K, D)，
距离双精度、参考值 = CVRPLIB 官方 BKS 路线复算（见 `参考口径核验_V2.md`）。

## 2. 逐方法矩阵

| 方法 | CVRP n=100 | CVRP n>100 | VRPTW 56 | 目标口径一致性 | V2 处理 |
|---|---|---|---|---|---|
| ALNS | 支持 | 支持 | 支持 | 距离 / 字典序均可显式最小化 | **复用** V1（源、初解、计时、目标未变） |
| VNS | 支持 | 支持 | 支持 | 同上 | **复用** V1 |
| OR-Tools | 支持 | 支持 | 支持 | CP-SAT/Routing 显式目标；V1 记为 distance proxy | 复用（作距离代理对照，不并入神经对比） |
| PyVRP | 支持 | 支持 | **支持（整数时间矩阵口径）** | CVRP 距离：一致；VRPTW：整数缩放可产生"仅截断口径可行"解 → TRUNCATED_OK 单列 | **补齐** VRPTW 168 端点（V1 仅 28 例 × 3 种子 = 83 端点）；CVRP 复用 |
| RouteFinder | 支持 | UNSUPPORTED（定长） | 支持 | 作者模型原生目标 | 复用 V1（VRPTW） |
| DeepACO | 支持 | UNSUPPORTED（定长/无大规模权重） | UNSUPPORTED（原仓库无 VRPTW） | — | 仅 CVRP 复用 |
| CaR | 支持 | UNSUPPORTED | UNSUPPORTED（无作者 VRPTW 权重） | — | 仅 CVRP 复用 |
| AttentionModel | 支持 | UNSUPPORTED（定长） | UNSUPPORTED（无作者 VRPTW 权重） | — | 仅 CVRP 复用 |
| POMO | 支持 | UNSUPPORTED（定长） | UNSUPPORTED（无作者 VRPTW 权重） | — | 仅 CVRP 复用 |
| MVMoE | 支持 | UNSUPPORTED | 支持 | 作者多任务含 VRPTW | 复用 V1（VRPTW） |
| CaDA | 支持 | UNSUPPORTED | 支持 | 作者多任务含 VRPTW | 复用 V1（VRPTW） |
| RRNCO | 支持 | UNSUPPORTED | 支持 | 作者多任务含 VRPTW | 复用 V1（VRPTW） |
| NDS-Econ | **UNSUPPORTED** | UNSUPPORTED | **UNSUPPORTED（V2 判定）** | 目标为冷链静态净效益；标准 VRPTW 权重与运行入口只存在于归档 legacy 树 `archive/cc_compare_reorganization_20261021/legacy/CC_Compare`（只读，无标准字典序接口） | **不并入**，登记理由（见 §3） |

## 3. NDS 判定（V2 计划修订，先于 V2 正式结果登记）

`P5_预登记.md` §5 表曾记 NDS-Econ 在 VRPTW 有作者权重（可行）。V2 复核发现：

1. 活动树 `CC_Compare/configs/methods.json` 的 10 项对比方法中**不含 NDS**；
2. NDS 相关实现与实验脚本仅存在于归档 legacy 树，且其目标函数为冷链静态净效益
   （与标准 VRPTW 字典序目标不同），归档为只读、不提供标准任务接口；
3. 因此"NDS VRPTW 168 端点"无法在**目标口径可证明一致**的前提下完成。

**处理**：登记为 UNSUPPORTED，不伪造端点、不混入主表、不以替代方法顶替。
V2 端点计划相应由 3852 调整为 **3684**：
3276（内部正式）+ 240（开发面板）+ 168（PyVRP VRPTW 补齐）。差额 168 = NDS 块。

## 4. 复用条件复核（每次复用必须逐项通过）

| 条件 | 复核方式 | 结论 |
|---|---|---|
| 源代码未改 | `p5_external_worker.py` 与 `results_external_*/phase.json` 的 SHA 对照冻结清单 | 未改（P5-V1 冻结） |
| 计时协议相同 | phase.json `seconds=60`，worker 内 `MaxRuntime(56)`+提取余量 | 相同 |
| 目标口径相同 | CVRP 逐边整数舍入；VRPTW (K,D) 字典序 | 相同 |
| 初解约定相同 | 外部方法用自身原生初始化（构造式），不用本方法见证 | 相同 |
| 实例与种子相同 | 同一 `_maskco_final_paper_p5` 数据，CVRP 68001–68003、VRPTW 69001–69003 | 相同 |

不满足任一条即重跑（本轮 PyVRP VRPTW 属"覆盖不足"，不是条件失效，故补齐而非替换）。
