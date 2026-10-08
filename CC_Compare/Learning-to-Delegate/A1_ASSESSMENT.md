# Learning-to-Delegate（L2D）A-v1 适配评估（2026-09-29，P4 评估项）

## 1. 源码与权重资产（本地已核）

- 论文：Learning to Delegate for Large-scale Vehicle Routing（NeurIPS 2021 Spotlight，
  Li, Yan & Wu；arXiv:2107.04139）。
- 代码：本地 `CC_Compare/Learning-to-Delegate/`（官方 clone：generate_*.py、
  preprocess*.py、supervised.py、run_{lkh,hgs}.py 等；`generations/`（10GB 数据/权重
  zip）与 `lkh3/`、`hgs/` 子求解器二进制**未下载**）。
- 权重：本地核到 L2D 官方 3 个回归模型权重（由 Omni-VRP 仓库自带的 L2D 基线副本）：
  `CC_Compare/Omni-VRP/L2D/exps/{regression_model_512,regression_model_2048,
  regression_maml_512}/models/40000.pth`。
- 子求解器：官方流程必须调用 LKH-3 / HGS 二进制（本地/服务器均无，需另行下载编译）。

## 2. 机制与 A-v1 的语义映射

| 维度 | L2D（官方口径） | A-v1 | 桥接评估 |
|---|---|---|---|
| 问题规模 | n ≥ 500–3000（大尺度） | 每决策可见池 ≤ ~90，全天 200 单 | **规模方向相反**：L2D 的核心收益在大尺度分区 |
| 机制 | 每步从全图选 k 个"子问题"（回归模型打分）→ 交给 LKH/HGS 子求解 → 拼回 | 揭示即接受/拒绝 + 在线路由 | 其"子问题选择"可映射为"可见池排序"，但无官方小尺度设定 |
| 决策接口 | 静态实例上的迭代构造轨迹 | 每揭示事件一次决策、10s 预算、不可撤销 | 无 reveal→commit 语义 |
| 约束 | CVRP/CVRPTW/VRPMPD 静态 | CVRPTW + C0 预算 + 承诺 | TW 变体有官方训练档，但规模与决策节奏不匹配 |
| 依赖 | LKH-3 / HGS 二进制 | 无外部子求解器 | 需补依赖 |

**结论**：L2D 的"委托子问题"机制在 n≈90 的 A-v1 池上无官方适用设定（官方模型训练于
n≥500 分布）；直接搬用等于把其回归模型当作通用排序打分器，这不是官方机制、也非官方
规模，可比性弱（与 §5.1 旧判断一致：硬适配 = 发明新决策策略）。**维持"评估完成、实施
排后"**：不进入当前 P2/P3 自动适配批次，除非主表需要"大尺度委托学习"锚点，届时需
预声明（补子求解器 + 小尺度再训练 + 验收标准）。

## 3. 记录与建议

- 状态：**可评估、暂不适配**（子求解器缺失 + 规模方向相反 + 无 reveal 决策层）；
  与 MAPT、CO-enriched-ML 同属"评估完成、实施排后"记录。
- 论文引用时按"大尺度委托学习"文献锚点处理，不冒充同口径对比。
