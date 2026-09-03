# 实验 Exp-03: CVRPTW 训练数据生成

- **日期**：2026-07-29 ✅ 完成
- **目标**：生成 CVRPTW 训练数据（Solomon 格式），验证 5D 特征 ([x, y, demand, tw_start, tw_end]) 的模型输入可行性
- **假设**：生成的数据格式能正确被 CVRPTWDataloader 加载，5D 特征能通过 CVRPTWModel 前向传播

## 实验设计

### 数据生成参数

| 参数 | 值 | 说明 |
|---|---|---|
| problem_size | 50 | 50 个 customer + 1 depot |
| num_instances | 1280 (train) / 128 (test) | test 已在 Exp-02 生成 |
| types | R1, C1, RC1 | 3 种 Solomon 类型 |
| capacity | 50 | 与 CVRP-100 一致 |

### 数据格式

每类数据包含：

| 字段 | 形状 | dtype | 说明 |
|---|---|---|---|
| coords | (N, 51, 2) | float32 | depot (idx 0) + 50 customers, [0,1]² 坐标 |
| demands | (N, 51) | int32 | depot=0, customer ∈ [1, 10] |
| tw_start | (N, 51) | float32 | 时间窗起始, [0, 24.0] (type 1 窄TW) |
| tw_end | (N, 51) | float32 | 时间窗结束, [0, 24.0] |
| service_time | (N, 51) | float32 | 基于 demand 估算, depot=0 |
| routes | (N, 200) | int32 | 贪心构造参考解, pad 到 200 |
| opt_costs | (N,) | float32 | 贪心解成本（非最优解） |

## 实际执行命令

```bash
cd /home/hzeng/project/MASKCO-Main/
source /home/hzeng/envs/MASKCO_env/bin/activate

# R1 (Random, 窄时间窗)
python -u "C-VRP_Cold-chainVehicleRoutingProblem/data/generate_cvrptw_data.py" \
    --problem_size 50 --num_instances 1280 --type R1 --capacity 50 \
    --output "C-VRP_Cold-chainVehicleRoutingProblem/data/cvrptw50_r1_train.npz"

# C1 (Clustered, 窄时间窗)
python -u "C-VRP_Cold-chainVehicleRoutingProblem/data/generate_cvrptw_data.py" \
    --problem_size 50 --num_instances 1280 --type C1 --capacity 50 \
    --output "C-VRP_Cold-chainVehicleRoutingProblem/data/cvrptw50_c1_train.npz"

# RC1 (Mixed, 窄时间窗)
python -u "C-VRP_Cold-chainVehicleRoutingProblem/data/generate_cvrptw_data.py" \
    --problem_size 50 --num_instances 1280 --type RC1 --capacity 50 \
    --output "C-VRP_Cold-chainVehicleRoutingProblem/data/cvrptw50_rc1_train.npz"
```

## 结果

| 检查项 | 结果 |
|---|---|
| 数据生成 — R1 train | ✅ (1280, 51, *) — ~4 MB |
| 数据生成 — C1 train | ✅ (1280, 51, *) — ~4 MB |
| 数据生成 — RC1 train | ✅ (1280, 51, *) — ~4 MB |
| 数据生成 — R1 test | ✅ (128, 51, *) — Exp-02 期间生成 |
| Dataloader 正常加载 | ✅ Features (4, 51, 5), TW range [0, 1] |
| 5D 特征前向传播通过 | ✅ Encoder (1, 51, 256), Logits (1, 51, 51) |
| Logits 无 NaN/Inf | ✅ |

### 已知局限

1. **参考解质量低**：`routes` 由贪心构造生成（最近邻 + TW 可行性检查），非最优解。训练时作为 mask-and-reconstruct 的 target，质量上限受限于贪心解。后续可用 LKH3 生成更优参考解。
2. **route 长度浪费**：pad 到 200，实际 CVRPTW-50 仅需 ~50-70 个位置（含 depot 分隔符）。可后续优化为动态 padding。
3. **TSPTW 数据未转换**：服务器上 TSPTW `.npz` 文件缺失（`服务器说明.md` 中列出的文件实际不存在），改用直接生成的 Solomon 数据。如需 TSPTW 转换，需从本地上传 TSPTW 文件。

## 结论

- **假设成立** ✅：数据生成、加载、模型前向传播全部通过
- 5D 特征正常进入 encoder，TW 归一化到 [0,1] 区间，模型无 NaN/Inf
- 三组 Solomon 类型训练数据就绪，可直接用于 Exp-04 消融训练

## 下一步

→ **Exp-04**：5D 输入特征消融实验（3D CVRP vs 5D CVRPTW 训练 loss 收敛对比）
