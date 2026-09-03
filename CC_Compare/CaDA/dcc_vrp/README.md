# CaDA → DCC-VRP 适配

CaDA（[CIAM-Group/CaDA](https://github.com/CIAM-Group/CaDA)，ICML 2025）的 MTVRP 框架支持 16 种 VRP 变体，**VRPTW 是其中之一**，因此可适配到 DCC-VRP（多车辆 CVRPTW）协议。

## 文件

| 文件 | 作用 |
|------|------|
| `dcc_data.py` | DCC .npz → CaDA `MTVRPEnv` TensorDict 输入（时间尺度对齐 + demand 归一 + 可见性） |
| `dcc_env.py` | `DCCEnv`：继承 `MTVRPEnv`，叠加 non-anticipatory 可见性门控（未来节点预标记已访问 + 只从可见客户出发） |
| `run_dcc.py` | 评估 harness：装载模型 + checkpoint → 贪心 best-of-N-starts 解码 → 报告 cost / TW feas / Cap feas / gap |

## 关键设计（诚实标注）

1. **成本可直接比**：CaDA 与 DynMaskCO 的坐标都在 `[0,1]²`，`get_reward` 用原始坐标欧氏距离，因此 `cost` 是纯行驶距离，与 DynMaskCO 同尺度。
2. **速度技巧**：CaDA 时间窗上界 `max_time=4.6`，DCC 为 24。为把时间特征放进 CaDA 训练分布且**严格保持可行性**，统一缩放时间轴 `s=4.6/24`、`speed=1/s`（`arrival' = s·arrival`，TW 可行性不变）。
3. **non-anticipatory**：`--mask_future` 下未来节点特征清零（坐标→0.5 中性，tw/service/demand→0）+ 解码器永不选未来节点 + 只从可见客户出发；处理 `batch=1`（实例可见性异构）。
4. **可行性复核**：CaDA 解码器的 action mask 逐步保证 TW+容量，本 harness 另用原始数据独立重放复核。

## 依赖风险（⚠️ 需在服务器处理）

CaDA 的 `requirements.txt` pin 了 `torch 2.0.1 + torchrl 0.1.1`（旧 API `CompositeSpec`/`BoundedTensorSpec`），与 RTX 5090（Blackwell, sm_120）要求的 **torch ≥ 2.7** 冲突。适配时需：
- 升级 torch ≥ 2.7、torchrl/tensordict 到新版；
- 旧 API 名（`CompositeSpec`→`Composite`、`BoundedTensorSpec`→`Bounded` 等）需按 `RRNCO/rrnco/utils.py` 的 `patch_torchrl_specs` 思路打补丁，或直接改 `CaDA/` 内引用。

## 运行（服务器, cada conda env）

```bash
cd /home/hzeng/project/MASKCO-Main/CC_Compare/CaDA

# 1) 确认 checkpoint 存在（CaDA 官方 VRPTW-50 权重）
ls 50/result/*/checkpoint-300.pt

# 2) non-anticipatory 评估
CUDA_VISIBLE_DEVICES=0 python dcc_vrp/run_dcc.py \
    --ckpt 50/result/2024-1111-1139/checkpoint-300.pt \
    --data ../../C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz \
    --mask_future --capacity 50

# 3) clairvoyant（对照）
CUDA_VISIBLE_DEVICES=0 python dcc_vrp/run_dcc.py \
    --ckpt 50/result/2024-1111-1139/checkpoint-300.pt \
    --data ../../C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz \
    --capacity 50
```

输出：`cost`（纯距离）、`TW feas`、`Cap feas`、`gap vs opt` 的均值。
