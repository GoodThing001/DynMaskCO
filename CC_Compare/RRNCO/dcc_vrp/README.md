# RRNCO → DCC-VRP 适配

RRNCO（[ai4co/real-routing-nco](https://github.com/ai4co/real-routing-nco)，ICLR 2026）的 `RMTVRPEnv`（`rcvrptw`）支持带时间窗的多车辆 CVRPTW，可适配到 DCC-VRP 协议。

## 文件

| 文件 | 作用 |
|------|------|
| `dcc_data.py` | DCC .npz → RRNCO `RMTVRPEnv` 输入（时间尺度对齐 + demand 归一 + 距离矩阵 + 可见性） |
| `dcc_env.py` | `DCCRMTVRPEnv`：`normalize=False` + 可见性门控；`VisibleStartNodes` 只从可见客户出发 |
| `run_dcc.py` | 评估 harness：镜像 `test.py`（关增强、batch=1、num_starts=可见客户数）→ 报告 cost / feas / gap |

## 关键决策（诚实标注）

1. **`normalize=False`（成本可比）**：RRNCO 默认 `normalize=True` 会把距离矩阵做 per-instance min-max 归一，且 `_get_reward` 的「去归一」是标量近似（有偏），导致 (a) TW 与 duration 尺度不一致、(b) 报告的 cost 不再是纯欧氏距离。因此用 `normalize=False` + 原始欧氏距离，`cost` 与 DynMaskCO 同尺度可直接比。
2. **速度技巧**：时间轴缩放 `s=4.6/24`、`speed=1/s`，让 TW/service/duration 落进 RRNCO 训练分布（`max_time=4.6`），且 `arrival' = s·arrival` 严格保持可行性。
3. **zero-shot 局限**：距离矩阵原始 ∈ `[0, √2]`，与 RRNCO 训练时的 min-max 归一 `[0,1]` 有轻微 OOD（影响 `DistanceExpert`）。属诚实报告的 zero-shot 迁移局限，不改结果语义。
4. **non-anticipatory**（`--mask_future`）：未来节点特征对模型屏蔽（坐标→0.5、距离/时长→中性、TW/service/demand→0），但 env 用 `_true_*` 真值做可行性 + 奖励，**serve 全部 50 客户**（不早停）。等价 DynMaskCO 的「causal encoder + full-service decoder」。只从可见客户出发，`batch=1`。

## 依赖

- `rl4co>=0.5.1`（仓库 lock 为 0.6.0）、`torch>=2.7`（Blackwell）。
- `rrnco/utils.py` 的 `patch_torchrl_specs()` 已在 `run_dcc.py` 里调用，兼容新旧 torchrl API。

## 运行（服务器, rrnco conda env）

```bash
cd /home/hzeng/project/MASKCO-Main/CC_Compare/RRNCO

# 1) 下载 rcvrptw checkpoint（HuggingFace）
python scripts/download_hf.py --no-data
ls checkpoints/rcvrptw/epoch_199.ckpt

# 2) non-anticipatory 评估
CUDA_VISIBLE_DEVICES=0 python dcc_vrp/run_dcc.py \
    --ckpt checkpoints/rcvrptw/epoch_199.ckpt \
    --data ../../C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz \
    --mask_future --capacity 50

# 3) clairvoyant（对照）
CUDA_VISIBLE_DEVICES=0 python dcc_vrp/run_dcc.py \
    --ckpt checkpoints/rcvrptw/epoch_199.ckpt \
    --data ../../C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz \
    --capacity 50
```

输出：`cost`（纯距离）、`TW feas`、`Cap feas`、`gap vs opt` 的均值。
