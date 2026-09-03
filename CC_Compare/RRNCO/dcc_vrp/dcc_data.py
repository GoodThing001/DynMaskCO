"""DCC-VRP → RRNCO RMTVRPEnv 数据桥接。

把 DynMaskCO 的 DCC .npz 实例转成 RRNCO `RMTVRPEnv`（rcvrptw）的 TensorDict 输入，
在保持可行性不变的前提下做时间尺度对齐，并让 cost 落在原始坐标距离尺度上。

DCC .npz 字段（51 节点 = 1 depot + 50 客户）:
    coords        (B, 51, 2)   float32   已归一化到 [0,1]^2, depot=(0.5,0.5) 在 index 0
    demands       (B, 51)      int32     depot=0
    tw_start/end  (B, 51)      float32   时间窗, depot 上界=24
    service_time  (B, 51)      float32
    reveal_time   (B, 51)      float32
    visible_mask  (B, 51)      int32     1=在 t=0 可见, 0=未来
    opt_costs     (B,)         float32   纯行驶距离（真值）

RRNCO 约定（RMTVRPEnv VRPTW, 训练分布）:
    locs ∈ [0,1]^2, speed, max_time=4.6, service ∈ [0.15,0.33],
    demand 按容量归一 (demand/cap, vehicle_capacity=1.0), 距离矩阵 min-max 归一 [0,1]。

关键决策 —— **normalize=False + 原始坐标单位**（诚实 zero-shot）:
    若用 RRNCO 默认 `normalize=True`，其 `_reset` 会把距离矩阵做 per-instance min-max
    归一，且 `_get_reward` 的「去归一化」是标量近似（有偏），导致 (a) TW 与 duration
    尺度不一致、(b) 报告的 cost 不再是纯欧氏距离，无法与 DynMaskCO 对齐。
    因此本桥接用 `normalize=False`，提供**原始欧氏距离**（与 DynMaskCO cost 同尺度、
    可直接比），并单独用「速度技巧」把时间轴缩进模型训练分布：

        s      = 4.6 / t_max            （时间缩放因子, t_max=24）
        TW     ← TW * s
        service← service * s
        speed  ← 1 / s                  （travel_time = dist/speed 也缩放 s 倍）

    这样 TW 可行性严格不变（arrival'_k = s · arrival_old_k），而 cost 仍是原始欧氏距离。
    代价：距离矩阵原始 ∈ [0, √2]，与训练时的 [0,1] 分布有轻微 OOD（DistanceExpert），
    属 zero-shot 迁移的诚实局限，已在 README 标注。
"""
import numpy as np
import torch
from tensordict import TensorDict

T_MAX = 4.6           # RRNCO RMTVRP 时间窗上界（depot horizon）


def _pairwise(c):
    """(M,2) → (M,M) 欧氏距离矩阵。"""
    return np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)


def load_dcc_tensordict(npz_path, capacity=50.0, mask_future=False, device="cpu"):
    """读取 DCC .npz → RRNCO 格式 TensorDict（normalize=False 原始单位）。

    non-anticipatory 语义（mask_future=True）与 DynMaskCO P0-2/P0-3a 对齐：
      - **encoder 看 masked**：未来节点 locs→0.5（depot 中心）、距离/时长→由 masked 坐标
        重算（未来≈depot）、TW/service/demand→0。模型看不到未来节点位置/需求。
      - **env 用真值**：把真正的 coords/TW/service/demand/distance 存进 `_true_*` 字段，
        DCCRMTVRPEnv 用它们做 TW/容量可行性 + 距离奖励，从而 **serve 全部 50 客户**。
      这样「encoder 被门控、decoder 仍服务全部节点」，等价 DynMaskCO 的 causal decode。

    Args:
        npz_path:     DCC 数据文件路径。
        capacity:     车辆容量（用于 demand 归一, 默认 50）。
        mask_future:  True = non-anticipatory（未来节点特征屏蔽，但仍服务全部）;
                      False = clairvoyant（全部可见，无屏蔽）。
        device:       TensorDict 设备。

    Returns:
        TensorDict, batch_size=[B]，字段与 RMTVRPEnv._reset 的输入约定一致，
        另含 visible_mask（bool, B,N+1）、opt_cost、以及 `_true_*` 真值字段。
    """
    d = np.load(npz_path, allow_pickle=True)
    coords = d["coords"].astype(np.float32)            # (B, N+1, 2), depot index 0
    demands = d["demands"].astype(np.float32)          # (B, N+1)
    tw_start = d["tw_start"].astype(np.float32)
    tw_end = d["tw_end"].astype(np.float32)
    service = d["service_time"].astype(np.float32)
    opt = d["opt_costs"].astype(np.float32) if "opt_costs" in d else np.zeros(coords.shape[0], np.float32)
    B, num_node = coords.shape[0], coords.shape[1]     # N+1

    # ---- 时间尺度对齐（保持可行性 + 放进 RRNCO 训练分布）----
    t_max = float(tw_end.max())                        # depot horizon（DCC = 24）
    s = T_MAX / t_max
    speed_val = 1.0 / s
    tw_start = tw_start * s
    tw_end = tw_end * s
    service = service * s

    # ---- demand 归一（RRNCO scale_demand: demand/cap, cap=1.0）----
    # 注意：RMTVRPEnv._reset 会为 depot prepend 一个 0，故这里传「仅客户」的 demand。
    demand_linehaul = (demands[:, 1:] / capacity).astype(np.float32)   # (B, N)

    # ---- 可见性（non-anticipatory vs clairvoyant）----
    if "visible_mask" in d:
        visible = d["visible_mask"].astype(np.float32)
    else:
        visible = np.ones((B, num_node), np.float32)
    visible[:, 0] = 1.0                               # depot 恒可见

    # ---- 真值距离/时长矩阵（env 用：可行性 + 奖励）----
    dist_true = np.stack([_pairwise(coords[b]) for b in range(B)]).astype(np.float32)
    dur_true = dist_true / speed_val

    # ---- encoder 看到的字段（默认 = 真值 = clairvoyant）----
    coords_enc, dist_enc, dur_enc = coords, dist_true, dur_true
    tw_start_enc, tw_end_enc, service_enc, demand_enc = tw_start, tw_end, service, demand_linehaul

    if mask_future:
        future = (visible == 0)                       # (B, N+1)
        # 未来节点坐标 → depot 中心（0.5, 0.5），与 DynMaskCO P0-2「coords→0.5」一致
        coords_enc = coords.copy()
        coords_enc[future] = 0.5
        # encoder 的距离/时长由 masked 坐标重算（未来 ≈ depot）
        dist_enc = np.stack([_pairwise(coords_enc[b]) for b in range(B)]).astype(np.float32)
        dur_enc = dist_enc / speed_val
        # 未来 TW/service/demand → 0（中性，encoder 看不到）
        tw_start_enc = tw_start.copy(); tw_start_enc[future] = 0.0
        tw_end_enc = tw_end.copy(); tw_end_enc[future] = 0.0
        service_enc = service.copy(); service_enc[future] = 0.0
        demand_enc = demand_linehaul.copy()
        demand_enc[future[:, 1:]] = 0.0

    td = TensorDict(
        {
            # ---- encoder 看到的字段（mask_future 时为屏蔽版）----
            "locs": torch.from_numpy(coords_enc),
            "demand_linehaul": torch.from_numpy(demand_enc),        # (B, N) 客户 only
            "demand_backhaul": torch.zeros((B, num_node - 1), dtype=torch.float32),
            "backhaul_class": torch.full((B, 1), 1, dtype=torch.int32),
            "distance_limit": torch.full((B, 1), float("inf"), dtype=torch.float32),
            "time_windows": torch.from_numpy(np.stack([tw_start_enc, tw_end_enc], axis=-1)),
            "service_time": torch.from_numpy(service_enc),
            "vehicle_capacity": torch.ones((B, 1), dtype=torch.float32),
            "capacity_original": torch.full((B, 1), capacity, dtype=torch.float32),
            "open_route": torch.zeros((B, 1), dtype=torch.bool),
            "speed": torch.full((B, 1), speed_val, dtype=torch.float32),
            "distance_matrix": torch.from_numpy(dist_enc),
            "duration_matrix": torch.from_numpy(dur_enc),
            "visible_mask": torch.from_numpy(visible).bool(),
            "opt_cost": torch.from_numpy(opt),
            # ---- 真值字段（DCCRMTVRPEnv 用：可行性 + 距离奖励，保证 serve 全部 50）----
            "_true_distance_matrix": torch.from_numpy(dist_true),
            "_true_duration_matrix": torch.from_numpy(dur_true),
            "_true_time_windows": torch.from_numpy(np.stack([tw_start, tw_end], axis=-1)),
            "_true_service_time": torch.from_numpy(service),
            "_true_demand_linehaul": torch.from_numpy(demand_linehaul),   # (B, N) 客户 only
        },
        batch_size=[B],
        device=device,
    )
    return td
