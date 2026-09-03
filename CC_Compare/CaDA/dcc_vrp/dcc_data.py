"""DCC-VRP → CaDA MTVRPEnv 数据桥接。

把 DynMaskCO 的 DCC .npz 实例转成 CaDA `MTVRPEnv` 的 TensorDict 输入约定，
在保持可行性不变的前提下做尺度对齐，并让 cost 落在原始坐标距离尺度上。

DCC .npz 字段（51 节点 = 1 depot + 50 客户）:
    coords        (B, 51, 2)   float32   已归一化到 [0,1]^2, depot=(0.5,0.5)
    demands       (B, 51)      int32     depot=0
    tw_start      (B, 51)      float32
    tw_end        (B, 51)      float32   depot 时间窗上界 = 24
    service_time  (B, 51)      float32
    reveal_time   (B, 51)      float32
    visible_mask  (B, 51)      int32     1=在 t=0 可见, 0=未来
    opt_costs     (B,)         float32   纯行驶距离（真值）
    routes        (B, 200)     int32     depot=0 分隔

CaDA 约定（MTVRPEnv VRPTW）:
    coords ∈ [0,1]^2, speed=1, 时间窗上界 max_time=4.6, service ∈ [0.15,0.33],
    demand 按容量归一 (demand/cap), vehicle_capacity=1.0。

尺度对齐技巧（关键）:
    CaDA 的 max_time=4.6, 而 DCC 的 depot 时间窗上界=24。为把时间特征放进
    CaDA 模型的训练分布, 同时**精确保持可行性**, 统一缩放时间轴:
        s      = 4.6 / t_max          （时间缩放因子）
        TW     ← TW   * s
        service← service * s
        speed  ← 1 / s                （travel_time = dist/speed 也缩放 s 倍）
    这样 arrival'_k = s · arrival_old_k 恒成立, TW 可行性严格不变；而 CaDA 的
    `get_reward` 用原始坐标欧氏距离, 所以 cost = 原始距离, 与 DynMaskCO 同尺度可直接比。
"""
import numpy as np
import torch
from tensordict import TensorDict

T_MAX_CADA = 4.6            # CaDA MTVRP 时间窗上界（depot horizon）
P_TAG = [1.0, 0.0, 1.0, 0.0, 0.0]   # [c, o, tw, l, b] —— 纯 VRPTW（闭路、无背载、无距离限制）


def load_dcc_tensordict(npz_path, capacity=50.0, mask_future=False, device="cpu"):
    """读取 DCC .npz → CaDA 格式 TensorDict。

    Args:
        npz_path:     DCC 数据文件路径。
        capacity:     车辆容量（用于 demand 归一化, 默认 50）。
        mask_future:  True = non-anticipatory（未来节点特征清零 + visible_mask 保留）;
                      False = clairvoyant（全部可见）。
        device:       TensorDict 设备。

    Returns:
        TensorDict, batch_size=[B], 含 CaDA MTVRPEnv 所需全部字段 + visible_mask/opt_cost。
    """
    d = np.load(npz_path, allow_pickle=True)
    coords = d["coords"].astype(np.float32)          # (B, N+1, 2)
    demands = d["demands"].astype(np.float32)        # (B, N+1)
    tw_start = d["tw_start"].astype(np.float32)
    tw_end = d["tw_end"].astype(np.float32)
    service = d["service_time"].astype(np.float32)
    opt = d["opt_costs"].astype(np.float32) if "opt_costs" in d else np.zeros(coords.shape[0], np.float32)
    B = coords.shape[0]
    num_node = coords.shape[1]                       # N+1

    # ---- 时间尺度对齐（保持可行性 + 放进 CaDA 训练分布）----
    t_max = float(tw_end.max())                      # depot horizon（DCC 数据 = 24）
    s = T_MAX_CADA / t_max
    speed = 1.0 / s
    tw_start = tw_start * s
    tw_end = tw_end * s
    service = service * s

    # ---- demand 归一（CaDA: demand/cap, vehicle_capacity=1.0）----
    demand_linehaul = demands / capacity
    vehicle_capacity = np.ones((B, 1), np.float32)
    capacity_original = np.full((B, 1), capacity, np.float32)

    # ---- 可见性（non-anticipatory vs clairvoyant）----
    if "visible_mask" in d:
        visible = d["visible_mask"].astype(np.float32)
    else:
        visible = np.ones_like(demands, np.float32)
    visible[:, 0] = 1.0                              # depot 恒可见
    if mask_future:
        future = (visible == 0)
        # DynMaskCO P0-2 等价：未来节点特征清零（坐标→0.5 中性, 其余→0）
        coords = coords.copy(); coords[future] = 0.5
        tw_start = tw_start.copy(); tw_start[future] = 0.0
        tw_end = tw_end.copy(); tw_end[future] = 0.0
        service = service.copy(); service[future] = 0.0
        demand_linehaul = demand_linehaul.copy(); demand_linehaul[future] = 0.0
    else:
        visible = np.ones_like(demands, np.float32)  # clairvoyant: 全部可见

    # ---- p_s_tag: [c, o, tw, l, b, size/2000] ----
    size_tag = (num_node - 1) / 2000.0               # 50/2000 = 0.025
    p_s_tag = np.tile(np.array(P_TAG + [size_tag], np.float32), (B, 1))

    td = TensorDict(
        {
            "locs": torch.from_numpy(coords),
            "demand_linehaul": torch.from_numpy(demand_linehaul),
            "demand_backhaul": torch.zeros((B, num_node), dtype=torch.float32),
            "distance_limit": torch.full((B, 1), float("inf"), dtype=torch.float32),
            "time_windows": torch.from_numpy(np.stack([tw_start, tw_end], axis=-1)),
            "service_time": torch.from_numpy(service),
            "vehicle_capacity": torch.from_numpy(vehicle_capacity),
            "capacity_original": torch.from_numpy(capacity_original),
            "open_route": torch.zeros((B, 1), dtype=torch.bool),
            "speed": torch.full((B, 1), speed, dtype=torch.float32),
            "p_s_tag": torch.from_numpy(p_s_tag),
            "visible_mask": torch.from_numpy(visible).bool(),
            "opt_cost": torch.from_numpy(opt),
        },
        batch_size=[B],
        device=device,
    )
    return td
