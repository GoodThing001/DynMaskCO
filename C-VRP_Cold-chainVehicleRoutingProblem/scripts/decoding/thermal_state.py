"""
Phase 3d: Endogenous Cold-Chain Physics — 内生的温度-品质-能耗状态

将预计算的 quality_loss / energy_mat 升级为 beam 内实时物理递推。

每段行驶 (i→j) 加入三个递推:
  1. 车厢温度:  Newton 冷却 + 制冷 + 开门冲击
  2. 货物品质:  累计 Arrhenius 积分 (每温区独立)
  3. 制冷能耗:  制冷功率 × 运行时间 + 开门恢复能耗

参数来源: 冷链文献 (Transportation Science, 2022/2026)

定义 (dimensionless, normalized to horizon=24h):
  T_ambient = 25.0  环境温度 (°C)
  T_target = [25, 4, -18]  温区目标温度 (°C)
  UA_over_C = 0.5  热传导系数 / 热容 (1/h) → 车厢自然升温速率
  P_cool = [0, 150, 400]  制冷功率 (W), 0=常温/150=冷藏/400=冷冻
  Q_door = 500  开门热冲击 (J per open)
  C_eff_base = 5000  基础热容 (J/K)
  C_eff_per_load = 50  货物热容贡献 (J/K per unit load)

Arrhenius 品质参数:
  k_0 = [0.01, 0.002, 0.0002]  常温/冷藏/冷冻 衰减速率 (1/h)
  Ea_over_R = 5000  活化能/气体常数 (K)

能耗:
  eta = 2.5  制冷效率 (COP, 每瓦电移走的热量)
"""
from dataclasses import dataclass
import numpy as np


# ============================================================
# Physical Constants
# ============================================================

T_AMBIENT = 25.0          # °C, 室外环境温度（GB 31605-2020 冷链环境）
T_TARGET = np.array([25.0, 4.0, -18.0], dtype=np.float32)  # 常温/冷藏/冷冻（GB 31605-2020: 冷藏 0-4°C, 冷冻 ≤-18°C）

# === 参数出处（2026-08-16，基于 docs/论文参考/v1 的 29 篇论文校准）===
# 制冷功率 P_COOL：真实制冷机 ~6-10 kW（电制冷拖车热/功耗模型, Int. J. Refrigeration）
#   - 连续冷却 ≈6 kW，TRU power 从 10 kW 降至 7 kW（随车厢温度下降）
#   - 此处为「有效值」(W)，因 1 unit ≈ 0.25h 的时间缩放，论文需说明缩放关系
# COP：非常数，随内外温差变化（Piecewise Linear Model, 电制冷拖车论文）
#   - Q_TRU = P_TRU × COP，COP 随温差增大而降低
# 品质衰减 K_0：一阶反应（first-order reaction）确认（ML 温度管理, Scientific Reports 2024）
#   - Q = 1 - exp(-k(T)t)，k(T) = k_0 exp(-E_a/RT)
# 热模型：集总热容（Lumped capacitance, 电制冷拖车论文）→ Newton 冷却方向正确
# Effective parameters rescaled for 50-node unit-square domain:
# Total route ~15-20 units → ~4-5 real hours → 1 unit ≈ 15 min = 0.25h
UA_OVER_C = 2.0            # 1/h, effective heat exchange (集总热容模型)
P_COOL = np.array([0.0, 800.0, 2000.0], dtype=np.float32)  # W, 有效制冷功率（真实 ~6-10 kW）
Q_DOOR = 2000.0            # J, effective door heat influx（开门热空气动力学, ATE 2022）
C_EFF_BASE = 3000.0        # J/K, effective base heat capacity（集总热容）
C_EFF_PER_LOAD = 30.0      # J/K per unit, cargo heat capacity contribution
K_0 = np.array([0.01, 0.002, 0.0002], dtype=np.float32)  # 1/h, Arrhenius 速率（一阶反应）
EA_OVER_R = 5000.0          # K, 活化能/气体常数

# Time scaling: beam's travel_t = normalized_dist / speed
# Real hours = travel_t × HOURS_PER_UNIT
# 1 normalized unit ≈ 10 km real → at 40 km/h ≈ 0.25 h
HOURS_PER_UNIT = 1.0  # rescaled: 1 unit = 1 "effective hour" for observable thermal dynamics


def compute_cop(delta_T: float) -> float:
    """COP 随温差变化（非常数）。

    依据：电制冷拖车热/功耗模型（Int. J. Refrigeration）——COP 是内外温差的
    Piecewise Linear Model 函数，温差越大 COP 越低（制冷越困难）。

    简单线性近似：COP ≈ 2.5 − 0.03 × ΔT（下限 1.0）。
    delta_T = T_ambient − T_cabin（温差，°C）。
    """
    return max(1.0, 2.5 - 0.03 * delta_T)


@dataclass
class ThermalState:
    """单个车辆在单个 beam 中的热状态。"""
    cabin_temp: float = 25.0       # 车厢当前温度 (°C)
    cumulative_energy: float = 0.0  # 累计制冷能耗 (kWh 等效)
    quality_ambient: float = 1.0    # 常温货物完好率 (0-1)
    quality_refrig: float = 1.0     # 冷藏货物完好率
    quality_frozen: float = 1.0     # 冷冻货物完好率
    quality_loss_total: float = 0.0 # 累计品质损失
    num_door_opens: int = 0
    num_thermal_violations: int = 0  # 温度越界次数 (> T_target + 5°C)


def compute_segment_thermal(
    state: ThermalState,
    travel_time: float,           # 行驶时间 (小时)
    service_time: float,          # 服务/开门时间 (小时)
    cargo_temp_class: int,        # 当前服务的货物温区 (0/1/2)
    vehicle_load: float,          # 当前载重 (单位)
    vehicle_temp_zone: int = 2,   # 车辆温区能力 (0/1/2, 2=多温区)
    ambient_temp: float = T_AMBIENT,
):
    """
    计算一段路线 (i→j + service at j) 后的热状态变化。

    Phase 1: 行驶阶段 (车厢封闭, 制冷运行)
      - T 受 Newton 冷却 + 制冷对抗
    Phase 2: 服务阶段 (开门卸货)
      - T 受开门冲击 → 快速向环境温度移动
    Phase 3: 品质更新
      - 每温区的 Arrhenius 衰减

    Returns:
        new_state: ThermalState (copy, 不修改原)
        segment_energy: float (kWh 等效)
        segment_quality_loss: float (品质损失总量)
    """
    T = state.cabin_temp

    # Time scaling: convert normalized units → effective hours
    dt_travel = travel_time * HOURS_PER_UNIT
    dt_service = service_time * HOURS_PER_UNIT

    # --- Phase 1: 行驶 ---
    t_target = T_TARGET[vehicle_temp_zone]

    # 热容随载重增加 (更多货物 → 更难升/降温)
    c_eff = C_EFF_BASE + C_EFF_PER_LOAD * vehicle_load

    # 自然升温: T → T_ambient
    delta_heat = (ambient_temp - T) * (1.0 - np.exp(-UA_OVER_C * travel_time))

    # 主动制冷: T → T_target
    if vehicle_temp_zone > 0 and T > t_target:
        # 制冷功率转化为温降: ΔT = (COP(ΔT) × P_cool × Δt) / C_eff
        # COP 随温差变化（非常数，电制冷拖车论文）
        cop = compute_cop(ambient_temp - T)
        cooling_capacity = cop * P_COOL[vehicle_temp_zone] * travel_time / c_eff
        delta_cool = -min(cooling_capacity, T - t_target + delta_heat)
    else:
        delta_cool = 0.0

    T += delta_heat + delta_cool

    # 制冷能耗 = P_cool × travel_time (仅当制冷运行时)
    if vehicle_temp_zone > 0 and state.cabin_temp > t_target:
        segment_energy = P_COOL[vehicle_temp_zone] * travel_time / 1000.0  # W·h → kWh (简化)
    else:
        segment_energy = 0.0

    # --- Phase 2: 服务 (开门) ---
    if service_time > 0 and cargo_temp_class >= 0:
        # 开门冲击: T 向环境温度快速移动
        door_impact = (ambient_temp - T) * (1.0 - np.exp(-3.0 * service_time))  # 3× faster
        T += door_impact

        # 开门后制冷恢复的额外能耗
        if vehicle_temp_zone > 0 and T > t_target:
            recovery_energy = Q_DOOR / 1000.0  # J → kWh
            segment_energy += recovery_energy

        # 温度越界检查: 服务时车厢温度超过目标温度 5°C
        if T > t_target + 5.0:
            state.num_thermal_violations += 1

    state.cabin_temp = T
    state.cumulative_energy += segment_energy
    state.num_door_opens += 1

    # --- Phase 3: 品质衰减 (累计) ---
    # 多温区（multi-compartment）：每温区在各自目标温度下独立衰减，k = K_0[cls]
    # （常温 0.01 / 冷藏 0.002 / 冷冻 0.0002，1/h）。
    # 旧版用共享车厢温度 + 绝对 Arrhenius（exp(-Ea/R·T)≈5e-8），会让常温货物
    # 在冷冻车厢里被过度冷藏、几乎不衰减（Num 恒为 0）。改为按温区独立速率。
    time_exposed = travel_time + service_time

    for cls in range(3):
        decay = 1.0 - np.exp(-K_0[cls] * time_exposed)
        if cls == 0:
            state.quality_ambient *= (1.0 - decay)
        elif cls == 1:
            state.quality_refrig *= (1.0 - decay)
        else:
            state.quality_frozen *= (1.0 - decay)

    # Total quality loss (weighted sum)
    ql_segment = (
        (1.0 - state.quality_ambient) * 0.3 +
        (1.0 - state.quality_refrig) * 0.4 +
        (1.0 - state.quality_frozen) * 0.3
    )
    prev_ql = state.quality_loss_total
    state.quality_loss_total = ql_segment

    return segment_energy, ql_segment - prev_ql


def compute_route_thermal_metrics(
    route: list,                  # 节点序列 (含 depot 0)
    coords: np.ndarray,           # (N+1, 2)
    tw_start: np.ndarray,         # (N+1,)
    tw_end: np.ndarray,           # (N+1,)
    service_time: np.ndarray,     # (N+1,)
    demands: np.ndarray,          # (N+1,)
    temp_class: np.ndarray,       # (N+1,) int
    capacity: float = 50.0,
    speed: float = 1.0,
    ambient_temp: float = T_AMBIENT,
    quality_salable_threshold: float = 0.1,  # 可售品质损耗阈值（Num 指标）
):
    """
    计算整条路线的冷链指标。

    Returns dict with:
      total_distance, total_energy_kwh, quality_loss, tti,
      thermal_violations, num_door_opens, avg_cabin_temp, num_unsalable
    """
    state = ThermalState()
    prev = 0
    total_dist = 0.0
    total_energy = 0.0
    total_quality = 0.0
    num_unsalable = 0
    current_load = 0.0
    temp_log = []  # [(node, arrival_temp, departure_temp)]

    for node in route:
        if node == 0:
            # Return to depot: close route
            if prev != 0:
                d = np.sqrt(((coords[prev] - coords[node]) ** 2).sum() + 1e-10)
                travel_t = d / speed
                en, ql = compute_segment_thermal(
                    state, travel_t, 0.0, 0, current_load, 0, ambient_temp)
                total_dist += d
                total_energy += en
                total_quality += ql
            prev = 0
            current_load = 0.0
            state.cabin_temp = T_AMBIENT  # reset for new route
            # 新车从 depot 出发：货物品质重置为新鲜（修复跨车辆品质未重置）
            state.quality_ambient = 1.0
            state.quality_refrig = 1.0
            state.quality_frozen = 1.0
            continue

        if node <= 0 or node >= len(coords):
            continue

        d = np.sqrt(((coords[prev] - coords[node]) ** 2).sum() + 1e-10)
        travel_t = d / speed
        svc_t = service_time[node]
        tc = int(temp_class[node]) if node < len(temp_class) else 0
        current_load += demands[node] if node < len(demands) else 0

        en, ql = compute_segment_thermal(
            state, travel_t, svc_t, tc, current_load, 2, ambient_temp)

        total_dist += d
        total_energy += en
        total_quality += ql
        temp_log.append((node, state.cabin_temp))
        prev = node

        # 不可售检查：该节点货物剩余品质 < 可售阈值
        if tc == 0 and state.quality_ambient < (1.0 - quality_salable_threshold):
            num_unsalable += 1
        elif tc == 1 and state.quality_refrig < (1.0 - quality_salable_threshold):
            num_unsalable += 1
        elif tc == 2 and state.quality_frozen < (1.0 - quality_salable_threshold):
            num_unsalable += 1

    # TTI: sum of |T_cabin - T_target| over service points
    tti = sum(abs(t[1] - T_TARGET[min(int(temp_class[t[0]]) if t[0] < len(temp_class) else 0, 2)])
              for t in temp_log)

    return {
        'total_distance': total_dist,
        'total_energy_kwh': total_energy,
        'quality_loss': total_quality,
        'tti': tti,
        'thermal_violations': state.num_thermal_violations,
        'num_door_opens': state.num_door_opens,
        'num_unsalable': num_unsalable,
        'avg_cabin_temp': np.mean([t[1] for t in temp_log]) if temp_log else T_AMBIENT,
        'temp_log': temp_log,
    }
