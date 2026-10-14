"""
历史 Phase H 滚动时域仿真器（非 C0 权威协议）。

本文件保留用于复现旧距离/TTI/edge-energy 消融。其时间步进、品质和
能耗均为代理量，不具备 pickup-to-depot cargo manifest，输出不得写入
当前论文结果。新实验统一使用 strict_online_env.py + coldchain_state.py +
coldchain_evaluator.py。

模拟场景:
  T=0: 已知部分订单 → MaskCO 初始求解 → 车辆出发
  T>0: 新订单不断到达 → 冻结在途节点 → 局部 Mask-Reconstruct → 输出新路径

用法:
    python -u simulation/rolling_horizon.py \
        --data dcc_50_r1_edod05_test.npz --ckpt phasec_st/step50000.ckpt \
        --edod 0.5 --horizon 24.0 --replan_interval 2.0
"""

import sys, os, argparse, time, numpy as np
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import MASKCO_ROOT, SCRIPTS_ROOT
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / 'models'))
sys.path.insert(0, str(MASKCO_ROOT))

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Vehicle:
    vid: int
    position: int = 0            # current node index (0=depot)
    next_node: int = -1          # next node to visit (-1 = idle)
    route: list = field(default_factory=list)  # remaining route [next_node, ...]
    load: float = 0.0
    temp_zone: int = 2           # 0=ambient, 1=refrig, 2=frozen (multi-temp capable)
    completed_nodes: list = field(default_factory=list)
    total_distance: float = 0.0
    spoilage_cost: float = 0.0
    # P0-4 Phase 4c: 温度轨迹追踪
    temp_log: list = field(default_factory=list)   # [(time, node, temp_class, segment_tti, segment_energy)]
    cumulative_tti: float = 0.0   # Time-Temperature Integrator
    cumulative_energy: float = 0.0  # Total refrigeration energy


@dataclass
class Order:
    oid: int                     # customer node index
    reveal_time: float           # when order becomes known
    tw_start: float
    tw_end: float
    demand: float
    temp_class: int              # 0=ambient, 1=refrig, 2=frozen
    service_time: float
    status: str = 'unknown'      # unknown → known → assigned → in_transit → completed
    assigned_vehicle: int = -1


class RollingHorizonSimulator:
    """历史动态仿真器；返回值带非权威 schema 标记。"""

    def __init__(self, dataset, capacity=50, speed=1.0, horizon=24.0):
        self.capacity = capacity
        self.speed = speed
        self.horizon = horizon
        self.clock = 0.0

        # 提取实例数据
        self.coords = dataset['coords']      # (N, nodes, 2)
        self.demands = dataset['demands']     # (N, nodes)
        self.tw_start = dataset['tw_start']   # (N, nodes)
        self.tw_end = dataset['tw_end']       # (N, nodes)
        self.service_time = dataset.get('service_time',
            np.zeros_like(self.demands, dtype=np.float32))
        self.temp_class = dataset.get('temp_class',
            np.zeros_like(self.demands, dtype=np.int32))
        self.reveal_time = dataset.get('reveal_time',
            np.zeros_like(self.demands, dtype=np.float32))
        self.num_instances = self.coords.shape[0]
        self.num_nodes = self.coords.shape[1]

        # 预计算距离矩阵
        diff = self.coords[:, :, None, :] - self.coords[:, None, :, :]
        self.dist_mat = np.sqrt((diff ** 2).sum(axis=-1))
        self.K_TEMP = np.array([0.01, 0.002, 0.0002])  # Arrhenius rates

        # P0-4 Phase 4c: 制冷能耗参数
        # T_target per temp_class: 0=ambient(25), 1=refrig(4), 2=frozen(-18)
        self.T_TARGET = np.array([25.0, 4.0, -18.0], dtype=np.float32)
        self.T_OUTSIDE = 25.0
        self.ENERGY_ALPHA = 1.0
        self.ENERGY_BETA = 0.05

    def _build_orders(self, inst_idx):
        """从数据集构建订单列表。"""
        orders = []
        for i in range(1, self.num_nodes):  # skip depot
            if self.demands[inst_idx, i] == 0:
                continue
            orders.append(Order(
                oid=i,
                reveal_time=float(self.reveal_time[inst_idx, i]),
                tw_start=float(self.tw_start[inst_idx, i]),
                tw_end=float(self.tw_end[inst_idx, i]),
                demand=float(self.demands[inst_idx, i]),
                temp_class=int(self.temp_class[inst_idx, i]),
                service_time=float(self.service_time[inst_idx, i]),
                status='known' if self.reveal_time[inst_idx, i] <= 0 else 'unknown',
            ))
        return orders

    def _distance(self, inst_idx, a, b):
        return self.dist_mat[inst_idx, a, b]

    def _travel_time(self, inst_idx, a, b):
        return self._distance(inst_idx, a, b) / self.speed

    def _spoilage(self, temp_class, travel_time):
        """Arrhenius 腐败成本。"""
        k = self.K_TEMP[min(int(temp_class), 2)]
        return 1.0 - np.exp(-k * travel_time)

    # P0-4 Phase 4c: 温度轨迹工具
    def _edge_tti(self, travel_time, cargo_temp_class, vehicle_temp_zone):
        """单段 Time-Temperature Integrator: travel_time × |T_target - T_ambient|"""
        t_target = self.T_TARGET[min(int(cargo_temp_class), 2)]
        # 车辆温区能力: multi-temp(2) → 可降至 frozen, refrig(1) → 可降至冷藏
        effective_temp = min(t_target, self.T_TARGET[min(int(vehicle_temp_zone), 2)])
        delta_t = abs(self.T_OUTSIDE - effective_temp)
        return travel_time * delta_t

    def _edge_energy(self, distance, cargo_temp_class):
        """单段制冷能耗: α × d × (1 + β × |T_outside - T_target|)"""
        t_target = self.T_TARGET[min(int(cargo_temp_class), 2)]
        delta_t = abs(self.T_OUTSIDE - t_target)
        return self.ENERGY_ALPHA * distance * (1.0 + self.ENERGY_BETA * delta_t)

    def _reveal_orders(self, orders):
        """在 clock 时刻揭示到期订单。"""
        newly_revealed = []
        for o in orders:
            if o.status == 'unknown' and o.reveal_time <= self.clock:
                o.status = 'known'
                newly_revealed.append(o)
        return newly_revealed

    def _freeze_mask(self, inst_idx, vehicles, orders):
        """
        构建冻结掩码：已完成的 + 车辆正在前往的节点 = 不可变。

        Returns:
            frozen_nodes: set of frozen node indices
            pending_orders: list of Order that need (re)planning
        """
        frozen = {0}  # depot always available
        for v in vehicles:
            frozen.add(v.position)
            if v.next_node > 0:
                frozen.add(v.next_node)
            frozen.update(v.completed_nodes)

        pending = [o for o in orders
                   if o.status in ('known',) and o.oid not in frozen]

        return frozen, pending

    def _route_to_vehicles(self, route, orders, inst_idx):
        """将路径分配给车辆，计算每辆车的状态。"""
        vehicles = []
        current_vehicle = None
        vehicle_id = 0

        for pos in range(len(route)):
            node = int(route[pos])
            if node == 0:  # depot = new vehicle starts
                if current_vehicle is not None:
                    vehicles.append(current_vehicle)
                current_vehicle = Vehicle(vid=vehicle_id)
                vehicle_id += 1
                continue
            if current_vehicle is None:
                continue
            current_vehicle.route.append(node)

        if current_vehicle is not None and current_vehicle.route:
            vehicles.append(current_vehicle)

        # 分配订单到车辆
        for v in vehicles:
            for node in v.route:
                for o in orders:
                    if o.oid == node:
                        o.status = 'assigned'
                        o.assigned_vehicle = v.vid

        return vehicles

    def simulate(self, inst_idx, initial_route, model_fn, replan_interval=2.0,
                 strategy='maskco_event', pending_threshold=3):
        """
        运行仿真。

        Args:
            inst_idx: 数据集中的实例索引
            initial_route: 初始路径 (pad_len,)
            model_fn: MaskCO 重规划函数 fn(pending_orders, frozen_nodes) → new_route
            replan_interval: 重规划检查间隔（小时）
            strategy: 'static' | 'full_reopt' | 'maskco_event'

        Returns:
            log: dict with metrics history
        """
        orders = self._build_orders(inst_idx)
        vehicles = self._route_to_vehicles(initial_route, orders, inst_idx)

        log = {
            'metric_schema': 'legacy-rolling-horizon-proxy-v1',
            'coldchain_authoritative': False,
            'clock': [], 'feas_rate': [], 'pending_count': [],
            'total_distance': [], 'spoilage_cost': [], 'replan_count': 0,
            # P0-4 Phase 4c: 温度/品质指标
            'total_tti': [], 'total_energy': [], 'quality_loss_rate': [],
        }

        self.clock = 0.0
        last_replan = -999

        # 初始出发
        for v in vehicles:
            if v.route:
                v.next_node = v.route.pop(0)
                v.position = 0

        while self.clock < self.horizon:
            # 1. 揭示新订单
            new_orders = self._reveal_orders(orders)
            pending = [o for o in orders if o.status == 'known']

            # 2. 推进车辆（含温度轨迹追踪 P0-4 Phase 4c）
            for v in vehicles:
                if v.next_node <= 0 and v.route:
                    v.next_node = v.route.pop(0)
                if v.next_node > 0:
                    travel_t = self._travel_time(inst_idx, v.position, v.next_node)
                    segment_dist = self._distance(inst_idx, v.position, v.next_node)
                    if self.clock + travel_t <= self.horizon:
                        # 到达
                        v.total_distance += segment_dist
                        v.spoilage_cost += self._spoilage(v.temp_zone, travel_t)

                        # P0-4: 记录温度轨迹
                        cargo_temp = int(self.temp_class[inst_idx, v.next_node]) if v.next_node < self.num_nodes else 0
                        segment_tti = self._edge_tti(travel_t, cargo_temp, v.temp_zone)
                        segment_energy = self._edge_energy(segment_dist, cargo_temp)
                        v.cumulative_tti += segment_tti
                        v.cumulative_energy += segment_energy
                        v.temp_log.append((self.clock, v.next_node, cargo_temp, segment_tti, segment_energy))

                        v.completed_nodes.append(v.next_node)
                        v.position = v.next_node
                        v.next_node = v.route.pop(0) if v.route else -1
                        # 标记订单完成
                        for o in orders:
                            if o.oid == v.position:
                                o.status = 'completed'

            # 3. 检查是否需要重规划
            pending_unassigned = [o for o in orders
                                  if o.status in ('known',) and o.assigned_vehicle < 0]
            should_replan = (
                len(pending_unassigned) >= pending_threshold and
                self.clock - last_replan >= replan_interval
            )

            if should_replan and strategy != 'static':
                last_replan = self.clock
                log['replan_count'] += 1

                if strategy == 'maskco_event':
                    frozen, _ = self._freeze_mask(inst_idx, vehicles, orders)
                    # 调用模型重规划（外部提供）
                    new_route = model_fn(inst_idx, pending_unassigned, frozen)
                    if new_route is not None:
                        vehicles = self._route_to_vehicles(new_route, orders, inst_idx)
                elif strategy == 'full_reopt':
                    # 全局重优化：所有未完成订单一起解
                    all_pending = [o for o in orders if o.status in ('known',)]
                    new_route = model_fn(inst_idx, all_pending, {0})
                    if new_route is not None:
                        vehicles = self._route_to_vehicles(new_route, orders, inst_idx)

            # 4. 记录日志
            pending_unassigned = [o for o in orders
                                  if o.status in ('known',) and o.assigned_vehicle < 0]
            completed = [o for o in orders if o.status == 'completed']
            tw_violations = sum(1 for o in completed
                               if o.tw_end < self.clock)  # rough estimate

            log['clock'].append(self.clock)
            log['pending_count'].append(len(pending_unassigned))
            log['total_distance'].append(sum(v.total_distance for v in vehicles))
            log['total_tti'].append(sum(v.cumulative_tti for v in vehicles))
            log['total_energy'].append(sum(v.cumulative_energy for v in vehicles))
            # 品质损失率 = spoilage / distance (归一化)
            total_sp = sum(v.spoilage_cost for v in vehicles)
            total_d = max(sum(v.total_distance for v in vehicles), 1e-6)
            log['quality_loss_rate'].append(total_sp / total_d)

            # 5. 推进时钟
            self.clock += 0.5  # 0.5h steps

        # 最终统计
        completed = [o for o in orders if o.status == 'completed']
        log['final_completed'] = len(completed)
        log['final_total_orders'] = len([o for o in orders if o.demand > 0])
        log['final_distance'] = sum(v.total_distance for v in vehicles)
        log['final_spoilage'] = sum(v.spoilage_cost for v in vehicles)
        log['final_vehicles_used'] = len([v for v in vehicles if v.total_distance > 0])
        # P0-4 Phase 4c: 冷链指标
        log['final_tti'] = sum(v.cumulative_tti for v in vehicles)
        log['final_energy'] = sum(v.cumulative_energy for v in vehicles)
        log['final_quality_loss_rate'] = log['final_spoilage'] / max(log['final_distance'], 1e-6)
        log['temp_logs'] = [v.temp_log for v in vehicles if v.temp_log]

        return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--edod', type=float, default=0.5)
    parser.add_argument('--horizon', type=float, default=24.0)
    parser.add_argument('--replan_interval', type=float, default=2.0)
    parser.add_argument('--pending_threshold', type=int, default=3)
    parser.add_argument('--num_instances', type=int, default=8)
    parser.add_argument('--output', type=str, default='simulation/logs/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"=== Rolling Horizon Simulation ===")
    print(f"  EDoD={args.edod}, Horizon={args.horizon}h, "
          f"ReplanInterval={args.replan_interval}h")

    dataset = dict(np.load(args.data))
    sim = RollingHorizonSimulator(dataset, horizon=args.horizon)

    # 简化版 model_fn：用贪心构造（后续替换为 MaskCO 模型调用）
    def dummy_model_fn(inst_idx, pending_orders, frozen_nodes):
        """贪心构造：最近邻 + TW 检查。"""
        pending_ids = [o.oid for o in pending_orders]
        if not pending_ids:
            return None

        route = [0]
        current = 0
        unvisited = set(pending_ids) - frozen_nodes
        current_time = sim.clock

        while unvisited:
            best, best_cost = None, float('inf')
            for j in unvisited:
                if j in frozen_nodes:
                    continue
                d = sim._distance(inst_idx, current, j)
                arr = current_time + sim.service_time[inst_idx, current] + d / sim.speed
                arr = max(arr, sim.tw_start[inst_idx, j])
                if arr <= sim.tw_end[inst_idx, j]:
                    if d < best_cost:
                        best_cost = d
                        best = j
            if best is None:
                route.append(0)
                current = 0
                current_time = 0
                continue
            route.append(best)
            unvisited.remove(best)
            current_time = max(current_time + sim.service_time[inst_idx, current] + sim._distance(inst_idx, current, best) / sim.speed, sim.tw_start[inst_idx, best])
            current = best

        route.append(0)
        return np.array(route, dtype=np.int32)

    # 跑 3 种策略
    for strategy in ['static', 'full_reopt', 'maskco_event']:
        print(f"\n--- Strategy: {strategy} ---")
        all_logs = []
        for i in range(min(args.num_instances, sim.num_instances)):
            initial = np.array([0] + [o.oid for o in sim._build_orders(i) if o.status == 'known'][:20] + [0], dtype=np.int32)
            log = sim.simulate(i, initial, dummy_model_fn,
                              replan_interval=args.replan_interval,
                              strategy=strategy,
                              pending_threshold=args.pending_threshold)
            all_logs.append(log)

        # 汇总
        completed_rates = [l['final_completed'] / max(l['final_total_orders'], 1) for l in all_logs]
        distances = [l['final_distance'] for l in all_logs]
        replans = [l['replan_count'] for l in all_logs]
        sp = [l['final_spoilage'] for l in all_logs]
        tti = [l['final_tti'] for l in all_logs]
        en = [l['final_energy'] for l in all_logs]
        ql_rate = [l['final_quality_loss_rate'] for l in all_logs]

        print(f"  Completed:     {np.mean(completed_rates):.1%}")
        print(f"  Distance:      {np.mean(distances):.1f}")
        print(f"  Replans:       {np.mean(replans):.1f}")
        print(f"  --- 历史代理指标（非 C0 / 不用于论文）---")
        print(f"  Spoilage proxy:{np.mean(sp):.4f}")
        print(f"  TTI proxy:     {np.mean(tti):.1f}")
        print(f"  Energy proxy:  {np.mean(en):.1f}")
        print(f"  Proxy/km:      {np.mean(ql_rate):.4f}")

        # 保存历史日志；文件字段只可用于旧实验复现。
        np.savez(os.path.join(args.output, f'sim_{strategy}_edod{str(args.edod).replace(".","")}.npz'),
                 completed_rates=completed_rates, distances=distances, replans=replans,
                 legacy_spoilage_proxy=sp, legacy_tti_proxy=tti,
                 legacy_energy_proxy=en, legacy_quality_loss_rate_proxy=ql_rate,
                 metric_schema=np.asarray('legacy-rolling-horizon-proxy-v1'),
                 coldchain_authoritative=np.asarray(False))


if __name__ == '__main__':
    main()
