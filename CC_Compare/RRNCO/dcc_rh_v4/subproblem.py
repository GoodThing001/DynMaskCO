"""逐车可见子问题构造（RRNCO-RH R0.5 Stage A.1，纯 NumPy + stdlib）。

每个可重规划车辆单独构造**自包含**子问题：节点严格 = depot + 当前 anchor + 当前
pool，未来 / 已服务 / 受保护客户结构性缺席。SubProblem 直接复制允许字段的数组，
**不保留对 view 的引用**——provider 无法访问 sub_nodes 之外的数据（结构隔离，
而非仅靠 canonical hash 过滤）。

canonical_payload()/canonical_hash() 只依赖子问题可见内容，供未来扰动测试与
身份冻结：改变未揭示订单的坐标/需求/TW/温区/数量，只要可见 pool 与 anchor 不变，
hash 不变。

本模块对 `view` 做 duck-typing，不 import DecisionView / method_adapter / Torch。
`view` 需提供：node_ids / coords / demands / tw_start / tw_end / service_time /
dist_mat / travel_mat / capacity / depot_tw_end / pool_customer_ids。
"""
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SubProblem:
    """单辆 replan 车的可见子问题（自包含不可变，无 view 引用）。"""
    vehicle_id: int
    anchor_node_id: int      # 真实节点 id（0=depot 或某客户）
    anchor_idx: int          # index into node_ids（sub_nodes）
    ready_time: float
    current_load: float
    pool_customer_ids: tuple  # 可见 pool 客户（真实 id，sorted）
    # 自包含数组（与 node_ids 对齐）
    node_ids: tuple          # = sub_nodes 真实 id（[0, anchor?, *pool]）
    coords: tuple
    demands: tuple
    tw_start: tuple
    tw_end: tuple
    service_time: tuple
    dist_mat: tuple          # (n_sub, n_sub)
    travel_mat: tuple        # (n_sub, n_sub)
    capacity: float
    depot_tw_end: float

    @property
    def sub_nodes(self) -> tuple:
        return self.node_ids

    def node_index(self, node_id) -> int:
        return self.node_ids.index(int(node_id))

    def canonical_payload(self) -> dict:
        return {
            'vehicle_id': int(self.vehicle_id),
            'anchor_node_id': int(self.anchor_node_id),
            'ready_time': float(self.ready_time),
            'current_load': float(self.current_load),
            'nodes': [int(n) for n in self.node_ids],
            'coords': [[float(x) for x in c] for c in self.coords],
            'demands': [float(d) for d in self.demands],
            'tw_start': [float(t) for t in self.tw_start],
            'tw_end': [float(t) for t in self.tw_end],
            'service_time': [float(s) for s in self.service_time],
            'dist_mat': [[float(x) for x in row] for row in self.dist_mat],
            'travel_mat': [[float(x) for x in row] for row in self.travel_mat],
            'capacity': float(self.capacity),
            'depot_tw_end': float(self.depot_tw_end),
        }

    def canonical_hash(self) -> str:
        payload = json.dumps(self.canonical_payload(), sort_keys=True).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()


def build_subproblem(view, vehicle) -> SubProblem:
    """从 view + 一个 VehicleView 构造自包含子问题（不保留 view 引用）。"""
    pool = tuple(sorted(int(c) for c in view.pool_customer_ids))
    nodes = [0]
    if int(vehicle.anchor_node_id) != 0:
        nodes.append(int(vehicle.anchor_node_id))
    for c in pool:
        if c not in nodes:
            nodes.append(c)
    idx = {n: i for i, n in enumerate(view.node_ids)}
    return SubProblem(
        vehicle_id=int(vehicle.vehicle_id),
        anchor_node_id=int(vehicle.anchor_node_id),
        anchor_idx=nodes.index(int(vehicle.anchor_node_id)),
        ready_time=float(vehicle.ready_time),
        current_load=float(vehicle.load),
        pool_customer_ids=pool,
        node_ids=tuple(nodes),
        coords=tuple(tuple(float(x) for x in view.coords[idx[n]]) for n in nodes),
        demands=tuple(float(view.demands[idx[n]]) for n in nodes),
        tw_start=tuple(float(view.tw_start[idx[n]]) for n in nodes),
        tw_end=tuple(float(view.tw_end[idx[n]]) for n in nodes),
        service_time=tuple(float(view.service_time[idx[n]]) for n in nodes),
        dist_mat=tuple(tuple(float(view.dist_mat[idx[a]][idx[b]]) for b in nodes)
                       for a in nodes),
        travel_mat=tuple(tuple(float(view.travel_mat[idx[a]][idx[b]]) for b in nodes)
                         for a in nodes),
        capacity=float(view.capacity),
        depot_tw_end=float(view.depot_tw_end),
    )
