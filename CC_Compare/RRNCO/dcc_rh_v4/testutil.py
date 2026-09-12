"""离线测试/快照共用的 mock view/vehicle 工厂（不 import method_adapter / Torch）。

构造 duck-typed view（满足 subproblem.py 声明的接口）+ 车辆快照。节点 id == 索引，
depot 恒为 0；dist = 欧氏距离，travel = dist（tw_speed=1）。

vehicle_specs：[(vehicle_id, anchor_node_id, ready_time, load, status,
                 mutable_suffix), ...]；anchor_idx 由本模块按 node_ids 反解。
"""
from types import SimpleNamespace


def euclid(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def make_vehicle(view, vehicle_id, anchor_node_id, ready_time, load,
                 status='ready', mutable_suffix=()):
    return SimpleNamespace(
        vehicle_id=int(vehicle_id),
        anchor_node_id=int(anchor_node_id),
        anchor_idx=view.node_ids.index(int(anchor_node_id)),
        ready_time=float(ready_time),
        load=float(load),
        status=status,
        mutable_suffix=tuple(int(x) for x in mutable_suffix),
    )


def make_view(coords, demands, tw_start, tw_end, service, vehicle_specs,
              replan_ids, capacity, depot_tw_end, pool_customer_ids,
              has_future_reveal=True):
    n = len(coords)
    node_ids = tuple(range(n))
    dist = [[euclid(coords[i], coords[j]) for j in range(n)] for i in range(n)]
    view = SimpleNamespace(
        node_ids=node_ids,
        coords=tuple(tuple(float(x) for x in c) for c in coords),
        demands=tuple(float(d) for d in demands),
        tw_start=tuple(float(t) for t in tw_start),
        tw_end=tuple(float(t) for t in tw_end),
        service_time=tuple(float(s) for s in service),
        dist_mat=tuple(tuple(float(x) for x in row) for row in dist),
        travel_mat=tuple(tuple(float(x) for x in row) for row in dist),
        capacity=float(capacity),
        depot_tw_end=float(depot_tw_end),
        pool_customer_ids=tuple(sorted(int(c) for c in pool_customer_ids)),
        vehicles=(),
        replan_ids=tuple(sorted(int(x) for x in replan_ids)),
        has_future_reveal=bool(has_future_reveal),
    )
    view.vehicles = tuple(
        make_vehicle(view, vid, anchor, rt, ld, st, suf)
        for (vid, anchor, rt, ld, st, suf) in vehicle_specs)
    return view
