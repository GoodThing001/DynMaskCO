"""DecisionView → OR-Tools RoutingModel 子问题构造（OR1-OR3）。

节点布局（索引 = RoutingIndexManager 节点）：
  0                = 真实 depot（end，所有车统一返仓）
  1 .. V           = virtual start（每辆 replan 车一个，V = len(replan_ids)）
  V+1 .. V+C       = 可变池客户

冻结整数化（与 PyVRP 同方向，协议级）：
  distance      = round(d × SCALE)
  travel time   = ceil(travel × SCALE)；arc transit(i,j) = travel(i,j) + service(i)
  tw_start      = ceil；tw_end = floor；depot deadline = floor(depot_tw_end)
  demand/capacity：整数不缩放（float 时 demand ceil、capacity floor）

关键语义：
  - 每车 virtual start：位置=anchor 坐标、service=0、时间 cumul 钉在
    ceil(ready_time)（SetRange 单点）；end=真实 depot；
  - 容量（标准建模，OR4.2）：fix_start_cumul_to_zero=False，start cumul 为
    自由变量，再用 capacity_dim.CumulVar(Start(v)).SetValue(ceil(current_load))
    钉死「start capacity cumul = current_load」；virtual start demand=0。
    后续 pickup 在其上累加，绝不可能在 depot 前释放当前货物；
  - 可选客户：每个 pool 客户 AddDisjunction(drop_penalty)；
    drop_penalty = distance_upper_bound + 1，保证「少 drop 一个客户优先于
    任意距离改善」（service-first）；int64 溢出检查；
  - 车辆数严格等于 len(replan_ids)，不补 parked 车。
"""
import math

import numpy as np

INT_SCALE = 1000
_MAX_INT64 = np.iinfo(np.int64).max


class BuildError(RuntimeError):
    pass


def ceil_int(x):
    return int(math.ceil(float(x) * INT_SCALE))


def floor_int(x):
    return int(math.floor(float(x) * INT_SCALE))


def round_int(x):
    return int(round(float(x) * INT_SCALE))


def compute_scales(n_pool, n_vehicles, max_arc):
    """service-first drop penalty（OR3）：覆盖任意距离差，检查 int64 溢出。"""
    distance_upper_bound = (int(n_pool) + int(n_vehicles)) * int(max_arc)
    drop_penalty = distance_upper_bound + 1
    if drop_penalty * (int(n_pool) + 1) >= _MAX_INT64 // 4:
        raise BuildError('drop penalty 可能超过 OR-Tools 安全整数范围：'
                         f'drop_penalty={drop_penalty} n_pool={n_pool}')
    return drop_penalty, distance_upper_bound


def empty_suffix_policy(view, v):
    """统一的空路线策略（OR4.2：builder 与 mapper 共用，禁止两套条件漂移）。

    WAIT（()）：有 future 且立即返仓可行 —— 车辆原地等待，
      求解模型中该车 virtual start→depot（及车辆 0 的 start→dummy）
      弧的成本与时间必须为 0，目标对账同计 0；
    CLOSE（(0,)）：真实 anchor→depot 成本与时间。
    """
    if not view.has_future_reveal:
        return (0,)
    i_anchor = v.anchor_idx
    return_time = (float(v.ready_time)
                   + float(view.travel_mat[i_anchor][view.node_index(0)]))
    if return_time <= float(view.depot_tw_end) + 1e-6:
        return ()
    return (0,)


class ProblemData:
    """构造产物：manager/routing + 身份映射 + 尺度审计。"""

    def __init__(self, manager, routing, vid_by_vehicle_index,
                 customer_id_by_node_index, pool_customers, scale_meta):
        self.manager = manager
        self.routing = routing
        self.vid_by_vehicle_index = vid_by_vehicle_index   # vehicle idx -> replan vid
        self.customer_id_by_node_index = customer_id_by_node_index  # node -> customer id
        self.pool_customers = list(pool_customers)
        self.scale_meta = scale_meta


def _demand_int(view, i):
    d = float(view.demands[i])
    return int(d) if float(int(d)) == d else int(math.ceil(d))


def build_model(view):
    from ortools.constraint_solver import pywrapcp

    node_idx = {n: i for i, n in enumerate(view.node_ids)}
    pool = [n for n, p in zip(view.node_ids, view.pool_mask) if p]
    replan_ids = sorted(view.replan_ids)
    V = len(replan_ids)
    C = len(pool)

    # ---- 输入边界校验（OR4.1）----
    if V == 0:
        raise BuildError('replan_ids 为空（无车辆可规划）')
    if len(set(replan_ids)) != len(replan_ids):
        raise BuildError(f'replan_ids 含重复: {replan_ids}')
    if len(set(pool)) != len(pool):
        raise BuildError('pool 含重复客户')
    if set(pool) & set(view.protected_customer_ids):
        raise BuildError('客户同时属于 protected 与 pool')
    veh_ids = [v.vehicle_id for v in view.vehicles]
    if len(set(veh_ids)) != len(veh_ids):
        raise BuildError('vehicle id 重复')
    # INTEGERIZED_TW_EMPTY：ceil(tw_start) > floor(tw_end) 的客户强制 drop
    # （整数化后 TW 为空），不让 OR-Tools 内部抛模糊错误
    forced_dropped = []
    for c in pool:
        i = node_idx[c]
        if ceil_int(view.tw_start[i]) > floor_int(view.tw_end[i]):
            forced_dropped.append(c)
    model_pool = [c for c in pool if c not in forced_dropped]
    C = len(model_pool)         # 实际建模客户数（forced-dropped 不建节点）
    n_nodes = 1 + V + C + V     # +V = 每车一个 dummy 终端
    # 节点布局：0=depot；1..V=virtual start；V+1..V+C=客户；
    # V+C+1..V+C+V=每车 dummy 终端（depot 坐标/0 需求/0 服务/全时段 TW）
    # dummy 意义（OR5 修复）：
    #   1) 保证「所有客户均不可服务」时 first-solution 仍非空；
    #   2) 钉死空路线的目标语义——OR-Tools 9.11.4210 对空路线 start→end 弧
    #      不计成本（已实测：2 空车 objective=0），而 env 侧 CLOSE 要真实返仓。
    #      每车 dummy 把空路线变成 start→dummy_v→end：
    #        CLOSE：dummy 弧 = 真实 anchor→depot 成本/时间（求解目标与写回一致）；
    #        WAIT：dummy 弧置 0（原地等待，与写回 () 一致）。
    DUMMY_SENTINEL = -1
    depot_node = 0
    virtual_start_nodes = list(range(1, 1 + V))
    customer_nodes = list(range(1 + V, 1 + V + C))
    dummy_nodes = list(range(1 + V + C, 1 + V + C + V))

    # 坐标/距离矩阵（int；virtual start 坐标 = anchor 坐标）
    coord_idx = {0: node_idx[0]}                    # depot
    for k, vid in enumerate(replan_ids):
        coord_idx[1 + k] = _vehicle_anchor(view, vid)
    for k, c in enumerate(model_pool):
        coord_idx[1 + V + k] = node_idx[c]
    for k in range(V):
        coord_idx[1 + V + C + k] = node_idx[0]    # 每车 dummy 坐标 = depot

    dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    travel_mat = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    service_int = np.zeros(n_nodes, dtype=np.int64)      # from-node service
    for i in range(n_nodes):
        for j in range(n_nodes):
            d = float(view.dist_mat[coord_idx[i]][coord_idx[j]])
            tr = float(view.travel_mat[coord_idx[i]][coord_idx[j]])
            dist_mat[i, j] = round_int(d)
            travel_mat[i, j] = ceil_int(tr)
    for k, c in enumerate(model_pool):
        service_int[1 + V + k] = ceil_int(view.service_time[node_idx[c]])

    # WAIT 车辆 dummy 弧零成本/零时间（OR5 修复）：
    #   空路线 = start→dummy_v→end；WAIT 时原地等待 → dummy 弧置 0；
    #   CLOSE 时保留真实 anchor→depot 成本/时间。
    for k, vid in enumerate(replan_ids):
        v = _vehicle(view, vid)
        if empty_suffix_policy(view, v) == ():
            dummy_k = 1 + V + C + k
            dist_mat[1 + k, dummy_k] = 0
            travel_mat[1 + k, dummy_k] = 0

    # transit(i,j) = travel(i,j) + service(i)
    transit = travel_mat + service_int[:, None]

    # demands：virtual start = 0（初始载重只由 capacity_dim.CumulVar(Start)
    # .SetValue(load) 表达）；客户 = demand；depot/dummy = 0
    demand = np.zeros(n_nodes, dtype=np.int64)
    for k, c in enumerate(model_pool):
        demand[1 + V + k] = _demand_int(view, node_idx[c])

    manager = pywrapcp.RoutingIndexManager(n_nodes, V, virtual_start_nodes,
                                           [depot_node] * V)
    routing = pywrapcp.RoutingModel(manager)

    # 关键：回调的 i/j 是 manager 索引（顺序 = starts 在前、ends/其余在后），
    # 与矩阵的节点号顺序不同——必须经 IndexToNode 转节点号。
    # end 索引（routing.End(v)）映射回 depot 节点 0，返仓弧自动正确。
    def dist_cb(i, j):
        return int(dist_mat[manager.IndexToNode(i)][manager.IndexToNode(j)])

    def transit_cb(i, j):
        return int(transit[manager.IndexToNode(i)][manager.IndexToNode(j)])

    def demand_cb(i, j):
        return int(demand[manager.IndexToNode(j)])

    routing.SetArcCostEvaluatorOfAllVehicles(
        routing.RegisterTransitCallback(dist_cb))

    # ---- 容量维度（OR4.1 标准建模，9.11.4210 实测稳定）----
    # fix_start_cumul_to_zero=False：start cumul 是自由变量，再用
    # capacity_dim.CumulVar(Start(v)).SetValue(ceil(current_load)) 精确钉死
    # 「start capacity cumul = current_load」；后续 pickup 在其上累加，
    # 绝不可能在 depot 前释放当前货物。virtual start demand=0。
    # 注意：必须用普通 (i,j) transit callback——RegisterUnaryTransitCallback
    # + AddDimensionWithVehicleCapacity 在本版本会返回错误状态（已实测隔离）。
    capacity = int(math.floor(float(view.capacity)))
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterTransitCallback(demand_cb),
        0, [capacity] * V, False, 'Capacity')
    capacity_dim = routing.GetDimensionOrDie('Capacity')
    for k, vid in enumerate(replan_ids):
        v = _vehicle(view, vid)
        capacity_dim.CumulVar(routing.Start(k)).SetValue(
            int(math.ceil(float(v.load))))

    # ---- 时间维度：start 时间自由、再逐点钉范围 ----
    horizon = floor_int(view.depot_tw_end)
    routing.AddDimension(routing.RegisterTransitCallback(transit_cb),
                         horizon, horizon, False, 'Time')
    td = routing.GetDimensionOrDie('Time')
    for k, vid in enumerate(replan_ids):
        v = _vehicle(view, vid)
        start = ceil_int(v.ready_time)
        td.CumulVar(manager.NodeToIndex(1 + k)).SetRange(start, start)
    for k, c in enumerate(model_pool):
        i = node_idx[c]
        td.CumulVar(manager.NodeToIndex(1 + V + k)).SetRange(
            ceil_int(view.tw_start[i]), floor_int(view.tw_end[i]))
    for k in range(V):
        td.CumulVar(manager.NodeToIndex(1 + V + C + k)).SetRange(0, horizon)
    # 注意：不对 end depot 节点 SetRange——dimension horizon 已约束所有 cumul
    # ∈ [0, horizon]（end 到达 ≤ floor(depot_tw_end) 由 horizon 强制）；
    # 且 ortools 9.11.4210 win wheel 对 end depot cumul SetRange 有原生
    # segfault（已实测隔离），该行为冗余约束。

    # ---- dummy 终端约束（OR5）：每车 dummy 只允许本车，且后继必为其 End ----
    for k in range(V):
        dummy_k_idx = manager.NodeToIndex(1 + V + C + k)
        routing.SetAllowedVehiclesForIndex([k], dummy_k_idx)
        routing.solver().Add(routing.NextVar(dummy_k_idx) == routing.End(k))

    # ---- 可选客户：service-first drop penalty（OR3）----
    # forced_dropped（INTEGERIZED_TW_EMPTY）不建 disjunction——不进模型，
    # 由 adapter 直接并入 dropped 集合（分区复核时仍是 pool 成员）
    max_arc = int(dist_mat.max())
    max_arc = max(max_arc, 1)
    drop_penalty, distance_upper_bound = compute_scales(C, V, max_arc)
    for k in range(C):
        routing.AddDisjunction([manager.NodeToIndex(1 + V + k)],
                               int(drop_penalty))

    vid_by_vehicle_index = {k: vid for k, vid in enumerate(replan_ids)}
    customer_id_by_node_index = {1 + V + k: c for k, c in enumerate(model_pool)}
    for k in range(V):
        customer_id_by_node_index[1 + V + C + k] = DUMMY_SENTINEL   # 映射时过滤
    scale_meta = {
        'n_pool': int(len(pool)), 'n_model_pool': int(C),
        'n_vehicles': int(V), 'max_arc': int(max_arc),
        'distance_upper_bound': int(distance_upper_bound),
        'drop_penalty': int(drop_penalty),
        'int_scale': INT_SCALE,
        'horizon_int': horizon,
        'initial_loads': [int(math.ceil(float(_vehicle(view, vid).load)))
                          for vid in replan_ids],
        'n_dummy': int(V),
        'forced_dropped': [int(c) for c in forced_dropped],
        'forced_drop_reason': 'INTEGERIZED_TW_EMPTY',
    }
    problem = ProblemData(manager, routing, vid_by_vehicle_index,
                          customer_id_by_node_index, pool, scale_meta)
    problem.forced_dropped = list(forced_dropped)
    return problem


def _vehicle(view, vid):
    return next(x for x in view.vehicles if x.vehicle_id == vid)


def _vehicle_anchor(view, vid):
    return _vehicle(view, vid).anchor_idx
