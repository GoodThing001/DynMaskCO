"""DecisionView → PyVRP Model 转换器（B2 步骤 2，单事件静态映射）。

冻结整数化规则（协议级，改动即重跑 B2 Gate）：
  - 距离：SCALE 倍率后四舍五入 round（仅目标值）；
  - 行驶时间/服务时间/tw_start：向上取整 ceil（保守）；
  - tw_end / depot 返仓 deadline / 有效容量：向下取整 floor（保守）；
  - demand/capacity：原数据为整数则不缩放（float 时 demand ceil、capacity floor）。

模型结构：
  - 客户：只含可变池（visible unserved − committed_next），required=True，
    pickup=demand（禁止 delivery）；
  - 每个 replan 车辆一个独立 vehicle type（num_available=1）：
      start_depot = 车辆当前位置（anchor），tw_early = ceil(ready_time)；
      end_depot = 真实 depot（node 0），tw_late = floor(depot_tw_end)；
      effective_capacity = floor(total_capacity − current_load)
      （不用 initial_load：depot 语义下可能被当作可在 start depot 卸载；
       当前舱内货物继续由 StrictOnlineEnv/C0 跟踪）；
  - 边：所有 location 全对（含各车辆独立 start depot），distance=round、
    duration=ceil(travel)；travel 时间 = dist / tw_speed。

身份映射（0.14.0 实测索引空间）：
  - route.vehicle_type() 返回 vehicle type 的**加入顺序索引**（m.vehicle_types[i]）；
  - route 迭代出的 Activity.idx 是**客户加入顺序索引**（m.clients[i]）。
"""
import math

import numpy as np

INT_SCALE = 1000          # 距离与时间统一 3 位小数量化
_MAX_INT = np.iinfo(np.int64).max
# 覆盖-可行性尺度（B1.1 P0 修复）：不再用固定 1e6（会被默认 penalty 压过，
# 导致超载换覆盖）；按当前子问题界限计算：
#   coverage_prize = distance_upper_bound + 1
#      → 多覆盖一个客户的收益 > 任意两条可行解之间的总距离差（覆盖第一、距离第二）
#   feasibility_penalty = n_pool * coverage_prize + distance_upper_bound + 1
#      → 任何 ≥1 单位的约束违反都不值得（min==max 固定，不靠自适应增长）


def compute_scales(n_pool, n_vehicles, max_edge):
    """返回 (coverage_prize, feasibility_penalty, distance_upper_bound)。"""
    distance_upper_bound = (int(n_pool) + int(n_vehicles)) * int(max_edge)
    coverage_prize = distance_upper_bound + 1
    feasibility_penalty = int(n_pool) * coverage_prize + distance_upper_bound + 1
    from pyvrp.constants import MAX_VALUE
    estimated_max_penalised_cost = int(n_pool) * coverage_prize + feasibility_penalty
    if estimated_max_penalised_cost >= MAX_VALUE // 4:
        raise BuildError('prize/penalty scaling 可能超过 PyVRP 安全整数范围：'
                         f'estimated={estimated_max_penalised_cost} '
                         f'MAX_VALUE={MAX_VALUE}')
    return coverage_prize, feasibility_penalty, distance_upper_bound


class BuildError(RuntimeError):
    pass


def ceil_int(x):
    return int(math.ceil(float(x) * INT_SCALE))


def floor_int(x):
    return int(math.floor(float(x) * INT_SCALE))


def round_int(x):
    return int(round(float(x) * INT_SCALE))


class ProblemData:
    """转换产物：模型 + 身份映射（客户加入序 ↔ 客户 id；vt 加入序 ↔ 车辆 id）
    + 覆盖-可行性尺度（审计落盘）。"""

    def __init__(self, model, vid_by_vt_idx, customer_id_by_client_idx,
                 pool_customers, require_complete=False, scale_meta=None):
        self.model = model
        self.vid_by_vt_idx = vid_by_vt_idx
        self.customer_id_by_client_idx = customer_id_by_client_idx
        self.pool_customers = list(pool_customers)
        self.require_complete = bool(require_complete)
        self.scale_meta = scale_meta or {}


def _demand_int(view, i):
    d = float(view.demands[i])
    return int(d) if float(int(d)) == d else int(math.ceil(d))


def build_model(view, require_complete=False):
    """把 DecisionView 转成 PyVRP Model（单事件静态子问题）。

    require_complete=False（默认，deferral 语义）：客户为可选 + 界限计算的覆盖
      prize（覆盖第一、距离第二；feasibility penalty 保证不超载换覆盖），
      当前子问题覆盖不了的客户留给后续事件（deferred 由公共 audit 记录）。
    require_complete=True（严格观察模式）：required=True，覆盖不了 → infeasible
      → SolveError 显式失败（第一阶段观察 PyVRP 原生完整覆盖率用）。
    """
    from pyvrp import Model

    m = Model()
    node_idx = {node: i for i, node in enumerate(view.node_ids)}
    pool_nodes = [n for n, p in zip(view.node_ids, view.pool_mask) if p]
    n_pool = len(pool_nodes)
    n_vehicles = len(view.replan_ids)

    # ---- 覆盖-可行性尺度（按当前子问题界限计算）----
    max_edge = max(round_int(float(view.dist_mat[i][j]))
                   for i in range(len(view.node_ids))
                   for j in range(len(view.node_ids)))
    max_edge = max(max_edge, 1)
    if not require_complete and n_pool > 0:
        coverage_prize, feasibility_penalty, dist_bound = compute_scales(
            n_pool, n_vehicles, max_edge)
    else:
        coverage_prize, feasibility_penalty, dist_bound = 0, 0, 0
    scale_meta = {'n_pool': int(n_pool), 'n_vehicles': int(n_vehicles),
                  'max_edge': int(max_edge),
                  'distance_upper_bound': int(dist_bound),
                  'coverage_prize': int(coverage_prize),
                  'feasibility_penalty': int(feasibility_penalty),
                  'require_complete': bool(require_complete)}

    # ---- 真实 depot（end，所有车统一返仓）----
    end_loc = m.add_location(view.coords[node_idx[0]][0], view.coords[node_idx[0]][1])
    end_depot = m.add_depot(end_loc, tw_early=0,
                            tw_late=floor_int(view.depot_tw_end))

    # ---- 客户（可变池；required 或 界限计算覆盖 prize）----
    customer_id_by_client_idx = {}
    client_locs = []
    for i, (node, in_pool) in enumerate(zip(view.node_ids, view.pool_mask)):
        if not in_pool or node == 0:
            continue
        loc = m.add_location(view.coords[i][0], view.coords[i][1])
        client_locs.append(loc)     # 保留 wrapper 引用（data() 按对象 id 对账）
        m.add_client(
            loc,
            pickup=_demand_int(view, i),
            service_duration=ceil_int(view.service_time[i]),
            tw_early=ceil_int(view.tw_start[i]),
            tw_late=floor_int(view.tw_end[i]),
            required=require_complete,
            prize=0 if require_complete else coverage_prize,
        )
        # 客户加入序 = len(m.clients)-1
        customer_id_by_client_idx[len(m.clients) - 1] = node

    # ---- 每辆 replan 车辆：独立 start depot + 独立 vehicle type ----
    vehicle_views = {v.vehicle_id: v for v in view.vehicles}
    vid_by_vt_idx = {}
    start_loc_coord_idx = {}     # start depot location -> view.node_ids 坐标索引
    all_start_locs = []
    for vid in sorted(view.replan_ids):
        v = vehicle_views[vid]
        ia = v.anchor_idx
        start_loc = m.add_location(view.coords[ia][0], view.coords[ia][1])
        all_start_locs.append(start_loc)
        start_loc_coord_idx[len(all_start_locs) - 1] = ia
        start_depot = m.add_depot(start_loc, tw_early=ceil_int(v.ready_time),
                                  tw_late=_MAX_INT)
        load = float(v.load)
        eff_capacity = int(math.floor(float(view.capacity) - load))
        if eff_capacity < 0:
            raise BuildError(f'车辆 {vid} 当前载重 {load} 超过容量 {view.capacity}')
        vt_idx = len(m.vehicle_types)
        m.add_vehicle_type(
            1,
            capacity=eff_capacity,
            start_depot=start_depot,
            end_depot=end_depot,
            fixed_cost=0,
            tw_early=0,
            tw_late=floor_int(view.depot_tw_end),
            unit_distance_cost=1,
            unit_duration_cost=0,
            name=f'vehicle_{vid}',
        )
        if len(m.vehicle_types) != vt_idx + 1:
            raise BuildError('add_vehicle_type 未产生恰好一个 vehicle type')
        vid_by_vt_idx[vt_idx] = vid

    # ---- 边：所有 location 全对（用坐标索引取 view 距离矩阵）----
    coord_of = {}
    coord_of[len(coord_of)] = node_idx[0]          # end depot location
    for ci, node in customer_id_by_client_idx.items():
        coord_of[len(coord_of)] = node_idx[node]   # client locations
    for si, ia in start_loc_coord_idx.items():
        coord_of[len(coord_of)] = ia               # start depot locations

    locs = [end_loc] + client_locs + all_start_locs
    for a_i, a in enumerate(locs):
        ia = coord_of[a_i]
        for b_i, b in enumerate(locs):
            ib = coord_of[b_i]
            if a_i == b_i:
                m.add_edge(a, b, 0, 0)
            else:
                m.add_edge(a, b,
                           round_int(float(view.dist_mat[ia][ib])),
                           ceil_int(float(view.travel_mat[ia][ib])))

    return ProblemData(m, vid_by_vt_idx, customer_id_by_client_idx,
                       [n for n, p in zip(view.node_ids, view.pool_mask) if p],
                       require_complete=require_complete, scale_meta=scale_meta)
