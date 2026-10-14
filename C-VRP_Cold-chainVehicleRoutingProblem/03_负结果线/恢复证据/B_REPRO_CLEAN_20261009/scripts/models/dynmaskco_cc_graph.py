"""DynMaskCO-CC 图适配（M1-edge-v1）：可见节点压缩 + 计划邻接 + event mask + timestep。

纯 NumPy，无 JAX/模型，可本地测。对称邻接只是 MaskCO 的计划条件与重构目标，**不是执行真值**
（执行仍由有向 FleetPlan + 车辆状态 + action certificate 决定）。

约定：
  - 可见节点集合 = {0(depot)} ∪ 已揭示客户，depot 固定局部索引 0；
  - served / committed / mutable 是独立属性，不把「不可修改」误当「不可见」；
  - 多车路线并集成对称二值邻接，无跨路线伪边，无 depot 分隔符 token；
  - event_mask=True = 需要重构的边位，False = 保留条件；遮蔽值 0（删边式条件）；
  - timestep = 保留的无向边比例（上三角），原邻接无边时 0。
"""
import numpy as np


def build_visible_index(visible_ids):
    """可见节点集合 = {0(depot)} ∪ visible_ids → 局部索引映射。

    Returns (node_to_local, local_to_node, N_visible)。
    """
    nodes = [0] + sorted(int(i) for i in visible_ids if int(i) != 0)
    node_to_local = {int(n): i for i, n in enumerate(nodes)}
    local_to_node = [int(n) for n in nodes]
    return node_to_local, local_to_node, len(nodes)


def _add_edge(A, i, j, node_to_local):
    if int(i) == int(j):
        return
    if int(i) not in node_to_local or int(j) not in node_to_local:
        raise KeyError(f"plan node {i} or {j} 不在可见节点集合")
    li, lj = node_to_local[int(i)], node_to_local[int(j)]
    A[li, lj] = 1.0
    A[lj, li] = 1.0


def build_plan_adjacency(plans, node_to_local, N_visible):
    """从 FleetPlan 构建对称二值邻接（剩余计划：anchor → suffix → depot）。

    每辆车的剩余计划 route = [anchor_node] + suffix + [0]；边 = 连续节点对（跳过自环）。
    多车取并集。空 suffix 且 anchor==depot 的车不产生边。
    """
    A = np.zeros((N_visible, N_visible), np.float32)
    for vid in sorted(plans):
        p = plans[vid]
        route = [int(p.anchor_node)] + [int(x) for x in p.suffix] + [0]
        for a, b in zip(route, route[1:]):
            _add_edge(A, a, b, node_to_local)
    return A


def build_event_mask(A0, candidate_adjs):
    """event mask M = ⋁_a (A_a != A_0)（加边与删边都进入），对角排除，对称。

    A_in = A0 ⊙ (1 - M)。Returns (M bool, A_in float32)。
    """
    N = A0.shape[0]
    M = np.zeros((N, N), bool)
    for Aa in candidate_adjs:
        M |= (np.asarray(Aa) != np.asarray(A0))
    M = M | M.T
    np.fill_diagonal(M, False)
    A_in = np.asarray(A0) * (1.0 - M.astype(np.float32))
    return M, A_in.astype(np.float32)


def compute_timestep(A0, A_in):
    """保留的无向边比例（上三角统计）；原邻接无边时 0。"""
    upper0 = np.triu(np.asarray(A0), k=1)
    upper_in = np.triu(np.asarray(A_in), k=1)
    n_orig = int((upper0 > 0.5).sum())
    if n_orig == 0:
        return 0.0
    n_retained = int((upper_in > 0.5).sum())
    return n_retained / n_orig


def resolve_target_anchor_node(slot, plans):
    """解析 slot 到实际 target VehiclePlan.anchor_node（负 anchor 是 depot-route 车编码）。

    - new_route（anchor=-1）→ 空 idle@depot 车，anchor_node=depot(0)；
    - anchored anchor<0（depot-route，vid=-anchor-1）→ anchor_node=depot(0)；
    - anchored anchor>0（ready@customer / committed）→ anchor_node=anchor。
    """
    if slot.kind == 'new_route':
        return 0
    from action_contract import _match_anchored
    vid = _match_anchored(plans, slot.anchor)
    return int(plans[vid].anchor_node)


def resolve_action_endpoints(customer, slot, predecessor, successor, plans, node_to_local):
    """候选动作 4 端点 → 局部可见索引 + valid 位。

    order = [customer, target_anchor, predecessor, successor]。
    slot=None（DEFER 伪动作）→ customer 有效，其余 valid=0（零向量占位，索引 0）。
    普通动作/KEEP 4 端点都有效（depot=局部 0）。缺失(valid=0)绝不与 depot(valid=1) 混同。
    """
    ends = np.zeros(4, np.int32)
    valid = np.zeros(4, bool)
    ends[0] = node_to_local[int(customer)]
    valid[0] = True
    if slot is None:
        return ends, valid
    anchor_node = resolve_target_anchor_node(slot, plans)
    ends[1] = node_to_local[int(anchor_node)]
    valid[1] = True
    ends[2] = node_to_local[int(predecessor)]
    valid[2] = True
    ends[3] = node_to_local[int(successor)]
    valid[3] = True
    return ends, valid
