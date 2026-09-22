"""DynMaskCO-CC 共享 context 构造（M1-edge-v1）：训练与在线共用。

修复三类接线问题：
  1. incumbent P0 必须来自 post-baseline planner（跑 continuation），不能读 pre-plan
     快照 suffix；
  2. 候选池边界 = decision_pool（只锁 committed + 非 mutable 车冻结 tail；mutable 车
     suffix 客户允许重分配）；
  3. 快照 JSON 往返后 state_hash 不复现（numpy→list 序列化差异），用有定义的类型恢复
     重建 vehicles（保留原 state_hash 作身份记录，不做静默关闭校验）。

纯 NumPy/对象层，无 JAX。训练（从 JSON snapshot）与在线（从 live vehicles）都经此构造
相同的「真实 incumbent / 可见压缩 / A0·M·A_in / 候选端点 / 显式特征 / 在线合法 mask」。
"""
import numpy as np

from strict_online_env import VehicleState
from recourse_snapshot import CC_SNAPSHOT_SCHEMA_VERSION, PAD
from action_contract import build_vehicle_plans


def cc_state_dict(state):
    """把 VehicleColdChainState 序列化为特征提取所需的 dict（舱温/分区载重）。

    训练 snapshot 里是 asdict(state) 的 dict；在线只有对象，这里取相同字段。
    """
    if state is None:
        return None
    return {
        'compartment_temperature_c': tuple(state.compartment_temperature_c),
        'zone_load': tuple(state.zone_load),
    }


def restore_vehicles_from_json(snapshot):
    """从 JSON 往返后的 snapshot 重建 vehicles（有定义的类型恢复，不做 state_hash 校验）。

    state_hash 在 JSON 往返后不可复现（numpy array → list 的序列化差异），因此这里按
    restore_recourse_snapshot 的同一字段语义重建 VehicleState；原始 state_hash 保留作
    身份记录（调用方负责记录）。缺失字段/非法状态会显式报错，不静默跳过。
    """
    if snapshot['schema_version'] not in ('recourse_snapshot_v2', CC_SNAPSHOT_SCHEMA_VERSION):
        raise ValueError("unsupported snapshot schema %s" % snapshot['schema_version'])
    K = int(snapshot['num_vehicles'])
    cc_enabled = snapshot['schema_version'] == CC_SNAPSHOT_SCHEMA_VERSION

    def _t(x):
        return None if (x is None or np.isnan(x)) else float(x)

    vehicles = []
    for k in range(K):
        v = VehicleState(vehicle_id=k)
        v.status = snapshot['vehicle_status'][k]
        v.current_node = int(snapshot['vehicle_node'][k])
        v.ready_time = float(snapshot['vehicle_ready'][k])
        v.current_load = float(snapshot['vehicle_load'][k])
        cn = int(snapshot['committed_next'][k])
        v.committed_next = None if cn == PAD else cn
        v.committed_arrive = _t(snapshot['committed_arrive'][k])
        v.committed_finish = _t(snapshot['committed_finish'][k])
        v.needs_replan = bool(snapshot['needs_replan'][k])
        v.replan_reason = snapshot['replan_reason'][k]
        lt = int(snapshot['mutable_suffix_len'][k])
        v.mutable_suffix = [int(x) for x in snapshot['mutable_suffix'][k][:lt]]
        ls = int(snapshot['served_route_len'][k])
        v.served_route = [int(x) for x in snapshot['served_route'][k][:ls]]
        v.dispatch_time = _t(snapshot['dispatch_time'][k])
        v.return_finish = _t(snapshot['return_finish'][k])
        if cc_enabled:
            from recourse_snapshot import _restore_coldchain_state
            v.coldchain_state = _restore_coldchain_state(
                snapshot['vehicle_coldchain_state'][k])
            v.coldchain_updated_time = _t(snapshot['coldchain_updated_time'][k])
        vehicles.append(v)
    return vehicles


def _decision_context(snapshot):
    inst_idx = int(snapshot['instance_id'])
    clock = float(snapshot['clock'])
    served_mask = snapshot['served_mask']
    visible_ids = [int(c) for c in snapshot['customer_universe']
                   if bool(snapshot['visible_mask'][int(c)])]
    return inst_idx, clock, served_mask, visible_ids


def compute_incumbent_plans(env, snapshot, continuation):
    """在决策点跑 baseline planner 得到 post-baseline incumbent P0（与 _incumbent_plans 一致）。"""
    inst_idx, clock, served_mask, visible_ids = _decision_context(snapshot)
    vehicles = restore_vehicles_from_json(snapshot)
    env.prepare_decision_point(clock, vehicles)
    if hasattr(continuation, 'restore_state'):
        continuation.restore_state(snapshot.get('replanner_state'))
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    continuation.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                      replan_ids=replan_ids)
    return build_vehicle_plans(env, inst_idx, vehicles), vehicles, replan_ids


def mutable_scope(vehicles):
    """本次 replan 的 mutable 车 id（idle/ready 且 needs_replan）。"""
    return {v.vehicle_id for v in vehicles
            if v.status in ('idle', 'ready') and v.needs_replan}


def decision_pool_from_vehicles(vehicles, served_mask, visible_ids):
    """decision_pool 边界：visible & unserved & 未 committed & 不在非 mutable 车冻结 tail。

    mutable 车 suffix 中的客户允许重分配；committed 客户与非 mutable 车的 tail 客户锁定。
    """
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    locked = set()
    for v in vehicles:
        if v.status == 'committed' and v.committed_next not in (None, 0):
            locked.add(int(v.committed_next))
        if v.vehicle_id not in replan_ids:
            for n in v.mutable_suffix:
                if int(n) != 0:
                    locked.add(int(n))
    return [int(c) for c in visible_ids if not served_mask[int(c)] and int(c) not in locked]


def mask_candidate_pool(vehicles, served_mask, visible_ids, protected=frozenset(),
                        plans=None):
    """统一 mask 候选池 = decision_pool_from_vehicles − protected 车辆完整 tail。

    训练 / 部署 / 回放 / 固定状态评估共用该入口，保证 mask 只来自可变范围：
      - 排除 committed 客户（committed_next）；
      - 排除 non-replan 车 tail（decision_pool_from_vehicles 已做）；
      - 排除 protected 车 tail（预留车）。
    plans 可选：传 FleetPlan（如 P0）时用 plan.suffix 取 protected tail；否则回退
    vehicles.mutable_suffix。
    """
    pool = list(decision_pool_from_vehicles(vehicles, served_mask, visible_ids))
    if not protected:
        return pool
    protected_tail = set()
    if plans is not None:
        for vid in protected:
            p = plans.get(vid)
            if p is not None:
                for x in p.suffix:
                    if int(x) > 0:
                        protected_tail.add(int(x))
    else:
        for v in vehicles:
            if v.vehicle_id in protected:
                for n in v.mutable_suffix:
                    if int(n) != 0:
                        protected_tail.add(int(n))
    return [int(c) for c in pool if int(c) not in protected_tail]


def validate_mask_scope(vehicles, served_mask, visible_ids, plans, mask, mutable_ids,
                        protected=frozenset()):
    """强制断言 mask 合规（越界则 raise AssertionError）。

    每个 mask 客户必须：⊆ decision_pool、来自 mutable_ids 车计划、非 committed_next、
    非 protected 车 tail。plans 为当前 FleetPlan（P0）。
    """
    decision_pool = set(decision_pool_from_vehicles(vehicles, served_mask, visible_ids))
    committed_next = {int(v.committed_next) for v in vehicles
                      if v.status == 'committed' and v.committed_next not in (None, 0)}
    protected_tail = set()
    for vid in protected:
        p = plans.get(vid)
        if p is not None:
            for x in p.suffix:
                if int(x) > 0:
                    protected_tail.add(int(x))
    mutable_plan_customers = set()
    for vid, p in plans.items():
        if vid in mutable_ids:
            for x in p.suffix:
                if int(x) > 0:
                    mutable_plan_customers.add(int(x))
    for c in mask:
        c = int(c)
        assert c in decision_pool, f"mask {c} not in decision_pool"
        assert c in mutable_plan_customers, f"mask {c} not from mutable_ids vehicle"
        assert c not in committed_next, f"mask {c} is committed_next"
        assert c not in protected_tail, f"mask {c} in protected tail"


def validate_full_partition(committed, plans, deferred, universe, served_mask=None,
                            future_fn=None):
    """完整状态分区校验：committed ⊎ planned suffix ⊎ explicit deferred 互斥且覆盖 universe。

    committed: 正在执行 leg 的客户集合（可空）。
    plans:     FleetPlan（vid -> VehiclePlan），suffix 为有序客户。
    deferred:  显式延期客户集合。
    universe: 期望集合 = 已揭示、未服务、未 committed 的客户。
    served_mask/future_fn 可选，用于额外校验「已服务/未来客户不得入计划」。

    返回 (ok, detail dict)。detail 分类：duplicate_suffix / committed_in_suffix /
    committed_and_deferred / suffix_and_deferred / missing / extra / served_planned /
    future_planned。
    """
    from collections import Counter
    committed = set(int(c) for c in committed)
    suffix_counts = Counter()
    for p in plans.values():
        for x in p.suffix:
            suffix_counts[int(x)] += 1
    suffix_set = set(suffix_counts)
    deferred = set(int(c) for c in deferred)
    universe = set(int(c) for c in universe)

    duplicate_suffix = sorted(c for c, k in suffix_counts.items() if k > 1)
    committed_in_suffix = sorted(committed & suffix_set)
    committed_and_deferred = sorted(committed & deferred)
    suffix_and_deferred = sorted(suffix_set & deferred)
    planned = suffix_set | deferred
    missing = sorted(c for c in universe if c not in planned)
    extra = sorted(c for c in planned if c not in universe and c not in committed)
    served_planned = []
    future_planned = []
    if served_mask is not None:
        served_planned = sorted(c for c in planned if served_mask[int(c)])
    if future_fn is not None:
        future_planned = sorted(c for c in planned if future_fn(int(c)))

    ok = (not duplicate_suffix and not committed_in_suffix
          and not committed_and_deferred and not suffix_and_deferred
          and not missing and not extra and not served_planned and not future_planned)
    return ok, {
        'duplicate_suffix': duplicate_suffix,
        'committed_in_suffix': committed_in_suffix,
        'committed_and_deferred': committed_and_deferred,
        'suffix_and_deferred': suffix_and_deferred,
        'missing': missing,
        'extra': extra,
        'served_planned': served_planned,
        'future_planned': future_planned,
    }
