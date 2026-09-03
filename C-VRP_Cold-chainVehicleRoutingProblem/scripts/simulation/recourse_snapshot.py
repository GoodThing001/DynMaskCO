"""
P0-S：RecourseSnapshotV2 —— 可精确恢复的 strict-online 决策点快照（02 手册 §P0-S2/S3）。

设计要点：
  - ordered mutable_suffix / served_route（padded -1，与 depot 0 区分）——旧 tail_mask 只能
    表达客户集合，无法恢复 insertion/continuation 所需的 route 顺序（P0-6）；
  - 完整 fleet/trace 状态（committed leg、dispatch/return、needs_replan/reason、
    services 记录、force_suffix）——恢复后与原 rollout 后缀 exact parity（Gate S）；
  - deterministic state_hash（不依赖 PYTHONHASHSEED，对 route 顺序敏感，风险 3.3）；
  - capture / clone / restore / resume / validate 五原语。

schema_version = 'recourse_snapshot_v2'
"""
import hashlib
import numpy as np

from strict_online_env import StrictOnlineEnv, VehicleState, VehicleTrace, ServiceRecord

SNAPSHOT_SCHEMA_VERSION = 'recourse_snapshot_v2'
PAD = -1


def _deep(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, list):
        return [_deep(y) for y in x]
    if isinstance(x, tuple):
        return tuple(_deep(y) for y in x)
    return x


def clone_snapshot(snapshot):
    """深拷贝（02 §P0-S3）；修改 clone 不影响原 snapshot。"""
    return {k: _deep(v) for k, v in snapshot.items()}


def _hash_update(h, v):
    if isinstance(v, np.ndarray):
        h.update(b'arr' + v.dtype.str.encode() + v.tobytes())
    elif isinstance(v, bool):
        h.update(b'bool1' if v else b'bool0')
    elif isinstance(v, int):
        h.update(b'int' + str(v).encode())
    elif isinstance(v, float):
        h.update(b'flt' + repr(v).encode())
    elif isinstance(v, str):
        h.update(b'str' + v.encode())
    elif isinstance(v, (list, tuple)):
        h.update(b'lst' + str(len(v)).encode())
        for x in v:
            _hash_update(h, x)
    elif v is None:
        h.update(b'none')
    else:
        h.update(b'oth' + repr(v).encode())


def snapshot_state_hash(snapshot):
    """确定性状态哈希（风险 3.3）：固定 dtype、canonical 遍历、对 route 顺序敏感。"""
    h = hashlib.sha256()
    h.update(SNAPSHOT_SCHEMA_VERSION.encode())
    for key in sorted(snapshot.keys()):
        if key == 'state_hash':
            continue
        h.update(key.encode())
        _hash_update(h, snapshot[key])
    return h.hexdigest()


def capture_recourse_snapshot(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                              served_mask, all_customers=None):
    """在决策点（_plan_and_commit 之前）捕获完整可恢复状态。"""
    K = len(vehicles)
    N = env.num_nodes
    if all_customers is None:
        all_customers = [i for i in range(1, N) if env.demands[inst_idx, i] > 0]
    max_tail = max((len(v.mutable_suffix) for v in vehicles), default=0)
    max_served = max((len(v.served_route) for v in vehicles), default=0)
    max_svcs = max((len(tr.services) for tr in traces), default=0)

    def _pad_nodes(arr, width):
        out = np.full(width, PAD, dtype=np.int32)
        if arr:
            out[:len(arr)] = np.asarray(arr, dtype=np.int32)
        return out

    def _pad_times(arr, width):
        out = np.full(width, np.nan, dtype=np.float64)
        if arr:
            out[:len(arr)] = np.asarray(arr, dtype=np.float64)
        return out

    def _f(x):
        return np.nan if x is None else float(x)

    vis = np.zeros(N, dtype=bool)
    vis[0] = True
    for c in all_customers:
        if env.reveal_time[inst_idx, c] <= clock + 1e-6:
            vis[c] = True

    snap = {
        'schema_version': SNAPSHOT_SCHEMA_VERSION,
        'instance_id': int(inst_idx),
        'event_id': int(event_id),
        'replan_count': int(env.replan_count),
        'clock': float(clock),
        'reveal_idx': int(reveal_idx),
        'num_vehicles': K,
        'num_nodes': N,
        'served_mask': np.asarray(served_mask, dtype=bool).copy(),
        'visible_mask': vis,
        'customer_universe': np.asarray(all_customers, dtype=np.int32),
        'vehicle_status': [str(v.status) for v in vehicles],
        'vehicle_node': np.asarray([int(v.current_node) for v in vehicles], dtype=np.int32),
        'vehicle_ready': np.asarray([float(v.ready_time) for v in vehicles], dtype=np.float64),
        'vehicle_load': np.asarray([float(v.current_load) for v in vehicles], dtype=np.float64),
        'committed_next': np.asarray(
            [PAD if v.committed_next is None else int(v.committed_next)
             for v in vehicles], dtype=np.int32),
        'committed_arrive': np.asarray([_f(v.committed_arrive) for v in vehicles], dtype=np.float64),
        'committed_finish': np.asarray([_f(v.committed_finish) for v in vehicles], dtype=np.float64),
        'needs_replan': np.asarray([bool(v.needs_replan) for v in vehicles], dtype=bool),
        'replan_reason': [v.replan_reason for v in vehicles],
        'mutable_suffix': np.stack([_pad_nodes(list(v.mutable_suffix), max_tail)
                                    for v in vehicles]) if K else np.zeros((0, 0), np.int32),
        'mutable_suffix_len': np.asarray([len(v.mutable_suffix) for v in vehicles], dtype=np.int32),
        'served_route': np.stack([_pad_nodes(list(v.served_route), max_served)
                                  for v in vehicles]) if K else np.zeros((0, 0), np.int32),
        'served_route_len': np.asarray([len(v.served_route) for v in vehicles], dtype=np.int32),
        'dispatch_time': np.asarray([_f(v.dispatch_time) for v in vehicles], dtype=np.float64),
        'return_finish': np.asarray([_f(v.return_finish) for v in vehicles], dtype=np.float64),
        'trace_dispatch_time': np.asarray([_f(tr.dispatch_time) for tr in traces], dtype=np.float64),
        'trace_return_depart': np.asarray([_f(tr.return_depart) for tr in traces], dtype=np.float64),
        'trace_return_arrival': np.asarray([_f(tr.return_arrival) for tr in traces], dtype=np.float64),
        'services_prev_node': np.stack([_pad_nodes([sr.prev_node for sr in tr.services], max_svcs)
                                        for tr in traces]) if K else np.zeros((0, 0), np.int32),
        'services_node': np.stack([_pad_nodes([sr.node for sr in tr.services], max_svcs)
                                   for tr in traces]) if K else np.zeros((0, 0), np.int32),
        'services_depart': np.stack([_pad_times([sr.depart_time for sr in tr.services], max_svcs)
                                     for tr in traces]) if K else np.zeros((0, 0), np.float64),
        'services_arrival': np.stack([_pad_times([sr.arrival_time for sr in tr.services], max_svcs)
                                      for tr in traces]) if K else np.zeros((0, 0), np.float64),
        'services_start': np.stack([_pad_times([sr.service_start for sr in tr.services], max_svcs)
                                    for tr in traces]) if K else np.zeros((0, 0), np.float64),
        'services_finish': np.stack([_pad_times([sr.service_finish for sr in tr.services], max_svcs)
                                     for tr in traces]) if K else np.zeros((0, 0), np.float64),
        'services_len': np.asarray([len(tr.services) for tr in traces], dtype=np.int32),
        'force_suffix': [[list(k), list(v)] for k, v in sorted(env.force_suffix.items())],
    }
    snap['state_hash'] = snapshot_state_hash(snap)
    return snap


def restore_recourse_snapshot(snapshot):
    """把 snapshot 恢复成 (vehicles, traces, served_mask) 运行态对象（exact round-trip）。"""
    if snapshot['schema_version'] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"snapshot schema {snapshot['schema_version']} != {SNAPSHOT_SCHEMA_VERSION}")
    K = int(snapshot['num_vehicles'])

    def _t(x):
        return None if np.isnan(x) else float(x)

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
        vehicles.append(v)

    traces = []
    for k in range(K):
        tr = VehicleTrace(vehicle_id=k)
        tr.dispatch_time = _t(snapshot['trace_dispatch_time'][k])
        tr.return_depart = _t(snapshot['trace_return_depart'][k])
        tr.return_arrival = _t(snapshot['trace_return_arrival'][k])
        ns = int(snapshot['services_len'][k])
        for s in range(ns):
            tr.services.append(ServiceRecord(
                vehicle_id=k,
                prev_node=int(snapshot['services_prev_node'][k][s]),
                node=int(snapshot['services_node'][k][s]),
                depart_time=float(snapshot['services_depart'][k][s]),
                arrival_time=float(snapshot['services_arrival'][k][s]),
                service_start=float(snapshot['services_start'][k][s]),
                service_finish=float(snapshot['services_finish'][k][s]),
            ))
        traces.append(tr)

    served_mask = snapshot['served_mask'].copy()
    return vehicles, traces, served_mask


def validate_snapshot_against_env(env, snapshot):
    """恢复前校验 snapshot 与 env/dataset 配置一致（schema、维度、universe、visible）。"""
    if snapshot['schema_version'] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"snapshot schema {snapshot['schema_version']} != {SNAPSHOT_SCHEMA_VERSION}")
    if int(snapshot['num_vehicles']) != env.num_vehicles:
        raise ValueError(f"num_vehicles {snapshot['num_vehicles']} != env {env.num_vehicles}")
    if int(snapshot['num_nodes']) != env.num_nodes:
        raise ValueError(f"num_nodes {snapshot['num_nodes']} != env {env.num_nodes}")
    inst = int(snapshot['instance_id'])
    if not (0 <= inst < env.num_instances):
        raise ValueError(f"instance_id {inst} out of range")
    custs = [i for i in range(1, env.num_nodes) if env.demands[inst, i] > 0]
    if list(snapshot['customer_universe']) != custs:
        raise ValueError("customer_universe mismatch")
    if not bool(snapshot['served_mask'][0]):
        raise ValueError("served_mask[0] (depot) must be True")
    clock = float(snapshot['clock'])
    vis = np.zeros(env.num_nodes, dtype=bool)
    vis[0] = True
    for c in custs:
        if env.reveal_time[inst, c] <= clock + 1e-6:
            vis[c] = True
    if not np.array_equal(vis, snapshot['visible_mask']):
        raise ValueError("visible_mask mismatch vs reveal_time/clock")


def resume_from_snapshot(dataset, snapshot, replanner, capacity=50, tw_speed=1.0, num_vehicles=25):
    """02 §P0-S3：构造可继续的环境；随后 env.run_resumed(snapshot) 得 (traces, served_mask)。"""
    env = StrictOnlineEnv(dataset, capacity, tw_speed, num_vehicles, replanner=replanner)
    validate_snapshot_against_env(env, snapshot)
    return env
