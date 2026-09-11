"""Proposal 模式适配接口（B1.1 P0-1）：结构隔离未来信息 + Bridge 深层写保护。

外部方法只实现 `ExternalReplanner.propose(view) -> PlanProposal`：
  - **不接收 env / dataset / VehicleState 对象**——只拿到 DecisionView（公开节点
    特征 + 车辆公开状态快照），未来客户的数量/身份/坐标/TW/需求在结构上不可达；
  - proposal 只包含 replan_ids 车辆的完整新 suffix（普通数据，无副作用）；
  - Bridge 负责校验 proposal（客户 ∈ 可变池、无重复、键 ⊆ replan_ids、键完整）、
    写回真实车辆、并重建 plan 核对 proposal hash（写回一致）。

non-anticipatory 从「统计测试保证」升级为「结构保证 + 统计测试双保险」。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from strict_online_env import Replanner
from action_contract import build_vehicle_plans, plan_hash


class ContractViolation(RuntimeError):
    """外部 adapter 违反 baseline contract（proposal 非法 / 写保护触发）。"""


# ---------------------------------------------------------------------------
# DecisionView / PlanProposal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleView:
    """车辆公开状态快照（不可变；anchor 为 node_ids 中的索引）。"""
    vehicle_id: int
    status: str                # idle / ready / committed / returning / closed
    anchor_idx: int            # index into DecisionView.node_ids
    anchor_node_id: int        # 真实节点 id（committed 车 = committed_next）
    ready_time: float          # 该 anchor 的可出发/可规划时刻（committed = committed_finish）
    load: float
    mutable_suffix: tuple      # 当前未 committed 计划（车队公开状态；尾部保留语义用）


@dataclass(frozen=True)
class DecisionView:
    """单个决策点的完整公开视图。不含任何未来客户信息（结构保证）。

    node_ids = [0] ∪ 车辆 anchor 节点 ∪ 可变池客户；
    pool_mask[node_ids 位置] = 该节点是可分配客户（可变池）。

    可变池 ownership 语义（B1.1 P0 修复，2026-09-09）：
      - 非 replan 车辆的 committed_next：冻结（protected）；
      - 非 replan 车辆的完整 mutable_suffix：冻结（protected，plan persistence）；
      - replan 车辆原有 suffix：进入 pool，允许在本次 replan 车辆之间重新分配；
      - future / served / invalid 客户：不进 pool。
    visible_unserved == protected_customer_ids ∪ pool_customer_ids（无交叉）。
    """
    inst_idx: int
    event_id: int
    clock: float
    objective: str
    capacity: float
    tw_speed: float
    node_ids: tuple                # 公开节点真实 id（含 0 与 anchor）
    pool_mask: tuple               # bool：是否可变客户池（可写入 proposal suffix）
    protected_customer_ids: tuple  # 非 replan 车辆冻结持有的客户（audit）
    pool_customer_ids: tuple       # 可变池客户（audit）
    coords: tuple                  # (len(node_ids), 2) float（仅公开节点）
    demands: tuple                 # 仅 pool 客户有意义；anchor 客户为真实需求（历史公开）
    tw_start: tuple
    tw_end: tuple
    service_time: tuple
    dist_mat: tuple                # (len, len) 公开节点间距离
    travel_mat: tuple              # dist / tw_speed
    temp_class: tuple              # coldchain 时有效；否则全 0
    initial_quality: tuple         # coldchain 时有效；否则全 1
    depot_tw_end: float
    vehicles: tuple                # 全部车辆公开快照（含 committed）
    replan_ids: tuple              # 本次可写车辆 id（idle/ready 且 needs_replan）
    has_future_reveal: bool        # 是否还有未 reveal 客户（二元信号，非内容）

    def node_index(self, node_id: int) -> int:
        try:
            return self.node_ids.index(int(node_id))
        except ValueError:
            raise ContractViolation(
                f'DecisionView 不包含节点 {node_id}（未来/已服务客户结构不可达）')


@dataclass(frozen=True)
class PlanProposal:
    """adapter 返回的纯数据计划（无副作用、无对象引用）。

    suffixes：replan_ids 全部车辆的完整新 mutable suffix（客户真实 id，
              末尾 0 = 服务完返仓；[] = WAIT）。
    solve_meta：可选的求解审计信息（如 coverage prize / penalty 尺度），
              由 runner 写入事件记录（审计用）。
    """
    suffixes: dict                 # vehicle_id -> tuple[int, ...]
    model_input_customers: tuple   # 实际喂给模型/求解器的客户集合（审计）
    fallback_triggered: bool
    model_runtime_s: float
    solve_meta: dict = None

    def proposal_hash(self) -> str:
        h = hashlib.sha256()
        for vid in sorted(self.suffixes):
            h.update(str(vid).encode())
            h.update(b':')
            h.update(str(tuple(self.suffixes[vid])).encode())
            h.update(b';')
        return h.hexdigest()


# ---------------------------------------------------------------------------
# DecisionView 构造（唯一读取 env 全量数据的代码路径；只挑选公开子集）
# ---------------------------------------------------------------------------

def build_decision_view(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                        replan_ids) -> DecisionView:
    replan_set = set(int(v) for v in replan_ids)
    committed_next_ids = set()
    protected_customers = set()
    anchors = []
    for v in vehicles:
        if v.status == 'committed' and v.committed_next not in (None, 0):
            committed_next_ids.add(int(v.committed_next))
            anchors.append(int(v.committed_next))
        elif v.status in ('idle', 'ready', 'returning') and int(v.current_node) != 0:
            anchors.append(int(v.current_node))
        # ownership 保护：非 replan 车辆的 committed_next 与完整 suffix 冻结
        if int(v.vehicle_id) not in replan_set:
            if v.committed_next not in (None, 0):
                protected_customers.add(int(v.committed_next))
            for n in v.mutable_suffix:
                if int(n) != 0:
                    protected_customers.add(int(n))

    visible_unserved = {int(c) for c in visible_ids
                        if not bool(served_mask[int(c)])}
    # 可变池 = 可见未服务 − protected（非 replan 冻结持有）− committed_next
    pool = sorted(visible_unserved - protected_customers)
    if pool and committed_next_ids:
        pool = sorted(set(pool) - committed_next_ids)
    node_ids = [0] + sorted(set(anchors) | set(pool))   # committed anchor 可能在可见未服务内 → union
    if len(set(node_ids)) != len(node_ids):
        raise ContractViolation('view 节点重复')
    idx = {n: i for i, n in enumerate(node_ids)}

    coords = np.asarray(env.coords[inst_idx], dtype=np.float32)[node_ids]
    demands = np.asarray(env.demands[inst_idx], dtype=np.float32)[node_ids]
    tw_start = np.asarray(env.tw_start[inst_idx], dtype=np.float32)[node_ids]
    tw_end = np.asarray(env.tw_end[inst_idx], dtype=np.float32)[node_ids]
    service = np.asarray(env.service_time[inst_idx], dtype=np.float32)[node_ids]
    dist = np.asarray(env.dist_mat[inst_idx], dtype=np.float32)[node_ids][:, node_ids]
    travel = dist / env.tw_speed
    if env.coldchain_contract is not None:
        temp_class = np.asarray(env.temp_class[inst_idx], dtype=np.int32)[node_ids]
        initial_quality = np.asarray(env.initial_quality[inst_idx],
                                     dtype=np.float32)[node_ids]
    else:
        temp_class = np.zeros(len(node_ids), dtype=np.int32)
        initial_quality = np.ones(len(node_ids), dtype=np.float32)

    vviews = []
    for v in vehicles:
        if v.status == 'committed' and v.committed_next not in (None, 0):
            anchor_node = int(v.committed_next)
            ready_t = float(v.committed_finish)
            load = float(v.current_load) + float(env.demands[inst_idx, anchor_node])
        else:
            anchor_node = int(v.current_node)
            ready_t = float(v.ready_time)
            load = float(v.current_load)
        vviews.append(VehicleView(
            vehicle_id=int(v.vehicle_id), status=str(v.status),
            anchor_idx=idx[anchor_node], anchor_node_id=anchor_node,
            ready_time=ready_t, load=load,
            mutable_suffix=tuple(int(x) for x in v.mutable_suffix)))

    return DecisionView(
        inst_idx=int(inst_idx), event_id=int(getattr(env, 'event_id', -1)),
        clock=float(clock), objective='coldchain' if env.coldchain_contract else 'distance',
        capacity=float(env.capacity), tw_speed=float(env.tw_speed),
        node_ids=tuple(node_ids),
        pool_mask=tuple(n in pool for n in node_ids),
        protected_customer_ids=tuple(sorted(protected_customers)),
        pool_customer_ids=tuple(pool),
        coords=tuple(map(tuple, coords.tolist())),
        demands=tuple(float(x) for x in demands),
        tw_start=tuple(float(x) for x in tw_start),
        tw_end=tuple(float(x) for x in tw_end),
        service_time=tuple(float(x) for x in service),
        dist_mat=tuple(map(tuple, dist.tolist())),
        travel_mat=tuple(map(tuple, travel.tolist())),
        temp_class=tuple(int(x) for x in temp_class),
        initial_quality=tuple(float(x) for x in initial_quality),
        depot_tw_end=float(env.tw_end[inst_idx, 0]),
        vehicles=tuple(vviews),
        replan_ids=tuple(sorted(int(x) for x in replan_ids)),
        has_future_reveal=bool(env.has_future_reveal(inst_idx, clock, served_mask)),
    )


# ---------------------------------------------------------------------------
# ExternalReplanner（propose 接口）
# ---------------------------------------------------------------------------

class ExternalReplanner(Replanner):
    """外部方法接口：实现 propose(view) -> PlanProposal。不接触 env/vehicles。"""

    method_name = 'external'
    method_revision = '0'
    adapter_revision = '0'
    checkpoint_hash = 'none'          # 权重身份：adapter 加载权重时计算并写入
    runtime_budget = None             # 每事件时间预算（秒）；None = 无预算（协议禁用）

    def __init__(self):
        self.fallback_triggered = False
        self.last_model_input_customers = []
        self.last_model_runtime_s = 0.0

    @classmethod
    def compute_files(cls):
        """adapter 计算依赖文件（参与 code_hash）。

        条目支持命名空间前缀：
          'project/<relpath>' → 项目 scripts 根（如 'project/simulation/jf1h_repair.py'）
          'common/<relpath>'  → common 根
          无前缀            → adapter 模块目录（逻辑路径 adapter/<name>）
        只列「直接改变决策/映射/子问题构造」的文件；入口驱动、环境身份检查
        等归 control 集合（SOURCE_MANIFEST 分类，不进 code_hash）。
        """
        return ()

    def propose(self, view: DecisionView) -> PlanProposal:
        raise NotImplementedError

    def time_model(self):
        """上下文管理器：记录模型/求解器实际耗时（写入 proposal.model_runtime_s）。"""
        parent = self

        class _Timer:
            def __enter__(self2):
                self2._t0 = time.perf_counter()
                return self2

            def __exit__(self2, *exc):
                parent.last_model_runtime_s = time.perf_counter() - self2._t0
                return False

        return _Timer()

    def mark_fallback(self):
        self.fallback_triggered = True

    # plan() 由 ExternalReplanner 统一实现（Bridge 入口）；子类不重写。
    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
             replan_ids=None):
        view = build_decision_view(env, inst_idx, clock, vehicles, served_mask,
                                   visible_ids, replan_ids or ())
        proposal = self.propose(view)
        if not isinstance(proposal, PlanProposal):
            raise ContractViolation('propose 必须返回 PlanProposal')
        self.fallback_triggered = bool(proposal.fallback_triggered)
        self.last_model_input_customers = [int(x) for x in proposal.model_input_customers]
        self.last_model_runtime_s = float(proposal.model_runtime_s)
        return view, proposal


# ---------------------------------------------------------------------------
# BridgeReplanner（写保护 + proposal 校验 + 写回核对）
# ---------------------------------------------------------------------------

_VEHICLE_FIELDS = ('status', 'current_node', 'ready_time', 'current_load',
                   'committed_next', 'committed_arrive', 'committed_finish',
                   'dispatch_time', 'return_finish', 'needs_replan', 'replan_reason')


class NativeReplanner(Replanner):
    """项目原生 Replanner 的包装基类（如 JF1-H-F）。

    不走 proposal 模式（原生 plan 直接接收 env/vehicles）；bridge 仍做
    写保护 + plans 捕获 + 公共审计。原生 replanner 自带 repair 层时，
    repair 审计字段由项目 _eval 附加（repair_applicable=True）。
    """

    method_name = 'native'
    method_revision = '0'
    adapter_revision = '0'
    checkpoint_hash = 'none'
    runtime_budget = None

    @classmethod
    def compute_files(cls):
        return ()

    def export_state(self):
        return None

    def restore_state(self, state):
        return None

    def sync_deferred_from_vehicles(self, vehicles):
        return None

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
             replan_ids=None):
        raise NotImplementedError


class NativeBridgeReplanner(Replanner):
    """原生 Replanner 的契约强制层（与 BridgeReplanner 同款写保护，
    无 proposal/view；plan_hook_records 的 view/solve_meta 为 None，
    model_input 为空——事件记录的 pool/protected 分区审计不适用）。

    隔离级别声明（B1.2）：External adapter 由 DecisionView 提供**结构性**
    non-anticipation；Native adapter 是受信任项目代码，通过环境不可变
    fingerprint（实例数组 + capacity/tw_speed）+ 车辆写保护 + future-
    perturbation prefix parity 审计约束，**不具备**同等结构隔离。
    """

    def __init__(self, inner, method_name=None, adapter_revision=None):
        self.inner = inner
        self.method_name = method_name or getattr(inner, 'method_name', 'native')
        self.adapter_revision = adapter_revision or getattr(inner,
                                                            'adapter_revision', '0')
        self.checkpoint_hash = getattr(inner, 'checkpoint_hash', 'none')
        self.plans_before = None
        self.plans_after = None
        self.plan_hook_records = []

    def export_state(self):
        return self.inner.export_state()

    def restore_state(self, state):
        self.inner.restore_state(state)

    def sync_deferred_from_vehicles(self, vehicles):
        self.inner.sync_deferred_from_vehicles(vehicles)

    # ---- repair 层透传（条件式：只有 inner 真正拥有 repair 字段时才暴露；
    # 无 repair 层的原生方法不得让 _eval 拿到 None.repair_stats）----
    def __getattr__(self, name):
        if name in ('repair_stats', 'deferred_customers'):
            inner = self.__dict__.get('inner')
            if inner is not None and hasattr(inner, name):
                return getattr(inner, name)
        raise AttributeError(name)

    # ---- 环境不可变 fingerprint（B1.2）：原生 replanner 直接接触 env，
    # 除车辆状态外还须保证实例数组与关键标量不被改写 ----
    @staticmethod
    def _env_fingerprint(env, inst_idx):
        h = hashlib.sha256()
        arrays = {}
        for name in ('coords', 'demands', 'tw_start', 'tw_end', 'service_time',
                     'reveal_time', 'dist_mat'):
            arrays[name] = getattr(env, name)[inst_idx]
        if env.coldchain_contract is not None:
            arrays['temp_class'] = env.temp_class[inst_idx]
            arrays['initial_quality'] = env.initial_quality[inst_idx]
        for name in sorted(arrays):
            a = np.asarray(arrays[name])
            h.update(name.encode())
            h.update(b'\x00')
            h.update(np.ascontiguousarray(a).tobytes())
            h.update(b'\x00')
        h.update(f'capacity={float(env.capacity)}'.encode())
        h.update(b'\x00')
        h.update(f'tw_speed={float(env.tw_speed)}'.encode())
        return h.hexdigest()

    @staticmethod
    def _capture(vehicles):
        return [(int(v.vehicle_id), tuple(getattr(v, k) for k in _VEHICLE_FIELDS),
                 tuple(int(x) for x in v.mutable_suffix)) for v in vehicles]

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
             replan_ids=None):
        replan_ids = set(int(x) for x in (replan_ids or ()))
        served_bytes = np.asarray(served_mask).tobytes()
        visible_tuple = tuple(int(x) for x in visible_ids)
        before = self._capture(vehicles)
        env_fp_before = self._env_fingerprint(env, inst_idx)
        self.plans_before = build_vehicle_plans(env, inst_idx, vehicles)

        self.inner.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                        replan_ids=replan_ids)

        after = self._capture(vehicles)
        if self._env_fingerprint(env, inst_idx) != env_fp_before:
            raise ContractViolation(
                'adapter 改写了环境数据（实例数组 / capacity / tw_speed 只读）')
        if [c[0] for c in before] != [c[0] for c in after]:
            raise ContractViolation('车辆列表被增删/重排/改写 vehicle_id')
        bmap = {c[0]: c for c in before}
        for vid, fields, suffix in after:
            old_fields, old_suffix = bmap[vid][1], bmap[vid][2]
            if fields != old_fields:
                raise ContractViolation(
                    f'adapter 修改了车辆 {vid} 的状态字段（只允许写 mutable_suffix；'
                    f'before={old_fields} after={fields}）')
            if suffix != old_suffix and vid not in replan_ids:
                raise ContractViolation(
                    f'adapter 修改了非 replan 车辆 {vid} 的 mutable_suffix '
                    f'({old_suffix} -> {suffix})')
        if np.asarray(served_mask).tobytes() != served_bytes:
            raise ContractViolation('adapter 改写了 served_mask（只读输入）')
        if tuple(int(x) for x in visible_ids) != visible_tuple:
            raise ContractViolation('adapter 改写了 visible_ids（只读输入）')

        self.plans_after = build_vehicle_plans(env, inst_idx, vehicles)
        self.plan_hook_records.append({
            'event_id': int(getattr(env, 'event_id', -1)),
            'clock': float(clock),
            'plans_before': self.plans_before,
            'plans_after': self.plans_after,
            'plan_hash_before': plan_hash(self.plans_before),
            'plan_hash_after': plan_hash(self.plans_after),
            'proposal_hash': None,
            'model_input_customers': [],
            'model_runtime_s': 0.0,
            'fallback_triggered': False,
            'solve_meta': None,
            'view': None,
        })


class BridgeReplanner(Replanner):
    """契约强制层：DecisionView 构造 → propose → 校验 → 写回 → 重建 plan 核对。"""

    def __init__(self, inner, method_name=None, adapter_revision=None):
        self.inner = inner
        self.method_name = method_name or getattr(inner, 'method_name', 'external')
        self.adapter_revision = adapter_revision or getattr(inner, 'adapter_revision', '0')
        self.checkpoint_hash = getattr(inner, 'checkpoint_hash', 'none')
        self.plans_before = None
        self.plans_after = None
        self.plan_hook_records = []

    def export_state(self):
        if hasattr(self.inner, 'export_state'):
            return self.inner.export_state()
        return None

    def restore_state(self, state):
        if hasattr(self.inner, 'restore_state'):
            self.inner.restore_state(state)

    def sync_deferred_from_vehicles(self, vehicles):
        if hasattr(self.inner, 'sync_deferred_from_vehicles'):
            self.inner.sync_deferred_from_vehicles(vehicles)

    @staticmethod
    def _capture(vehicles):
        return [(int(v.vehicle_id), tuple(getattr(v, k) for k in _VEHICLE_FIELDS),
                 tuple(int(x) for x in v.mutable_suffix)) for v in vehicles]

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
             replan_ids=None):
        replan_ids = set(int(x) for x in (replan_ids or ()))
        served_bytes = np.asarray(served_mask).tobytes()
        visible_tuple = tuple(int(x) for x in visible_ids)
        before = self._capture(vehicles)
        self.plans_before = build_vehicle_plans(env, inst_idx, vehicles)

        view, proposal = self.inner.plan(env, inst_idx, clock, vehicles,
                                         served_mask, visible_ids,
                                         replan_ids=replan_ids)

        # ---- 车辆列表完整性：数量/顺序/ID 不变，非 mutable 车逐字段不变 ----
        after = self._capture(vehicles)
        if [c[0] for c in before] != [c[0] for c in after]:
            raise ContractViolation('车辆列表被增删/重排/改写 vehicle_id')
        bmap = {c[0]: c for c in before}
        for vid, fields, suffix in after:
            old_fields, old_suffix = bmap[vid][1], bmap[vid][2]
            if fields != old_fields:
                raise ContractViolation(
                    f'adapter 修改了车辆 {vid} 的状态字段（只允许写 mutable_suffix；'
                    f'before={old_fields} after={fields}）')
            if suffix != old_suffix and vid not in replan_ids:
                raise ContractViolation(
                    f'adapter 修改了非 replan 车辆 {vid} 的 mutable_suffix '
                    f'({old_suffix} -> {suffix})')
        if np.asarray(served_mask).tobytes() != served_bytes:
            raise ContractViolation('adapter 改写了 served_mask（只读输入）')
        if tuple(int(x) for x in visible_ids) != visible_tuple:
            raise ContractViolation('adapter 改写了 visible_ids（只读输入）')

        # ---- proposal 校验 ----
        pool_ids = {n for n, m in zip(view.node_ids, view.pool_mask) if m}
        if set(proposal.suffixes.keys()) != replan_ids:
            missing = replan_ids - set(proposal.suffixes.keys())
            extra = set(proposal.suffixes.keys()) - replan_ids
            raise ContractViolation(
                f'proposal 键集合 != replan_ids：missing={sorted(missing)} '
                f'extra={sorted(extra)}')
        seen = set()
        for vid, suffix in proposal.suffixes.items():
            suf = [int(x) for x in suffix]
            for c in suf:
                if c == 0:
                    continue
                if c not in pool_ids:
                    raise ContractViolation(
                        f'proposal 车辆 {vid} suffix 含非法客户 {c}'
                        f'（不在可变池：未来/已服务/无效客户）')
                if c in seen:
                    raise ContractViolation(f'proposal 跨车重复客户 {c}')
                seen.add(c)
        for vid, suffix in proposal.suffixes.items():
            suf = [int(x) for x in suffix]
            if len(set(x for x in suf if x != 0)) != len([x for x in suf if x != 0]):
                raise ContractViolation(f'proposal 车辆 {vid} suffix 内重复')

        # ---- 写回（只写 replan_ids 的 idle/ready 车）----
        for v in vehicles:
            if v.vehicle_id in replan_ids:
                if v.status not in ('idle', 'ready'):
                    raise ContractViolation(
                        f'replan_ids 含非 idle/ready 车辆 {v.vehicle_id}')
                v.mutable_suffix = [int(x) for x in proposal.suffixes[v.vehicle_id]]

        # ---- 重建 plan 核对 proposal（写回一致）----
        plans_after = build_vehicle_plans(env, inst_idx, vehicles)
        for vid in replan_ids:
            expected = tuple(int(x) for x in proposal.suffixes[vid] if int(x) != 0)
            actual = plans_after[vid].suffix
            if actual != expected:
                raise ContractViolation(
                    f'写回不一致：车辆 {vid} plan suffix={actual} '
                    f'proposal={expected}')

        self.plans_after = plans_after
        self.plan_hook_records.append({
            'event_id': int(getattr(env, 'event_id', -1)),
            'clock': float(clock),
            'plans_before': self.plans_before,
            'plans_after': self.plans_after,
            'plan_hash_before': plan_hash(self.plans_before),
            'plan_hash_after': plan_hash(self.plans_after),
            'proposal_hash': proposal.proposal_hash(),
            'model_input_customers': [int(x) for x in proposal.model_input_customers],
            'model_runtime_s': float(proposal.model_runtime_s),
            'fallback_triggered': bool(proposal.fallback_triggered),
            'solve_meta': dict(proposal.solve_meta) if proposal.solve_meta else None,
            'view': view,
        })
