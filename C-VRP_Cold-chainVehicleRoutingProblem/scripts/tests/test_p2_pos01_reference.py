"""P2-POS-01A synthetic CPU acceptance. No datasets/results/checkpoints are written.

Run this file explicitly with CPU env, PYTHONDONTWRITEBYTECODE=1 and pytest's
cacheprovider disabled. Only the original cvrp100.ckpt is read. One and only one
synthetic head optimizer step is executed. The real backbone is never optimized.

Synthetic fixed-head bias=1 is used to exercise COMMIT independently of learning;
it is NOT a trained/calibrated policy. Fault doubles are ONLY fault-injection
counterexamples. Real M-pre encoding, decoding, scoring, reconstruction and
coldchain transitions are exercised separately and together without doubles.

Real SAA/mask continuation, future generation, real-day smoke and training remain
NOT_RUN/NOT_QUALIFIED; the engine parity test uses an explicitly synthetic fixed
acceptance ledger, not a substitute claimed to be the strong SAA baseline.

实施验收记录（2026-10-19，基线 5a20221c8a2a1cccfb22e8d5d328e8b741122b62）：
  * 11项非优化器测试最终全部通过（19.30s）；梯度测试单独运行1次通过（9.09s）。
    后续复验排除了梯度项，未执行第二次优化器更新或200步拟合。
  * 真实cvrp100.ckpt：CPU、(16,6)层、(512,512)维；固定公开51节点桶实际前向。
    完整参数摘要 7b43ece321619b7cea8d88b35c3053e8e6cd82f40f968631380adef4a0f88a2e，
    单步前后逐字节摘要一致；仅小头参数改变，合成损失2.06931471824646。
  * 4个mask -> 4个不同修复/接受计划 -> 真实StrictOnlineEnv的3种不同首动作。
    这是手工合成例的辨识证据，不是训练收益或真实日改善机会的证明。
  * KEEP恢复：合成固定接拒账的完整真实引擎轨迹/计划结果/账本/utility一致。
    强制挂起已加载真实模型的worker后，在2s槽内终止回收，原PublicView不变。
  * VALUE_KEEP使用默认零输出；COMMIT接线使用显式合成常数头bias=1，未伪称训练。
    故障double仅用于NaN、无合法动作、第二次插入失败等故障注入。
  * 未实现/未验收：真实SAA+SGBS接拒/续程适配、事件键RNG在SAA中的消费、训练历史
    未来生成与真实mask配对采集、完整接拒+投影+修复的总10s合同、整批资源监督、
    200步训练/校准、连续温度/浮点全认证、真实两日/OPP/L2/L3/最终测试。
    本次worker的10s仅是PublicView修复组件时限，不能称完整在线合同已合格。
  * 与设计的最小接口差异：stepwise_sample没有合法枚举回调；新模块复用原动作结构、
    apply_action、logit增删边映射和全局argmax语义，以局部循环加入统一保守容量谓词；
    没有猴子补丁、修改旧枚举器或训练新插入头。只支持声明的v2/Euclidean/N50公域。
  * 未改历史results、正式裁决、生产模块或MASKCO_code源文件；本轮结束即停止。
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import copy
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import configure_imports, MASKCO_ROOT
configure_imports(('experimental', 'models', 'training', 'simulation', 'coldchain', 'evaluation'))
import p2_pos01_reference as ref
from p2_pos01_mask_value import FrozenBackbone, MaskValueModel, state_digest
from p2_pos01_teacher import (EventRNG, PHASE, seal_capture, resume_keep, paired_label,
                              collect_real_labels)
from coldchain_contract import ColdChainContractV2
from coldchain_state import create_vehicle_state, dispatch_vehicle, CargoLotState
from strict_online_env import StrictOnlineEnv, Replanner
from recourse_snapshot import capture_recourse_snapshot, snapshot_state_hash
from coldchain_evaluator_a1 import evaluate_trace_a1
from mpre_replanner import retained_edge_ratio
from flax import nnx

CHECKPOINT = MASKCO_ROOT/'ckpts/cvrp100.ckpt'


def synthetic_case():
    contract = ColdChainContractV2()
    xy = [(0., 0.), (.01, .03), (.02, .01), (.04, .025),
          (.07, .01), (.08, .04), (.055, .075)]
    rows = [dict(oid=i, x=x, y=y, demand=0. if i == 0 else float((i % 3)+1),
                 tw_start=0., tw_end=10., service=0.01 if i else 0.,
                 temp_class=i % 3, initial_quality=1. if i % 3 == 0 else 9., reveal=0.)
            for i, (x, y) in enumerate(xy)]
    vehicles = (ref.Vehicle(0, mutable_suffix=(3, 1, 2)),
                ref.Vehicle(1, mutable_suffix=(6, 4, 5)))
    kwargs = dict(accepted=tuple(range(1, 7)), rejected=(), served=(), replan_ids=(0, 1),
                  clock=0., booking_horizon=0., capacity=50., tw_speed=1.,
                  energy_budget=1000., contract=contract)
    return rows, vehicles, kwargs


def synthetic_view():
    return ref.project_public(*synthetic_case()[:2], **synthetic_case()[2])


@pytest.fixture(scope='module')
def backbone():
    started = time.monotonic()
    model = FrozenBackbone(CHECKPOINT)
    assert time.monotonic()-started < 300, 'STOP: CPU initialization exceeded design slot'
    assert model.manifest['layers'] == (16, 6)
    assert model.manifest['encoder_dim'] == model.manifest['decoder_dim'] == 512
    return model


@pytest.fixture
def model(backbone):
    # Fixture bias lets a genuine head forward choose a mask before training.
    return MaskValueModel(backbone, synthetic_bias=1.)


def test_public_boundary_and_future_invariance(model):
    rows, vehicles, args = synthetic_case()
    class Hidden(dict):
        def __getitem__(self, key):
            if key != 'reveal':
                raise AssertionError('hidden field accessed: '+key)
            return super().__getitem__(key)
    v1 = ref.project_public(rows+[Hidden(reveal=1.)], vehicles, **args)
    v2 = ref.project_public(rows+[Hidden(reveal=100.), Hidden(reveal=200.)], vehicles, **args)
    assert v1 == v2
    base = ref.plans_from_view(v1)
    masks = ref.mask_pool(v1, base)
    f1, f2 = model.features(v1, base, masks), model.features(v2, base, masks)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(model.score_masks(v1, base, masks), model.score_masks(v2, base, masks))
    n1, l1 = ref.decide(v1, model, time.monotonic()+10)
    n2, l2 = ref.decide(v2, model, time.monotonic()+10)
    assert n1 == n2 and l1 == l2
    ready = replace(v1.vehicles[0], status='ready', current_node=1)
    before = replace(v1, booking_horizon=1.)
    assert ref.wait_or_return(before, ready) == ref.wait_or_return(replace(v2, booking_horizon=1.), ready) == ()
    assert ref.wait_or_return(replace(before, clock=1.), ready) == (0,)


def test_real_model_chain_and_candidate_identity(model, backbone):
    view = synthetic_view()
    original = ref.identity(view)
    base = ref.plans_from_view(view)
    assert ref.reference_check(view, base)['ok']
    masks = ref.mask_pool(view, base)
    assert 1 <= len(masks) <= 4
    record = []
    for m in masks:
        H, Z, ix, timestep = backbone.context(view, base, ref.remove_mask(base, m))
        assert H.shape == Z.shape == (1, 51, 512)
        assert np.isfinite(np.asarray(Z)).all() and 0 <= timestep <= 1
        candidate, fail = ref.reconstruct(view, base, m, backbone, time.monotonic()+10)
        assert fail is None and candidate is not None
        check = ref.reference_check(view, candidate, m)
        accepted = candidate if check['ok'] else base
        executed = execute_synthetic_plan(accepted, view)
        assert executed == ref.action_signature(view, accepted)
        record.append({'mask': m, 'repair': ref.plan_hash(candidate), 'reference_ok': check['ok'],
                       'accepted': ref.plan_hash(accepted), 'action': executed})
    new_view, log = ref.decide(view, model, time.monotonic()+10)
    assert log['status'] == 'COMMIT', log
    assert new_view.accepted == view.accepted and new_view.served == view.served
    assert ref.identity(view) == original
    assert log['qualification'] == 'REFERENCE_ONLY'
    assert len({r['repair'] for r in record}) >= 1  # diagnostic, NOT a gain assertion
    print('\nCANDIDATE_DIAGNOSTIC', {'rows': record, 'unique_repair': len({r['repair'] for r in record}),
          'unique_accepted': len({r['accepted'] for r in record}),
          'unique_action': len({r['action'] for r in record}), 'head': 'SYNTHETIC_CONSTANT_NOT_TRAINED'})
    print('MODEL_IDENTITY', backbone.manifest)


def test_one_cpu_gradient_updates_only_head(model, backbone):
    v = synthetic_view()
    base = ref.plans_from_view(v)
    f = model.features(v, base, ref.mask_pool(v, base))
    before, head_before = backbone.parameter_digest(), state_digest(nnx.state(model.head, nnx.Param))
    value = model.one_synthetic_step(f, np.arange(len(f), dtype=np.float32)+2.)
    assert np.isfinite(value)
    assert before == backbone.parameter_digest() == backbone.manifest['parameter_sha256']
    assert head_before != state_digest(nnx.state(model.head, nnx.Param))
    print('ONE_GRADIENT', {'loss': value, 'backbone_unchanged': True, 'head_changed': True})


def test_retained_edges_do_not_count_new_bridge(backbone):
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    b = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], np.float32)
    assert retained_edge_ratio(a, b) == 0.
    v = synthetic_view()
    base = ref.plans_from_view(v)
    _, z0, _, _ = backbone.context(v, base, base)
    _, z1, _, _ = backbone.context(v, base, ref.remove_mask(base, (1, 3)))
    assert not np.array_equal(z0, z1), 'masked adjacency did not reach actual decoder'


def test_commitment_ledger_and_prefix_are_immutable(backbone):
    v = synthetic_view()
    cc = dispatch_vehicle(create_vehicle_state(v.contract), v.contract)
    frozen = replace(v.vehicles[0], status='committed', committed_next=3,
                     committed_arrive=.04, committed_finish=.05,
                     mutable_suffix=(1, 2), coldchain_state=cc, coldchain_updated_time=0.)
    v = ref.validate_view(replace(v, vehicles=(frozen, v.vehicles[1])))
    base = ref.plans_from_view(v)
    assert ref.reference_check(v, base)['ok']
    assert all(3 not in m and 1 not in m and 2 not in m for m in ref.mask_pool(v, base))
    tampered = copy.deepcopy(base)
    tampered[0] = replace(tampered[0], anchor_node=1)
    assert not ref.reference_check(v, tampered, (4, 5))['ok']
    deleted = copy.deepcopy(base)
    deleted[1] = replace(deleted[1], suffix=(4, 5))
    assert not ref.reference_check(v, deleted, (4, 5))['ok']
    _, fail = ref.reconstruct(v, base, (1, 4), backbone, time.monotonic()+10)
    assert fail == 'ILLEGAL_MASK'
    same = ref.commit_view(v, base, ())
    assert same.vehicles[0] == frozen and same.accepted == v.accepted


def test_capacity_domain_at_all_four_stages(backbone):
    v = synthetic_view()
    orders = list(v.orders)
    limit = v.capacity-ref.CAPACITY_MARGIN
    orders[3] = replace(orders[3], demand=25.)
    orders[1] = replace(orders[1], demand=limit-25.)
    orders[2] = replace(orders[2], demand=1.)
    # One customer moved to the other vehicle so the boundary route has exactly two.
    veh = (replace(v.vehicles[0], mutable_suffix=(3, 1)),
           replace(v.vehicles[1], mutable_suffix=(6, 4, 5, 2)))
    at = replace(v, orders=tuple(orders), vehicles=veh)
    p = ref.plans_from_view(at)
    assert ref.capacity_ok(at, limit) and not ref.capacity_ok(at, limit+5e-5)
    assert ref.route_check(at, p[0]) and ref.reference_check(at, p)['ok']
    partial = ref.remove_mask(p, (1, 3))
    assert ref.legal_actions(at, partial, (1, 3))
    above_orders = list(orders)
    above_orders[1] = replace(orders[1], demand=orders[1].demand+5e-5)
    above = replace(at, orders=tuple(above_orders))
    assert not ref.route_check(above, p[0])
    assert ref.mask_pool(above, p) == ()
    assert not ref.reference_check(above, p)['ok']
    trial_actions = ref.legal_actions(above, ref.remove_mask(p, (1, 3)), (1, 3))
    for _, action, _ in trial_actions:
        trial = ref.apply_action(ref.remove_mask(p, (1, 3)), action,
                                 allowed_vehicle_ids=ref.mutable_ids(above))
        assert all(ref.route_check(above, x) for x in trial.values())


def test_ninth_cargo_lot_affects_public_features(model):
    v = synthetic_view()
    additions = tuple(ref.Order(i, .01, .03, 1., 0., 10., 0., 0, 1., 0.) for i in range(7, 15))
    ids = (7, 8, 9, 10, 11, 12, 13, 14, 1)
    orders = tuple(replace(o, demand=1., temp_class=0, initial_quality=1.) if o.oid == 1 else o for o in v.orders)+additions
    lots = tuple(CargoLotState(i, 1., 0, 0., 1., 1.) for i in ids)
    cc = replace(dispatch_vehicle(create_vehicle_state(v.contract), v.contract),
                 zone_load=(9., 0., 0.), cargo_manifest=lots)
    vehicle = replace(v.vehicles[0], status='ready', current_node=1, current_load=9.,
                      served_route=ids, mutable_suffix=(3, 2), coldchain_state=cc,
                      coldchain_updated_time=0.)
    view = ref.validate_view(replace(v, orders=orders, vehicles=(vehicle, v.vehicles[1]),
                     accepted=tuple(sorted(set(v.accepted)|set(ids))), served=tuple(sorted(ids))))
    base, masks = ref.plans_from_view(view), ((2, 3),)
    before = model.features(view, base, masks)
    changed = replace(cc, cargo_manifest=lots[:8]+(replace(lots[8], quality_remaining=.2),))
    after = model.features(replace(view, vehicles=(replace(vehicle, coldchain_state=changed), v.vehicles[1])), base, masks)
    assert not np.array_equal(before, after)
    kept = ref.commit_view(view, base, ())
    assert kept.vehicles[0].served_route == ids and kept.served == view.served


def test_failures_never_write_partial_state(model, backbone, monkeypatch):
    v = synthetic_view()
    stamp = ref.identity(v)
    class NaNHead:
        def score_masks(self, _v, _b, masks):
            return np.full(len(masks), np.nan)
    assert ref.decide(v, NaNHead(), time.monotonic()+10)[0] == v
    assert ref.decide(v, NaNHead(), time.monotonic()+10)[1]['status'] == 'NAN'
    class BrokenBackbone:
        def insertion_scores(self, *_):
            raise ValueError('injected reconstruction failure')
    class BrokenModel:
        backbone = BrokenBackbone()
        def score_masks(self, _v, _b, masks):
            return np.ones(len(masks))
    assert ref.decide(v, BrokenModel(), time.monotonic()+10)[1]['status'] == 'RECONSTRUCT_FAIL'
    class FailAfterOneInsertion:
        calls = 0
        def insertion_scores(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise ValueError('injected after first local insertion')
            return backbone.insertion_scores(*args)
    partial_failure = BrokenModel()
    partial_failure.backbone = FailAfterOneInsertion()
    new, log = ref.decide(v, partial_failure, time.monotonic()+10)
    assert partial_failure.backbone.calls == 2 and new == v and log['status'] == 'RECONSTRUCT_FAIL'
    assert ref.decide(v, model, time.monotonic()-1)[0] == v
    with monkeypatch.context() as patch:
        patch.setattr(ref, 'legal_actions', lambda *_: [])
        new, log = ref.decide(v, model, time.monotonic()+10)
        assert new == v and log['status'] == 'NO_LEGAL_ACTION'
    assert ref.identity(v) == stamp


def test_high_value_cannot_override_energy_or_salability(model):
    v = synthetic_view()
    base = ref.plans_from_view(v)
    low_budget = replace(v, energy_budget=.01)
    assert not ref.reference_check(low_budget, base)['ok']
    with pytest.raises(ValueError, match='KEEP_INVALID'):
        ref.decide(low_budget, model, time.monotonic()+10)
    # Fish below the actual v2 hard salability threshold cannot be rescued by q.
    low_q = tuple(replace(o, initial_quality=4.) if o.temp_class in (1, 2) else o for o in v.orders)
    assert not ref.reference_check(replace(v, orders=low_q), base)['ok']


class SyntheticFixedLedger(Replanner):
    """Handwritten synthetic admission fixture, explicitly NOT the SAA baseline."""
    def __init__(self, plan=None):
        # All six orders are already public at t=0 and admitted in this fixture.
        self._accepted, self._rejected = set(range(1, 7)), set()
        self._plan = {0: [3, 1, 2], 1: [6, 4, 5]} if plan is None else copy.deepcopy(plan)
    def on_reveal(self, env, inst_idx, clock, vehicles, served, visible):
        self._accepted.update(int(c) for c in visible)
        for v in vehicles:
            if v.status == 'idle':
                v.mutable_suffix = list(self._plan[v.vehicle_id])
    def plan(self, env, inst_idx, clock, vehicles, served, visible, replan_ids=None):
        for v in vehicles:
            if v.status in ('idle', 'ready') and (replan_ids is None or v.vehicle_id in replan_ids):
                route = [c for c in self._plan[v.vehicle_id] if not served[c]]
                v.mutable_suffix = route if route else [0]
    def export_state(self):
        return {'accepted': sorted(self._accepted), 'rejected': sorted(self._rejected),
                'plan': copy.deepcopy(self._plan)}
    def restore_state(self, state):
        self._accepted, self._rejected = set(state['accepted']), set(state['rejected'])
        self._plan = copy.deepcopy(state['plan'])


def execute_synthetic_plan(plans, view):
    """Observe actual StrictOnlineEnv services; no future or new admission here."""
    rows, _, _ = synthetic_case()
    data = {'coords': np.array([[[r['x'], r['y']] for r in rows]], np.float32)}
    for field, key in (('demands', 'demand'), ('tw_start', 'tw_start'), ('tw_end', 'tw_end'),
                       ('service_time', 'service'), ('temp_class', 'temp_class'),
                       ('initial_quality', 'initial_quality'), ('reveal_time', 'reveal')):
        data[field] = np.array([[r[key] for r in rows]], np.float32)
    policy = SyntheticFixedLedger({vid: list(p.suffix) for vid, p in plans.items()})
    engine = StrictOnlineEnv(data, view.capacity, num_vehicles=2, replanner=policy,
                             coldchain_contract=view.contract, booking_horizon=view.booking_horizon)
    traces, _ = engine.run(0)
    evaluation = evaluate_trace_a1(traces, {k: a[0] for k, a in data.items()}, view.contract,
        set(view.accepted), set(view.rejected), view.energy_budget, {0: 1., 1: 1., 2: 1.})
    assert evaluation['hard_feasible'] and np.isfinite(evaluation['utility'])
    return tuple((t.vehicle_id, ('SERVE', t.services[0].node)) for t in traces)


def test_real_engine_keep_noop_exact_restore():
    rows, _, args = synthetic_case()
    fields = {'coords': [[[r['x'], r['y']] for r in rows]],
              **{target: [[r[source] for r in rows]] for target, source in
                 (('demands', 'demand'), ('tw_start', 'tw_start'), ('tw_end', 'tw_end'),
                  ('service_time', 'service'), ('temp_class', 'temp_class'),
                  ('initial_quality', 'initial_quality'), ('reveal_time', 'reveal'))}}
    dataset = {k: np.array(val, dtype=np.float32) for k, val in fields.items()}
    policy = SyntheticFixedLedger()
    env = StrictOnlineEnv(dataset, 50., num_vehicles=2, replanner=policy,
                         coldchain_contract=args['contract'], booking_horizon=0.)
    captures = []
    def hook(env, inst, clock, event, reveal_idx, vehicles, traces, served, universe):
        if captures:
            return
        snap = capture_recourse_snapshot(env, inst, clock, event, reveal_idx, vehicles,
                                         traces, served, universe, preserve_scalar_types=True)
        view = ref.project_public(rows, vehicles, **{**args, 'clock': clock,
                    'accepted': policy._accepted, 'rejected': policy._rejected,
                    'served': [int(c) for c in np.flatnonzero(served) if c != 0]})
        captures.append(seal_capture(snap, phase=PHASE, public_view=view,
                                    continuation_identity='SYNTHETIC_FIXED_NOT_SAA', source_dataset=dataset))
    env.snapshot_hook = hook
    original, served = env.run(0)
    assert captures
    restored, rs, ledger = resume_keep(captures[0], dataset, SyntheticFixedLedger,
        args['contract'], capacity=50., tw_speed=1., booking_horizon=0.,
        continuation_identity='SYNTHETIC_FIXED_NOT_SAA')
    assert ledger == policy.export_state()
    np.testing.assert_array_equal(served, rs)
    assert snapshot_state_hash({'traces': [asdict(t) for t in original]}) == snapshot_state_hash(
        {'traces': [asdict(t) for t in restored]})
    a = evaluate_trace_a1(original, {k: val[0] for k, val in dataset.items()}, args['contract'],
                          policy._accepted, policy._rejected, 1000., {0: 1., 1: 1., 2: 1.})
    b = evaluate_trace_a1(restored, {k: val[0] for k, val in dataset.items()}, args['contract'],
                          set(ledger['accepted']), set(ledger['rejected']), 1000., {0: 1., 1: 1., 2: 1.})
    assert a == b and a['hard_feasible']
    with pytest.raises(ValueError, match='public protocol/dataset changed'):
        resume_keep(captures[0], dataset, SyntheticFixedLedger, args['contract'],
                    capacity=50., tw_speed=1., booking_horizon=.1,
                    continuation_identity='SYNTHETIC_FIXED_NOT_SAA')
    print('ENGINE_KEEP_PARITY', {'exact_trace': True, 'ledger': True, 'utility': a['utility'],
                                'controller': 'SYNTHETIC_FIXED_NOT_SAA'})


def test_keyed_randomness_and_failure_preserving_labels():
    a, b = EventRNG('synthetic-F', 11), EventRNG('synthetic-F', 11)
    b.normal(('unrelated-event', 0), 19)
    np.testing.assert_array_equal(a.normal(('reveal', .5, 'external-3'), 4),
                                  b.normal(('reveal', .5, 'external-3'), 4))
    keep = dict(status='OK', hard_feasible=True, utility=10., scenario_identity='F', continuation_identity='B')
    candidate = {**keep, 'utility': 12.}
    assert paired_label(candidate, keep)['delta_utility'] == 2.
    missing = paired_label({**candidate, 'status': 'ROLLOUT_TIMEOUT', 'utility': None}, keep)
    assert missing['delta_utility'] is None and missing['training_weight'] == 0.
    with pytest.raises(ValueError, match='unpaired'):
        paired_label({**candidate, 'scenario_identity': 'OTHER'}, keep)
    with pytest.raises(NotImplementedError, match='NOT AUTHORISED'):
        collect_real_labels()


def test_worker_real_cpu_chain_and_forced_timeout():
    v = synthetic_view()
    stamp = ref.identity(v)
    with ref.ReferenceSession(CHECKPOINT, synthetic_head_bias=1.) as session:
        assert session.manifest['platform'] == 'cpu'
        # The initialization/shape warmup has its own declared 300s slot.
        new, record = session.decide(v, seconds=300)
        assert record['status'] == 'COMMIT', record
        same, steady = session.decide(v, seconds=10)
        assert steady['status'] == 'COMMIT' and same == new
        t0 = time.monotonic()
        unchanged, fault = session.decide(v, seconds=2, synthetic_fault='hang')
        assert time.monotonic()-t0 < 2
        assert unchanged == v and not fault['worker_alive'] and not session.process.is_alive()
        assert ref.identity(v) == stamp
        print('WORKER', {'steady': steady['status'], 'forced_timeout': fault, 'tree_reclaimed': True})
