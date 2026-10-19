"""P2-POS-01A/B synthetic CPU acceptance. No datasets/results/checkpoints are written.

P2-B command: pytest -p no:cacheprovider -k 'not one_cpu_gradient' <this file>.
The historical A-only gradient test is retained but MUST NOT run in the B scope.
The following A record is historical; the B acceptance record follows it.

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

P2-POS-01B 实施验收记录（2026-10-19，起始HEAD 3789e4956d44740b9357722f9a12a09938149558）：
  * 17项非训练合成测试通过，1项历史梯度测试明确排除，最终整组46.76s。
    本轮优化器更新=0、真实数据日=0、真实训练标签采集=0、GPU/正式实验=0。
  * 真实ReplaySafeSaaReplanner的_saa_decide、条件历史采样、ls影子搜索与StrictOnlineEnv
    实际执行；K=4、marginal能源口径、同事件共享6s接拒截止，配置/历史/源码均绑定身份。
    公共51节点桶的未发布行仅零需求占位，未来值不传入当前SAA或MaskCO。
  * POST_ADMISSION_PRE_REPAIR在SAA本事件完成接拒和形成P0后、引擎commit前捕获。
    mask候选只改合法mutable后缀，并写入恢复后SAA的真实_plan；首个恢复计划hash与
    候选相符，接拒计数不重复，随后真实未来揭示事件继续执行SAA决策。
  * 一个手工源状态、4个合法mask、2个HIST外生未来，加KEEP共10条合成续程均OK。
    4个配对均值约为[-0.18045335,-0.31647326,-0.21642810,-0.31647326]，负值原样保留。
    每个未来上的mask产生3种不同的实际首批commit向量；KEEP是第4种。
    这些数只验接线和标签含义，不是正向收益、可学习性或统计显著性结论。
  * KEEP/直接运行完整轨迹、账本和utility一致；各mask重复终局身份一致；真实SAA的
    共用揭示键场景相同，额外消耗无关随机数或反转动作顺序不会移动配对外生未来。
    EventRNG同时用于HIST外生未来生成和SAA续程采样，不以真实未来hash作决策seed。
  * 原权重字节身份与A一致，完整backbone参数摘要仍为上列7b43ece3...，无训练。
    A的4修复/3实际首动作测试已加>=2防退化断言，不再以unique>=1充当通过。
  * 非法mask、独立参考拒绝、EXCEPTION、ROLLOUT_TIMEOUT均保留utility=null；
    故障分支不污染源快照或下一分支。身份损坏停止队列，剩余预定槽保持NOT_ATTEMPTED；
    标签有效性要求KEEP与候选在全部K上均有效，不删除失败或填造有限负奖励。
  * 资格：LABEL_CHAIN_SYNTHETIC_PASS / REFERENCE_ONLY。已达到标签链的合成前置，
    不是两日执行资格或授权。真实源日与HIST的TRAIN隔离/逐内容清单、整队列资源封顶和
    监督验收仍需在单独授权的两日烟测范围收口；真实状态覆盖率/标签有效率尚未测量。
    完整接拒+投影+模型+检查10s在线合同、同预算distance+SGBS比较、连续温度及浮点
    全认证、价值头拟合/校准、正式L2/L3与最终测试均未建立，不能据本件升级。
    真实标签采集入口仍受REAL_COLLECTION_NOT_AUTHORISED门控；没有新CLI或自动续跑。
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
    assert len({r['repair'] for r in record}) >= 2, 'degenerate synthetic repair pool'
    assert len({r['action'] for r in record}) >= 2, 'no distinguishable actual first actions'
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
    with pytest.raises(PermissionError, match='NOT_AUTHORISED'):
        collect_real_labels()


# P2-POS-01B: actual ReplaySafeSaaReplanner + CondHistoricalSampler + StrictOnlineEnv.
# These fixtures use handcrafted arrays only; no NPZ/results/real-day assets read.
import p2_pos01_teacher as teacher
from scenario_saa import make_c0_contract_v2, ScenarioOrder


def saa_synthetic_dataset():
    rows, _, _ = synthetic_case()
    data = {k: np.zeros((1, 51), np.float32) for k in
            ('demands', 'tw_start', 'tw_end', 'service_time', 'temp_class',
             'initial_quality', 'reveal_time')}
    data['coords'] = np.zeros((1, 51, 2), np.float32)
    data['tw_end'][:] = 5.
    data['initial_quality'][:] = 1.
    data['reveal_time'][:] = 1.5  # zero-demand PUBLIC padding, not hidden cardinality
    for r in rows:
        c = r['oid']
        data['coords'][0, c] = [r['x'], r['y']]
        for name, value in (('demands', r['demand']), ('service_time', r['service']),
                             ('temp_class', r['temp_class']), ('initial_quality', r['initial_quality']),
                             ('reveal_time', .05 if c else 0.)):
            data[name][0, c] = value
    for c, xy, reveal in ((7, (.12, .09), .23), (8, (.02, .11), .31)):
        data['coords'][0, c] = xy
        data['demands'][0, c] = 2.
        data['service_time'][0, c] = .01
        data['temp_class'][0, c] = 1
        data['initial_quality'][0, c] = 9.
        data['reveal_time'][0, c] = reveal
    history = []
    for shift in (0., .015):
        day = []
        for c in range(1, 9):
            day.append(ScenarioOrder(c, float(data['reveal_time'][0, c]),
                float(data['coords'][0, c, 0])+(shift if c > 6 else 0.),
                float(data['coords'][0, c, 1]), float(data['demands'][0, c]),
                int(data['temp_class'][0, c]), 0., 5., float(data['service_time'][0, c])))
        history.append(tuple(day))
    spec = teacher.SaaSpec(make_c0_contract_v2(), tuple(history), booking_horizon=.5,
                           budget=1000., stream_id='HANDCRAFTED_SYNTHETIC_ONLY')
    return data, spec


def run_actual_saa(data, spec, capture=False):
    controller = teacher.PublicSaaReplanner(spec)
    env = teacher.SaaPhaseEnv(data, spec.capacity, spec.tw_speed, num_vehicles=2,
        replanner=controller, coldchain_contract=spec.contract, booking_horizon=spec.booking_horizon)
    captures = []
    if capture:
        def hook(engine, view, context):
            if not captures and ref.mask_pool(view, ref.plans_from_view(view)):
                captures.append(teacher.capture_saa_phase(engine, view, context, spec))
        env.p2_capture = hook
    traces, served = env.run(0)
    ev = evaluate_trace_a1(traces, {k: a[0] for k, a in data.items()}, spec.contract,
        controller._accepted, controller._rejected, spec.budget, dict(enumerate(spec.reject_penalty)))
    return captures, traces, served, controller, ev


@pytest.fixture(scope='module')
def saa_bundle(backbone):
    data, spec = saa_synthetic_dataset()
    captures, _, _, controller, _ = run_actual_saa(data, spec, True)
    assert captures, 'actual SAA did not expose a legal post-admission state'
    capture = captures[0]
    assert isinstance(controller, teacher.ReplaySafeSaaReplanner)
    teacher.validate_saa_capture(capture, spec)
    actions = teacher.prepare_mask_actions(capture, spec, backbone)
    assert actions and any(a['status'] == 'OK' for a in actions), actions
    tapes = teacher.make_future_tapes(capture, actions, spec, K=2)
    report = teacher.collect_real_labels(capture, actions, tapes, spec, synthetic=True)
    print('\nSAA_LABEL_CHAIN', {'actions': [(a['mask'], a['status']) for a in actions],
        'n_planned': report['n_planned'], 'statuses': [r['status'] for r in report['rows']],
        'labels': report['labels'], 'entry_commits': [r.get('actual_entry_commits') for r in report['rows']]})
    return capture, actions, tapes, spec, report


def test_saa_keep_matches_direct_and_current_admission_not_repeated(saa_bundle):
    capture, _, tapes, spec, report = saa_bundle
    _, traces, served, policy, ev = run_actual_saa(tapes[0]['dataset'], spec)
    keep = report['rows'][0]
    assert keep['status'] == 'OK', keep
    assert keep['trace_identity'] == snapshot_state_hash({'traces': [asdict(t) for t in traces]})
    assert keep['evaluation'] == ev
    assert keep['ledger'] == policy.export_state()
    assert keep['first_entry']['admission_count'] == capture['snapshot']['replanner_state']['p2_saa']['admission_count']


def test_saa_real_mask_enters_controller_and_repeats_exactly(saa_bundle, backbone):
    capture, actions, tapes, spec, report = saa_bundle
    teacher.validate_saa_capture(capture, spec)
    assert len({ref.plan_hash(a['plan']) for a in actions if a['status'] == 'OK'}) >= 2
    for action in actions:
        if action['status'] != 'OK':
            continue
        row = next(r for r in report['rows'] if r['scenario_identity'] == tapes[0]['scenario_identity']
                   and r['action_id'] == action['action_id'])
        assert row['status'] == 'OK', row
        assert row['first_entry']['plan_hash'] == ref.plan_hash(action['plan'])
        assert row['first_entry']['accepted'] == list(capture['public_view'].accepted)
        repeat = teacher.run_branch(capture, action, tapes[0], spec)
        assert row['terminal_identity'] == repeat['terminal_identity']
        assert row['utility'] == repeat['utility']
    assert all(x['valid'] and np.isfinite(x['delta_utility']) for x in report['labels']), report['labels']
    assert backbone.parameter_digest() == backbone.manifest['parameter_sha256']


def test_saa_hidden_future_cannot_change_current_state_or_action(backbone):
    source, spec = saa_synthetic_dataset()
    captures1, *_ = run_actual_saa(source, spec, True)
    altered = copy.deepcopy(source)
    altered['coords'][0, 7:] = 100.
    altered['demands'][0, 7:9] = 19.
    altered['temp_class'][0, 7:9] = 2
    altered['reveal_time'][0, 7:9] = [.27, .45]
    captures2, *_ = run_actual_saa(altered, spec, True)
    one, two = captures1[0], captures2[0]
    assert one['public_identity'] == two['public_identity']
    assert one['snapshot']['replanner_state'] == two['snapshot']['replanner_state']
    a1 = teacher.prepare_mask_actions(one, spec, backbone)
    a2 = teacher.prepare_mask_actions(two, spec, backbone)
    assert [a['action_identity'] for a in a1] == [a['action_identity'] for a in a2]
    m = MaskValueModel(backbone)  # no optimization, default untrained KEEP head
    v1, v2 = one['public_view'], two['public_view']
    b1, b2 = ref.plans_from_view(v1), ref.plans_from_view(v2)
    masks = ref.mask_pool(v1, b1)
    np.testing.assert_array_equal(m.score_masks(v1, b1, masks), m.score_masks(v2, b2, masks))
    np.testing.assert_array_equal(m.features(v1, b1, masks), m.features(v2, b2, masks))
    np.testing.assert_array_equal(one['snapshot']['mutable_suffix'], two['snapshot']['mutable_suffix'])


def test_saa_external_rng_pairing_and_branch_order_independence(saa_bundle, monkeypatch):
    capture, actions, tapes, spec, report = saa_bundle
    again = teacher.make_future_tapes(capture, actions, spec, K=2)
    assert [t['scenario_identity'] for t in tapes] == [t['scenario_identity'] for t in again]
    reverse = teacher.make_future_tapes(capture, tuple(reversed(actions)), spec, K=2)
    assert [t['scenario_identity'] for t in tapes] == [t['scenario_identity'] for t in reverse]
    # Actual SAA draws at common external reveal keys agree even when intervening
    # service events, plans, admission decisions and branch order differ.
    rows = [r for r in report['rows'] if r['status'] == 'OK' and
            r['scenario_identity'] == tapes[0]['scenario_identity']]
    audits = [{repr(d['key']): d['scenarios'] for d in r['ledger']['p2_saa']['draw_audit']} for r in rows]
    assert len(audits[0]) >= 2, 'no actual future SAA event consumed keyed RNG'
    for audit in audits[1:]:
        common = set(audit) & set(audits[0])
        assert len(common) >= 2 and all(audit[k] == audits[0][k] for k in common)
    row = teacher.run_branch(capture, actions[-1], tapes[0], spec)
    repeated_keep = teacher.run_branch(capture, None, tapes[0], spec)
    assert repeated_keep['terminal_identity'] == report['rows'][0]['terminal_identity']
    assert row['status'] == 'OK'
    generator = teacher.EventRNG.generator
    def extra_consumption(self, key):
        generator(self, ('UNRELATED_SYNTHETIC_DRAW', key)).normal(size=17)
        return generator(self, key)
    monkeypatch.setattr(teacher.EventRNG, 'generator', extra_consumption)
    extra = teacher._run_branch(capture, actions[-1], tapes[0], spec)
    assert extra['terminal_identity'] == row['terminal_identity']


def test_saa_broken_ledger_prefix_and_cutoff_rejected(saa_bundle):
    capture, actions, tapes, spec, _ = saa_bundle
    for kind in ('ledger', 'prefix'):
        bad = copy.deepcopy(capture)
        snap = bad['snapshot']
        if kind == 'ledger':
            snap['replanner_state']['accepted'].pop()
        else:
            snap['vehicle_node'][0] = 1
            snap['runtime_scalars_v18']['vehicles'][0]['current_node'] = ('python', 1)
        snap['state_hash'] = snapshot_state_hash(snap)  # semantic test, not just checksum failure
        bad['capture_identity'] = ref.identity([teacher.PHASE, bad['public_identity'], snap['state_hash'],
            spec.signature, bad['protocol'], bad['dataset_identity']])
        bad['saa_identity'] = teacher._saa_capture_identity(bad)
        with pytest.raises(ValueError):
            teacher.run_branch(bad, actions[0], tapes[0], spec)
    with pytest.raises(ValueError):
        teacher.run_branch(capture, actions[0], tapes[0], replace(spec, booking_horizon=.6))
    # Test the ACTUAL SAA route predicate on both sides of the same capacity rule.
    policy = teacher.PublicSaaReplanner(spec)
    view = capture['public_view']
    e = ref.public_env(view)
    e.demands[0, 1] = 50.-ref.CAPACITY_MARGIN
    assert policy._route_feasible(e, 0, 0, view.clock, 0., [1])
    e.demands[0, 1] += 5e-5
    assert not policy._route_feasible(e, 0, 0, view.clock, 0., [1])


def test_saa_faults_invalid_labels_and_snapshot_survive(saa_bundle, backbone):
    capture, actions, tapes, spec, report = saa_bundle
    stamp = snapshot_state_hash(capture['snapshot'])
    exception = teacher.run_branch(capture, actions[0], tapes[0], spec,
                                   synthetic_fault='raise_after_install')
    timeout = teacher.run_branch(capture, actions[0], tapes[0], spec, seconds=5,
                                 synthetic_fault='hang_after_install')
    assert exception['status'] == 'EXCEPTION' and exception['utility'] is None
    assert timeout['status'] == 'ROLLOUT_TIMEOUT' and timeout['worker_reclaimed']
    assert not paired_label(exception, report['rows'][0])['valid']
    assert not paired_label(timeout, report['rows'][0])['valid']
    invalid = teacher.prepare_mask_actions(capture, spec, backbone, masks=((0, 1),))
    itapes = teacher.make_future_tapes(capture, invalid, spec, K=1)
    invalid_report = teacher.collect_real_labels(capture, invalid, itapes, spec, synthetic=True)
    assert invalid_report['n_planned'] == 2 and len(invalid_report['rows']) == 2
    assert invalid_report['labels'][0]['delta_utility'] is None
    assert invalid_report['rows'][1]['status'] == 'ILLEGAL_MASK'
    failed_report = teacher.collect_real_labels(capture, actions[:1], tapes[:1], spec,
        synthetic=True, synthetic_faults={(0, actions[0]['action_id']): 'raise_after_install'})
    assert len(failed_report['rows']) == 2 and failed_report['rows'][1]['status'] == 'EXCEPTION'
    assert failed_report['labels'][0]['delta_utility'] is None
    damaged = copy.deepcopy(tapes[0])
    damaged['dataset']['coords'][0, 7, 0] += .1
    stopped = teacher.collect_real_labels(capture, actions, (damaged,), spec, synthetic=True)
    assert stopped['status'] == 'BLOCKED' and len(stopped['rows']) == len(actions)+1
    assert stopped['rows'][0]['status'] == 'BLOCKED'
    assert all(r['status'] == 'NOT_ATTEMPTED' for r in stopped['rows'][1:])
    assert all(x['delta_utility'] is None for x in stopped['labels'])
    assert snapshot_state_hash(capture['snapshot']) == stamp
    after = teacher.run_branch(capture, actions[0], tapes[0], spec)
    original = next(r for r in report['rows'] if r['action_id'] == actions[0]['action_id'])
    assert after['terminal_identity'] == original['terminal_identity']


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
