"""P2-POS-01B: isolated actual-SAA counterfactual label chain, REFERENCE_ONLY.

Implements event-keyed randomness, immutable capture identity, strict-engine
KEEP/no-op resume with explicit public booking_horizon, and failure-preserving
paired-label arithmetic. Synthetic integration is executable; real-day collection
and training remain disabled. Production files and frozen results are untouched.
"""
from __future__ import annotations

import copy
import hashlib
import multiprocessing as mp
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from functools import lru_cache

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import configure_imports
configure_imports(('experimental', 'simulation', 'coldchain', 'evaluation'))
from recourse_snapshot import (snapshot_state_hash, validate_snapshot_against_env,
                              capture_recourse_snapshot, restore_recourse_snapshot)
from strict_online_env import StrictOnlineEnv
import p2_pos01_reference as ref
from p2_pos01_reference import identity, finite
from i3_counterfactual_teacher import ReplaySafeSaaReplanner
from scenario_saa import CondHistoricalSampler, ScenarioOrder
from coldchain_evaluator_a1 import evaluate_trace_a1

PHASE = 'POST_ADMISSION_PRE_REPAIR'


@lru_cache(maxsize=1)
def source_signature():
    """Relevant source bytes at process initialization; not a platform certificate."""
    root = Path(__file__).resolve().parents[1]
    paths = ('training/p2_pos01_teacher.py', 'experimental/p2_pos01_reference.py',
        'models/p2_pos01_mask_value.py', 'evaluation/scenario_saa.py',
        'training/i3_counterfactual_teacher.py', 'simulation/strict_online_env.py',
        'simulation/recourse_snapshot.py', 'simulation/action_contract.py',
        'coldchain/coldchain_state.py', 'coldchain/coldchain_contract.py',
        'evaluation/coldchain_evaluator_a1.py', 'evaluation/run_exp_energy_c0.py')
    return identity({p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for key in sorted(dataset):
        array = np.ascontiguousarray(dataset[key])
        digest.update(str((key, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class EventRNG:
    """Independent draw for each external event key, never a branch call counter.

    Caller must use external scenario/event identity; no action or branch ID.
    A different key may be drawn first without shifting subsequent event draws.
    """
    def __init__(self, scenario_identity, seed):
        self.scenario_identity, self.seed = str(scenario_identity), int(seed)

    def normal(self, external_event_key, size):
        return self.generator(external_event_key).normal(size=size)

    def generator(self, external_event_key):
        """Seed namespace is a predeclared stream ID, never a true-future hash."""
        key = identity([self.scenario_identity, self.seed, external_event_key])
        words = np.frombuffer(bytes.fromhex(key), dtype='<u4').copy()
        return np.random.default_rng(np.random.SeedSequence(words))


def seal_capture(snapshot, *, phase, public_view, continuation_identity, source_dataset):
    if phase != PHASE or not continuation_identity:
        raise ValueError('capture phase/continuation identity required')
    snap = copy.deepcopy(snapshot)
    if snap.get('state_hash') != snapshot_state_hash(snap):
        raise ValueError('unsealed engine snapshot')
    ledger = snap.get('replanner_state')
    if not isinstance(ledger, dict) or 'accepted' not in ledger or 'rejected' not in ledger:
        raise ValueError('complete acceptance ledger required')
    if set(ledger['accepted']) != set(public_view.accepted) or set(ledger['rejected']) != set(public_view.rejected):
        raise ValueError('public/private ledger mismatch')
    protocol = {'booking_horizon': public_view.booking_horizon, 'capacity': public_view.capacity,
                'tw_speed': public_view.tw_speed, 'contract_hash': public_view.contract.contract_hash}
    source = dataset_digest(source_dataset)
    return {'phase': phase, 'public_identity': identity(public_view), 'snapshot': snap,
            'protocol': protocol, 'dataset_identity': source,
            'continuation_identity': continuation_identity,
            'scope': 'KEEP_REPLAY_ONLY_NOT_MASK_SAA',
            'capture_identity': identity([phase, identity(public_view), snap['state_hash'],
                                          continuation_identity, protocol, source])}


def resume_keep(capture, dataset, policy_factory, contract, *, capacity, tw_speed,
                booking_horizon, continuation_identity):
    """Real existing StrictOnlineEnv replay; private teacher side only.

    No legacy resume factory and no old future-reading force_suffix function.
    This is KEEP plumbing, not the qualified mask+SAA continuation required by B.
    """
    if capture['phase'] != PHASE or capture['continuation_identity'] != continuation_identity:
        raise ValueError('continuation/phase changed')
    snap = copy.deepcopy(capture['snapshot'])
    if snap['state_hash'] != snapshot_state_hash(snap):
        raise ValueError('engine snapshot identity changed')
    protocol = {'booking_horizon': booking_horizon, 'capacity': capacity, 'tw_speed': tw_speed,
                'contract_hash': contract.contract_hash}
    if protocol != capture['protocol'] or dataset_digest(dataset) != capture['dataset_identity']:
        raise ValueError('public protocol/dataset changed during recovery')
    expected = identity([capture['phase'], capture['public_identity'], snap['state_hash'],
                         continuation_identity, protocol, capture['dataset_identity']])
    if expected != capture['capture_identity'] or not np.isfinite(booking_horizon):
        raise ValueError('capture/public booking identity changed')
    before = dataset_digest(dataset)
    policy = policy_factory()
    env = StrictOnlineEnv(copy.deepcopy(dataset), capacity=capacity, tw_speed=tw_speed,
        num_vehicles=int(snap['num_vehicles']), replanner=policy,
        coldchain_contract=contract, booking_horizon=booking_horizon)
    validate_snapshot_against_env(env, snap)
    traces, served = env.run_resumed(snap)
    after = dataset_digest(dataset)
    if after != before or snapshot_state_hash(capture['snapshot']) != snap['state_hash']:
        raise RuntimeError('private replay mutated source')
    return traces, served, policy.export_state()


def paired_label(candidate, keep):
    """No finite reward invented for failures. Caller retains all input rows."""
    valid = all(x.get('status') == 'OK' and x.get('hard_feasible') is True and
                x.get('utility') is not None and finite(x['utility']) for x in (candidate, keep))
    if (candidate.get('scenario_identity') != keep.get('scenario_identity') or
            candidate.get('continuation_identity') != keep.get('continuation_identity') or
            not candidate.get('scenario_identity') or not candidate.get('continuation_identity')):
        raise ValueError('unpaired future/continuation')
    return {'delta_utility': candidate['utility']-keep['utility'] if valid else None,
            'valid': valid, 'candidate_status': candidate.get('status'),
            'keep_status': keep.get('status'), 'training_weight': 1.0 if valid else 0.0}


@dataclass(frozen=True)
class SaaSpec:
    contract: object
    history: tuple
    booking_horizon: float
    budget: float
    capacity: float = 50.
    tw_speed: float = 1.5
    K: int = 4
    n_neighbors: int = 10
    time_limit: float = 6.
    cooling_share: float = 1.
    reject_penalty: tuple = (5., 10., 15.)
    stream_id: str = 'p2-pos01b-registered-stream'
    stream_seed: int = 101
    shadow_mode: str = 'ls'
    energy_pricing: str = 'marginal'
    standby_orders: float = 8.

    @property
    def signature(self):
        return snapshot_state_hash({'spec': asdict(self), 'source': source_signature()})


class PublicSaaReplanner(ReplaySafeSaaReplanner):
    """Actual ReplaySafe SAA decisions, with a public-only boundary and keyed draws.

    No alternate admission scorer or toy controller. Parent _saa_decide, scenario
    simulation/local search and certified insertion execute unchanged. The extra
    independent gate enforces the new reference hard domain on real admissions.
    """
    def __init__(self, spec):
        self.spec = spec
        spec.contract.validate()
        if (not spec.history or spec.K != 4 or spec.time_limit <= 0 or
                abs(spec.tw_speed*spec.contract.units.distance_km_per_unit/
                    spec.contract.units.hours_per_time_unit-spec.contract.units.speed_kmph) > 1e-9):
            raise ValueError('invalid/unaligned SAA reference specification')
        super().__init__(budget=spec.budget, capacity=spec.capacity,
            booking_horizon=spec.booking_horizon, contract=spec.contract,
            cooling_share=spec.cooling_share,
            sampler=CondHistoricalSampler(copy.deepcopy(spec.history), spec.n_neighbors),
            reject_penalty=dict(enumerate(spec.reject_penalty)), K=spec.K,
            time_limit=spec.time_limit, arm_seed=spec.stream_seed,
            shadow_mode=spec.shadow_mode, energy_pricing=spec.energy_pricing,
            standby_orders=spec.standby_orders, future_policy='reveal', incr_eval=False)
        self.admission_count = 0
        self.draw_audit = []
        self.admission_checks = []

    def export_state(self):
        state = super().export_state()
        state['p2_saa'] = {'signature': self.spec.signature,
            'admission_count': self.admission_count, 'draw_audit': copy.deepcopy(self.draw_audit),
            'admission_checks': copy.deepcopy(self.admission_checks)}
        return state

    def restore_state(self, state):
        extra = state.get('p2_saa', {})
        if extra.get('signature') != self.spec.signature:
            raise ValueError('SAA history/configuration identity changed')
        super().restore_state(copy.deepcopy(state))
        self.admission_count = int(extra['admission_count'])
        self.draw_audit = copy.deepcopy(extra['draw_audit'])
        self.admission_checks = copy.deepcopy(extra['admission_checks'])

    def _try_insert_certified(self, env, inst, o, vehicles, served, clock):
        saved = copy.deepcopy(self._plan)
        ok = super()._try_insert_certified(env, inst, o, vehicles, served, clock)
        if ok:
            view = ref.view_from_saa(env, self, clock, vehicles, served,
                                     accepted_override=self._accepted | {int(o)})
            gate = ref.reference_check(view, ref.plans_from_view(view))
            self.admission_checks.append((float(clock), int(o), gate['ok'], gate.get('reason')))
            ok = gate['ok']
        if not ok:
            self._plan = saved
        return ok

    def on_reveal(self, engine, inst, clock, vehicles, served, visible_ids):
        proxy, published = ref.saa_public_env(engine, inst, clock)
        self._reset_if_new(inst)
        reserved = proxy.get_reserved_customers(vehicles)
        new = sorted((c for c in published if c and not served[c] and c not in reserved
                      and c not in self._accepted and c not in self._rejected),
                     key=lambda c: (proxy.reveal_time[0, c], c))
        if not new:
            return
        start = time.perf_counter()
        self._sync_id_map(proxy, 0, clock)
        visible = self._build_snapshot(proxy, 0, clock, vehicles, served)
        # External key contains only published arrival identity/time, not a plan,
        # branch number, engine event counter or private future dataset identity.
        arrivals = tuple(c for c in published if c and
                         abs(proxy.reveal_time[0, c]-clock) <= ref.EVENT_TOL)
        key = ('SAA_REVEAL', float(clock).hex(), arrivals)
        rng = EventRNG(self.spec.stream_id, self.spec.stream_seed).generator(key)
        scenarios = self.sampler.sample(visible, rng, self.K)
        self.draw_audit.append({'key': key, 'scenarios': snapshot_state_hash(
            {'scenarios': [[asdict(o) for o in day] for day in scenarios]})})
        for o in new:
            self.admission_count += 1
            # One event-level deadline, not a fresh 6s per new order.
            self._saa_decide(proxy, 0, clock, vehicles, served, o, scenarios, start)

    def plan(self, engine, inst, clock, vehicles, served, visible_ids, replan_ids=None):
        proxy, published = ref.saa_public_env(engine, inst, clock)
        # StrictOnlineEnv does not call on_reveal at t=0. Admit published initial
        # orders once, before exposing POST_ADMISSION_PRE_REPAIR.
        if any(c and c not in self._accepted | self._rejected for c in published):
            self.on_reveal(engine, inst, clock, vehicles, served, visible_ids)
        super().plan(proxy, inst, clock, vehicles, served, published, replan_ids)
        view = ref.view_from_saa(proxy, self, clock, vehicles, served, replan_ids)
        if set(view.rejected) != self._rejected:
            raise ValueError('phase is not fully post-admission')
        if getattr(engine, 'p2_observe', None) is not None:
            engine.p2_observe(view, copy.deepcopy(self.export_state()))
        if getattr(engine, 'p2_capture', None) is not None:
            context = engine.p2_context
            engine.p2_capture(engine, view, context)


class SaaPhaseEnv(StrictOnlineEnv):
    """Existing engine with an observation context; scheduling/commit are inherited."""
    p2_capture = None
    p2_observe = None
    p2_commit_observe = None

    def _maybe_snapshot(self, inst, clock, reveal_idx, vehicles, traces, served, universe):
        self.p2_context = (inst, clock, self.event_id, reveal_idx, vehicles, traces, served, universe)
        return super()._maybe_snapshot(inst, clock, reveal_idx, vehicles, traces, served, universe)

    def _commit(self, inst, clock, vehicle, traces, next_node):
        result = super()._commit(inst, clock, vehicle, traces, next_node)
        if self.p2_commit_observe is not None:
            self.p2_commit_observe(float(clock), vehicle.vehicle_id, int(next_node), vehicle.status)
        return result


def capture_saa_phase(engine, view, context, spec):
    if engine.booking_horizon != spec.booking_horizon:
        raise ValueError('public booking cutoff changed')
    inst, clock, event, reveal_idx, vehicles, traces, served, universe = context
    if inst != 0 or engine.num_instances != 1 or engine.num_nodes != 51:
        raise ValueError('teacher requires one identified root day in the declared N50 bucket')
    snap = capture_recourse_snapshot(engine, inst, clock, event, reveal_idx, vehicles,
                                     traces, served, universe, preserve_scalar_types=True)
    sealed = seal_capture(snap, phase=PHASE, public_view=view,
                          continuation_identity=spec.signature, source_dataset=engine.dataset)
    sealed.update(scope='SAA_MASK_REFERENCE', public_view=copy.deepcopy(view),
                  spec_signature=spec.signature)
    sealed['saa_identity'] = _saa_capture_identity(sealed)
    return sealed


def _saa_capture_identity(capture):
    return identity([capture['capture_identity'], capture['public_identity'],
                     capture['spec_signature'], ref.identity(capture['public_view'])])


def validate_saa_capture(capture, spec):
    if (capture.get('scope') != 'SAA_MASK_REFERENCE' or capture['phase'] != PHASE or
            capture['spec_signature'] != spec.signature or
            capture['saa_identity'] != _saa_capture_identity(capture) or
            capture['public_identity'] != ref.identity(capture['public_view'])):
        raise ValueError('SAA phase/public/configuration identity drift')
    snap, view = capture['snapshot'], capture['public_view']
    if snap['state_hash'] != snapshot_state_hash(snap):
        raise ValueError('frozen engine snapshot changed')
    expected_seal = identity([PHASE, capture['public_identity'], snap['state_hash'],
        spec.signature, capture['protocol'], capture['dataset_identity']])
    if capture['capture_identity'] != expected_seal:
        raise ValueError('frozen capture seal changed')
    expected_protocol = {'booking_horizon': spec.booking_horizon, 'capacity': spec.capacity,
                         'tw_speed': spec.tw_speed, 'contract_hash': spec.contract.contract_hash}
    if capture['protocol'] != expected_protocol or view.energy_budget != spec.budget:
        raise ValueError('public protocol/budget changed')
    policy = PublicSaaReplanner(spec)
    policy.restore_state(snap['replanner_state'])
    if policy.budget != spec.budget or set(policy._id_map) != {o.oid for o in view.orders if o.oid}:
        raise ValueError('restored budget/visible identity map changed')
    vehicles, _, served = restore_recourse_snapshot(snap)
    if (tuple(sorted(policy._accepted)) != view.accepted or
            tuple(sorted(policy._rejected)) != view.rejected):
        raise ValueError('acceptance ledger changed')
    # Verify both public vehicle/prefix state AND provisional actual C plan.
    by_id = {v.vehicle_id: v for v in vehicles}
    for v in view.vehicles:
        raw = by_id[v.vehicle_id]
        for name in ref.Vehicle.__dataclass_fields__:
            if name == 'ready_time':
                expected = max(view.clock, raw.ready_time) if raw.status in ('idle', 'ready') else raw.ready_time
            elif name == 'mutable_suffix':
                expected = tuple(c for c in policy._plan.get(v.vehicle_id, ())
                                 if not served[c] and c != raw.committed_next)
            elif name == 'served_route':
                expected = tuple(raw.served_route)
            else:
                expected = getattr(raw, name)
            if getattr(v, name) != expected:
                raise ValueError('physical prefix/commitment/P0 plan mismatch: '+name)
    if not ref.reference_check(view, ref.plans_from_view(view))['ok']:
        raise ValueError('KEEP reference check failed')


def prepare_mask_actions(capture, spec, backbone, masks=None, seconds=10.):
    """Freeze every current action BEFORE constructing/reading any exogenous future."""
    validate_saa_capture(capture, spec)
    view = capture['public_view']
    base = ref.plans_from_view(view)
    masks = ref.mask_pool(view, base) if masks is None else tuple(masks)
    actions = []
    for i, mask in enumerate(masks):
        start = time.monotonic()
        try:
            candidate, fail = ref.reconstruct(view, base, tuple(mask), backbone, start+seconds)
            gate = None if fail else ref.reference_check(view, candidate, tuple(mask))
            status = fail or ('OK' if gate['ok'] else 'REFERENCE_REJECT')
            reason = fail if fail else gate.get('reason')
        except Exception as exc:
            candidate, status, reason = None, 'EXCEPTION', type(exc).__name__+': '+str(exc)
        action = {'action_id': 'mask-%d' % i, 'mask': tuple(mask), 'status': status,
                  'reason': reason, 'plan': candidate if status == 'OK' else None,
                  'public_identity': capture['public_identity'],
                  'backbone_identity': copy.deepcopy(backbone.manifest)}
        action['action_identity'] = _action_identity(action)
        actions.append(action)
    return tuple(actions)


def _action_identity(action):
    return snapshot_state_hash({k: ([asdict(p) for _, p in sorted(v.items())]
                                    if k == 'plan' and v is not None else v)
                               for k, v in action.items() if k != 'action_identity'})


def make_future_tapes(capture, actions, spec, K=2):
    """Generate teacher-private external futures from HIST AFTER actions are sealed.

    Replaces hidden slots rather than retaining the source day's true tail/count.
    Physical future generation and SAA planning draws use separate keyed namespaces.
    """
    validate_saa_capture(capture, spec)
    if not actions or any(a['action_identity'] != _action_identity(a) for a in actions):
        raise ValueError('current action set must be frozen before future generation')
    view = capture['public_view']
    orders = tuple(ScenarioOrder(o.oid, o.reveal, o.x, o.y, o.demand, o.temp_class,
                                 o.tw_start, o.tw_end, o.service) for o in view.orders if o.oid)
    sampler = CondHistoricalSampler(copy.deepcopy(spec.history), spec.n_neighbors)
    prefix = SimpleNamespace(orders=orders, clock=view.clock)
    tapes = []
    for k in range(K):
        key = ('EXOGENOUS_FUTURE', float(view.clock).hex(), k)
        rng = EventRNG(spec.stream_id, spec.stream_seed).generator(key)
        future = sampler.sample(prefix, rng, 1)[0]
        if any(o.reveal <= view.clock+ref.EVENT_TOL or o.reveal >= spec.booking_horizon for o in future):
            raise ValueError('history future outside public booking domain')
        data = {name: np.zeros((1, 51), np.float32) for name in
                ('demands', 'tw_start', 'tw_end', 'service_time', 'temp_class',
                 'initial_quality', 'reveal_time')}
        data['coords'] = np.zeros((1, 51, 2), np.float32)
        data['tw_end'][:] = next(o.tw_end for o in view.orders if o.oid == 0)
        data['initial_quality'][:] = 1.
        data['reveal_time'][:] = spec.booking_horizon+1.
        for o in view.orders:
            c = o.oid
            data['coords'][0, c] = [o.x, o.y]
            for name, val in (('demands', o.demand), ('tw_start', o.tw_start), ('tw_end', o.tw_end),
                              ('service_time', o.service), ('temp_class', o.temp_class),
                              ('initial_quality', o.initial_quality), ('reveal_time', o.reveal)):
                data[name][0, c] = val
        free = [c for c in range(1, 51) if c not in {o.oid for o in view.orders}]
        if len(future) > len(free):
            raise ValueError('external history future exceeds public N50 bucket')
        for c, o in zip(free, future):
            data['coords'][0, c] = [o.x, o.y]
            for name, val in (('demands', o.demand), ('tw_start', o.tw_start), ('tw_end', o.tw_end),
                              ('service_time', o.service_time), ('temp_class', o.temp_class),
                              ('initial_quality', spec.contract.quality.initial_value[o.temp_class]),
                              ('reveal_time', o.reveal)):
                data[name][0, c] = val
        # The source engine itself uses float32 inputs; all public prefix bytes
        # must round-trip exactly. Any incompatible representation blocks replay.
        tape = {'dataset': data, 'draw_key': key, 'dataset_identity': dataset_digest(data),
                'frozen_actions': tuple(a['action_identity'] for a in actions)}
        tape['scenario_identity'] = identity([spec.signature, key, tape['dataset_identity']])
        tapes.append(tape)
    return tuple(tapes)


def _snapshot_for_branch(capture, action, tape, spec):
    validate_saa_capture(capture, spec)
    if (dataset_digest(tape['dataset']) != tape['dataset_identity'] or
            tape['scenario_identity'] != identity([spec.signature, tape['draw_key'], tape['dataset_identity']])):
        raise ValueError('exogenous tape identity changed')
    snap = copy.deepcopy(capture['snapshot'])
    data, view = tape['dataset'], capture['public_view']
    prefix_env = StrictOnlineEnv(data, spec.capacity, tw_speed=spec.tw_speed,
        num_vehicles=snap['num_vehicles'], coldchain_contract=spec.contract,
        booking_horizon=spec.booking_horizon)
    policy = PublicSaaReplanner(spec)
    policy.restore_state(snap['replanner_state'])
    vehicles, _, served = restore_recourse_snapshot(snap)
    proxy, _ = ref.saa_public_env(prefix_env, 0, view.clock)
    replay_view = ref.view_from_saa(proxy, policy, view.clock, vehicles, served, view.replan_ids)
    if ref.identity(replay_view) != capture['public_identity']:
        raise ValueError('exogenous future changed the frozen public prefix')
    snap['customer_universe'] = np.flatnonzero(data['demands'][0] > 0).astype(np.int32)
    if action is not None:
        if (action['action_identity'] != _action_identity(action) or
                action['action_identity'] not in tape['frozen_actions'] or
                action['public_identity'] != capture['public_identity'] or action['status'] != 'OK'):
            raise ValueError('invalid/unfrozen intervention')
        candidate = action['plan']
        gate = ref.reference_check(view, candidate, action['mask'])
        if not gate['ok']:
            raise ValueError('independent intervention gate: '+gate['reason'])
        state = snap['replanner_state']
        by_id = {v.vehicle_id: v for v in view.vehicles}
        for vid, p in candidate.items():
            if vid not in ref.mutable_ids(view):
                continue
            old = state['plan'].get(str(vid), [])
            # Keep C's historical ledger entries; only the mutable suffix changes.
            prefix = [c for c in old if c in view.served or c == by_id[vid].committed_next]
            state['plan'][str(vid)] = prefix+list(p.suffix)
    snap['state_hash'] = snapshot_state_hash(snap)
    return snap


def _run_branch(capture, action, tape, spec, fault=None):
    snap = _snapshot_for_branch(capture, action, tape, spec)
    policy = PublicSaaReplanner(spec)
    env = SaaPhaseEnv(copy.deepcopy(tape['dataset']), spec.capacity, tw_speed=spec.tw_speed,
        num_vehicles=snap['num_vehicles'], replanner=policy, coldchain_contract=spec.contract,
        booking_horizon=spec.booking_horizon)
    validate_snapshot_against_env(env, snap)
    first, commits = [], []
    def observe(view, state):
        if first:
            return
        entry = {'plan_hash': ref.plan_hash(ref.plans_from_view(view)),
                 'state_plan': copy.deepcopy(state['plan']), 'accepted': state['accepted'],
                 'admission_count': state['p2_saa']['admission_count'],
                 'public_identity': ref.identity(view)}
        first.append(entry)
        if fault == 'hang_after_install':
            time.sleep(3600)
        if fault == 'raise_after_install':
            raise RuntimeError('synthetic fault after actual SAA plan installation')
    env.p2_observe = observe
    env.p2_commit_observe = lambda clock, vid, node, status: commits.append((vid, node, status)) \
        if clock == capture['public_view'].clock else None
    traces, served = env.run_resumed(snap)
    ev = evaluate_trace_a1(traces, {k: v[0] for k, v in tape['dataset'].items()}, spec.contract,
                           policy._accepted, policy._rejected, spec.budget,
                           dict(enumerate(spec.reject_penalty)))
    failures = list(ev['failures'])
    if not finite([asdict(t) for t in traces]) or not finite(ev):
        failures.append('nonfinite_trace_or_result')
    for t in traces:
        load = 0.
        for service in t.services:
            load += float(tape['dataset']['demands'][0, service.node])
            if load > spec.capacity-ref.CAPACITY_MARGIN:
                failures.append('conservative_capacity_violation')
        cc = t.final_coldchain_state
        if cc is not None and cc.thermal_violation_count:
            failures.append('reference_endpoint_temperature_violation')
        if cc is not None:
            for r in cc.delivered_to_depot:
                if spec.contract.quality.threshold_type[r.temp_class] == 'absolute' and (
                        r.salable is not True or r.delivered_quality < spec.contract.quality.threshold[r.temp_class]):
                    failures.append('independent_fish_salability')
    if policy.timeouts:
        failures.append('SAA_DECISION_TIMEOUT')
    ledger = policy.export_state()
    end = {'traces': [asdict(t) for t in traces], 'served': served, 'ledger': ledger, 'evaluation': ev}
    return {'status': 'OK' if not failures else 'HARD_FAIL',
            'hard_feasible': not failures, 'utility': ev['utility'] if not failures else None,
            'raw_reference_utility': ev['utility'], 'failures': sorted(set(failures)),
            'terminal_identity': snapshot_state_hash(end), 'trace_identity': snapshot_state_hash({'traces': end['traces']}),
            'first_entry': first[0], 'actual_entry_commits': tuple(commits),
            'ledger': ledger, 'evaluation': ev}


def _branch_worker(conn, capture, action, tape, spec, fault):
    try:
        conn.send(_run_branch(capture, action, tape, spec, fault))
    except BaseException as exc:
        conn.send({'status': 'EXCEPTION', 'utility': None, 'hard_feasible': False,
                   'failures': [type(exc).__name__+': '+str(exc)]})
    finally:
        conn.close()


def run_branch(capture, action, tape, spec, seconds=60., *, synthetic_fault=None):
    """One complete private branch in its own process; failure rows are never omitted."""
    metadata = {'scenario_identity': tape['scenario_identity'], 'continuation_identity': spec.signature,
        'capture_identity': capture['saa_identity'], 'action_id': 'KEEP' if action is None else action['action_id'],
        'action_identity': 'KEEP' if action is None else action['action_identity'],
        'qualification': 'REFERENCE_ONLY'}
    if action is not None and action['status'] != 'OK':
        return {**metadata, 'status': action['status'], 'hard_feasible': False,
                'utility': None, 'failures': [action['reason']]}
    if seconds <= 1:
        raise ValueError('branch slot must include one second termination reserve')
    # Reject tampering before spawning. No broken capture is silently converted
    # into a legitimate counterfactual observation.
    _snapshot_for_branch(capture, action, tape, spec)
    before = snapshot_state_hash(capture['snapshot'])
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_branch_worker, args=(child, capture, action, tape, spec, synthetic_fault))
    start = time.monotonic()
    process.start()
    child.close()
    result = None
    try:
        if parent.poll(max(0., seconds-1-(time.monotonic()-start))):
            try:
                result = parent.recv()
            except EOFError:
                result = {'status': 'EXCEPTION', 'utility': None, 'hard_feasible': False,
                          'failures': ['worker exited without receipt']}
        else:
            result = {'status': 'ROLLOUT_TIMEOUT', 'utility': None, 'hard_feasible': False,
                      'failures': ['branch slot expired']}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(.4)
        if process.is_alive():
            process.kill()
            process.join(.4)
        parent.close()
    if process.is_alive() or time.monotonic()-start > seconds:
        raise RuntimeError('STOP: branch could not be reclaimed within slot')
    if snapshot_state_hash(capture['snapshot']) != before:
        raise RuntimeError('STOP: branch mutated source snapshot')
    return {**metadata, **result, 'worker_reclaimed': True}


def collect_real_labels(capture=None, actions=None, tapes=None, spec=None, *, synthetic=False,
                        seconds=60., synthetic_faults=None):
    """Implemented queue/labels; THIS TURN real training-data collection stays locked."""
    if not synthetic:
        raise PermissionError('REAL_COLLECTION_NOT_AUTHORISED; synthetic integration only')
    validate_saa_capture(capture, spec)
    faults = synthetic_faults or {}
    queue = [(k, action) for k in range(len(tapes)) for action in (None, *actions)]
    rows = [{'scenario_identity': tapes[k]['scenario_identity'], 'continuation_identity': spec.signature,
        'action_id': 'KEEP' if action is None else action['action_id'],
        'status': 'NOT_ATTEMPTED', 'utility': None, 'hard_feasible': False,
        'failures': []} for k, action in queue]
    blocked = False
    for i, (k, action) in enumerate(queue):
        keep_index = k*(len(actions)+1)
        if blocked:
            rows[i]['failures'] = ['queue stopped after identity/supervision failure']
            continue
        if action is not None and rows[keep_index]['status'] != 'OK':
            rows[i]['failures'] = ['KEEP unavailable; branch not attempted']
            continue
        try:
            rows[i] = run_branch(capture, action, tapes[k], spec, seconds,
                synthetic_fault=faults.get((k, rows[i]['action_id'])))
        except (ValueError, RuntimeError) as exc:
            rows[i].update(status='BLOCKED', failures=[type(exc).__name__+': '+str(exc)])
            blocked = True
    for i, (k, action) in enumerate(queue):
        if action is not None:
            rows[i]['paired'] = paired_label(rows[i], rows[k*(len(actions)+1)])
    labels = []
    for action in actions:
        selected = [r for r in rows if r['action_id'] == action['action_id']]
        valid = len(selected) == len(tapes) and all(r['paired']['valid'] for r in selected)
        labels.append({'action_id': action['action_id'], 'valid': valid,
            'delta_utility': float(np.mean([r['paired']['delta_utility'] for r in selected])) if valid else None,
            'statuses': [r['status'] for r in selected],
            'keep_statuses': [r['paired']['keep_status'] for r in selected],
            'training_weight': 1. if valid else 0.})
    if len(rows) != len(tapes)*(len(actions)+1):
        raise RuntimeError('failure ledger lost a planned branch')
    return {'qualification': 'REFERENCE_ONLY', 'rows': rows, 'labels': labels,
            'n_planned': len(rows), 'status': 'BLOCKED' if blocked else 'COMPLETE',
            'source_identity': source_signature(),
            'capture_manifest': {k: copy.deepcopy(capture[k]) for k in
                ('phase', 'saa_identity', 'capture_identity', 'public_identity', 'dataset_identity', 'protocol')},
            'action_manifest': [{k: copy.deepcopy(a[k]) for k in
                ('action_id', 'action_identity', 'mask', 'status', 'reason', 'backbone_identity')} for a in actions],
            'tape_manifest': [{k: copy.deepcopy(t[k]) for k in
                ('scenario_identity', 'draw_key', 'dataset_identity')} for t in tapes],
            'real_collection_authorised': False}
