"""I3 engineering primitives: pre-decision capture and isolated action replay.

Not a registered teacher dataset export or a training runner. Imports existing
SAA/environment without editing them. Only visible is a generator input;
physical snapshots, dataset, current-event scenarios and branch outcomes are
teacher-private. A declared train seed alone does not prove data provenance:
formal export must also seal generator config and the dataset byte identity.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for rel in ('evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(SCRIPTS / rel))
from scenario_saa import SaaReplanner
from recourse_snapshot import (capture_recourse_snapshot, clone_snapshot,
                              restore_recourse_snapshot, snapshot_state_hash,
                              validate_snapshot_against_env)
from strict_online_env import StrictOnlineEnv
from coldchain_evaluator_a1 import evaluate_trace_a1
from run_identity import dataset_identity, dataset_meta_sha256

TRAIN_SEED = 20260925


@dataclass(frozen=True)
class TeacherRecord:
    train_seed: int
    dataset_sha256: str
    dataset_meta_sha256: str
    continuation_signature: str
    snapshot: dict
    visible: object
    candidate: int
    pending: tuple
    event_scenarios: tuple
    record_hash: str = ''


def record_identity(record):
    """Seal candidate/pending/visible/event draws, not just the physical snapshot."""
    return snapshot_state_hash({
        'train_seed': record.train_seed, 'dataset_sha256': record.dataset_sha256,
        'dataset_meta_sha256': record.dataset_meta_sha256,
        'continuation_signature': record.continuation_signature,
        'snapshot_hash': snapshot_state_hash(record.snapshot),
        'visible': asdict(record.visible), 'candidate': record.candidate,
        'pending': record.pending,
        'event_scenarios': [[asdict(o) for o in s] for s in record.event_scenarios]})


def continuation_signature(policy):
    """Current teacher supports the stateless historical sampler protocol only.

    A checkpoint/stateful sampler must add its own sealed state before use.
    Including history bytes prevents factories from silently changing teachers.
    """
    from scenario_saa import CondHistoricalSampler, UncondHistoricalSampler
    if type(policy.sampler) not in (CondHistoricalSampler, UncondHistoricalSampler):
        raise TypeError('Teacher requires a sealed stateless historical sampler')
    keys = ('budget', 'capacity', 'booking_horizon', 'cooling_share', 'reject_penalty',
            'K', 'time_limit', 'arm_seed', 'shadow_mode', 'energy_pricing',
            'standby_orders', 'future_policy', 'overage_price', 'incr_eval',
            'anytime_vote', 'anytime_early_stop')
    legacy_defaults = {'incr_eval': False, 'anytime_vote': False,
                       'anytime_early_stop': True}
    state = {key: getattr(policy, key, legacy_defaults.get(key)) for key in keys}
    state['available_optional_fields'] = sorted(
        key for key in legacy_defaults if hasattr(policy, key))
    state['contract_hash'] = policy.contract.contract_hash
    state['sampler'] = type(policy.sampler).__name__
    state['n_neighbors'] = getattr(policy.sampler, 'n_neighbors', None)
    state['history'] = [[asdict(o) for o in day] for day in policy.sampler.history]
    return snapshot_state_hash(state)


class ReplaySafeSaaReplanner(SaaReplanner):
    """Preserve reveal-order ID mapping and counters in private teacher snapshots."""
    def export_state(self):
        state = super().export_state()
        state['i3_replay'] = {
            'id_map': {int(k): int(v) for k, v in self._id_map.items()},
            'next_oid': int(self._next_oid), 'timeouts': int(self.timeouts),
            'partial_commits': int(getattr(self, 'partial_commits', 0)),
            'early_stops': int(getattr(self, 'early_stops', 0))}
        return state

    def restore_state(self, state):
        super().restore_state(state)
        extra = state.get('i3_replay')
        if extra is None:
            raise ValueError('Teacher snapshot lacks reveal-order identity and counters')
        self._id_map = {int(k): int(v) for k, v in extra['id_map'].items()}
        self._next_oid = int(extra['next_oid'])
        for name in ('timeouts', 'partial_commits', 'early_stops'):
            setattr(self, name, int(extra[name]))


class CaptureTeacherReplanner(ReplaySafeSaaReplanner):
    """Capture selected decision ordinals for an offline teacher.

    capture_ordinals is declared BEFORE a day runs, independent of utility or
    feasibility. Ordinals count all _saa_decide entries, including rejected ones.
    The physical engine hook occurs after on_reveal but before commit; replace
    only its planner state with the state saved BEFORE this individual decision.
    Observer work is excluded from the decision timer. Wall-time replay can
    still differ near the deadline; exact parity is tested with an infinite
    decision budget, not claimed for deployment's 10 seconds.
    """
    def __init__(self, *args, capture_ordinals, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_ordinals = frozenset(int(i) for i in capture_ordinals)
        if any(i < 0 for i in self.capture_ordinals):
            raise ValueError('Negative capture ordinal')
        self._capture_day = None
        self._capture_ordinal = 0
        self._event_entries = []
        self._event_tag = None
        self._generation_input = None

    def _build_snapshot(self, env, inst_idx, clock, vehicles, served_mask):
        # Saa.on_reveal samples ONCE before any same-event decisions. Keep its
        # actual generator input, not an updated per-order planner snapshot.
        visible = super()._build_snapshot(env, inst_idx, clock, vehicles, served_mask)
        self._generation_input = ((int(inst_idx), int(env.event_id), float(clock)), visible)
        return visible

    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
        capture_t0 = time.perf_counter()
        if self._capture_day != inst_idx:
            self._capture_day, self._capture_ordinal = inst_idx, 0
        tag = (int(inst_idx), int(env.event_id), float(clock))
        if self._event_tag != tag:
            self._event_tag, self._event_entries = tag, []
        selected = self._capture_ordinal in self.capture_ordinals
        if self._generation_input is None or self._generation_input[0] != tag:
            raise ValueError('No sealed generation input for this reveal event')
        self._event_entries.append({
            'candidate': int(o),
            'state': deepcopy(self.export_state()) if selected else None,
            'visible': self._generation_input[1] if selected else None,
            'scenarios': tuple(tuple(s) for s in scenarios) if selected else None})
        self._capture_ordinal += 1
        capture_elapsed = time.perf_counter() - capture_t0
        return super()._saa_decide(env, inst_idx, clock, vehicles, served_mask,
                                 o, scenarios, t0 + capture_elapsed)

    def capture_hook(self, records, *, train_seed, dataset_sha256, dataset_meta_sha256):
        if int(train_seed) != TRAIN_SEED:
            raise ValueError('I3 teacher capture accepts only the registered training seed')

        def hook(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                 served_mask, all_customers):
            if self._event_tag != (int(inst_idx), int(event_id), float(clock)):
                return
            for j, entry in enumerate(self._event_entries):
                if entry['state'] is None:
                    continue
                physical = capture_recourse_snapshot(env, inst_idx, clock, event_id,
                    reveal_idx, vehicles, traces, served_mask, all_customers)
                physical['replanner_state'] = deepcopy(entry['state'])
                physical['state_hash'] = snapshot_state_hash(physical)
                visible = entry['visible']
                if any(o.reveal > clock + 1e-6 for o in visible.orders):
                    raise ValueError('Future order in visible teacher model input')
                record = TeacherRecord(int(train_seed), dataset_sha256,
                    dataset_meta_sha256, continuation_signature(self), physical,
                    visible, entry['candidate'],
                    tuple(e['candidate'] for e in self._event_entries[j + 1:]),
                    entry['scenarios'])
                records.append(replace(record, record_hash=record_identity(record)))
            self._event_entries = []
        return hook


def _restored(record, dataset, policy_factory, contract, booking_horizon, tw_speed,
              require_continuation=False):
    if record.train_seed != TRAIN_SEED:
        raise ValueError('Not a registered training record')
    if dataset_identity(dataset) != record.dataset_sha256:
        raise ValueError('Teacher dataset identity changed')
    if dataset_meta_sha256(dataset) != record.dataset_meta_sha256:
        raise ValueError('Teacher dataset shape/dtype identity changed')
    if not record.record_hash or record_identity(record) != record.record_hash:
        raise ValueError('Teacher record metadata changed')
    snap = clone_snapshot(record.snapshot)
    if snapshot_state_hash(snap) != snap.get('state_hash'):
        raise ValueError('Teacher decision snapshot changed')
    policy = policy_factory()
    if not isinstance(policy, ReplaySafeSaaReplanner):
        raise TypeError('Counterfactual continuation must restore full SAA state')
    if require_continuation and continuation_signature(policy) != record.continuation_signature:
        raise ValueError('Counterfactual continuation configuration/history changed')
    env = StrictOnlineEnv(dataset, capacity=record.visible.capacity, tw_speed=tw_speed,
        num_vehicles=int(snap['num_vehicles']), replanner=policy,
        coldchain_contract=contract, booking_horizon=booking_horizon)
    validate_snapshot_against_env(env, snap)
    vehicles, _, served = restore_recourse_snapshot(snap)
    policy.restore_state(snap['replanner_state'])
    candidate = record.candidate
    if (candidate in policy._accepted or candidate in policy._rejected or
            served[candidate] or not snap['visible_mask'][candidate]):
        raise ValueError('Candidate is not visible and undecided in saved state')
    return snap, policy, env, vehicles, served


def _force_action(policy, env, inst_idx, clock, vehicles, served, candidate, accept):
    if accept:
        saved = deepcopy(policy._plan)
        try:
            ok = policy._try_insert_certified(env, inst_idx, candidate, vehicles, served, clock)
        except ValueError:
            ok = False
        if not ok:
            policy._plan = saved
            return False
        policy._accepted.add(candidate)
    else:
        policy._rejected.add(candidate)
    return True


def replay_action(record, dataset, policy_factory, contract, booking_horizon,
                  tw_speed, accept):
    """Force one decision; replay remaining same-event decisions then full future.

    Action insertion certification is an offline teacher intervention, not a
    deployed decision with a 10s budget. Continuation uses factory's fixed SAA
    configuration and actual wall-time fallbacks, identical in the two branches.
    The source record/dataset must remain unchanged. Return None utility for a
    hard-infeasible forced accept; callers keep the sample with zero policy weight.
    """
    snap, policy, env, vehicles, served = _restored(
        record, dataset, policy_factory, contract, booking_horizon, tw_speed,
        require_continuation=True)
    inst, clock = int(snap['instance_id']), float(snap['clock'])
    feasible = _force_action(policy, env, inst, clock, vehicles, served, record.candidate, accept)
    if not feasible:
        return {'action_feasible': False, 'utility': None, 'hard_feasible': None}
    for candidate in record.pending:
        policy._saa_decide(env, inst, clock, vehicles, served, candidate,
                          record.event_scenarios, time.perf_counter())
    snap['replanner_state'] = deepcopy(policy.export_state())
    snap['state_hash'] = snapshot_state_hash(snap)
    traces, _ = env.run_resumed(snap)
    ev = evaluate_trace_a1(traces, {k: v[inst] for k, v in dataset.items()}, contract,
                           policy._accepted, policy._rejected, policy.budget, policy.reject_penalty)
    if dataset_identity(dataset) != record.dataset_sha256:
        raise RuntimeError('Teacher replay mutated dataset')
    return {'action_feasible': True, **ev,
            'accepted': sorted(policy._accepted), 'rejected': sorted(policy._rejected),
            'timeouts': int(policy.timeouts), 'traces': traces}


def teacher_preference(accept, reject, *, minimum_gap):
    """Keep invalid/tie samples; give the objective finite zero placeholders.

    The teacher difference is meaningful only if BOTH full branches satisfy
    hard constraints and have finite utility. Formal export must retain their
    status/failures instead of silently dropping a day or state.
    minimum_gap is a caller-declared training setting, not a selected default.
    """
    if not np.isfinite(minimum_gap) or minimum_gap < 0:
        raise ValueError('Invalid declared teacher threshold')
    valid = all(b.get('action_feasible') and b.get('hard_feasible') and
                b.get('utility') is not None and np.isfinite(b['utility'])
                for b in (accept, reject))
    delta = float(accept['utility'] - reject['utility']) if valid else None
    if delta is not None and not np.isfinite(delta):
        valid, delta = False, None
    eligible = bool(valid and delta != 0 and abs(delta) >= minimum_gap)
    return {'teacher_valid': bool(valid), 'teacher_delta': delta,
            'decision_eligible': eligible, 'loss_teacher_margin': delta if eligible else 0.0}


def scenario_margin(record, dataset, policy_factory, contract, booking_horizon,
                    tw_speed, scenarios):
    """Average accept-reject utility over this generated K-set, same SAA sign rule.

    Simulation is outside autodiff; only the generated draw likelihood receives
    the score gradient. All raw future data is teacher-private. The downstream
    uses a sum; dividing by positive K preserves its decision and makes units
    per action, avoiding a hidden K multiplier in the preference temperature.
    """
    if not scenarios:
        raise ValueError('A decision set must contain at least one scenario')
    if any(o.reveal <= record.visible.clock + 1e-6 for s in scenarios for o in s):
        raise ValueError('Generated scenario contains a past or current order')
    snap, policy, env, vehicles, served = _restored(
        record, dataset, policy_factory, contract, booking_horizon, tw_speed)
    inst, clock = int(snap['instance_id']), float(snap['clock'])
    saved = deepcopy(policy._plan)
    if not _force_action(policy, env, inst, clock, vehicles, served, record.candidate, True):
        return {'action_feasible': False, 'margin': None}
    acc_plan = deepcopy(policy._plan)
    policy._plan = saved
    space, orders = policy._build_space(env, inst, scenarios)
    idx = {id(o): space.n_real + i for i, o in enumerate(orders)}
    aa = policy._sim_states(env, inst, clock, vehicles, served, acc_plan)
    rr = policy._sim_states(env, inst, clock, vehicles, served, saved)
    pc = policy.reject_penalty[policy._class_of(env, inst, record.candidate)]
    differences = []
    for scenario in scenarios:
        ii = [idx[id(o)] for o in scenario]
        differences.append(policy._sim_scenario(env, inst, clock, aa, ii, space)
                           - policy._sim_scenario(env, inst, clock, rr, ii, space) + pc)
    if not np.isfinite(differences).all():
        raise ValueError('Nonfinite teacher-side shadow margin')
    return {'action_feasible': True, 'margin': float(np.mean(differences)),
            'per_scenario_margin': differences, 'K': len(scenarios)}
