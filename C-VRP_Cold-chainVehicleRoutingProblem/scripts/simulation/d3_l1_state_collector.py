#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 F-FREE state collector (user-authorised; strict-online causal).

Hard constraints enforced here:
  * This module and its process MUST NOT import any F / D-Geo code.  `assert_f_free_process()`
    fails loudly if `mpre*`, `mtrained*` or `d3_geo_adapter` are already in `sys.modules`.
  * The environment is always constructed with the explicit, pre-specified public horizon
    `booking_horizon = 24.0` (data-generator declaration, generate_coldchain_data.py:119).
    The `None` default is never used: it makes `has_future_reveal` read real future reveal data.
  * R1 slots are keyed on the reveal EVENT, decided by OBSERVED VISIBLE-SET GROWTH and cross-checked
    against `sorted(set(reveal_time))` -- never on `reveal_idx` alone (t=0 carries reveal_idx=0 and
    the first real reveal may too) and never on `event_id` (it counts service completions and
    returns as well).
  * Restoring is done by building the restore environment HERE with the identical explicit config
    (`booking_horizon=24.0`) before calling the frozen `env.run_resumed(snapshot)`, because
    `recourse_snapshot.resume_from_snapshot` (:350-356) omits `booking_horizon` and would restore
    with None.  The frozen module is NOT modified.

Nothing here scores anything; no plan is chosen by any model.
"""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[2]
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data'):
    if str(EXT / _p) not in sys.path:
        sys.path.insert(0, str(EXT / _p))

HORIZON = 24.0                     # pre-specified public booking horizon (NOT data-derived)
FORBIDDEN_MODULES = ('mpre', 'mpre_trained', 'mpre_policy', 'mtrained_replanner', 'd3_geo_adapter')


def assert_f_free_process():
    hit = sorted(m for m in sys.modules
                 if m.split('.')[0] in FORBIDDEN_MODULES or m.startswith('mpre'))
    if hit:
        raise RuntimeError('F-FREE VIOLATION: forbidden modules already imported: %s' % hit)
    return True


def sha256_file(p):
    if not p:
        return None
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def config_identity(dataset_path, profile_path, contract_path, capacity, num_vehicles, tw_speed,
                    horizon=HORIZON, K=4):
    """Normalised, pre-registration-grade configuration fingerprint."""
    items = {
        'dataset_sha256': sha256_file(dataset_path),
        'objective_profile_sha256': sha256_file(profile_path),
        'coldchain_contract_sha256': sha256_file(contract_path),
        'capacity': float(capacity), 'num_vehicles': int(num_vehicles),
        'tw_speed': float(tw_speed), 'booking_horizon': float(horizon), 'K': int(K),
        'collector_sha256': sha256_file(__file__),
        'strict_online_env_sha256': sha256_file(EXT / 'scripts/simulation/strict_online_env.py'),
        'jf1h_repair_sha256': sha256_file(EXT / 'scripts/simulation/jf1h_repair.py'),
        'recourse_snapshot_sha256': sha256_file(EXT /
                                                'scripts/simulation/recourse_snapshot.py'),
    }
    canon = json.dumps(items, sort_keys=True, separators=(',', ':'))
    return {'items': items, 'fingerprint_sha256': hashlib.sha256(canon.encode()).hexdigest()}


def _plan_fingerprint(vehicles):
    rows = []
    for v in vehicles:
        rows.append((int(v.vehicle_id), str(getattr(v, 'status', '')), int(getattr(v, 'current_node', -1)),
                     getattr(v, 'committed_next', None), tuple(getattr(v, 'mutable_suffix', []) or [])))
    return hashlib.sha256(repr(sorted(rows)).encode()).hexdigest()


def reveal_event_times(env, inst_idx):
    rt = np.asarray(env.reveal_time[inst_idx], dtype=float)
    return sorted(set(float(x) for x in rt if x > 0))


class RevealSlotCollector:
    """snapshot_hook-compatible collector for R1 = the first K reveal events."""

    def __init__(self, env, inst_idx, K, capture=True, clock_eps=1e-6, pre_capture=None,
                 preserve_scalar_types=False):
        self.preserve_scalar_types = preserve_scalar_types
        self.pre_capture = pre_capture
        self.env = env
        self.inst_idx = int(inst_idx)
        self.K = int(K)
        self.capture = capture
        self.eps = float(clock_eps)
        self.expected_reveals = reveal_event_times(env, inst_idx)
        self.slots = []
        self.events = []
        self._seen_visible = None
        self._taken_clocks = set()

    # -- hook -----------------------------------------------------------------------------
    def __call__(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                 all_customers):
        clock = float(clock)
        visible = {int(i) for i in all_customers
                   if float(env.reveal_time[int(inst_idx), int(i)]) <= clock + self.eps}
        grew = (self._seen_visible is not None and len(visible - self._seen_visible) > 0)
        first_hook = self._seen_visible is None
        self._seen_visible = visible
        is_reveal = bool(grew) and not first_hook
        # cross-check against the de-duplicated reveal-time set of this instance
        matches_declared = any(abs(clock - t) < self.eps for t in self.expected_reveals)
        slot = None
        reject = None
        if is_reveal and not matches_declared:
            reject = 'reveal_by_visible_growth_but_no_declared_reveal_time'
        elif is_reveal and matches_declared:
            if clock in self._taken_clocks:
                reject = 'duplicate_slot_for_same_reveal_clock'
            elif len(self.slots) >= self.K:
                reject = 'beyond_K'                      # not an error: extra reveals are ignored
            else:
                slot = len(self.slots) + 1
                self._taken_clocks.add(clock)
        elif not is_reveal and matches_declared and not first_hook:
            reject = 'declared_reveal_time_but_visible_set_did_not_grow'
        record = {'slot': slot, 'event_id': int(event_id), 'reveal_idx': int(reveal_idx),
                  'clock': clock, 'is_first_hook_t0': bool(first_hook),
                  'visible_growth': bool(grew), 'n_visible': len(visible),
                  'declared_reveal_match': bool(matches_declared),
                  'plan_fingerprint': _plan_fingerprint(vehicles), 'reject_reason': reject}
        if slot is not None and self.capture:
            from recourse_snapshot import capture_recourse_snapshot
            if self.pre_capture is not None:
                # the ledger is written (and fsynced) BEFORE the capture happens
                self.pre_capture(slot, clock, int(event_id), int(reveal_idx))
            t0 = time.time()
            snap = capture_recourse_snapshot(env, int(inst_idx), clock, int(event_id),
                                            int(reveal_idx), vehicles, traces, served_mask,
                                            all_customers=list(all_customers),
                                            preserve_scalar_types=self.preserve_scalar_types)
            record['capture_seconds'] = time.time() - t0
            record['snapshot'] = snap
        self.events.append({k: v for k, v in record.items() if k != 'snapshot'})
        if slot is not None:
            self.slots.append(record)
        return None

    def slot_records(self):
        return [r for r in self.slots]

    def assertions(self):
        """The five mandatory assertions the user required of R1 slotting."""
        clocks = [r['clock'] for r in self.slots]
        return {
            't0_never_occupies_a_slot': all(not (r['is_first_hook_t0'] and r['slot'])
                                           for r in self.events),
            'non_reveal_events_never_occupy_a_slot': all(
                (r['slot'] is None) for r in self.events
                if not r['visible_growth'] or r['is_first_hook_t0']),
            'one_slot_per_reveal_clock': len(clocks) == len(set(clocks)),
            'each_slot_captured_exactly_once': all(
                sum(1 for r in self.slots if r['slot'] == s) == 1
                for s in range(1, len(self.slots) + 1)),
            'slots_match_declared_reveal_times': all(r['declared_reveal_match'] for r in self.slots),
            'slots_are_prefix_of_declared_reveals': clocks == self.expected_reveals[:len(clocks)],
        }


def make_replanner(capacity, horizon=HORIZON, contract=None):
    """Construct the JF1-H-F baseline replanner without importing anything model-related."""
    from jf1h_repair import JF1HRepairReplanner
    sig = inspect.signature(JF1HRepairReplanner.__init__)
    params = {n for n in sig.parameters if n != 'self'}
    avail = {'capacity': float(capacity), 'booking_horizon': float(horizon),
             'coldchain_contract': contract, 'contract': contract}
    kwargs = {n: avail[n] for n in params if n in avail}
    if not kwargs and not params:
        return JF1HRepairReplanner(), {'params': [], 'mode': 'no_args'}
    try:
        return JF1HRepairReplanner(**kwargs), {'params': sorted(kwargs), 'mode': 'kwargs'}
    except TypeError as exc:
        raise RuntimeError('cannot construct JF1HRepairReplanner; signature=%s kwargs=%s err=%s'
                           % (str(sig), sorted(kwargs), exc))


def build_env(dataset, capacity, num_vehicles, tw_speed, replanner, contract, horizon=HORIZON):
    """The ONLY sanctioned env constructor: the horizon is always explicit."""
    from strict_online_env import StrictOnlineEnv
    env = StrictOnlineEnv(dataset, capacity, tw_speed, num_vehicles, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=float(horizon))
    assert env.booking_horizon == float(horizon), 'horizon not applied'
    return env


def discover_rollout(env):
    for name in ('run_instance', 'run_episode', 'simulate_instance', 'simulate', 'rollout',
                 'run_one', 'run'):
        fn = getattr(env, name, None)
        if callable(fn):
            return name, fn
    raise RuntimeError('no rollout entry found on StrictOnlineEnv: %s'
                       % [n for n in dir(env) if 'run' in n or 'sim' in n])


def run_baseline_rollout(env, inst_idx, hook=None):
    if hook is not None:
        env.snapshot_hook = hook
    name, fn = discover_rollout(env)
    t0 = time.time()
    out = fn(int(inst_idx))
    return {'entry': name, 'seconds': time.time() - t0, 'result': out}
