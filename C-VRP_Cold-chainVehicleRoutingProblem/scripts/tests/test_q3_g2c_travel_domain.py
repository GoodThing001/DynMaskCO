#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-2 ERRATA ROUND 2 CONTROLS -- the distance-model domain (D-*) and the execution monitor (M-*).

Answers the two things the independent review demanded before G-2 can be closed:

  S-8  `sqrt(2)/tw_speed` is only a travel bound inside a CERTIFIED distance model.  `_travel` divides
       `dataset['dist_mat']` (when present) by `tw_speed`, and the generator has branches that scale
       edge weights up, so the bound has to be derived per model and WITNESSED per instance.
  S-9  a call counter is not a monitor.  The monitor must also detect a replaced entry point, a
       callback that rewrites cold-chain state directly, and an undeclared restore -- and a monitor
       that fired must not yield a certification-grade receipt.

Usage: python scripts/tests/test_q3_g2c_travel_domain.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'coldchain'))
sys.path.insert(0, str(EXT / 'scripts' / 'm1b_adapter'))
sys.path.insert(0, str(EXT / 'scripts' / 'simulation'))

import coldchain_contract as CC                                            # noqa: E402
import q3_g2c_travel_domain_experimental as G2C                            # noqa: E402

ROWS = []
LIMITS = ('read-only static analysis + tiny synthetic episodes on the CPU; no training, no GPU, no '
          'formal rollout, no final-test seed, no production wiring')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-6s %s' % (st, cid, name[:94]))


def try_call(fn, *a, **k):
    try:
        return {'ok': True, 'value': fn(*a, **k)}
    except G2C.Refusal as exc:
        return {'ok': False, 'code': exc.code, 'detail': exc.detail}


# ---------------------------------------------------------------------------------------------
# tiny episodes
# ---------------------------------------------------------------------------------------------
class ServeOne(object):
    def __init__(self, tamper=None, hide_advance=None, advance_in_plan=None):
        self.done = False
        self.tamper = tamper
        self.hide_advance = hide_advance
        self.advance_in_plan = advance_in_plan

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        if self.hide_advance is not None:
            env._advance_coldchain_to = self.hide_advance
        if self.advance_in_plan is not None:
            for v in vehicles:
                if v.status in ('ready', 'committed'):
                    env._advance_coldchain_to(v, float(clock))
                    break
        free = [c for c in visible_ids if not served_mask[c]]
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            v.mutable_suffix = [] if self.done or not free else [int(free[0])]
        if free:
            self.done = True
        if self.tamper is not None:
            self.tamper(vehicles, float(clock))


def make_env(replanner, contract, coords=None, n_nodes=2, tw_start=None, dist_mat=None):
    import numpy as np
    if coords is None:
        coords = np.array([[0.0, 0.0], [0.9, 0.0]], dtype=np.float32)
    data = {
        'coords': coords[None, ...].astype(np.float32),
        'demands': np.array([[0.0] + [5.0] * (n_nodes - 1)], dtype=np.float32),
        'tw_start': (np.array([[0.0, 23.5]]) if tw_start is None
                     else np.asarray(tw_start, dtype=np.float32)[None, ...]),
        'tw_end': np.array([[24.0] + [24.0] * (n_nodes - 1)], dtype=np.float32),
        'service_time': np.array([[0.0] + [0.4] * (n_nodes - 1)], dtype=np.float32),
        'reveal_time': np.zeros((1, n_nodes), dtype=np.float32),
        'temp_class': np.zeros((1, n_nodes), dtype=np.int32),
        'initial_quality': np.ones((1, n_nodes), dtype=np.float32),
    }
    if dist_mat is not None:
        data['dist_mat'] = np.asarray(dist_mat, dtype=np.float32)[None, ...]
    from strict_online_env import StrictOnlineEnv
    return StrictOnlineEnv(data, capacity=contract.operational.shared_vehicle_capacity,
                           num_vehicles=1, replanner=replanner, coldchain_contract=contract)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import numpy as np
    contract = CC.default_pilot_contract()
    speed = 1.0

    # ============================================================================ D-1 Euclidean
    splits = sorted(EXT.glob('data/m0_scale/*.npz'))
    witnesses, bad = [], []
    for path in splits:
        d = np.load(path, allow_pickle=True)
        for i in range(int(d['coords'].shape[0])):
            w = try_call(G2C.certify_distance_model, d['dist_mat'][i], d['coords'][i], 'euclidean')
            if not w['ok']:
                bad.append({'file': path.name, 'inst': i, 'code': w['code']})
            elif i < 2:
                witnesses.append({'file': path.name, 'inst': i,
                                  'max_edge': w['value']['max_edge'],
                                  'max_abs_dev_from_euclidean':
                                      w['value']['max_abs_dev_from_euclidean'],
                                  'symmetric': w['value']['symmetric']})
    eu = G2C.travel_bound(('euclidean',), speed)
    ok1 = not bad and eu['travel_bound'] >= float(np.sqrt(2.0)) - 1e-15
    ck('D-1', 'the registered splits pass the Euclidean witness per instance (dist_mat symmetric and '
       'equal to the Euclidean matrix of coords within 1e-6), so `sqrt(2)/tw_speed` IS a valid bound '
       'there -- as a WITNESSED instance property, not as a repo-wide invariant',
       ok1,
       {'n_instances_checked': sum(int(np.load(p, allow_pickle=True)['coords'].shape[0])
                                   for p in splits),
        'failures': bad[:5], 'examples': witnesses, 'travel_bound': eu},
       conclusion='Euclidean is admissible here, and only here')

    # ============================================================================ D-2 asymmetric
    d0 = np.load(splits[0], allow_pickle=True)
    coords = np.asarray(d0['coords'][0], dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    euclid = np.sqrt((diff ** 2).sum(axis=-1))
    asym = euclid.copy()
    # the generator's two mechanisms, at their maxima
    asym[1:, 1:] = euclid[1:, 1:] * G2C.ASYM_MAX * G2C.RUSH_MAX
    w_asym = try_call(G2C.certify_distance_model, asym, coords, 'euclidean')
    bound_eu = G2C.travel_bound(('euclidean',), speed)['travel_bound']
    bound_asym = G2C.travel_bound(('asymmetric_transit',), speed)['travel_bound']
    ok2 = (not w_asym['ok'] and w_asym['code'] == 'DISTANCE_MODEL_WITNESS_FAILED'
           and float(asym.max()) > bound_eu
           and bound_asym >= float(asym.max()))
    ck('D-2', 'asymmetric traffic multipliers break the Euclidean bound: the witness rejects the '
       'matrix, `sqrt(2)` is strictly smaller than the largest edge, and only the model-derived '
       'bound (multiplier ceiling 1.0455 x 2.0 from the generator ranges) covers it',
       ok2,
       {'asym_multiplier_ceiling': G2C.ASYM_MAX * G2C.RUSH_MAX,
        'max_edge_injected': float(asym.max()), 'bound_euclidean': bound_eu,
        'bound_asymmetric': bound_asym, 'witness': w_asym},
       conclusion='S-8: sqrt(2) is not a universal bound')

    # ============================================================================ D-3 perturbation
    pert = euclid.copy()
    idx = np.array([1, 2, 3])
    for i in idx:
        for j in idx:
            if i != j:
                pert[i, j] *= G2C.PERTURB_MAX
    w_pert = try_call(G2C.certify_distance_model, pert, coords, 'euclidean')
    bound_pert = G2C.travel_bound(('long_tail_perturbation',), speed)['travel_bound']
    ok3 = (not w_pert['ok'] and float(pert.max()) > bound_eu and bound_pert >= float(pert.max()))
    ck('D-3', 'long-tail edge perturbations (the generator multiplies selected edges by U(1.5, 5.0)) '
       'also break the Euclidean bound; the perturbation model bound covers them and the Euclidean '
       'witness refuses the matrix',
       ok3,
       {'perturb_multiplier_ceiling': G2C.PERTURB_MAX, 'max_edge_injected': float(pert.max()),
        'bound_euclidean': bound_eu, 'bound_perturbation': bound_pert, 'witness': w_pert},
       conclusion='a second source branch, a second model')

    # ============================================================================ D-4 external
    unb = try_call(G2C.travel_bound, ('external_undeclared',), speed)
    w_ext = try_call(G2C.certify_distance_model, euclid, coords, 'external_undeclared')
    ok4 = (not unb['ok'] and unb['code'] == 'DISTANCE_MODEL_UNBOUNDED'
           and not w_ext['ok'] and w_ext['code'] == 'DISTANCE_MODEL_UNBOUNDED')
    ck('D-4', 'an externally supplied distance matrix with no declared edge-weight bound gets NO '
       'bound at all: both the bound derivation and the per-instance witness refuse, instead of '
       'quietly reusing sqrt(2)',
       ok4, {'travel_bound': unb, 'witness': w_ext},
       conclusion='an unknown cost model is refused, not approximated')

    # ============================================================================ D-5 tw_speed
    speeds = [0.0, -1.0, float('nan'), float('inf'), None, 'fast']
    sp = [{'tw_speed': repr(s), **(lambda r: {'ok': r['ok'], 'code': r.get('code')})(
        try_call(G2C.travel_bound, ('euclidean',), s))} for s in speeds]
    ok5 = all(not e['ok'] and e['code'] == 'TW_SPEED_INVALID' for e in sp)
    ck('D-5', 'a non-positive, non-finite, missing or non-numeric `tw_speed` refuses: the speed is '
       'part of the travel bound, so an unvalidated speed silently rescales every clock', ok5,
       {'cases': sp}, conclusion='the speed is validated before it divides anything')

    # ============================================================================ M-1 counting
    env = make_env(ServeOne(), contract)
    mon = G2C.ExecutionMonitor(env, n_max=20003, clock_sup=24.0 + 1.5).install()
    G2C.instrument(env, mon, replanner=env.replanner)
    env.run(0)
    rec = mon.receipt()
    clocks = list(mon.clocks)
    okm1 = (sum(mon.counts.values()) >= 2 and rec['certifiable'] is True
            and rec['unmonitored_writes'] == [] and rec['bypass_events'] == []
            and max(mon.clocks) > 24.0)
    ck('M-1', 'the monitor counts every advance and stays certifiable on a clean episode, while still '
       'recording the timestamp above the horizon (the E-01 mechanism, now observed through the '
       'monitor)',
       okm1, {'receipt': {k: rec[k] for k in ('counts', 'max_clock', 'n_max', 'problems',
                                             'certifiable', 'coverage', 'not_covered')},
              'event_clocks': clocks},
       conclusion='counting works, and the receipt carries its own caveats')

    # ============================================================================ M-2 dynamic forward
    env = make_env(ServeOne(advance_in_plan=True), contract)
    mon = G2C.ExecutionMonitor(env, n_max=20003, clock_sup=100.0).install()
    G2C.instrument(env, mon, replanner=env.replanner)
    env.snapshot_hook = lambda *a, **k: None
    env.run(0)
    rec2 = mon.receipt()
    okm2 = (mon.callback_advances > 0 and rec2['certifiable'] is True
            and rec2['unmonitored_writes'] == [])
    ck('M-2', 'a callback that advances the chain itself is still COUNTED (the entry point is the '
       'audited attribute) and is attributed to a callback, so the count bound remains meaningful '
       'when an OPEN sink is used',
       okm2, {'callback_advances': mon.callback_advances, 'counts': mon.counts,
              'problems': rec2['problems']},
       conclusion='forwarded calls are visible as long as they go through the attribute')

    # ============================================================================ M-3 direct write
    # The tamper replaces the frozen cold-chain state with a forged copy (one field bumped) WITHOUT
    # calling the advance.  This does not trip the env's own backwards-clock guard, so nothing else
    # in the engine notices it -- only the monitor's state diff does.
    import dataclasses

    def tamper(vehicles, clock):
        for v in vehicles:
            if v.coldchain_state is not None:
                v.coldchain_state = dataclasses.replace(
                    v.coldchain_state,
                    cumulative_energy_kwh=v.coldchain_state.cumulative_energy_kwh + 1.0,
                    thermal_violation_count=v.coldchain_state.thermal_violation_count + 1)

    env = make_env(ServeOne(tamper=tamper), contract)
    mon = G2C.ExecutionMonitor(env, n_max=20003, clock_sup=100.0).install()
    G2C.instrument(env, mon, replanner=env.replanner)
    env.snapshot_hook = lambda *a, **k: None
    env.run(0)
    rec3 = mon.receipt()
    okm3 = (len(rec3['unmonitored_writes']) >= 1
            and 'UNMONITORED_STATE_WRITE' in rec3['problems']
            and rec3['certifiable'] is False)
    ck('M-3', 'a callback that replaces the cold-chain STATE with a forged copy (energy and violation '
       'counters bumped, no advance call, the env own backwards-clock guard does not fire) is caught '
       'by the state-diff window: the receipt lists the unmonitored write and is no longer '
       'certifiable, so counting alone would have missed it',
       okm3, {'unmonitored_writes': rec3['unmonitored_writes'][:2],
              'problems': rec3['problems'], 'certifiable': rec3['certifiable']},
       conclusion='S-9: the monitor covers state writes, not just call counts')

    # ============================================================================ M-4 bypass
    env = make_env(ServeOne(hide_advance=lambda v, c: None), contract)
    mon = G2C.ExecutionMonitor(env, n_max=20003, clock_sup=100.0).install()
    G2C.instrument(env, mon, replanner=env.replanner)
    env.snapshot_hook = lambda *a, **k: None
    caught = None
    try:
        env.run(0)
    except G2C.Refusal as exc:
        caught = exc.code
    okm4 = (caught == 'MONITOR_BYPASSED' and len(mon.bypass_events) >= 1)
    ck('M-4', 'a callback that REPLACES `env._advance_coldchain_to` is caught by the entry-point '
       'identity check, because the monitor owns that attribute and verifies it before every '
       'audited window',
       okm4, {'caught': caught, 'bypass_events': mon.bypass_events[:2]},
       conclusion='replacement is a refusal, not a silent blind spot')

    # ============================================================================ M-5 restore
    env = make_env(ServeOne(), contract)
    mon = G2C.ExecutionMonitor(env, n_max=20003, clock_sup=100.0).install()
    undeclared = try_call(mon.guard_restore, [], False)
    declared = try_call(mon.guard_restore, [], True)
    okm5 = (not undeclared['ok'] and undeclared['code'] == 'UNDECLARED_ANCHOR_INJECTION'
            and declared['ok'] is True)
    ck('M-5', 'an anchor injected by a restore path is either DECLARED or refused '
       '(`UNDECLARED_ANCHOR_INJECTION`); the control is at the monitor API level because exercising a '
       'real `run_resumed` needs a captured snapshot, and a resume inside an interval is already '
       'refused by G-2',
       okm5, {'undeclared': undeclared, 'declared': declared,
              'limitation': 'API-level control, not a full run_resumed round trip'},
       conclusion='the restore path cannot move the anchor silently')

    # ============================================================================ M-6 over limit
    env = make_env(ServeOne(), contract)
    mon = G2C.ExecutionMonitor(env, n_max=1, clock_sup=100.0).install()
    G2C.instrument(env, mon, replanner=env.replanner)
    env.snapshot_hook = lambda *a, **k: None
    caught6 = None
    try:
        env.run(0)
    except G2C.Refusal as exc:
        caught6 = exc.code
    rec6 = mon.receipt()
    okm6 = (caught6 == 'COUNT_BOUND_VIOLATED' and rec6['certifiable'] is False
            and 'COUNT_BOUND_VIOLATED' in rec6['problems'])
    ck('M-6', 'exceeding the count bound aborts the episode AND leaves the receipt non-certifiable, so '
       'no installable certification can be formed after the limit fired',
       okm6, {'caught': caught6, 'problems': rec6['problems'], 'certifiable': rec6['certifiable']},
       conclusion='the limit is terminal, not a warning')

    # ============================================================================ F-1 folded verdict
    q_clean = G2C.qualify_travel_and_clock(horizon=24.0, models=('euclidean',), tw_speed=1.0,
                                           n_max=20003, observed_max_clock=24.8,
                                           monitor_receipt={'certifiable': True, 'problems': []},
                                           distance_witness={'satisfied': True})
    q_no_monitor = G2C.qualify_travel_and_clock(horizon=24.0, models=('euclidean',), tw_speed=1.0,
                                                n_max=20003, observed_max_clock=24.8,
                                                monitor_receipt=None,
                                                distance_witness={'satisfied': True})
    q_bad_monitor = G2C.qualify_travel_and_clock(horizon=24.0, models=('euclidean',), tw_speed=1.0,
                                                 n_max=20003, observed_max_clock=24.8,
                                                 monitor_receipt={'certifiable': False,
                                                                  'problems': [
                                                                      'UNMONITORED_STATE_WRITE']},
                                                 distance_witness={'satisfied': True})
    q_unbounded = G2C.qualify_travel_and_clock(horizon=24.0, models=('external_undeclared',),
                                               tw_speed=1.0, n_max=20003,
                                               monitor_receipt={'certifiable': True,
                                                                'problems': []})
    okf = (q_clean['verdict'] == G2C.PROVED_CONDITIONAL
           and q_no_monitor['verdict'] == G2C.NOT_CERTIFIABLE
           and q_bad_monitor['verdict'] == G2C.REFUSED
           and q_unbounded['verdict'] == G2C.REFUSED)
    ck('F-1', 'the folded verdict behaves as a ladder: a clean Euclidean episode with a monitor is '
       '`PROVED_CONDITIONAL`; the same without a monitor is `NOT_CERTIFIABLE` '
       '(`CALLBACK_ADVANCES_UNWITNESSED`); a fired monitor is `REFUSED`; an unbounded distance model '
       'is `REFUSED`',
       okf,
       {'clean': q_clean['verdict'], 'no_monitor': q_no_monitor['verdict'],
        'bad_monitor': q_bad_monitor['verdict'], 'unbounded': q_unbounded['verdict'],
        'clean_T_clock_sup': q_clean['T_clock_sup'],
        'coverage': {'monitor_required': True, 'distance_model_required': True}},
       conclusion='qualification needs both the domain witness and the monitor')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G2C_FAIL' if n_fail else 'G2C_PASS'
    payload = {
        'schema': 'q3_g2c_travel_domain_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'PRODUCTION_NOT_QUALIFIED': True,
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'distance_models': G2C.DISTANCE_MODELS,
        'generator_ceilings': {'asym': G2C.ASYM_MAX, 'rush': G2C.RUSH_MAX,
                               'perturb_edge': G2C.PERTURB_MAX},
        'rows': ROWS,
        'status_lines': {
            'GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED': 'permanent, not retro-approved',
            'THERMAL_CONTINUOUS_NOT_ESTABLISHED': True, 'Q3_FORMAL_FREEZE_NOT_AUTHORISED': True,
            'M2_NOT_AUTHORISED': True, 'L1': 'BLOCKED', 'L2_NOT_ESTABLISHED': True,
            'A3_NOT_ESTABLISHED': True, 'stable_online_gain': False,
            'HISTORICAL_BACKBONE_UNVERIFIED': True, 'C0': 'AWAITING_USER_BUDGET_RULING',
            't_min': 'always 0', 'final_test_seeds': 'SEALED'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d controls: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
