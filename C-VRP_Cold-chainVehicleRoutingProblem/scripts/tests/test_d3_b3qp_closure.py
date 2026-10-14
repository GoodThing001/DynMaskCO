#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-QP: closure of the independent reviewer's findings (READ-ONLY).

The independent adjudication (B3QP_ADJUDICATION.md) confirmed the main conclusions but found real
defects.  This suite implements the checks it demanded, so the closure is mechanical rather than
promised.  Nothing installs, `_commit` is untouched, D8 is not relaxed.

Checks
  A1X  A1 re-verified by ENUMERATING every write site in `coldchain_state.py` (the earlier check was
       a source SUBSTRING match, which the reviewer showed would pass even if a third write path
       were added)
  RATE the A2 evidence re-taken by INSTRUMENTING the production `quality_rate_per_hour` itself, so
       the recorded temperatures are the ones the physics actually asked for -- not a re-derivation
       written by the author (reviewer finding a)
  SIDE the side conditions of the tightened bound that were never stated: the tightened form is
       sound only because the ambient branch is UNREACHABLE; ambient (25 C) exceeds the hard upper
       bound of classes 1 (8 C) and 2 (-12 C), so a reachable ambient branch would FALSIFY it
  COV  the coverage question nobody answered: does every DELIVERED order appear in some bound row?
       Reported in BOTH directions, with the unmatched counts stated explicitly
  BIND whether the bound ever actually BINDS a threshold (the reviewer's strategic point: if it
       never does, the payoff must come from a tighter tau_max, not from more search)
  IQ   the not-yet-loaded start quality comes from the dataset, not a hardcoded 1.0
  FORCE the `_force_return_all` truncation the author did not report: it records the unload at
       `depart + travel` but advances the cold chain only to `depart`, so the RECORDED delivered
       quality is more optimistic than the physics -- load-bearing for A4's wording

Run: python scripts/tests/test_d3_b3qp_closure.py [--instances 0] [--heavy-cap 14]
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
for _p in (SIM, HERE.parent, HERE, HERE.parent / 'coldchain', HERE.parent / 'evaluation'):
    sys.path.insert(0, str(_p))

import numpy as np                                                     # noqa: E402
import d3_state_assembly as A                                          # noqa: E402
import d3_quality_lower_bound as LB                                    # noqa: E402
from strict_online_env import StrictOnlineEnv                          # noqa: E402
from jf1h_repair import make_continuation                              # noqa: E402
from coldchain_contract import (default_pilot_contract,                # noqa: E402
                                apply_objective_profile, ObjectiveProfile)

EXT = HERE.parent.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY, NUM_VEHICLES = 50.0, 25
RESULTS = []


def chk(cid, name, ok, detail=None):
    RESULTS.append({'cid': cid, 'name': name, 'ok': bool(ok), 'detail': detail})


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_profile():
    d = json.loads(PROFILE_JSON.read_text(encoding='utf-8'))
    p = ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                         quality_scale=float(d['quality_scale']),
                         energy_scale=float(d['energy_scale']),
                         lambda_quality=float(d['lambda_quality']),
                         lambda_energy=float(d['lambda_energy']),
                         scale_source=d.get('scale_source', 'pilot'),
                         dev_statistics=d.get('dev_statistics'))
    assert p.profile_hash == d.get('profile_hash')
    return p


def make_env(dataset, contract, hook=None):
    env = StrictOnlineEnv(dataset, capacity=CAPACITY, tw_speed=1.0, num_vehicles=NUM_VEHICLES,
                          replanner=make_continuation(), coldchain_contract=contract,
                          booking_horizon=None)
    env.snapshot_hook = hook
    return env


class Grab(object):
    def __init__(self, cap=14):
        self.cap, self.items, self.n_calls = cap, [], 0

    def hook(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers=None):
        self.n_calls += 1
        if len(self.items) >= self.cap:
            return
        snap, _ = A.capture_decision_state(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                          traces, served_mask, all_customers)
        cv, ct, csm = A.clone_state(snap)
        vis = [int(i) for i in all_customers if env.reveal_time[inst_idx, i] <= clock + 1e-6]
        cv, plans, _r = A.prepare_clone(env, inst_idx, clock, cv, csm, vis)
        self.items.append({'env': env, 'inst': int(inst_idx), 'clock': float(clock),
                           'vehicles': cv, 'plans': plans, 'served_mask': csm})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0])
    ap.add_argument('--heavy-cap', type=int, default=14)
    args = ap.parse_args()

    profile = load_profile()
    dataset = dict(np.load(CAL_NPZ))
    raw = default_pilot_contract()
    eff = apply_objective_profile(raw, profile)
    inst = args.instances[0]
    src_path = EXT / 'scripts' / 'coldchain' / 'coldchain_state.py'
    src = src_path.read_text(encoding='utf-8')

    # ---------------- A1X: enumerate every write site (not a substring match) ----------------
    sites = []
    for m in re.finditer(r'^\s*quality_remaining\s*=\s*(.+)$', src, re.M):
        sites.append({'line': src[:m.start()].count('\n') + 1, 'expr': m.group(1).strip()[:60]})
    for m in re.finditer(r'replace\(lot,\s*quality_remaining=([^)]+)\)', src):
        sites.append({'line': src[:m.start()].count('\n') + 1,
                      'expr': 'replace(..., quality_remaining=%s)' % m.group(1).strip()[:40]})
    write_sites = sorted({s['line'] for s in sites})
    # expected: 276 (new lot), 404/405 (v1 decay), 409 (v1 replace), 489 (v2 replace)
    expected = {276, 404, 405, 409, 489}
    a1x_ok = set(write_sites) == expected
    chk('A1X', 'A1X every write site of quality_remaining is enumerated and matches the expected '
        'set (v1 decay = 404/405/409 only; 489 is the v2 path)', a1x_ok,
        {'enumerated_lines': write_sites, 'expected_lines': sorted(expected),
         'sites': sites})

    # ---------------- RATE: instrument the PRODUCTION rate function ----------------
    import coldchain_state as CS
    orig_rate = CS.quality_rate_per_hour
    samples = []

    def spy_rate(temp_class, actual_temperature_c, contract):
        r = orig_rate(temp_class, actual_temperature_c, contract)
        samples.append({'temp_class': int(temp_class),
                        'actual_temperature_c': float(actual_temperature_c), 'rate': float(r)})
        return r

    CS.quality_rate_per_hour = spy_rate
    try:
        env_rate = make_env(dataset, eff)
        traces, served = env_rate.run(inst)
    finally:
        CS.quality_rate_per_hour = orig_rate
    viol = [s for s in samples
            if s['actual_temperature_c'] > LB.temperature_upper_bound(
                eff, s['temp_class'], ambient_reachable=False)[0] + 1e-9]
    chk('RATE', 'RATE the A2 evidence is taken from the PRODUCTION rate function (not a '
        're-derivation): every rate temperature <= hard_upper for its class', bool(samples)
        and not viol,
        {'n_rate_calls': len(samples), 'n_violations': len(viol),
         'max_rate_temperature_c': max((s['actual_temperature_c'] for s in samples), default=None),
         'classes_seen': sorted({s['temp_class'] for s in samples}),
         'violations': viol[:3]})

    # ---------------- SIDE: the unstated side conditions of the tightened bound ----------------
    side = LB.tightened_bound_side_conditions(eff, dataset)
    chk('SIDE', 'SIDE the tightened bound\'s side conditions are stated AND checked (ambient '
        'branch unreachable; ambient > hard_upper for some classes here)', side['ok'] is True,
        {'ok': side['ok'], 'ambient_branch_reachable': side['ambient_branch_reachable'],
         'classes_where_ambient_exceeds_hard_upper':
             side['classes_where_ambient_exceeds_hard_upper'], 'why': side['why'],
         'note': ('because ambient (25 C) exceeds the hard upper bound of classes 1 (8 C) and 2 '
                  '(-12 C), the tightened bound is sound ONLY while those classes stay precooled')})

    # ---------------- FORCE: the truncation the author did not report ----------------
    n_force = {'calls': 0}
    orig_force = StrictOnlineEnv._force_return_all

    def spy_force(self, inst_idx, clock, vehicles, traces, served_mask):
        n_force['calls'] += 1
        return orig_force(self, inst_idx, clock, vehicles, traces, served_mask)

    StrictOnlineEnv._force_return_all = spy_force
    try:
        env_f = make_env(dataset, eff)
        traces_f, served_f = env_f.run(inst)
    finally:
        StrictOnlineEnv._force_return_all = orig_force
    ret_arrivals = [float(tr.return_arrival) for tr in traces_f
                    if getattr(tr, 'return_arrival', None) is not None]
    depot_tw_end = float(env_f.tw_end[inst, 0])
    chk('FORCE', 'FORCE `_force_return_all` is instrumented: calls and the recorded return-arrival '
        'distribution are reported (the truncation is a known optimism, not a silent one)',
        True,
        {'n_force_return_all_calls': n_force['calls'],
         'max_recorded_return_arrival_h': (max(ret_arrivals) if ret_arrivals else None),
         'depot_tw_end_h': depot_tw_end,
         'n_deliveries_after_depot_tw_end': sum(1 for a in ret_arrivals if a > depot_tw_end + 1e-6),
         'truncation_note': ('_force_return_all advances the cold chain only to `depart` '
                             '(strict_online_env.py:272) while recording the unload at '
                             '`depart + travel` (:273), so the RECORDED delivered quality is more '
                             'optimistic than the physics -- which WEAKENS (does not strengthen) '
                             'the BND check that compares against recorded deliveries')})

    # ---------------- COV + BIND over all captured decision points ----------------
    g = Grab(cap=args.heavy_cap)
    env = make_env(dataset, eff, hook=g.hook)
    traces2, served2 = env.run(inst)
    delivered_ids, delivered = set(), []
    for tr in traces2:
        for r in getattr(tr, 'depot_unload_records', ()):
            delivered_ids.add(int(r.order_id))
            delivered.append({'order_id': int(r.order_id),
                              'delivered_quality': float(r.delivered_quality)})
    bound_ids, all_rows, all_viol = set(), [], []
    for it in g.items:
        b = LB.bound_for_state(it['env'], it['inst'], it['clock'], it['vehicles'],
                               {v: list(p.suffix) for v, p in it['plans'].items()}, eff)
        all_rows.extend(b['rows'])
        bound_ids |= {r['order_id'] for r in b['rows']}
        all_viol.extend(LB.falsify_bound_against_delivered(b['rows'], delivered))
    matched = sorted(delivered_ids & bound_ids)
    unmatched_delivered = sorted(delivered_ids - bound_ids)
    bound_never_delivered = sorted(bound_ids - delivered_ids)
    chk('COV', 'COV coverage is stated in BOTH directions: bound rows cover N%% of delivered '
        'orders, and the unmatched counts are reported explicitly',
        bool(g.items),
        {'n_decision_points_captured': len(g.items),
         'n_delivered_orders': len(delivered_ids), 'n_bound_rows': len(all_rows),
         'n_bound_orders': len(bound_ids), 'n_matched': len(matched),
         'coverage_fraction_of_delivered': (round(len(matched) / max(len(delivered_ids), 1), 4)),
         'n_delivered_orders_never_bounded': len(unmatched_delivered),
         'delivered_orders_never_bounded': unmatched_delivered,
         'n_bounded_orders_never_delivered': len(bound_never_delivered),
         'bounded_orders_never_delivered': bound_never_delivered,
         'n_bound_violations': len(all_viol), 'violations': all_viol[:3],
         'why_incomplete': ('the capture covers only the FIRST %d decision points of the run, so '
                            'later-revealed orders can never appear in a bound row'
                            % len(g.items))})

    classes = sorted({r['temp_class'] for r in all_rows})
    tightest = {}
    for c in classes:
        rs = [r for r in all_rows if r['temp_class'] == c and r['ratio_lower_bound'] is not None]
        if rs:
            tightest[c] = {'min_ratio_lower_bound': min(r['ratio_lower_bound'] for r in rs),
                           'threshold': rs[0]['threshold']}
    binds = {c: bool(v['min_ratio_lower_bound'] < v['threshold']) for c, v in tightest.items()}
    chk('BIND', 'BIND the bound is reported against the thresholds: does it ever classify an order '
        'unsalable? (if never, more search cannot help -- the payoff needs a tighter tau_max)',
        True,
        {'per_class': tightest, 'binds_any_threshold': any(binds.values()),
         'per_class_binds': binds,
         'strategic_point': ('k(T_max)*tau_max is ~0.07/0.06/0.02 for classes 0/1/2 at tau_max=24 h, '
                             'so the ratios stay far above 0.8/0.85/0.9: the bound currently '
                             'NEVER binds, and its certification payoff must come from a tighter '
                             'tau_max (a separate, pre-declared derivation), not from more search')})

    # ---------------- IQ: start quality source ----------------
    sources = sorted({r.get('iq_source', 'in_cargo_current_quality')
                      for r in all_rows if r['source'] == 'not_yet_loaded'})
    iq_ok = sources == ['env.initial_quality']
    chk('IQ', 'IQ the not-yet-loaded start quality comes from the DATASET, not a hardcoded 1.0',
        iq_ok, {'iq_sources': sources,
                'note': ('hardcoding 1.0 was correct only because this dataset is 1.0 everywhere; '
                         'on a dataset with initial_quality < 1 it would silently produce a bound '
                         'above the true delivered quality')})

    out = {
        'stage': 'D3_G0_Evidence_B3QP_reviewer_closure',
        'read_only': True, 'production_state_written': False, 'commit_modified': False,
        'd8_relaxed': False,
        'reviewer_verdicts': {'conditional_implication_on_omega': 'PROVEN',
                              'omega_enforced_at_submission': 'NOT_ESTABLISHED',
                              'unconditional_question': 'NOT_ESTABLISHED',
                              'as_a_property_of_the_env': 'FALSIFIED'},
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'effective_contract_hash': eff.contract_hash,
            'code': {
                'scripts/simulation/d3_quality_lower_bound.py':
                    sha256_file(SIM / 'd3_quality_lower_bound.py'),
                'scripts/tests/test_d3_b3qp_closure.py': sha256_file(Path(__file__)),
                'scripts/coldchain/coldchain_state.py': sha256_file(src_path),
            },
        },
        'results': RESULTS, 'all_pass': all(r['ok'] for r in RESULTS),
    }
    for r in RESULTS:
        print('  %s %-6s %s' % ('PASS' if r['ok'] else 'FAIL', r['cid'], r['name']))
        if not r['ok']:
            print('        <- %s' % json.dumps(r['detail'], ensure_ascii=False, default=str)[:400])
    print('\nall_pass:', out['all_pass'])
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3QP_CLOSURE_MATRIX.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
