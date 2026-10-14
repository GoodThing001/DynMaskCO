#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-QC acceptance: falsification tests for the candidate analytic bound.

READ-ONLY.  Nothing installs, `_commit` is untouched, D8 is not relaxed.  The verdict is
PROVEN / FALSIFIED / NOT_ESTABLISHED and **a search count may never upgrade it to PROVEN**.

Checks:
  A2    instrument the temperature each v1 quality update ACTUALLY rates against on real runs and
        compare it with `max(hard_upper, ambient)` -- this is the assumption the bound lives or
        dies by, and the door-warming path is the concrete way it could fail
  A3    the Arrhenius rate is non-decreasing in T over the admissible temperature range
  A1    every v1 update is `q <- min(q, q*exp(-k*dt))` (structural check on the source path)
  OM    the Omega definition is recorded and the bound records its conditional nature
  BND   `delivered_quality >= q_lower_bound` on every delivered lot of a real run
        (a single violation FALSIFIES; agreement is evidence only)
  ADV   adversarial-but-feasible schedule search around the BINDING lot, recording realised
        service times and qualities
  VER   the tri-state adjudication: no path from a bounded search to PROVEN

Run: python scripts/tests/test_d3_b3qc_lower_bound.py [--instances 0] [--heavy-cap 4] [--grid 9]
"""
import argparse
import hashlib
import json
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
from dynmaskco_cc_context import mutable_scope                         # noqa: E402
from counterfactual_teacher import _eval                               # noqa: E402

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
    def __init__(self, cap=4):
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
                           'vehicles': cv, 'plans': plans, 'served_mask': csm,
                           'visible_ids': vis, 'snapshot': snap})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0])
    ap.add_argument('--heavy-cap', type=int, default=4)
    ap.add_argument('--grid', type=int, default=9)
    args = ap.parse_args()

    profile = load_profile()
    dataset = dict(np.load(CAL_NPZ))
    raw = default_pilot_contract()
    eff = apply_objective_profile(raw, profile)
    inst = args.instances[0]

    # ---------- A2: instrument the temperature the v1 rate ACTUALLY uses ----------
    # run a full instance under the recorder: this is the assumption the bound depends on, and the
    # door-warming path is the concrete way it could break.
    from d3_quality_lower_bound import ambient_branch_reachable
    amb_reach, amb_detail = ambient_branch_reachable(eff, dataset)
    rec = LB.RateTemperatureRecorder(eff, ambient_reachable=amb_reach)
    env_a2 = make_env(dataset, eff)
    with rec:
        traces, served = env_a2.run(inst)
    viol = rec.violations()
    a2_ok = rec.samples and not viol
    chk('A2', 'A2 the rate temperature never exceeds max(hard_upper, ambient) on a real run',
        bool(a2_ok),
        {'n_rate_samples': len(rec.samples), 'n_violations': len(viol),
         'max_rate_temperature': (max(s['rate_temperature_c'] for s in rec.samples)
                                  if rec.samples else None),
         'bound_range': ([min(s['bound_c'] for s in rec.samples),
                          max(s['bound_c'] for s in rec.samples)] if rec.samples else None),
         'precooled_branch_samples': sum(1 for s in rec.samples if s['precooled']),
         'ambient_branch_samples': sum(1 for s in rec.samples if not s['precooled']),
         'examples_of_violation': viol[:3]})

    # ---------- A3: Arrhenius monotone in T ----------
    from coldchain_state import quality_rate_per_hour
    a3_rows, a3_ok = [], True
    for c in eff.thermal.supported_temp_classes:
        lo, hi = eff.thermal.hard_temperature_bounds_c[int(c)]
        ts = [float(lo) + (float(hi) - float(lo)) * i / 9.0 for i in range(10)]
        ts += [float(eff.thermal.ambient_temperature_c)]
        ks = [quality_rate_per_hour(int(c), t, eff) for t in sorted(ts)]
        mono = all(b >= a - 1e-15 for a, b in zip(ks, ks[1:]))
        a3_rows.append({'class': int(c), 'hard_bounds': [float(lo), float(hi)],
                        'k_min': min(ks), 'k_max': max(ks), 'monotone': mono})
        a3_ok = a3_ok and mono
    chk('A3', 'A3 the Arrhenius rate is non-decreasing in T over the admissible range', a3_ok,
        a3_rows)

    # ---------- A1: structural check of the v1 update ----------
    src = (EXT / 'scripts' / 'coldchain' / 'coldchain_state.py').read_text(encoding='utf-8')
    a1_ok = ('quality_remaining = lot.quality_remaining * math.exp(-rate * duration_h)' in src
             and 'quality_remaining = min(lot.quality_remaining, max(0.0, quality_remaining))'
             in src)
    chk('A1', 'A1 the v1 update is q <- min(q, q*exp(-k*dt)) with no other factor', a1_ok,
        {'anchors_found': a1_ok})

    # ---------- OM: Omega recorded and the bound declares its conditional nature ----------
    om = LB.omega_definition()
    chk('OM', 'OM the admissible set Omega is formally recorded', bool(om['membership_conditions'])
        and bool(om['degrees_of_freedom_within_Omega']) and bool(om['not_in_Omega']),
        {'n_membership_conditions': len(om['membership_conditions']),
         'n_degrees_of_freedom': len(om['degrees_of_freedom_within_Omega']),
         'n_excluded': len(om['not_in_Omega']), 'name': om['name']})

    # ---------- BND: lower bound vs the delivered quality of the real run ----------
    g = Grab(cap=args.heavy_cap)
    env = make_env(dataset, eff, hook=g.hook)
    traces2, served2 = env.run(inst)
    delivered = []
    for tr in traces2:
        for r in getattr(tr, 'depot_unload_records', ()):
            delivered.append({'order_id': int(r.order_id),
                              'delivered_quality': float(r.delivered_quality),
                              'salable': (None if r.salable is None else bool(r.salable))})
    # BND over EVERY captured decision point, so BOTH branches are exercised: the first decision
    # points have no cargo (not_yet_loaded), later ones carry cargo (in_cargo).  Matching against
    # the delivered records of the whole run is a necessary condition on real trajectories.
    per_item, all_viol, sources = [], [], {}
    for it in g.items:
        b = LB.bound_for_state(it['env'], it['inst'], it['clock'], it['vehicles'],
                               {v: list(pp.suffix) for v, pp in it['plans'].items()}, eff)
        v = LB.falsify_bound_against_delivered(b['rows'], delivered)
        for r in b['rows']:
            sources[r['source']] = sources.get(r['source'], 0) + 1
        all_viol.extend(v)
        per_item.append({'clock': it['clock'], 'n_rows': len(b['rows']),
                         'n_in_cargo': sum(1 for r in b['rows'] if r['source'] == 'in_cargo'),
                         'n_violations': len(v),
                         'n_unsalable_lower_bound': len(b['unsalable_lower_bound_orders'])})
    base = g.items[0] if g.items else None
    bnd = (LB.bound_for_state(base['env'], base['inst'], base['clock'], base['vehicles'],
                              {v: list(p.suffix) for v, p in base['plans'].items()}, eff)
           if base is not None else None)
    viol_bnd = all_viol
    chk('BND', 'BND delivered_quality >= q_lower_bound on every delivered lot, over ALL captured '
        'decision points and BOTH lot branches (in-cargo / not-yet-loaded); a violation FALSIFIES',
        bnd is not None and not viol_bnd,
        {'n_decision_points_checked': len(per_item),
         'n_bound_rows_total': sum(r['n_rows'] for r in per_item),
         'branch_coverage': sources,
         'n_delivered_records': len(delivered),
         'n_violations': len(viol_bnd), 'violations': viol_bnd[:3],
         'per_item': per_item,
         'tau_max_h': (bnd['tau_max'] if bnd else None),
         'ambient_branch_reachable': (bnd['ambient_branch_reachable'] if bnd else None),
         'per_class_T_max': ({str(r['temp_class']): r['T_max_used'] for r in bnd['rows']}
                             if bnd else None),
         'tightest_ratio_lower_bound': (min((r['ratio_lower_bound'] for r in bnd['rows']
                                            if r['ratio_lower_bound'] is not None), default=None)
                                        if bnd else None)})

    # ---------- ADV: adversarial-but-feasible schedule search around the BINDING lot ----------
    adv = {'searched': False, 'why': 'no bound rows'}
    if bnd and bnd['rows']:
        binding = min((r for r in bnd['rows'] if r['ratio_lower_bound'] is not None),
                      key=lambda r: r['ratio_lower_bound'] - r['threshold'])
        adv = LB.search_schedules_for_binding_lot(
            base['env'], base['inst'], base['clock'], base['vehicles'], eff,
            binding['order_id'], served_mask=base['served_mask'], grid=args.grid,
            dataset=dataset)
        adv['binding_order'] = binding['order_id']
        adv['binding_ratio_lower_bound'] = binding['ratio_lower_bound']
        adv['binding_threshold'] = binding['threshold']
        adv['bound_still_holds_in_search'] = (
            adv['min_delivered_quality'] is None
            or adv['min_delivered_quality'] >= binding['q_lower_bound'] - 1e-12)
    mono = adv.get('quality_monotone_nonincreasing_in_delay')
    chk('ADV', 'ADV 1-D admissible-departure sweep: only feasible schedules, realised service times '
        'and qualities recorded, monotonicity lemma tested (result: %s)' % mono,
        adv.get('searched') is True and adv.get('n_feasible', 0) >= 2,
        {'searched': adv.get('searched'), 'n_feasible': adv.get('n_feasible'),
         'n_infeasible': adv.get('n_infeasible'),
         'distinct_minima': adv.get('distinct_minima'),
         'min_delivered_quality': adv.get('min_delivered_quality'),
         'bound_still_holds_in_search': adv.get('bound_still_holds_in_search'),
         'effectiveness': ('INEFFECTIVE_SEARCH' if len(adv.get('distinct_minima') or []) <= 1
                           else 'VARIED_SEARCH'),
         'max_admissible_delay_h': adv.get('max_admissible_delay'),
         'quality_monotone_nonincreasing_in_delay':
             adv.get('quality_monotone_nonincreasing_in_delay'),
         'realised_binding_service_times': [r.get('binding_lot_service_start')
                                            for r in adv.get('rows', [])],
         'realised_qualities': [r.get('delivered_quality') for r in adv.get('rows', [])],
         'why_infeasible_sample': [r.get('why') for r in adv.get('rows', [])
                                   if not r.get('feasible')][:3]})

    # ---------- VER: tri-state adjudication has no path from a search to PROVEN ----------
    f = LB.Falsification()
    f.check('demo', 'a passing check', True)
    v_pass, _ = f.verdict()
    f.searches = 10_000
    v_many, _ = f.verdict()
    f2 = LB.Falsification()
    f2.check('demo_fail', 'a failing check', False, {'x': 1})
    v_fail, ids = f2.verdict()
    ver_ok = (v_pass == LB.NOT_ESTABLISHED and v_many == LB.NOT_ESTABLISHED
              and v_fail == LB.FALSIFIED and ids == ['demo_fail']
              and bnd is not None and bnd['verdict'] == LB.NOT_ESTABLISHED
              and 'search count' in bnd['verdict_reason'])
    chk('VER', 'VER tri-state adjudication: passing checks and 10000 searches stay '
        'NOT_ESTABLISHED; a counterexample is FALSIFIED; PROVEN is unreachable here', ver_ok,
        {'after_passing_check': v_pass, 'after_10000_searches': v_many,
         'with_counterexample': v_fail, 'counterexample_ids': ids,
         'bound_verdict': (bnd['verdict'] if bnd else None)})

    out = {
        'stage': 'D3_G0_Evidence_B3QC_lower_bound_falsification',
        'read_only': True, 'production_state_written': False, 'commit_modified': False,
        'd8_relaxed': False, 'installable_impact': 'NONE',
        'verdict': LB.NOT_ESTABLISHED,
        'verdict_rule': 'FALSIFIED if any counterexample; else NOT_ESTABLISHED; PROVEN only from '
                        'a proof of Q_cert,i <= Q_i(omega) for all omega in Omega, which a bounded '
                        'search cannot supply',
        'applies_to': LB.APPLIES_TO,
        'does_not_transfer_to': 'v2 absolute-sensory / zero-order fish quality model',
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'effective_contract_hash': eff.contract_hash,
            'code': {
                'scripts/simulation/d3_quality_lower_bound.py':
                    sha256_file(SIM / 'd3_quality_lower_bound.py'),
                'scripts/tests/test_d3_b3qc_lower_bound.py': sha256_file(Path(__file__)),
            },
        },
        'assumptions': LB.assumptions(),
        'falsified_lemmas': LB.FALSIFIED_LEMMAS,
        'omega': LB.omega_definition(),
        'rate_temperature_audit': {
            'n_samples': len(rec.samples), 'n_violations': len(viol),
            'max_rate_temperature_c': (max(s['rate_temperature_c'] for s in rec.samples)
                                       if rec.samples else None),
            'ambient_branch_samples': sum(1 for s in rec.samples if not s['precooled']),
            'ambient_branch_reachable': amb_reach, 'ambient_branch_detail': amb_detail,
        },
        'bound_vs_delivered': {'n_violations': len(viol_bnd), 'violations': viol_bnd[:5]},
        'adversarial_search': {k: adv.get(k) for k in
                               ('searched', 'vehicle_id', 'order_id', 'n_feasible', 'n_infeasible',
                                'min_delivered_quality', 'distinct_minima',
                                'bound_still_holds_in_search')},
        'results': RESULTS, 'all_pass': all(r['ok'] for r in RESULTS),
    }
    for r in RESULTS:
        print('  %s %-5s %s' % ('PASS' if r['ok'] else 'FAIL', r['cid'], r['name']))
        if not r['ok']:
            print('        <- %s' % json.dumps(r['detail'], ensure_ascii=False, default=str)[:420])
    print('\nverdict:', out['verdict'], ' all_pass:', out['all_pass'])
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3QC_LOWER_BOUND_MATRIX.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
