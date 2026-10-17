#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S-8 ERRATA CONTROLS -- the analytic distance-multiplier ceilings (K-1 .. K-8).

Two mechanisms in `generate_coldchain_data.py` make the previous ceilings unsound:

  K-1  the asymmetric noise divides by the TRANSPOSED draw, so the supremum pairs 1.35 with 0.85
  K-2  with the rush-hour factor the ceiling is 2.526471, not 2.090741, so the old travel bound
       2.956992 is NOT an upper bound of the domain
  K-3  edge pairs are drawn WITH replacement, so one edge can be scaled by 5^k, not by 5
  K-4  both mechanisms compose multiplicatively
  K-5  the forced-return leg uses the DIRECTED depot in-edge `dist_mat[i, 0]`; an amplified in-edge
       is exercised through the real `_force_return_all` path
  K-6  an undeclared k, or too small an analytic ceiling, must be NOT_CERTIFIABLE / REFUSED
  K-7  the per-instance measurements of the registered splits stand independently
  K-8  the effect on T_clock_sup and on the time-error budget

Usage: python scripts/tests/test_q3_g2d_analytic_ceiling.py --out <json path>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'coldchain'))
sys.path.insert(0, str(EXT / 'scripts' / 'm1b_adapter'))
sys.path.insert(0, str(EXT / 'scripts' / 'simulation'))

import coldchain_contract as CC                                            # noqa: E402
import q3_g2c_travel_domain_experimental as G2C                            # noqa: E402  (retracted constants)
import q3_g2d_analytic_ceiling_experimental as G2D                         # noqa: E402

ROWS = []
LIMITS = ('exact rational arithmetic + two synthetic episodes on the CPU; generator and frozen data '
          'untouched; no training, no GPU, no rollout, no production wiring')


def ck(cid, name, ok, detail=None, conclusion='', grading='PROVED'):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'grading': grading, 'limits': LIMITS})
    print('[%s] %-5s %s' % (st, cid, name[:94]))


def try_call(fn, *a, **k):
    try:
        return {'ok': True, 'value': fn(*a, **k)}
    except G2D.Refusal as exc:
        return {'ok': False, 'code': exc.code, 'detail': exc.detail}


class ServeOne(object):
    def __init__(self):
        self.done = False

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        free = [c for c in visible_ids if not served_mask[c]]
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            v.mutable_suffix = [] if self.done or not free else [int(free[0])]
        if free:
            self.done = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import numpy as np
    contract = CC.default_pilot_contract()
    speed = 1.0
    sqrt2 = math.sqrt(2.0)

    # ============================================================ K-1 the asymmetric supremum
    a_max = G2D.ASYM_MAX_EXACT
    old = G2D.ASYM_MAX_RETRACTED_EXACT
    # the concrete pair the supremum is attained at
    x_ij, x_ji = Fraction('1.35'), Fraction('0.85')
    attained = (x_ij + 1 / (x_ji + Fraction('1e-8'))) / 2
    ok1 = (attained == a_max and a_max > old
           and abs(float(a_max) - 1.263235287) < 1e-9
           and abs(float(old) - 1.045370370) < 1e-9
           and float(a_max / old) > 1.20)
    ck('K-1', 'the asymmetric ceiling is 1.263235287 (not 1.045370): the generator divides by the '
       'TRANSPOSED draw, so the supremum pairs X[i,j]=1.35 with X[j,i]=0.85, two independent draws',
       ok1,
       {'corrected_exact': str(a_max), 'corrected': float(a_max),
        'retracted_exact': str(old), 'retracted': float(old),
        'understatement_factor': float(a_max / old),
        'attaining_pair': {'X[i,j]': '1.35', 'X[j,i]': '0.85'},
        'source': 'generate_coldchain_data.py:84-85'},
       conclusion='the old expression paired a draw with its own reciprocal')

    # ============================================================ K-2 the old bound is unsound
    with_rush_new = a_max * G2D.RUSH_MAX
    with_rush_old = old * G2D.RUSH_MAX
    old_bound = sqrt2 * float(with_rush_old)
    new_bound = G2D.travel_bound_analytic(('asymmetric_transit',), speed)
    # a legal instance: corners of the unit square (euclid = sqrt2) at the maximum multipliers
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    eu = np.array([[0.0, sqrt2], [sqrt2, 0.0]])
    legal = eu * float(with_rush_new)
    ok2 = (float(with_rush_new) > float(with_rush_old)
           and legal.max() > old_bound
           and new_bound['travel_bound'] >= legal.max())
    ck('K-2', 'the retracted bound is NOT an upper bound of the domain: a legal instance at the '
       'corrected supremum reaches 3.572969, above the old 2.956992; the corrected analytic bound '
       'covers it',
       ok2,
       {'corrected_with_rush': float(with_rush_new), 'retracted_with_rush': float(with_rush_old),
        'old_travel_bound': old_bound, 'new_travel_bound': new_bound['travel_bound'],
        'legal_instance_max_edge': float(legal.max()),
        'retracted_bound_would_be_violated_by': float(legal.max() - old_bound)},
       conclusion='an empirical case below the old bound could not certify it')

    # ============================================================ K-3 repeated perturbation
    twice = G2D.PERTURB_HIT_MAX ** 2
    legal2 = np.array([[0.0, sqrt2 * float(twice)], [sqrt2 * float(twice), 0.0]])
    old_perturb_bound = sqrt2 * 5.0
    new_k1 = G2D.travel_bound_analytic(('long_tail_perturbation',), speed,
                                       perturb_multiplicity=1)['travel_bound']
    new_k2 = G2D.travel_bound_analytic(('long_tail_perturbation',), speed,
                                       perturb_multiplicity=2)['travel_bound']
    ok3 = (float(twice) == 25.0 and legal2.max() > old_perturb_bound
           and legal2.max() > new_k1 and new_k2 >= legal2.max())
    ck('K-3', 'edge pairs are drawn WITH replacement, so one ordered pair can be multiplied twice '
       '(5^2 = 25): a legal instance reaches 35.355, above both the old 7.071 and the k=1 ceiling; '
       'only the declared-k model covers it',
       ok3,
       {'draw_sampling': 'rng.integers(size=n) pairs, no de-duplication (:103-106)',
        'multiplier_for_k2': float(twice), 'legal_instance_max_edge': float(legal2.max()),
        'old_bound_(k assumed 1)': old_perturb_bound, 'new_k1': new_k1, 'new_k2': new_k2},
       conclusion='5^k, and k is not bounded by the source')

    # ============================================================ K-4 composition
    combined_mult = with_rush_new * twice
    combined_bound = G2D.travel_bound_analytic(('asymmetric_transit', 'long_tail_perturbation'),
                                               speed, perturb_multiplicity=2)
    legal3 = np.array([[0.0, sqrt2 * float(combined_mult)], [sqrt2 * float(combined_mult), 0.0]])
    ok4 = (combined_bound['travel_bound'] >= legal3.max()
           and combined_bound['multiplier_ceiling'] > float(with_rush_new))
    ck('K-4', 'the two mechanisms compose: the multiplier ceiling is (asym x rush) x 5^k and the '
       'combined analytic bound covers an instance built at the product',
       ok4,
       {'combined_multiplier': float(combined_mult),
        'combined_bound': combined_bound['travel_bound'],
        'legal_instance_max_edge': float(legal3.max()),
        'contributions': combined_bound['contributions']},
       conclusion='composition is multiplicative, so the ceilings must be too')

    # ============================================================ K-5 the directed depot edge
    import numpy as _np
    coords_env = _np.array([[[0.0, 0.0], [0.9, 0.0]]], dtype=_np.float32)
    dm = _np.array([[[0.0, 0.9], [3.0, 0.0]]], dtype=_np.float32)   # only the IN-edge is amplified
    data = {
        'coords': coords_env,
        'dist_mat': dm,
        'demands': _np.array([[0.0, 5.0]], dtype=_np.float32),
        'tw_start': _np.array([[0.0, 23.5]], dtype=_np.float32),
        'tw_end': _np.array([[24.0, 24.0]], dtype=_np.float32),
        'service_time': _np.array([[0.0, 0.4]], dtype=_np.float32),
        'reveal_time': _np.zeros((1, 2), dtype=_np.float32),
        'temp_class': _np.zeros((1, 2), dtype=_np.int32),
        'initial_quality': _np.ones((1, 2), dtype=_np.float32),
    }
    from strict_online_env import StrictOnlineEnv
    env = StrictOnlineEnv(data, capacity=contract.operational.shared_vehicle_capacity,
                          num_vehicles=1, replanner=ServeOne(), coldchain_contract=contract)
    clocks = []
    orig = env._advance_coldchain_to

    def counting(v, clock):
        clocks.append(float(clock))
        return orig(v, clock)

    env._advance_coldchain_to = counting
    env.run(0)
    max_clock = max(clocks) if clocks else None
    witness = G2D.per_instance_witness(dm[0], coords_env[0])
    per_inst = G2D.travel_bound_per_instance(witness, speed)
    eu_bound = sqrt2
    ok5 = (max_clock is not None and max_clock > 24.0 + eu_bound
           and witness['depot_column_max'] == 3.0
           and per_inst['travel_bound'] >= max_clock - 24.0)
    ck('K-5', 'the forced-return clock uses the DIRECTED depot in-edge: with only `dist_mat[1, 0]` '
       'amplified to 3.0 (and every other edge Euclidean) the real `_force_return_all` path advances '
       'to 26.9, which is above horizon+sqrt(2) = 25.414 -- so a bound on customer-customer edges '
       'does not bound this path, while the depot-column witness does',
       ok5,
       {'dist_mat': dm.tolist(), 'observed_clocks': clocks, 'max_clock': max_clock,
        'euclidean_ceiling': 24.0 + eu_bound,
        'depot_column_max': witness['depot_column_max'],
        'per_instance_bound': per_inst['travel_bound'],
        'per_instance_leg_bound': per_inst['depot_leg_bound'],
        'mechanism': 'strict_online_env.py:273 -> :664 -> :304'},
       conclusion='the relevant maximum is the depot column, not the pair maximum')

    # ============================================================ K-6 refusals
    undeclared = try_call(G2D.travel_bound_analytic, ('long_tail_perturbation',), speed)
    undeclared_q = G2D.qualify_s8(models=('long_tail_perturbation',), tw_speed=speed,
                                  perturb_multiplicity=None)
    # a declared k=1 ceiling is sqrt(2)*5 = 7.071 h, so T_clock_sup = 31.07 h; an instance whose
    # return leg is 5^2-amplified (35.355) lands at 59.355 h, far above it -> must be REFUSED
    k1_ceiling = G2D.travel_bound_analytic(('long_tail_perturbation',), speed,
                                           perturb_multiplicity=1)['travel_bound']
    too_small = G2D.qualify_s8(models=('long_tail_perturbation',), tw_speed=speed,
                               perturb_multiplicity=1, declared_horizon=24.0,
                               observed_max_clock=24.0 + float(twice) * sqrt2)
    unbounded = try_call(G2D.travel_bound_analytic, ('external_undeclared',), speed)
    ok6 = (not undeclared['ok'] and undeclared['code'] == 'PERTURB_MULTIPLICITY_UNDECLARED'
           and undeclared_q['verdict'] == 'NOT_CERTIFIABLE'
           and too_small['verdict'] == 'REFUSED'
           and not unbounded['ok'] and unbounded['code'] == 'DISTANCE_MODEL_UNBOUNDED')
    ck('K-6', 'an undeclared k is `NOT_CERTIFIABLE` (never silently k=1); a declared k=1 ceiling '
       '(T_clock_sup = 31.07 h) is REFUSED once an instance needs a 5^2-amplified return leg '
       '(59.36 h observed); an external matrix stays `DISTANCE_MODEL_UNBOUNDED`',
       ok6,
       {'undeclared_k': undeclared, 'undeclared_verdict': undeclared_q['verdict'],
        'k1_ceiling_h': k1_ceiling,
        'k1_T_clock_sup': 24.0 + k1_ceiling,
        'observed_for_the_k2_instance': 24.0 + float(twice) * sqrt2,
        'k1_too_small': {'verdict': too_small['verdict'],
                         'refusals': too_small['refusals']},
        'external': unbounded},
       conclusion='the ceiling must come with a declared domain, and be big enough')

    # ============================================================ K-7 the per-instance witness
    splits = sorted(EXT.glob('data/m0_scale/*.npz'))
    in_scope, offenders, per_split = 0, [], []
    for path in splits:
        d = np.load(path, allow_pickle=True)
        for i in range(int(d['coords'].shape[0])):
            w = G2D.per_instance_witness(d['dist_mat'][i], d['coords'][i])
            in_scope += 1
            if not w['euclidean_consistent']:
                offenders.append({'file': path.name, 'inst': i,
                                  'max_abs_dev': w['max_abs_dev_from_euclidean']})
            if i == 0:
                per_split.append({'file': path.name, 'max_edge': w['max_edge'],
                                  'max_multiplier_observed': w['max_multiplier_observed'],
                                  'depot_column_max': w['depot_column_max'],
                                  'symmetric': w['symmetric'],
                                  'bound': G2D.travel_bound_per_instance(w, speed)['travel_bound']})
    ok7 = (in_scope == 96 and not offenders)
    ck('K-7', 'the per-instance witness is INDEPENDENT of the retracted ceiling and still holds: all '
       '96 registered instances are Euclidean to float32 rounding, so the measured multiplier is 1.0 '
       'and the finite-domain conclusion is untouched', ok7,
       {'n_instances': in_scope, 'offenders': offenders[:5], 'examples': per_split,
        'separation': 'the analytic ceiling is retracted; the measurement of these instances is not'},
       conclusion='a retraction does not erase a measurement')

    # ============================================================ K-8 budget effect
    old_T = 24.0 + old_bound
    new_T = 24.0 + new_bound['travel_bound']


    def ulp_ceil(x):
        v = Fraction(float(x))
        k, p = 0, Fraction(1)
        while p < v:
            p *= 2
            k += 1
        return p / Fraction(2 ** 52)


    same_binade = ulp_ceil(old_T) == ulp_ceil(new_T)
    n_max = 20003
    skip = Fraction(1, 10 ** 9) + ulp_ceil(new_T) / 2
    rounding = n_max * ulp_ceil(new_T) / 2
    slack_new = float(skip + rounding)
    skip_old = Fraction(1, 10 ** 9) + ulp_ceil(old_T) / 2
    rounding_old = n_max * ulp_ceil(old_T) / 2
    slack_old = float(skip_old + rounding_old)
    ok8 = (new_T > old_T and abs(slack_new - slack_old) < 1e-24
           and same_binade is True)
    ck('K-8', 'budget effect: the asymmetric-domain clock bound rises 26.957 -> 27.573 h, which is '
       'the SAME binade, so the ulp and therefore the slack are numerically unchanged '
       '(1.0710684844307253e-09 h) -- the value and the identity digest change, the number does not',
       ok8,
       {'T_clock_sup_old': old_T, 'T_clock_sup_new': new_T,
        'ulp_old': str(ulp_ceil(old_T)), 'ulp_new': str(ulp_ceil(new_T)),
        'same_binade': same_binade, 'slack_old': slack_old, 'slack_new': slack_new,
        'digest_effect': 'T_clock_sup is bound into the identity, so the digest changes even though '
                         'the slack does not',
        'registered_domain': 'unaffected: it is Euclidean (K-7), so no ceiling change reaches it'},
       conclusion='retracting a ceiling changes a value and a digest, not necessarily a number')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G2D_FAIL' if n_fail else 'G2D_PASS'
    payload = {
        'schema': 'q3_g2d_analytic_ceiling_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'PRODUCTION_NOT_QUALIFIED': True,
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'retracts': {'scripts/coldchain/q3_g2c_travel_domain_experimental.py':
                     ['ASYM_MAX (1.045370 -> 1.263235287)',
                      'PERTURB_MAX (5 -> 5^k with declared k)',
                      'the derived bounds 2.956992 / 7.071068']},
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'retraction_table': G2D.retraction_table(),
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
