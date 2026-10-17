#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 STRICT FLOATING-POINT QUALIFICATION CONTROLS (DIAGNOSTIC, not for production).

Every row records input, expectation, measurement, conclusion, limits and source sha256, and the
file fails (non-zero exit) if any row fails.  The point of the set is that the PRE-FIX module must
FAIL where it is wrong: SFC-04 shows a legal executor state at which the pre-fix 4-ulp guard is below
a rigorous upper bound even with a perfectly correct exponential, and SFC-06 shows that the pre-fix
count bound is below the executor for large door counts.  A control set that only proved "the
functions return" would pass a broken implementation and is therefore not used here.

Usage: python scripts/tests/test_q3_strict_float_bounds.py --out <json path>
"""
from __future__ import annotations

import argparse
import decimal
import hashlib
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts/coldchain'))

from coldchain_contract import ColdChainContract, ColdChainContractV2   # noqa: E402
import coldchain_state as cs                                            # noqa: E402
import q3_strict_float_experimental as SF                               # noqa: E402
import q3_thermal_interval_experimental as PRE                          # noqa: E402

CONTRACT = ColdChainContractV2()
CONTRACT_V1 = ColdChainContract()
TH = CONTRACT.thermal
ZONES = SF.zone_thermals_exact(CONTRACT)
PRE_ZONES = PRE.zone_thermals(CONTRACT)          # the pre-fix module's own zone objects
N_ZONES = len(ZONES)
EPS = float(cs._EPS)
ROWS = []
LIMITS = ('synthetic fixed-input controls only; no instance data, no training, no rollout, no GPU; '
          'a passing row is a DIAGNOSTIC, never a gain or a certification')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-6s %s' % ('PASS' if ok else 'FAIL', cid, name[:96]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ executor helpers (diagnostic)
def executor_doors(zone, dt_h, n_doors):
    """The executor's internal pair (`_advance_temperatures` then `_apply_door_event`).

    The same pair is validated bit-for-bit against the PUBLIC api (`transition_segment`) by
    `q3_theorem_b_audit.py` N-02/N-03.  The public api is used here only while the pickup stays inside
    `shared_vehicle_capacity`; the internal pair is needed beyond it (the reference audit's own
    PATH-I).
    """
    temps = tuple(float(t) for t in TH.target_temperature_c)
    for _ in range(int(n_doors)):
        temps, _e = cs._advance_temperatures(temps, float(dt_h), (True,) * N_ZONES, CONTRACT)
        temps, _h, _r = cs._apply_door_event(temps, (True,) * N_ZONES, int(zone), CONTRACT)
    return temps[int(zone)]


def decimal_exp_neg(x):
    ctx = decimal.Context(prec=60)
    with decimal.localcontext(ctx):
        return decimal.Decimal(-decimal.Decimal(repr(float(x)))).exp()


def decimal_contains(lo, hi, x, prec=400):
    """Compare an exact rational enclosure against a correctly-rounded decimal reference.

    Two traps this function exists to avoid:
      * the argument must be the EXACT binary value of x, so it is built from the rational
        numerator/denominator, not from `repr(float(x))` (which is a nearby decimal number);
      * the divisions must be done far above the enclosure width, otherwise their own rounding
        breaks containment.
    """
    ctx = decimal.Context(prec=prec)
    with decimal.localcontext(ctx):
        arg = decimal.Decimal(x.numerator) / decimal.Decimal(x.denominator)
        ref = (-arg).exp()
        flo = decimal.Decimal(lo.numerator) / decimal.Decimal(lo.denominator)
        fhi = decimal.Decimal(hi.numerator) / decimal.Decimal(hi.denominator)
        return bool(flo <= ref <= fhi), ref, flo, fhi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    src = {
        'scripts/coldchain/q3_strict_float_experimental.py':
            sha256(EXT / 'scripts/coldchain/q3_strict_float_experimental.py'),
        'scripts/coldchain/q3_thermal_interval_experimental.py':
            sha256(EXT / 'scripts/coldchain/q3_thermal_interval_experimental.py'),
        'scripts/coldchain/coldchain_state.py':
            sha256(EXT / 'scripts/coldchain/coldchain_state.py'),
        'scripts/coldchain/coldchain_contract.py':
            sha256(EXT / 'scripts/coldchain/coldchain_contract.py'),
        'scripts/tests/test_q3_strict_float_bounds.py': sha256(Path(__file__)),
    }

    # ================================================================ SFC-01 exp enclosure
    xs = [Fraction(v) for v in (0.0, 1e-9, 1e-4, 0.01, 0.1, 0.5, 0.96, 1.0, 2.4, 3.0, 4.8, 5.0,
                                6.0, 12.0)]
    worst_margin = None
    all_contain = True
    widths = {}
    for x in xs:
        lo, hi = SF.exp_neg_enclosure(x)
        inside, _ref, _flo, _fhi = decimal_contains(lo, hi, x)
        all_contain = all_contain and inside
        width = float((hi - lo) / hi) if hi > 0 else 0.0
        widths[str(float(x))] = width
        worst_margin = width if worst_margin is None else max(worst_margin, width)
    ck('SFC-01', 'the rational exp enclosure contains the 60-digit decimal reference for every '
       'sampled exponent, and its relative width stays below binary64 resolution over the whole '
       'declared domain and beyond',
       all_contain and worst_margin < 1e-15,
       {'n_x': len(xs), 'worst_relative_width': worst_margin, 'per_x_width': widths,
        'domain_bound': 'x = h*dt <= 0.2*24 = 4.8 for the default contract',
        'reference': 'decimal.Context(prec=60).exp() (correctly rounded, independent of math.exp)',
        'what_it_proves': 'lemma L-A on the sampled exponents; the general statement is analytic',
        'what_it_does_not_prove': 'anything about the platform libm'},
       conclusion='the enclosure is rigorous and tight enough that no ulp guess is needed on the '
                  'R-level (model) statements')

    # ================================================================ SFC-02 libm is NOT certified
    random.seed(20261017)
    ks = []
    for _ in range(300):
        x = Fraction(random.uniform(0.0, 10.0))
        lo, hi = SF.exp_neg_enclosure(x, terms=200)
        mid = (lo + hi) / 2
        libm = Fraction(math.exp(-float(x)))
        ks.append(abs(float(libm - mid)) / float(mid))
    ks.sort()
    kappa_measured = ks[-1]
    ck('SFC-02', 'the platform math.exp deviates from the rigorous enclosure by ~1e-16 relative but '
       'IEEE-754 gives NO bound, so the measured value cannot be promoted to a certificate',
       kappa_measured > 0 and kappa_measured < 2 ** -50,
       {'n_samples': len(ks), 'max_relative_deviation': kappa_measured,
        'max_deviation_in_ulps_of_one': kappa_measured / 2.220446049250313e-16,
        'p99': ks[int(0.99 * len(ks))], 'median': ks[len(ks) // 2],
        'label': 'DIAGNOSTIC_ONLY',
        'why_not_a_proof': 'the value is measured on THIS platform/version; libm is not required by '
                           'IEEE-754 to be correctly rounded, and the executor may run elsewhere'},
       conclusion='an F-level (executor float) claim must state kappa explicitly; the module refuses '
                  'to produce one without it')

    # prove the refusal is real
    refused = False
    try:
        SF.executor_segment_enclosure(ZONES[1], 4.0, 0.1, False, None)
    except SF.Refusal as exc:
        refused = exc.code == 'EXP_ERROR_BOUND_NOT_DECLARED'
    ck('SFC-03', 'an F-level enclosure without a declared exp error bound is REFUSED instead of '
       'silently returning a number',
       refused, {'exception_code': 'EXP_ERROR_BOUND_NOT_DECLARED'},
       conclusion='no caller can obtain an executor-float claim without stating its assumption')

    # ================================================================ SFC-04 the 4-ulp guard fails
    DTS = (1e-9 * 1.0000001, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1,
           0.5, 1.0, 2.0, 6.0, 12.0, 24.0)
    failures = []
    for z, pz in zip(ZONES, PRE_ZONES):
        tau, d = z.target_c, float(z.delta_real)
        for t0 in (tau, tau + 0.25 * d, tau + 0.5 * d, tau + 0.75 * d, tau + d):
            for dt in DTS:
                up = Fraction(PRE.advance_upward(pz, t0, dt))
                f_hi = SF.executor_segment_enclosure(z, t0, dt, False, Fraction(0))[1]
                r_hi = SF.real_psi_enclosure(z, Fraction(t0), Fraction(dt))[1]
                failures.append({'zone': z.zone, 't0': t0, 'dt_h': dt,
                                 'deficit_vs_F_kappa0_ulp': SF.ulp_gap(up, f_hi),
                                 'deficit_vs_R_ulp': SF.ulp_gap(up, r_hi)})
    worst = max(failures, key=lambda r: r['deficit_vs_F_kappa0_ulp'])
    n_bad = sum(1 for r in failures if r['deficit_vs_F_kappa0_ulp'] > 0)
    ck('SFC-04', 'the pre-fix 4-ulp guard is NOT an upper bound: at legal executor states it is below '
       'the rigorous F-level bound even with a PERFECTLY correct exponential (kappa = 0)',
       n_bad > 0 and worst['deficit_vs_F_kappa0_ulp'] > 0,
       {'n_grid': len(failures), 'n_insufficient': n_bad, 'worst_case': worst,
        'reading': 'a positive deficit means `advance_upward` returns less than a guaranteed upper '
                   'bound on what the executor will store; the outward-rounding claim is false',
        'label': 'MATHEMATICAL_STATEMENT + PROVED_BY_ENCLOSURE (no libm assumption used here)'},
       conclusion='nextafter x4 cannot be upgraded to a certificate of the real-value upper bound')

    # ================================================================ SFC-05 the implicit kappa
    table = []
    for z in ZONES:
        tau = z.target_c
        d = float(z.delta_real)
        worst_k = None
        for t0 in (tau, tau + 0.5 * d, tau + d):
            for dt in (1e-9 * 1.0000001, 1e-7, 1e-5, 1e-3, 1e-2, 0.1, 1.0, 6.0, 24.0):
                k = SF.required_exp_error_budget(z, t0, dt)
                if k is None:
                    continue
                if worst_k is None or k < worst_k[0]:
                    worst_k = (k, t0, dt)
        table.append({'zone': z.zone,
                      'min_kappa': float(worst_k[0]),
                      'min_kappa_ulps_of_one': float(worst_k[0]) / 2.220446049250313e-16,
                      'at': {'t0': worst_k[1], 'dt_h': worst_k[2]}})
    ck('SFC-05', 'the 4-ulp shift is exactly equivalent to declaring kappa <= kappa*, and kappa* is '
       'as low as 0 ulp in two of the three zones',
       any(r['min_kappa_ulps_of_one'] < 1.0 for r in table),
       {'per_zone': table,
        'measured_platform_kappa': kappa_measured,
        'reading': 'the pre-fix module never states kappa, so the guard silently asserts one; in '
                   'zones 1 and 2 that assertion is false even for an exact exponential'},
       conclusion='quantifying kappa is a mandatory input, not a cosmetic margin')

    # ================================================================ SFC-06 count bound unsound
    KAPPA = Fraction(1, 2 ** 52)
    rows = []
    for z, pz in zip(ZONES, PRE_ZONES):
        for n in (1, 9, 20, 45, 354, 360):
            ex = executor_doors(z.zone, 0.0, n)
            pre = PRE.door_count_bound(pz, n)
            strict = float(SF.count_bound_strict(z, n))
            rows.append({'zone': z.zone, 'n_doors': n, 'executor_c': ex,
                         'pre_fix_bound_c': pre, 'pre_fix_deficit_ulps': SF.ulp_gap(
                             Fraction(pre), Fraction(ex)),
                         'strict_bound_c': strict, 'strict_dominates': strict >= ex})
    bad = [r for r in rows if r['pre_fix_deficit_ulps'] > 0]
    ck('SFC-06', 'the pre-fix door-count bound tau+N*delta is BELOW the executor for N >= 20 because '
       'it ignores the N accumulated roundings; the corrected bound dominates everywhere',
       len(bad) > 0 and all(r['strict_dominates'] for r in rows),
       {'n_rows': len(rows), 'n_pre_fix_unsound': len(bad),
        'worst': max(rows, key=lambda r: r['pre_fix_deficit_ulps']),
        'rows': rows,
        'reading': 'the pre-fix bound rounds ONE expression and shifts 4 ulp, while the executor '
                   'performs N separate additions; lemma L-E in the module docstring',
        'label': 'EXECUTOR_REPRODUCED + PROVED (lemma L-E)'},
       conclusion='P2 must be implemented with an accumulation-aware outward rounding')

    # ================================================================ SFC-07 razor-thin verdict
    z0, pz0 = ZONES[0], PRE_ZONES[0]
    n = 360
    ex = executor_doors(0, 0.0, n)
    pre = PRE.door_count_bound(pz0, n)
    upre = max(PRE.u_recursion(pz0, [(0.0, True)] * n))
    tight = min(upre, pre)
    ck('SFC-07', 'at the exact door count that crosses the zone-0 hard bound the pre-fix verdict is '
       'correct only by the 4-ulp shift itself, while its stated bound is far BELOW the executor',
       ex > z0.hard_high_c and tight > z0.hard_high_c
       and SF.ulp_gap(Fraction(tight), Fraction(ex)) > 100,
       {'zone': 0, 'n_doors': n, 'executor_c': ex, 'hard_high_c': z0.hard_high_c,
        'pre_fix_tight_bound_c': tight,
        'tight_minus_hard_high_ulps': SF.ulp_gap(Fraction(tight), Fraction(z0.hard_high_c)),
        'executor_minus_tight_ulps': SF.ulp_gap(Fraction(tight), Fraction(ex)),
        'reading': 'the verdict "EXCEEDS" rests on 4 ulp (1.4e-14 C) of accident, while the number '
                   'reported as the upper bound understates the executor by >160 ulp'},
       conclusion='the design is not merely unproven, it is margin-free: correctness of this verdict '
                  'is a numerical coincidence of the 4-ulp shift')

    # ================================================================ SFC-08 malformed inputs
    old_raised, new_refused = [], []
    zone1 = ZONES[1]
    cases = [
        ('nan_duration', {0: [(1.0, False)], 1: [(float('nan'), True)], 2: [(1.0, False)]},
         CONTRACT, 'zone_segments'),
        ('inf_duration', {0: [(1.0, False)], 1: [(float('inf'), True)], 2: [(1.0, False)]},
         CONTRACT, 'zone_segments'),
        ('string_duration', {0: [(1.0, False)], 1: [('0.01', True)], 2: [(1.0, False)]},
         CONTRACT, 'zone_segments'),
        ('three_tuple', {0: [(1.0, False)], 1: [(0.01, True, 9)], 2: [(1.0, False)]},
         CONTRACT, 'zone_segments'),
        ('negative_duration', {0: [(1.0, False)], 1: [(-0.01, True)], 2: [(1.0, False)]},
         CONTRACT, 'zone_segments'),
        ('segments_container_is_a_dict', {'0': [(0.01, True)]}, CONTRACT, 'container'),
        ('contract_none', {0: [(1.0, False)], 1: [(0.01, True)], 2: [(1.0, False)]},
         None, 'zone_segments'),
    ]
    for name, segs, contract, kind in cases:
        try:
            rec = PRE.certify_interval(contract, contract_identity='c', state_identity='s',
                                       segments_by_zone=segs,
                                       active_zone_mask=(True,) * N_ZONES)
            old_raised.append({'case': name, 'outcome': 'returned',
                               'verdict': rec.get('experimental_verdict')})
        except BaseException as exc:                                      # noqa: BLE001
            old_raised.append({'case': name, 'outcome': 'RAISED',
                               'exception': type(exc).__name__})
        code = 'NO_REFUSAL'
        try:
            if kind == 'container':
                SF.require_segments(segs, 'case')
            elif contract is None:
                SF.zone_thermals_exact(None)
            else:
                for seg in SF.require_segments(segs.get(1, []), 'case'):
                    dt, _o = SF.segment_pair(seg, 'case')
                    SF.executor_segment_enclosure(zone1, zone1.target_c, float(dt), True, KAPPA)
        except SF.Refusal as exc:
            code = exc.code
        new_refused.append({'case': name, 'code': code})
    raw_exceptions = [r for r in old_raised if r['outcome'] == 'RAISED']
    ck('SFC-08', 'the pre-fix module raises raw TypeError/ValueError/AttributeError on malformed '
       'input instead of a deterministic refusal; the strict module returns a coded refusal for '
       'every case',
       len(raw_exceptions) >= 3 and all(r['code'] != 'NO_REFUSAL' for r in new_refused),
       {'pre_fix': old_raised, 'strict': new_refused,
        'defect': 'a raw exception is not a documented refusal and the refusal list already recorded '
                  'by the pre-fix code does NOT stop the recursion from consuming the bad value',
        'label': 'DIAGNOSTIC (observed behaviour)'},
       conclusion='malformed segment data must be validated before the recursion, not reported '
                  'after it')

    # ================================================================ SFC-09 NaN silently swallowed
    nan_rec = PRE.u_recursion(PRE_ZONES[1], [(float('nan'), True)])
    tau1 = float(ZONES[1].target_c)
    d1 = float(ZONES[1].delta_real)
    ck('SFC-09', 'a NaN duration is silently swallowed inside the recursion (`max(tau, nan)` returns '
       'tau), so the pre-fix per-zone numbers look like a healthy post-door state even though the '
       'overall verdict happens to be REFUSED',
       all(x == x and abs(x) != float('inf') for x in nan_rec)
       and abs(nan_rec[1] - (tau1 + d1)) < 1e-12,
       {'pre_fix_u_sequence': nan_rec, 'healthy_looking_value_c': nan_rec[1],
        'tau_plus_delta_c': tau1 + d1,
        'nan_semantics': 'max(4.0, nan) == 4.0 in CPython, so a NaN looks like a healthy floor',
        'strict_module': 'refuses with NON_FINITE_INPUT before any arithmetic',
        'label': 'DEFECT_RECORDED_IN_PRE_FIX_MODULE'},
       conclusion='refusal must precede evaluation, otherwise the receipt carries meaningless numbers')

    # ================================================================ SFC-10 extreme segments
    extreme = []
    ok_all = True
    for z in ZONES:
        for dt in (0.0, 1e-12, 1e-9, 1e-8, 0.001, 12.0, 24.0, 1e6, 1e300):
            lo, hi = SF.executor_segment_enclosure(z, z.target_c, dt, True, KAPPA)
            ex = executor_doors(z.zone, dt, 1)
            inside = float(lo) - 1e-9 <= ex <= float(hi) + 1e-9
            ok_all = ok_all and inside
            extreme.append({'zone': z.zone, 'dt_h': dt, 'executor_c': ex,
                            'enclosure': [float(lo), float(hi)], 'contains': inside})
    ck('SFC-10', 'zero-length, ultra-short, long and absurd segments are all enclosed by the strict '
       'bound and none produces an overflow-driven unsound result',
       ok_all, {'rows': extreme, 'n': len(extreme)},
       conclusion='the enclosure is total on the legal input domain, including the dt <= _EPS branch')

    # ================================================================ SFC-11 contract identity
    ok_id = False
    try:
        SF.require_contract_identity(CONTRACT, 'MADE_UP')
    except SF.Refusal as exc:
        ok_id = exc.code == 'CONTRACT_IDENTITY_MISMATCH'
    pre_accepts_v1 = PRE.certify_interval(
        CONTRACT_V1, contract_identity='MADE_UP', state_identity='MADE_UP',
        segments_by_zone={0: [(1.0, False)], 1: [(0.01, True)], 2: [(1.0, False)]},
        active_zone_mask=(True,) * N_ZONES)['experimental_verdict']
    ck('SFC-11', 'the strict module refuses a contract identity that is not the contract hash; the '
       'pre-fix module accepts ANY string, including a v1 contract labelled by hand',
       ok_id and pre_accepts_v1 == 'INTERVAL_BOUND_HOLDS',
       {'strict_exception': 'CONTRACT_IDENTITY_MISMATCH',
        'pre_fix_verdict_with_v1_contract_and_made_up_identity': pre_accepts_v1,
        'v2_contract_hash': CONTRACT.contract_hash, 'v1_contract_hash': CONTRACT_V1.contract_hash,
        'delta_star_v2_h': {z.zone: float(z.return_time_h) for z in ZONES},
        'label': 'DEFECT_RECORDED_IN_PRE_FIX_MODULE'},
       conclusion='a contract change must invalidate the bound automatically; a caller-declared '
                  'string cannot do that')

    # ================================================================ SFC-12 P1 dominance
    random.seed(424242)
    n_checks, n_miss_pre, n_miss_strict = 0, 0, 0
    for z, pz in zip(ZONES, PRE_ZONES):
        for _ in range(60):
            k = random.randint(1, 12)
            segs = [(10 ** random.uniform(-9, 0.9), random.random() < 0.6) for _ in range(k)]
            ex = z.target_c
            temps = tuple(float(t) for t in TH.target_temperature_c)
            for dt, opens in segs:
                temps, _e = cs._advance_temperatures(temps, float(dt), (True,) * N_ZONES, CONTRACT)
                if opens:
                    temps, _h, _r = cs._apply_door_event(temps, (True,) * N_ZONES, z.zone, CONTRACT)
                ex = temps[z.zone]
            pre_u = max(PRE.u_recursion(pz, segs))
            strict_u = SF.executor_u_recursion(z, segs, KAPPA)['u_max_float']
            n_checks += 1
            n_miss_pre += 0 if ex <= pre_u else 1
            n_miss_strict += 0 if ex <= strict_u else 1
    ck('SFC-12', 'the pre-fix P1 recursion and the strict F-level recursion both dominate the '
       'executor on randomized mixed segment sequences',
       n_miss_pre == 0 and n_miss_strict == 0,
       {'n_sequences': n_checks, 'pre_fix_misses': n_miss_pre, 'strict_misses': n_miss_strict,
        'why_pre_fix_still_works_here': 'under bit-identical arithmetic the certifier recomputes the '
                                        "executor's own float, and the 4-ulp shift only has to cover "
                                        'certifier/executor divergence (lemma L-D); that is a '
                                        'different object from the real-value bound in SFC-04',
        'label': 'DIAGNOSTIC (randomized search, not a proof)'},
       conclusion='P1 is the sound part of the pre-fix module; P1 + P2 combined is not, because the '
                  'min() admits the unsound P2 branch')

    # ================================================================ SFC-13 the cost of no kappa
    exp_free = {z.zone: float(SF.exp_free_worst_case(z)) for z in ZONES}
    dstar = {z.zone: float(z.return_time_h) for z in ZONES}
    ck('SFC-13', 'refusing to assume ANYTHING about exp costs a factor of 60-130: the kappa-free '
       'requirement is 3.5-5.4 h per segment against Delta* of 2.5-3.3 minutes',
       all(exp_free[z] > 50 * dstar[z] for z in exp_free),
       {'kappa_free_segment_requirement_h': exp_free, 'delta_star_h': dstar,
        'ratio': {z: exp_free[z] / dstar[z] for z in exp_free},
        'why': 'with exp unconstrained in (0,1] the segment end can reach the ambient temperature, '
               'so only dt >= (A - tau)/c can be certified non-warming',
        'label': 'PROVED (R-level, no assumption)'},
       conclusion='a kappa assumption is not decorative: without it no realistic cold-chain segment '
                  'can be certified, which is why signing an eta/kappa is unavoidable')

    # ================================================================ SFC-14 R vs F separation
    z = ZONES[1]
    t0 = float(z.target_c) + float(z.delta_real)
    r_lo, r_hi = SF.real_psi_enclosure(z, Fraction(t0), Fraction(0.01))
    f_lo, f_hi = SF.executor_segment_enclosure(z, t0, 0.01, False, KAPPA)
    inside = float(f_lo) <= SF.plain_advance_float(z, t0, 0.01) <= float(f_hi)
    ck('SFC-14', 'the R-level and F-level enclosures are different objects: R needs no assumption and '
       'is exact to far below one ulp, F carries the exp assumption and is about one ulp wide',
       float(r_hi - r_lo) < 1e-40 and float(f_hi - f_lo) >= float(r_hi - r_lo) and inside,
       {'t0': t0, 'dt_h': 0.01, 'real_enclosure_width': float(r_hi - r_lo),
        'executor_enclosure_width': float(f_hi - f_lo),
        'real_enclosure': [float(r_lo), float(r_hi)],
        'executor_enclosure': [float(f_lo), float(f_hi)],
        'executor_value': SF.plain_advance_float(z, t0, 0.01)},
       conclusion='P1/P2/P3 as statements about the MODEL can be proved without any float '
                  'assumption; only the transfer to stored floats needs kappa')

    # ================================================================ SFC-15 false REFUSAL
    z2, pz2 = ZONES[2], PRE_ZONES[2]
    n = 50
    ex2 = executor_doors(2, 0.0, n)
    pre2 = PRE.door_count_bound(pz2, n)
    upre2 = max(PRE.u_recursion(pz2, [(0.0, True)] * n))
    tight2 = min(upre2, pre2)
    ck('SFC-15', 'the 4-ulp shift also fails in the OTHER direction: in the negative-target zone it '
       'turns an executor-safe boundary state into INTERVAL_BOUND_EXCEEDS_HARD_BOUND (false refusal)',
       ex2 <= z2.hard_high_c and tight2 > z2.hard_high_c,
       {'zone': 2, 'n_doors': n, 'executor_c': ex2, 'hard_high_c': z2.hard_high_c,
        'pre_fix_tight_bound_c': tight2,
        'executor_at_or_below_hard_high': ex2 <= z2.hard_high_c,
        'bound_above_hard_high': tight2 > z2.hard_high_c,
        'reading': 'a fixed upward shift of 4 ulp on a -12 C boundary refuses a plan whose executor '
                   'state is exactly at the bound; over-refusal is safe but it is not evidence, and '
                   'it is caused by the same unquantified margin'},
       conclusion='the guard is unsound in one direction and over-conservative in the other, which is '
                  'exactly what an unquantified margin does; a derived margin removes both')

    payload = {
        'schema': 'q3_strict_float_bounds_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'contract_constants': {
            'ambient_c': float(TH.ambient_temperature_c),
            'target_c': [float(x) for x in TH.target_temperature_c],
            'hard_bounds_c': [[float(a), float(b)] for a, b in TH.hard_temperature_bounds_c],
            'heat_transfer_per_hour': [float(x) for x in TH.heat_transfer_per_hour],
            'cooling_rate_c_per_hour': [float(x) for x in TH.cooling_rate_c_per_hour],
            'door_heat_kwh': [float(x) for x in TH.door_heat_kwh],
            'thermal_capacity_kwh_per_c': [float(x) for x in TH.thermal_capacity_kwh_per_c],
            'delta_door_c': [float(z.delta_real) for z in ZONES],
            'delta_door_float': [float(z.delta_float) for z in ZONES],
            'drift_margin_c_per_hour': [float(z.drift_margin_c_per_hour) for z in ZONES],
            'return_time_h': [float(z.return_time_h) for z in ZONES],
            'kappa_free_segment_requirement_h': [float(SF.exp_free_worst_case(z)) for z in ZONES]},
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('STRICT_FLOAT_CONTROLS_PASS' if all(r['ok'] for r in ROWS)
                    else 'STRICT_FLOAT_CONTROLS_FAIL'),
        'labels': {
            'PROVED': 'analytic statements plus exact-rational enclosures (SFC-01, SFC-06, SFC-13)',
            'DIAGNOSTIC': 'randomized or single-platform measurements (SFC-02, SFC-12)',
            'PRE_FIX_DEFECT': 'behaviour of the pre-fix module that blocks certification '
                              '(SFC-04, SFC-06, SFC-08, SFC-09, SFC-11)',
            'UNKNOWN': 'any statement requiring a platform-independent bound on math.exp'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False), encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL -> %s ==='
          % (len(ROWS), payload['n_pass'], payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
