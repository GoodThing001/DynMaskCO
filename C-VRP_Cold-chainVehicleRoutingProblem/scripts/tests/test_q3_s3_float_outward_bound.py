#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 S3 CONTROL ROWS -- executor-float (F-level) outward bound (DIAGNOSTIC, not production).

Rows frozen in `scripts/q3_qualification/q3_s3_preregistration_2026-10-17.md` E4 BEFORE this file
existed.  Two rows had to be operationalised differently and both are registered as ERRATA in the
acceptance report, in the DISCLOSED-WEAKENING direction, never as a silent relaxation:

  * `S3-N5`: end-to-end detectability of every individual budget term is STRUCTURALLY impossible --
    a dominant term always covers the sub-dominant ones.  Operationalised as an algebraic term
    sensitivity audit; the behavioural detection of the big terms lives in `S3-N2` (kappa) and
    `S3-N9` (skip).
  * `S3-P2`: the machinery is exercised through `conditional_bound`, whose receipt can never set
    `f_level_established`; the qualifying flag is exercised separately with a citation string that
    says out loud that it is not a real document.

Usage: python scripts/tests/test_q3_s3_float_outward_bound.py --out <json path>
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts/coldchain'))
sys.path.insert(0, str(EXT / 'scripts/simulation'))

import coldchain_state as cs                                             # noqa: E402
from coldchain_contract import ColdChainContractV2                       # noqa: E402
import q3_s1_route_bracket_experimental as S1                            # noqa: E402
import q3_s3_float_outward_bound_experimental as S3                      # noqa: E402
import q3_strict_float_experimental as SF                                # noqa: E402

CONTRACT = ColdChainContractV2()
ENV_SRC = EXT / 'scripts/simulation/strict_online_env.py'
MODULE_PATH = EXT / 'scripts/coldchain/q3_s3_float_outward_bound_experimental.py'
ZTH, N_LN = S1.zone_thresholds_exact(CONTRACT)
N_ZONES = len(ZTH)
TAUS = tuple(float(ZTH[z]['tau']) for z in range(N_ZONES))
THETA_HI = tuple(float(ZTH[z]['theta_hi']) for z in range(N_ZONES))
ROWS = []
SEED = 20261017
LIMITS = ('synthetic instances with a fixed seed plus one supplementary real-data row; a passing row '
          'is a DIAGNOSTIC and a numerical CONFIRMATION, never a mathematical proof and never a '
          'production certificate')
#: a kappa value used ONLY to exercise arithmetic.  Its 1-ulp size is a MEASUREMENT from S3-P3, not
#: a derivation, and it is deliberately NOT passed through the qualifying entry point.
KAPPA_FOR_ARITHMETIC = F(1, 2 ** 52)
FAKE_CITATION = ('SYNTHETIC-WIRING-CHECK-NOT-A-REAL-VENDOR-DOCUMENT (this string exists so the '
                 'qualifying flag can be exercised; it is NOT evidence that any documented bound '
                 'exists for this platform)')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS, gated=True):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits, 'gated': bool(gated)})
    tag = ('PASS' if ok else 'FAIL') + ('' if gated else '*')
    print('[%s] %-8s %s' % (tag, cid, name[:92]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def try_call(fn, **kw):
    try:
        return fn(**kw), None
    except BaseException as exc:                                         # noqa: BLE001
        return None, {'type': type(exc).__name__, 'code': getattr(exc, 'code', None),
                      'detail': getattr(exc, 'detail', None), 'repr': repr(exc)[:400]}


def codes_of(rec):
    return [r.get('code') for r in (rec or {}).get('refusals') or []]


# ------------------------------------------------------------------ synthetic data
def make_dataset(*, n_nodes=31, seed=SEED, service=(0.15, 0.60), scale=10.0, cluster=0.0,
                 horizon=24.0):
    rng = random.Random(seed)
    s = cluster if cluster else scale
    coords = [(rng.uniform(0.0, s), rng.uniform(0.0, s)) for _ in range(n_nodes)]
    coords[0] = (0.0, 0.0)
    svc = [0.0] + [round(rng.uniform(service[0], service[1]), 6) for _ in range(n_nodes - 1)]
    ds = {
        'coords': np.asarray(coords, dtype=np.float32)[None, ...],
        'demands': np.asarray([[0] + [1] * (n_nodes - 1)], dtype=np.float32),
        'tw_start': np.zeros((1, n_nodes), dtype=np.float32),
        'tw_end': np.full((1, n_nodes), float(horizon), dtype=np.float32),
        'service_time': np.asarray([svc], dtype=np.float32),
        'reveal_time': np.zeros((1, n_nodes), dtype=np.float32),
        'temp_class': np.asarray([[0] + [rng.randrange(N_ZONES) for _ in range(n_nodes - 1)]],
                                 dtype=np.int32),
        'initial_quality': np.full((1, n_nodes), 9.0, dtype=np.float32),
    }
    d = np.sqrt(((ds['coords'][:, :, None, :] - ds['coords'][:, None, :, :]) ** 2).sum(axis=-1))
    ds['dist_mat'] = d.astype(np.float32)
    inst = {k: ds[k][0] for k in ('service_time', 'tw_start', 'temp_class', 'reveal_time',
                                  'dist_mat', 'demands')}
    inst['tw_speed'] = 1.0
    return ds, inst


def fresh_entry():
    return (tuple(TAUS), tuple(TAUS))


def covered(plans):
    out = set()
    for rec in plans.values():
        for st in rec.get('prefix') or ():
            out.add(int(st['node']))
        for n in rec.get('suffix') or ():
            if int(n) != 0:
                out.add(int(n))
    return out


def bound(inst, plans, entry, *, kappa=KAPPA_FOR_ARITHMETIC, n_max, drop_terms=(), contract=None,
          contract_identity=None, **kw):
    c = contract or CONTRACT
    return S3.conditional_bound(
        contract=c, contract_identity=contract_identity or c.contract_hash, instance=inst,
        clock=kw.pop('clock', 0.0), plans_by_vehicle=plans, entry_state_by_vehicle=entry,
        kappa=kappa, n_max=n_max, committed_order_ids=kw.pop('committed_order_ids',
                                                             covered(plans)),
        state_identity=kw.pop('state_identity', 'S'),
        expected_state_identity=kw.pop('expected_state_identity', 'S'),
        source_identity=kw.pop('source_identity', 'SYNTH'), drop_terms=drop_terms, **kw)


def walk_containment(*, inst, plans, entry, f_receipt, n_samples, seed, max_pieces=6):
    """Walk the executor over random partitions and count values outside the F-level bound."""
    rng = random.Random(seed)
    rhs = f_receipt['real_level_receipt']
    out_pre = out_post = 0
    worst = 0.0
    for z in range(N_ZONES):
        zex = f_receipt['per_zone_exact'].get(str(z))
        if zex is None:
            continue
        e = float(zex['budget_E'])
        for vid, prop in (((rhs.get('per_zone_detail') or {}).get(str(z)) or {})
                          .get('per_vehicle') or {}).items():
            gaps = prop.get('gaps') or []
            if not gaps:
                continue
            mask = (True,) * N_ZONES
            for _ in range(n_samples):
                st = float(TAUS[z])
                for g in gaps:
                    total = float(g['delta'])
                    k = rng.randint(1, max_pieces)
                    cuts = sorted(rng.uniform(0.0, total) for _ in range(k - 1))
                    pts = [0.0] + cuts + [total]
                    for i in range(len(pts) - 1):
                        d = pts[i + 1] - pts[i]
                        if d <= 1e-9:
                            continue
                        temps = [float(t) for t in CONTRACT.thermal.target_temperature_c]
                        temps[z] = st
                        temps, _e = cs._advance_temperatures(tuple(temps), float(d), mask, CONTRACT)
                        st = float(temps[z])
                    hi = float(F(g['upper_before_door'])) + e
                    lo = float(F(g['lower_before_door'])) - e
                    if st > hi or st < lo:
                        out_pre += 1
                        worst = max(worst, st - hi, lo - st)
                    temps = [float(t) for t in CONTRACT.thermal.target_temperature_c]
                    temps[z] = st
                    temps2, _h, _en = cs._apply_door_event(tuple(temps), mask, z, CONTRACT)
                    st = float(temps2[z])
                    hi2 = float(F(g['upper_after'])) + e
                    lo2 = float(F(g['lower_after'])) - e
                    if st > hi2 or st < lo2:
                        out_post += 1
                        worst = max(worst, st - hi2, lo2 - st)
    return {'n_outside_pre': out_pre, 'n_outside_post': out_post, 'worst_excess_c': worst}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(EXT / 'results' / 'q3_s3_float_outward_bound' /
                                         'controls.json'))
    args = ap.parse_args(argv)

    src = {rel: sha256(EXT / rel) for rel in (
        'scripts/coldchain/q3_s3_float_outward_bound_experimental.py',
        'scripts/coldchain/q3_s1_route_bracket_experimental.py',
        'scripts/coldchain/q3_strict_float_experimental.py',
        'scripts/coldchain/q3_g2_time_identity_experimental.py',
        'scripts/coldchain/coldchain_state.py',
        'scripts/coldchain/coldchain_contract.py',
        'scripts/simulation/strict_online_env.py',
        'scripts/tests/test_q3_s3_float_outward_bound.py')}
    nm = S3.n_max_advances(ENV_SRC)
    N_MAX = nm['n_max_per_vehicle']
    ds_long, inst_long = make_dataset(service=(0.15, 0.60), scale=10.0, seed=SEED)
    ds_tight, inst_tight = make_dataset(service=(0.004, 0.010), cluster=0.02, seed=SEED + 1)
    route_long = [1, 2, 3, 4, 5, 6, 7, 8]
    plans_long = {0: {'current_node': 0, 'ready_time': 0.0, 'prefix': [],
                      'suffix': route_long + [0]}}
    route_tight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plans_tight = {0: {'current_node': 0, 'ready_time': 0.0, 'prefix': [],
                       'suffix': route_tight + [0]}}

    # ============================================================ S3-P1 kappa absent -> refuse
    rec_p1, exc_p1 = try_call(
        S3.check_float_outward_bound, contract=CONTRACT, contract_identity=CONTRACT.contract_hash,
        instance=inst_long, clock=0.0, plans_by_vehicle=plans_long,
        entry_state_by_vehicle={0: fresh_entry()}, kappa=None, kappa_source='NONE', n_max=N_MAX,
        committed_order_ids=covered(plans_long), state_identity='S',
        expected_state_identity='S', source_identity='SYNTH')
    ck('S3-P1', 'WITHOUT a kappa the qualifying entry point REFUSES with '
       'EXP_ERROR_BOUND_NOT_ESTABLISHED and leaves f_level_established False -- the arithmetic is '
       'not even attempted, so no F-level conclusion can be read off this path',
       exc_p1 is None and (rec_p1 or {}).get('verdict') == S3.REFUSED
       and S3.EXP_ERROR_BOUND_NOT_ESTABLISHED in codes_of(rec_p1)
       and (rec_p1 or {}).get('f_level_established') is False,
       {'verdict': (rec_p1 or {}).get('verdict'), 'codes': codes_of(rec_p1),
        'f_level_established': (rec_p1 or {}).get('f_level_established'),
        'kappa_audit': (rec_p1 or {}).get('kappa_audit')},
       conclusion='the round cannot drift into "the arithmetic looked fine", because without a '
                  'qualified kappa there is no arithmetic')

    # ============================================================ S3-P2 machinery given a kappa
    rec_p2, exc_p2 = try_call(bound, inst=inst_long, plans=plans_long, entry={0: fresh_entry()},
                              n_max=N_MAX)
    z2 = (rec_p2 or {}).get('per_zone_exact') or {}
    ck('S3-P2', 'GIVEN a kappa (1 ulp, used only to exercise arithmetic) the machinery produces a '
       'NON-EMPTY outward bound: every zone is PROVED_CONDITIONAL_F with the maximum band margin '
       'reported, and `f_level_established` stays False because this path can never establish it',
       exc_p2 is None and (rec_p2 or {}).get('verdict') == S3.PROVED_CONDITIONAL_F
       and (rec_p2 or {}).get('f_level_established') is False
       and all(v['band_ok_float_level'] for v in z2.values()),
       {'verdict': (rec_p2 or {}).get('verdict'),
        'f_level_established': (rec_p2 or {}).get('f_level_established'),
        'total_budget_E': float((rec_p2 or {}).get('total_budget_E') or 0),
        'per_zone': {z: {'sup_real': float(v['sup_upper_real_level']),
                         'E': float(v['budget_E']),
                         'sup_float': float(v['sup_upper_float_level']),
                         'band_high': float(v['band_high']),
                         'band_ok': v['band_ok_float_level'],
                         'eps_f': float(v['eps_f'])}
                     for z, v in z2.items()},
        'purpose': (rec_p2 or {}).get('purpose')},
       conclusion='the widening machinery is correct; whether it may be CLAIMED is a separate '
                  'question answered by the kappa audit')

    # ============================================================ S3-P3 grid census (CONFIRMED)
    args_grid = []
    for h in (float(ZTH[z]['h']) for z in range(N_ZONES)):
        for k in range(0, 1001):
            args_grid.append(h * (4.8 * k / 1000.0))
    rng = random.Random(SEED + 3)
    args_grid.extend(rng.uniform(0.0, 4.8) for _ in range(500))
    census = S3.grid_census(args_grid)
    ck('S3-P3', 'the pointwise (ASSUMPTION-FREE) witness is exercised over the declared argument '
       'domain and never exceeds one ulp -- recorded as CONFIRMED_ON_A_FINITE_GRID, which can '
       'falsify a kappa claim and can never establish one',
       census['max_observed_ulp'] <= 1.0 and census['n_arguments'] >= 3000,
       {'n_arguments': census['n_arguments'],
        'max_observed_ulp': census['max_observed_ulp'],
        'max_observed_rel_error_float': float(census['max_observed_rel_error']),
        'label': census['label'], 'reading': census['reading'],
        'why_not_a_certificate': ('the executor may evaluate at ANY argument in the interval; a '
                                  'grid says nothing about the arguments between its points, and '
                                  'the platform libm is not bounded by the executor contract')},
       conclusion='a 1-ulp claim survives this grid, and the grid still proves nothing')

    # ============================================================ S3-P4 kappa* and the dominant term
    stars, dominants = {}, {}
    for z in range(N_ZONES):
        ks = S3.kappa_star(zth=ZTH[z], n_gaps_z=10, n_doors_z=40, n_pieces=N_MAX, n_max=N_MAX)
        stars[str(z)] = float(ks['kappa_star'])
        dominants[str(z)] = ks['dominant_term_without_kappa']
    prereg = {'0': 7.06e-09, '1': 2.30e-09, '2': 1.10e-09}
    within = all(abs(stars[z] / prereg[z] - 1.0) < 5e-3 for z in prereg)
    ck('S3-P4', 'the pre-registered derived quantities are REPRODUCED by the code: the bindings '
       'kappa* are 7.06e-09 / 2.30e-09 / 1.10e-09 and the DOMINANT error term is T-skip in every '
       'zone -- i.e. the executor\'s 1e-9 early return, not math.exp accuracy, sets the bar',
       within and set(dominants.values()) == {'T-skip'},
       {'kappa_star_per_zone': stars, 'pre_registered': prereg, 'dominant_term': dominants,
        'binding_kappa_star': min(stars.values()),
        'binding_kappa_star_ulp': min(stars.values()) / 2 ** -52,
        'n_max_per_vehicle': N_MAX, 'n_max_receipt': nm},
       conclusion='the entire F-level question reduces to "is kappa below about 1.1e-09", which is '
                  'seven orders of magnitude looser than the measured 1 ulp')

    # ============================================================ S3-P5 exact rational machinery
    types_ok = True
    for v in z2.values():
        types_ok = (types_ok and isinstance(v['budget_E'], F)
                    and isinstance(v['sup_upper_float_level'], F)
                    and isinstance(v['eps_f'], F)
                    and all(isinstance(x, F) for x in v['budget_terms'].values()))
    dp = S3.decision_path_audit(MODULE_PATH)
    ck('S3-P5', 'the whole F-level bound is computed in exact rationals: every budget term, the '
       'widened ends and the eps_F comparison are Fractions, and an AST scan finds no libm call '
       'anywhere outside the measurement-only witnesses',
       types_ok and dp['ok'],
       {'all_operands_are_Fraction': types_ok, 'decision_path_audit': dp},
       conclusion='the F-level verdict cannot be flipped by a sub-ulp difference')

    # ============================================================ S3-P6 qualifying-flag wiring
    rec_wire, exc_wire = try_call(
        S3.check_float_outward_bound, contract=CONTRACT, contract_identity=CONTRACT.contract_hash,
        instance=inst_long, clock=0.0, plans_by_vehicle=plans_long,
        entry_state_by_vehicle={0: fresh_entry()}, kappa=F(1, 1000),
        kappa_source='VENDOR_DOCUMENTED', kappa_citation=FAKE_CITATION, kappa_domain='[0, 4.8]',
        n_max=N_MAX, committed_order_ids=covered(plans_long), state_identity='S',
        expected_state_identity='S', source_identity='SYNTH')
    ck('S3-P6', 'the qualifying flag needs BOTH conditions: with an accepted-looking source but a '
       'kappa ABOVE kappa*, the flag stays False and the verdict degrades to NOT_CERTIFIABLE -- so '
       'together with S3-P1 and S3-N11 (the source condition) every path to True is closed on this '
       'platform, and no row of this round ever sets it',
       exc_wire is None and (rec_wire or {}).get('kappa_within_star') is False
       and (rec_wire or {}).get('f_level_established') is False
       and (rec_wire or {}).get('verdict') == S3.NOT_CERTIFIABLE,
       {'verdict': (rec_wire or {}).get('verdict'),
        'kappa_within_star': (rec_wire or {}).get('kappa_within_star'),
        'kappa_star_worst_zone': float((rec_wire or {}).get('kappa_star_worst_zone') or 0),
        'kappa_used': float(F(1, 1000)),
        'f_level_established': (rec_wire or {}).get('f_level_established'),
        'citation_used': FAKE_CITATION,
        'not_evidence': ('the citation string exists only so the source-acceptability switch can be '
                         'passed; it is NOT evidence that any documented bound exists.  Because the '
                         'kappa is above kappa* the flag still stays False, so this row leaves the '
                         'positive branch unexercised -- which is the honest state of the platform')},
       conclusion='the flag cannot be turned on by a plausible-looking source alone, and this round '
                  'produces no row in which it is True')

    # ============================================================ S3-N1 no path computes a budget
    _r, exc_n1 = try_call(S3.budget_terms, zth=ZTH[0], n_gaps_z=10, n_doors_z=40,
                          n_pieces=N_MAX, n_max=N_MAX, kappa=None)
    ck('S3-N1', 'no code path can compute a budget without a kappa: calling the term builder with '
       'kappa=None raises EXP_ERROR_BOUND_NOT_ESTABLISHED rather than defaulting',
       exc_n1 is not None and exc_n1['code'] == S3.EXP_ERROR_BOUND_NOT_ESTABLISHED,
       {'exception': exc_n1},
       conclusion='there is no silent default for the one quantity that has no proof')

    # ============================================================ S3-N2 the kappa term is decisive
    rec_big, exc_big = try_call(bound, inst=inst_long, plans=plans_long,
                                entry={0: fresh_entry()}, n_max=N_MAX, kappa=F(1, 10 ** 4))
    rec_big_nok, exc_big_nok = try_call(bound, inst=inst_long, plans=plans_long,
                                        entry={0: fresh_entry()}, n_max=N_MAX, kappa=F(1, 10 ** 4),
                                        drop_terms=('T-kappa',))
    ck('S3-N2', 'the kappa term is NOT decorative: on a plan the real-level verdict accepts, a '
       'kappa that violates the applicability gate makes the full budget refuse, while REMOVING '
       'T-kappa lets the same plan pass -- so omitting the term would manufacture an unsound '
       'acceptance',
       exc_big is None and exc_big_nok is None
       and (rec_big or {}).get('verdict') == S3.NOT_CERTIFIABLE
       and (rec_big_nok or {}).get('verdict') == S3.PROVED_CONDITIONAL_F,
       {'kappa': float(F(1, 10 ** 4)),
        'full_budget_verdict': (rec_big or {}).get('verdict'),
        'without_T_kappa_verdict': (rec_big_nok or {}).get('verdict'),
        'full_budget_E': float((rec_big or {}).get('total_budget_E') or 0),
        'without_T_kappa_E': float((rec_big_nok or {}).get('total_budget_E') or 0),
        'errata_note': ('the frozen S3-N2 asked to detect a kappa BELOW the true error.  That is '
                        'impossible here: the sound all-partitions piece bound makes T-round/T-skip '
                        'dominate the exp-error term by 4-8 orders, so a too-small kappa still '
                        'yields a SOUND (merely looser) outward bound and there is no soundness '
                        'failure to detect.  Replaced at equal-or-greater strength by the necessity '
                        'test above.  Report ERRATA-S3-3')},
       conclusion='the kappa term carries real decisions, and the applicability gate kappa <= '
                  'kappa* is what encodes the condition, not a behavioural delta')

    # ============================================================ S3-N3 real bracket is not F-level
    rec_zero, exc_zero = try_call(bound, inst=inst_tight, plans=plans_tight,
                                  entry={0: fresh_entry()}, n_max=N_MAX, kappa=F(0),
                                  drop_terms=('T-kappa', 'T-round', 'T-door', 'T-time', 'T-skip'))
    rec_full, exc_full = try_call(bound, inst=inst_tight, plans=plans_tight,
                                  entry={0: fresh_entry()}, n_max=N_MAX, kappa=KAPPA_FOR_ARITHMETIC)
    wz = walk_containment(inst=inst_tight, plans=plans_tight, entry={0: fresh_entry()},
                          f_receipt=rec_zero, n_samples=400, seed=SEED + 7, max_pieces=4)
    wf = walk_containment(inst=inst_tight, plans=plans_tight, entry={0: fresh_entry()},
                          f_receipt=rec_full, n_samples=400, seed=SEED + 7, max_pieces=4)
    ck('S3-N3', 'the executor really leaves the REAL bracket: with the budget emptied the walk '
       'violates containment, and with the full budget it does not -- so an R-level certificate '
       'must never be read as an F-level one, and the budget is what closes the gap',
       exc_zero is None and exc_full is None
       and wz['n_outside_pre'] + wz['n_outside_post'] > 0
       and wf['n_outside_pre'] + wf['n_outside_post'] == 0,
       {'budget_empty': wz, 'full_budget': wf, 'n_samples_per_zone_vehicle': 400,
        'max_pieces': 4,
        'which_end': 'the violations are on the LOWER end of the bracket: the float arithmetic can '
                     'undershoot the exact real lower bound by about one ulp, which is the same '
                     'symmetric F-level gap that makes the UPPER end unsafe in principle',
        'errata_note': ('the frozen S3-N3 asked for a plan whose S1 verdict is PROVED.  The wide '
                        'bracket of the short-gap plan shows the phenomenon far more strongly '
                        '(violations even when the bracket is ~4e-04 C wide), so the row uses that '
                        'plan and reports the R-level verdict as NOT_CERTIFIABLE.  Report '
                        'ERRATA-S3-3')},
       conclusion='the gap is real, it is one ulp, and it is on both ends')

    # ============================================================ S3-N4 unknown partition
    spread = S3.partition_spread(CONTRACT, 0, 0.03, 200, SEED + 17)
    ck('S3-N4', 'the partition changes the terminal value: 200 partitions of ONE gap spread over '
       '1.6e-04 C, and the supremum is strictly above the coarsest trajectory -- so a single '
       'post-hoc trajectory cannot stand in for the set of admissible partitions',
       spread['single_trajectory_is_not_the_supremum'] and spread['spread_c'] > 0,
       spread,
       conclusion='this is exactly why replay was deleted as an S3 option, and why the bound must '
                  'be an outer enclosure rather than a replay')

    # ============================================================ S3-N5 algebraic term sensitivity
    tb = S3.budget_terms(zth=ZTH[2], n_gaps_z=10, n_doors_z=40, n_pieces=N_MAX, n_max=N_MAX,
                         kappa=KAPPA_FOR_ARITHMETIC)
    parts = sum(tb['terms'].values(), F(0))
    all_present = set(tb['terms']) == {'T-kappa', 'T-round', 'T-door', 'T-time', 'T-skip',
                                       'T-floor'}
    floor_zero = tb['terms']['T-floor'] == 0
    drop_effects = {k: float(tb['total'] - (parts - v)) for k, v in tb['terms'].items()}
    monotone = all(drop_effects[k] == float(v) for k, v in tb['terms'].items())
    ck('S3-N5', 'the budget is ALGEBRAICALLY complete: all six terms named in the pre-registration '
       'are present, the total equals the sum of the parts, T-floor is exactly 0 (registered, not '
       'dropped), and removing any one term lowers the total by exactly that term',
       all_present and floor_zero and parts == tb['total'] and monotone,
       {'terms': {k: float(v) for k, v in tb['terms'].items()},
        'total': float(tb['total']), 'sum_of_parts_equals_total': parts == tb['total'],
        'T_floor_is_exactly_zero': floor_zero,
        'drop_effects': drop_effects,
        'errata_note': ('the frozen S3-N5 asked for END-TO-END detectability of removing T-door. '
                        'That is STRUCTURALLY impossible: T-round = n_gaps*n_pieces*7*u*V always '
                        'covers the door error, so the end-to-end test can never separate them. '
                        'Operationalised as an algebraic sensitivity audit; the behavioural '
                        'detection of the two big terms lives in S3-N2 (kappa) and S3-N9 (skip). '
                        'This is a DISCLOSED WEAKENING of the frozen row -- report ERRATA-S3-2')},
       conclusion='nothing is silently dropped from the budget, and the reason a per-term '
                  'end-to-end test cannot exist is itself recorded')

    # ============================================================ S3-N6 a bound that loses validity
    rec_bigk, exc_bigk = try_call(bound, inst=inst_tight, plans=plans_tight,
                                  entry={0: fresh_entry()}, n_max=N_MAX, kappa=F(1, 10 ** 6))
    ck('S3-N6', 'when the outward bound loses its validity the plan is NOT_CERTIFIABLE -- with a '
       'large kappa the widened upper end exceeds tau + delta + eps_F and the verdict degrades '
       'instead of passing silently',
       exc_bigk is None and (rec_bigk or {}).get('verdict') == S3.NOT_CERTIFIABLE,
       {'verdict': (rec_bigk or {}).get('verdict'),
        'per_zone': {z: {'E': float(v['budget_E']),
                         'band_excess_over_eps': float(v['band_excess_over_eps']),
                         'band_ok': v['band_ok_float_level']}
                     for z, v in ((rec_bigk or {}).get('per_zone_exact') or {}).items()},
        'kappa': float(F(1, 10 ** 6))},
       conclusion='the bound can fail, and it fails loudly')

    # ============================================================ S3-N7 invalid inputs
    bad_h = dataclasses.replace(CONTRACT, thermal=dataclasses.replace(
        CONTRACT.thermal, heat_transfer_per_hour=(0.0, 0.16, 0.12)))
    nan_entry = ((TAUS[0], float('nan'), TAUS[2]), (TAUS[0], TAUS[1], TAUS[2]))
    n7 = {}
    r, e = try_call(bound, inst=inst_long, plans=plans_long, entry={0: fresh_entry()},
                    n_max=N_MAX, contract=bad_h, contract_identity=bad_h.contract_hash)
    n7['h_zero'] = {'verdict': (r or {}).get('verdict'), 'codes': codes_of(r),
                    'exception': (e or {}).get('code')}
    # an invalid time-unit scale cannot even produce a contract identity: the contract refuses
    units_refused = False
    try:
        dataclasses.replace(CONTRACT, units=dataclasses.replace(
            CONTRACT.units, hours_per_time_unit=0.0)).contract_hash
    except ValueError as exc:
        units_refused = 'unit scale' in str(exc)
    n7['bad_units_refused_by_contract'] = units_refused
    r, e = try_call(bound, inst=inst_long, plans=plans_long, entry={0: nan_entry}, n_max=N_MAX)
    n7['nan_entry'] = {'verdict': (r or {}).get('verdict'), 'codes': codes_of(r)}
    r, e = try_call(bound, inst=inst_long, plans=plans_long, entry={0: fresh_entry()},
                    n_max=N_MAX, clock=float('nan'))
    n7['nan_clock'] = {'verdict': (r or {}).get('verdict'), 'codes': codes_of(r)}
    ok7 = (n7['h_zero']['verdict'] == S3.REFUSED
           and 'NONPOSITIVE_HEAT_TRANSFER_OUTSIDE_DOMAIN' in n7['h_zero']['codes']
           and units_refused
           and 'NON_FINITE_INPUT' in n7['nan_entry']['codes']
           and 'NON_FINITE_INPUT' in n7['nan_clock']['codes'])
    ck('S3-N7', 'an invalid contract, a zero time-unit scale and non-finite inputs are all REFUSED '
       '-- the zero time scale cannot even produce a contract identity, so it is rejected before '
       'any F-level arithmetic exists',
       ok7, {'cases': n7}, conclusion='fail-closed on the declared domain')

    # ============================================================ S3-N8 unbound state identity
    rec_n8, exc_n8 = try_call(bound, inst=inst_long, plans=plans_long, entry={0: fresh_entry()},
                              n_max=N_MAX, state_identity=None, expected_state_identity=None,
                              source_identity=None, committed_order_ids=None)
    ck('S3-N8', 'unbound state / source identity or an unbound commitment set can only reach '
       'DIAGNOSTIC_ONLY: an F-level verdict is never issued on a state whose provenance is unknown',
       exc_n8 is None and (rec_n8 or {}).get('verdict') == S3.DIAGNOSTIC_ONLY
       and set((rec_n8 or {}).get('binding_notes') or ()) >= {'STATE_IDENTITY_UNBOUND',
                                                              'SOURCE_IDENTITY_UNBOUND',
                                                              'COMMITMENT_UNBOUND'},
       {'verdict': (rec_n8 or {}).get('verdict'),
        'notes': (rec_n8 or {}).get('binding_notes')},
       conclusion='the conditional certificate is bound to the state it was made for')

    # ============================================================ S3-N9 the skipped-piece budget
    sk = S3.skip_demonstration(CONTRACT, 2, N_MAX)
    ck('S3-N9', 'THE R-LEVEL UPPER END IS NOT AN F-LEVEL ONE: a partition made of n_max pieces of '
       '1e-9 h is early-returned as the IDENTITY, so the executor stays at tau+delta while the ODE '
       'flow has already cooled -- the naive margin-only T-skip under-covers and the corrected '
       '(margin + h*delta) T-skip covers it',
       sk['t_skip_covers_surplus'] and not sk['naive_margin_only_covers_surplus'],
       {k: (float(v) if isinstance(v, F) else v) for k, v in sk.items()},
       conclusion='the dominant F-level error is the executor template itself, not math.exp; this '
                  'is why S3 could not be closed by a better exp bound alone')

    # ============================================================ S3-N10 future information
    fi = S3.future_information_audit(MODULE_PATH)
    served = covered(plans_long)
    rv = np.array(inst_long['reveal_time'], dtype=np.float64)
    rv[[n for n in range(rv.shape[0]) if n not in served]] = 1.0e6
    inst_rv = dict(inst_long)
    inst_rv['reveal_time'] = rv
    rec_a, exc_a = try_call(bound, inst=inst_long, plans=plans_long, entry={0: fresh_entry()},
                            n_max=N_MAX)
    rec_b, exc_b = try_call(bound, inst=inst_rv, plans=plans_long, entry={0: fresh_entry()},
                            n_max=N_MAX)

    def _fingerprint(rec):
        z = (rec or {}).get('per_zone_exact') or {}
        return (str((rec or {}).get('total_budget_E')),
                tuple(sorted((k, str(v['sup_upper_real_level']), str(v['budget_E']),
                              str(v['sup_upper_float_level']),
                              str(v['budget_terms']['T-skip']))
                             for k, v in z.items())),
                (rec or {}).get('verdict'))

    fa, fb = _fingerprint(rec_a), _fingerprint(rec_b)
    ck('S3-N10', 'the bound is decision-time, proved TWO ways: the S3 module contains no reveal '
       'reference at all (AST), and moving every unserved order\'s reveal time to a far future '
       'instant leaves the entire bound -- budget, both widened ends, the T-skip term and the '
       'verdict -- bit-identical',
       fi['ok'] and exc_a is None and exc_b is None and fa == fb,
       {'s3_module_ast_findings': fi['findings'],
        'perturbation': 'reveal_time of every node NOT in the plan set to 1.0e6',
        'fingerprint_identical': fa == fb,
        'fingerprint': {'total_budget_E': fa[0], 'verdict': fa[2],
                        'per_zone_first': fa[1][0] if fa[1] else None},
        'why_S1_may_read_reveal_time': ('S1 reads reveal_time only to check that a SERVED node is '
                                        'visible at the decision clock -- a legality check.  The '
                                        'invariance test above shows that changing reveal times '
                                        'cannot move the bound, so the legality check does not leak '
                                        'into the enclosure')},
       conclusion='no future-segmentation or future-reveal information enters the bound, and that '
                  'is demonstrated by perturbation rather than asserted')

    # ============================================================ S3-N11 empirical kappa refused
    n11 = {}
    for src_name in ('EMPIRICAL_MEASURED', 'FINITE_GRID_CONFIRMED', 'COMMUNITY_CONVENTION',
                     'IEEE754_INFERENCE'):
        r, e = try_call(S3.check_float_outward_bound, contract=CONTRACT,
                        contract_identity=CONTRACT.contract_hash, instance=inst_long, clock=0.0,
                        plans_by_vehicle=plans_long, entry_state_by_vehicle={0: fresh_entry()},
                        kappa=KAPPA_FOR_ARITHMETIC, kappa_source=src_name,
                        kappa_citation='x', kappa_domain='[0,4.8]', n_max=N_MAX,
                        committed_order_ids=covered(plans_long), state_identity='S',
                        expected_state_identity='S', source_identity='SYNTH')
        n11[src_name] = {'verdict': (r or {}).get('verdict'), 'codes': codes_of(r),
                         'f_level_established': (r or {}).get('f_level_established')}
    ck('S3-N11', 'every empirical or inferential kappa source is REFUSED with '
       'KAPPA_SOURCE_NOT_ACCEPTABLE, even though the same value passes the arithmetic -- a measured '
       'kappa is never laundered into a certificate',
       all(v['verdict'] == S3.REFUSED and 'KAPPA_SOURCE_NOT_ACCEPTABLE' in v['codes']
           and v['f_level_established'] is False for v in n11.values()),
       {'cases': n11, 'accepted_sources': list(S3.KAPPA_SOURCES_ACCEPTED)},
       conclusion='the only way in is a document this platform does not have')

    # ============================================================ S3-E1 non-emptiness, no filtering
    planset, idx = [], 1
    for i in range(24):
        n_stops = 2 + (i % 7)
        n_veh = 1 + (i % 3)
        inst = inst_long if i % 2 == 0 else inst_tight
        pl, nxt = {}, 1
        for v in range(n_veh):
            stops = []
            for _ in range(n_stops):
                if nxt >= 31:
                    break
                stops.append(nxt)
                nxt += 1
            pl[v] = {'current_node': 0, 'ready_time': 0.0, 'prefix': [], 'suffix': stops + [0]}
        planset.append((f'plan-{i:02d}', inst, pl))
    outcomes, n_proved, fails = [], 0, []
    n_rej_real, n_rej_float_only = 0, 0
    for name, inst, pl in planset:
        r, e = try_call(bound, inst=inst, plans=pl,
                        entry={v: fresh_entry() for v in pl}, n_max=N_MAX)
        v = (r or {}).get('verdict')
        ex = ((r or {}).get('per_zone_exact') or {})
        real_zones = ((r or {}).get('real_level_receipt') or {}).get('per_zone') or {}
        bad_real = [z for z, i in real_zones.items() if not i.get('band_ok')]
        bad_float = [z for z, i in ex.items() if not i['band_ok_float_level']]
        if v == S3.PROVED_CONDITIONAL_F:
            n_proved += 1
        else:
            if bad_real:
                n_rej_real += 1
            elif bad_float:
                n_rej_float_only += 1
            fails.append({'plan': name, 'verdict': v, 'exception': (e or {}).get('code'),
                          'codes': codes_of(r), 'bad_zones_float_level': bad_float,
                          'bad_zones_real_level': bad_real,
                          'rejected_at': 'real_level' if bad_real else 'float_level_only'})
        outcomes.append({'plan': name, 'n_stops': sum(len(p['suffix']) - 1 for p in pl.values()),
                         'n_vehicles': len(pl), 'verdict': v,
                         'max_band_margin_c': min(
                             (float(i['band_high'] - i['sup_upper_float_level'])
                              for i in ex.values()), default=None),
                         'budget_E': float((r or {}).get('total_budget_E') or 0)})
    ck('S3-E1', 'the F-level bound is NON-EMPTY on a pre-declared plan set: every plan is reported '
       'with its verdict, its budget and its band margin, nothing is filtered out, and the '
       'PROVED_CONDITIONAL_F share is stated as it came out -- with the rejections split by WHICH '
       'level rejected them',
       len(outcomes) == 24 and (n_proved + len(fails)) == 24,
       {'n_plans': len(outcomes), 'n_proved_conditional_f': n_proved,
        'n_not_proved': len(fails),
        'n_rejected_at_real_level': n_rej_real,
        'n_rejected_at_float_level_only': n_rej_float_only,
        'failures': fails, 'outcomes': outcomes,
        'kstar_binding': float(min(S3.kappa_star(zth=ZTH[z], n_gaps_z=10, n_doors_z=40,
                                                n_pieces=N_MAX, n_max=N_MAX)['kappa_star']
                                   for z in range(N_ZONES))),
        'note': ('"PROVED_CONDITIONAL_F" here means CONDITIONAL ON A KAPPA THAT WAS SUPPLIED FOR '
                 'ARITHMETIC.  It is not an F-level qualification of this platform.  The failures '
                 'are the ones the REAL-level bracket already rejected (short gaps), i.e. the '
                 'F-level widening is not what rejected them.')},
       conclusion='the bound is not vacuous, the rejections are attributed to the right level, and '
                  'the non-emptiness result is reported without picking favourable plans')

    # ============================================================ S3-E2 the 40 ms gate
    plans_fleet, nxt = {}, 1
    for v in range(8):
        stops = []
        for _ in range(5):
            stops.append(nxt)
            nxt += 1
        plans_fleet[v] = {'current_node': 0, 'ready_time': 0.0, 'prefix': [], 'suffix': stops + [0]}
    entry_fleet = {v: fresh_entry() for v in plans_fleet}

    def run_once():
        return bound(inst=inst_long, plans=plans_fleet, entry=entry_fleet, n_max=N_MAX)

    for _ in range(10):
        run_once()
    samples = []
    for _ in range(60):
        t0 = time.perf_counter()
        run_once()
        samples.append(time.perf_counter() - t0)
    med, mx = statistics.median(samples), max(samples)
    ck('S3-E2', 'the single-candidate qualification gate G-T1 (40 ms, median AND maximum reported) '
       'is met by the F-level certificate on a 40-door fleet',
       mx <= 0.040,
       {'median_s': med, 'max_s': mx, 'n_samples': len(samples),
        'g_t1_seconds': 0.040, 'median_within': med <= 0.040, 'max_within': mx <= 0.040,
        'max_candidates_within_budget_by_median': (0.040 / med) if med else None,
        'reading': 'this is a MEDIAN-of-60 measurement on one machine, not a worst-case guarantee '
                   'over inputs and machine load (round-1 review opinion 2)'},
       conclusion='the cost gate is met on this machine and is reported as a distribution, not as '
                  'a guarantee')

    # ============================================================ S3-E3 the 10 s gate
    ck('S3-E3', 'the full online decision gate G-T2 (10 s) is NOT MEASURED: nothing here times the '
       'scoring, the search, the other mandatory certifications or the accept decision, and those '
       'unmeasured parts MUST NOT be counted as zero cost',
       True,
       {'g_t2_seconds': 10.0, 'status': 'FULL_ONLINE_10S_BUDGET_UNMEASURED',
        'measured_here': ['the F-level certificate only, per candidate'],
        'not_measured': ['MaskCO forward pass', 'search / candidate generation',
                         'the other mandatory certifications', 'the accept decision',
                         'machine-load variation'],
        'reading': ('by review opinion 2 a median of isolated timings is not a worst-case'
                    ' guarantee; the honest status of G-T2 is UNMEASURED, not PASS')},
       conclusion='no claim is made that a decision fits in 10 s')

    # ============================================================ payload
    gated = [r for r in ROWS if r.get('gated', True)]
    payload = {
        'schema': 'q3_s3_float_outward_bound_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'platform': S3.platform_record(),
        'seed': SEED,
        'eps_f_c': float(S3.EPS_F),
        'n_max_per_vehicle': N_MAX,
        'n_max_receipt': nm,
        'kappa_sources_accepted': list(S3.KAPPA_SOURCES_ACCEPTED),
        'kappa_sources_refused': list(S3.KAPPA_SOURCES_REFUSED),
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in gated if r['ok']),
        'n_fail': sum(1 for r in gated if not r['ok']),
        'verdict': ('S3_CONTROLS_PASS' if all(r['ok'] for r in gated) else 'S3_CONTROLS_FAIL'),
        'kappa_verdict': 'EXP_ERROR_BOUND_NOT_ESTABLISHED',
        'f_level_established': False,
        'labels': {'PROVED': 'S3-IMP, S3-SKIP, L-B, L-C, the Theta/exp enclosures',
                   'CONFIRMED': 'S3-P3 grid census, S3-N2/N3 containment walks, S3-P4 numbers',
                   'NOT_ESTABLISHED': 'kappa for this platform -> EXP_ERROR_BOUND_NOT_ESTABLISHED'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL -> %s ==='
          % (len(ROWS), payload['n_pass'], payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
