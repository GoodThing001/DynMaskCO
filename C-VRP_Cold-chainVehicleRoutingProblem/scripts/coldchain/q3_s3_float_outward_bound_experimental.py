#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 S3: the EXECUTOR-FLOAT (F-level) outward bound.

WHAT S3 HAS TO DELIVER
======================
A bound that is ALL of:

  (i)   DECISION-TIME computable -- a function of (frozen prefix, candidate plan, instance arrays,
        contract, declared kappa) only, never of a future reveal clock or of a partition that is
        itself future information;
  (ii)  an OUTWARD enclosure of what the executor actually STORES in
        `VehicleColdChainState.compartment_temperature_c` (a binary64 per zone);
  (iii) valid for EVERY admissible partitioning of every door-less interval, not just the one a
        recorded run happened to use.

Bit-identical replay is DELETED as an option.  It is already proved that the executor's partition is
cut at reveal clocks and that refining a door-less interval CHANGES the terminal value (1 segment
`4.054313` vs 4 segments `4.054433`, `1.20e-04 C`), so replay of a future suffix is not a property
that exists at decision time.  Replay answers "did this happen?"; S3 needs "can that happen?".

THE ONE THING S3 CANNOT GET FOR FREE -- and the lemma that says so
==================================================================
S3-IMP (impossibility).  Let the bound have to hold for every admissible partition of a gap of total
`D`, and take the partition into `k` pieces; the executor's end value is a function of the `k`
platform values `E_i = fl_exp(-h*D/k)` with `dC/dE_i != 0` (magnitude `|v_i - A|`).  The executor
contract, IEEE-754 and the C standard say nothing about the VALUE of `fl_exp` beyond "it is the
platform's exponential".  Hence:

  * the ASSUMPTION-FREE bound is `fl_exp in [0, 1]` (non-positive argument), which needs every
    door-less segment to last at least `(A - tau_z)/c_z` = 3.5 / 4.2 / 5.375 h -- an EMPTY
    certificate for any admissible plan (`q3_strict_float_experimental.exp_free_worst_case`);
  * every non-trivial bound therefore IMPLIES a uniform relative-accuracy statement on `fl_exp`
    over the plan's argument set, i.e. a `kappa`.

So S3's whole outcome reduces to the QUALIFICATION of kappa, and this module refuses to pretend
otherwise: `check_float_outward_bound` refuses (`EXP_ERROR_BOUND_NOT_ESTABLISHED`) unless the caller
supplies a kappa whose source is one of the three documented kinds, and `f_level_established` stays
False even when the arithmetic lands inside the band.

THE ERROR BUDGET (every term has a closed form; nothing is measured into it)
===========================================================================
For zone z, over the whole route (all terms are exact rationals):

    T-kappa  = kappa * S_z * n_pieces        S_z = A - tau_z  (the largest |v - A| reachable)
    T-round  = n_gaps * n_pieces * N_OPS * u * V_max
    T-door   = n_doors_z * N_OPS_DOOR * u * V_max
    T-time   = u * V_max                     (`end - depart` and the `* hours_per_time_unit`)
    T-skip   = margin_z * n_max * _EPS_TIME  (`coldchain_state.py:361-362` early-returns the
                                              IDENTITY when duration_h <= 1e-9, so that elapsed time
                                              is never cooled -- see S3-SKIP below)
    T-floor  = 0                             (`max(tau, .)` is exact and can only raise; registered
                                              explicitly so it cannot be silently dropped)

    E_z = T-kappa + T-round + T-door + T-time + T-skip

S3-SKIP is the finding that matters most: the dominant F-level error term is NOT `math.exp`'s
accuracy but the executor's `duration_h <= 1e-9` early return.  A partition may contain at most
`n_max` advances per vehicle, so at most `n_max * 1e-9 h` of elapsed time is never cooled, worth up
to `margin_z * n_max * 1e-9` = 1.20e-5 / 3.28e-5 / 5.68e-5 C.  That is ~8 orders of magnitude above
the entire rounding budget, so S1's ODE upper end is NOT an upper bound for the executor even with a
perfect `exp`.  `skip_demonstration` exhibits it against the real executor.

WHAT THIS MODULE DOES NOT DO
============================
  * it does not establish that any platform's `math.exp` meets any kappa -- that is the audit's job,
    and its verdict is `EXP_ERROR_BOUND_NOT_ESTABLISHED` unless a documented source is supplied;
  * it makes no claim about an intra-segment (continuous) profile: the executor defines none;
  * it says nothing about quality, energy, time windows, capacity or joint certification;
  * it is not wired into any certifier and `production_consumable` is hard-coded False.
"""
from __future__ import annotations

import ast
import math
import platform
import random
import sys
from fractions import Fraction as F
from pathlib import Path

import q3_g2_time_identity_experimental as G2
import q3_s1_route_bracket_experimental as S1
import q3_strict_float_experimental as SF

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_S3_FLOAT_OUTWARD_BOUND_EXPERIMENTAL_V1'
SCHEMA = 'q3_s3_float_outward_bound'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL_F = 'PROVED_CONDITIONAL_F'

#: pre-registered in `q3_s3_preregistration_2026-10-17.md` E2.3 -- not a free parameter and not
#: adjustable after seeing a result.
EPS_F = F(1, 1000)
#: the executor's own short-circuit threshold (`coldchain_state.py:_EPS`)
EPS_TIME = F(1, 10 ** 9)
#: elementary operations per door-less piece / per door event, counted from the source
N_OPS_PER_PIECE = 7
N_OPS_DOOR = 2
#: the three admissible kappa sources (pre-registration E2.1).  Anything else is refused, including
#: every empirical / grid-measured / "all real libms do it" argument.
KAPPA_SOURCES_ACCEPTED = ('FORMAL_SPECIFICATION', 'VENDOR_DOCUMENTED', 'CERTIFIED_IMPLEMENTATION')
KAPPA_SOURCES_REFUSED = ('EMPIRICAL_MEASURED', 'FINITE_GRID_CONFIRMED', 'COMMUNITY_CONVENTION',
                         'IEEE754_INFERENCE', 'NONE')
EXP_ERROR_BOUND_NOT_ESTABLISHED = 'EXP_ERROR_BOUND_NOT_ESTABLISHED'

#: functions allowed to touch `math` -- measurement only, never a verdict
MATH_MEASUREMENT_ONLY = frozenset({'exp_pointwise_error_bound', 'grid_census', 'platform_record',
                                   'skip_demonstration', 'partition_spread'})
FORBIDDEN_IN_DECISION = frozenset({'exp', 'expm1', 'log', 'log1p', 'log2', 'log10', 'pow'})

PROPOSITIONS = (
    'S3-IMP_NO_ASSUMPTION_FREE_OUTWARD_BOUND_IS_NON_TRIVIAL',
    'S3-SKIP_THE_EXECUTOR_EARLY_RETURN_SKIPS_COOLING',
    'L-B_ROUND_TO_NEAREST_RELATIVE_BOUND',
    'L-C_MONOTONICITY_OF_THE_EXECUTOR_STEP',
    'L-A_RATIONAL_EXP_ENCLOSURE_USED_ONLY_TO_WITNESS_THE_KAPPA_GAP',
    'S1_ROUTE_BRACKET_REUSED_VERBATIM_ONE_CONVENTION',
)
NOT_CLAIMED = (
    'that any platform math.exp meets any kappa: the audit verdict is '
    'EXP_ERROR_BOUND_NOT_ESTABLISHED unless a documented source is supplied',
    'an intra-segment (continuous) profile: the executor defines no intra-segment trajectory, so '
    'nothing here says anything about the temperature between two segment boundaries',
    'quality salability, energy budget, time windows, capacity or joint certification',
    'THERMAL_CONTINUOUS_NOT_ESTABLISHED / PRODUCTION_NOT_QUALIFIED / '
    'Q3_FORMAL_FREEZE_NOT_AUTHORISED: none of these change even if every row of this round passes',
    'that a future suffix may be certified by replay: replay is deleted as an option',
)


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                     # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


# ---------------------------------------------------------------------------------------------
# 1. platform record, kappa audit, pointwise (assumption-free) witness
# ---------------------------------------------------------------------------------------------
def platform_record():
    """Identity of the platform whose `math.exp` any kappa statement would be about."""
    return {
        'sys_version': sys.version.replace('\n', ' '),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'float_info': {k: getattr(sys.float_info, k) for k in
                       ('radix', 'mant_dig', 'rounds', 'epsilon', 'max', 'min')},
        'exp_semantics': ('CPython mathmodule.c calls the C library exp; CPython does not implement '
                          'exp itself, so an accuracy statement is a statement about the platform '
                          'libm, not about Python'),
        'read_at': 'run time',
    }


def exp_pointwise_error_bound(x):
    """Rigorous, ASSUMPTION-FREE relative-error bound of `math.exp(-x)` AT THIS x.

    `[lo, hi]` encloses the TRUE `exp(-x)` (exact rational, `SF.exp_neg_enclosure`); hence for the
    true value `t in [lo, hi]`, `|fl_exp(-x) - t| <= max(|v - lo|, |hi - v|)`.  Nothing is assumed
    about the platform.  This is a statement about ONE argument: it can WITNESS the kappa gap, it
    can never certify a plan -- which is exactly what S3-IMP says.
    """
    if not isinstance(x, F):
        raise SF.Refusal('NOT_AN_EXACT_RATIONAL', {'field': 'x', 'type': type(x).__name__})
    if x < 0:
        raise SF.Refusal('NEGATIVE_EXPONENT', {'x': str(x)})
    lo, hi = SF.exp_neg_enclosure(x)
    if lo == hi:
        return {'x': x, 'true_lower': lo, 'true_upper': hi, 'executor_value': math.exp(-float(x)),
                'abs_error_upper': F(0), 'rel_error_upper': F(0), 'ulp_multiple': 0.0}
    v = math.exp(-float(x))
    err = max(abs(F(v) - lo), abs(hi - F(v)))
    return {'x': x, 'true_lower': lo, 'true_upper': hi, 'executor_value': v,
            'abs_error_upper': err, 'rel_error_upper': err / lo,
            'ulp_multiple': float(err / lo) / 2.0 ** -52}


def grid_census(arguments):
    """Observed pointwise error over a declared grid.  CONFIRMED ONLY -- never a proof."""
    worst, n = F(0), 0
    for a in arguments:
        r = exp_pointwise_error_bound(F(float(a)))
        n += 1
        if r['rel_error_upper'] > worst:
            worst = r['rel_error_upper']
    return {'n_arguments': n, 'max_observed_rel_error': worst,
            'max_observed_ulp': (float(worst) / 2.0 ** -52) if worst else 0.0,
            'label': 'CONFIRMED_ON_A_FINITE_GRID_NOT_PROVED',
            'reading': ('a finite grid can FALSIFY a kappa claim; it can never establish one, because '
                        'the executor may evaluate at any argument in the interval and nothing in '
                        'the executor contract bounds the platform libm')}


def exp_accuracy_audit(*, kappa, kappa_source, citation=None, domain=None):
    """Decide whether a kappa may be used in an F-level claim.

    Only the three documented sources qualify; every empirical or inferential source is refused with
    `KAPPA_SOURCE_NOT_ACCEPTABLE`, and a missing kappa with `EXP_ERROR_BOUND_NOT_ESTABLISHED`.
    """
    rec = {'kappa': None, 'kappa_source': kappa_source, 'citation': citation, 'domain': domain,
           'acceptable': False, 'code': None, 'reason': None}
    if kappa is None or kappa_source in (None, 'NONE'):
        rec['code'] = EXP_ERROR_BOUND_NOT_ESTABLISHED
        rec['reason'] = ('no kappa was supplied; by S3-IMP every non-trivial outward bound implies a '
                         'uniform relative-accuracy statement about the platform math.exp, and the '
                         'executor contract, IEEE-754 and the C standard provide none')
        return rec
    if kappa_source in KAPPA_SOURCES_REFUSED or kappa_source not in KAPPA_SOURCES_ACCEPTED:
        rec['code'] = 'KAPPA_SOURCE_NOT_ACCEPTABLE'
        rec['reason'] = {'supplied': kappa_source, 'accepted': list(KAPPA_SOURCES_ACCEPTED),
                         'why': ('an empirical value, a finite-grid check, "every real libm does '
                                 'it" and an IEEE-754 inference are all excluded by the '
                                 'pre-registration E2.1: a measured kappa is not a proof, and this '
                                 'module will not launder one into a certificate')}
        return rec
    try:
        k = kappa if isinstance(kappa, F) else F(kappa)
    except (TypeError, ValueError):
        rec['code'] = 'KAPPA_NOT_A_NUMBER'
        return rec
    if k < 0:
        rec['code'] = 'KAPPA_NEGATIVE'
        return rec
    if not citation:
        rec['code'] = 'KAPPA_CITATION_MISSING'
        rec['reason'] = 'a documented source must name the document and the clause it relies on'
        return rec
    if not domain:
        rec['code'] = 'KAPPA_DOMAIN_MISSING'
        rec['reason'] = 'a documented bound is only usable over the input domain it covers'
        return rec
    rec.update({'kappa': k, 'acceptable': True})
    return rec


# ---------------------------------------------------------------------------------------------
# 2. structural constants reused, never copied
# ---------------------------------------------------------------------------------------------
def n_max_advances(env_source_path):
    """Recompute `n_max_per_vehicle` from the G-2 call-graph census instead of copying 20003."""
    text = Path(env_source_path).read_text(encoding='utf-8')
    census = G2.call_graph_census(text, target='_advance_coldchain_to')
    return {'n_max_per_vehicle': int(census['n_max_per_vehicle']),
            'formula': census.get('n_max_formula'), 'max_iter': census.get('max_iter'),
            'recomputed_not_copied': True}


class _ZoneView(object):
    """A duck-typed zone carrying the exact fields `SF.exp_free_worst_case` reads."""

    def __init__(self, v):
        self.zone = v['zone']
        self.A = v['A']
        self.tau = v['tau']
        self.c = v['c']


def no_assumption_free_bound(contract):
    """S3-IMP as data: with only `fl_exp in [0, 1]` the required segment length is absurd."""
    zth, _n_ln = S1.zone_thresholds_exact(contract)
    rows = {}
    for z, v in sorted(zth.items()):
        need = SF.exp_free_worst_case(_ZoneView(v))
        rows[str(z)] = {'required_segment_h_without_exp_assumption': need,
                        'required_segment_min': float(need) * 60.0,
                        'smallest_theta_z_h': v['theta_hi']}
    worst = max((r['required_segment_h_without_exp_assumption'] for r in rows.values()), default=F(0))
    return {'rows': rows, 'binding_required_segment_h': worst,
            'verdict': 'ASSUMPTION_FREE_OUTWARD_BOUND_IS_EMPTY',
            'reading': ('if nothing is assumed about math.exp, every door-less segment must last at '
                        'least (A - tau)/c > 3.5 h, which no admissible service or travel segment '
                        'reaches, so the certificate is EMPTY.  That is why S3 reduces to kappa.')}


# ---------------------------------------------------------------------------------------------
# 3. the F-level error budget
# ---------------------------------------------------------------------------------------------
def budget_terms(*, zth, n_gaps_z, n_doors_z, n_pieces, n_max, kappa):
    """Every F-level error term as an exact rational.  No measurement enters this function."""
    if kappa is None:
        raise SF.Refusal(EXP_ERROR_BOUND_NOT_ESTABLISHED, {'why': 'budget requested without kappa'})
    kappa = kappa if isinstance(kappa, F) else F(kappa)
    if kappa < 0:
        raise SF.Refusal('KAPPA_NEGATIVE', {'kappa': str(kappa)})
    for name, v in (('n_gaps_z', n_gaps_z), ('n_doors_z', n_doors_z), ('n_pieces', n_pieces),
                    ('n_max', n_max)):
        if isinstance(v, bool) or int(v) != v or v < 0:
            raise SF.Refusal('BAD_BUDGET_COUNT', {'field': name, 'value': repr(v)})
    u = SF.UNIT_ROUNDOFF
    s_z = zth['A'] - zth['tau']
    v_max = max(abs(zth['A']), abs(zth['tau']) + zth['delta'], F(1))
    # T-skip uses (margin + h*delta), NOT margin alone: `T - F_d(T) = (T - T_inf)(1 - e^{-h d}) <=
    # (T - T_inf)*h*d` and `T - T_inf <= (tau - T_inf) + delta = margin/h + delta`, so the cooling
    # that a skipped piece loses is at most (margin + h*delta)*d.  The pre-registration froze
    # `margin * n_max * eps`, which UNDER-covers by the factor `1 + h*delta/margin` (0.507 % for
    # zone 2) -- measured, then corrected here; see the report ERRATA-S3-1.
    t_skip_rate = zth['drift_margin'] + zth['h'] * zth['delta']
    terms = {
        'T-kappa': kappa * s_z * int(n_pieces),
        'T-round': F(int(n_gaps_z) * int(n_pieces) * N_OPS_PER_PIECE) * u * v_max,
        'T-door': F(int(n_doors_z) * N_OPS_DOOR) * u * v_max,
        'T-time': u * v_max,
        'T-skip': t_skip_rate * int(n_max) * EPS_TIME,
        'T-floor': F(0),
    }
    total = sum(terms.values(), F(0))
    dominant = max(terms, key=lambda k: terms[k])
    return {'terms': terms, 'total': total, 'dominant_term': dominant,
            'dominant_share': (float(terms[dominant] / total) if total > 0 else 0.0),
            'constants': {'S_z': s_z, 'V_max': v_max, 'u': u, 'n_pieces': int(n_pieces),
                          'n_max': int(n_max), 'n_gaps_z': int(n_gaps_z),
                          'n_doors_z': int(n_doors_z), 'eps_time': EPS_TIME,
                          't_skip_rate': t_skip_rate},
            'eps_f': EPS_F, 'within_eps_f': total <= EPS_F}


def kappa_star(*, zth, n_gaps_z, n_doors_z, n_pieces, n_max):
    """The largest kappa for which `E_z <= EPS_F`.  Derived, never chosen."""
    base = budget_terms(zth=zth, n_gaps_z=n_gaps_z, n_doors_z=n_doors_z, n_pieces=n_pieces,
                        n_max=n_max, kappa=F(0))
    fixed = base['total']
    s_z = zth['A'] - zth['tau']
    denom = s_z * int(n_pieces)
    if fixed >= EPS_F:
        return {'kappa_star': F(0), 'kappa_independent_total': fixed,
                'dominant_term_without_kappa': base['dominant_term'],
                'why': ('the kappa-independent terms alone already exceed eps_F, so NO kappa '
                        'qualifies -- the outward bound would be empty')}
    if denom <= 0:                                                        # pragma: no cover
        return {'kappa_star': None, 'why': 'degenerate zone (A == tau)'}
    return {'kappa_star': (EPS_F - fixed) / denom, 'kappa_independent_total': fixed,
            'sensitivity_S_z': s_z, 'n_pieces': int(n_pieces),
            'dominant_term_without_kappa': base['dominant_term'],
            'why': 'solved from kappa * S_z * n_pieces + fixed <= eps_F'}


# ---------------------------------------------------------------------------------------------
# 4. the conditional F-level bound (machinery only -- never an establishment)
# ---------------------------------------------------------------------------------------------
def _gap_lower_min(r_receipt, z, default):
    lows = []
    for prop in ((r_receipt.get('per_zone_detail') or {}).get(str(z)) or {}).get(
            'per_vehicle', {}).values():
        for g in prop.get('gaps') or []:
            lows.append(g['lower_before_door'])
    return min(lows) if lows else default


def conditional_bound(*, contract, contract_identity, instance, clock, plans_by_vehicle,
                      entry_state_by_vehicle, kappa, n_max, committed_order_ids=None,
                      revealed_nodes=None, state_identity=None, expected_state_identity=None,
                      source_identity=None, active_zone_mask=None, travel_fn=None,
                      reveal_tolerance=S1.DEFAULT_REVEAL_TOLERANCE, thresholds=None,
                      drop_terms=()):
    """Widen S1's exact real bracket by the F-level budget and evaluate the containment predicate.

    ALWAYS returns `f_level_established=False`: this entry point exists to exercise and falsify the
    machinery and must never be readable as an F-level qualification.  The qualifying entry point is
    `check_float_outward_bound`.
    """
    r = S1.check_route_certificate(
        contract=contract, contract_identity=contract_identity, instance=instance, clock=clock,
        plans_by_vehicle=plans_by_vehicle, entry_state_by_vehicle=entry_state_by_vehicle,
        committed_order_ids=committed_order_ids, revealed_nodes=revealed_nodes,
        state_identity=state_identity, expected_state_identity=expected_state_identity,
        source_identity=source_identity, active_zone_mask=active_zone_mask, travel_fn=travel_fn,
        reveal_tolerance=reveal_tolerance, thresholds=thresholds)
    if r['verdict'] == S1.REFUSED:
        return _receipt(REFUSED, list(r.get('refusals') or []), r.get('binding_notes'), {}, None,
                        kappa, n_max, 'MACHINERY_DEMONSTRATION_ONLY', r)
    zth_all = ({int(k): v for k, v in thresholds.items()} if thresholds is not None
               else S1.zone_thresholds_exact(contract)[0])
    zth = {int(k): v for k, v in (r['per_zone_detail'] or {}).items()}
    per_zone, per_zone_exact = {}, {}
    for z, info in sorted(zth.items()):
        zc = zth_all[z]
        n_gaps_z = int(info.get('n_gaps') or 0)
        n_doors_z = int(info.get('n_doors') or 0)
        b = budget_terms(zth=zc, n_gaps_z=n_gaps_z, n_doors_z=n_doors_z, n_pieces=n_max,
                         n_max=n_max, kappa=kappa)
        e = sum((v for k, v in b['terms'].items() if k not in set(drop_terms)), F(0))
        sup_r = info['sup_upper']
        hi_f = sup_r + e
        lo_f = _gap_lower_min(r, z, zc['band_low']) - e
        band_ok = hi_f <= info['band_high'] + EPS_F
        hard_ok = (hi_f <= info['hard_high']) and (lo_f >= info['hard_low'])
        per_zone_exact[str(z)] = {
            'zone': z, 'n_gaps': n_gaps_z, 'n_doors': n_doors_z,
            'sup_upper_real_level': sup_r, 'budget_E': e, 'budget_terms': b['terms'],
            'dominant_term': b['dominant_term'], 'dominant_share': b['dominant_share'],
            'sup_upper_float_level': hi_f, 'lower_float_level': lo_f,
            'band_high': info['band_high'], 'hard_low': info['hard_low'],
            'hard_high': info['hard_high'], 'eps_f': EPS_F,
            'band_ok_float_level': band_ok,
            'band_excess_over_eps': max(F(0), hi_f - info['band_high'] - EPS_F),
            'hard_ok_float_level': hard_ok, 'kstar': kappa_star(
                zth=zc, n_gaps_z=n_gaps_z, n_doors_z=n_doors_z, n_pieces=n_max, n_max=n_max),
        }
        per_zone[str(z)] = {k: (float(v) if isinstance(v, F) else v)
                            for k, v in per_zone_exact[str(z)].items()
                            if k in ('zone', 'n_gaps', 'n_doors', 'sup_upper_real_level',
                                     'budget_E', 'sup_upper_float_level', 'band_high',
                                     'band_excess_over_eps', 'band_ok_float_level',
                                     'hard_ok_float_level', 'dominant_term', 'dominant_share')}
    bad = [z for z, i in per_zone_exact.items() if not i['band_ok_float_level']]
    if r['verdict'] == S1.NOT_CERTIFIABLE:
        verdict = NOT_CERTIFIABLE
    elif bad:
        verdict = NOT_CERTIFIABLE
    elif r.get('binding_notes'):
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL_F
    return _receipt(verdict, list(r.get('refusals') or []), r.get('binding_notes'), per_zone,
                    per_zone_exact, kappa, n_max, 'MACHINERY_DEMONSTRATION_ONLY', r)


def check_float_outward_bound(*, contract, contract_identity, instance, clock, plans_by_vehicle,
                              entry_state_by_vehicle, kappa, kappa_source, n_max,
                              kappa_citation=None, kappa_domain=None, committed_order_ids=None,
                              revealed_nodes=None, state_identity=None,
                              expected_state_identity=None, source_identity=None,
                              active_zone_mask=None, travel_fn=None,
                              reveal_tolerance=S1.DEFAULT_REVEAL_TOLERANCE, thresholds=None):
    """The qualifying entry point: refuses without a documented kappa and never over-claims."""
    audit = exp_accuracy_audit(kappa=kappa, kappa_source=kappa_source, citation=kappa_citation,
                               domain=kappa_domain)
    if not audit['acceptable']:
        rec = _receipt(REFUSED,
                       [{'code': audit['code'],
                         'detail': {'reason': audit['reason'], 'kappa_source': kappa_source}}],
                       ['KAPPA_UNQUALIFIED'], {}, None, kappa, n_max,
                       'F_LEVEL_QUALIFYING_PATH', None)
        rec['kappa_audit'] = audit
        rec['f_level_established'] = False
        return rec
    rec = conditional_bound(
        contract=contract, contract_identity=contract_identity, instance=instance, clock=clock,
        plans_by_vehicle=plans_by_vehicle, entry_state_by_vehicle=entry_state_by_vehicle,
        kappa=audit['kappa'], n_max=n_max, committed_order_ids=committed_order_ids,
        revealed_nodes=revealed_nodes, state_identity=state_identity,
        expected_state_identity=expected_state_identity, source_identity=source_identity,
        active_zone_mask=active_zone_mask, travel_fn=travel_fn,
        reveal_tolerance=reveal_tolerance, thresholds=thresholds)
    rec['purpose'] = 'F_LEVEL_QUALIFYING_PATH'
    rec['kappa_audit'] = audit
    exact = rec.get('per_zone_exact') or {}
    stars = [i['kstar'].get('kappa_star') for i in exact.values()
             if isinstance(i['kstar'].get('kappa_star'), F)]
    worst = min(stars) if stars else None
    rec['kappa_star_worst_zone'] = worst
    rec['kappa_within_star'] = bool(worst is not None and audit['kappa'] <= worst)
    rec['f_level_established'] = bool(rec['verdict'] == PROVED_CONDITIONAL_F
                                      and rec['kappa_within_star'])
    return rec


def _receipt(verdict, refusals, notes, per_zone, per_zone_exact, kappa, n_max, purpose, r):
    total = sum((i['budget_E'] for i in (per_zone_exact or {}).values()), F(0))
    return {
        'schema': SCHEMA,
        'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
        'production_consumable': PRODUCTION_CONSUMABLE,
        'implementation_version': IMPLEMENTATION_VERSION,
        'purpose': purpose,
        'verdict': verdict,
        'f_level_established': False,     # only the qualifying path may raise this
        'kappa': kappa, 'n_max_per_vehicle': n_max, 'eps_f_c': float(EPS_F),
        'refusals': refusals,
        'binding_notes': list(notes or []),
        'per_zone': per_zone or {},
        'per_zone_exact': per_zone_exact or {},
        'total_budget_E': total,
        'real_level_receipt': r,
        'propositions': list(PROPOSITIONS),
        'not_claimed': list(NOT_CLAIMED),
        'claim': ("conditional on the declared kappa, the executor's stored temperature at every "
                  "door event lies in [lo_S3, hi_S3] for EVERY admissible partition of every "
                  "door-less interval, where lo_S3/hi_S3 are S1's exact real bracket widened by the "
                  "itemised F-level error budget"),
        'not_established': ('the kappa itself: by S3-IMP no non-trivial F-level bound exists without '
                            'a uniform relative-accuracy statement about the platform math.exp, and '
                            'none is available for this platform'),
    }


# ---------------------------------------------------------------------------------------------
# 5. witnesses for the two structural facts
# ---------------------------------------------------------------------------------------------
def skip_demonstration(contract, zone, n_pieces):
    """S3-SKIP: `n_pieces` advances of exactly 1e-9 h are IDENTITY, so the value stays put while
    S1's ODE flow claims it cooled.  Returns the surplus, which is what `T-skip` has to cover."""
    import coldchain_state as cs
    zth, _n = S1.zone_thresholds_exact(contract)
    zth = zth[int(zone)]
    n_zones = len(contract.thermal.supported_temp_classes)
    mask = (True,) * n_zones
    temps = [float(t) for t in contract.thermal.target_temperature_c]
    temps[int(zone)] = float(zth['tau'] + zth['delta'])
    start = temps[int(zone)]
    for _ in range(int(n_pieces)):
        temps, _e = cs._advance_temperatures(tuple(temps), 1e-9, mask, contract)
    executor_end = float(temps[int(zone)])
    flow_end = float(zth['t_infinity'] + (F(zth['tau']) + zth['delta'] - zth['t_infinity'])
                     * SF.exp_neg_enclosure(zth['h'] * F(int(n_pieces)) * EPS_TIME)[0])
    surplus = executor_end - flow_end
    t_skip_rate = zth['drift_margin'] + zth['h'] * zth['delta']
    t_skip = t_skip_rate * int(n_pieces) * EPS_TIME
    t_skip_naive = zth['drift_margin'] * int(n_pieces) * EPS_TIME
    return {'zone': int(zone), 'n_pieces': int(n_pieces), 'start_c': start,
            'executor_end_c': executor_end, 'ode_flow_end_c': flow_end,
            'executor_minus_flow_c': surplus,
            't_skip_budget_c': float(t_skip),
            't_skip_budget_naive_margin_only_c': float(t_skip_naive),
            't_skip_covers_surplus': F(surplus) <= t_skip,
            'naive_margin_only_covers_surplus': F(surplus) <= t_skip_naive,
            'reading': ('the executor early-returns the IDENTITY for duration_h <= 1e-9, so a '
                        'partition made of such pieces never cools: its end value sits ABOVE the ODE '
                        'flow that S1 uses as the upper end.  This is why an F-level bound needs '
                        'T-skip and why S1 alone is not one.  The naive margin-only T-skip '
                        'under-covers, which is EXACTLY why the term carries the h*delta part.')}


def partition_spread(contract, zone, delta, n_samples, seed):
    """S3-N4: different partitions of the SAME total give different executor end values, so a
    single trajectory cannot stand in for the supremum over all partitions."""
    import coldchain_state as cs
    zth, _n = S1.zone_thresholds_exact(contract)
    zth = zth[int(zone)]
    n_zones = len(contract.thermal.supported_temp_classes)
    mask = (True,) * n_zones
    rng = random.Random(int(seed))
    total = float(delta)
    ends = []

    def run(parts):
        temps = [float(t) for t in contract.thermal.target_temperature_c]
        temps[int(zone)] = float(zth['tau'] + zth['delta'])
        for d in parts:
            if d <= 1e-9:
                continue
            temps, _e = cs._advance_temperatures(tuple(temps), float(d), mask, contract)
        return float(temps[int(zone)])

    coarse = run([total])
    ends.append(coarse)
    for _ in range(int(n_samples)):
        k = rng.randint(2, 12)
        cuts = sorted(rng.uniform(0.0, total) for _ in range(k - 1))
        pts = [0.0] + cuts + [total]
        ends.append(run([pts[i + 1] - pts[i] for i in range(len(pts) - 1)]))
    return {'zone': int(zone), 'delta_h': total, 'n_partitions': len(ends),
            'min_end_c': min(ends), 'max_end_c': max(ends),
            'spread_c': max(ends) - min(ends),
            'coarsest_end_c': coarse, 'sup_over_partitions_minus_coarsest_c':
                max(ends) - coarse,
            'single_trajectory_is_not_the_supremum': (max(ends) - coarse) > 0,
            'reading': ('refining a door-less interval raises the end value, so the maximum over '
                        'admissible partitions is strictly above any ONE recorded trajectory; a '
                        'replay-based claim would be unsound by exactly this spread')}


# ---------------------------------------------------------------------------------------------
# 6. mechanical audits
# ---------------------------------------------------------------------------------------------
def _math_calls(path, allowlist):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in allowlist:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                base = sub.func.value
                if isinstance(base, ast.Name) and base.id == 'math':
                    hits.append((node.name, sub.func.attr))
    return hits


def decision_path_audit(module_path, *, allowlist=MATH_MEASUREMENT_ONLY):
    """The decision path must compute the verdict with exact rationals only."""
    findings = [{'function': f, 'call': 'math.' + a} for f, a in _math_calls(module_path, allowlist)
                if a in FORBIDDEN_IN_DECISION]
    return {'ok': not findings, 'findings': findings, 'allowlist': sorted(allowlist),
            'reading': ('math.exp appears ONLY as an oracle under measurement '
                        '(exp_pointwise_error_bound / grid_census / the two witnesses); every '
                        'verdict-relevant quantity is an exact rational')}


def future_information_audit(module_path, *, extra_paths=()):
    """No function that computes lo_S3/hi_S3 may read a reveal clock."""
    forbidden = {'reveal_time', 'revealed_at', 'reveal_events', 'next_reveal'}
    hits = []
    for p in (module_path,) + tuple(extra_paths):
        tree = ast.parse(Path(p).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in forbidden:
                    hits.append({'module': str(p), 'function': node.name, 'name': sub.attr})
                elif isinstance(sub, ast.Name) and sub.id in forbidden:
                    hits.append({'module': str(p), 'function': node.name, 'name': sub.id})
    return {'ok': not hits, 'findings': hits,
            'scanned': [str(module_path)] + [str(p) for p in extra_paths],
            'reading': ('the bound is a function of the frozen prefix, the candidate plan, the '
                        'instance arrays and the declared kappa only; S1 reads reveal_time only to '
                        'check VISIBILITY of a served node, never to choose a partition')}
