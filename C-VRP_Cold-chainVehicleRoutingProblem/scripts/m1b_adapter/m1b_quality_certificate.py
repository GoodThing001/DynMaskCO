#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-3 + Q2-5: the Omega10 INITIALIZATION gate and the analytic v2 QUALITY certificate.

Omega10 (new, proposed in the Q1 independent review): every fish lot's ACTUAL inbound initial
quality must equal the contract's `initial_value[c]` within a pre-fixed `epsilon_init = 1e-9`.
The Q1 bound was proved only CONDITIONALLY on this, because
`coldchain_state.py:248-252,270-278` accepts ANY positive initial value, and
`strict_online_env.py:159-161` even defaults to all-ones when the dataset lacks the field.

The certificate uses the CONSERVATIVE initial value `Q0_safe = Q0_contract - epsilon_init`, and
the tolerance-corrected horizon `H_max = max(0, T_end + 1e-6 - t_min) * hours_per_time_unit`
(Omega5-prime), giving

    Q_LB = max(0, Q0_safe - k * H_max)

with the SAME salability semantics as `_delivery_record_v2`:
`Q_LB >= threshold - 1e-9`.

SCOPE: v2 zero-order + absolute-threshold classes ONLY.  Tomato (first order, threshold_type
'none') is out of scope and its `salable` is None -- it must NEVER be counted as unsalable, and
this certificate must NEVER be reported as certifying the tomato's continuous quality.

NO FUTURE INFORMATION: this gate reads only (a) the lots actually in the candidate plans'
vehicles' manifests, (b) the currently revealed orders the candidate plans reference, and
(c) currently committed visible orders.  It never scans unrevealed orders' true initial values;
those are checked at the reveal / accept / pickup boundary instead (`check_boundary_initial_value`).
"""
from __future__ import annotations

OMEGA10_OK = 'OMEGA10_OK'
OMEGA10_INITIAL_VALUE_MISMATCH = 'OMEGA10_INITIAL_VALUE_MISMATCH'
OMEGA10_LOT_FIELD_INVALID = 'OMEGA10_LOT_FIELD_INVALID'
OMEGA10_NO_RELEVANT_LOTS = 'OMEGA10_NO_RELEVANT_LOTS'

QUALITY_LB_OK = 'QUALITY_LB_OK'
QUALITY_LB_BELOW_THRESHOLD = 'QUALITY_LB_BELOW_THRESHOLD'
QUALITY_CLASS_OUT_OF_SCOPE = 'QUALITY_CLASS_OUT_OF_SCOPE'
QUALITY_TIME_UNIT_INVALID = 'QUALITY_TIME_UNIT_INVALID'
QUALITY_NUMERIC_INVALID = 'QUALITY_NUMERIC_INVALID'
QUALITY_HORIZON_UNAVAILABLE = 'QUALITY_HORIZON_UNAVAILABLE'

EPSILON_INIT = 1e-9            # PRE-FIXED by the spec; not to be tuned per run
EPSILON_THRESHOLD = 1e-9       # matches _delivery_record_v2
EPSILON_RETURN = 1e-6          # matches action_contract.certify_route / the A-v1 evaluator


def _finite(x):
    try:
        f = float(x)
    except Exception:                                                   # noqa: BLE001
        return False
    return f == f and f not in (float('inf'), float('-inf'))


# ---------------------------------------------------------------------------------------------
# Omega10
# ---------------------------------------------------------------------------------------------
def check_omega10_initialization(lots, per_class_constants, *, epsilon_init=EPSILON_INIT):
    """Gate: every in-scope fish lot's ACTUAL initial quality matches the contract value.

    `lots` is an iterable of objects with `temp_class` and `initial_quality` -- in practice the
    cargo-manifest lots of the vehicles involved in the candidate plans, plus any revealed order
    the candidate plans reference.  Unrevealed orders are NEVER passed in here by the caller.

    Returns (ok, verdict, detail).
    """
    if not _finite(epsilon_init) or epsilon_init < 0:
        return False, OMEGA10_LOT_FIELD_INVALID, {'field': 'epsilon_init',
                                                  'got': repr(epsilon_init)}
    checked, out_of_scope, mismatches, bad = 0, 0, [], []
    q0_safe = {}
    for lot in lots or ():
        c = getattr(lot, 'temp_class', None)
        iq = getattr(lot, 'initial_quality', None)
        if c is None or iq is None:
            bad.append({'order_id': getattr(lot, 'order_id', None), 'why': 'missing field'})
            continue
        if not _finite(c) or not _finite(iq):
            bad.append({'order_id': getattr(lot, 'order_id', None), 'why': 'non-finite',
                        'temp_class': repr(c), 'initial_quality': repr(iq)})
            continue
        c = int(c)
        const = (per_class_constants or {}).get(c)
        if const is None:
            bad.append({'order_id': getattr(lot, 'order_id', None),
                        'why': 'temperature class absent from contract', 'temp_class': c})
            continue
        if not const.get('in_scope', True):
            out_of_scope += 1
            continue
        checked += 1
        want = float(const['initial_value'])
        got = float(iq)
        q0_safe[c] = want - float(epsilon_init)
        if abs(got - want) > float(epsilon_init):
            mismatches.append({'order_id': getattr(lot, 'order_id', None), 'temp_class': c,
                               'expected': want, 'actual': got,
                               'abs_dev': abs(got - want), 'epsilon_init': float(epsilon_init)})
    detail = {'n_checked': checked, 'n_out_of_scope': out_of_scope,
              'n_mismatch': len(mismatches), 'mismatches': mismatches[:20],
              'n_field_errors': len(bad), 'field_errors': bad[:20],
              'epsilon_init': float(epsilon_init), 'q0_safe_used': q0_safe}
    if bad:
        return False, OMEGA10_LOT_FIELD_INVALID, detail
    if mismatches:
        return False, OMEGA10_INITIAL_VALUE_MISMATCH, detail
    if checked == 0:
        # No in-scope lots to check.  This is NOT a failure (a day may genuinely hold no fish at
        # this decision point), but it is reported distinctly so it can never be mistaken for a
        # positive verification of something that was not examined.
        return True, OMEGA10_NO_RELEVANT_LOTS, detail
    return True, OMEGA10_OK, detail


def check_boundary_initial_value(order, per_class_constants, *, epsilon_init=EPSILON_INIT):
    """Equivalent check at the reveal / accept / pickup boundary for a SINGLE order.

    This is how a future order is validated once it becomes known: at the moment it is revealed,
    accepted or actually picked, its true initial quality is checked against the contract.  That
    keeps the invariant 'every fish lot's inbound value equals the contract value' without ever
    reading an unrevealed order's data during an online decision.
    """
    return check_omega10_initialization([order], per_class_constants,
                                        epsilon_init=epsilon_init)


def required_initial_value_invariant_report():
    """The loading-pipeline requirement, stated so it can be audited (not a runtime check)."""
    return {
        'invariant': 'for every temperature class c and every fish order i: '
                     'initial_quality(i) == QualityConfigV2.initial_value[c] within '
                     'epsilon_init = 1e-9',
        'current_enforcement': {
            'runtime_guarantee': 'NONE (this gate)',
            'coldchain_state.py:248-252,270-278': 'accepts ANY positive initial_quality',
            'strict_online_env.py:159-161': 'takes dataset initial_quality, DEFAULTS TO ALL ONES',
            'coldchain_evaluator_a1.add_v2_initial_quality:20-33': 'the intended converter; a '
                                                                   'plain function, applied by '
                                                                   'calling convention at 60+ sites',
        },
        'boundaries_where_the_invariant_must_hold': [
            'data loading (dataset initial_quality == add_v2_initial_quality output)',
            'reveal / accept (check_boundary_initial_value)',
            'actual pickup (the lot written into CargoLotState)',
        ],
        'why_not_scanned_online': 'scanning unrevealed orders during an online decision would use '
                                  'non-public information; the gate therefore reads only manifest '
                                  'lots, candidate-referenced revealed orders and committed '
                                  'visible orders.',
    }


# ---------------------------------------------------------------------------------------------
# Q2-5  the analytic certificate
# ---------------------------------------------------------------------------------------------
def horizon_hours(T_end, t_min, hours_per_time_unit, *,
                  epsilon_return=EPSILON_RETURN, require_t_min_verified=True):
    """Omega5-prime: H_max = max(0, (T_end + eps_return) - t_min) * hours_per_time_unit.

    `t_min` is the VERIFIED earliest pickup_finish.  When it cannot be verified the caller must
    pass 0.0 (the clock origin), which is the conservative choice.
    """
    if not _finite(T_end) or not _finite(t_min) or not _finite(hours_per_time_unit):
        return None, QUALITY_NUMERIC_INVALID, {'T_end': repr(T_end), 't_min': repr(t_min),
                                               'hours_per_time_unit': repr(hours_per_time_unit)}
    if hours_per_time_unit <= 0:
        return None, QUALITY_TIME_UNIT_INVALID, {'hours_per_time_unit': hours_per_time_unit}
    tmin = float(t_min)
    if require_t_min_verified is False:
        tmin = 0.0
    span = (float(T_end) + float(epsilon_return)) - tmin
    if span < 0:
        span = 0.0
    return span * float(hours_per_time_unit), None, {
        'T_end': float(T_end), 't_min': tmin, 'epsilon_return': float(epsilon_return),
        'hours_per_time_unit': float(hours_per_time_unit),
        'note': 'NOT a linear rescaling of a unitless horizon; recomputed from the formula'}


def certify_quality_lb(per_class_constants, *, T_end, t_min, hours_per_time_unit,
                       epsilon_init=EPSILON_INIT, epsilon_threshold=EPSILON_THRESHOLD):
    """Return (result, verdict, detail).

    `result[c]` carries `q_lb`, `threshold`, `salable` for every IN-SCOPE class.  Out-of-scope
    classes are reported with `in_scope=False` and NO salable claim.
    """
    H, verdict, hdetail = horizon_hours(T_end, t_min, hours_per_time_unit)
    if H is None:
        return None, verdict, hdetail
    out, below = {}, []
    any_in_scope = False
    for c, const in sorted((per_class_constants or {}).items()):
        if not const.get('in_scope', True):
            out[c] = {'in_scope': False, 'dynamics_order': const.get('dynamics_order'),
                      'threshold_type': const.get('threshold_type'),
                      'salable': None, 'claim': 'NOT_CERTIFIED_BY_THIS_BOUND'}
            continue
        any_in_scope = True
        q0_safe = float(const['q0_safe'])
        k = float(const['rate_per_hour'])
        thr = float(const['threshold'])
        q_lb = max(0.0, q0_safe - k * H)
        if not _finite(q_lb):
            return out, QUALITY_NUMERIC_INVALID, {'class': c, 'q_lb': repr(q_lb)}
        salable = q_lb >= thr - float(epsilon_threshold)
        out[c] = {'in_scope': True, 'q0_contract': float(const['initial_value']),
                  'q0_safe': q0_safe, 'k': k, 'H_max_hours': H, 'q_lb': q_lb,
                  'threshold': thr, 'epsilon_threshold': float(epsilon_threshold),
                  'salable': salable,
                  'margin': q_lb - thr,
                  'H_crit': (float(const['initial_value']) - thr + float(epsilon_threshold)) / k
                  if k > 0 else None}
        if not salable:
            below.append(c)
    detail = {'H_max': H, 'horizon_detail': hdetail, 'per_class': out,
              'classes_below_threshold': below,
              'tomato_note': 'class 0 (first order, threshold_type none) is OUT OF SCOPE: '
                             'salable is None and is NOT counted as unsalable; this bound makes '
                             'NO claim about the tomato continuous metric'}
    if not any_in_scope:
        return out, QUALITY_CLASS_OUT_OF_SCOPE, detail
    if below:
        return out, QUALITY_LB_BELOW_THRESHOLD, detail
    return out, QUALITY_LB_OK, detail
