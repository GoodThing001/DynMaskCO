#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION  --  S-8 ERRATA: the ANALYTIC distance-multiplier ceilings.

TWO ERRORS RETRACTED HERE
=========================
(1) ASYMMETRIC CEILING.  The previous errata computed the ceiling as `max_x (x + 1/x)/2` over
    `x in [0.85, 1.35]`, giving 1.045370.  The generator does NOT pair a draw with its own
    reciprocal; it pairs a draw with the reciprocal of the TRANSPOSED draw
    (`generate_coldchain_data.py:84-85`):

        A[i,j] = (X[i,j] + 1/(X[j,i] + 1e-8)) / 2

    `X[i,j]` and `X[j,i]` are independent draws, so the supremum over the real domain is

        A_max = (1.35 + 1/(0.85 + 1e-8)) / 2 = 4295000027/3400000040 = 1.263235287...

    which is 1.263235, NOT 1.045370.  With the rush-hour factor (max 2.0) the ceiling is
    2.526471 instead of 2.090741, and the travel bound becomes sqrt(2)*2.526471 = 3.572969
    instead of 2.956992.  The old constant was therefore NOT an upper bound of the domain: an
    admissible instance can exceed it.  The measured D-2 case (2.594) happened to sit below it,
    which is exactly why an empirical case cannot certify an analytic ceiling.

(2) PERTURBATION MULTIPLICITY.  `perturbed_edges` is built from `rng.integers(..., size=n)` PAIRS
    with replacement (`:103-106`), then applied one by one (`:141-143`):

        for (i, j) in perturbed_edges: dist_mat[i, j] *= rng.uniform(1.5, 5.0)

    The same ordered pair can be drawn several times, so the multiplier of one edge is `5^k`, NOT
    `5`.  Nothing in the generator bounds `k` below `n_edge_perturb`, so without a DECLARED `k` the
    perturbation model has no usable ceiling (for N=51, p=0.1 the draw count is 78, i.e. 5^78).
    The previous `PERTURB_MAX = 5.0` is retracted.

WHAT SURVIVES
=============
The per-instance measurements are untouched: for the registered splits `dist_mat` is Euclidean to
within float32 rounding, which is a fact about those 96 instances and is NOT a consequence of any
analytic ceiling.  The two must be reported separately, and that separation is the main content of
this module: an analytic ceiling may be retracted while the per-instance witness still stands.

THE DEPOT COLUMN
================
`_force_return_all` advances to `depart + travel(current_node, 0)` -- the directed edge INTO the
depot (`strict_online_env.py:273`).  A bound on the largest customer-to-customer edge says nothing
about that specific leg, so this module also reports the depot column `max_i dist_mat[i, 0]`, and the
control exercises an amplified depot in-edge through the real forced-return path.
"""
from __future__ import annotations

import math
from fractions import Fraction

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'G2D_ANALYTIC_CEILING_V1'
SCHEMA = 'q3_g2d_analytic_ceiling'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'

# ---- generator ranges, exactly (generate_coldchain_data.py:84-89, :103-106, :141-143) ----------
ASYM_LO = Fraction('0.85')
ASYM_HI = Fraction('1.35')
ASYM_EPS = Fraction('1e-8')
RUSH_MAX = Fraction(2)
PERTURB_HIT_MAX = Fraction(5)

#: the CORRECTED supremum: one draw at its maximum against the transposed draw at its minimum
ASYM_MAX_EXACT = (ASYM_HI + 1 / (ASYM_LO + ASYM_EPS)) / 2
#: the RETRACTED expression, kept only so a control can show it is too small
ASYM_MAX_RETRACTED_EXACT = (ASYM_HI + 1 / ASYM_HI) / 2

#: relative float inflation applied to the analytic bound (deterministic, documented)
REL_INFLATION_ULPS = 4
REL_INFLATION = Fraction(int(REL_INFLATION_ULPS), 2 ** 52)


class Refusal(Exception):
    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail or {}

    def as_dict(self):
        return {'code': self.code, 'detail': self.detail}


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                      # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


def _sqrt2_up():
    """sqrt(2) inflated outward by 2 ulp, as an exact Fraction."""
    v = math.sqrt(2.0)
    for _ in range(2):
        v = math.nextafter(v, math.inf)
    return Fraction(v)


def require_offline(stage):
    if stage != 'offline':
        raise Refusal('ONLINE_MUST_NOT_READ_HIDDEN_DATA', {
            'stage': stage,
            'why': 'the full distance matrix includes edges to unrevealed orders; derive the bound '
                   'once in the qualification stage and consume it as a scalar'})


# =============================================================================================
# the analytic ceiling
# =============================================================================================
def analytic_multiplier(models, perturb_multiplicity=None):
    """Product of the model ceilings, exactly.  A perturbation model NEEDS a declared multiplicity."""
    mult = Fraction(1)
    used, notes = [], []
    for m in models:
        if m == 'euclidean':
            used.append({'model': m, 'ceiling': str(Fraction(1))})
        elif m == 'asymmetric_transit':
            mult *= ASYM_MAX_EXACT * RUSH_MAX
            used.append({'model': m, 'ceiling': str(ASYM_MAX_EXACT * RUSH_MAX),
                         'formula': '(1.35 + 1/(0.85 + 1e-8))/2 * 2.0'})
        elif m == 'long_tail_perturbation':
            if perturb_multiplicity is None:
                raise Refusal('PERTURB_MULTIPLICITY_UNDECLARED', {
                    'why': 'the generator draws edge pairs WITH replacement, so one edge can be '
                           'scaled by 5^k; k is not bounded by the source and must be declared',
                    'ceiling_if_k_were_1': str(PERTURB_HIT_MAX)})
            k = int(perturb_multiplicity)
            if k < 0:
                raise Refusal('PERTURB_MULTIPLICITY_INVALID', {'k': k})
            mult *= PERTURB_HIT_MAX ** k
            used.append({'model': m, 'ceiling': str(PERTURB_HIT_MAX ** k), 'k': k,
                         'formula': '5^k'})
            notes.append('the ceiling grows as 5^k; a declared k is a DOMAIN assumption, not a '
                         'consequence of the generator')
        elif m == 'external_undeclared':
            raise Refusal('DISTANCE_MODEL_UNBOUNDED', {'model': m})
        else:
            raise Refusal('DISTANCE_MODEL_UNKNOWN', {'model': m})
    return {'multiplier_ceiling_exact': mult, 'multiplier_ceiling': float(mult),
            'contributions': used, 'notes': notes}


def travel_bound_analytic(models, tw_speed, perturb_multiplicity=None, *,
                          box_diagonal=1.0):
    """sqrt2*box*sqrt(mult)/speed, inflated by a declared relative amount."""
    if not _finite(tw_speed) or float(tw_speed) <= 0:
        raise Refusal('TW_SPEED_INVALID', {'tw_speed': repr(tw_speed)})
    if not _finite(box_diagonal) or float(box_diagonal) <= 0:
        raise Refusal('COORDINATE_BOX_INVALID', {'box_diagonal': repr(box_diagonal)})
    comp = analytic_multiplier(models, perturb_multiplicity)
    exact = _sqrt2_up() * Fraction(float(box_diagonal)) \
        * comp['multiplier_ceiling_exact'] / Fraction(float(tw_speed))
    inflated = exact * (1 + REL_INFLATION)
    b = float(inflated)
    f = Fraction(b)
    if f < inflated:
        b = math.nextafter(b, math.inf)
    return {'travel_bound': b, 'travel_bound_exact': str(inflated),
            'multiplier_ceiling_exact': str(comp['multiplier_ceiling_exact']),
            'multiplier_ceiling': comp['multiplier_ceiling'],
            'contributions': comp['contributions'], 'notes': comp['notes'],
            'box_diagonal': float(box_diagonal), 'tw_speed': float(tw_speed),
            'rel_inflation_ulps': int(REL_INFLATION_ULPS),
            'formula': 'sqrt2_up*box*prod(ceilings)/speed * (1 + %d*2^-52)' % int(REL_INFLATION_ULPS)}


def retraction_table():
    """The two retracted constants beside their corrections, exactly."""
    old_asym = ASYM_MAX_RETRACTED_EXACT
    new_asym = ASYM_MAX_EXACT
    out = {
        'asymmetric': {
            'retracted_expression': 'max_x (x + 1/x)/2 over x in [0.85, 1.35]',
            'retracted_value_exact': str(old_asym), 'retracted_value': float(old_asym),
            'corrected_expression': '(1.35 + 1/(0.85 + 1e-8))/2  -- transposed draws are independent',
            'corrected_value_exact': str(new_asym), 'corrected_value': float(new_asym),
            'understatement_factor': float(new_asym / old_asym),
            'why_retracted': ('the generator divides by the TRANSPOSED draw (:85), so the maximum '
                              'pairs a maximum with an independent minimum'),
            'with_rush_retracted': float(old_asym * RUSH_MAX),
            'with_rush_corrected': float(new_asym * RUSH_MAX),
        },
        'perturbation': {
            'retracted_expression': 'one hit of U(1.5, 5.0) => ceiling 5',
            'retracted_value': 5.0,
            'corrected_expression': '5^k with a DECLARED k (the same ordered pair can be drawn k '
                                    'times, sampling is with replacement)',
            'k_is_bounded_by': 'n_edge_perturb (the draw count), nothing tighter in the source',
            'example_draw_count_N51_p01': 78,
            'example_ceiling_k78_log10': 78 * math.log10(5),
        },
        'travel_bound_impact': {
            'old_asymmetric_bound': math.sqrt(2) * float(old_asym * RUSH_MAX),
            'new_asymmetric_bound': float(travel_bound_analytic(
                ('asymmetric_transit',), 1.0)['travel_bound']),
            'old_perturbation_bound': math.sqrt(2) * 5.0,
            'new_perturbation_bound_k1': float(travel_bound_analytic(
                ('long_tail_perturbation',), 1.0, perturb_multiplicity=1)['travel_bound']),
            'new_perturbation_bound_k2': float(travel_bound_analytic(
                ('long_tail_perturbation',), 1.0, perturb_multiplicity=2)['travel_bound']),
        },
    }
    out['travel_bound_impact']['asymmetric_bound_ratio'] = (
        out['travel_bound_impact']['new_asymmetric_bound']
        / out['travel_bound_impact']['old_asymmetric_bound'])
    return out


# =============================================================================================
# the per-instance witness (untouched by the retraction)
# =============================================================================================
def per_instance_witness(dist_mat, coords, *, tol=1e-6, stage='offline'):
    """Measure the multipliers THIS matrix actually uses, and its depot column.

    This is the finite-domain statement the retraction does NOT touch: it is a measurement of one
    instance, not a claim about the generator.
    """
    require_offline(stage)
    import numpy as np
    dm = np.asarray(dist_mat, dtype=np.float64)
    c = np.asarray(coords, dtype=np.float64)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise Refusal('DIST_MAT_SHAPE_INVALID', {'shape': list(dm.shape)})
    if not np.all(np.isfinite(dm)):
        raise Refusal('DIST_MAT_NON_FINITE', {'n': int((~np.isfinite(dm)).sum())})
    diff = c[:, None, :] - c[None, :, :]
    eu = np.sqrt((diff ** 2).sum(axis=-1))
    n = dm.shape[0]
    off = ~np.eye(n, dtype=bool)
    ratio = np.where(off & (eu > 0), dm / np.where(eu > 0, eu, 1.0), 1.0)
    depot_col = dm[:, 0]
    return {
        'max_edge': float(dm.max()),
        'max_multiplier_observed': float(ratio[off].max()) if off.any() else None,
        'max_multiplier_edge': (np.unravel_index(int(np.argmax(np.where(off, ratio, -1.0))),
                                                 ratio.shape) if off.any() else None),
        'max_abs_dev_from_euclidean': float(np.abs(dm - eu).max()),
        'symmetric': bool(np.allclose(dm, dm.transpose(), rtol=0.0, atol=tol)),
        'depot_column_max': float(depot_col[1:].max()) if n > 1 else None,
        'depot_column_argmax': int(np.argmax(depot_col[1:])) + 1 if n > 1 else None,
        'euclidean_consistent': bool(np.abs(dm - eu).max() <= tol),
        'reading': ('a per-instance measurement; it says what this matrix does, never what the '
                    'generator guarantees'),
    }


def travel_bound_per_instance(witness, tw_speed, *, rel_inflation_ulps=REL_INFLATION_ULPS):
    """The bound implied by the MEASURED multiplier of one instance (offline result)."""
    if not _finite(tw_speed) or float(tw_speed) <= 0:
        raise Refusal('TW_SPEED_INVALID', {'tw_speed': repr(tw_speed)})
    mult = witness.get('max_multiplier_observed')
    if mult is None or not _finite(mult):
        raise Refusal('PER_INSTANCE_MULTIPLIER_UNKNOWN', {'witness': 'no usable multiplier'})
    exact = _sqrt2_up() * Fraction(float(mult)) / Fraction(float(tw_speed))
    exact = exact * (1 + Fraction(int(rel_inflation_ulps), 2 ** 52))
    return {'travel_bound': float(exact), 'travel_bound_exact': str(exact),
            'measured_multiplier': float(mult), 'scope': 'THIS instance only',
            'depot_leg_bound': (witness['depot_column_max'] / float(tw_speed)
                                if witness.get('depot_column_max') is not None else None)}


# =============================================================================================
# the folded S-8 verdict
# =============================================================================================
def qualify_s8(*, models, tw_speed, perturb_multiplicity=None, per_instance=None,
               declared_horizon=None, observed_max_clock=None):
    """Five separate statements, never merged:
       1. what the retraction got wrong;
       2. what remains true for the measured instances;
       3. the NEW conditional analytic ceiling;
       4. external matrices that stay uncertifiable;
       5. the effect on T_clock_sup and on the time-error budget.
    """
    refusals, conditional, notes = [], [], []
    analytic, per_inst = None, None
    try:
        analytic = travel_bound_analytic(models, tw_speed,
                                         perturb_multiplicity=perturb_multiplicity)
    except Refusal as exc:
        if exc.code in ('PERTURB_MULTIPLICITY_UNDECLARED', 'DISTANCE_MODEL_UNBOUNDED'):
            conditional.append(exc.as_dict())        # a domain-permission question
        else:
            refusals.append(exc.as_dict())
    if per_instance is not None:
        try:
            per_inst = travel_bound_per_instance(per_instance, tw_speed)
        except Refusal as exc:
            refusals.append(exc.as_dict())

    budget = None
    if declared_horizon is not None and _finite(declared_horizon):
        if analytic is not None:
            budget = {'T_clock_sup_analytic': float(declared_horizon) + analytic['travel_bound'],
                      'T_clock_sup_per_instance': (
                          float(declared_horizon) + per_inst['travel_bound']
                          if per_inst is not None else None),
                      'basis': 'analytic ceiling vs measured multiplier -- two different scopes'}
            if observed_max_clock is not None and _finite(observed_max_clock):
                bound = (budget['T_clock_sup_analytic'] if per_inst is None
                         else budget['T_clock_sup_per_instance'])
                if float(observed_max_clock) > bound + 1e-12:
                    refusals.append({'code': 'CLOCK_EXCEEDS_UPPER_BOUND',
                                     'detail': {'observed': float(observed_max_clock),
                                                'bound_used': bound}})
        else:
            notes.append('no analytic ceiling (perturbation multiplicity undeclared or the distance '
                         'model is unbounded); only the per-instance measurement is available')

    if refusals:
        verdict = REFUSED
    elif conditional:
        verdict = NOT_CERTIFIABLE
    elif notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL
    return {'schema': SCHEMA, 'verdict': verdict,
            'EXPERIMENTAL_NOT_FOR_PRODUCTION': True,
            'production_consumable': False,
            'PRODUCTION_NOT_QUALIFIED': True,
            'implementation_version': IMPLEMENTATION_VERSION,
            'retraction': retraction_table(),
            'analytic_ceiling': analytic,
            'per_instance': ({'witness': per_instance, 'bound': per_inst}
                             if per_instance is not None else None),
            'budget_effect': budget,
            'refusals': refusals, 'conditional': conditional, 'notes': notes,
            'external_matrix': {'certifiable': False,
                                'why': 'an externally supplied matrix has no ceiling this module can '
                                       'derive; only a per-instance measurement is available'},
            'separations': [
                'the retracted ANALYTIC ceiling',
                'the measured PER-INSTANCE witness (unchanged by the retraction)',
                'the new CONDITIONAL analytic ceiling (requires a declared k)',
                'external distance matrices: still uncertifiable',
                'the effect on T_clock_sup and the time-error budget',
            ],
            'not_claimed': ['that the generator bounds k', 'that an empirical case can certify an '
                            'analytic ceiling', 'that the depot column is bounded by a '
                            'customer-customer edge bound']}
