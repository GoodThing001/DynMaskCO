#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 TIME BUDGET AUDIT for the inter-door cooling bound.

WHY THIS FILE EXISTS
====================
The previous round certified an inter-door condition with the slack `2*_EPS` and the sentence
"effective cooling >= wall clock - 2*_EPS".  That sentence is TRUE for the default contract but its
*reason* was incomplete, and the incompleteness is exactly what the reviewer flagged:

  (i)   the anchor skip (`_advance_coldchain_to` early return) contributes at most
        `_EPS + ulp/2`, not zero;
  (ii)  every real advance passes `fl(end - depart * hpu)` to the executor, so the SUM of the
        durations actually processed differs from the exact wall interval by up to
        `n_advances * ulp/2`;
  (iii) the time-unit conversion `float(duration) * hours_per_time_unit` adds another rounding when
        `hours_per_time_unit != 1.0`;
  (iv)  the anchor has SIX write sites, two of which are snapshot-RESTORE paths that can inject an
        anchor from outside the event loop.

So the honest statement is a BUDGET, not a single constant:

    Delta_eff  >=  (clock_n - clock_{n-1})  -  SLACK(T_max, n_max, hpu)

    SLACK = [ _EPS + ulp_c(T_max)/2 ]                 (anchor skip, one constant per interval)
          + [ n_max * ulp_c(T_max)/2 ]                (per-advance subtraction rounding)
          + [ 0 ]                                     (only if hpu == 1.0 exactly; else NOT_CERTIFIABLE)

with `ulp_c(T_max)` an outward-rounded ulp ceiling for every clock in `[0, T_max]`, `n_max` the
structural bound on the number of advances (`max_iter = 10000` in `_run_loop`,
`strict_online_env.py:481`), and `T_max` the episode horizon (`tw_end.max()`, `:454`).

For the default contract this evaluates to ~1.0e-9 h, comfortably below `Delta* ~ 0.04 h`.  For a
horizon of 1e6 h the same formula gives ~1.2e-6 h, which is NOT covered by `2*_EPS`: the previous
sentence was true by arithmetic coincidence in this configuration, not by construction.

THREE LAYERS THAT MUST NOT BE CONFUSED (the reviewer's boundary)
===============================================================
  L1  DURATION   this module: how much cooling time did the executor actually process?  Exact
                 rationals + an explicit float budget.  No `math.exp` involved.
  L2  REAL VALUE L-F/L-G in `q3_event_segmentation_experimental`: the real-valued temperature given
                 that duration.  Still no `math.exp` assumption (the flow is a closed form).
  L3  FLOATS     the double actually stored in `compartment_temperature_c`: needs `math.exp` error
                 (kappa) or the S3 bitwise audit.  **This module says NOTHING about L3.**

An exact `Fraction` comparison at L1 is therefore NOT a continuous-temperature certification; it is
the first of three layers, and the third one is still open.

WHAT THIS MODULE DOES NOT DO
============================
No production receipt (`production_consumable` False); no claim about stored floats; no claim about
the intra-segment profile; no claim that the restore paths are irrelevant when a resume happens
between two doors (they must be refused, see `ANCHOR_WRITE_SITES`).
"""
from __future__ import annotations

import math
from fractions import Fraction

import q3_strict_float_experimental as SF

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_TIME_BUDGET_AUDIT_V1'
SCHEMA = 'q3_time_budget_audit'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'

EPS_EXACT = Fraction(1e-9)

#: Every write to `coldchain_updated_time` in the tree, with the file and line as of this freeze.
#: Enumerated mechanically (`grep -n 'coldchain_updated_time\s*='`) and re-verified by TBA-01.
ANCHOR_WRITE_SITES = (
    ('strict_online_env.py:206', 'dispatch', 'sets the anchor to the dispatch time'),
    ('strict_online_env.py:230', 'real advance', 'sets the anchor to the advance end'),
    ('strict_online_env.py:332', 'door', 'sets the anchor to the door clock'),
    ('strict_online_env.py:403', 'return completion', 'sets the anchor to the return clock'),
    ('recourse_snapshot.py:305', 'SNAPSHOT RESTORE', 'injects the anchor from a snapshot'),
    ('dynmaskco_cc_context.py:72', 'SNAPSHOT RESTORE', 'injects the anchor from a snapshot'),
)
RESTORE_SITES = ('recourse_snapshot.py:305', 'dynmaskco_cc_context.py:72')

#: `_run_loop` bounds the number of iterations, and every iteration performs at most one advance per
#: vehicle, so this is a structural bound on the number of advances in one episode.
MAX_ITER = 10000
MAX_ITER_SOURCE = 'strict_online_env.py:481 (`max_iter = 10000`)'

#: The horizon actually used by the loop; every clock in an episode is <= this.
HORIZON_SOURCE = 'strict_online_env.py:454 (`horizon = float(self.tw_end[inst_idx].max())`)'

#: Preconditions for L1.  Any of them failing means NOT_CERTIFIABLE, never "use the wall clock".
L1_PRECONDITIONS = (
    ('L1-1', 'the only advance path is _advance_coldchain_to with served_customer=None',
     'strict_online_env.py:209-230'),
    ('L1-2', 'the early return does not move the anchor, so a skipped span is covered later',
     'strict_online_env.py:215-220'),
    ('L1-3', 'the anchor write sites are exactly the six listed in ANCHOR_WRITE_SITES',
     'TBA-01 re-enumerates them from the source'),
    ('L1-4', 'NO snapshot restore happens between the two doors being checked',
     'recourse_snapshot.py:305 / dynmaskco_cc_context.py:72 can inject an anchor'),
    ('L1-5', 'hours_per_time_unit == 1.0 exactly, so no unit-conversion rounding occurs',
     'coldchain_contract.py:36, coldchain_state.py:521'),
    ('L1-6', 'every clock in the episode is <= T_max, which bounds the subtraction ulp',
     HORIZON_SOURCE),
    ('L1-7', 'the number of advances in one episode is <= MAX_ITER', MAX_ITER_SOURCE),
    ('L1-8', 'clocks are non-decreasing inside the interval (the checker refuses otherwise)',
     'checked in effective_cooling_budget'),
)


# ---------------------------------------------------------------------------------------------
# ulp ceiling, exactly
# ---------------------------------------------------------------------------------------------
def ulp_ceiling(x):
    """Outward-rounded ulp bound: `ulp(y) <= result` for every 0 <= y <= x (exact Fraction)."""
    v = Fraction(float(x))
    if v <= 0:
        return Fraction(5e-324)
    k = 0
    p = Fraction(1)
    while p < v:
        p *= 2
        k += 1
        if k > 2000:                                                      # pragma: no cover
            raise SF.Refusal('ULP_CEILING_OVERFLOW', {'x': float(x)})
    return p / Fraction(2 ** 52)


def half_ulp_ceiling(x):
    return ulp_ceiling(x) / 2


# ---------------------------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------------------------
def time_budget(horizon, max_iter=MAX_ITER, hours_per_time_unit=1.0):
    """All three slack terms, exactly, with the reason for each."""
    hpu = float(hours_per_time_unit)
    hu = half_ulp_ceiling(horizon)
    skip = EPS_EXACT + hu
    rounding = Fraction(int(max_iter)) * hu
    unit = Fraction(0) if hpu == 1.0 else None
    total = None if unit is None else skip + rounding + unit
    return {
        'horizon_h': float(horizon), 'max_iter': int(max_iter),
        'hours_per_time_unit': hpu,
        'ulp_ceiling_h': str(ulp_ceiling(horizon)),
        'half_ulp_ceiling_h': float(hu),
        'skip_term_h': float(skip), 'skip_term_exact': str(skip),
        'rounding_term_h': float(rounding), 'rounding_term_exact': str(rounding),
        'unit_term_h': (None if unit is None else float(unit)),
        'total_slack_h': (None if total is None else float(total)),
        'total_slack_exact': (None if total is None else str(total)),
        'unit_conversion_exact': (hpu == 1.0),
        'old_constant_2eps_h': float(2 * EPS_EXACT),
        'old_constant_covers_budget': (None if total is None
                                       else Fraction(2) * EPS_EXACT >= total),
        'reasons': {
            'skip': '_advance_coldchain_to early return: the skipped span is <= _EPS + ulp/2 and is '
                    'covered later unless it is the LAST sample of the interval (one constant per '
                    'interval)',
            'rounding': 'each real advance passes fl(end - depart); the n_x sum of those roundings is '
                        'bounded by n * ulp/2 and does NOT cancel',
            'unit': 'duration = fl(end - depart) * hours_per_time_unit adds one more rounding unless '
                    'the unit is exactly 1.0',
        },
    }


def effective_cooling_budget(clock_prev, clock_next, horizon, max_iter=MAX_ITER,
                            hours_per_time_unit=1.0):
    """Exact wall interval minus the full budget: the rigorous lower bound on processed cooling."""
    a = Fraction(float(clock_prev))
    b = Fraction(float(clock_next))
    if b < a:
        raise SF.Refusal('CLOCK_MOVED_BACKWARDS', {'a': float(a), 'b': float(b)})
    bud = time_budget(horizon, max_iter, hours_per_time_unit)
    wall = b - a
    if bud['total_slack_exact'] is None:
        return {'wall_h': float(wall), 'lower_bound_exact': None, 'budget': bud,
                'certifiable': False,
                'why': 'hours_per_time_unit != 1.0, so the unit-conversion rounding is not budgeted'}
    slack = Fraction(bud['total_slack_exact'])
    lb = wall - slack
    return {'wall_h': float(wall), 'wall_exact': str(wall),
            'lower_bound_h': float(lb), 'lower_bound_exact': str(lb),
            'budget': bud, 'certifiable': True,
            'relative_cost': (float(slack / wall) if wall > 0 else None)}


# ---------------------------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------------------------
def audit(*, contract, contract_identity, event_stream, horizon, max_iter=MAX_ITER,
          resumed_between_doors=False, state_identity=None, expected_state_identity=None,
          anchor_write_sites=None):
    """Full L1 audit: precondition ledger + per-interval budget + exact Delta* comparison.

    Returns a receipt whose verdict can be compared against the Delta* certificate.
    """
    import q3_door_interval_thresholds_experimental as TH
    refusals, notes, per_zone = [], [], {}
    try:
        cid = SF.require_contract_identity(contract, contract_identity)
    except SF.Refusal as exc:
        cid = None
        refusals.append({'code': exc.code, 'detail': exc.detail})
    if state_identity is None or expected_state_identity is None:
        notes.append('STATE_IDENTITY_UNBOUND')
    elif str(state_identity) != str(expected_state_identity):
        refusals.append({'code': 'STATE_IDENTITY_MISMATCH'})
    hpu = float(getattr(getattr(contract, 'units', None), 'hours_per_time_unit', 1.0))
    if hpu != 1.0:
        refusals.append({'code': 'TIME_UNIT_CONVERSION_NOT_BUDGETED',
                         'detail': {'hours_per_time_unit': hpu,
                                    'why': 'L1-5: a non-unit scale adds a rounding per advance that '
                                           'this budget does not cover'}})
    if resumed_between_doors:
        refusals.append({'code': 'SNAPSHOT_RESTORE_INSIDE_INTERVAL',
                         'detail': {'why': 'L1-4: a restore path injects the anchor '
                                           '(recourse_snapshot.py:305 / dynmaskco_cc_context.py:72), '
                                           'so the wall clock no longer bounds the processed cooling'}})
    if horizon is None or not (float(horizon) > 0):
        refusals.append({'code': 'MISSING_HORIZON'})
        horizon = 1.0
    sites = tuple(anchor_write_sites) if anchor_write_sites is not None else ANCHOR_WRITE_SITES
    if resumed_between_doors is False and len(sites) != len(ANCHOR_WRITE_SITES):
        refusals.append({'code': 'ANCHOR_WRITE_SITES_UNEXPECTED',
                         'detail': {'found': [s[0] for s in sites],
                                    'expected': [s[0] for s in ANCHOR_WRITE_SITES]}})
    bud = time_budget(horizon, max_iter, hpu)
    for z in SF.zone_thermals_exact(contract):
        th = TH.zone_thresholds_exact(contract, z.zone)
        need = th['delta_star_exact']
        per_zone[z.zone] = {'threshold_h': float(need), 'threshold_min': float(need) * 60.0,
                            'intervals': [], 'violations': []}
        zz = z
        for vid in sorted(event_stream):
            prev = None
            for r in event_stream[vid]:
                zc = r.get('zone')
                clk = r.get('door_clock')
                if zc is None or not _finite(clk) or int(zc) != zz.zone:
                    continue
                if prev is not None:
                    b = effective_cooling_budget(prev, clk, horizon, max_iter, hpu)
                    if not b['certifiable']:
                        per_zone[z.zone]['violations'].append({'why': b['why']})
                    else:
                        lb = Fraction(b['lower_bound_exact'])
                        slack = lb - need
                        row = {'wall_h': b['wall_h'], 'lower_bound_h': b['lower_bound_h'],
                               'threshold_h': float(need),
                               'slack_exact': str(slack), 'slack_h': float(slack),
                               'ok': slack >= 0}
                        per_zone[z.zone]['intervals'].append(row)
                        if not row['ok']:
                            per_zone[z.zone]['violations'].append(row)
                prev = clk
    bad = [k for k, v in per_zone.items() if v['violations']]
    if refusals:
        verdict = REFUSED
    elif bad:
        verdict = NOT_CERTIFIABLE
    elif notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL
    return {'schema': SCHEMA, 'verdict': verdict,
            'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
            'production_consumable': PRODUCTION_CONSUMABLE,
            'implementation_version': IMPLEMENTATION_VERSION,
            'contract_hash': cid, 'refusals': refusals, 'binding_notes': notes,
            'budget': bud,
            'per_zone': {str(k): v for k, v in sorted(per_zone.items())},
            'n_preconditions': len(L1_PRECONDITIONS),
            'preconditions': [{'id': a, 'condition': b, 'where': c} for a, b, c in L1_PRECONDITIONS],
            'anchor_write_sites': [{'site': a, 'role': b, 'effect': c}
                                   for a, b, c in ANCHOR_WRITE_SITES],
            'layers': {
                'L1_duration': 'covered by this audit (exact rationals + the budget above)',
                'L2_real_value': 'covered by q3_event_segmentation_experimental (L-F/L-G); no libm '
                                 'assumption',
                'L3_stored_floats': 'NOT COVERED: the executor updates through math.exp and the door '
                                    'adds delta in floating point; that needs kappa or the S3 audit',
            },
            'claim': ('every inter-door interval processed at least Delta*_z of effective cooling, so '
                      'the real-valued band holds at every door event'),
            'not_claimed': [
                'that the STORED FLOATS are within the band (layer L3)',
                'that the intra-segment continuous profile is safe',
                'anything about an interval that contains a snapshot restore',
                'anything when hours_per_time_unit != 1.0'],
            }


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                     # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


def source_anchor_sites(path, pattern='coldchain_updated_time'):
    """Re-enumerate the anchor write sites from a source file (mechanical, no hard-coded list)."""
    import re
    out = []
    text = path.read_text(encoding='utf-8').splitlines()
    rx = re.compile(r'%s\s*=' % pattern)
    for i, line in enumerate(text, start=1):
        if rx.search(line):
            out.append((path.name, i, line.strip()))
    return out
