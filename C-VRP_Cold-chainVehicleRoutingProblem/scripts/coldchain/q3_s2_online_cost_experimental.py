#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 S2: ONLINE COST of the S1 route bracket.

WHAT THIS ANSWERS (Q3-3 of `M1_L2最小可行性与成本裁决建议_2026-10-17.md` §4.2)
=============================================================================
S1 makes the continuous thermal band decidable at decision time.  It does not say whether the
decision FITS IN AN ONLINE BUDGET.  This module supplies the three pieces that question needs:

  1. a CLOSED FORM for the work the certificate performs, derived from the shape of the plan
     (number of doors, number of short gaps, number of vehicles, number of zones) -- not from a
     timing run, so a campaign can be sized before anything is executed;
  2. an EXTERNAL count of the expensive evaluations (the rigorous `exp` / `ln` enclosures), taken by
     patching `q3_strict_float_experimental.exp_neg_enclosure` and
     `q3_s1_route_bracket_experimental.rational_ln_enclosure`, so the count is measured on the
     running code rather than self-reported by it;
  3. an ISOLATED wall-clock measurement with a pre-declared ceiling.

THE COST IS NOT WHAT ONE WOULD GUESS
====================================
The bracket needs `exp` only for gaps that are too SHORT to bring the zone back to the floor.  A gap
that is at least `Theta_z` collapses the bracket to `tau_z` with zero transcendental evaluations, and
for the default contract every `Theta_z` is about 3.2-3.3 minutes while every gap is at least the
later stop's service time (9.0 minutes on the frozen R1 data).  So a normal plan costs ZERO, and the
closed form says so exactly: `n_exp == n_short_gaps`, and `n_short_gaps == 0` unless the plan really
does contain a short gap.

WHAT THIS MODULE DOES NOT DO
============================
  * it does not decide thermal safety (that is S1) and it does not touch the F-level question (S3);
  * the wall clock is a LOCAL diagnostic on one machine and is never a cross-platform guarantee;
  * no GPU, no training, no rollout; `production_consumable` is hard-coded False.
"""
from __future__ import annotations

import ast
import contextlib
import statistics
import time
from pathlib import Path

import q3_s1_route_bracket_experimental as S1
import q3_strict_float_experimental as SF

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_S2_ONLINE_COST_EXPERIMENTAL_V1'
SCHEMA = 'q3_s2_online_cost'

#: pre-registered BEFORE any measurement (see the round's E5.3).  Recorded here so the numbers the
#: tests compare against cannot drift with the measurement.
B_DEC_SECONDS = 10.0
C_ADMISSIBLE = 250
T_MAX_SECONDS = B_DEC_SECONDS / C_ADMISSIBLE

#: functions allowed to touch `math` at module-import scope.  `_theta_via_math_log` is a REPORT-ONLY
#: probe that no verdict depends on; `InstanceArrays.travel_fn` is the travel-model input, whose
#: Euclidean fallback uses `math.hypot` (not a bound-relevant transcendental) and whose default path
#: (`dist_mat / tw_speed`) uses no transcendental at all.
MATH_ALLOWLIST = frozenset({'_theta_via_math_log', 'travel_fn'})
FORBIDDEN_MATH = frozenset({'exp', 'expm1', 'log', 'log1p', 'log2', 'log10', 'pow'})
#: dependency functions this module also audits, so the guarantee is not defeated by a helper
AUDITED_DEPENDENCIES = ('exp_neg_enclosure', 'zone_thermals_exact', 'require_contract_identity',
                        'float_ceil', 'float_floor', 'segment_pair')


# ---------------------------------------------------------------------------------------------
# 1. the closed form
# ---------------------------------------------------------------------------------------------
def closed_form(*, n_doors, n_short_gaps, n_vehicles, n_zones, cached_thresholds=False):
    """Exact work model of the S1 certificate for one candidate plan.

    `n_gap_rows  = n_doors`            one bracket row per door (the entry interval is merged into
                                       the first door's gap: `prev` starts at the decision clock)
    `n_exp       = n_short_gaps`       one rigorous exp enclosure per gap that is NOT provably long
                                       enough to return the zone to the floor
    `n_ln        = 0 if cached_thresholds else n_zones`
                                       one rigorous ln enclosure per zone with a non-zero door
                                       increment, a ONE-OFF per contract -- `n_ln_required_by_contract`
                                       always reports the contract-level requirement so the two
                                       numbers cannot be confused
    The elementary `+ - * / max` work is linear in the same counts; `elementary_ops_bound` records the
    constant used by the report (16 exact rational operations per gap row, 4 per vehicle, 8 fixed).
    """
    for name, v in (('n_doors', n_doors), ('n_short_gaps', n_short_gaps),
                    ('n_vehicles', n_vehicles), ('n_zones', n_zones)):
        if isinstance(v, bool) or int(v) != v or v < 0:
            raise SF.Refusal('BAD_SHAPE_COUNT', {'field': name, 'value': repr(v)})
    n_doors, n_short = int(n_doors), int(n_short_gaps)
    if n_short > n_doors:
        raise SF.Refusal('SHORT_GAPS_EXCEED_DOORS',
                         {'n_short_gaps': n_short, 'n_doors': n_doors,
                          'why': 'there is one gap row per door, so this shape is impossible'})
    return {
        'n_gap_rows': n_doors,
        'n_exp_exact': n_short,
        'n_exp_upper_bound': n_doors,
        'n_exp_upper_bound_tighter': max(0, n_doors - int(n_vehicles)),
        # the pre-registration's E4.2 S2-7 formula, kept verbatim so its literal status is visible
        'n_exp_pre_registered_formula': n_doors + int(n_zones) * int(n_vehicles),
        'n_ln_one_off_per_contract': 0 if cached_thresholds else int(n_zones),
        'n_ln_required_by_contract': int(n_zones),
        'cached_thresholds': bool(cached_thresholds),
        'elementary_ops_bound': 16 * n_doors + 4 * int(n_vehicles) + 8,
        'shape': {'n_doors': n_doors, 'n_short_gaps': n_short,
                  'n_vehicles': int(n_vehicles), 'n_zones': int(n_zones)},
    }


def plan_shape(receipt):
    """Extract the closed-form inputs from an S1 receipt."""
    s = receipt.get('summary') or {}
    th = receipt.get('thresholds') or {}
    return {'n_doors': int(s.get('n_doors') or 0),
            'n_short_gaps': int(s.get('n_short_gaps') or 0),
            'n_vehicles': int(s.get('n_vehicles') or 0),
            'n_zones': len(th),
            'n_stops': int(s.get('n_stops') or 0),
            'n_exp_reported': int(s.get('n_exp') or 0),
            'n_gaps_reported': int(s.get('n_gaps') or 0),
            'n_ln_reported': int(s.get('n_ln_enclosures') or 0)}


# ---------------------------------------------------------------------------------------------
# 2. external counting
# ---------------------------------------------------------------------------------------------
@contextlib.contextmanager
def count_transcendentals():
    """Count the rigorous `exp` / `ln` enclosure calls made while the block runs.

    The patch is applied to the MODULES the S1 code looks the names up in, so it counts calls made
    by the running certificate, not a number the certificate reports about itself.
    """
    counts = {'exp': 0, 'ln': 0}
    orig_exp = SF.exp_neg_enclosure
    orig_ln = S1.rational_ln_enclosure

    def _exp(*args, **kwargs):
        counts['exp'] += 1
        return orig_exp(*args, **kwargs)

    def _ln(*args, **kwargs):
        counts['ln'] += 1
        return orig_ln(*args, **kwargs)

    SF.exp_neg_enclosure = _exp
    S1.rational_ln_enclosure = _ln
    try:
        yield counts
    finally:
        SF.exp_neg_enclosure = orig_exp
        S1.rational_ln_enclosure = orig_ln


# ---------------------------------------------------------------------------------------------
# 3. isolated timing
# ---------------------------------------------------------------------------------------------
def isolated_timing(fn, *, warmup=20, repeat=200):
    """Median wall clock of `fn()` in seconds.  No I/O and no numpy inside the timed region."""
    for _ in range(int(warmup)):
        fn()
    samples = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples)
    return {'median_s': med, 'min_s': min(samples), 'max_s': max(samples),
            'n_samples': len(samples), 'warmup': int(warmup),
            't_max_s': T_MAX_SECONDS, 'within_budget': med <= T_MAX_SECONDS,
            'max_candidates_within_budget': (B_DEC_SECONDS / med) if med > 0 else float('inf')}


# ---------------------------------------------------------------------------------------------
# 4. mechanical audit: the decision path must not call a libm transcendental
# ---------------------------------------------------------------------------------------------
def _math_calls(path, allowlist):
    """[(function_name, attribute)] for every `math.<attr>` CALL found outside `allowlist`."""
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


def audit_decision_path(module_paths, *, allowlist=MATH_ALLOWLIST,
                        dependencies=AUDITED_DEPENDENCIES, sf_path=None):
    """Refuse to claim a libm-free decision if any audited function calls math.exp/log/pow."""
    findings, scanned = [], []
    for p in module_paths:
        for fn_name, attr in _math_calls(p, allowlist):
            if attr in FORBIDDEN_MATH:
                findings.append({'module': str(p), 'function': fn_name, 'call': 'math.' + attr})
        scanned.append(str(p))
    if sf_path is not None:
        tree = ast.parse(Path(sf_path).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in dependencies:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    base = sub.func.value
                    if isinstance(base, ast.Name) and base.id == 'math':
                        if sub.func.attr in FORBIDDEN_MATH:
                            findings.append({'module': str(sf_path), 'function': node.name,
                                             'call': 'math.' + sub.func.attr})
            scanned.append('%s:%s' % (sf_path, node.name))
    return {'ok': not findings, 'findings': findings, 'scanned': scanned,
            'allowlist': sorted(allowlist),
            'forbidden': sorted(FORBIDDEN_MATH),
            'reading': ('the DECISION path calls no libm transcendental at all; the only two math '
                        'uses in the audited set are the report-only probe '
                        '`_theta_via_math_log` and the travel-model input `travel_fn` (math.hypot), '
                        'neither of which can move a bound')}


# ---------------------------------------------------------------------------------------------
# 5. the assembled cost report
# ---------------------------------------------------------------------------------------------
def cost_report(receipt, *, closed=None, measured_exp=None, measured_ln=None, timing=None):
    """Assemble one candidate plan's cost record and check it against the closed form."""
    shape = plan_shape(receipt)
    cached = bool((receipt.get('summary') or {}).get('thresholds_supplied'))
    cf = closed or closed_form(n_doors=shape['n_doors'], n_short_gaps=shape['n_short_gaps'],
                               n_vehicles=shape['n_vehicles'], n_zones=shape['n_zones'],
                               cached_thresholds=cached)
    out = {'schema': SCHEMA,
           'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
           'production_consumable': PRODUCTION_CONSUMABLE,
           'implementation_version': IMPLEMENTATION_VERSION,
           'shape': shape, 'closed_form': cf,
           'budget': {'B_dec_seconds': B_DEC_SECONDS, 'C_admissible': C_ADMISSIBLE,
                      't_max_seconds': T_MAX_SECONDS},
           'measured_exp': measured_exp, 'measured_ln': measured_ln,
           'timing': timing}
    checks = []
    if measured_exp is not None:
        checks.append({'id': 'exp_equals_closed_form',
                       'ok': measured_exp == cf['n_exp_exact'],
                       'measured': measured_exp, 'expected': cf['n_exp_exact']})
        checks.append({'id': 'exp_equals_reported',
                       'ok': measured_exp == shape['n_exp_reported'],
                       'measured': measured_exp, 'reported': shape['n_exp_reported']})
        checks.append({'id': 'exp_within_door_upper_bound',
                       'ok': measured_exp <= cf['n_exp_upper_bound'],
                       'measured': measured_exp, 'upper': cf['n_exp_upper_bound']})
    if measured_ln is not None:
        checks.append({'id': 'ln_equals_closed_form',
                       'ok': measured_ln == cf['n_ln_one_off_per_contract'],
                       'measured': measured_ln, 'expected': cf['n_ln_one_off_per_contract']})
    if timing is not None:
        checks.append({'id': 'within_budget', 'ok': bool(timing['within_budget']),
                       'median_s': timing['median_s'], 't_max_s': T_MAX_SECONDS})
    out['checks'] = checks
    out['ok'] = all(c['ok'] for c in checks)
    return out
