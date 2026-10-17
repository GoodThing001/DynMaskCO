#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION  --  G-2 ERRATA: the clock upper bound, the callback scope, and
the install gate.

WHY THIS FILE EXISTS
====================
The first G-2 report (`G2_时间参数身份与strict_online来源资格审查_2026-10-17.md`) treated the episode
horizon as if it bounded every timestamp handed to the temperature-advance function:

    L1-6:  "every clock in the episode is <= T_max"

That is WRONG, and an independent review found the counterexample in the source itself.  When the
event loop runs out of events it calls `_force_return_all`, which (a) commits a return leg and
(b) calls `_advance_fleet(..., finish)` where

    finish = depart + travel(current_node, depot)

so the LAST timestamp that reaches `_advance_coldchain_to` -- and therefore the last one in the ulp
budget -- can EXCEED the horizon by one travel leg.  Three consequences, all handled here:

    (i)   the ulp ceiling must be taken over a CLOCK bound, not over the horizon;
    (ii)  the difference is invisible in the registered domain (24 and 24+1.42 share a binade) but
          it is not invisible in general (a horizon just below a power of two doubles the ulp), so
          the derivation must be fixed even though the numbers do not move;
    (iii) the tighter iteration bound `1 + floor(T_max/eps_event)` was an UNPROVEN tightening and is
          now NOT ADOPTED; the adopted advance bound uses `max_iter` alone.

The same review also corrected the framing of the install gap: `run_repair_transaction` is
FAIL-CLOSED when no certifier is supplied (it refuses before sampling, zero installs).  "There is no
certifier" is not the same statement as "an uncertified plan was installed", and this module verifies
the former as source fact.

WHAT THIS MODULE IS NOT
=======================
Not a production change, not a new certificate, not a claim that the clock bound is tight.  It does
not touch the certifier, the env, or the install entry.  It does not decide that the horizon is
public; it re-derives what the horizon is being asked to bound.
"""
from __future__ import annotations

import ast
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'G2B_CLOCK_SCOPE_V1'
SCHEMA = 'q3_g2b_clock_scope'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'

#: the advance path's own tolerances (unchanged; the errata is about the CLOCK, not about them)
EPS_SKIP = Fraction(1, 10 ** 9)
EPS_BACKWARD = 1e-6
EPS_EVENT = 1e-6


class Refusal(Exception):
    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail or {}

    def as_dict(self):
        return {'code': self.code, 'detail': self.detail}


# =============================================================================================
# exact ulp ceiling (kept local so this module never depends on a moved constant)
# =============================================================================================
def ulp_ceiling(x):
    """Outward-rounded ulp bound: ulp(y) <= result for every 0 <= y <= x, exact Fraction."""
    v = Fraction(float(x))
    if v <= 0:
        return Fraction(5e-324)
    k, p = 0, Fraction(1)
    while p < v:
        p *= 2
        k += 1
        if k > 4000:                                                       # pragma: no cover
            raise Refusal('ULP_CEILING_OVERFLOW', {'x': float(x)})
    return p / Fraction(2 ** 52)


def half_ulp_ceiling(x):
    return ulp_ceiling(x) / 2


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                      # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


# =============================================================================================
# (i) the clock upper bound
# =============================================================================================
#: every way this module is willing to obtain `max(clock) - horizon`.
TRAVEL_BOUND_SOURCES = {
    'coordinate_box_diagonal': {
        'visibility': 'ONLINE-DECISION',
        'formula': 'sqrt(2) / tw_speed',
        'why': ('the coordinate box is a DOMAIN invariant, not instance data: every coordinate in '
                'the registered splits lies inside the unit square, so any depot leg is at most the '
                'diagonal.  Measured range in data/m0_scale: [0.0201, 0.9798], observed max '
                'pairwise distance 1.276295.'),
    },
    'served_leg_max': {
        'visibility': 'ONLINE-DECISION',
        'formula': 'max over already-served legs of dist/speed',
        'why': ('a return leg starts at a node the vehicle physically reached, so the leg length is '
                'disclosed by the time it is used; this source is a per-episode runtime witness, '
                'not a static bound'),
    },
    'caller_declared_with_witness': {
        'visibility': 'CERTIFIER-PROOF',
        'formula': 'declared value, witnessed against observed clocks',
        'why': 'a declared bound is accepted only while the runtime monitor stays under it',
    },
}


def clock_upper_bound(horizon, travel_bound, source):
    """T_clock_sup = horizon + travel_bound, with the source of the travel bound named.

    The horizon alone is NOT the clock bound: `_force_return_all` advances to
    `depart + travel(current, 0)` after the last in-loop event, so the final timestamp is
    `horizon + at most one leg`.  """
    if source not in TRAVEL_BOUND_SOURCES:
        raise Refusal('TRAVEL_BOUND_SOURCE_UNKNOWN', {'source': source})
    if not _finite(horizon) or float(horizon) <= 0:
        raise Refusal('HORIZON_INVALID', {'value': repr(horizon)})
    if not _finite(travel_bound) or float(travel_bound) < 0:
        raise Refusal('TRAVEL_BOUND_INVALID', {'value': repr(travel_bound)})
    sup = float(horizon) + float(travel_bound)
    return {
        'T_horizon': float(horizon),
        'travel_bound': float(travel_bound),
        'T_clock_sup': sup,
        'source': source,
        'source_visibility': TRAVEL_BOUND_SOURCES[source]['visibility'],
        'derivation': ('last in-loop clock <= horizon; _force_return_all then advances to '
                       'depart + travel(current, 0) with depart <= horizon, so '
                       'max clock <= horizon + max one-leg travel'),
        'ulp_ceiling_horizon': str(ulp_ceiling(horizon)),
        'ulp_ceiling_clock': str(ulp_ceiling(sup)),
        'same_binade': ulp_ceiling(horizon) == ulp_ceiling(sup),
        'why_the_bound_is_needed': (
            'the ulp budget must dominate every timestamp that reaches the executor, including the '
            'return-completion advance, which happens AFTER the loop has run out of events'),
    }


def slack_budget(clock_sup, n_max, hours_per_time_unit=1.0):
    """The three slack terms, with the ulp taken over the CLOCK bound (not over the horizon)."""
    hpu = float(hours_per_time_unit)
    hu = half_ulp_ceiling(clock_sup)
    skip = EPS_SKIP + hu
    rounding = Fraction(int(n_max)) * hu
    unit = None if hpu != 1.0 else Fraction(0)
    total = None if unit is None else skip + rounding + unit
    return {'clock_sup': float(clock_sup), 'n_max_per_vehicle': int(n_max),
            'hours_per_time_unit': hpu, 'half_ulp_ceiling': float(hu),
            'skip_term': float(skip), 'rounding_term': float(rounding),
            'total_slack': (None if total is None else float(total)),
            'total_slack_exact': (None if total is None else str(total)),
            'certifiable': total is not None,
            'basis': 'ulp taken over the maximum CLOCK, not over the event-loop horizon'}


def basis_comparison(horizon, clock_sup, n_max, hours_per_time_unit=1.0):
    """Does using the horizon instead of the clock bound change the number?  Report, do not assume."""
    a = slack_budget(horizon, n_max, hours_per_time_unit)
    b = slack_budget(clock_sup, n_max, hours_per_time_unit)
    ratio = (None if (a['total_slack'] is None or b['total_slack'] is None or a['total_slack'] == 0)
             else b['total_slack'] / a['total_slack'])
    return {'slack_with_horizon': a['total_slack'], 'slack_with_clock_sup': b['total_slack'],
            'ratio_clock_over_horizon': ratio,
            'numbers_differ': (a['total_slack'] != b['total_slack']),
            'ulp_horizon': str(ulp_ceiling(horizon)), 'ulp_clock': str(ulp_ceiling(clock_sup)),
            'reading': ('equal here because both values fall in the same binade; the ratio becomes 2 '
                        'as soon as the clock bound crosses a power of two, so the horizon-only '
                        'basis is not a valid derivation even where it agrees numerically')}


# =============================================================================================
# (ii) the advance bound, without the unproven time-side tightening
# =============================================================================================
def advance_budget_max_iter(max_iter, pre_loop, per_iteration, termination, n_iter_current=None):
    """The ADOPTED bound: the iteration count is bounded by `max_iter` alone.

    The earlier report also offered `n_iter <= 1 + floor(T_max/eps_event)`, which is a real statement
    only if every iteration is proven to move the clock by more than `eps_event`.  The review is
    right that the proof is not in the corpus, so this tightening is recorded as NOT ADOPTED and the
    number is unchanged either way (max_iter binds in the registered domain)."""
    n = int(pre_loop) + int(per_iteration) * int(max_iter) + int(termination)
    vari = None
    if n_iter_current is not None:
        vari = {'time_side_tightening': 'NOT_ADOPTED',
                'why': ('it requires "every iteration moves the clock by more than eps_event", which '
                        'is asserted nowhere in the source and only observed in one synthetic '
                        'episode; using it would make the budget depend on an unproven premise'),
                'n_iter_would_be': int(n_iter_current),
                'n_would_be': int(pre_loop) + int(per_iteration) * int(n_iter_current)
                               + int(termination)}
    return {'n_max_per_vehicle': n,
            'n_iter_bound': int(max_iter),
            'n_iter_basis': 'max_iter (the explicit loop cap in `_run_loop`)',
            'formula': '%d + %d*max_iter + %d' % (int(pre_loop), int(per_iteration),
                                                  int(termination)),
            'not_adopted': vari}


# =============================================================================================
# (iii) the callback scope
# =============================================================================================
#: Every extension point that runs caller-supplied code INSIDE the event loop.  A callback that
#: calls `_advance_coldchain_to` (or `transition_segment`) makes the static advance bound false, and
#: nothing in the source prevents it, so each sink is classified rather than assumed harmless.
CALLBACK_SINKS = (
    ('snapshot_hook', 'strict_online_env.py:552', '_maybe_snapshot -> self.snapshot_hook(...)',
     'OPEN', 'receives `env`; may call any env method'),
    ('oracle_hook', 'strict_online_env.py:564', '_run_oracle_hook -> self.oracle_hook(...)',
     'OPEN', 'receives `env`; may call any env method'),
    ('replanner.plan', 'strict_online_env.py:634', '_plan_and_commit -> self.replanner.plan(...)',
     'OPEN', 'receives `env`; may call any env method'),
    ('replanner.on_reveal', 'strict_online_env.py:539', 'reveal event -> self.replanner.on_reveal(...)',
     'OPEN', 'receives `env`; may call any env method'),
    ('replanner.restore_state', 'strict_online_env.py:584', 'run_resumed -> restore_state(...)',
     'OPEN', 'receives the snapshot; may touch the env through the closure'),
    ('replanner.sync_deferred_from_vehicles', 'strict_online_env.py:645',
     '_plan_and_commit -> sync_deferred_from_vehicles(vehicles)', 'MUTATES_VEHICLES',
     'mutates `mutable_suffix` only; may not advance the cold chain'),
    ('force_suffix', 'strict_online_env.py:639', 'branch intervention', 'DATA_ONLY',
     'a dict of (event_id, vehicle) -> suffix; no code runs'),
)

#: functions that may read the FULL instance array (hidden orders) and are therefore OFFLINE ONLY
OFFLINE_ONLY = ('classify_horizon', 'horizon_witness', 'tw_end_max_scan')


def require_offline_stage(stage):
    """Reading hidden orders to decide whether a certificate passes is a strict-online violation.

    The equality witness `max(tw_end) == tw_end[0]` needs the full array, so it may only be computed
    in the qualification/identity stage.  The online path consumes the already-qualified scalars."""
    if stage != 'offline':
        raise Refusal('ONLINE_MUST_NOT_READ_HIDDEN_DATA', {
            'stage': stage,
            'why': ('the horizon witness scans tw_end over orders that have not been revealed; '
                    'used online it would make the certificate depend on future information')})


def offline_horizon_witness(tw_end_row, reveal_time_row, stage='offline'):
    """The witness, with the stage guard attached (this is the ONLY place the full row is read)."""
    require_offline_stage(stage)
    import numpy as np
    row = np.asarray(tw_end_row, dtype=np.float64)
    if row.ndim != 1 or row.size < 2:
        raise Refusal('HORIZON_ROW_INVALID', {'shape': list(row.shape)})
    rt = None if reveal_time_row is None else np.asarray(reveal_time_row, dtype=np.float64)
    hidden = (rt > 0) if rt is not None else np.zeros(row.shape, dtype=bool)
    return {
        'tw_end_max': float(row.max()),
        'depot_tw_end': float(row[0]),
        'hidden_customer_max': (float(row[hidden].max()) if bool(hidden.any()) else None),
        'witness_equal': bool(float(row.max()) == float(row[0])),
        'stage': stage,
        'reads_hidden_data': True,
        'handoff_rule': ('only {depot_tw_end, witness_equal, T_clock_sup, n_max, contract_hash} '
                         'may cross into the online path'),
    }


def online_consumable(identity):
    """The projection an online consumer is allowed to see: scalars that are already qualified."""
    keep = ('contract_hash', 'hours_per_time_unit', 'T_clock_sup', 'n_max_per_vehicle',
            'witness_equal', 'eps_skip', 'eps_backward', 'eps_event')
    out, missing = {}, []
    flat = {}
    for section in identity.values():
        if isinstance(section, dict):
            flat.update(section)
    flat.update(identity)
    for k in keep:
        if k in flat:
            out[k] = flat[k]
        else:
            missing.append(k)
    out['certifiable'] = bool(out.get('witness_equal')) and out.get('hours_per_time_unit') == 1.0
    if missing:
        raise Refusal('ONLINE_PARAMETER_MISSING', {'missing': missing})
    return out


# =============================================================================================
# (iv) the install gate is fail-closed (source fact, not inference)
# =============================================================================================
def install_gate_witness(source_text):
    """Prove from the source that `run_repair_transaction` refuses when `certify_fn is None`."""
    tree = ast.parse(source_text)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'run_repair_transaction':
            fn = node
            break
    if fn is None:
        raise Refusal('INSTALL_ENTRY_NOT_FOUND')
    guard_line, refuse_line, fails_closed = None, None, False
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test)
        if 'certify_fn' in test and 'None' in test:
            guard_line = node.lineno
            body_text = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            if 'no_certifier' in body_text and 'finish' in body_text:
                fails_closed = True
                refuse_line = node.body[0].lineno
            if 'refused_before_sampling' in body_text:
                fails_closed = fails_closed and True
    if guard_line is None:
        raise Refusal('INSTALL_GATE_MISSING',
                      {'why': 'run_repair_transaction no longer refuses a missing certifier'})
    return {'entry': 'run_repair_transaction', 'guard_line': guard_line,
            'refuse_line': refuse_line, 'fails_closed': bool(fails_closed),
            'reading': ('without a certifier the transaction refuses BEFORE sampling '
                        '(failure_reason=no_certifier, refused_before_sampling=True) and installs '
                        'nothing -- it does not silently install an uncertified plan'),
            'not_claimed': ['that a certifier is reachable in production (it is not)']}


# =============================================================================================
# runtime monitor: the mechanical witness for "no callback advanced the chain extra times"
# =============================================================================================
def attach_advance_monitor(env, n_max):
    """Count every call into `_advance_coldchain_to` and record the maximum clock it was given.

    This is the only way to keep a static bound honest in the presence of OPEN callbacks: the bond
    is "the static core bound PLUS zero advances from callbacks", and the monitor is what observes
    the second half."""
    state = {'counts': {}, 'clocks': [], 'n_max': int(n_max)}
    orig = env._advance_coldchain_to

    def counting(v, clock):
        state['counts'][v.vehicle_id] = state['counts'].get(v.vehicle_id, 0) + 1
        state['clocks'].append(float(clock))
        if state['counts'][v.vehicle_id] > state['n_max']:
            raise Refusal('COUNT_BOUND_VIOLATED', {
                'vehicle': int(v.vehicle_id), 'count': state['counts'][v.vehicle_id],
                'n_max': state['n_max']})
        return orig(v, clock)

    env._advance_coldchain_to = counting
    state['max_observed'] = lambda: (max(state['counts'].values()) if state['counts'] else 0)
    state['max_clock'] = lambda: (max(state['clocks']) if state['clocks'] else None)
    return state


# =============================================================================================
# the audit
# =============================================================================================
def qualify_clock(*, horizon, travel_bound, travel_source, n_max, hours_per_time_unit=1.0,
                  observed_max_clock=None, n_callback_advances=0, monitor_attached=False,
                  offline_witness=None):
    """Fold the errata into one verdict.  Anything unknown is reported, never assumed."""
    refusals, notes, conditional = [], [], []
    bound = None
    try:
        bound = clock_upper_bound(horizon, travel_bound, travel_source)
    except Refusal as exc:
        refusals.append(exc.as_dict())

    slack = None
    if bound is not None:
        slack = slack_budget(bound['T_clock_sup'], n_max, hours_per_time_unit)
        if not slack['certifiable']:
            conditional.append({'code': 'TIME_UNIT_CONVERSION_NOT_BUDGETED',
                                'detail': {'hours_per_time_unit': hours_per_time_unit}})
        if observed_max_clock is not None and _finite(observed_max_clock):
            if float(observed_max_clock) > bound['T_clock_sup'] + 1e-12:
                refusals.append({'code': 'CLOCK_EXCEEDS_UPPER_BOUND',
                                 'detail': {'observed': float(observed_max_clock),
                                            'T_clock_sup': bound['T_clock_sup']}})
        if not bound['same_binade']:
            notes.append('the clock bound crosses a binade, so the corrected basis CHANGES the '
                         'slack (horizon-only would have under-estimated it)')

    if n_callback_advances and not monitor_attached:
        conditional.append({'code': 'CALLBACK_ADVANCES_UNWITNESSED',
                            'detail': {'n_callback_advances': int(n_callback_advances),
                                       'why': 'an OPEN callback advanced the cold chain and no '
                                              'monitor bounds the count'}})
    if offline_witness is not None and not offline_witness.get('witness_equal'):
        conditional.append({'code': 'HIDDEN_DERIVED_HORIZON',
                            'detail': {'leak': offline_witness.get('tw_end_max', 0.0)
                                               - offline_witness.get('depot_tw_end', 0.0)}})

    if refusals:
        verdict = REFUSED
    elif conditional:
        verdict = NOT_CERTIFIABLE
    elif notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL
    return {'schema': SCHEMA, 'verdict': verdict,
            'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
            'implementation_version': IMPLEMENTATION_VERSION,
            'clock_bound': bound, 'slack': slack,
            'refusals': refusals, 'conditional': conditional, 'notes': notes,
            'callback_sinks': [{'sink': a, 'where': b, 'call_path': c, 'scope': d, 'note': e}
                               for a, b, c, d, e in CALLBACK_SINKS],
            'claim': ('the ulp budget is taken over a bound on the actual maximum timestamp, and the '
                      'advance count is bounded by max_iter plus a monitor that catches callback '
                      'advances'),
            'not_claimed': [
                'that the travel bound is tight',
                'that an OPEN callback cannot advance the chain (only that a monitor would see it)',
                'that a certifier is reachable in production',
            ]}
