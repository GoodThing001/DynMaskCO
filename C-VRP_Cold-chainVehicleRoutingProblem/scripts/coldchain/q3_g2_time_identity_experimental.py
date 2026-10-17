#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION  --  G-2: time-parameter identity and strict-online source
qualification for the Q3 / G-TMIN certification chain.

WHY THIS FILE EXISTS
====================
The Q3 L1 layer certifies an inter-door cooling bound that depends on four TIME quantities:

    (a) the episode horizon T_max      horizon = float(self.tw_end[inst_idx].max())
                                       strict_online_env.py:454 (and :596 in run_resumed)
    (b) the number n of updates        loosely: max_iter = 10000, strict_online_env.py:481
    (c) the unit scale                 hours_per_time_unit, coldchain_contract.py:36 / :521
    (d) the tolerances                 1e-9 / 1e-6 / 1e-6

Each of them exists, has a value, and is used.  What did NOT exist is an answer to three questions
that are about QUALIFICATION, not about arithmetic:

    Q1  where does the value come from, and is that source allowed to be read at decision time?
    Q2  does the value carry information about orders that have not been revealed yet?
    Q3  can an identity be built that fails closed when any of the above changes?

The reviewer's finding that motivates this module is precise: tw_end is the FULL instance array, so
the horizon is by definition a function of unrevealed orders' deadlines.  A simulator may hold hidden
data in order to advance the real world; a CERTIFIER may not silently reuse it.  Those are different
privileges and this module keeps them apart instead of merging them.

THREE LAYERS THAT MUST NOT BE MERGED
====================================
    SIMULATOR-TRUTH   what the world simulator may know in order to advance events:
                      max(tw_end), the full reveal_time vector, future service completions.
    ONLINE-DECISION   what the strict-online decider may use at the current decision point:
                      revealed orders, the public booking horizon, the depot return deadline
                      (_plan_and_commit builds visible_ids from reveal_time <= clock).
    CERTIFIER-PROOF   what a certificate may depend on: quantities the certifier can re-derive from
                      admissible inputs, plus an explicit WITNESS for anything borrowed from above.

A value that lives in layer 1 and is numerically equal to a layer-2 value is not thereby a layer-2
value.  It becomes admissible only with a witness, and this module demands the witness mechanically
(max(tw_end) == tw_end[0] per instance) instead of trusting the coincidence.

WHAT THIS MODULE DERIVES INSTEAD OF INHERITING
=============================================
n is NOT taken from max_iter.  An earlier local module
(q3_time_budget_audit_experimental.MAX_ITER) wrote "every iteration performs at most one advance per
vehicle, so 10000 bounds the number of advances".  That reading is incomplete: _commit also
advances, and _plan_and_commit runs once per iteration, so the per-vehicle bound is
2*n_iter + 3 -- about twice the inherited number.  This module therefore re-derives the bound from a
STATIC CALL-GRAPH CENSUS of strict_online_env.py (AST, no execution) and refuses a budget built on
the inherited reading (INSUFFICIENT_COUNT_BOUND).

WHAT THIS MODULE DOES NOT DO
===========================
No production receipt; no change to the certifier; no claim that T_end currently HAS a production
caller (it does not -- the class is only constructed inside tests).  No claim about L2 (real-valued
temperature) or L3 (stored floats).  It does not decide whether the horizon SHOULD be public; it
decides whether a given value is ADMISSIBLE and records why.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'G2_TIME_IDENTITY_V1'
SCHEMA = 'q3_g2_time_identity'

#: verdict ladder, most severe first
REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'

#: the three tolerances, named by role (never by value only).
#: eps_skip is the EXACT rational 1/10^9; the source literal `1e-9` is the nearest double, which is
#: larger than the rational by ~6.2e-26.  The slack budget spends a half-ulp of the horizon on the
#: addition `start + 1e-9` anyway (>= 1.8e-15 at T=24), which outweighs that difference by more than
#: ten orders of magnitude, so the rational is still an upper bound on the skipped span.
EPS_SKIP = Fraction(1, 10 ** 9)          # end <= start + 1e-9  -> the anchor is not moved
EPS_BACKWARD = 1e-6                      # end < start - 1e-6   -> ValueError, else silent skip
EPS_EVENT = 1e-6                         # event / termination comparisons

#: the files whose identity a time-parameter certificate depends on
AUDITED_FILES = (
    'scripts/simulation/strict_online_env.py',
    'scripts/coldchain/coldchain_state.py',
    'scripts/coldchain/coldchain_contract.py',
    'scripts/m1b_adapter/m1b_quality_certificate.py',
    'scripts/m1b_adapter/m1b_q2_certifier.py',
)

#: the advance-count derivation, as formulas (the census fills in the coefficients)
ADVANCE_BOUND_FORMULA = 'n_max_per_vehicle = pre_loop + per_iter * n_iter + termination'
ITER_BOUND_FORMULA = 'n_iter <= min(max_iter, 1 + floor(T_max / eps_event))'

PUBLIC_BY_WITNESS = 'PUBLIC_BY_WITNESS'
HIDDEN_DERIVED = 'HIDDEN_DERIVED'
VISIBILITY_LAYERS = ('SIMULATOR-TRUTH', 'ONLINE-DECISION', 'CERTIFIER-PROOF')


class Refusal(Exception):
    """A hard stop: the input is not admissible.  Never downgraded to a warning."""

    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail or {}

    def as_dict(self):
        return {'code': self.code, 'detail': self.detail}


# =============================================================================================
# exact float bounds (self-contained: this module must not depend on a local unaudited asset)
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
        if k > 2000:                                                       # pragma: no cover
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


def canon(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def sha256_json(payload):
    return hashlib.sha256(canon(payload).encode('utf-8')).hexdigest()


# =============================================================================================
# code identity (committed blob first -- a worktree digest is not portable under autocrlf)
# =============================================================================================
def _git_blob(rel_path, repo_root):
    try:
        proc = subprocess.run(['git', 'show', 'HEAD:%s' % rel_path], cwd=str(repo_root),
                              capture_output=True)
    except Exception:                                                      # noqa: BLE001
        return None
    return proc.stdout or None


def source_identity(ext_root, repo_root=None, rel_prefix=''):
    """sha256 of each audited file plus the basis.  Failure is explicit, never silent."""
    ext_root = Path(ext_root)
    repo_root = Path(repo_root) if repo_root is not None else ext_root.parent
    out, bases = {}, set()
    for rel in AUDITED_FILES:
        blob = _git_blob(rel_prefix + rel, repo_root)
        if blob:
            out[rel] = hashlib.sha256(blob).hexdigest()
            bases.add('committed blob')
        else:
            path = ext_root / rel
            if not path.exists():
                raise Refusal('SOURCE_FILE_MISSING', {'file': rel})
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            bases.add('worktree bytes (git unavailable)')
    basis = sorted(bases)[0] if len(bases) == 1 else sorted(bases)
    return {'files': out, 'basis': basis}


# =============================================================================================
# static call-graph census of the advance path (no execution, no data)
# =============================================================================================
def _class_methods(tree, class_name='StrictOnlineEnv'):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {f.name: f for f in node.body if isinstance(f, ast.FunctionDef)}
    raise Refusal('CLASS_NOT_FOUND', {'class': class_name})


def _block_ends_with_return(body):
    return bool(body) and isinstance(body[-1], ast.Return)


def _enclosing_exclusive_if(function_node, call_node):
    """Is the call inside an `if` branch that ends with `return`?

    Such a branch is mutually exclusive with everything written AFTER the `if`, which is what
    licenses counting the site as at most one execution.
    """
    for node in ast.walk(function_node):
        if not isinstance(node, ast.If):
            continue
        for sub in node.body:
            hit = (sub is call_node) or any(inner is call_node for inner in ast.walk(sub))
            if hit:
                return ((True, node.lineno) if _block_ends_with_return(node.body)
                        else (False, None))
    return False, None


def _calls_in(function_node):
    """Every self.<method>(...) call in the body, in source order."""
    found = []
    for node in ast.walk(function_node):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
            found.append((node.func.attr, node))
    return sorted(found, key=lambda t: t[1].lineno)


def _calls_in_loop(loop_node):
    if loop_node is None:
        return []
    found = []
    for node in ast.walk(loop_node):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
            found.append((node.func.attr, node))
    return sorted(found, key=lambda t: t[1].lineno)


def _loop_over_vehicles(function_node):
    """Line numbers of `for ... in vehicles` loops (the repetition this module accounts for)."""
    hits = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) \
                and node.iter.id == 'vehicles':
            hits.append(node.lineno)
    return sorted(hits)


def _inside_node(method, node, other):
    """Is `node` inside AST subtree `other`?  Separates pre-loop calls from in-loop calls."""
    if other is None:
        return False
    return any(cand is node for cand in ast.walk(other))


def _is_break_terminated(loop_node, call_node):
    """Is the call the statement immediately before a `break` in its own block?"""
    for node in ast.walk(loop_node):
        for field in ('body', 'orelse'):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for idx, stmt in enumerate(body[:-1]):
                if not isinstance(body[idx + 1], ast.Break):
                    continue
                if any(inner is call_node for inner in ast.walk(stmt)):
                    return True
    return False


def call_graph_census(source_text, target='_advance_coldchain_to'):
    """Derive the per-vehicle advance-count bound from the source structure.

    The bound is an UPPER bound built from three structurally distinct contributions:

        pre_loop     the `_plan_and_commit` call each entry point makes BEFORE the event loop
                     (`run` / `run_resumed`), which commits at most once per vehicle
        per_iter     one `_advance_fleet` advance plus one `_commit` advance, per iteration
        termination  `_force_return_all` (advance + single-vehicle fleet advance), at most once,
                     because the only call site is immediately followed by `break`

    Exclusivity inside `_commit` is NOT assumed: it is WITNESSED -- the `if node == 0:` branch ends
    with `return`, so its advance site cannot be added to the site after the `if`.  If that witness
    disappears the multiplicity becomes the naive call-site count and the bound grows.
    """
    tree = ast.parse(source_text)
    methods = _class_methods(tree)

    # --- the iteration cap of the entry loop -------------------------------------------------
    max_iter, max_iter_line = None, None
    for node in ast.walk(methods['_run_loop']):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, int) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'max_iter':
            max_iter, max_iter_line = int(node.value.value), node.lineno
    if max_iter is None:
        raise Refusal('MAX_ITER_NOT_FOUND', {'method': '_run_loop'})

    # --- per-function multiplicity of the advance, with the exclusivity witnesses -------------
    direct, witnesses = {}, {}
    for name, fn in methods.items():
        exclusive, plain = [], []
        for callee, node in _calls_in(fn):
            if callee != target:
                continue
            is_excl, if_line = _enclosing_exclusive_if(fn, node)
            (exclusive if is_excl else plain).append((node.lineno, if_line))
        if exclusive or plain:
            direct[name] = max(len(plain), 1 if exclusive else 0)
            witnesses[name] = ([{'line': ln, 'exclusive_via_return_at': ifl, 'how': 'return-branch'}
                                for ln, ifl in exclusive]
                               + [{'line': ln, 'exclusive_via_return_at': None, 'how': 'additive'}
                                  for ln, _ in plain])

    # --- advance cost of one invocation of a method, for ONE vehicle -------------------------
    def cost(name, seen):
        if name == target:
            return 1
        if name not in methods:
            return 0
        if name in seen:
            raise Refusal('CALL_GRAPH_CYCLE', {'at': name, 'path': sorted(seen | {name})})
        total = direct.get(name, 0)
        for callee, _node in _calls_in(methods[name]):
            if callee == target:
                continue
            total += cost(callee, seen | {name})
        return total

    # --- split the loop body into per-iteration work and the break-terminated branch ---------
    loop_node = None
    for node in ast.walk(methods['_run_loop']):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) \
                and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
            loop_node = node
            break
    if loop_node is None:
        raise Refusal('EVENT_LOOP_NOT_FOUND', {'method': '_run_loop'})

    once_calls, per_iter_calls = [], []
    for callee, node in _calls_in_loop(loop_node):
        if _is_break_terminated(loop_node, node):
            once_calls.append((callee, node.lineno))
        else:
            per_iter_calls.append((callee, node.lineno))

    per_iteration = sum(cost(c, set()) for c, _ln in per_iter_calls)
    termination = sum(cost(c, set()) for c, _ln in once_calls)

    # --- entry points call `_plan_and_commit` once before the loop ---------------------------
    pre_loop_per_entry, entry_points = {}, []
    for name in ('run', 'run_resumed'):
        if name not in methods:
            continue
        n = 0
        for callee, node in _calls_in(methods[name]):
            if callee == '_plan_and_commit' and not _inside_node(methods[name], node, loop_node):
                n += cost(callee, set())
        pre_loop_per_entry[name] = n
        entry_points.append(name)
    pre_loop = max(pre_loop_per_entry.values()) if pre_loop_per_entry else 0
    if pre_loop == 0:                                                      # pragma: no cover
        raise Refusal('PRE_LOOP_CONTRIBUTION_NOT_FOUND', {'entry_points': entry_points})

    n_max = pre_loop + per_iteration * max_iter + termination
    receipt = {
        'target': target,
        'entry_points': entry_points, 'pre_loop_per_entry': pre_loop_per_entry,
        'pre_loop': pre_loop, 'per_iteration': per_iteration, 'termination': termination,
        'per_iteration_call_sites': [{'method': c, 'line': ln} for c, ln in per_iter_calls],
        'once_call_sites': [{'method': c, 'line': ln} for c, ln in once_calls],
        'max_iter': max_iter, 'max_iter_line': max_iter_line,
        'advance_call_sites': {name: witnesses[name] for name in sorted(witnesses)},
        'multiplicity': {name: direct[name] for name in sorted(direct)},
        'n_max_per_vehicle': n_max,
        'n_max_formula': '%d + %d*max_iter + %d' % (pre_loop, per_iteration, termination),
        'formula': ADVANCE_BOUND_FORMULA,
        'iteration_formula': ITER_BOUND_FORMULA,
        'loop_over_vehicles_at': _loop_over_vehicles(methods['_run_loop']),
        'exclusivity_is_witnessed_not_assumed': True,
        'note': ('the bound is an UPPER bound obtained from call sites, not a measurement; an edit '
                 'that adds an advance path changes n_max_per_vehicle and therefore the identity'),
    }
    receipt['census_digest'] = sha256_json(receipt)
    return receipt


# =============================================================================================
# horizon: source classification and witness
# =============================================================================================
HORIZON_SOURCES = {
    'tw_end.max': ('SIMULATOR-TRUTH', HIDDEN_DERIVED,
                   'defined over the FULL tw_end array, including unrevealed orders'),
    'tw_end[0]': ('ONLINE-DECISION', PUBLIC_BY_WITNESS,
                  'the depot return deadline, already read online by the replanner'),
    'caller_declared_public': ('CERTIFIER-PROOF', PUBLIC_BY_WITNESS,
                               'a caller-declared value; this module still demands the witness'),
}


def classify_horizon(tw_end_row, reveal_time_row=None):
    """Classify the horizon of ONE instance and produce the witness the certifier needs.

    The witness is max(tw_end) == tw_end[0]: it turns "the hidden-derived maximum happens to equal a
    public quantity" into a checkable statement about THIS instance instead of a dataset-wide hope.
    hidden_max is reported separately because it is the quantity that WOULD leak.
    """
    import numpy as np
    row = np.asarray(tw_end_row, dtype=np.float64)
    if row.ndim != 1 or row.size < 2:
        raise Refusal('HORIZON_ROW_INVALID', {'shape': list(row.shape)})
    if not np.all(np.isfinite(row)):
        raise Refusal('HORIZON_INVALID', {'non_finite': int((~np.isfinite(row)).sum())})
    tw_max = float(row.max())
    depot = float(row[0])
    if reveal_time_row is None:
        hidden_max, hidden_ids = None, None
    else:
        rt = np.asarray(reveal_time_row, dtype=np.float64)
        mask = rt > 0
        hidden_max = (float(row[mask].max()) if bool(mask.any()) else None)
        hidden_ids = int(mask.sum())
    equal = (tw_max == depot)
    return {
        'tw_end_max': tw_max, 'depot_tw_end': depot, 'hidden_customer_max': hidden_max,
        'n_hidden_customers': hidden_ids,
        'witness_equal': bool(equal),
        'classification': PUBLIC_BY_WITNESS if equal else HIDDEN_DERIVED,
        'defined_source': 'tw_end.max',
        'admissible_source': 'tw_end[0]' if equal else None,
        'leak_magnitude': (tw_max - depot) if not equal else 0.0,
        'why': ('the full-array maximum equals the depot deadline, which the strict-online decider '
                'already reads; without this equality the maximum would carry an unrevealed '
                "order's deadline"),
    }


def require_public_horizon(witness, claimed_value):
    """Fail closed unless the claimed value is the witnessed public quantity."""
    if witness is None:
        raise Refusal('HORIZON_WITNESS_MISSING')
    if not _finite(claimed_value):
        raise Refusal('HORIZON_INVALID', {'value': repr(claimed_value)})
    if not witness.get('witness_equal'):
        raise Refusal('HIDDEN_DERIVED_HORIZON', {
            'tw_end_max': witness.get('tw_end_max'), 'depot_tw_end': witness.get('depot_tw_end'),
            'leak_magnitude': witness.get('leak_magnitude'),
            'why': 'the episode maximum exceeds the public depot deadline, so it is a function of an '
                   'unrevealed order and is not admissible for an online certificate'})
    if float(claimed_value) != float(witness['depot_tw_end']):
        raise Refusal('HORIZON_VALUE_MISMATCH', {
            'claimed': float(claimed_value), 'witnessed_public': float(witness['depot_tw_end'])})
    return float(witness['depot_tw_end'])


def require_upper_bound(horizon, observed_max_clock):
    """A horizon smaller than an observed clock is not an upper bound."""
    if not _finite(horizon):
        raise Refusal('HORIZON_INVALID', {'value': repr(horizon)})
    if float(horizon) <= 0:
        raise Refusal('HORIZON_INVALID', {'value': float(horizon), 'why': 'non-positive'})
    if observed_max_clock is not None and _finite(observed_max_clock):
        if float(observed_max_clock) > float(horizon) + 1e-12:
            raise Refusal('HORIZON_NOT_AN_UPPER_BOUND', {
                'horizon': float(horizon), 'observed_max_clock': float(observed_max_clock)})
    return float(horizon)


# =============================================================================================
# iteration bound and the advance / slack budgets
# =============================================================================================
def iteration_bound(horizon, max_iter, eps_event=EPS_EVENT):
    """Iterations are bounded BOTH by max_iter and by the clock advancing > eps_event each step."""
    time_side = 1 + int(math.floor(Fraction(float(horizon)) / Fraction(float(eps_event))))
    both = min(int(max_iter), time_side)
    return {'max_iter': int(max_iter), 'time_side': time_side, 'bound': both,
            'binding': 'max_iter' if int(max_iter) <= time_side else 'time',
            'eps_event': float(eps_event), 'formula': ITER_BOUND_FORMULA,
            'why_time_side': ('every iteration either breaks or moves the clock by more than '
                              'eps_event, so n_iter <= 1 + floor(T_max / eps_event)')}


def advance_budget(horizon, census, declared_n=None):
    """Per-vehicle advance bound from the census, and the gate that rejects the inherited number.

    declared_n is the number the caller wants to use.  If it is smaller than the derived bound the
    call is REFUSED -- that gate is what rejects the inherited n = max_iter reading.
    """
    it = iteration_bound(horizon, census['max_iter'])
    derived = census['pre_loop'] + census['per_iteration'] * it['bound'] + census['termination']
    if declared_n is not None:
        if int(declared_n) < derived:
            raise Refusal('INSUFFICIENT_COUNT_BOUND', {
                'declared_n': int(declared_n), 'derived_n_max': derived,
                'why': 'the declared count bound is smaller than the one derived from the call '
                       'graph, so the rounding budget would be under-estimated'})
        used = int(declared_n)
    else:
        used = derived
    return {'n_max_per_vehicle': used, 'derived_n_max': derived, 'iterations': it,
            'census': {'per_iteration': census['per_iteration'], 'pre_loop': census['pre_loop'],
                       'termination': census['termination'],
                       'max_iter': census['max_iter'],
                       'census_digest': census['census_digest']},
            'formula': ADVANCE_BOUND_FORMULA}


def slack_budget(horizon, n_max, hours_per_time_unit=1.0):
    """The three slack terms for an interval, exactly, with the unit question made explicit."""
    hpu = float(hours_per_time_unit)
    hu = half_ulp_ceiling(horizon)
    skip = EPS_SKIP + hu
    rounding = Fraction(int(n_max)) * hu
    unit = None if hpu != 1.0 else Fraction(0)
    total = None if unit is None else skip + rounding + unit
    return {'horizon': float(horizon), 'n_max_per_vehicle': int(n_max),
            'hours_per_time_unit': hpu,
            'half_ulp_ceiling': float(hu),
            'skip_term': float(skip), 'skip_term_exact': str(skip),
            'rounding_term': float(rounding), 'rounding_term_exact': str(rounding),
            'unit_term': (None if unit is None else float(unit)),
            'total_slack': (None if total is None else float(total)),
            'total_slack_exact': (None if total is None else str(total)),
            'certifiable': total is not None,
            'reasons': {
                'skip': 'the anchor is not moved when the span is <= 1e-9; the skipped span is '
                        'covered by a later advance unless it is the last one in the interval',
                'rounding': 'each real advance passes fl(end - start) to the executor; the sum of '
                            'n such roundings is bounded by n * ulp/2 and does not cancel',
                'unit': 'a non-unit hours_per_time_unit adds one rounding per advance that this '
                        'budget does not cover',
            }}


# =============================================================================================
# identity object
# =============================================================================================
def build_identity(*, source, census, contract_hash, contract, horizon_witness, max_iter,
                   domain, tolerances=None):
    """The G-2 shared time-identity object.

    It binds: contract identity, code identity, the unit scale, the PUBLIC horizon with its witness,
    the event/advance budget with its derivation, the tolerances and the applicable domain.  The
    digest covers every field, so any drift changes the identity.
    """
    units = getattr(contract, 'units', None)
    hpu = None if units is None else float(getattr(units, 'hours_per_time_unit', float('nan')))
    payload = {
        'schema': SCHEMA, 'version': IMPLEMENTATION_VERSION,
        'not_for_production': True, 'production_consumable': False,
        'contract': {
            'contract_hash': contract_hash,
            'hours_per_time_unit': hpu,
            'distance_km_per_unit': None if units is None else float(
                getattr(units, 'distance_km_per_unit', float('nan'))),
        },
        'code': {'files': dict(source['files']), 'basis': source['basis'],
                 'max_iter': int(max_iter), 'census_digest': census['census_digest']},
        'time_unit': {
            'hours_per_time_unit': hpu,
            'identity_bound': 'inside contract_hash (asdict(contract) is hashed)',
            'certifiable': (hpu is not None and hpu == 1.0),
        },
        'horizon': {
            'value': float(horizon_witness['depot_tw_end']),
            'defined_source': horizon_witness['defined_source'],
            'admissible_source': horizon_witness['admissible_source'],
            'classification': horizon_witness['classification'],
            'witness_equal': bool(horizon_witness['witness_equal']),
            'leak_magnitude': float(horizon_witness['leak_magnitude']),
        },
        'advance_budget': {
            'n_max_per_vehicle_at_max_iter': int(
                census['pre_loop'] + census['per_iteration'] * int(max_iter)
                + census['termination']),
            'per_iteration': census['per_iteration'], 'pre_loop': census['pre_loop'],
            'termination': census['termination'], 'max_iter': int(max_iter),
            'formula': ADVANCE_BOUND_FORMULA, 'iteration_formula': ITER_BOUND_FORMULA,
            'inherited_reading_rejected': 'max_iter alone is NOT accepted as the advance bound',
        },
        'tolerances': {
            'eps_skip': str(EPS_SKIP), 'eps_backward': EPS_BACKWARD, 'eps_event': EPS_EVENT,
            'in_slack_budget': ['eps_skip'],
            'not_in_slack_budget': ['eps_backward', 'eps_event'],
            'why_not': ('eps_backward only decides whether a non-positive span is refused or '
                        'silently skipped, and a skipped span moves no anchor; eps_event shifts '
                        'EVENT TIMESTAMPS, while the durations inside one interval telescope to '
                        '(clock_next - clock_prev) regardless of when the events fired'),
        },
        'domain': dict(domain),
    }
    if tolerances:
        payload['tolerances'].update(tolerances)
    payload['identity_digest'] = sha256_json(payload)
    return payload


def verify_identity(identity, recomputed_source=None, recomputed_census=None, contract_hash=None):
    """Fail closed if any bound field disagrees with a freshly recomputed value."""
    problems = []
    if recomputed_source is not None and identity['code']['files'] != recomputed_source['files']:
        problems.append({'field': 'code.files',
                         'why': 'a source file changed after the identity was built'})
    if recomputed_census is not None:
        if identity['code']['census_digest'] != recomputed_census['census_digest']:
            problems.append({'field': 'code.census_digest',
                             'why': 'the call graph changed after the identity was built',
                             'was': identity['code']['census_digest'],
                             'now': recomputed_census['census_digest']})
        if identity['advance_budget']['max_iter'] != recomputed_census['max_iter']:
            problems.append({'field': 'advance_budget.max_iter',
                             'was': identity['advance_budget']['max_iter'],
                             'now': recomputed_census['max_iter']})
    if contract_hash is not None and identity['contract']['contract_hash'] != contract_hash:
        problems.append({'field': 'contract.contract_hash',
                         'why': 'the unit scale (or another contract field) changed',
                         'was': identity['contract']['contract_hash'], 'now': contract_hash})
    if problems:
        raise Refusal('IDENTITY_MISMATCH', {'problems': problems})
    return True


# =============================================================================================
# the matrix and the qualification audit
# =============================================================================================
def parameter_matrix():
    """field -> defined source -> visibility -> admissibility, one row per time parameter."""
    return [
        {'field': 'horizon / T_max', 'where': 'strict_online_env.py:454, :596',
         'defined_source': 'tw_end.max() over the FULL instance array',
         'visibility': 'SIMULATOR-TRUTH', 'carries_unrevealed_info': True,
         'admissible_source': 'tw_end[0] with witness tw_end.max() == tw_end[0]',
         'admissible_visibility': 'ONLINE-DECISION',
         'admissibility': 'ADMISSIBLE_ONLY_WITH_WITNESS',
         'if_witness_absent': 'NOT_CERTIFIABLE (HIDDEN_DERIVED_HORIZON)'},
        {'field': 'tw_end[0] (the admitted horizon)',
         'where': 'read online by the replanner, e.g. strict_online_env.py:717',
         'defined_source': 'the depot return deadline of the instance',
         'visibility': 'ONLINE-DECISION', 'carries_unrevealed_info': False,
         'admissible_source': 'itself, provided max(tw_end) == tw_end[0] is witnessed',
         'admissible_visibility': 'ONLINE-DECISION', 'admissibility': 'ADMISSIBLE',
         'if_witness_absent': 'the substitute is not available either; NOT_CERTIFIABLE'},
        {'field': 'n (temperature updates per vehicle)',
         'where': 'strict_online_env.py:481 with :272, :286, :304',
         'defined_source': 'derived from the static call graph; max_iter is only the loop cap',
         'visibility': 'CERTIFIER-PROOF', 'carries_unrevealed_info': False,
         'admissible_source': 'pre_loop + per_iter*n_iter + termination (this module derives it)',
         'admissible_visibility': 'CERTIFIER-PROOF',
         'admissibility': 'ADMISSIBLE_WHEN_DERIVED',
         'if_derivation_absent': 'REFUSED (INSUFFICIENT_COUNT_BOUND)'},
        {'field': 'hours_per_time_unit', 'where': 'coldchain_contract.py:36, :521',
         'defined_source': 'UnitScale, covered by contract_hash',
         'visibility': 'CERTIFIER-PROOF', 'carries_unrevealed_info': False,
         'admissible_source': 'the contract, and only when exactly 1.0',
         'admissible_visibility': 'CERTIFIER-PROOF',
         'admissibility': 'ADMISSIBLE_IF_UNIT',
         'if_non_unit': 'NOT_CERTIFIABLE (TIME_UNIT_CONVERSION_NOT_BUDGETED)'},
        {'field': 'eps_skip = 1e-9', 'where': 'strict_online_env.py:219',
         'defined_source': 'literal', 'visibility': 'CERTIFIER-PROOF',
         'carries_unrevealed_info': False,
         'admissible_source': 'literal; enters the slack budget as one constant per interval',
         'admissible_visibility': 'CERTIFIER-PROOF', 'admissibility': 'ADMISSIBLE',
         'if_changed': 'the identity digest changes'},
        {'field': 'eps_backward = 1e-6', 'where': 'strict_online_env.py:217',
         'defined_source': 'literal', 'visibility': 'CERTIFIER-PROOF',
         'carries_unrevealed_info': False,
         'admissible_source': 'literal; NOT in the slack budget (a skipped span moves no anchor)',
         'admissible_visibility': 'CERTIFIER-PROOF', 'admissibility': 'ADMISSIBLE',
         'if_changed': 'the identity digest changes'},
        {'field': 'eps_event = 1e-6', 'where': 'strict_online_env.py:306, :490, :506, :514',
         'defined_source': 'literal', 'visibility': 'CERTIFIER-PROOF',
         'carries_unrevealed_info': False,
         'admissible_source': ('literal; NOT in the slack budget -- it shifts event timestamps, '
                               'and the durations inside an interval telescope exactly'),
         'admissible_visibility': 'CERTIFIER-PROOF', 'admissibility': 'ADMISSIBLE',
         'if_changed': 'the identity digest changes'},
    ]


def qualify(*, identity, horizon_witness, horizon_value, census, contract, n_declared=None,
            resumed_between_doors=False, observed_max_clock=None, observed_advances=None,
            count_bound_violated=False, recomputed_source=None, recomputed_census=None):
    """Run the G-2 audit on one identity.  Verdict ladder: REFUSED > NOT_CERTIFIABLE > DIAGNOSTIC."""
    refusals, notes, not_certifiable = [], [], []
    units = getattr(contract, 'units', None)
    hpu = float(getattr(units, 'hours_per_time_unit', float('nan')))

    # 1. the horizon must be the witnessed public quantity and must bound the observed clocks
    pub = None
    try:
        pub = require_public_horizon(horizon_witness, horizon_value)
        pub = require_upper_bound(pub, observed_max_clock)
    except Refusal as exc:
        if exc.code == 'HIDDEN_DERIVED_HORIZON':
            not_certifiable.append(exc.as_dict())      # a source question, not a malformed input
        else:
            refusals.append(exc.as_dict())

    # 2. the unit scale
    if not (hpu == 1.0):
        not_certifiable.append({'code': 'TIME_UNIT_CONVERSION_NOT_BUDGETED',
                                'detail': {'hours_per_time_unit': hpu}})

    # 3. a resume inside the interval injects an anchor from outside the event loop
    if resumed_between_doors:
        refusals.append({'code': 'SNAPSHOT_RESTORE_INSIDE_INTERVAL',
                         'detail': {'why': 'the anchor can be injected from a snapshot, so the wall '
                                           'clock no longer bounds the processed cooling'}})

    # 4. the identity must still match the corpus
    try:
        verify_identity(identity, recomputed_source=recomputed_source,
                        recomputed_census=recomputed_census,
                        contract_hash=(contract.contract_hash if contract is not None else None))
    except Refusal as exc:
        refusals.append(exc.as_dict())

    # 5. the count bound must come from the call graph, not from max_iter
    budget, slack = None, None
    if census is not None and pub is not None:
        try:
            budget = advance_budget(pub, census, declared_n=n_declared)
        except Refusal as exc:
            refusals.append(exc.as_dict())
        if budget is not None:
            if count_bound_violated:
                not_certifiable.append({'code': 'COUNT_BOUND_VIOLATED',
                                        'detail': {'why': 'an advance count exceeded the derived '
                                                          'bound'}})
            elif observed_advances is not None \
                    and int(observed_advances) > budget['n_max_per_vehicle']:
                not_certifiable.append({'code': 'COUNT_BOUND_VIOLATED',
                                        'detail': {'observed_advances': int(observed_advances),
                                                   'n_max_per_vehicle':
                                                       budget['n_max_per_vehicle']}})
            elif observed_advances is not None:
                notes.append('advances_observed=%d <= bound=%d'
                             % (int(observed_advances), budget['n_max_per_vehicle']))
            slack = slack_budget(pub, budget['n_max_per_vehicle'], hpu)
            if not slack['certifiable']:
                not_certifiable.append({'code': 'TIME_UNIT_CONVERSION_NOT_BUDGETED',
                                        'detail': {'hours_per_time_unit': hpu, 'stage': 'slack'}})

    if refusals:
        verdict = REFUSED
    elif not_certifiable:
        verdict = NOT_CERTIFIABLE
    elif notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL

    return {
        'schema': 'q3_g2_time_qualification', 'verdict': verdict,
        'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
        'production_consumable': PRODUCTION_CONSUMABLE,
        'implementation_version': IMPLEMENTATION_VERSION,
        'identity_digest': identity.get('identity_digest'),
        'refusals': refusals, 'not_certifiable': not_certifiable, 'notes': notes,
        'horizon': {'witness': horizon_witness, 'admitted_value': pub,
                    'admitted_source': 'tw_end[0]' if pub is not None else None},
        'advance_budget': budget, 'slack': slack,
        'tolerances': {'eps_skip': str(EPS_SKIP), 'eps_backward': EPS_BACKWARD,
                       'eps_event': EPS_EVENT},
        'matrix': parameter_matrix(),
        'layers': {k: 'kept separate by this module' for k in VISIBILITY_LAYERS},
        'claim': ('the time parameters used by a certificate are named, identity-bound, and either '
                  'public or witnessed; anything else is refused'),
        'not_claimed': [
            'that T_end currently has a production caller (it does not)',
            'that the horizon SHOULD be public',
            'any statement about L2 (real-valued temperature) or L3 (stored floats)',
            'anything about an interval containing a snapshot restore',
        ],
    }
