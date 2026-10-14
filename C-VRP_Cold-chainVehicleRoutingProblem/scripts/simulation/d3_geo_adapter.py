"""D3-Engineering: D-Geo adapter (non-learning geometric scorer) + B-mask pre-filter.

Authorised scope (user ruling 2026-10-14): D3-Engineering code development and offline
engineering verification only.  This module does NOT compare F with D-Geo, does NOT train,
does NOT collect data, and does NOT modify any historical behaviour.

Design constraints taken from the user's three technical corrections:
  * 32-state index is an INDEX only: state restoreability must be proven; otherwise those
    indices may serve static schema checks only (enforced elsewhere, see the state-audit tool).
  * The distance tie rule must NOT use a pairwise approximate-equality relation (not
    transitive).  It uses a UNIFIED MINIMUM reference instead:
        d_min = min_i delta_d[i];  T = {i : delta_d[i] <= d_min + eps};  pick in T by key.
    eps is PROVISIONAL in engineering and must be frozen only after a numerical-stability
    check.  It applies to D-Geo's distance score ONLY -- F's learned argmax rule is
    untouched (see `select_historical_argmax`, which reproduces the historical rule exactly).
  * Tie keys use REAL field types: `slot.kind` is the string 'anchored' | 'new_route'
    (ranked by a declared map), `slot.anchor` is an int, and a `new_route` slot's physical
    vehicle is resolved by deterministic physical matching (`action_contract._match_new_route`).

Why an independent adapter instead of editing `mpre_policy.stepwise_sample`:
the historical sampler's `np.argmax` semantics are what F was trained and tuned under.
Changing the shared decision path before the attribution comparison would silently change
the very policy being compared.  So the loop is mirrored here, the historical behaviour is
left byte-intact, and a test proves the historical mode reproduces it exactly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# declared constants
# ---------------------------------------------------------------------------
GEO_SCORE_KEY = 'geo_incremental_distance'
"""`state` key holding the per-action incremental TRAVEL distance, aligned with `legal`."""

SLOT_KIND_RANK = {'anchored': 0, 'new_route': 1}
"""Declared string ordering for `slot.kind`: never rely on container iteration order."""

# PROVISIONAL engineering defaults for the ROBUSTNESS VARIANT -- NOT frozen, and NOT the
# primary comparison.  Freezing requires a numerical-stability check on the distance
# computation (see `distance_stability_report`); the ~1e-5 figure seen elsewhere in this
# project is MODEL padding noise and is NOT a distance-error basis.
#
# PRIMARY comparison uses eps == 0 (see EPS_D_PRIMARY_*): with the legal list in canonical
# order, a zero tolerance makes D-Geo's tie resolution structurally IDENTICAL to F's
# first-maximum argmax, so the two arms then differ in exactly one factor -- the score
# vector.  Any eps > 0 introduces a SECOND factor (tolerance band) and is therefore only a
# registered robustness variant, never the primary contrast.
EPS_D_PRIMARY_REL = 0.0
EPS_D_PRIMARY_ABS = 0.0
EPS_D_REL_PROVISIONAL = 1e-9
EPS_D_ABS_PROVISIONAL = 1e-12

SELECTOR_GEO = 'geo_unified_minimum'
SELECTOR_HISTORICAL = 'historical_argmax'
SELECTORS = (SELECTOR_GEO, SELECTOR_HISTORICAL)

FAILURE_REASONS = (None, 'no_legal', 'nan', 'timeout', 'rollback', 'cert_fail')


def sha256_of(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_identity(paths):
    """Per-file SHA of every module this adapter depends on (for T-G9)."""
    return {str(p): sha256_of(p) for p in paths}


# ---------------------------------------------------------------------------
# ordered-mask discipline
# ---------------------------------------------------------------------------
def assert_ordered_mask(mask, *, name='mask_set'):
    """Reject a bare `set`/`frozenset`; return a list preserving the caller's order.

    `mpre_policy.stepwise_sample` does `remaining = list(mask_set)`, so a bare set makes the
    action order unspecified -- and `argmax`'s first-maximum rule then makes the tie outcome
    unspecified too.  The adapter therefore refuses unordered input.
    """
    if isinstance(mask, (set, frozenset)):
        raise TypeError(
            '%s must be an ORDERED sequence (list/tuple); got %s. A bare set makes the action '
            'order unspecified, and with first-maximum argmax the tie outcome becomes '
            'unspecified as well.' % (name, type(mask).__name__))
    out = [int(c) for c in mask]
    if len(set(out)) != len(out):
        raise ValueError('%s contains duplicates: %s' % (name, out))
    return out


def build_pool_B(vehicles, served_mask, visible_ids, plans, mutable_ids,
                 protected=frozenset(), pool_fn=None):
    """B-semantics mask pool, constructed BEFORE sampling (user ruling 1).

    B semantics = every mask customer must come from a `mutable_ids` vehicle's PLAN suffix
    (the training-era `validate_mask_scope` clause).  The pool is built by filtering, not by
    sampling from a deferred-containing pool and then relying on an assertion to fail.

    Returns `(pool_B_ordered, diag)`.  An empty pool is returned as `[]` with
    `diag['empty']=True`; callers MUST record it as-is and MUST NOT drop the state.
    """
    if pool_fn is None:
        from dynmaskco_cc_context import mask_candidate_pool as pool_fn
    # NOTE (P0 audit R1): the injected `pool_fn` must actually be used.  The pre-fix version
    # imported into a local name inside `if pool_fn is None:` and then called
    # `mask_candidate_pool(...)` unconditionally, so passing a pool_fn raised
    # UnboundLocalError and the injection silently never took effect.
    pool = [int(c) for c in pool_fn(vehicles, served_mask, visible_ids,
                                    protected=protected, plans=plans)]
    mutable_plan_customers = set()
    for vid, p in (plans or {}).items():
        if vid in mutable_ids:
            for x in p.suffix:
                if int(x) > 0:
                    mutable_plan_customers.add(int(x))
    pool_B = [c for c in pool if c in mutable_plan_customers]
    excluded = [c for c in pool if c not in mutable_plan_customers]
    diag = {'pool_full_ordered': pool, 'mutable_plan_customers': sorted(mutable_plan_customers),
            'excluded_as_deferred_or_outside_mutable_plans': excluded,
            'n_full': len(pool), 'n_B': len(pool_B), 'empty': len(pool_B) == 0,
            'rule': 'B: pool filtered by mutable_ids plan suffix BEFORE sampling; empty pool '
                    'is recorded as-is and the state is not dropped'}
    return pool_B, diag


# ---------------------------------------------------------------------------
# extract wrapper: expose the distance vector, index-aligned with `legal`
# ---------------------------------------------------------------------------
def resolve_vehicle_target(plans, action, allowed_vehicle_ids=None):
    """Deterministic physical vehicle resolution for an action (ties key + steps record)."""
    from action_contract import _match_anchored, _match_new_route
    if action.slot.kind == 'new_route':
        return _match_new_route(plans, allowed_vehicle_ids)
    return _match_anchored(plans, action.slot.anchor)


def make_geo_extract_fn(base_extract_fn=None, *, allowed_vehicle_ids=None):
    """Wrap a shared `extract_fn` so the state carries the geo distance vector.

    `base_extract_fn(P, remaining, legal)` -> dict (the SAME extractor every arm uses).
    The wrapper only APPENDS `GEO_SCORE_KEY` and `vehicle_target_vid`, both aligned with
    `legal`; it never re-enumerates and never recomputes the distance
    (`Candidate.incremental_distance` is the environment's own value).
    """
    def _extract(P, remaining, legal):
        state = dict(base_extract_fn(P, remaining, legal)) if base_extract_fn else {}
        state[GEO_SCORE_KEY] = [float(cand.incremental_distance) for _c, _a, cand in legal]
        state['vehicle_target_vid'] = [
            resolve_vehicle_target(P, a, allowed_vehicle_ids) for _c, a, _cand in legal]
        return state
    return _extract


def geo_score(state):
    """D-Geo: `s_D(a) = -delta_d(a)`.  Reads ONLY the distance vector (G2).

    Missing key or a mismatched length is a hard error: a silent fallback would let a wrong
    scorer masquerade as D-Geo.
    """
    if GEO_SCORE_KEY not in state:
        raise KeyError('%s missing from state; D-Geo may only score the environment distance'
                       % GEO_SCORE_KEY)
    return [-float(x) for x in state[GEO_SCORE_KEY]]


# ---------------------------------------------------------------------------
# tie-break: unified minimum reference + canonical action key
# ---------------------------------------------------------------------------
def canonical_action_key(customer_id, action, vehicle_id):
    """Total order over actions, using the REAL field types.

    Order: (customer_id, vehicle_id, slot_kind_rank, slot_anchor, position, predecessor,
            successor).
    * `slot.kind` is ranked through the declared `SLOT_KIND_RANK` map (string compare is fine
      but a declared rank makes the intent explicit and stable);
    * an anonymous `new_route` slot has `anchor == -1`, so it can never collide with an
      anchored slot's anchor;
    * `vehicle_id` is the deterministically RESOLVED physical vehicle, so an anonymous
      `new_route` and a normal anchored action form the same kind of key in any process.
    """
    kind = action.slot.kind
    if kind not in SLOT_KIND_RANK:
        raise ValueError('unknown slot.kind %r' % kind)
    return (int(customer_id), int(vehicle_id), SLOT_KIND_RANK[kind], int(action.slot.anchor),
            int(action.position), int(action.predecessor), int(action.successor))


def _eps_for(d_min, eps_rel, eps_abs):
    return max(float(eps_abs), float(eps_rel) * abs(float(d_min)))


def unified_minimum_tie_set(deltas, eps_rel=EPS_D_PRIMARY_REL, eps_abs=EPS_D_PRIMARY_ABS):
    """`d_min = min_i delta_d[i]`; `T = {i : delta_d[i] <= d_min + eps}`.

    Deliberately NOT a pairwise approximate-equality relation: approximate equality is not
    transitive, so pairwise comparisons would make the outcome depend on comparison order.
    Anchoring every comparison to the same `d_min` makes the tie set order-independent.
    """
    d = [float(x) for x in deltas]
    if not d:
        raise ValueError('empty distance vector')
    d_min = min(d)
    eps = _eps_for(d_min, eps_rel, eps_abs)
    T = [i for i, x in enumerate(d) if x <= d_min + eps]
    return d_min, eps, T


def select_geo(deltas, keys, eps_rel=EPS_D_PRIMARY_REL, eps_abs=EPS_D_PRIMARY_ABS):
    """D-Geo selection: argmin under the unified-minimum tie rule, tie broken by key.

    `single_factor_with_F` is True only when eps == 0: then, with the legal list in canonical
    order, `first-maximum argmax over -delta_d` and `argmin over delta_d with canonical-key
    tie-break` select the SAME index -- i.e. F and D-Geo differ in exactly one factor.  Any
    eps > 0 adds a tolerance band as a second factor and must be declared as such.
    """
    d_min, eps, T = unified_minimum_tie_set(deltas, eps_rel, eps_abs)
    idx = min(T, key=lambda i: keys[i])
    return idx, {'d_min': d_min, 'eps_used': eps, 'tie_set': list(T),
                 'selected': int(idx), 'n_candidates': len(deltas),
                 'eps_is_provisional': eps > 0.0,
                 'single_factor_with_F': eps == 0.0,
                 'rule': 'unified minimum reference; tie decided by canonical action key'}


def select_historical_argmax(scores):
    """F's historical rule, reproduced exactly: `np.argmax` = FIRST maximum.

    Kept separate on purpose.  Applying a distance tolerance to learned logits would be a
    different decision policy and would require its own equivalence test.
    """
    return int(np.argmax(np.asarray(scores)))


# ---------------------------------------------------------------------------
# budget accounting (user: engineering maxima are capacity planning ONLY)
# ---------------------------------------------------------------------------
class Budget(object):
    """Counter set; every unit is reported separately because they are NOT equal cost.

    P0 audit R2: `n_action_applications` counts calls to `apply_fn`.  It is NOT a solver-call
    count -- this adapter never invokes a solver.  Real solver invocations must be reported by
    the caller through `report_solver_calls()`; conflating the two would let a comparison
    claim equal solver work while the arms did different amounts of it.
    """

    KEYS = ('n_actions_enumerated', 'n_actions_scored', 'n_candidate_plans_certified',
            'n_action_applications', 'n_solver_calls', 'wall_seconds')

    def __init__(self):
        self.d = {k: 0 for k in self.KEYS}
        self.d['wall_seconds'] = 0.0
        self.started = time.perf_counter()

    def bump(self, key, n=1):
        if key not in self.d:
            raise KeyError(key)
        self.d[key] += n

    def report_solver_calls(self, n):
        """Called by the assembly layer with the REAL solver invocations it performed."""
        self.d['n_solver_calls'] += int(n)

    def add_wall(self, seconds):
        self.d['wall_seconds'] += float(seconds)

    def snapshot(self):
        out = dict(self.d)
        out['wall_seconds'] = round(out['wall_seconds'], 6)
        out['n_solver_calls_note'] = ('reported by the caller; stays 0 inside this adapter '
                                      'because no solver is invoked here')
        return out


# ---------------------------------------------------------------------------
# the D3 sampler loop (mirrors the historical contract; historical mode == original)
# ---------------------------------------------------------------------------
def stepwise_sample_d3(*, score_fn, extract_fn, legal_fn, apply_fn, plans, mask_set,
                       mutable_ids=None, selector=SELECTOR_GEO,
                       eps_rel=EPS_D_REL_PROVISIONAL, eps_abs=EPS_D_ABS_PROVISIONAL,
                       temperature=1.0, rng=None, deterministic=False, certify_fn=None,
                       budget=None, wall_clock_limit_s=None):
    """Sample/derive a full repair.  Returns `(plan, steps, records, failure_reason, diag)`.

    `legal_fn(P, remaining) -> legal` MUST be the shared `mpre_policy.enumerate_legal_actions`
    (injected so this module never re-implements enumeration).  `failure_reason` follows the
    historical vocabulary plus `cert_fail`, `timeout`, `rollback`; every failure is recorded
    and stays in the denominator.

    P0 audit R4 -- FAIL-CLOSED ROLLBACK: on ANY failure the returned plan is the ORIGINAL
    `plans`, never the partially repaired working plan.  Recording `failure_reason` alone
    would not stop a careless caller from installing a half-repaired plan; the partial plan is
    kept in `diag['partial_plan_at_failure']` as evidence instead.
    """
    if selector not in SELECTORS:
        raise ValueError('unknown selector %r' % selector)
    if rng is None:
        rng = np.random.default_rng(0)
    budget = budget if budget is not None else Budget()
    mask = assert_ordered_mask(mask_set)
    P = dict(plans)
    remaining = list(mask)
    records, steps = [], []
    diag = {'selector': selector, 'mask_order': list(mask), 'divergence': None,
            'n_steps': 0}

    def _rollback(reason):
        """Return the ORIGINAL plan + keep the partial one as evidence."""
        diag['rollback'] = True
        diag['rollback_to'] = 'the original plans passed in by the caller'
        diag['partial_plan_at_failure'] = {int(vid): [int(x) for x in p.suffix]
                                           for vid, p in P.items()}
        diag['n_steps'] = len(steps)
        budget.add_wall(time.perf_counter() - t0)
        return dict(plans), steps, records, reason, diag

    t0 = time.perf_counter()
    while remaining:
        if wall_clock_limit_s is not None and (time.perf_counter() - t0) > wall_clock_limit_s:
            return _rollback('timeout')
        legal = legal_fn(P, remaining)
        budget.bump('n_actions_enumerated', len(legal))
        if not legal:
            return _rollback('no_legal')
            diag['n_steps'] = len(steps)
            return _rollback('no_legal')
        state = extract_fn(P, remaining, legal)
        scores = np.asarray(score_fn(state))
        budget.bump('n_actions_scored', len(scores))
        if not np.isfinite(scores).all():
            return _rollback('nan')
        if selector == SELECTOR_HISTORICAL or not deterministic:
            # historical semantics: first-maximum argmax when deterministic, softmax when not
            if deterministic:
                idx = select_historical_argmax(scores)
                sel = {'rule': 'historical first-maximum argmax (F rule, unchanged)'}
            else:
                logits = scores / max(temperature, 1e-9)
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                idx = int(rng.choice(len(legal), p=probs))
                sel = {'rule': 'historical softmax sampling'}
        else:
            keys = [canonical_action_key(c, a, state['vehicle_target_vid'][i])
                    for i, (c, a, _cand) in enumerate(legal)]
            idx, sel = select_geo(state[GEO_SCORE_KEY], keys, eps_rel, eps_abs)
        c, a, _cand = legal[idx]
        vid = state['vehicle_target_vid'][idx]
        records.append({'state_keys': sorted(state.keys()), 'chosen': idx,
                        'selection': sel, 'customer': int(c), 'vehicle_id': int(vid),
                        'incremental_distance': float(_cand.incremental_distance)})
        P = apply_fn(P, a, allowed_vehicle_ids=mutable_ids)
        steps.append({'customer': int(c), 'vehicle_id': int(vid), 'slot_kind': a.slot.kind,
                      'slot_anchor': int(a.slot.anchor), 'position': int(a.position),
                      'predecessor': int(a.predecessor), 'successor': int(a.successor),
                      'incremental_distance': float(_cand.incremental_distance)})
        remaining.remove(c)
        budget.bump('n_action_applications')
    if certify_fn is not None:
        ok, detail = certify_fn(P)
        budget.bump('n_candidate_plans_certified')
        if not ok:
            diag['certify_detail'] = detail
            return _rollback('cert_fail')
    budget.add_wall(time.perf_counter() - t0)
    diag['n_steps'] = len(steps)
    return P, steps, records, None, diag


def distance_stability_report(pairs):
    """Evidence for freezing `eps_d` later: is the distance computation exact on ties?

    `pairs` = [(label, value_a, value_b)] where the two values are MATHEMATICALLY equal
    distances obtained by two different computation routes.  Reports the absolute spread so
    `eps_d` can be chosen from measured distance arithmetic rather than from unrelated model
    numerics.  Returns (ok, report); `ok` here only means "the spread is finite".
    """
    rows, worst = [], 0.0
    for label, a, b in pairs:
        d = abs(float(a) - float(b))
        worst = max(worst, d)
        rows.append({'label': label, 'a': float(a), 'b': float(b), 'abs_diff': d})
    return True, {'n': len(rows), 'max_abs_diff': worst, 'rows': rows,
                  'note': 'engineering evidence only; eps_d is NOT frozen by this report'}
