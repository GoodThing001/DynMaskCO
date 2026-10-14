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

---------------------------------------------------------------------------
G0-Safety Green Phase (batch 1, user-authorised 2026-10-14) -- SAFETY CONTRACT CHANGE
---------------------------------------------------------------------------
The Red Phase counterexample harness found nine reproduced interface-contract gaps.  This
batch closes the five that are verifiable with synthetic hooks.  Every change below is a NEW
safety semantic for INVALID / UNVERIFIED input; on legal input the decision path is unchanged
(see `test_d3_g0_green_phase.py` block E, which diffs live vs the frozen pre-fix module).

  S1 score-vector contract (A-1a/A-1b/A-1c): `ndim(scores) == 1` and
     `len(scores) == len(legal)` are now enforced BEFORE selection, and a violation fails
     CLOSED (`failure_reason in ('score_shape','score_len')`) -- it is not an exception, and it
     is not a silent prefix/flattened-argmax selection.
  S2 GEO index alignment (A-1d, and the A-1e index too): the state-supplied
     `GEO_SCORE_KEY` vector and `vehicle_target_vid` must both be index-aligned with `legal`,
     else fail closed (`'geo_len'`).
  S3 mandatory certification at the INSTALL boundary (A-2): `stepwise_sample_d3` stays a pure
     sampler, but `run_repair_transaction` refuses to proceed without a real certifier, and
     `InstallLedger.finish` installs ONLY with certification evidence bound to the installed
     plan's identity.  Counting certifications is NOT sufficient: the certified plan hash must
     equal the installed plan hash.
  S4 identity binding + immutability (A-4b): the certified plan hash is recorded at
     certification; the ledger stores a VALUE COPY together with `plan_hash` /
     `certified_plan_hash` / certifier identity, so the ledger cannot later "change its story".
  S5 install idempotency + three-state rollback (A-5b/A-5d): `finish` may be called at most
     once per ledger (one ledger = one transaction); a rollback now distinguishes
     `candidate_discarded` / `committed_state_restored` / `rollback_unverified` instead of
     counting every failure as a completed restoration.

STILL OPEN after this batch (deferred to G0-Evidence, batch 2 -- do NOT read as fixed):
  * A-3b: the adapter does not itself freeze/verify the committed prefix.  The real guard lives
    on `VehicleState` + the mask pool.  Adding a same-named fake field would NOT fix it.
  * A-5c: no durable `P_after_install` receipt.  The ledger receipts are IN-MEMORY only and
    vanish with the process; no persistence surface is added by this batch.
  * Contract/state identity binding for the certifier: only the PLAN identity is bound here.
    Binding the C0 contract / state identity needs the assembly layer.
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

# Tie-break modes.  PRIMARY = legal-order-first, because F's historical rule is
# `np.argmax` = FIRST maximum in the order `enumerate_legal_actions` produced.  Re-sorting both
# arms by canonical key would be formally symmetric but could change F's original selection and
# thereby introduce a new policy version.  Canonical keys are still RECORDED for audit and
# cross-process checks; they must not drive the primary tie.
TIE_BREAK_LEGAL_ORDER = 'legal_order_first'
TIE_BREAK_CANONICAL = 'canonical_key'
TIE_BREAKS = (TIE_BREAK_LEGAL_ORDER, TIE_BREAK_CANONICAL)

FAILURE_REASONS = (None, 'no_legal', 'nan', 'timeout', 'rollback', 'cert_fail',
                   # G0-Safety batch 1: new CLEAN fail-closed reasons.  These are returned as
                   # ordinary failure reasons (never raised), so a caller can never install on
                   # them and an audit can tell them apart from a crash.
                   'score_shape', 'score_len', 'geo_len', 'no_certifier')

# Declarative safety contract (G0-Safety batch 1).  A plan may be installed only when a real
# certifier ran, bound to the installed plan's identity.  The gate is enforced at the INSTALL
# boundary (`run_repair_transaction` + `InstallLedger.finish`), NOT inside `stepwise_sample_d3`:
# the sampler produces a candidate and installs nothing, and several historical tests exercise
# it without a certifier on purpose.
INSTALL_REQUIRES_CERTIFICATION = True

# Reasons the INSTALL gate refuses to install, even with `failure_reason is None`.
REFUSAL_REASONS = ('already_finished', 'certification_missing', 'certified_plan_hash_missing',
                   'certified_plan_identity_mismatch', 'certifier_identity_missing',
                   'decision_deadline_exceeded')


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
# plan identity (G0-Safety batch 1, S4)
# ---------------------------------------------------------------------------
def _plan_identity(plans):
    """Deterministic identity of a full-fleet plan, or None when it cannot be computed.

    Returns None rather than raising: the identity is used by a REFUSING gate, and a gate that
    crashes on a malformed candidate would be strictly worse than one that refuses it.  A None
    identity can never satisfy an identity comparison, so an unhashable plan is refused.
    """
    try:
        from action_contract import plan_hash
        return plan_hash(plans)
    except Exception:                                            # noqa: BLE001 (refuse, don't crash)
        return None


def _copy_plan(plan):
    """Value copy of a plan, so a ledger entry cannot be changed by an external mutation.

    Handles the real `VehiclePlan` mapping and, for robustness, any other mapping by deep copy.
    """
    import copy as _copy
    if not isinstance(plan, dict):
        return _copy.deepcopy(plan)
    out = {}
    for vid, p in plan.items():
        if all(hasattr(p, a) for a in ('vehicle_id', 'anchor_node', 'anchor_time',
                                       'anchor_load', 'suffix')):
            out[vid] = type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                               list(p.suffix))
        else:
            out[vid] = _copy.deepcopy(p)
    return out


def _callable_identity(fn):
    """Stable, human-auditable identity of an injected callable (module + qualname)."""
    if fn is None:
        return None
    mod = getattr(fn, '__module__', None)
    name = getattr(fn, '__qualname__', None) or getattr(fn, '__name__', None)
    return ('%s.%s' % (mod, name)) if mod and name else (name or repr(fn))


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


def select_geo(deltas, keys, eps_rel=EPS_D_PRIMARY_REL, eps_abs=EPS_D_PRIMARY_ABS,
               tie_break=TIE_BREAK_LEGAL_ORDER):
    """D-Geo selection: argmin under the unified-minimum tie rule.

    `tie_break='legal_order_first'` (PRIMARY) picks the FIRST index in `T`, which is exactly
    F's first-maximum semantics applied to `-delta_d` on the same `legal` order.  Only with
    `eps == 0` AND `tie_break='legal_order_first'` are F and D-Geo structurally identical, so
    that the two arms differ in exactly one factor (the score vector); `single_factor_with_F`
    reports this.  `'canonical_key'` is the audit/variant mode and must not be the primary.
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError('unknown tie_break %r' % tie_break)
    d_min, eps, T = unified_minimum_tie_set(deltas, eps_rel, eps_abs)
    if tie_break == TIE_BREAK_LEGAL_ORDER:
        idx = min(T)
    else:
        idx = min(T, key=lambda i: keys[i])
    return idx, {'d_min': d_min, 'eps_used': eps, 'tie_set': list(T),
                 'selected': int(idx), 'n_candidates': len(deltas),
                 'tie_break': tie_break,
                 'eps_is_provisional': eps > 0.0,
                 'single_factor_with_F': (eps == 0.0 and tie_break == TIE_BREAK_LEGAL_ORDER),
                 'canonical_key_of_selected': list(keys[idx]),
                 'rule': 'unified minimum reference; tie decided by %s' % tie_break}


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
                       eps_rel=EPS_D_PRIMARY_REL, eps_abs=EPS_D_PRIMARY_ABS,
                       tie_break=TIE_BREAK_LEGAL_ORDER,
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
    if selector == SELECTOR_GEO and not deterministic:
        # Fail closed instead of silently degrading: the geo rule (unified-minimum tie
        # selection) is only defined for deterministic selection.  Falling through to the
        # historical softmax sampler would ignore the requested selector, and a scorer whose
        # length does not match `legal` used to crash inside rng.choice.
        raise ValueError(
            'selector=%r requires deterministic=True; refusing to silently fall back to the '
            'historical softmax sampler' % SELECTOR_GEO)
    if tie_break not in TIE_BREAKS:
        raise ValueError('unknown tie_break %r' % tie_break)
    if rng is None:
        rng = np.random.default_rng(0)
    budget = budget if budget is not None else Budget()
    mask = assert_ordered_mask(mask_set)
    P = dict(plans)
    remaining = list(mask)
    records, steps = [], []
    diag = {'selector': selector, 'mask_order': list(mask), 'divergence': None,
            'n_steps': 0,
            # G0-Safety batch 1: certification/deadline evidence is ALWAYS present, so the install
            # gate never has to guess whether a field is absent or False.  Defaults describe the
            # truthful initial state: nothing has been certified yet.
            'certified': False, 'certifier': None, 'certified_plan_hash': None,
            'deadline_ok': None}

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
            diag['timed_out'] = True
            return _rollback('timeout')
        legal = legal_fn(P, remaining)
        budget.bump('n_actions_enumerated', len(legal))
        if not legal:
            return _rollback('no_legal')
            diag['n_steps'] = len(steps)
            return _rollback('no_legal')
        state = extract_fn(P, remaining, legal)
        scores = np.asarray(score_fn(state))
        # ---- S1 score-vector contract (A-1a / A-1b / A-1c) --------------------------------
        # The injected scorer is an INTERFACE.  A short vector used to be silently scored as a
        # prefix (wrong action chosen, failure_reason None); an over-long one raised only when
        # the flattened argmax happened to fall outside `legal`; a 2-D one was flattened by
        # `np.argmax` (silently denoting a different candidate) and then crashed inside this
        # function's own evidence recorder.  All three are now CLEAN fail-closed refusals, and
        # the check runs BEFORE `np.isfinite`, so `scores[idx]` below is always a scalar.
        if scores.ndim != 1:
            diag['score_contract'] = {'violation': 'ndim != 1', 'ndim': int(scores.ndim),
                                      'shape': list(scores.shape), 'n_legal': len(legal)}
            return _rollback('score_shape')
        if scores.shape[0] != len(legal):
            diag['score_contract'] = {'violation': 'len(scores) != len(legal)',
                                      'len_scores': int(scores.shape[0]),
                                      'n_legal': len(legal)}
            return _rollback('score_len')
        # counted only once the vector is a valid score for every legal action
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
            # ---- S2 GEO index alignment (A-1d, and the A-1e index as well) -----------------
            # `select_geo` anchors its tie set to `min(deltas)` over the vector it is GIVEN, so a
            # truncated vector silently drops the true minimum.  The keys are indexed by
            # `vehicle_target_vid`, so a short one would mis-key the tie.  Both are validated
            # against `legal` before use: an injected state supplier is an interface too.
            try:
                geo_vec = state[GEO_SCORE_KEY]
            except (KeyError, IndexError, TypeError):
                diag['geo_contract'] = {'violation': 'missing %s' % GEO_SCORE_KEY,
                                       'n_legal': len(legal)}
                return _rollback('geo_len')
            try:
                vids = state['vehicle_target_vid']
            except (KeyError, IndexError, TypeError):
                diag['geo_contract'] = {'violation': 'missing vehicle_target_vid',
                                       'n_legal': len(legal)}
                return _rollback('geo_len')
            if len(geo_vec) != len(legal) or len(vids) != len(legal):
                diag['geo_contract'] = {'violation': 'index vector not aligned with legal',
                                       'len_geo': len(geo_vec), 'len_vids': len(vids),
                                       'n_legal': len(legal)}
                return _rollback('geo_len')
            keys = [canonical_action_key(c, a, vids[i])
                    for i, (c, a, _cand) in enumerate(legal)]
            idx, sel = select_geo(geo_vec, keys, eps_rel, eps_abs, tie_break)
        c, a, _cand = legal[idx]
        vid = state['vehicle_target_vid'][idx]
        # A4 evidence: raw legal order, canonical keys, distance ties and EXACT score ties are
        # all recorded, so "changed because of the tolerance" and "changed because of the
        # score" can be told apart afterwards.
        deltas = [float(t[2].incremental_distance) for t in legal]
        d_min_all = min(deltas) if deltas else None
        records.append({'state_keys': sorted(state.keys()), 'chosen': idx,
                        'selection': sel, 'customer': int(c), 'vehicle_id': int(vid),
                        'incremental_distance': float(_cand.incremental_distance),
                        'legal_order': [int(t[0]) for t in legal],
                        'canonical_keys': [list(k) for k in
                                           [canonical_action_key(t[0], t[1],
                                                                 state['vehicle_target_vid'][i])
                                            for i, t in enumerate(legal)]],
                        'distance_ties': [i for i, dv in enumerate(deltas)
                                          if dv == d_min_all],
                        'score_exact_ties': [i for i, sv in enumerate(scores)
                                             if sv == scores[idx]],
                        'eps_rel': eps_rel, 'eps_abs': eps_abs, 'tie_break': tie_break})
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
        # S3/S4: bind the certification to the EXACT plan object that was certified.  A count of
        # certifications cannot distinguish "certified A, installed B"; the hash can.
        diag['certified'] = True
        diag['certifier'] = _callable_identity(certify_fn)
        diag['certified_plan_hash'] = _plan_identity(P)
        if not ok:
            diag['certify_detail'] = detail
            return _rollback('cert_fail')
        # A4: re-check the decision ceiling AFTER certification.  Checking only at the top of
        # each step would let a run that certifies successfully but blows the budget be
        # reported as a normal completion.
        if wall_clock_limit_s is not None and (time.perf_counter() - t0) > wall_clock_limit_s:
            diag['timed_out'] = True
            diag['timed_out_after_certification'] = True
            return _rollback('timeout')
    else:
        # NOT an error here: this sampler installs nothing, so it stays a pure candidate
        # generator.  The MANDATORY gate is at the install boundary (`run_repair_transaction`
        # refuses without a certifier; `InstallLedger.finish` refuses without bound evidence).
        diag['certified'] = False
        diag['certifier'] = None
        diag['certified_plan_hash'] = None
        diag['certification_note'] = ('no certifier supplied: this is a candidate only and MUST '
                                      'NOT be installed; the install gate refuses it')
    budget.add_wall(time.perf_counter() - t0)
    diag['n_steps'] = len(steps)
    diag['deadline_ok'] = True          # every ceiling check above was passed
    return P, steps, records, None, diag


class InstallLedger(object):
    """SAFETY ADDITION (not historical-behaviour identity).  ONE LEDGER == ONE TRANSACTION.

    The historical decision path has no transaction concept: it returns a plan and the caller
    decides.  This ledger makes the invariant explicit and testable -- a plan may be installed
    ONLY when all of the following hold (G0-Safety batch 1):
      * `failure_reason is None` (the candidate itself succeeded);
      * a real certifier ran (`certified is True`) and its identity is recorded;
      * the certification was bound to a plan identity (`certified_plan_hash` is present);
      * the installed plan's identity EQUALS the certified plan's identity -- counting
        certifications cannot distinguish "certified A, installed B";
      * the decision deadline was not blown (`deadline_ok is not False`).

    Anything else is a REFUSAL (`n_refusals`, event `action='refuse'`), which is different from a
    rollback: nothing was changed, so there is nothing to restore.  Three rollback states are now
    distinguished rather than counted as one:
        candidate_discarded       - the failed candidate is not installed (always True on failure)
        committed_state_restored  - `rollback_fn` actually ran
        rollback_unverified       - no `rollback_fn` was supplied, so restoration is NOT proven
    `n_rollbacks` keeps its historical meaning ("failure paths taken") and is NOT a claim that
    committed state was restored -- read `n_state_restorations` / `n_rollback_unverified`.  A
    transaction refused BEFORE any candidate existed counts in `n_pre_sampling_refusals`
    instead, so it can never be read as a rollback that restored something.

    Receipts are IN-MEMORY ONLY (G0-Evidence batch 2 owns persistence); this class exposes no
    persistence surface and must not be described as producing a durable `P_after_install`.
    """

    def __init__(self, install_fn, rollback_fn=None):
        self.install_fn = install_fn
        self.rollback_fn = rollback_fn
        self.n_installs = 0
        self.n_rollbacks = 0                 # failure paths taken (== candidates discarded)
        self.n_refusals = 0                  # install refused by the safety gate
        self.n_pre_sampling_refusals = 0     # refused BEFORE any candidate existed
        self.n_state_restorations = 0        # `rollback_fn` actually invoked
        self.n_rollback_unverified = 0       # failure with no rollback_fn -> NOT proven restored
        self.installed_plans = []            # value copies, so an external mutation cannot lie
        self.installs = []                   # identity-bound receipts
        self.events = []
        self.finished = False                # S5 idempotency guard

    def finish(self, *, failure_reason, plan, original_plan,
               certified=None, certified_plan_hash=None, certifier=None, deadline_ok=None,
               refused_before_sampling=False):
        """Install iff the repair succeeded AND the certification is bound to this plan."""
        # ---- S5: one ledger is one transaction; a second finish is refused, never repeated -----
        if self.finished:
            self.n_refusals += 1
            self.events.append({'action': 'refuse', 'reason': 'already_finished',
                                'candidate_discarded': True,
                                'committed_state_restored': True,
                                'rollback_unverified': False})
            return False
        if failure_reason is not None:
            self.finished = True
            if refused_before_sampling:
                # No candidate was ever produced and nothing was changed: this is a REFUSAL, not
                # a rollback, and it must not be counted as a restoration of committed state.
                self.n_pre_sampling_refusals += 1
                self.events.append({'action': 'refuse', 'reason': failure_reason,
                                    'refused_before_sampling': True,
                                    'candidate_discarded': True,
                                    'committed_state_restored': True,
                                    'rollback_unverified': False})
                return False
            self.n_rollbacks += 1
            restored = False
            if self.rollback_fn is not None:
                self.rollback_fn(original_plan)
                restored = True
                self.n_state_restorations += 1
            else:
                self.n_rollback_unverified += 1
            self.events.append({'action': 'rollback', 'reason': failure_reason,
                                'candidate_discarded': True,
                                'committed_state_restored': restored,
                                'rollback_unverified': not restored})
            return False
        # ---- S3/S4: the install gate ---------------------------------------------------------
        installed_hash = _plan_identity(plan)
        why = None
        if certified is not True:
            why = 'certification_missing'
        elif not certified_plan_hash:
            why = 'certified_plan_hash_missing'
        elif installed_hash != certified_plan_hash:
            why = 'certified_plan_identity_mismatch'
        elif not certifier:
            why = 'certifier_identity_missing'
        elif deadline_ok is False:
            why = 'decision_deadline_exceeded'
        if why is not None:
            self.finished = True
            self.n_refusals += 1
            self.events.append({'action': 'refuse', 'reason': why,
                                'installed_plan_hash': installed_hash,
                                'certified_plan_hash': certified_plan_hash,
                                'certifier': certifier,
                                'candidate_discarded': True,
                                'committed_state_restored': True,   # nothing was changed
                                'rollback_unverified': False})
            return False
        self.install_fn(plan)
        self.n_installs += 1
        stored = _copy_plan(plan)
        self.installed_plans.append(stored)
        self.installs.append({'plan_hash': installed_hash,
                              'certified_plan_hash': certified_plan_hash,
                              'certifier': certifier,
                              'plan': stored})
        self.finished = True
        self.events.append({'action': 'install', 'reason': None,
                            'plan_hash': installed_hash,
                            'certified_plan_hash': certified_plan_hash,
                            'certifier': certifier,
                            'candidate_discarded': False,
                            'committed_state_restored': False,
                            'rollback_unverified': False})
        return True


def run_repair_transaction(*, ledger, **kwargs):
    """Run `stepwise_sample_d3` under the install transaction.

    Any exception raised by the injected hooks is converted into a failure reason, so the
    caller can never install on an exception path.  Returns the ledger's `finish()` result.

    G0-Safety batch 1 (S3): certification is MANDATORY here.  Without a `certify_fn` the
    transaction refuses BEFORE sampling (`failure_reason='no_certifier'`, zero installs, the
    original plan returned) instead of silently installing an uncertified plan.  This is a
    deliberate safety-contract change: the previous behaviour -- install with
    `n_candidate_plans_certified == 0` -- is the G0-01 counterexample.
    """
    plans = kwargs.get('plans')
    if INSTALL_REQUIRES_CERTIFICATION and kwargs.get('certify_fn') is None:
        return ledger.finish(failure_reason='no_certifier',
                             plan=dict(plans) if plans is not None else {},
                             original_plan=dict(plans) if plans is not None else {},
                             refused_before_sampling=True)
    try:
        plan, _steps, _records, failure_reason, diag = stepwise_sample_d3(**kwargs)
    except BaseException as exc:
        plan = dict(plans) if plans is not None else {}
        failure_reason = 'exception:%s' % type(exc).__name__
        diag = {'exception': repr(exc)[:400], 'rollback': True, 'n_steps': 0,
                'certified': False, 'certifier': None, 'certified_plan_hash': None,
                'deadline_ok': None}
        ledger.events.append({'action': 'exception', 'reason': failure_reason})
    if failure_reason is not None and diag.get('timed_out'):
        diag['counted_as'] = 'timeout (never a normal completion)'
    return ledger.finish(failure_reason=failure_reason, plan=plan,
                         original_plan=dict(plans) if plans is not None else {},
                         certified=diag.get('certified'),
                         certified_plan_hash=diag.get('certified_plan_hash'),
                         certifier=diag.get('certifier'),
                         deadline_ok=diag.get('deadline_ok'))


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
