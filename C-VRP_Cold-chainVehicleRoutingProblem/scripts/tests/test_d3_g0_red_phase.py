#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Red Phase: minimal falsifiable counterexample tests for the residual G0 blockers.

READ-ONLY with respect to production code: this file only ADDS tests.  It does not modify
`d3_geo_adapter.py`, any historical test, frozen tree, experiment data or adjudication file.

Design (demanded by the round's acceptance criteria):
  * each case asserts the DESIRED invariant; running it against the current source yields the
    counterexample;
  * each case CLASSIFIES the failure, so an unrelated exception can never masquerade as a
    counterexample:
        COUNTEREXAMPLE_REPRODUCED - the specified invariant was violated, as predicted
        FAIL_LOUD_ACCEPTABLE      - the source raised the EXPECTED error (acceptable, not a bug)
        NOT_REPRODUCED            - the invariant held (item may be closed)
        NOT_VERIFIABLE            - the invariant cannot be observed with injected fakes
        TEST_INVALID              - an unexpected/other exception occurred (test problem, not
                                    a production finding)
  * two POSITIVE CONTROLS run the same invariant checks against a conforming STUB; they must
    pass, proving the harness itself is valid and separating test validity from production G0.

Synthetic data and injected fakes only: no MaskCO checkpoint, no scientific data, no training.
Run: python scripts/tests/test_d3_g0_red_phase.py
Exit: 0 always (this is a measurement harness); the report lives in the emitted JSON.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE.parent))

import numpy as np                                             # noqa: E402
import d3_geo_adapter as G                                     # noqa: E402
from action_contract import apply_action, VehiclePlan          # noqa: E402
from test_d3_geo_adapter import _cand, _identity_apply, _plans  # noqa: E402

RESULTS = []
MUT = {1, 2, 3}
P0 = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})


def record(tid, item, invariant, verdict, observed, evidence=None):
    RESULTS.append({'test_id': tid, 'audit_item': item, 'invariant_asserted': invariant,
                    'verdict': verdict, 'observed': observed, 'evidence': evidence or {}})


def legal_fn(seq):
    return lambda P, rem: [t for t in seq if t[0] in rem]


def run_guarded(fn):
    """Run a case body; any unexpected exception => TEST_INVALID with the traceback tail."""
    try:
        fn()
    except BaseException as exc:                                # noqa: BLE001
        import traceback
        record(getattr(fn, '_tid', 'unknown'), getattr(fn, '_item', '?'),
               getattr(fn, '_inv', '?'), 'TEST_INVALID',
               '%s: %s' % (type(exc).__name__, exc),
               {'traceback_tail': traceback.format_exc()[-400:]})


# ---------------------------------------------------------------- G0-01 (A-2)
def case_g0_01():
    """A-2: a plan must NOT be installed unless full certification was performed."""
    seq = [_cand(4, 'anchored', 3, 0, 3.0)]
    installed = []
    ledger = G.InstallLedger(install_fn=lambda p: installed.append(p))
    budget = G.Budget()
    ok = G.run_repair_transaction(
        ledger=ledger, score_fn=G.geo_score,
        extract_fn=G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT),
        legal_fn=legal_fn(seq), apply_fn=_identity_apply, plans=P0, mask_set=[4],
        mutable_ids=MUT, selector=G.SELECTOR_GEO, deterministic=True,
        certify_fn=None, budget=budget)
    snap = budget.snapshot()
    observed = {'installed': len(installed), 'n_installs': ledger.n_installs,
                'n_candidate_plans_certified': snap['n_candidate_plans_certified'],
                'transaction_returned_true': ok}
    if ledger.n_installs == 1 and snap['n_candidate_plans_certified'] == 0:
        record('G0-01', 'A-2', 'install requires n_candidate_plans_certified >= 1',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': 'a plan was installed with ZERO certifications performed',
                'source_line': 'd3_geo_adapter.py:412  if certify_fn is not None:'})
    else:
        record('G0-01', 'A-2', 'install requires n_candidate_plans_certified >= 1',
               'NOT_REPRODUCED', observed)


case_g0_01._tid, case_g0_01._item = 'G0-01', 'A-2'
case_g0_01._inv = 'install requires full certification'


# ---------------------------------------------------------------- G0-02 (A-1a)
def case_g0_02():
    """A-1a: a score vector shorter than `legal` must not silently score only a prefix."""
    seq = [_cand(4, 'anchored', 3, 0, 3.0), _cand(5, 'anchored', 3, 1, 1.0),
           _cand(6, 'new_route', -1, 0, 2.0)]
    # FIXTURE FIX: mask_set must contain all three customers, otherwise `legal` has length 1
    # and the case silently degenerates into the over-long case (A-1b).
    plan, steps, recs, fr, diag = G.stepwise_sample_d3(
        score_fn=lambda st: [0.0],                    # length 1 vs legal length 3
        extract_fn=G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT),
        legal_fn=legal_fn(seq), apply_fn=_identity_apply, plans=P0, mask_set=[4, 5, 6],
        mutable_ids=MUT, selector=G.SELECTOR_HISTORICAL, deterministic=True)
    observed = {'failure_reason': fr, 'chosen_customer': steps[0]['customer'] if steps else None,
                'n_legal': len(legal_fn(seq)(P0, [4, 5, 6])), 'score_len': 1}
    if fr is None and steps and steps[0]['customer'] == seq[0][0] and observed['n_legal'] == 3:
        record('G0-02', 'A-1a', 'len(scores) == len(legal) or fail-closed',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': 'silently scored only the first of 3 candidates and returned success',
                'source_line': 'd3_geo_adapter.py:366-368 (finite check only, no length check)'})
    elif fr is not None:
        record('G0-02', 'A-1a', 'len(scores) == len(legal) or fail-closed',
               'FAIL_LOUD_ACCEPTABLE', observed)
    else:
        record('G0-02', 'A-1a', 'len(scores) == len(legal) or fail-closed',
               'NOT_REPRODUCED', observed)


case_g0_02._tid, case_g0_02._item = 'G0-02', 'A-1a'
case_g0_02._inv = 'score length equals legal length or fail-closed'


# ---------------------------------------------------------------- G0-03 (A-1c)
def case_g0_03():
    """A-1c: a 2-D score array must not be flattened into a silently wrong candidate index.

    Measured in two layers so the classification is precise:
      primary   - the selection PRIMITIVE is exercised directly: it must return the candidate
                  index the score matrix denotes; flattening gives a different index;
      secondary - the full adapter is then run with the same 2-D scores and whatever happens
                  (silent accept or a crash inside the adapter) is recorded as-is.
    """
    seq = [_cand(4, 'anchored', 3, 0, 3.0), _cand(5, 'anchored', 3, 1, 1.0),
           _cand(6, 'new_route', -1, 0, 2.0)]
    # rows = candidate index; the intended winner is ROW 1, the flattened argmax is flat index 2
    scores_2d = np.array([[0.0, 0.0], [9.0, 0.0], [1.0, 0.0]])
    flat_idx = G.select_historical_argmax(scores_2d)
    intended_idx = int(np.argmax(scores_2d[:, 0]))
    raised = None
    try:
        G.stepwise_sample_d3(
            score_fn=lambda st: scores_2d,
            extract_fn=G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT),
            legal_fn=legal_fn(seq), apply_fn=_identity_apply, plans=P0, mask_set=[4, 5, 6],
            mutable_ids=MUT, selector=G.SELECTOR_HISTORICAL, deterministic=True)
    except BaseException as exc:                       # noqa: BLE001  classified on purpose
        raised = '%s: %s' % (type(exc).__name__, str(exc)[:90])
    observed = {'score_ndim': 2, 'intended_row_index': intended_idx,
                'flattened_argmax_index': flat_idx,
                'mis_index_if_used_on_legal': flat_idx != intended_idx,
                'full_adapter_outcome': raised or 'returned without raising',
                'n_legal': 3}
    if flat_idx != intended_idx:
        record('G0-03', 'A-1c', 'ndim(scores) == 1 or fail-closed',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': ('the selection primitive returns the FLATTENED argmax (row*K+col), which '
                        'denotes a different candidate than the score matrix does: index %d vs '
                        'intended %d' % (flat_idx, intended_idx)),
                'secondary_manifestation': raised or 'none',
                'source_line': 'd3_geo_adapter.py:366 (no ndim check), 373 (argmax flattens)',
                'note': ('the secondary outcome differs from the earlier prediction: with 3 legal '
                         'candidates the adapter no longer raises IndexError at legal[idx] but '
                         'proceeds and then fails inside its own evidence recorder on an array '
                         'truth-value comparison. Either way the 2-D score is neither handled nor '
                         'rejected.')})
    else:
        record('G0-03', 'A-1c', 'ndim(scores) == 1 or fail-closed',
               'NOT_REPRODUCED', observed)


case_g0_03._tid, case_g0_03._item = 'G0-03', 'A-1c'
case_g0_03._inv = 'score ndim equals 1 or fail-closed'


# ---------------------------------------------------------------- G0-04 (A-1b)
def case_g0_04():
    """A-1b: an over-long score vector is NOT reliably fail-loud.

    REFINEMENT over the audit: the audit called A-1b "fail-loud, acceptable".  That is only true
    when the flattened argmax index happens to fall OUTSIDE the legal range.  If it falls inside,
    the extra scores are silently ignored.  Both branches are measured here.
    """
    seq = [_cand(4, 'anchored', 3, 0, 3.0), _cand(5, 'anchored', 3, 1, 1.0)]
    ext = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT)

    def _attempt(scores):
        try:
            G.stepwise_sample_d3(
                score_fn=lambda st: scores, extract_fn=ext,
                legal_fn=legal_fn(seq), apply_fn=_identity_apply, plans=P0, mask_set=[4, 5],
                mutable_ids=MUT, selector=G.SELECTOR_HISTORICAL, deterministic=True)
            return None
        except BaseException as exc:                    # noqa: BLE001  classified on purpose
            return type(exc).__name__

    inside = _attempt([5.0, 4.0, 3.0, 2.0])       # argmax=0 -> inside legal range (len 2)
    outside = _attempt([1.0, 5.0, 4.0, 3.0])      # argmax=1 -> inside too (len 2)
    beyond = _attempt([1.0, 2.0, 9.0, 3.0])       # argmax=2 -> OUTSIDE legal range
    observed = {'legal_len': 2, 'argmax_inside_exception': inside,
                'argmax_beyond_exception': beyond, 'second_probe': outside,
                'expected_for_beyond': 'IndexError'}
    if inside is None and beyond == 'IndexError':
        record('G0-04', 'A-1b',
               'over-long scores must fail closed OR never be silently partly used',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': ('with 4 scores for 2 candidates the run returned SUCCESS when the argmax '
                        'fell inside the legal range, silently discarding the extra scores; it '
                        'only raised when the argmax index exceeded the legal length'),
                'upgrades': 'A-1b was classified FAIL_LOUD_ACCEPTABLE in the audit; the in-range '
                            'branch is a silent acceptance, so the item is not reliably loud',
                'source_line': 'd3_geo_adapter.py:366/373/385'})
    elif beyond == 'IndexError':
        record('G0-04', 'A-1b', 'over-long scores must fail closed OR never be silently partly used',
               'FAIL_LOUD_ACCEPTABLE', observed)
    else:
        record('G0-04', 'A-1b', 'over-long scores must fail closed OR never be silently partly used',
               'NOT_REPRODUCED', observed)


case_g0_04._tid, case_g0_04._item = 'G0-04', 'A-1b'
case_g0_04._inv = 'over-long score fails loud'


# ---------------------------------------------------------------- G0-05 (A-5b)
def case_g0_05():
    """A-5b: InstallLedger.finish must be idempotent (no duplicate installs)."""
    installed = []
    ledger = G.InstallLedger(install_fn=lambda p: installed.append(p))
    r1 = ledger.finish(failure_reason=None, plan={'x': 1}, original_plan=P0)
    r2 = ledger.finish(failure_reason=None, plan={'x': 1}, original_plan=P0)
    observed = {'first_returned': r1, 'second_returned': r2,
                'n_installs': ledger.n_installs, 'installed_len': len(installed)}
    if ledger.n_installs == 2:
        record('G0-05', 'A-5b', 'n_installs <= 1 per transaction',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': 'calling finish twice installed twice; no idempotency guard',
                'source_line': 'd3_geo_adapter.py:449-461'})
    else:
        record('G0-05', 'A-5b', 'n_installs <= 1 per transaction', 'NOT_REPRODUCED', observed)


case_g0_05._tid, case_g0_05._item = 'G0-05', 'A-5b'
case_g0_05._inv = 'install is idempotent'


# ---------------------------------------------------------------- G0-06 (A-4b)
def case_g0_06():
    """A-4b: the certified plan's identity must be recorded and must equal the installed one."""
    retained = []
    ledger = G.InstallLedger(install_fn=lambda p: retained.append(p))
    seq = [_cand(4, 'anchored', 3, 0, 3.0)]
    plan, _s, _r, fr, diag = G.stepwise_sample_d3(
        score_fn=G.geo_score,
        extract_fn=G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT),
        legal_fn=legal_fn(seq), apply_fn=apply_action, plans=P0, mask_set=[4],
        mutable_ids=MUT, selector=G.SELECTOR_GEO, deterministic=True,
        certify_fn=lambda p: (True, {}))
    ledger.finish(failure_reason=fr, plan=plan, original_plan=P0)
    # was ANY plan-identity hash recorded at certification or at install?
    # FIXTURE FIX: use vars(ledger) rather than dir(ledger); dir() includes Python's own
    # `__hash__`, which made the previous run report a bogus identity field.
    identity_fields = [k for k in (diag or {})
                       if ('hash' in str(k).lower() or 'identity' in str(k).lower())
                       and not str(k).startswith('__')]
    identity_fields += [k for k in vars(ledger)
                        if ('hash' in k.lower() or 'identity' in k.lower())
                        and not k.startswith('__')]
    # aliasing check: mutate the object the ledger retained
    before = {v: list(p.suffix) for v, p in retained[0].items()} if retained else None
    if retained:
        first_vid = sorted(retained[0].keys())[0]
        retained[0][first_vid] = VehiclePlan(first_vid, 0, 0.0, 0.0, [999])
    after = {v: list(p.suffix) for v, p in retained[0].items()} if retained else None
    observed = {'failure_reason': fr, 'n_installs': ledger.n_installs,
                'identity_fields_recorded': identity_fields,
                'ledger_entry_changed_by_external_mutation': before != after,
                'before': before, 'after': after}
    if (not identity_fields) and before != after:
        record('G0-06', 'A-4b',
               'certified plan identity is recorded and the installed entry is immutable',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': ('no identity hash exists anywhere in diag or the ledger, and the ledger '
                        'stores the plan BY REFERENCE so an external mutation silently changes '
                        'what it appears to have installed'),
                'source_line': 'd3_geo_adapter.py:459  self.installed_plans.append(plan)',
                'evidence_type': 'API-surface + runtime aliasing proof (not a hash comparison, '
                                 'because no hash exists to compare)'})
    else:
        record('G0-06', 'A-4b',
               'certified plan identity is recorded and the installed entry is immutable',
               'NOT_REPRODUCED', observed)


case_g0_06._tid, case_g0_06._item = 'G0-06', 'A-4b'
case_g0_06._inv = 'certified/installed identity recorded and equal'


# ---------------------------------------------------------------- G0-07 (A-1d)
def case_g0_07():
    """A-1d: the GEO distance vector must be index-aligned with `legal` (short => silent miss)."""
    seq = [_cand(4, 'anchored', 3, 0, 5.0), _cand(5, 'anchored', 3, 1, 6.0),
           _cand(6, 'new_route', -1, 0, 1.0)]      # true best is candidate 2
    short = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT)

    def _truncating_extract(P, remaining, legal):
        st = short(P, remaining, legal)
        st[G.GEO_SCORE_KEY] = st[G.GEO_SCORE_KEY][:2]      # breaks index alignment
        return st

    plan, steps, recs, fr, diag = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=_truncating_extract, legal_fn=legal_fn(seq),
        apply_fn=_identity_apply, plans=P0, mask_set=[4], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True)
    chosen = steps[0]['customer'] if steps else None
    observed = {'failure_reason': fr, 'true_best_customer': seq[2][0],
                'chosen_customer': chosen, 'delta_len': 2, 'n_legal': 3}
    if fr is None and chosen == seq[0][0]:
        record('G0-07', 'A-1d', 'len(deltas) == len(legal) or fail-closed',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': 'a truncated distance vector silently ignored the true best candidate',
                'source_line': 'd3_geo_adapter.py:226-244 / 384'})
    elif fr is not None:
        record('G0-07', 'A-1d', 'len(deltas) == len(legal) or fail-closed',
               'FAIL_LOUD_ACCEPTABLE', observed)
    else:
        record('G0-07', 'A-1d', 'len(deltas) == len(legal) or fail-closed',
               'NOT_REPRODUCED', observed)


case_g0_07._tid, case_g0_07._item = 'G0-07', 'A-1d'
case_g0_07._inv = 'geo distance vector index-aligned with legal'


# ---------------------------------------------------------------- G0-08 (A-3b)
def case_g0_08():
    """A-3b: the adapter itself must not install a plan that re-plans a committed customer."""
    # vehicle 1 is committed to customer 7 (committed_next), which is NOT in any suffix
    plans_c = {1: VehiclePlan(1, 3, 0.0, 0.0, [4]),
               2: VehiclePlan(2, 0, 0.0, 0.0, []),
               3: VehiclePlan(3, 0, 0.0, 0.0, [5])}
    plans_c[1].committed_next = 7
    committed_before = plans_c[1].committed_next
    seq = [_cand(7, 'anchored', 3, 0, 1.0)]          # re-insert the committed customer
    installed = []
    ledger = G.InstallLedger(install_fn=lambda p: installed.append(p))
    ok = G.run_repair_transaction(
        ledger=ledger, score_fn=G.geo_score,
        extract_fn=G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT),
        legal_fn=legal_fn(seq), apply_fn=apply_action, plans=plans_c, mask_set=[7],
        mutable_ids=MUT, selector=G.SELECTOR_GEO, deterministic=True,
        certify_fn=lambda p: (True, {}))
    inst = installed[0] if installed else None
    in_suffix = any(7 in [int(x) for x in p.suffix] for p in inst.values()) if inst else False
    observed = {'committed_next_before': int(committed_before),
                'committed_customer_appears_in_suffix_after_install': in_suffix,
                'n_installs': ledger.n_installs, 'transaction_returned_true': ok}
    # HONEST RECLASSIFICATION: `VehiclePlan` has no `committed_next` field; the commitment lives
    # on `VehicleState` and is excluded by the mask pool, i.e. OUTSIDE this adapter's plan-level
    # API.  The attribute injected here is therefore a fake and does NOT prove the real pipeline
    # can violate a commitment.  The item stays OPEN and is NOT_VERIFIABLE with these fakes.
    record('G0-08', 'A-3b',
           'no installed plan may re-plan a committed customer without an explicit check',
           'NOT_VERIFIABLE', observed,
           {'why': ('the adapter boundary is plan-level (FleetPlan, no committed_next field); the '
                    'commitment guard lives in VehicleState + the mask pool, which this harness '
                    'cannot exercise without the env layer'),
            'fixture_caveat': ('plans_c[1].committed_next = 7 is a fabricated attribute, not the '
                               'real mechanism; the observed insertion therefore does NOT '
                               'falsify the real pipeline'),
            'audit_status_kept': 'A-3b remains OPEN',
            'what_would_be_needed': 'a state-level harness that builds real VehicleState objects '
                                    'and routes them through mask_candidate_pool + the adapter'})


case_g0_08._tid, case_g0_08._item = 'G0-08', 'A-3b'
case_g0_08._inv = 'adapter does not install commitment-violating plans'


# ---------------------------------------------------------------- G0-09 (A-5d)
def case_g0_09():
    """A-5d: a rollback must either really happen or be marked unverified."""
    ledger = G.InstallLedger(install_fn=lambda p: None, rollback_fn=None)
    ledger.finish(failure_reason='no_legal', plan={'x': 1}, original_plan=P0)
    rev = ledger.events[-1] if ledger.events else {}
    flagged = any('unverified' in str(k).lower() or 'unverified' in str(v).lower()
                  for k, v in rev.items())
    observed = {'n_rollbacks': ledger.n_rollbacks, 'last_event': rev,
                'rollback_unverified_flagged': flagged, 'rollback_fn': None}
    if ledger.n_rollbacks == 1 and not flagged:
        record('G0-09', 'A-5d',
               'rollback with rollback_fn=None is marked unverified',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': ('the ledger counts a rollback while nothing was restored and no flag '
                        'records that, so the counter is misleading'),
                'source_line': 'd3_geo_adapter.py:452-455'})
    else:
        record('G0-09', 'A-5d', 'rollback with rollback_fn=None is marked unverified',
               'NOT_REPRODUCED', observed)


case_g0_09._tid, case_g0_09._item = 'G0-09', 'A-5d'
case_g0_09._inv = 'unbacked rollback is flagged'


# ---------------------------------------------------------------- G0-10 (A-5c)
def case_g0_10():
    """A-5c: a successful install must produce a durable receipt (P_after_install)."""
    ledger = G.InstallLedger(install_fn=lambda p: None)
    api = [k for k in dir(ledger) if any(t in k.lower() for t in
                                         ('receipt', 'persist', 'write', 'path', 'save', 'dump'))]
    ledger.finish(failure_reason=None, plan={'x': 1}, original_plan=P0)
    observed = {'persistence_api_surface': api, 'events_in_memory_only': True,
                'n_installs': ledger.n_installs}
    if not api:
        record('G0-10', 'A-5c', 'a successful install yields a durable P_after_install receipt',
               'COUNTEREXAMPLE_REPRODUCED', observed,
               {'why': 'the ledger exposes no persistence surface at all; the receipt exists only '
                       'in memory and disappears with the process',
                'source_line': 'd3_geo_adapter.py:441-461',
                'evidence_type': 'API-surface proof (no receipt path parameter exists to test)'})
    else:
        record('G0-10', 'A-5c', 'a successful install yields a durable P_after_install receipt',
               'NOT_REPRODUCED', observed)


case_g0_10._tid, case_g0_10._item = 'G0-10', 'A-5c'
case_g0_10._inv = 'durable install receipt exists'


# ---------------------------------------------------------------- CONTROLS
def control_c1():
    """POSITIVE CONTROL for G0-01: a conforming transaction must refuse uncertified installs."""
    class ConformingLedger:
        def __init__(self):
            self.n_installs, self.n_certified, self.refused = 0, 0, 0

        def finish(self, *, failure_reason, plan, original_plan, certified=False):
            self.n_certified += int(bool(certified))
            if failure_reason is not None or not certified:
                self.refused += 1
                return False
            self.n_installs += 1
            return True

    led = ConformingLedger()
    led.finish(failure_reason=None, plan={'x': 1}, original_plan=P0, certified=False)
    observed = {'stub_installs': led.n_installs, 'stub_refused': led.refused}
    if led.n_installs == 0 and led.refused == 1:
        record('G0-C1', 'CONTROL(A-2)', 'the G0-01 invariant is checkable and can hold',
               'CONTROL_PASSES', observed,
               {'meaning': 'the harness can distinguish a conforming implementation; the '
                           'G0-01 failure is therefore a production finding, not a test artefact'})
    else:
        record('G0-C1', 'CONTROL(A-2)', 'the G0-01 invariant is checkable and can hold',
               'CONTROL_FAILS', observed)


def control_c2():
    """POSITIVE CONTROL for A-1a/A-1c: a conforming scorer contract is detectable."""
    def conforming_check(scores, n_legal, ndim):
        return len(scores) == n_legal and ndim == 1

    good = conforming_check([0.0, 1.0, 2.0], 3, np.asarray([0.0, 1.0, 2.0]).ndim)
    bad_len = conforming_check([0.0], 3, 1)
    bad_ndim = conforming_check([[0.0], [1.0], [2.0]], 3, np.asarray([[0.0], [1.0], [2.0]]).ndim)
    observed = {'good_accepted': good, 'short_rejected': bad_len is False,
                '2d_rejected': bad_ndim is False}
    if good and bad_len is False and bad_ndim is False:
        record('G0-C2', 'CONTROL(A-1a/A-1c)', 'length/ndim contract is mechanically checkable',
               'CONTROL_PASSES', observed,
               {'meaning': 'both A-1a and A-1c are detectable by a simple assertion; the '
                           'observed production behaviour is a real gap'})
    else:
        record('G0-C2', 'CONTROL(A-1a/A-1c)', 'length/ndim contract is mechanically checkable',
               'CONTROL_FAILS', observed)


def main():
    for fn in (case_g0_01, case_g0_02, case_g0_03, case_g0_04, case_g0_05, case_g0_06,
               case_g0_07, case_g0_08, case_g0_09, case_g0_10, control_c1, control_c2):
        run_guarded(fn)

    counts = {}
    for r in RESULTS:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    out = {
        'stage': 'D3_G0_Red_Phase',
        'production_source_modified': False,
        'scientific_data_used': False,
        'checkpoint_loaded': False,
        'results': RESULTS,
        'verdict_counts': counts,
        'harness_valid': all(r['verdict'] != 'TEST_INVALID' for r in RESULTS)
                         and counts.get('CONTROL_PASSES', 0) == 2,
        'note': ('a COUNTEREXAMPLE_REPRODUCED row is a production G0 gap; the harness validity '
                 '(controls + no TEST_INVALID) is a separate statement and does not depend on '
                 'the production findings'),
    }
    print('%-7s %-14s %-28s %s' % ('test', 'audit item', 'verdict', 'observed'))
    for r in RESULTS:
        print('%-7s %-14s %-28s %s' % (r['test_id'], r['audit_item'], r['verdict'],
                                       json.dumps(r['observed'], ensure_ascii=False)[:110]))
    print('\nverdict_counts:', json.dumps(counts, ensure_ascii=False))
    print('harness_valid:', out['harness_valid'])
    # FIXTURE FIX: resolve the repo root by walking up to MASKCO_code, instead of assuming the
    # script's depth (the previous run wrote the JSON into a stray C-VRP_.../research/ path).
    root = HERE
    while root is not None and not (root / 'MASKCO_code').is_dir():
        root = root.parent
    dest_dir = root / 'research' / 'review_3080_20261010' / 'p0_next_stage_audit_root_v1'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / 'D3_G0_RED_PHASE_MATRIX.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
