#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 assembly-layer invariants A1-A4 (user-authorised round).

Written against the CORRECT (post-fix) behaviour on purpose: running it against the pre-fix
code yields the constructive counterexamples for the two newly found risks.

New risks addressed
  N1 `stepwise_sample_d3()` still DEFAULTED to a POSITIVE tolerance while `select_geo()` had
     already defaulted to zero, so "primary eps == 0" did not materialise unless the assembly
     layer passed zeros explicitly.
  N2 even with eps == 0, D-Geo's canonical-key tie-break is NOT the same rule as F's
     first-maximum-in-`legal`-order rule when the actual `legal` order differs from canonical
     key order and there is an exact tie.

Design decision implemented (user ruling): the PRIMARY control keeps F's historical `legal`
order and first-maximum rule; D-Geo at eps == 0 selects the FIRST minimum-distance action in
the SAME `legal` order.  Canonical keys are still RECORDED for audit and cross-process checks
but must not replace F's historical tie rule.

Test categories
  A1 primary parameter passing          (assembly entry -> selection uses eps == 0)
  A2 F historical selection equivalence (incl. reversed canonical order + exact tie)
  A3 install transaction safety         (zero installs on every failure/exception)
  A4 records and accounting             (raw order, keys, ties, workload, wall clock;
                                         timeout must not count as normal completion)

Run: python scripts/tests/test_d3_assembly_invariants.py    (exit 0 == all invariants hold)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE.parent))

import d3_geo_adapter as G                                        # noqa: E402
import mpre_policy as MP                                          # noqa: E402
from test_d3_geo_adapter import _cand, _identity_apply, _plans     # noqa: E402

CASES = []


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))


P0 = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})
MUT = {1, 2, 3}
EXTRACT = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT)

# A missing feature must be a FAIL, never a crash: resolve the post-fix API defensively.
TIE_BREAK_LEGAL_ORDER = getattr(G, 'TIE_BREAK_LEGAL_ORDER', '<missing>')
InstallLedger = getattr(G, 'InstallLedger', None)
run_repair_transaction = getattr(G, 'run_repair_transaction', None)


def _legal_fn(seq):
    return lambda P, rem: [t for t in seq if t[0] in rem]


# canonical key order is the REVERSE of legal order: index 0 (customer 5) carries the LARGER
# canonical key, so a canonical-key tie-break would pick index 1 while F picks index 0.
_TIE_SEQ = [_cand(5, 'anchored', 3, 0, 5.0), _cand(4, 'anchored', 3, 1, 5.0)]


def main():
    # ---------------- A1 primary parameter passing ----------------
    import inspect
    sig = inspect.signature(G.stepwise_sample_d3)
    d_rel = sig.parameters['eps_rel'].default
    d_abs = sig.parameters['eps_abs'].default
    ck('A1 stepwise_sample_d3 defaults to eps_rel == 0 (primary comparison)', d_rel == 0.0, d_rel)
    ck('A1 stepwise_sample_d3 defaults to eps_abs == 0', d_abs == 0.0, d_abs)
    sig2 = inspect.signature(G.select_geo)
    ck('A1 select_geo also defaults to eps == 0',
       sig2.parameters['eps_rel'].default == 0.0 and sig2.parameters['eps_abs'].default == 0.0)
    ck('A1 the primary tie-break is legal-order-first (NOT canonical key)',
       sig.parameters.get('tie_break') is not None
       and sig.parameters['tie_break'].default == TIE_BREAK_LEGAL_ORDER,
       sig.parameters.get('tie_break'))
    ck('A1 provisional positive tolerance still exists for the non-primary variant',
       G.EPS_D_REL_PROVISIONAL > 0 and G.EPS_D_ABS_PROVISIONAL > 0)
    ck('A1 the install transaction API exists', InstallLedger is not None
       and run_repair_transaction is not None,
       {'InstallLedger': InstallLedger, 'run_repair_transaction': run_repair_transaction})
    # N3 (found while fixing this suite): requesting the GEO selector without deterministic
    # selection used to fall through to the historical softmax sampler -- the geo rule was
    # silently ignored, and a score/legal length mismatch crashed instead of failing closed.
    raised = False
    try:
        G.stepwise_sample_d3(score_fn=G.geo_score, extract_fn=EXTRACT,
                             legal_fn=_legal_fn(_TIE_SEQ), apply_fn=_identity_apply,
                             plans=P0, mask_set=[4, 5], mutable_ids=MUT,
                             selector=G.SELECTOR_GEO, deterministic=False)
    except ValueError:
        raised = True
    ck('A1 GEO selector with deterministic=False is REFUSED (no silent degradation to '
       'the historical sampler)', raised)

    # ---------------- A2 F historical selection equivalence ----------------
    # exact tie, and canonical key order is the reverse of legal order
    _p, steps_g, rec_g, fr_g, _d = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=EXTRACT, legal_fn=_legal_fn(_TIE_SEQ),
        apply_fn=_identity_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True)
    idx_f = G.select_historical_argmax([-t[2].incremental_distance for t in _TIE_SEQ])
    ck('A2 on an exact tie with reversed canonical order D-Geo picks the FIRST action in '
       'legal order, exactly like F',
       steps_g[0]['customer'] == _TIE_SEQ[idx_f][0] == 5,
       {'geo': steps_g[0]['customer'], 'F': _TIE_SEQ[idx_f][0],
        'note': 'canonical-key tie-break would have picked customer 4'})
    ck('A2 canonical keys are still RECORDED even though they do not drive the primary tie',
       rec_g[0].get('canonical_keys') is not None and len(rec_g[0]['canonical_keys']) == 2,
       rec_g[0].get('canonical_keys'))
    ck('A2 the selection is flagged single-factor with F',
       rec_g[0]['selection'].get('single_factor_with_F') is True, rec_g[0]['selection'])

    # full F-equivalence on the same ordered legal list, end to end
    seq_f = [_cand(4, 'anchored', 3, 0, 3.0), _cand(5, 'new_route', -1, 0, 3.0)]
    plans_f = _plans({1: (3, [4]), 2: (0, [])})
    ext_f = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids={1, 2})
    real_enum = MP.enumerate_legal_actions
    MP.enumerate_legal_actions = lambda env, inst, P, rem, mut: _legal_fn(seq_f)(P, rem)
    try:
        _pa, s_f, _ra, fr_f = MP.stepwise_sample(
            score_fn=lambda st: [1.0, 1.0], extract_fn=ext_f, env=None, inst_idx=0,
            plans=plans_f, mask_set=[5], mutable_ids={1, 2}, deterministic=True)
    finally:
        MP.enumerate_legal_actions = real_enum
    _pb, s_g, _rb, fr_g2, _d = G.stepwise_sample_d3(
        score_fn=lambda st: [1.0, 1.0], extract_fn=ext_f, legal_fn=_legal_fn(seq_f),
        apply_fn=__import__('action_contract').apply_action, plans=plans_f, mask_set=[5],
        mutable_ids={1, 2}, selector=G.SELECTOR_GEO, deterministic=True)
    proj_g = [{k: v for k, v in s.items() if k != 'incremental_distance'} for s in s_g]
    extra = set()
    for a_, b_ in zip(s_f, s_g):
        extra |= (set(b_) - set(a_))
    ck('A2 end-to-end: on an exact logits tie the new assembly layer matches the ORIGINAL '
       'stepwise_sample step for step (projected onto the historical fields)',
       proj_g == s_f and fr_f == fr_g2, {'orig': s_f, 'new_projected': proj_g})
    ck('A2 the only added step field is the pinned evidence field '
       '(incremental_distance), nothing else',
       extra == {'incremental_distance'}, extra)

    # ---------------- A3 install transaction safety ----------------
    if InstallLedger is None or run_repair_transaction is None:
        for nm in ('A3 no_legal installs ZERO times and rolls back',
                   'A3 an injected exception installs ZERO times and does not escape the transaction',
                   'A3 the success path installs exactly once',
                   'A3 NaN score installs ZERO times', 'A3 timeout installs ZERO times',
                   'A3 the original plan is unchanged on failure'):
            ck(nm, False, 'install transaction API missing in the adapter')
    else:
        installed = []
        ledger = InstallLedger(install_fn=lambda p: installed.append(p),
                               rollback_fn=lambda p: None)

        def _never(P, rem):
            return []

        plan_a, _s, _r, fr_a, _dg = G.stepwise_sample_d3(
            score_fn=G.geo_score, extract_fn=EXTRACT, legal_fn=_never,
            apply_fn=_identity_apply, plans=P0, mask_set=[4], mutable_ids=MUT,
            selector=G.SELECTOR_GEO, deterministic=True)
        ledger.finish(failure_reason=fr_a, plan=plan_a, original_plan=P0)
        ck('A3 no_legal installs ZERO times and rolls back',
           fr_a == 'no_legal' and ledger.n_installs == 0 and ledger.n_rollbacks == 1,
           {'installs': ledger.n_installs, 'rollbacks': ledger.n_rollbacks})

        ledger2 = InstallLedger(install_fn=lambda p: installed.append(p))
        try:
            run_repair_transaction(
                ledger=ledger2, score_fn=G.geo_score, extract_fn=EXTRACT,
                legal_fn=lambda P, rem: (_ for _ in ()).throw(RuntimeError('injected')),
                apply_fn=_identity_apply, plans=P0, mask_set=[4], mutable_ids=MUT,
                selector=G.SELECTOR_GEO, deterministic=True)
            raised = False
        except BaseException:
            raised = True
        ck('A3 an injected exception installs ZERO times and does not escape the transaction',
           ledger2.n_installs == 0 and raised is False,
           {'installs': ledger2.n_installs, 'raised': raised})

        ledger3 = InstallLedger(install_fn=lambda p: installed.append(p))
        ok = run_repair_transaction(
            ledger=ledger3, score_fn=G.geo_score, extract_fn=EXTRACT,
            legal_fn=_legal_fn(seq_f), apply_fn=_identity_apply, plans=plans_f, mask_set=[5],
            mutable_ids={1, 2}, selector=G.SELECTOR_GEO, deterministic=True)
        ck('A3 the success path installs exactly once',
           ok is True and ledger3.n_installs == 1, {'installs': ledger3.n_installs})

        ledger4 = InstallLedger(install_fn=lambda p: installed.append(p))
        run_repair_transaction(
            ledger=ledger4, score_fn=lambda st: [float('nan')] * len(st[G.GEO_SCORE_KEY]),
            extract_fn=EXTRACT, legal_fn=_legal_fn(seq_f), apply_fn=_identity_apply,
            plans=plans_f, mask_set=[5], mutable_ids={1, 2}, selector=G.SELECTOR_GEO, deterministic=True)
        ck('A3 NaN score installs ZERO times', ledger4.n_installs == 0)

        ledger5 = InstallLedger(install_fn=lambda p: installed.append(p))
        run_repair_transaction(
            ledger=ledger5, score_fn=G.geo_score, extract_fn=EXTRACT,
            legal_fn=_legal_fn(seq_f), apply_fn=_identity_apply, plans=plans_f, mask_set=[5],
            mutable_ids={1, 2}, selector=G.SELECTOR_GEO, wall_clock_limit_s=-1.0)
        ck('A3 timeout installs ZERO times', ledger5.n_installs == 0)

        orig_suffix = {v: list(p.suffix) for v, p in P0.items()}
        after_suffix = {v: list(p.suffix) for v, p in plan_a.items()}
        ck('A3 the original plan is unchanged on failure', after_suffix == orig_suffix,
           {'after': after_suffix, 'orig': orig_suffix})

    # ---------------- A4 records, accounting and the post-certification timeout ----------
    budget = G.Budget()
    _p, _s, rec4, fr4, diag4 = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=EXTRACT, legal_fn=_legal_fn(_TIE_SEQ),
        apply_fn=_identity_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True, budget=budget)
    r0 = rec4[0]
    ck('A4 records the raw legal order, canonical keys, distance tie set and final action',
       all(k in r0 for k in ('legal_order', 'canonical_keys', 'distance_ties', 'chosen',
                             'customer')),
       sorted(r0.keys()))
    ck('A4 records exact score ties separately from distance ties',
       'score_exact_ties' in r0 and 'distance_ties' in r0, sorted(r0.keys()))
    snap = budget.snapshot()
    ck('A4 workload and wall clock are reported with separate units',
       snap['n_actions_enumerated'] > 0 and snap['n_action_applications'] == 2
       and snap['wall_seconds'] >= 0.0, snap)
    ck('A4 the timeout flag is exposed so a timeout cannot be recorded as a normal completion',
       'timed_out' in diag4 or fr4 in (None, 'timeout'), {'fr': fr4, 'diag_keys': sorted(diag4)})

    # post-certification timeout: certification succeeds but the decision ceiling is blown
    def _slow_certify(P):
        import time
        time.sleep(0.02)
        return True, {}
    _p, _s, _r, fr6, diag6 = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=EXTRACT, legal_fn=_legal_fn(_TIE_SEQ),
        apply_fn=_identity_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True, certify_fn=_slow_certify,
        wall_clock_limit_s=0.005)
    ck('A4 a decision that only exceeds the ceiling during/after certification is FAILED, '
       'not reported as a normal completion',
       fr6 == 'timeout' and diag6.get('timed_out') is True, {'fr': fr6, 'diag': diag6})

    passed = sum(1 for _n, ok, _d in CASES if ok)
    for n, ok, detail in CASES:
        print(('  PASS ' if ok else '  FAIL ') + n + ('' if ok else ('   <- ' + str(detail))))
    failed = len(CASES) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
