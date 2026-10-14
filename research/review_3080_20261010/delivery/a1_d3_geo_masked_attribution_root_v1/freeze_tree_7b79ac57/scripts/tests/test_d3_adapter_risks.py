#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-Engineering P0: counterexample suite for the four assembly risks.

Each test is written against the CORRECT (post-fix) behaviour on purpose, so running it
against the pre-fix code yields constructive counterexamples -- the failure output is kept
as evidence, then the fix is applied and the same suite is re-run.

Risks covered:
  R1 `build_pool_B(pool_fn=...)` accepts an injected pool but never calls it (and leaves
     `mask_candidate_pool` unbound).
  R2 `n_solver_calls` actually counts ACTION APPLICATIONS; no solver is invoked at all.
  R3 F and D-Geo do NOT share an identical tie-break when a distance tolerance > 0 is used.
  R4 on `timeout` / `no_legal` / `nan` the adapter returns a PARTIALLY repaired plan, so a
     careless caller could install it instead of rolling back to P_before.

Run: python scripts/tests/test_d3_adapter_risks.py     (exit 0 == all closed)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE.parent))

import d3_geo_adapter as G                                        # noqa: E402
from test_d3_geo_adapter import _cand, _plans, _V                  # noqa: E402

CASES = []


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))


def _mutating_apply(P, action, allowed_vehicle_ids=None):
    """apply that ACTUALLY changes the plan, so a partial repair is observable."""
    out = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
           for vid, p in P.items()}
    if action.slot.kind == 'new_route':
        out[2] = type(out[2])(2, 0, 0.0, 0.0, list(out[2].suffix) + [int(action.customer)])
    else:
        out[1] = type(out[1])(1, out[1].anchor_node, 0.0, 0.0,
                              list(out[1].suffix) + [int(action.customer)])
    return out


def main():
    P0 = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})
    MUT = {1, 2, 3}
    extract = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT)

    # ---------------- R1 injected pool_fn must actually be used ----------------
    calls = {'n': 0}

    def _fake_pool(vehicles, served_mask, visible_ids, protected=frozenset(), plans=None):
        calls['n'] += 1
        return [7, 8]

    veh = [_V(1, 'idle', True, None, (4,))]
    served = [False] * 32
    plans_R = _plans({1: (3, [4, 7, 8])})
    try:
        pool, diag = G.build_pool_B(veh, served, [4, 7, 8], plans_R, {1}, pool_fn=_fake_pool)
        r1_ok = (calls['n'] == 1 and pool == [7, 8])
        r1_detail = {'calls': calls['n'], 'pool': pool}
    except Exception as exc:
        r1_ok, r1_detail = False, '%s: %s' % (type(exc).__name__, exc)
    ck('R1 injected pool_fn is actually invoked and its order preserved', r1_ok, r1_detail)

    # ---------------- R2 budget units must not be conflated ----------------
    seq = [_cand(4, 'anchored', 3, 0, 7.0), _cand(5, 'anchored', -4, 1, 7.0),
           _cand(6, 'new_route', -1, 0, 9.0)]
    budget = G.Budget()
    _p, _s, _r, _fr, _d = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=extract,
        legal_fn=lambda P, rem: [t for t in seq if t[0] in rem],
        apply_fn=_mutating_apply, plans=P0, mask_set=[4, 5, 6], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True, budget=budget)
    snap = budget.snapshot()
    ck('R2 action applications are counted separately from solver calls',
       snap.get('n_action_applications') == 3 and snap.get('n_solver_calls') == 0,
       snap)
    ck('R2 Budget.KEYS exposes both units explicitly',
       'n_action_applications' in G.Budget.KEYS and 'n_solver_calls' in G.Budget.KEYS,
       G.Budget.KEYS)

    # ---------------- R3 eps == 0 makes the tie rule structurally identical to F -------
    # The near-tie must be built so the LARGER-distance candidate carries the SMALLER canonical
    # key; otherwise both arms agree by accident and the counterexample proves nothing.
    #   index 0: customer 4 (smaller key), distance 1.0 + 0.5e-12  (larger)
    #   index 1: customer 5 (larger key),  distance 1.0            (smaller)
    near = [_cand(4, 'anchored', 3, 0, 1.0 + 0.5e-12), _cand(5, 'anchored', 3, 1, 1.0)]
    keys = [G.canonical_action_key(4, near[0][1], 1), G.canonical_action_key(5, near[1][1], 1)]
    assert keys[0] < keys[1], 'fixture must give the larger-distance candidate the smaller key'
    idx_f = G.select_historical_argmax([-x[2].incremental_distance for x in near])
    idx_tol, _sel = G.select_geo([x[2].incremental_distance for x in near], keys,
                                 eps_rel=1e-9, eps_abs=1e-12)
    idx_zero, sel0 = G.select_geo([x[2].incremental_distance for x in near], keys,
                                  eps_rel=0.0, eps_abs=0.0)
    ck('R3 with a positive tolerance D-Geo and F DIVERGE on a near-tie (risk is real)',
       idx_tol != idx_f and idx_f == 1 and idx_tol == 0,
       {'F': idx_f, 'D-Geo(eps>0)': idx_tol, 'keys': keys})
    ck('R3 with eps == 0 D-Geo reproduces F exactly (single-factor comparison)',
       idx_zero == idx_f, {'F': idx_f, 'D-Geo(eps=0)': idx_zero})
    ck('R3 the zero-tolerance configuration is flagged as single-factor with F',
       sel0.get('single_factor_with_F') is True, sel0)
    ck('R3 a positive tolerance is flagged as NOT single-factor',
       _sel.get('single_factor_with_F') is False, _sel)

    # ---------------- R4 every failure path rolls back to the ORIGINAL plan ------------
    orig = {vid: list(p.suffix) for vid, p in P0.items()}
    # step 1 has a legal action, step 2 does not -> partial repair happens internally
    seq_partial = [_cand(4, 'anchored', 3, 0, 5.0)]
    p_out, steps, _r, fr, diag4 = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=extract,
        legal_fn=lambda P, rem: [t for t in seq_partial if t[0] in rem],
        apply_fn=_mutating_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True)
    returned = {vid: list(p.suffix) for vid, p in p_out.items()}
    ck('R4 no_legal after a successful step returns the ORIGINAL plan (fail-closed)',
       fr == 'no_legal' and returned == orig, {'returned': returned, 'orig': orig})
    ck('R4 the partial plan is preserved as EVIDENCE, not returned as the result',
       diag4.get('rollback') is True and diag4.get('partial_plan_at_failure') is not None)
    p_out2, _s2, _r2, fr2, diag42 = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=extract,
        legal_fn=lambda P, rem: [t for t in seq_partial if t[0] in rem],
        apply_fn=_mutating_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True, wall_clock_limit_s=-1.0)
    ck('R4 timeout also returns the ORIGINAL plan',
       fr2 == 'timeout' and {v: list(p.suffix) for v, p in p_out2.items()} == orig)

    passed = sum(1 for _n, ok, _d in CASES if ok)
    for n, ok, detail in CASES:
        print(('  PASS ' if ok else '  FAIL ') + n + ('' if ok else ('   <- ' + str(detail))))
    failed = len(CASES) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
