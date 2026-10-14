#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-Engineering: T-G1..T-G9 for the D-Geo adapter.

Pure offline: no env, no data, no model, no experiment.  `legal` lists are hand-built so the
tests are hermetic; the shared enumerator is verified via a spy/monkeypatch instead of a real
environment.

Fleet fixture (physically consistent, so `_match_anchored` / `_match_new_route` resolve):
    vid 1: anchor_node = 3, suffix = [4]   -> anchored slot anchor = 3
    vid 2: anchor_node = 0, suffix = []    -> the empty idle depot vehicle (NEW_ROUTE)
    vid 3: anchor_node = 0, suffix = [5]   -> anchored slot anchor = -(3+1) = -4

Extras beyond the nine:
  * X1 old/new behaviour DIFF: the adapter's historical mode reproduces
    `mpre_policy.stepwise_sample` exactly on the SAME legal list (isolated by monkeypatching
    the shared enumerator), so the compared F policy is not silently changed.
  * X2 F behaviour REGRESSION: F's rule is still first-maximum `np.argmax` and no distance
    tolerance leaks into it.

Run: python scripts/tests/test_d3_geo_adapter.py      (exit 0 == all pass)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE.parent))

import numpy as np                                        # noqa: E402
import d3_geo_adapter as G                                 # noqa: E402
import mpre_policy as MP                                   # noqa: E402
from action_contract import (ActionSlot, Candidate, FleetAction, VehiclePlan,  # noqa: E402
                             apply_action)

CASES = []


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))


def _cand(customer, kind, anchor, position, dist, feasible=True):
    pred = anchor if position == 0 else 0
    act = FleetAction(customer=int(customer), slot=ActionSlot(kind, int(anchor)),
                      position=int(position), predecessor=int(pred), successor=int(pred))
    return (int(customer), act,
            Candidate(action=act, feasible=feasible, reason=None,
                      incremental_distance=float(dist), tw_slack=0.0, cap_slack=0.0,
                      return_slack=0.0, plan_hash='h'))


def _plans(spec):
    return {vid: VehiclePlan(vehicle_id=vid, anchor_node=a, anchor_time=0.0, anchor_load=0.0,
                             suffix=list(s)) for vid, (a, s) in spec.items()}


def _identity_apply(P, action, allowed_vehicle_ids=None):
    """Test stand-in for `action_contract.apply_action`.

    NOTE the kwarg name: `apply_action(plans, action, src_vid=None, allowed_vehicle_ids=None)`
    -- the third POSITIONAL parameter is `src_vid`, so passing mutable_ids positionally
    silently corrupts the call.  This stand-in mirrors the real signature on purpose.
    """
    return dict(P)


class _V:
    def __init__(self, vid, status='idle', needs_replan=True, committed_next=None, suffix=()):
        self.vehicle_id, self.status = vid, status
        self.needs_replan, self.committed_next = needs_replan, committed_next
        self.mutable_suffix = list(suffix)


def _legal_fn(seq):
    return lambda P, remaining: [t for t in seq if t[0] in remaining]


MUT = {1, 2, 3}


def main():
    P0 = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})
    extract = G.make_geo_extract_fn(lambda P, rem, legal: {'n_legal': len(legal)},
                                    allowed_vehicle_ids=MUT)

    # ---------------- T-G1 index one-to-one ----------------
    legal = [_cand(4, 'anchored', 3, 0, 11.0), _cand(5, 'anchored', -4, 1, 7.5),
             _cand(6, 'new_route', -1, 0, 9.25)]
    st = extract(P0, [4, 5, 6], legal)
    ck('T-G1 len(geo vector) == len(legal) == len(vehicle_target_vid)',
       len(st[G.GEO_SCORE_KEY]) == len(legal) == len(st['vehicle_target_vid']))
    ck('T-G1 item i is exactly legal[i].incremental_distance (bit-equal, no recompute)',
       all(st[G.GEO_SCORE_KEY][i] == legal[i][2].incremental_distance for i in range(3)))
    ck('T-G1 geo_score == -delta_d in legal order',
       G.geo_score(st) == [-11.0, -7.5, -9.25])
    ck('T-G1 vehicle targets resolved deterministically (anchored vid=1, depot-route vid=3, '
       'new_route vid=2)', st['vehicle_target_vid'] == [1, 3, 2], st['vehicle_target_vid'])

    # ---------------- T-G2 score purity ----------------
    legal2 = [_cand(4, 'anchored', 3, 0, 11.0), _cand(5, 'anchored', -4, 1, 7.5)]
    st2 = extract(P0, [4, 5], legal2)
    before = G.geo_score(st2)
    for _c, _a, cand in legal2:                 # perturb every non-distance field
        cand.projected_energy_kwh = 999.0
        cand.projected_quality_loss = 123.0
        cand.coldchain_reject_reason = 'x'
        cand.temperature_compatible = False
        cand.plan_hash = 'zzz'
    ck('T-G2 cold-chain / shadow / reject fields do NOT affect geo_score',
       G.geo_score(st2) == before)
    raised = False
    try:
        G.geo_score({'something_else': [1.0]})
    except KeyError:
        raised = True
    ck('T-G2 missing distance key raises (no silent fallback scorer)', raised)

    # ---------------- T-G3 B mask built BEFORE sampling; empty pool recorded --------------
    veh = [_V(1, 'idle', True, None, (3, 4)), _V(2, 'idle', True, None, ())]
    served = [False] * 32
    plans_B = _plans({1: (0, [3, 4]), 2: (0, [])})
    pool, diag = G.build_pool_B(veh, served, [3, 4, 9], plans_B, {1, 2})
    ck('T-G3 pool_B excludes the deferred customer 9 (in no mutable plan suffix)',
       9 not in pool and 9 in diag['excluded_as_deferred_or_outside_mutable_plans'])
    ck('T-G3 pool_B keeps the pool order', pool == [3, 4], pool)
    empty_pool, empty_diag = G.build_pool_B(veh, served, [9], plans_B, {1, 2})
    ck('T-G3 empty pool returned as [] and flagged, NOT dropped',
       empty_pool == [] and empty_diag['empty'] is True)

    # ---------------- T-G4 order discipline + determinism ----------------
    try:
        G.assert_ordered_mask({4, 5, 6})
        raised = False
    except TypeError:
        raised = True
    ck('T-G4 bare set is REJECTED (order would be unspecified)', raised)
    ck('T-G4 ordered input preserved verbatim', G.assert_ordered_mask([6, 4, 5]) == [6, 4, 5])
    seq = [_cand(4, 'anchored', 3, 0, 7.0), _cand(5, 'anchored', -4, 1, 7.0),
           _cand(6, 'new_route', -1, 0, 9.0)]
    runs = []
    for _ in range(20):
        _p, steps, _r, fr, _d = G.stepwise_sample_d3(
            score_fn=G.geo_score, extract_fn=extract, legal_fn=_legal_fn(seq),
            apply_fn=_identity_apply, plans=P0, mask_set=[4, 5, 6], mutable_ids=MUT,
            selector=G.SELECTOR_GEO, deterministic=True)
        runs.append(json.dumps(steps, sort_keys=True))
    ck('T-G4 20 in-process runs give byte-identical steps', len(set(runs)) == 1)
    code = (
        "import sys, json;"
        "sys.path.insert(0, %r); sys.path.insert(0, %r);"
        "import d3_geo_adapter as G, test_d3_geo_adapter as T;"
        "P0 = T._plans({1:(3,[4]),2:(0,[]),3:(0,[5])});"
        "seq = [T._cand(4,'anchored',3,0,7.0), T._cand(5,'anchored',-4,1,7.0),"
        "       T._cand(6,'new_route',-1,0,9.0)];"
        "ext = G.make_geo_extract_fn(lambda P,r,l: {}, allowed_vehicle_ids=T.MUT);"
        "p,steps,rec,fr,d = G.stepwise_sample_d3(score_fn=G.geo_score, extract_fn=ext,"
        "  legal_fn=T._legal_fn(seq), apply_fn=T._identity_apply, plans=P0,"
        "  mask_set=[4,5,6], mutable_ids=T.MUT, selector=G.SELECTOR_GEO,"
        "  deterministic=True);"
        "print(json.dumps(steps, sort_keys=True))" % (str(SIM), str(HERE)))
    outs = set()
    for _ in range(2):
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        outs.add((r.stdout or '').strip())
    ck('T-G4 two separate processes give identical steps',
       len(outs) == 1 and all(o for o in outs), outs)

    # ---------------- T-G5 unified-minimum tie rule (no pairwise equality) ---------------
    # Non-transitivity demo: with eps = 1e-12 (abs, rel 0), |d0-d1| and |d1-d2| are within eps
    # but |d0-d2| is NOT.  Pairwise approximate equality would chain 0~1~2; anchoring every
    # comparison to d_min must exclude 2.
    _dmin, _eps, T = G.unified_minimum_tie_set([1.0, 1.0 + 0.9e-12, 1.0 + 1.8e-12],
                                               eps_rel=0.0, eps_abs=1e-12)
    ck('T-G5 tie set is anchored to d_min, so the non-transitive pair chain 0~1~2 is NOT used',
       T == [0, 1] and 2 not in T, {'d_min': _dmin, 'eps': _eps, 'T': T})
    keys = [G.canonical_action_key(*t) for t in
            ((5, _cand(5, 'anchored', 3, 1, 7.0)[1], 1),
             (4, _cand(4, 'anchored', 3, 0, 7.0)[1], 1),
             (6, _cand(6, 'new_route', -1, 0, 7.0)[1], 2))]
    idx_a, sel_a = G.select_geo([7.0, 7.0, 7.0], keys)
    perm = [2, 0, 1]
    idx_b, _ = G.select_geo([7.0, 7.0, 7.0], [keys[i] for i in perm])
    ck('T-G5 all-equal ties resolve to the smallest canonical key (customer 4)',
       keys[idx_a] == min(keys) and keys[idx_a][0] == 4)
    ck('T-G5 tie outcome is invariant to permuting the candidate order',
       keys[perm[idx_b]] == keys[idx_a])
    ck('T-G5 tie rule recorded and marked provisional',
       sel_a['rule'].startswith('unified minimum reference') and sel_a['eps_is_provisional'])
    ok_rep, rep = G.distance_stability_report([('equal_route_delta', 1.0 + 1e-12, 1.0)])
    ck('T-G5 eps evidence report produced, explicitly NOT frozen',
       ok_rep and 'NOT frozen' in rep['note'])

    # ---------------- T-G6 full certification required ----------------
    calls = {'n': 0}

    def _cert_fail(P):
        calls['n'] += 1
        return False, {'extra': [9]}

    _p, _s, _r, fr, diag6 = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=extract, legal_fn=_legal_fn(seq),
        apply_fn=_identity_apply, plans=P0, mask_set=[4], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True, certify_fn=_cert_fail)
    ck('T-G6 failing full certification gives failure_reason=cert_fail (not adoption)',
       fr == 'cert_fail' and calls['n'] == 1)
    ck('T-G6 certification detail retained', diag6.get('certify_detail') == {'extra': [9]})

    # ---------------- T-G7 every failure mode is booked ----------------
    seen = {}
    _p, _s, _r, fr, _d = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=extract, legal_fn=lambda P, rem: [],
        apply_fn=_identity_apply, plans=P0, mask_set=[4], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True)
    seen['no_legal'] = fr
    _p, _s, _r, fr, _d = G.stepwise_sample_d3(
        score_fn=lambda st: [float('nan')], extract_fn=extract, legal_fn=_legal_fn(seq),
        apply_fn=_identity_apply, plans=P0, mask_set=[4], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True)
    seen['nan'] = fr
    _p, _s, _r, fr, _d = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=extract, legal_fn=_legal_fn(seq),
        apply_fn=_identity_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True, wall_clock_limit_s=-1.0)
    seen['timeout'] = fr
    seen['cert_fail'] = 'cert_fail'
    ck('T-G7 no_legal / nan / timeout / cert_fail all surface as failure_reason',
       seen == {'no_legal': 'no_legal', 'nan': 'nan', 'timeout': 'timeout',
                'cert_fail': 'cert_fail'}, seen)
    budget = G.Budget()
    _p, _s, _r, _fr, _d = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=extract, legal_fn=_legal_fn(seq),
        apply_fn=_identity_apply, plans=P0, mask_set=[4, 5, 6], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True, budget=budget)
    snap = budget.snapshot()
    ck('T-G7 budget reports the five units SEPARATELY (they are not equal-cost)',
       sorted(snap.keys()) == sorted(G.Budget.KEYS) and snap['n_actions_scored'] > 0
       and snap['n_solver_calls'] == 3, snap)

    # ---------------- T-G8 divergence is not identity mismatch ----------------
    seq_div = [_cand(4, 'anchored', 3, 0, 5.0), _cand(5, 'anchored', -4, 1, 5.0)]
    _p, _s, rec_g, fr_g, _d = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=extract, legal_fn=_legal_fn(seq_div),
        apply_fn=_identity_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True)
    def _last_is_best(st):
        # length-aware: the shared sampler does NOT check len(scores) == len(legal), so a
        # fixed-length scorer would IndexError on the shorter second step.  Making the last
        # candidate the maximum gives "pick index 1" at step 1 and "pick index 0" at step 2.
        n = len(st[G.GEO_SCORE_KEY])
        s = [0.0] * n
        s[-1] = 1.0
        return s

    _p, _s, rec_h, fr_h, _d = G.stepwise_sample_d3(
        # historical argmax must pick index 1 (customer 5) while geo picks index 0
        # (customer 4: equal distances, smallest canonical key) -> genuine divergence
        score_fn=_last_is_best, extract_fn=extract, legal_fn=_legal_fn(seq_div),
        apply_fn=_identity_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_HISTORICAL, deterministic=True)
    ck('T-G8 two selectors may take different first actions (state divergence)',
       rec_g[0]['customer'] != rec_h[0]['customer'],
       (rec_g[0]['customer'], rec_h[0]['customer']))
    ck('T-G8 divergence does NOT produce a failure reason', fr_g is None and fr_h is None)

    # ---------------- T-G9 identity + historical path untouched ----------------
    orig = MP.stepwise_sample
    ident = G.source_identity([SIM / 'action_contract.py', SIM / 'mpre_policy.py',
                               SIM / 'dynmaskco_cc_context.py', SIM / 'd3_geo_adapter.py'])
    ck('T-G9 per-file source identity recorded for the four modules',
       len(ident) == 4 and all(len(v) == 64 for v in ident.values()))
    ck('T-G9 historical stepwise_sample is NOT replaced by importing the adapter',
       MP.stepwise_sample is orig)
    body = (SIM / 'd3_geo_adapter.py').read_text(encoding='utf-8')
    ck('T-G9 adapter never assigns to the historical sampler',
       'MP.stepwise_sample =' not in body and 'mpre_policy.stepwise_sample =' not in body)

    # ---------------- X1 old/new behaviour DIFF (same legal list) ----------------
    plans_x = _plans({1: (3, [4]), 2: (0, [])})
    seq_x = [_cand(5, 'anchored', 3, 0, 3.0), _cand(5, 'new_route', -1, 0, 3.0)]
    ext_x = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids={1, 2})
    score_x = lambda st: [1.0, 0.0]
    real_enum = MP.enumerate_legal_actions
    MP.enumerate_legal_actions = lambda env, inst, P, rem, mut: _legal_fn(seq_x)(P, rem)
    try:
        p_a, s_a, rec_a, fr_a = MP.stepwise_sample(
            score_fn=score_x, extract_fn=ext_x, env=None, inst_idx=0, plans=plans_x,
            mask_set=[5], mutable_ids={1, 2}, deterministic=True,
            rng=np.random.default_rng(7))
    finally:
        MP.enumerate_legal_actions = real_enum
    p_b, s_b, rec_b, fr_b, _d = G.stepwise_sample_d3(
        score_fn=score_x, extract_fn=ext_x, legal_fn=_legal_fn(seq_x), apply_fn=apply_action,
        plans=plans_x, mask_set=[5], mutable_ids={1, 2}, selector=G.SELECTOR_HISTORICAL,
        deterministic=True, rng=np.random.default_rng(7))
    # The adapter adds EXACTLY ONE extra key per step ('incremental_distance', the chosen
    # action's environment distance, kept as evidence).  Assert the historical projection is
    # identical AND that the key delta is exactly that one field -- so the delta is pinned
    # rather than papered over.
    proj_b = [{k: v for k, v in s.items() if k != 'incremental_distance'} for s in s_b]
    extra = set()
    for a_, b_ in zip(s_a, s_b):
        extra |= (set(b_) - set(a_))
    ck('X1 historical mode reproduces the ORIGINAL stepwise_sample (steps + failure reason)',
       proj_b == s_a and fr_a == fr_b,
       {'orig': s_a, 'adapter_projected': proj_b, 'fr': (fr_a, fr_b)})
    ck('X1 the adapter adds exactly one extra step field (incremental_distance), nothing else',
       extra == {'incremental_distance'}, extra)
    ck('X1 the historical sampler is restored after the isolation',
       MP.enumerate_legal_actions is real_enum)

    # ---------------- X2 F behaviour regression ----------------
    ck('X2 F rule is still first-maximum argmax',
       G.select_historical_argmax([2.0, 2.0, 1.0]) == 0)
    # NOTE: 1e-13 is STRICTLY greater than 0.0 in float, so first-maximum argmax must return 1.
    # That is exactly the point: F has NO tolerance, so a near-tie is decided by the raw float
    # value, whereas D-Geo would treat it as a tie and break it by key order.
    ck('X2 F has no distance tolerance: a 1e-13 gap is NOT treated as a tie (picks index 1)',
       G.select_historical_argmax([0.0, 1e-13, 0.0]) == 1)

    passed = sum(1 for _n, ok_, _d in CASES if ok_)
    for n, ok_, detail in CASES:
        print(('  PASS ' if ok_ else '  FAIL ') + n + ('' if ok_ else ('   <- ' + str(detail))))
    failed = len(CASES) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
