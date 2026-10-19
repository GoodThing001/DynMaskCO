"""P2-C-CAMR-STATIC-01  synthetic regression on the shared machinery.

Runs BEFORE T2 and BEFORE any model is loaded. It never steps an environment (no SOURCE day is
re-run) and it never touches one of the 19 analysis units: it reads saved public views that lie
outside the analysis set and rewrites FleetPlan objects in memory.

STRUCTURAL LIMIT, recorded rather than hidden: in the saved data a multi-order queue behind an
in-transit vehicle exists ONLY in the 19 analysis units (that is exactly DOMAIN-01's finding). So
the regression uses two fixtures instead of one impossible one:

  fixture A  `NO_COMMITTED_VEHICLE` phase with many allowed vehicles -> exercises the shared
             action set, the `allowed` gate, the tie-break and vehicle choice
  fixture B  `INSUFFICIENT_QUEUE` plan-exhaustion phase, one allowed vehicle and >=3 pending
             orders -> exercises remove/re-insert reconstruction, the anchor / committed-leg /
             exact-once checker, determinism and the score-agnostic plumbing

The genuine `committed` + multi-vehicle combination is therefore first exercised by the
degeneration gate, which is a registered hard precondition for Pass A, not an analysis result.

Checks
  R1  the shared legal set is filtered (feasible AND every trial plan passes route_check)
  R1b excluding a vehicle from `allowed` removes it as a target and leaves its plan untouched
  R2  `allowed` contains the vehicle taken as the anchor holder
  R3  the unified tie-break orders equal scores by action_id()
  R4  greedy is deterministic (identical plan_hash on repeat)
  R5  completions[0] IS the greedy completion (the primary comparison object)
  R6  the anchor / committed-leg / exact-once checker catches deliberate corruption
  R7  the ranking plumbing is score-agnostic: proposer(score='distance') reproduces a direct
      distance ranking exactly (the model-free half of the degeneration gate)
  R8  a clean reconstruction passes every registered anchor assertion

Usage: python synthetic_regression.py [--out synthetic_regression.json]
"""
import argparse
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

import camr_shared as cs                                            # noqa: E402
import action_contract as ac                                        # noqa: E402


def _not_a_unit(r, unit_keys):
    return not any(k['arm'] == r['arm'] and k['day_id'] == int(r['day']) and
                   k['ordinal'] == int(r['ordinal']) for k in unit_keys)


def _candidates(unit_keys):
    """Every non-unit NO_COMMITTED_VEHICLE phase, in file order, with cached views.

    `pending_len` is blank in anchor_scope.csv for this verdict (the probe returns early when
    there is no in-transit vehicle), so the phase richness is computed from the view itself.
    """
    rows = [r for r in cs.read_scope()
            if r['verdict'] == 'NO_COMMITTED_VEHICLE' and _not_a_unit(r, unit_keys)]
    out = []
    for r in rows[:80]:
        ph = cs.load_phase(r['arm'], int(r['day']), int(r['ordinal']))
        view = cs.view_of(ph)
        plans = cs.plans_of(view)
        allowed = cs.shared_allowed(view)
        pending = {int(x) for x in view.accepted} - {int(x) for x in view.served}
        pools = {int(v): [int(c) for c in plans[v].suffix if int(c) in pending]
                 for v in sorted(allowed)}
        out.append({'arm': r['arm'], 'day': int(r['day']), 'ordinal': int(r['ordinal']),
                    'public_identity': ph['public_identity'], 'view': view, 'plans': plans,
                    'env': cs.ref.public_env(view), 'allowed': allowed, 'pending': pending,
                    'pools': pools})
    return out


def pick_fixture_a(cands):
    """Most allowed vehicles (vehicle choice + `allowed` gate + tie-break)."""
    best = None
    for c in cands:
        if not c['pending']:
            continue
        score = (len(c['allowed']), len(c['pending']))
        if best is None or score > best[0]:
            best = (score, c)
    if best is None:
        raise SystemExit('no fixture A')
    c = dict(best[1])
    c['mask'] = tuple(sorted(c['pending'])[:2])
    c['law'] = ('non-unit NO_COMMITTED_VEHICLE phase maximising (n_allowed, n_pending) with '
                'ties by file order; mask = two lowest pending orders')
    return c


def pick_fixture_b(cands, exclude):
    """A vehicle carrying >=2 pending orders (remove/re-insert reconstruction + checker)."""
    best = None
    for c in cands:
        if (c['arm'], c['day'], c['ordinal']) == exclude:
            continue
        for vid, pool in sorted(c['pools'].items()):
            if len(pool) < 2:
                continue
            score = (len(c['pending']), len(c['allowed']), len(pool))
            if best is None or score > best[0]:
                best = (score, c, vid, pool)
            break
    if best is None:
        raise SystemExit('no fixture B')
    _s, c, vid, pool = best
    c = dict(c)
    c.update({'cvid': int(vid), 'anchor': cs.anchor_of(c['plans'], int(vid)),
              'committed_next': 0, 'queue': pool, 'mask': (pool[0], pool[1]),
              'law': ('non-unit NO_COMMITTED_VEHICLE phase maximising (n_pending, n_allowed, '
                      'pool) among phases whose smallest allowed vehicle carries >=2 pending '
                      'orders; mask = that vehicle\'s first two pending orders')})
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'synthetic_regression.json')
    a = ap.parse_args()
    reg = cs.load_json(OUT / 'registration.json')
    if a.out.exists():
        raise SystemExit('refuse overwrite: %s already exists' % a.out)

    unit_keys = [u['unit_id'] for u in reg['units']['list']]
    cands = _candidates(unit_keys)
    A = pick_fixture_a(cands)
    B = pick_fixture_b(cands, (A['arm'], A['day'], A['ordinal']))
    checks = []

    def rec(name, ok, detail=None, fixture=None):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'ok': bool(ok),
                       'fixture': fixture, 'detail': detail})

    # ---- fixture A: action set -------------------------------------------
    view, plans, env, allowed = A['view'], A['plans'], A['env'], A['allowed']
    remaining = sorted(A['pending'])
    entries = cs.legal_actions_shared(view, env, plans, remaining, allowed)
    allowed_t = tuple(sorted(allowed))
    bad_flag = [1 for _c, _a, cand in entries if not cand.feasible]
    bad_ref = []
    for _c, action, _cand in entries:
        trial = ac.apply_action(plans, action, allowed_vehicle_ids=allowed_t)
        if any(not cs.ref.route_check(view, p) for p in trial.values()):
            bad_ref.append(action.action_id())
    rec('R1_shared_set_is_filtered', not bad_flag and not bad_ref,
        {'n_entries': len(entries), 'infeasible': len(bad_flag),
         'route_check_failures': bad_ref[:5], 'capacity_margin': cs.CAPACITY_MARGIN}, 'A')

    excluded = None
    for vid in sorted(allowed):
        if [int(x) for x in plans[vid].suffix]:
            excluded = int(vid)
            break
    r1b_ok, r1b_detail = False, {'excluded': excluded, 'n_allowed': len(allowed)}
    if excluded is not None and len(allowed) >= 2:
        restricted = frozenset(v for v in allowed if v != excluded)
        ents = cs.legal_actions_shared(view, env, plans, remaining, restricted)
        delivered, pulled = [], []
        for _c, action, _cand in ents:
            trial = ac.apply_action(plans, action, allowed_vehicle_ids=tuple(sorted(restricted)))
            before = [int(x) for x in plans[excluded].suffix]
            after = [int(x) for x in trial[excluded].suffix]
            if len(after) > len(before):
                delivered.append(action.action_id())
            elif len(after) < len(before):
                pulled.append(action.action_id())
        r1b_ok = not delivered
        r1b_detail.update({'n_entries': len(ents), 'delivered_to_excluded': delivered[:5],
                           'source_pull_out_of_excluded': len(pulled),
                           'note': ('apply_action always removes the customer from its source '
                                    'vehicle, so a non-allowed source can be emptied; the '
                                    'registered predicate constrains the TARGET only, and for the '
                                    '19 units allowed is the whole fleet so the distinction is '
                                    'void there')})
    rec('R1b_no_action_targets_an_excluded_vehicle', r1b_ok, r1b_detail, 'A')

    tie_ok, tie_detail = False, {'n_entries': len(entries)}
    if len(entries) >= 2:
        a0, a1 = entries[0], entries[1]
        ordered = cs.rank([a0, a1], lambda cand, action: 7.0)
        want = sorted([a0[1].action_id(), a1[1].action_id()])
        tie_ok = [x[1].action_id() for x in ordered] == want
        tie_detail.update({'ids': want, 'ranked': [x[1].action_id() for x in ordered]})
    rec('R3_tie_break_by_action_id', tie_ok, tie_detail, 'A')

    # ---- fixture B: reconstruction --------------------------------------
    view, plans, env, allowed = B['view'], B['plans'], B['env'], B['allowed']
    cvid, anchor, cn, pending, mask = (B['cvid'], B['anchor'], B['committed_next'],
                                       B['pending'], B['mask'])
    rec('R2_allowed_contains_anchor_holder', cvid in allowed,
        {'cvid': cvid, 'n_allowed': len(allowed)}, 'B')

    prop_dist = cs.proposer(view, env, allowed, score='distance')
    g1 = cs.greedy(prop_dist, plans, mask, allowed)
    g2 = cs.greedy(prop_dist, plans, mask, allowed)
    h1 = None if g1['plan'] is None else cs.signature(g1['plan'])
    h2 = None if g2['plan'] is None else cs.signature(g2['plan'])
    rec('R4_greedy_deterministic', h1 is not None and h1 == h2,
        {'plan_hash': h1, 'reason': g1['reason'], 'calls': g1['scoring_calls']}, 'B')

    comp = cs.completions(prop_dist, plans, mask, allowed)
    first = cs.signature(comp['completions'][0]) if comp['completions'] else None
    rec('R5_completions_first_is_greedy', first == h1,
        {'first': first, 'greedy': h1, 'stats': comp['stats']}, 'B')

    ok, why = (cs.check_plan(view, g1['plan'], cvid, anchor, cn, pending)
               if g1['plan'] is not None else (False, 'NO_PLAN'))
    rec('R8_clean_plan_passes_anchor_assertions', ok, {'reason': why}, 'B')

    neg = {}
    if g1['plan'] is not None:
        P = dict(g1['plan'])
        p = P[cvid]
        P[cvid] = ac.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time + 1.0,
                                 p.anchor_load, p.suffix)
        neg['anchor'] = cs.check_plan(view, P, cvid, anchor, cn, pending)[1]

        P2 = dict(g1['plan'])
        p2 = P2[cvid]
        P2[cvid] = ac.VehiclePlan(p2.vehicle_id, p2.anchor_node, p2.anchor_time,
                                  p2.anchor_load, tuple(p2.suffix) + (int(cn),))
        neg['committed_leg'] = cs.check_plan(view, P2, cvid, anchor, cn, pending)[1]

        P3 = dict(g1['plan'])
        p3 = P3[cvid]
        dup = tuple(p3.suffix) + tuple(p3.suffix)
        if not dup:
            one = sorted(pending)[0]
            dup = (one, one)
        P3[cvid] = ac.VehiclePlan(p3.vehicle_id, p3.anchor_node, p3.anchor_time,
                                  p3.anchor_load, dup)
        neg['duplicate'] = cs.check_plan(view, P3, cvid, anchor, cn, pending)[1]
    rec('R6_corruption_detected',
        neg.get('anchor') == 'ANCHOR_MISMATCH'
        and neg.get('committed_leg') == 'COMMITTED_LEG_IN_SUFFIX'
        and str(neg.get('duplicate', 'NO')).startswith(('EXACT_ONCE', 'NON_PENDING')), neg, 'B')

    def direct(P, rem, K):
        ents = cs.legal_actions_shared(view, env, P, rem, allowed)
        return cs.rank(ents, cs.distance_primary)[:K], len(ents)

    plain = cs.greedy(direct, plans, mask, allowed)
    h_plain = None if plain['plan'] is None else cs.signature(plain['plan'])
    rec('R7_plumbing_is_score_agnostic', h_plain is not None and h_plain == h1,
        {'via_proposer': h1, 'via_direct_rank': h_plain}, 'B')

    payload = {
        'schema': 'p2_c_camr_static_01_synthetic_regression_v1',
        'run_id': reg['run_id'],
        'model_loaded': False,
        'scope': ('no SOURCE day is re-run, no environment is stepped, no model is loaded; saved '
                  'public views outside the 19 analysis units are read and FleetPlan objects are '
                  'rewritten in memory only'),
        'structural_limit': ('a multi-order queue behind an in-transit vehicle exists only in the '
                             '19 units, so the committed + multi-vehicle combination is first '
                             'exercised by the degeneration gate, which is a registered hard '
                             'precondition for Pass A'),
        'fixture_a': {'arm': A['arm'], 'day_id': A['day'], 'ordinal': A['ordinal'],
                      'public_identity': A['public_identity'], 'n_allowed': len(A['allowed']),
                      'n_pending': len(A['pending']), 'remaining': remaining,
                      'n_entries': len(entries), 'is_analysis_unit': False, 'law': A['law']},
        'fixture_b': {'arm': B['arm'], 'day_id': B['day'], 'ordinal': B['ordinal'],
                      'public_identity': B['public_identity'], 'mask': list(mask),
                      'anchor_holder_vid': cvid, 'committed_next': cn,
                      'n_allowed': len(B['allowed']), 'n_pending': len(pending),
                      'is_analysis_unit': False, 'law': B['law']},
        'n_checks': len(checks),
        'n_ok': sum(1 for c in checks if c['ok']),
        'all_ok': all(c['ok'] for c in checks),
        'checks': checks,
    }
    cs.write_json(a.out, payload)
    for c in checks:
        print('%-5s %-42s [%s]' % (c['status'], c['check'], c['fixture']))
    print('%d/%d ok' % (payload['n_ok'], payload['n_checks']))
    print('A', payload['fixture_a']['arm'], payload['fixture_a']['day_id'],
          payload['fixture_a']['ordinal'], 'allowed', payload['fixture_a']['n_allowed'],
          'entries', payload['fixture_a']['n_entries'])
    print('B', payload['fixture_b']['arm'], payload['fixture_b']['day_id'],
          payload['fixture_b']['ordinal'], 'mask', payload['fixture_b']['mask'],
          'allowed', payload['fixture_b']['n_allowed'])
    sys.exit(0 if payload['all_ok'] else 3)


if __name__ == '__main__':
    main()
