"""P2-C-ANCHOR-01: commitment-anchored masked reconstruction (CAMR) scope feasibility.

READ-ONLY. For every saved public phase with exactly one in-transit (`committed`) vehicle,
this asks: if the in-transit leg stays frozen at (committed_next, committed_finish, load after
pickup), does the not-yet-executed suffix admit >=2 DISTINCT candidate plans that pass the same
route-level reference check, and are they distinguishable by the first action AFTER the anchor?

It does not modify the engine, does not execute a source day, does not evaluate utility.
The committed vehicle is only added to `allowed_vehicle_ids` for a hypothetical enumeration;
no module-level behaviour is changed.

Usage: python probe_anchor.py --out-dir <this directory>       (refuses to overwrite)
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
OUT = Path(__file__).resolve().parent
SAA = ROOT / 'research/p2_c_opp_01_20261019' / 'phases.json'
CTRL = ROOT / 'research/p2_c_ctrl_01_20261019' / 'phases.json'

sys.path.insert(0, str(ROOT / 'scripts'))
from project_paths import configure_imports                       # noqa: E402
configure_imports(('experimental', 'simulation', 'coldchain', 'evaluation', 'training'))
import p2_pos01_reference as ref                                   # noqa: E402
import action_contract as ac                                       # noqa: E402
from p2_pos01c_gap import view_from_json                           # noqa: E402

VERDICTS = ('ANCHOR_LEGAL_MULTI', 'ANCHOR_LEGAL_SAME_NEXT', 'NO_LEGAL_CANDIDATE',
            'INSUFFICIENT_QUEUE', 'NO_COMMITTED_VEHICLE', 'MULTIPLE_COMMITTED_VEHICLES')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def legal_actions_with(view, env, plans, remaining, allowed):
    """Exact 1-parameter generalisation of ref.legal_actions (same predicates, explicit allowed).

    ref.legal_actions hardcodes mutable_ids(view); CAMR needs the same test with the in-transit
    vehicle additionally admitted, so this mirrors those six lines instead of changing the module.
    """
    out = []
    for c in remaining:
        if c not in view.accepted or c in view.served:
            raise ValueError('illegal repair customer')
        candidates, _ = ac.enumerate_actions_from_plans(env, 0, plans, c,
                                                        allowed_vehicle_ids=allowed)
        for cand in candidates:
            if cand.feasible:
                trial = ac.apply_action(plans, cand.action, allowed_vehicle_ids=allowed)
                if all(ref.route_check(view, p) for p in trial.values()):
                    out.append((c, cand.action, cand))
    return out


def enumerate_candidates(env, view, plans, mask, allowed):
    """All distinct full-fleet plans reachable by reconstructing `mask` legally."""
    seen = {}

    def rec(P, remaining):
        if not remaining:
            seen.setdefault(ac.plan_hash(P), P)
            return
        for c, action, _cand in legal_actions_with(view, env, P, remaining, allowed):
            rec(ac.apply_action(P, action, allowed_vehicle_ids=allowed),
                [x for x in remaining if x != c])

    rec(plans, sorted(mask))
    return seen


def freeze_ok(view, plans, cand, cvid, anchor, committed_next, accepted_unserved):
    """Anchor frozen, committed leg untouched, every accepted-unserved order exactly once."""
    p = cand[cvid]
    if (p.anchor_node, p.anchor_time, p.anchor_load) != anchor:
        return False, 'ANCHOR_MUTATED'
    for q in cand.values():
        if committed_next in q.suffix:
            return False, 'COMMITTED_LEG_IN_SUFFIX'
    positions = Counter()
    for q in cand.values():
        for c in q.suffix:
            positions[int(c)] += 1
    for o in accepted_unserved:
        if o == committed_next:
            if positions.get(o, 0) != 0:
                return False, 'ANCHOR_ORDER_IN_SUFFIX'
        elif positions.get(o, 0) != 1:
            return False, 'EXACT_ONCE_VIOLATED_%d' % positions.get(o, 0)
    for c, n in positions.items():
        if c not in accepted_unserved:
            return False, 'NON_PENDING_ORDER_IN_PLAN'
    return True, 'OK'


def probe_phase(ph):
    view = view_from_json({'public_view': ph['public_view'],
                           'public_identity': ph['public_identity']})
    plans = ref.plans_from_view(view)
    allowed = set(ref.mutable_ids(view))
    committed = [v for v in view.vehicles
                 if v.status == 'committed' and v.committed_next not in (None, 0)]
    row = {'day': ph['day'], 'ordinal': ph['ordinal'], 'clock_h': round(float(ph['clock']), 6),
           'n_committed': len(committed)}
    if not committed:
        row['verdict'] = 'NO_COMMITTED_VEHICLE'
        return row
    if len(committed) > 1:
        row['verdict'] = 'MULTIPLE_COMMITTED_VEHICLES'
        return row
    v = committed[0]
    cvid = int(v.vehicle_id)
    committed_next = int(v.committed_next)
    base = plans[cvid]
    anchor = (base.anchor_node, base.anchor_time, base.anchor_load)
    accepted = {int(x) for x in view.accepted}
    served = {int(x) for x in view.served}
    pending = accepted - served
    queue = [c for c in base.suffix if c in pending]
    row.update({'committed_vid': cvid, 'committed_next': committed_next,
                'queue_len': len(queue), 'pending_len': len(pending),
                'current_mutable_total': sum(1 for vid, p in plans.items() if vid in allowed
                                             for c in p.suffix if c in pending)})
    if len(queue) < 2:
        row['verdict'] = 'INSUFFICIENT_QUEUE'
        return row

    env = ref.public_env(view)
    allowed_anchor = allowed | {cvid}
    masks = [tuple(sorted(p)) for p in combinations(queue, 2)]
    cands = {}
    for m in masks:
        cands.update(enumerate_candidates(env, view, plans, m, allowed_anchor))

    ok_cands = {}
    freeze_fail = Counter()
    for h, P in cands.items():
        good, why = freeze_ok(view, plans, P, cvid, anchor, committed_next, pending)
        if good and all(ref.route_check(view, q) for q in P.values()):
            ok_cands[h] = P
        elif not good:
            freeze_fail[why] += 1

    next_actions = set()
    for P in ok_cands.values():
        s = list(P[cvid].suffix)
        next_actions.add(s[0] if s else 0)

    row.update({'n_masks': len(masks), 'n_raw_candidates': len(cands),
                'n_legal_candidates': len(ok_cands),
                'n_next_action_after_anchor': len(next_actions),
                'next_actions': sorted(next_actions)[:8],
                'freeze_failures': dict(freeze_fail),
                'anchor': [anchor[0], round(float(anchor[1]), 6), round(float(anchor[2]), 6)]})
    if len(ok_cands) >= 2 and len(next_actions) >= 2:
        row['verdict'] = 'ANCHOR_LEGAL_MULTI'
    elif len(ok_cands) >= 2:
        row['verdict'] = 'ANCHOR_LEGAL_SAME_NEXT'
    else:
        row['verdict'] = 'NO_LEGAL_CANDIDATE'
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=OUT)
    a = ap.parse_args()
    if a.out_dir.resolve() != OUT.resolve():
        raise SystemExit('refuse output outside this directory')

    reg = load(OUT / 'registration.json')
    for rel, want in reg['inputs'].items():
        if sha(WORKSPACE / rel) != want:
            raise SystemExit('INPUT_IDENTITY_DRIFT: %s' % rel)

    rows = []
    for arm, path in (('SAA', SAA), ('MYOPIC', CTRL)):
        for ph in load(path):
            r = probe_phase(ph)
            r['arm'] = arm
            rows.append(r)

    # keep the anchor test rows first for readability
    order = {k: i for i, k in enumerate(VERDICTS)}
    rows.sort(key=lambda r: (order[r['verdict']], r['arm'], r['day'], r['ordinal']))

    fields = sorted({k for r in rows for k in r if k != 'freeze_failures'}) + ['freeze_failures']
    with (OUT / 'anchor_scope.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2['freeze_failures'] = json.dumps(r.get('freeze_failures', {}), sort_keys=True)
            w.writerow(r2)

    summary = {'schema': 'p2_c_anchor_01_scope_v1', 'run_id': reg['run_id'],
               'qualification': 'REFERENCE_ONLY', 'phases': len(rows),
               'verdict_counts': dict(Counter(r['verdict'] for r in rows)),
               'per_arm': {}}
    for arm in ('SAA', 'MYOPIC'):
        sub = [r for r in rows if r['arm'] == arm]
        tested = [r for r in sub if r['verdict'] in ('ANCHOR_LEGAL_MULTI', 'ANCHOR_LEGAL_SAME_NEXT',
                                                     'NO_LEGAL_CANDIDATE')]
        summary['per_arm'][arm] = {
            'phases': len(sub),
            'verdict_counts': dict(Counter(r['verdict'] for r in sub)),
            'anchor_tested_phases': len(tested),
            'phases_with_ge2_legal_candidates': sum(1 for r in tested
                                                    if r['n_legal_candidates'] >= 2),
            'phases_with_ge2_next_actions': sum(1 for r in tested
                                                if r['n_next_action_after_anchor'] >= 2),
            'total_legal_candidates': sum(r['n_legal_candidates'] for r in tested),
            'freeze_failure_total': sum(sum(r['freeze_failures'].values()) for r in tested),
            'detail': [{k: r.get(k) for k in ('day', 'ordinal', 'clock_h', 'committed_next',
                                              'queue_len', 'pending_len', 'n_masks',
                                              'n_raw_candidates', 'n_legal_candidates',
                                              'n_next_action_after_anchor', 'next_actions',
                                              'anchor', 'verdict')} for r in tested],
        }
    summary['h1_verdict'] = ('ANCHOR_SCOPE_ESTABLISHED'
                             if any(v['phases_with_ge2_legal_candidates'] > 0
                                    for v in summary['per_arm'].values())
                             else 'ANCHOR_SCOPE_NOT_ESTABLISHED')
    with (OUT / 'anchor_summary.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')

    print('phases', len(rows), summary['verdict_counts'])
    for arm, v in summary['per_arm'].items():
        print(' ', arm, 'tested', v['anchor_tested_phases'],
              'ge2_cands', v['phases_with_ge2_legal_candidates'],
              'ge2_next', v['phases_with_ge2_next_actions'],
              'freeze_fail', v['freeze_failure_total'])
    print('H1:', summary['h1_verdict'])


if __name__ == '__main__':
    main()
