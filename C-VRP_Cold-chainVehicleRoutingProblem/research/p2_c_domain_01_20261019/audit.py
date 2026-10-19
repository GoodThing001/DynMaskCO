"""P2-C-DOMAIN-01: legal mask scope and decision-timing audit (READ-ONLY).

Re-reads the frozen CTRL/OPP phase records; never executes a controller, a repair,
a rollout, an evaluator or a source day. Legality-only enumeration is used to count
DISTINCT candidate plan identities under four pre-declared mask granularities.

Bucket rule = registration.json as CORRECTED by amendment_01.json (the registered
precedence misfiled a `committed` vehicle's queued suffix as a replan-selection
exclusion; the amendment is registered before this rerun).

Usage: python audit.py --out-dir <this directory>       (refuses to overwrite)
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
SAA = ROOT / 'research/p2_c_opp_01_20261019'
CTRL = ROOT / 'research/p2_c_ctrl_01_20261019'

sys.path.insert(0, str(ROOT / 'scripts'))
from project_paths import configure_imports                       # noqa: E402
configure_imports(('experimental', 'simulation', 'coldchain', 'evaluation', 'training'))
import p2_pos01_reference as ref                                   # noqa: E402
import action_contract as ac                                       # noqa: E402
from p2_pos01c_gap import view_from_json                           # noqa: E402
from p2_pos01c_control import occupancy                            # noqa: E402

BUCKETS = ('MUTABLE', 'ON_COMMITTED_VEHICLE_ANCHOR', 'ON_COMMITTED_VEHICLE_QUEUE',
           'ON_CLOSED_VEHICLE', 'ON_PROTECTED_VEHICLE', 'ON_EXCLUDED_IDLE_VEHICLE',
           'UNREPRESENTED')
# must stay empty if the whole lock is the commitment contract (amendment_01 falsification test)
RESIDUAL = ('ON_CLOSED_VEHICLE', 'ON_PROTECTED_VEHICLE', 'ON_EXCLUDED_IDLE_VEHICLE',
            'UNREPRESENTED')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def regime(view):
    """The engine has exactly three decision-point regimes; recorded, not assumed."""
    n_replan = len(set(view.replan_ids or ()))
    n_committed = sum(1 for v in view.vehicles if v.status == 'committed')
    n_active = sum(1 for v in view.vehicles if v.status != 'closed')
    return n_replan, n_committed, n_active


def classify_lock(view, occ_row):
    """Amended rule: attribute every accepted-and-unserved order to exactly one bucket."""
    pending = sorted(set(view.accepted) - set(view.served))
    allowed = set(occ_row['legal_vehicle_ids'])
    protected = set(view.protected or ())
    vmap = {v.vehicle_id: v for v in view.vehicles}
    anchors = {int(v.committed_next): v.vehicle_id for v in view.vehicles
               if v.status == 'committed' and v.committed_next not in (None, 0)}
    owner = {}
    for v in view.vehicles:
        for c in (v.mutable_suffix or ()):
            c = int(c)
            if c:
                owner.setdefault(c, v.vehicle_id)
    buckets = {k: [] for k in BUCKETS}
    for o in pending:
        if o in anchors:
            buckets['ON_COMMITTED_VEHICLE_ANCHOR'].append(o)
            continue
        vid = owner.get(o)
        if vid is None:
            buckets['UNREPRESENTED'].append(o)
            continue
        v = vmap[vid]
        if vid in allowed:
            buckets['MUTABLE'].append(o)
        elif v.status == 'committed':
            buckets['ON_COMMITTED_VEHICLE_QUEUE'].append(o)
        elif v.status == 'closed':
            buckets['ON_CLOSED_VEHICLE'].append(o)
        elif vid in protected:
            buckets['ON_PROTECTED_VEHICLE'].append(o)
        else:
            buckets['ON_EXCLUDED_IDLE_VEHICLE'].append(o)
    return buckets, pending


def distinct_candidates(view, plans, mask, allowed):
    seen = set()

    def rec(P, remaining):
        if not remaining:
            seen.add(ac.plan_hash(P))
            return
        for c, action, _cand in ref.legal_actions(view, P, list(remaining)):
            rec(ac.apply_action(P, action, allowed_vehicle_ids=allowed),
                [x for x in remaining if x != c])

    rec(plans, sorted(mask))
    return seen


def pool_of(view, plans, allowed):
    return sorted(c for vid, p in plans.items() if vid in allowed for c in p.suffix
                  if c in view.accepted and c not in view.served)


def census(view, plans, allowed, pool):
    res = {}

    def run(name, masks):
        cand = set()
        for m in masks:
            cand |= distinct_candidates(view, plans, m, allowed)
        res[name] = {'n_masks': len(masks), 'n_candidates': len(cand)}

    run('CURRENT', list(ref.mask_pool(view, plans, seed=0)))
    s1 = [(c,) for c in pool]
    run('S1_ONLY', s1)
    m12 = s1 + [tuple(sorted(p)) for p in combinations(pool, 2)]
    run('MIXED_1_2', m12)
    run('MIXED_1_2_3', m12 + [tuple(sorted(p)) for p in combinations(pool, 3)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=OUT)
    a = ap.parse_args()
    if a.out_dir.resolve() != OUT.resolve():
        raise SystemExit('refuse output outside this diagnostic directory')

    reg = load(OUT / 'registration.json')
    amend = load(OUT / 'amendment_01.json')
    for rel, want in reg['inputs'].items():
        if sha(WORKSPACE / rel) != want:
            raise SystemExit('INPUT_IDENTITY_DRIFT: %s' % rel)
    policies = reg['declared_policies']

    lock_rows = []
    per_arm = {}
    prev_pool = {}
    residual_violations = []
    for arm, folder in (('SAA', SAA), ('MYOPIC', CTRL)):
        phases = load(folder / 'phases.json')
        acc = Counter()
        reg_hist = Counter()
        reg_stats = {}
        census_tot = {p: Counter() for p in policies}
        became_frozen = 0
        scope_blocked = 0
        for p in phases:
            view = view_from_json({'public_view': p['public_view'],
                                   'public_identity': p['public_identity']})
            plans = ref.plans_from_view(view)
            allowed = ref.mutable_ids(view)
            if [list(m) for m in ref.mask_pool(view, plans, seed=0)] != p['masks']['masks']:
                raise SystemExit('MASK_REPRODUCTION_MISMATCH %s %s %s'
                                 % (arm, p['day'], p['ordinal']))
            occ = occupancy(view)
            buckets, pending = classify_lock(view, occ)
            pool = pool_of(view, plans, allowed)
            if len(pool) != occ['total_mutable_customers']:
                raise SystemExit('POOL_MISMATCH %s %s' % (arm, p['day']))
            n_replan, n_committed, n_active = regime(view)
            reg_key = '|replan|=%d,committed=%d,active=%d' % (n_replan, n_committed, n_active)
            reg_hist[reg_key] += 1
            rs = reg_stats.setdefault(reg_key, {'phases': 0, 'pool': Counter(),
                                                'pending': Counter(), 'cands': Counter()})
            rs['phases'] += 1
            rs['pool'][len(pool)] += 1
            rs['pending'][len(pending)] += 1

            for k in RESIDUAL:
                if buckets[k]:
                    residual_violations.append({'arm': arm, 'day': p['day'],
                                                'ordinal': p['ordinal'], 'bucket': k,
                                                'orders': buckets[k]})

            key = (arm, p['day'])
            prev = prev_pool.get(key)
            frozen_now = set(pending) - set(pool)
            if prev is not None:
                became_frozen += len(frozen_now & prev)
            prev_pool[key] = set(pool)
            if len(pool) < 2 and len(pending) >= 2:
                scope_blocked += 1

            cand = census(view, plans, allowed, pool)
            row = {'arm': arm, 'day': p['day'], 'ordinal': p['ordinal'],
                   'clock_h': round(float(p['clock']), 6),
                   'regime': reg_key, 'n_replan_ids': n_replan,
                   'n_committed_vehicles': n_committed,
                   'n_pending': len(pending), 'n_mutable': len(pool),
                   'n_locked': len(pending) - len(pool),
                   'masks_current': cand['CURRENT']['n_masks']}
            for k, v in buckets.items():
                row['lock_' + k.lower()] = len(v)
                acc[k] += len(v)
            for pol in policies:
                row['%s_masks' % pol.lower()] = cand[pol]['n_masks']
                row['%s_cands' % pol.lower()] = cand[pol]['n_candidates']
                census_tot[pol]['masks'] += cand[pol]['n_masks']
                census_tot[pol]['phases_ge1'] += (cand[pol]['n_candidates'] >= 1)
                census_tot[pol]['phases_ge2'] += (cand[pol]['n_candidates'] >= 2)
                rs['cands'][pol] += (cand[pol]['n_candidates'] >= 2)
            lock_rows.append(row)

        per_arm[arm] = {
            'phases': len(phases),
            'orders_total': {k: int(acc[k]) for k in BUCKETS},
            'locked_share': (float(acc['ON_COMMITTED_VEHICLE_ANCHOR']
                                   + acc['ON_COMMITTED_VEHICLE_QUEUE'])
                             / max(1, sum(acc.values()))),
            'regime_histogram': {k: int(v) for k, v in reg_hist.items()},
            'regime_detail': {k: {'phases': v['phases'],
                                  'mutable_pool_histogram': dict(sorted(v['pool'].items())),
                                  'pending_histogram': dict(sorted(v['pending'].items())),
                                  'phases_with_ge2_candidates': {p: int(v['cands'][p])
                                                                 for p in policies}}
                              for k, v in reg_stats.items()},
            'phases_with_fewer_than_2_mutable_but_pending_ge2': scope_blocked,
            'orders_mutable_at_prev_phase_and_frozen_now': became_frozen,
            'census': {p: {k: int(v) for k, v in c.items()} for p, c in census_tot.items()},
        }

    fields = sorted({k for r in lock_rows for k in r})
    with (OUT / 'lock_attribution.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(lock_rows)

    summary = {
        'schema': 'p2_c_domain_01_summary_v2',
        'run_id': reg['run_id'],
        'amendment': amend['amendment_id'],
        'qualification': 'DOMAIN_ONLY',
        'analyzed_phases': len(lock_rows),
        'per_arm': per_arm,
        'residual_bucket_violations': residual_violations,
        'q3_timing': {
            'single_captured_phase_type': 'POST_ADMISSION_PRE_REPAIR',
            'alternative_capture_points': 'UNASSESSABLE_FROM_ARTIFACT',
            'why': ('all 289 saved phases are captured at one fixed point; evaluating an earlier '
                    'or later legal decision boundary requires an instrumented engine run, which '
                    'is not authorised in this round'),
            'legal_option_caveat': ('"mutable at the previous phase, frozen now" is a timing fact '
                                    'about the saved trajectory. It is NOT an option: the committed '
                                    'leg cannot be reopened without violating the strict-online '
                                    'commitment contract.'),
        },
        'outputs': {'lock_attribution.csv': sha(OUT / 'lock_attribution.csv')},
        'audit_source_sha256': sha(Path(__file__)),
        'superseded': 'superseded_v1/ holds the pre-amendment run; those numbers must not be cited',
        'limitations': [
            'two previously inspected developer roots only; phase exposures are not independent days',
            'candidate identity is a plan hash under legality-only enumeration; it does not measure '
            'quality, utility, certifiability or executability',
            'no controller, repair, rollout, evaluator or source day was executed',
        ],
    }
    with (OUT / 'domain_summary.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')

    print('PASS phases=%d  residual_violations=%d' % (len(lock_rows), len(residual_violations)))
    for arm, d in per_arm.items():
        print(' ', arm, 'orders:', d['orders_total'], ' locked_share=%.4f' % d['locked_share'])
        print('        regimes:', d['regime_histogram'])
        for rk, rv in sorted(d['regime_detail'].items()):
            print('        %-32s phases=%3d pool=%s' % (rk, rv['phases'],
                                                        rv['mutable_pool_histogram']))
        for pol in policies:
            c = d['census'][pol]
            print('        %-11s masks=%4d phases>=1cand=%3d phases>=2cand=%3d'
                  % (pol, c['masks'], c['phases_ge1'], c['phases_ge2']))


if __name__ == '__main__':
    main()
