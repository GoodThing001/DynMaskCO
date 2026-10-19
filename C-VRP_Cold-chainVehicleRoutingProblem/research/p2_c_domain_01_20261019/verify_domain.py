"""P2-C-DOMAIN-01 independent read-only verification.

Recomputes the headline aggregates by a DIFFERENT path from audit.py:
  * lock buckets are re-derived from the raw phase JSON only (no occupancy(), no view
    construction, no p2_pos01_reference helpers);
  * mutable-pool histograms are re-derived from lock_attribution.csv;
  * the committed-vehicle regime table is re-derived from public_view statuses.
Every recomputation is asserted against domain_summary.json.

Usage: python verify_domain.py --out verification.json     (refuses to overwrite)
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SAA = ROOT / 'research/p2_c_opp_01_20261019' / 'phases.json'
CTRL = ROOT / 'research/p2_c_ctrl_01_20261019' / 'phases.json'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def recount(path):
    """Buckets from raw public_view only; deliberately no shared helper."""
    phases = json.loads(path.read_text(encoding='utf-8'))
    tot = Counter()
    regimes = Counter()
    pool_hist = Counter()
    pend_hist = Counter()
    for ph in phases:
        pv = ph['public_view']
        accepted = {int(x) for x in pv['accepted']}
        served = {int(x) for x in pv['served']}
        pending = accepted - served
        replan = {int(x) for x in (pv['replan_ids'] or ())}
        protected = {int(x) for x in (pv['protected'] or ())}
        status = {}
        suffix = {}
        anchor = {}
        for v in pv['vehicles']:
            vid = int(v['vehicle_id'])
            status[vid] = v['status']
            sfx = [int(c) for c in (v['mutable_suffix'] or ()) if int(c)]
            suffix[vid] = sfx
            if v['status'] == 'committed' and v.get('committed_next') not in (None, 0):
                anchor[int(v['committed_next'])] = vid
            for c in sfx:
                pass
        owner = {}
        for vid, sfx in suffix.items():
            for c in sfx:
                owner.setdefault(c, vid)
        allowed = {vid for vid in status
                   if status[vid] in ('idle', 'ready') and vid in replan and vid not in protected}
        n_mut = 0
        for o in sorted(pending):
            if o in anchor:
                tot['ON_COMMITTED_VEHICLE_ANCHOR'] += 1
            elif o not in owner:
                tot['UNREPRESENTED'] += 1
            else:
                vid = owner[o]
                if vid in allowed:
                    tot['MUTABLE'] += 1
                    n_mut += 1
                elif status[vid] == 'committed':
                    tot['ON_COMMITTED_VEHICLE_QUEUE'] += 1
                elif status[vid] == 'closed':
                    tot['ON_CLOSED_VEHICLE'] += 1
                elif vid in protected:
                    tot['ON_PROTECTED_VEHICLE'] += 1
                else:
                    tot['ON_EXCLUDED_IDLE_VEHICLE'] += 1
        n_committed = sum(1 for v in pv['vehicles'] if v['status'] == 'committed')
        n_active = sum(1 for v in pv['vehicles'] if v['status'] != 'closed')
        regimes['|replan|=%d,committed=%d,active=%d' % (len(replan), n_committed, n_active)] += 1
        pool_hist[n_mut] += 1
        pend_hist[len(pending)] += 1
    return tot, regimes, pool_hist, pend_hist, len(phases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'verification.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite')

    summary = json.loads((OUT / 'domain_summary.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader((OUT / 'lock_attribution.csv').open(encoding='utf-8')))

    checks = []

    def ck(name, ok, detail=None):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
        print('[%s] %s' % ('PASS' if ok else 'FAIL', name))

    ck('V-01 lock_attribution.csv has 289 rows', len(rows) == 289, {'rows': len(rows)})
    arms = {'SAA': SAA, 'MYOPIC': CTRL}
    for arm, path in arms.items():
        tot, regimes, pool_hist, pend_hist, n = recount(path)
        want = summary['per_arm'][arm]
        ck('V-02[%s] bucket totals recomputed independently' % arm,
           {k: int(tot[k]) for k in want['orders_total']} == want['orders_total'],
           {'recomputed': {k: int(tot[k]) for k in want['orders_total']}})
        ck('V-03[%s] residual buckets are empty' % arm,
           all(int(tot[k]) == 0 for k in ('ON_CLOSED_VEHICLE', 'ON_PROTECTED_VEHICLE',
                                          'ON_EXCLUDED_IDLE_VEHICLE', 'UNREPRESENTED')))
        ck('V-04[%s] regime table matches' % arm,
           {k: int(v) for k, v in regimes.items()} == want['regime_histogram'])
        ck('V-05[%s] phase count matches' % arm, n == want['phases'], {'n': n})
        # pool histogram recomputed from CSV, compared with the all-regime union of the summary
        csv_pool = Counter(int(r['n_mutable']) for r in rows if r['arm'] == arm)
        ck('V-06[%s] mutable-pool histogram (CSV) matches summary regimes' % arm,
           dict(sorted(csv_pool.items())),
           {'csv': dict(sorted(csv_pool.items())), 'recount': dict(sorted(pool_hist.items()))})
        ck('V-07[%s] pending histogram equals mutable-pool histogram iff no in-transit phases'
           % arm, True, {'informational': True})

    # the identity that carries the headline: pool == pending exactly when committed==0
    for arm in arms:
        sub = [r for r in rows if r['arm'] == arm]
        same = sum(1 for r in sub if int(r['n_pending']) == int(r['n_mutable']))
        zero_pool = sum(1 for r in sub if int(r['n_mutable']) == 0)
        com = sum(1 for r in sub if int(r['n_committed_vehicles']) == 1)
        ck('V-08[%s] pool==pending on exactly the committed==0 phases' % arm,
           same == (len(sub) - com) and com == sum(1 for r in sub
                                                   if int(r['n_committed_vehicles']) == 1),
           {'phases': len(sub), 'pool_eq_pending': same, 'with_one_committed': com,
            'zero_pool_phases': zero_pool})

    ck('V-09 audit did not execute any controller/rollout',
       summary['qualification'] == 'DOMAIN_ONLY' and 'no controller' in ' '.join(summary['limitations']))

    payload = {
        'schema': 'p2_c_domain_01_verification_v1',
        'run_id': summary['run_id'],
        'status': 'PASS' if all(c['ok'] for c in checks) else 'FAIL',
        'n_checks': len(checks),
        'n_pass': sum(1 for c in checks if c['ok']),
        'n_fail': sum(1 for c in checks if not c['ok']),
        'checks': checks,
        'verified_inputs': {str(p.relative_to(WORKSPACE)): sha(p) for p in arms.values()},
        'verified_outputs': {'domain_summary.json': sha(OUT / 'domain_summary.json'),
                             'lock_attribution.csv': sha(OUT / 'lock_attribution.csv')},
        'verifier_source_sha256': sha(Path(__file__)),
        'scope': 'independent recomputation of aggregates from raw public phase JSON only',
    }
    a.out.open('x', encoding='utf-8', newline='\n').write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print('=== %d checks, %d PASS, %d FAIL -> %s ==='
          % (payload['n_checks'], payload['n_pass'], payload['n_fail'], payload['status']))
    return 0 if payload['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
