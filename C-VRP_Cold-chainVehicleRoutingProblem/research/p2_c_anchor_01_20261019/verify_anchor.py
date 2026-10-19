"""P2-C-ANCHOR-01 independent verification of the anchor-scope aggregates.

READ-ONLY. Re-derives every headline number from `anchor_scope.csv` and from the source phase
files instead of trusting `anchor_summary.json`, and asserts the invariants the report leans on.
Also checks that the execution engine is byte-identical to its committed blob, which is what
lets the synthetic transaction test claim "engine not modified".

Checks
  A  input identity: every registered input still has the recorded sha256
  B  baseline identity: HEAD == registration['baseline'], and the worktree engine file is
     byte-identical to its blob at HEAD
  C  csv/json agreement: verdict counts, per-arm counts and the H1 verdict recomputed from
     `anchor_scope.csv` must equal `anchor_summary.json`
  D  row invariants for the anchor-tested phases (mask count, candidate identity, queue/pending
     relation, distinct-next-action count, zero freeze failures)
  E  independent recompute of the committed-vehicle census and the queue/pending identity from
     the source phase files
  F  the synthetic transaction result is internally consistent and all its checks pass

Usage: python verify_anchor.py --out verification.json        (refuses to overwrite)
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
OUT = Path(__file__).resolve().parent
SAA = ROOT / 'research/p2_c_opp_01_20261019' / 'phases.json'
CTRL = ROOT / 'research/p2_c_ctrl_01_20261019' / 'phases.json'
ENGINE = ROOT / 'scripts' / 'simulation' / 'strict_online_env.py'

sys.path.insert(0, str(ROOT / 'scripts'))
from project_paths import configure_imports                       # noqa: E402
configure_imports(('experimental', 'simulation', 'coldchain', 'evaluation', 'training'))
import p2_pos01_reference as ref                                   # noqa: E402
from p2_pos01c_gap import view_from_json                           # noqa: E402

TESTED = ('ANCHOR_LEGAL_MULTI', 'ANCHOR_LEGAL_SAME_NEXT', 'NO_LEGAL_CANDIDATE')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=str(ROOT), capture_output=True)
    if r.returncode != 0:
        raise SystemExit('git %s failed: %s' % (' '.join(args), r.stderr.decode('utf-8', 'replace')))
    return r.stdout


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, got=None, want=None, detail=None):
        self.items.append({'check': name, 'status': 'PASS' if ok else 'FAIL',
                           'ok': bool(ok), 'got': got, 'want': want, 'detail': detail})

    @property
    def all_ok(self):
        return all(i['ok'] for i in self.items)


def check_a(ck, reg):
    bad = []
    for rel, want in sorted(reg['inputs'].items()):
        p = WORKSPACE / rel
        got = sha_file(p) if p.is_file() else 'MISSING'
        if got != want:
            bad.append({'path': rel, 'want': want, 'got': got})
    ck.add('A_input_identity', not bad, got=len(reg['inputs']) - len(bad),
           want=len(reg['inputs']), detail=bad or 'all registered inputs match')


def check_b(ck, reg):
    head = git('rev-parse', 'HEAD').decode().strip()
    ck.add('B1_head_is_registered_baseline', head == reg['baseline'], got=head,
           want=reg['baseline'])
    blob = git('cat-file', 'blob',
               ':' + str(ENGINE.relative_to(WORKSPACE)).replace('\\', '/'))
    blob_sha = sha_bytes(blob)
    work_sha = sha_file(ENGINE)
    ck.add('B2_engine_worktree_equals_head_blob', blob_sha == work_sha,
           got={'worktree': work_sha, 'head_blob': blob_sha},
           want='equal', detail='strict_online_env.py was not modified by this work package')


def read_rows():
    with (OUT / 'anchor_scope.csv').open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('queue_len', 'pending_len', 'n_masks', 'n_raw_candidates',
                  'n_legal_candidates', 'n_next_action_after_anchor', 'n_committed'):
            if r[k] not in (None, ''):
                r[k] = int(r[k])
        if r.get('freeze_failures'):
            r['freeze_failures'] = json.loads(r['freeze_failures'])
        if r.get('next_actions'):
            r['next_actions'] = json.loads(r['next_actions'])
    return rows


def recompute_summary(rows):
    verdict_counts = dict(Counter(r['verdict'] for r in rows))
    per_arm = {}
    for arm in sorted({r['arm'] for r in rows}):
        sub = [r for r in rows if r['arm'] == arm]
        tested = [r for r in sub if r['verdict'] in TESTED]
        per_arm[arm] = {
            'phases': len(sub),
            'verdict_counts': dict(Counter(r['verdict'] for r in sub)),
            'anchor_tested_phases': len(tested),
            'phases_with_ge2_legal_candidates': sum(1 for r in tested
                                                    if r['n_legal_candidates'] >= 2),
            'phases_with_ge2_next_actions': sum(1 for r in tested
                                                if r['n_next_action_after_anchor'] >= 2),
            'total_legal_candidates': sum(r['n_legal_candidates'] for r in tested),
            'freeze_failure_total': sum(sum((r['freeze_failures'] or {}).values())
                                        for r in tested),
        }
    h1 = ('ANCHOR_SCOPE_ESTABLISHED'
          if any(v['phases_with_ge2_legal_candidates'] > 0 for v in per_arm.values())
          else 'ANCHOR_SCOPE_NOT_ESTABLISHED')
    return {'phases': len(rows), 'verdict_counts': verdict_counts,
            'per_arm': per_arm, 'h1_verdict': h1}


def check_c(ck, summary, recomp):
    mism = []
    if summary['phases'] != recomp['phases']:
        mism.append({'field': 'phases', 'json': summary['phases'], 'csv': recomp['phases']})
    if summary['verdict_counts'] != recomp['verdict_counts']:
        mism.append({'field': 'verdict_counts', 'json': summary['verdict_counts'],
                     'csv': recomp['verdict_counts']})
    if summary['h1_verdict'] != recomp['h1_verdict']:
        mism.append({'field': 'h1_verdict', 'json': summary['h1_verdict'],
                     'csv': recomp['h1_verdict']})
    for arm, want in recomp['per_arm'].items():
        got = summary['per_arm'].get(arm, {})
        for k, v in want.items():
            if got.get(k) != v:
                mism.append({'field': '%s.%s' % (arm, k), 'json': got.get(k), 'csv': v})
    ck.add('C_csv_json_agreement', not mism, got='recomputed from anchor_scope.csv',
           want='anchor_summary.json', detail=mism or 'identical')


def check_d(ck, rows):
    tested = [r for r in rows if r['verdict'] in TESTED]
    problems = []
    for r in tested:
        tag = '%s d%s ord%s' % (r['arm'], r['day'], r['ordinal'])
        q = r.get('queue_len')
        if r['n_committed'] != 1:
            problems.append([tag, 'n_committed', r['n_committed']])
        if not (q and q >= 2):
            problems.append([tag, 'queue_len', q])
        if r['n_masks'] != q * (q - 1) // 2:
            problems.append([tag, 'n_masks', r['n_masks'], 'want', q * (q - 1) // 2])
        if r['n_legal_candidates'] != r['n_raw_candidates']:
            problems.append([tag, 'candidate_filtering', r['n_legal_candidates'],
                             r['n_raw_candidates']])
        if r['n_next_action_after_anchor'] != len(r['next_actions']):
            problems.append([tag, 'next_action_count', r['n_next_action_after_anchor'],
                             len(r['next_actions'])])
        if r['n_next_action_after_anchor'] < 2:
            problems.append([tag, 'next_actions_lt_2', r['n_next_action_after_anchor']])
        if sum((r['freeze_failures'] or {}).values()) != 0:
            problems.append([tag, 'freeze_failures', r['freeze_failures']])
        if r['verdict'] != 'ANCHOR_LEGAL_MULTI':
            problems.append([tag, 'verdict', r['verdict']])
    ck.add('D_row_invariants', not problems, got=len(tested),
           want='>=1 anchor-tested phase', detail=problems or 'all invariants hold')
    ck.add('D_queue_is_pending_minus_one',
           all(r['pending_len'] == r['queue_len'] + 1 for r in tested), got=None,
           want='pending_len == queue_len + 1 for every anchor-tested phase',
           detail=[[r['arm'], r['day'], r['ordinal'], r['pending_len'], r['queue_len']]
                   for r in tested if r['pending_len'] != r['queue_len'] + 1])


def check_e(ck, rows):
    """Recount committed vehicles and the queue length straight from the saved phases."""
    by_key = {(r['arm'], int(r['day']), int(r['ordinal'])): r for r in rows}
    mism = []
    n_phases = 0
    census = Counter()
    for arm, path in (('SAA', SAA), ('MYOPIC', CTRL)):
        for ph in load(path):
            n_phases += 1
            view = view_from_json({'public_view': ph['public_view'],
                                   'public_identity': ph['public_identity']})
            committed = [v for v in view.vehicles
                         if v.status == 'committed' and v.committed_next not in (None, 0)]
            census[(arm, len(committed) > 1 and 'MULTI' or len(committed))] += 1
            plans = ref.plans_from_view(view)
            key = (arm, int(ph['day']), int(ph['ordinal']))
            r = by_key.get(key)
            if r is None:
                mism.append([key, 'row_missing'])
                continue
            if int(r['n_committed']) != len(committed):
                mism.append([key, 'n_committed', r['n_committed'], len(committed)])
            if len(committed) == 1:
                v = committed[0]
                base = plans[int(v.vehicle_id)]
                pending = ({int(x) for x in view.accepted}
                           - {int(x) for x in view.served})
                queue = [c for c in base.suffix if c in pending]
                if int(r['pending_len']) != len(pending):
                    mism.append([key, 'pending_len', r['pending_len'], len(pending)])
                if int(r['queue_len']) != len(queue):
                    mism.append([key, 'queue_len', r['queue_len'], len(queue)])
                if [int(r['committed_next'])] != [int(v.committed_next)]:
                    mism.append([key, 'committed_next', r['committed_next'],
                                 int(v.committed_next)])
    ck.add('E_independent_census', not mism, got=n_phases, want=289,
           detail=mism[:20] or 'committed census / pending / queue recomputed identically')
    ck.add('E_phase_count', n_phases == 289, got=n_phases, want=289)
    ck.add('E_no_multiple_committed_vehicles',
           not any(k[1] == 'MULTI' for k in census), got=dict(
               ('%s:%s' % (k[0], k[1]), v) for k, v in sorted(census.items(), key=str)),
           want='no phase has 2+ committed vehicles')


def check_f(ck, syn):
    ck.add('F1_all_synthetic_checks_pass', syn['all_checks_pass'],
           got=syn['checks'], want='every registered synthetic check true')
    ck.add('F2_engine_not_modified', syn['engine_modified'] is False,
           got=syn['engine_modified'], want=False)
    ck.add('F3_no_source_day', syn['source_day_used'] is False,
           got=syn['source_day_used'], want=False)
    arms = syn['arms']
    ck.add('F4_maintain_exact_once', arms['maintain']['exact_once'] is True,
           got=arms['maintain']['served_set'], want='all customers exactly once')
    static = arms['write_on_reveal_only']
    dropped = {c for d in static['dropped_write_log'] for c in d['tail']}
    ck.add('F5_static_write_loses_orders',
           (not static['exact_once']) and set(static['missing_orders']) <= dropped,
           got={'missing': static['missing_orders'], 'dropped_tail': sorted(dropped)},
           want='missing orders are a subset of the tail destroyed at the reveal')
    late = arms['write_on_reveal_only_late']
    ck.add('F6_late_reveal_control_exact_once',
           late['exact_once'] is True and late['n_dropped_writes'] == 0,
           got={'exact_once': late['exact_once'],
                'dropped': late['n_dropped_writes']}, want='True / 0')
    ck.add('F7_reconciliation_is_necessary',
           arms['maintain_no_reconcile']['duplicate_serves'] != [],
           got=arms['maintain_no_reconcile']['duplicate_serves'],
           want='non-empty duplicate list without head reconciliation')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'verification.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite: %s already exists' % a.out)

    reg = load(OUT / 'registration.json')
    summary = load(OUT / 'anchor_summary.json')
    syn = load(OUT / 'synthetic_tx.json')
    rows = read_rows()
    recomp = recompute_summary(rows)

    ck = Checks()
    check_a(ck, reg)
    check_b(ck, reg)
    check_c(ck, summary, recomp)
    check_d(ck, rows)
    check_e(ck, rows)
    check_f(ck, syn)

    artefacts = {}
    for name in ('registration.json', 'anchor_scope.csv', 'anchor_summary.json',
                 'synthetic_tx.json', 'probe_anchor.py', 'synthetic_tx.py', 'verify_anchor.py'):
        p = OUT / name
        artefacts[name] = sha_file(p) if p.is_file() else 'MISSING'
    artefacts['scripts/simulation/strict_online_env.py'] = sha_file(ENGINE)

    payload = {
        'schema': 'p2_c_anchor_01_verification_v1',
        'run_id': reg['run_id'],
        'baseline': reg['baseline'],
        'recomputed': recomp,
        'n_checks': len(ck.items),
        'n_ok': sum(1 for i in ck.items if i['ok']),
        'all_ok': ck.all_ok,
        'checks': ck.items,
        'artefact_sha256': artefacts,
    }
    a.out.open('x', encoding='utf-8', newline='\n').write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

    for i in ck.items:
        print('%-5s %s' % (i['status'], i['check']))
    print('%d/%d checks ok' % (payload['n_ok'], payload['n_checks']))
    print('recomputed', json.dumps(recomp, sort_keys=True))
    sys.exit(0 if ck.all_ok else 3)


if __name__ == '__main__':
    main()
