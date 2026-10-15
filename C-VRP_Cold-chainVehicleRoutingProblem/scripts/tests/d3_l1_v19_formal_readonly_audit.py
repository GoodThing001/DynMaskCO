"""Independent 64-slot formal collection audit. Read-only; no rollouts or F scoring."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
for part in ('scripts/tests', 'scripts/simulation', 'scripts/coldchain'):
    sys.path.insert(0, str(EXT / part))
from d3_l1_v18_readonly_audit import typed_bytes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--runtime-identity', required=True)
    ap.add_argument('--authorization', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    import numpy as np
    from recourse_snapshot import snapshot_state_hash, restore_recourse_snapshot
    ident = json.loads(Path(args.runtime_identity).read_text(encoding='utf-8'))
    auth = json.loads(Path(args.authorization).read_text(encoding='utf-8'))
    rows, errors = [], []
    def ck(name, ok, detail=None):
        rows.append({'name': name, 'ok': bool(ok), 'detail': detail})
    try:
        ledger = [json.loads(line) for line in (root / 'STATE_LEDGER.jsonl').read_text().splitlines()]
        manifest = [json.loads(line) for line in (root / 'STATE_MANIFEST.jsonl').read_text().splitlines()]
        result = json.loads((root / 'SUPERVISOR_RESULT.json').read_text())
        data_path = EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'
        ck('dataset_bytes_pinned', sha(data_path) == ident['dataset_sha256'] == auth['dataset_sha256'])
        data = dict(np.load(data_path))
        for name, rec in ident['modules'].items():
            ck('source:' + name, sha(EXT / rec['path']) == rec['sha256'])
        ck('formal_scope', result['purpose'] == 'formal' and result['slots_in_denominator'] == 64)
        ck('formal_summary_pass', result['verdict'] == 'BATCH_COMPLETE')
        ck('batch_budget', result['batch_wall_s'] <= 3600.)
        ck('no_model', result['model_identity'] == 'F_FREE_NO_MODEL')
        ck('source_stable', result['source_stable'] is True and all(
            row['source_stable'] for row in result['source_stability'].values()))
        ck('human_authorization_binding', result['authorization_sha256'] == sha(args.authorization)
           and auth['runtime_identity_sha256'] == sha(args.runtime_identity))
        token = json.loads(Path(args.authorization).with_suffix('.CONSUMED.json').read_text())
        ck('one_pass_token_binding', token['authorization_sha256'] == sha(args.authorization)
           and token['run_id'] == result['run_id'] and token['one_pass_only'] is True)
        for prefix in ('SUPERVISOR_IDENTITY_START', 'SUPERVISOR_IDENTITY_END'):
            state = json.loads((root / (prefix + '.json')).read_text())
            ck(prefix, state['ok'] and state['runtime_identity_sha256'] == sha(args.runtime_identity))
        published = []
        failed = []
        for day in range(16):
            dayrows = [r for r in ledger if r.get('day_id') == day]
            starts = [r for r in dayrows if r['event'] == 'DAY_ROLLOUT_STARTED']
            finishes = [r for r in dayrows if r['event'] == 'DAY_ROLLOUT_FINISHED']
            ck('day%d:one_rollout' % day, len(starts) == len(finishes) == 1)
            if starts and finishes:
                ck('day%d:rollout_budget' % day, finishes[0]['mono'] - starts[0]['mono'] <= 120.)
            for phase in ('START', 'END'):
                state = json.loads((root / ('WORKER_IDENTITY_%s_DAY%02d.json' % (phase, day))).read_text())
                ck('day%d:identity_%s' % (day, phase), state['ok']
                   and state['runtime_identity_sha256'] == sha(args.runtime_identity)
                   and {n: r['sha256'] for n, r in state['identity']['modules'].items()}
                   == {n: r['sha256'] for n, r in ident['modules'].items()})
            reveal_times = sorted(set(float(x) for x in data['reveal_time'][day] if x > 0))
            for pos in range(1, 5):
                pre = 'day%d/slot%d:' % (day, pos)
                slot = [r for r in dayrows if r.get('position') == pos]
                attempts = [r for r in slot if r['event'] == 'ATTEMPT_STARTED']
                pubs = [r for r in slot if r['event'] == 'PUBLISHED']
                done = [r for r in slot if r['event'] == 'SLOT_FINISHED']
                terminals = [r for r in slot if r['event'] in
                             ('PUBLISHED', 'FAILED', 'NOT_ATTEMPTED_DUE_TO_TERMINATION')]
                ck(pre + 'one_terminal', len(terminals) == 1)
                if len(pubs) != 1 or terminals[-1]['event'] != 'PUBLISHED':
                    failed.append({'day_id': day, 'position': pos, 'records': slot})
                    ck(pre + 'qualified', False, 'failed or uncompleted; retained in denominator')
                    continue
                pub = pubs[0]
                published.append((day, pos))
                ck(pre + 'wal_before_publication', len(attempts) == 1 and attempts[0]['mono'] < pub['mono'])
                ck(pre + 'durable_and_budget', len(done) == 1 and done[0]['mono'] >= pub['mono']
                   and 0 <= done[0]['slot_span_wall_s'] <= 60.)
                entries = [r for r in manifest if (r['day_id'], r['position']) == (day, pos)]
                ck(pre + 'manifest_one', len(entries) == 1)
                snap_path = root / pub['snapshot_path']
                ck(pre + 'snapshot_sha_bytes', sha(snap_path) == pub['snapshot_sha256']
                   and snap_path.stat().st_size == pub['bytes'])
                snap = pickle.loads(snap_path.read_bytes())
                restore_recourse_snapshot(snap)
                ck(pre + 'state_hash', snapshot_state_hash(snap) == snap['state_hash'] == pub['state_hash'])
                ck(pre + 'contract', snap['coldchain_contract_hash'] == ident['effective_contract_hash'])
                ck(pre + 'R1_first_four_reveals', snap['instance_id'] == day
                   and typed_bytes(snap['clock']) == typed_bytes(reveal_times[pos - 1]))
                visible = np.asarray(data['reveal_time'][day] <= snap['clock'] + 1e-6, dtype=bool)
                visible[0] = True
                ck(pre + 'visible_mask', visible.tobytes() == snap['visible_mask'].tobytes())
                validation_path = root / pub['validation_path']
                ck(pre + 'validation_sha', sha(validation_path) == pub['validation_sha256'])
                ev = pickle.loads(validation_path.read_bytes())
                for title, a, b in (
                    ('full_trace_exact', ev['original_traces'], ev['resumed_traces']),
                    ('served_exact', ev['original_served'], ev['resumed_served']),
                    ('serialization_exact', snap, ev['reloaded_snapshot']),
                    ('recapture_exact', snap, ev['recaptured_snapshot'])):
                    ck(pre + title, typed_bytes(a) == typed_bytes(b))
                p, q = ev['original_projection'], ev['perturbed_projection']
                ck(pre + 'two_independent_calls', p['replanner_calls'] == q['replanner_calls'] == 1)
                ck(pre + 'no_projection_errors', not p['errors'] and not q['errors'])
                for key in ('visible_inputs_sha', 'legal_candidates_sha', 'planned_suffix_sha',
                            'planned_suffix_sha_boundary'):
                    ck(pre + key, p[key] is not None and p[key] == q[key])
                ck(pre + 'valid_perturbation', pub['n_invisible_modified'] > 0)
                ck(pre + 'reported_call_budget', pub['validation']['replanner_calls'] == 2
                   and pub['validation']['capture_calls'] == 1
                   and pub['validation']['explicit_restore_calls'] == 1
                   and pub['validation']['hash_calls'] == 2)
            print('verified day %d/16' % (day + 1), flush=True)
        ck('64_unique_registered_slots', len(published) == len(set(published)) == 64)
        ck('formal_summary_count', result['formal_slots_collected'] == result['slots_published'] == len(published))
    except Exception as exc:
        errors.append('%s: %s' % (type(exc).__name__, exc))
        failed = locals().get('failed', [])
        published = locals().get('published', [])
    report = {'classification': 'independent_readonly_formal_64_slot_audit', 'rows': rows,
        'n_pass': sum(r['ok'] for r in rows), 'n_fail': sum(not r['ok'] for r in rows) + len(errors),
        'qualified_slots': len(published), 'failed_or_missing_slots': failed, 'errors': errors,
        'denominator': 64, 'runtime_identity_sha256': sha(args.runtime_identity),
        'authorization_sha256': sha(args.authorization), 'auditor_sha256': sha(__file__),
        'algorithmic_gain_claim': 'none', 'stable_online_gain': False, 'argv': sys.argv}
    with open(args.out, 'x', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps({k: report[k] for k in ('n_pass', 'n_fail', 'qualified_slots', 'errors')}))
    return int(report['n_fail'] != 0)


if __name__ == '__main__':
    sys.exit(main())
