"""Independently verify saved engineering inputs; no planning, scoring or installation."""
import hashlib
import argparse
import json
import pickle
import sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
for part in ('scripts/tests', 'scripts/simulation', 'scripts/coldchain'):
    sys.path.insert(0, str(EXT / part))
from d3_l1_v18_readonly_audit import typed_bytes
import numpy as np
from recourse_snapshot import restore_recourse_snapshot

ROOT = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
E = ROOT / 'v110_evidence'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out=None):
    result = json.loads((E / 'INPUT_QUALIFICATION_V110.json').read_text())
    start = json.loads((E / 'START.json').read_text())
    manifest = json.loads((ROOT / 'v19_evidence/STATE_MANIFEST_64_V19.json').read_text())
    rows = []
    def ck(name, ok):
        rows.append({'name': name, 'ok': bool(ok)})
    ck('denominator_all_64_original_order', [(r['day_id'], r['position']) for r in result['rows']]
       == [(d, p) for d in range(16) for p in range(1, 5)])
    ck('no_science_or_resampling', result['F_forward_calls'] == result['F_D_Geo_comparisons']
       == result['install_calls'] == result['rollouts'] == 0 and result['baseline_calls'] == 128)
    ck('no_F_modules_and_no_science_upgrade', not result['F_module_imports'] and not result['science_ready'])
    ck('start_source_identity', start['sources'] == result['sources_at_start'])
    for name, rec in result['sources_at_start'].items():
        ck('source:' + name, sha(EXT / rec['path']) == rec['sha256']
           == result['sources_at_end'][name]['sha256'])
    for entry, row in zip(manifest, result['rows']):
        prefix = 'day%d/slot%d:' % (row['day_id'], row['position'])
        ck(prefix + 'input_state_identity', entry['state_hash'] == row['state_hash'])
        raw = E / row['derived_path']
        ck(prefix + 'raw_sha256', sha(raw) == row['derived_sha256'])
        saved = pickle.loads(raw.read_bytes())
        snap_path = ROOT / 'v19_evidence/formal_collection' / entry['snapshot_path']
        ck(prefix + 'source_snapshot_sha256', sha(snap_path) == entry['snapshot_sha256'])
        snap = pickle.loads(snap_path.read_bytes())
        before, _, served = restore_recourse_snapshot(snap)
        a, b = saved['original'], saved['hidden_perturbed']
        for key in ('P0', 'deferred', 'pool', 'mask', 'legal', 'inputs', 'mutable_ids', 'protected_ids'):
            ck(prefix + 'exact_' + key, typed_bytes(a[key]) == typed_bytes(b[key]))
        ck(prefix + 'P0_fingerprint', hashlib.sha256(typed_bytes(a['P0'])).hexdigest() == row['P0_sha256'])
        ck(prefix + 'inputs_fingerprint', hashlib.sha256(typed_bytes(a['inputs'])).hexdigest() == row['input_sha256'])
        ck(prefix + 'legal_fingerprint', hashlib.sha256(typed_bytes(a['legal'])).hexdigest() == row['legal_sha256'])
        ck(prefix + 'mask_engineering_first_two_only', a['mask'] == a['pool'][:2] and len(a['mask']) == 2)
        ck(prefix + 'partial_only_removes_mask', all(
            a['partial'][vid].suffix == tuple(c for c in p.suffix if c not in a['mask'])
            for vid, p in a['P0'].items()))
        ck(prefix + 'anchors_prepared_to_clock', all(v.ready_time >= snap['clock']
            for v in a['vehicles'] if v.status in ('idle', 'ready')))
        ck(prefix + 'frozen_vehicles_preserved', all(typed_bytes(v) == typed_bytes(a['vehicles'][v.vehicle_id])
            for v in before if v.status not in ('idle', 'ready')))
        ck(prefix + 'served_immutable', typed_bytes(served) == typed_bytes(a['served']))
        ck(prefix + 'visible_mapping_only', set(a['inputs']['local_to_node']) <= set(a['visible_ids']) | {0})
        ck(prefix + 'shape_action_count', a['inputs']['action_feats'].shape == (1, len(a['legal']), 20))
        ck(prefix + 'finite_model_inputs', all(np.isfinite(a['inputs'][k]).all()
            for k in ('raw_3d', 'node_feats', 'adjmat', 'action_feats')))
        ck(prefix + 'target_range', set(a['inputs']['vehicle_target_vid']) <= set(a['mutable_ids']))
        ck(prefix + 'C0_missing_remains_blocked', row['C0_full_certification'].startswith('NOT_RUN_MISSING_'))
        ck(prefix + 'all_recorded_checks', all(row['checks'].values()) and all(a['checks'].values())
           and all(b['checks'].values()) and row['status'] == 'INPUT_CHECKS_PASS')
    report = {'rows': rows, 'n_pass': sum(r['ok'] for r in rows), 'n_fail': sum(not r['ok'] for r in rows),
              'classification': 'independent_readonly_existing_input_artifact_audit',
              'source_sha256': sha(__file__), 'input_report_sha256': sha(E / 'INPUT_QUALIFICATION_V110.json'),
              'states': 64, 'new_plan_calls': 0, 'F_forward_calls': 0, 'install_calls': 0,
              'science_ready': False, 'stable_online_gain': False}
    target = Path(out) if out else E / 'INDEPENDENT_READONLY_AUDIT_V110.json'
    with open(target, 'x', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    return int(report['n_fail'] != 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', help='New independent output path; never overwrite archived results')
    sys.exit(main(parser.parse_args().out))
