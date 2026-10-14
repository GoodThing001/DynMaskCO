"""Read-only reproducible search and stored-artifact engineering audit.
No model imports, pickle execution, source edits or numerical research runs.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def decode(x):
    if isinstance(x, dict):
        if '__mapping__' in x: return {decode(k): decode(v) for k, v in x['__mapping__']}
        if '__sequence__' in x: return [decode(v) for v in x['values']]
        # Do not instantiate dataclasses or arrays from archived logs.
        return {k: decode(v) for k, v in x.items()}
    if isinstance(x, list): return [decode(v) for v in x]
    return x


def run(root, out):
    review = root / 'research/review_3080_20261010'
    ext = root / 'C-VRP_Cold-chainVehicleRoutingProblem'
    pins, checks = {}, []
    def get(rel):
        p = root / rel
        pins[rel] = {'sha256': sha(p), 'bytes': p.stat().st_size}
        return decode(json.loads(p.read_text(encoding='utf-8')))
    def check(name, value): checks.append({'name': name, 'pass': value is True})
    scopes = [ext / 'results', root / 'research/pickup_20round_audit',
              review / 'delivery', review / 'pickup']
    targets = [23.4229, 2213.5795, 2.2939]
    text_scanned, raw_joint, near, opaque, parse_errors = 0, [], [], [], []
    excluded = {'source_snapshot', 'selftest_fixtures', 'v3_selftest_fixtures', '__pycache__', '.git', 'parent', 'parent_v2'}
    for base in scopes:
        for p in base.rglob('*'):
            if not p.is_file() or excluded.intersection(p.parts): continue
            rel = p.relative_to(root).as_posix()
            if p.suffix in {'.pkl', '.npz', '.npy', '.ckpt'}:
                new_scope = str(p).startswith(str(review)) or str(p).startswith(str(root / 'research/pickup_20round_audit')) or 'research_20261010_continuation_v2' in p.parts
                opaque.append({'path': rel, 'sha256': sha(p) if new_scope else None, 'bytes': p.stat().st_size,
                               'hash_scope': 'new-work full bytes' if new_scope else 'legacy binary listing only',
                               'decoded': False, 'numeric_combination_binding': 'unknown'})
            if p.suffix not in {'.json', '.jsonl', '.csv', '.txt'}: continue
            text_scanned += 1
            try: text = p.read_text(encoding='utf-8')
            except (UnicodeError, OSError) as exc:
                parse_errors.append({'path': rel, 'error': str(exc)}); continue
            found = [bool(re.search(r'(?<![0-9.])' + str(t).replace('.', r'\.') + r'(?![0-9])', text)) for t in targets]
            if any(found):
                item = {'path': rel, 'sha256': sha(p), 'target_exact_tokens': found}
                near.append(item)
                if all(found): raw_joint.append(item)
    numerical = {'searched_scopes': [str(p.relative_to(root)) for p in scopes],
                 'text_files_scanned': text_scanned, 'exact_target_tokens': targets,
                 'joint_raw_token_candidates': raw_joint, 'individual_matches': near,
                 'opaque_original_artifacts': opaque, 'read_errors': parse_errors,
                 'raw_three_metric_binding_established': False,
                 'absence_proven_globally': False,
                 'conclusion': 'unknown: no joint raw source identified in searched text; opaque files are not semantically decoded',
                 'quoted_arithmetic_only': {'composite': targets[0] + .1 * targets[1] + targets[2],
                   'distance_share_percent': 100 * targets[0] / (targets[0] + .1 * targets[1] + targets[2])}}
    base = 'research/review_3080_20261010/delivery/remote_actual_bundle_v2/downloaded/'
    full = get(base + 'ACTUAL_FULLSTEP_QA.json')
    before, after = full['before']['leaves'], full['after']['leaves']
    check('3D actual234 inventory rows', len(before) == len(after) == 234)
    check('3D shape element sum76259840', sum(math.prod(v['shape']) for v in before) == 76259840)
    check('3D stored before/after same paths', [v['path'] for v in before] == [v['path'] for v in after])
    check('3D source ledgers equal', full['source_start'] == full['source_end'])
    check('3D one fixed step', type(full['trained_fixed_steps']) is int and full['trained_fixed_steps'] == 1)
    check('3D engineering flags cannot claim scientific', full['research_model_eligible'] is False and full['stable_online_gain_established'] is False)
    for name, count in [('full', 234), ('encoder', 160), ('decoder', 60)]:
        g = full['gradient_groups'][name]
        check('3D stored gradient stats ' + name, g['leaves'] == g['leaves_finite'] == g['leaves_nonzero'] == count)
    check('3D stored output checkpoint nonempty hash', len(full['saved_sha256']) == 64)
    check('3D config null softcap', full['model_config_loaded']['softcap'] is None and full['model_config_loaded']['final_softcap'] is None)
    featurebase = 'research/review_3080_20261010/delivery/remote_features_bundle_v1/downloaded/'
    feat = get(featurebase + 'FEATURES_OBJECTIVE_QA.json')
    inp = get(featurebase + 'INPUT_AND_TEACHER_QUALIFICATION.json')
    check('10D field10 names', len(feat['fields']) == 10)
    check('10D both starts identical', len({v['same_initial_state_digest'] for v in feat['arms'].values()}) == 1)
    check('10D source ledgers equal', feat['source_start'] == feat['source_end'])
    check('10D element count independently sum', sum(math.prod(v['shape']) for v in inp['mapped_inventory'].values()) == 76263424)
    unchanged = [k for k, v in inp['original_inventory'].items() if k != "('init_proj', 'kernel')" and inp['mapped_inventory'][k] == v]
    check('10D only projection shape changes, other233 metadata equal', len(unchanged) == 233)
    check('10D registration flags false', feat['research_model_eligible'] is False and feat['dynamic_online_qualified'] is False and feat['stable_gain_established'] is False)
    for name, arm in feat['arms'].items():
        check(name + ' single step integer', type(arm['fixed_steps']) is int and arm['fixed_steps'] == 1)
        check(name + ' objective arithmetic', abs(arm['loss_total'] - arm['loss_CE'] - arm['objective_coefficient'] * arm['loss_preference']) < 5e-7)
        check(name + '234 stored updated leaves', len(arm['parameter_inventory']) == 234)
        check(name + 'stored hard complete projection', arm['projection']['hard'] is True and arm['projection']['complete'] is True)
        check(name + ' retained mask edges stored', arm['retained_edges_preserved'] is True and not arm['remaining'])
        for group, count in [('full',234), ('encoder',160), ('decoder',60)]:
            g = arm['gradient_stats'][group]
            check(name + ' stored gradient '+group, g['leaves'] == g['finite'] == g['nonzero'] == count)
    featureweight = review / 'delivery/remote_features_bundle_v1/downloaded/INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json'
    weight_evidence = {'independent_byte_read_report_available': featureweight.is_file(),
                       'independent_local_weight_loading_established': False}
    if featureweight.is_file(): weight_evidence['actual_read_report'] = get(featureweight.relative_to(root).as_posix())
    pbase = 'research/review_3080_20261010/pickup/actual_masked_fleet_engineering_v1/'
    pactual = get(pbase + 'actual_qa_v1/result.json')
    resident = get(pbase + 'resident_qa_v1/result.json')
    wiring = get(pbase + 'wiring_qa_v1/result.json')
    check('P actual on archived packet only', pactual['known_archived_only'] is True and pactual['formal_online_gain_established'] is False)
    check('P stored full model call', pactual['full_encoder_decoder_used'] is True and pactual['random_params_differ'] is True)
    check('P resident timeout keepsown/reaped stored', resident['timeout_worker_reaped'] is True and resident['timeout_keeps_own'] is True)
    check('P wiring legacy process cap limitation retained', wiring['registration']['legacy_process_kill_qualified'] is False)
    for rel in [base+'RUN_EXIT.json', featurebase+'RUN_EXIT.json']:
        x = get(rel); check(rel + ' exit0', type(x['exit_code']) is int and x['exit_code'] == 0 and x['failure'] is None)
    report = {'kind': 'independent_stored_package_and_scope_audit_v1', 'pins': pins,
              'numeric_source_search': numerical, 'stored_package_checks': checks,
              'stored_package_failures': [v for v in checks if not v['pass']],
              'saved_weight_evidence_scope': weight_evidence,
              'actual_activation_independently_reexecuted_here': False,
              'raw_parameter_gradients_independently_loaded_here': False,
              'raw_model_forward_independently_reexecuted_here': False,
              'stable_online_gain_established': False,
              'full_scientific_qualification_established': False,
              'boundaries': {'3D_elements': 76259840, '10D_elements': 76263424,
                  'runtime_dtype': 'float32 vs stored config bfloat16', 'engineering_steps_each_arm': 1,
                  'D_state_scope': 'one previously observed all-release/ready teacher snapshot',
                  'P_state_scope': 'one public archived packet; model proposal on <=5 mutable orders/two vehicles',
                  'P_budget_gap': 'resident actor kill works as stored QA; legacy synchronous admission still lacks total compute kill',
                  'physics': 'fixed category quality rate; contract target temperature is not measured cargo/air temperature'}}
    out.mkdir(parents=True, exist_ok=True)
    with (out/'ACTUAL_PACKAGE_SCOPE_AUDIT.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'checks': len(checks), 'failures': len(report['stored_package_failures']),
                      'joint_raw_numeric_candidates': len(raw_joint)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); run(a.root, a.out)
