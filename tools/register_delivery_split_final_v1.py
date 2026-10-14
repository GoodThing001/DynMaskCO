"""Seal the one exclusive 8+2 request package for the fixed split-final head."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1/downloaded'
NEW = ROOT / 'research/review_3080_20261010/delivery/split_final_timed_execution_root_v1'
R12 = ROOT / 'research/review_3080_20261010/goal_round_20261013_r12_learned_proposal_and_split_head'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (NEW / 'LOCAL_EXECUTION_BINDING.json').exists()
    reg = json.loads((OLD / 'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    assert reg['requests'] == [{'arm': 'pretrained_initial', 'fault_stage': 0}, {'arm': 'random888_initial', 'fault_stage': 0}]
    assert (reg['total_request_seconds'], reg['child_seconds'], reg['reap_reserve_seconds']) == (10, 8, 2)
    assert reg['input_sha256'] == sha(NEW / 'CHAIN_INPUT.json')
    old = reg['package_files']
    changed = []
    package = {}
    for name, prior in old.items():
        path = NEW / name
        assert path.is_file() and path.suffix in ('.py', '.json')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'))
        current = sha(path)
        if current != prior:
            changed.append(name)
        package[name] = current
    assert changed == ['delivery_score_split_engine_v1.py'], changed
    split = json.loads((R12 / 'DELIVERY_SPLIT_HEAD_RESULT.json').read_text(encoding='utf-8'))
    assert split['strict_equal'] and len(split['rows']) == 34 and split['total_different_elements'] == 0
    reg.update({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'NEW_SPLIT_FINAL_HEAD_COMPLETE_TIMED_REQUEST_TWO_FIXED_INITIAL_ARMS',
        'selection_rule': 'One new split-before-final head candidate; fixed key12,3 and initial arms; original 8+2 and all C0/source/parameter gates unchanged; no old request replay or benefit selection.',
        'remote_directory': 'research/review_3080_20261010/split_final_timed_execution_root_v1',
        'package_files': package,
        'transport_tool_sha256': sha(ROOT / 'tools/transport_delivery_split_final_v1.py'),
        'prior_timed_request_result_sha256': sha(OLD / 'RESULT.json'),
        'prospective_split_head_registration_sha256': sha(R12 / 'DELIVERY_SPLIT_HEAD_REGISTRATION.json'),
        'saved_input_strict_result_sha256': sha(R12 / 'DELIVERY_SPLIT_HEAD_RESULT.json'),
        'candidate_method': 'first layer at fixed 750 rows, slice n rows before final output layer',
        'saved_input_gate': '34/34 exact float32 bit patterns, 21148 actions',
        'model_execution_authorized': True,
        'execution_authorized': True,
        'old_QA_repeated': False,
        'full_request_budget_qualified': False,
        'learned_output_qualified': False,
        'scientific_family3_started': False,
        'stable_online_gain': False,
    })
    (NEW / 'EXECUTION_REGISTRATION.json').write_text(json.dumps(reg, indent=2) + '\n', encoding='utf-8')
    binding = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'registration_sha256': sha(NEW / 'EXECUTION_REGISTRATION.json'),
               'only_changed_package_file': changed[0],
               'new_engine_sha256': sha(NEW / changed[0]),
               'prior_timed_request_result_sha256': reg['prior_timed_request_result_sha256'],
               'resource_must_be_rechecked_immediately_before_upload': True,
               'stable_online_gain': False}
    (NEW / 'LOCAL_EXECUTION_BINDING.json').write_text(json.dumps(binding, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'package_files': len(package), 'changed': changed, 'registered': binding['registration_sha256']}))


if __name__ == '__main__':
    main()
