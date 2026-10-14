"""Preserve pre-execution static transport failure and bind actual local helper."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/review_3080_20261010/delivery/split_final_timed_execution_root_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reg_path = BASE / 'EXECUTION_REGISTRATION.json'
    binding_path = BASE / 'LOCAL_EXECUTION_BINDING.json'
    assert not (BASE / 'DEPLOY_ATTEMPT.json').exists()
    assert not (BASE / 'TRANSPORT_STATIC_FAILURE_V1.json').exists()
    reg = json.loads(reg_path.read_text(encoding='utf-8'))
    old = reg['local_transport_helpers']
    drift = {name: {'registered': digest, 'actual': sha(name)} for name, digest in old.items() if sha(name) != digest}
    assert list(drift) == [str(ROOT / 'tools/_srv.py')], drift
    shutil.copyfile(reg_path, BASE / 'EXECUTION_REGISTRATION_V1_STATIC_FAIL.json')
    shutil.copyfile(binding_path, BASE / 'LOCAL_EXECUTION_BINDING_V1_STATIC_FAIL.json')
    (BASE / 'TRANSPORT_STATIC_FAILURE_V1.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'before upload/launch',
        'failed_registration_sha256': sha(BASE / 'EXECUTION_REGISTRATION_V1_STATIC_FAIL.json'),
        'helper_drift': drift, 'server_mutation': False, 'model_executed': False}, indent=2) + '\n', encoding='utf-8')
    reg['created_utc'] = datetime.now(timezone.utc).isoformat()
    reg['registration_revision'] = 'R2: actual _srv.py helper bytes rebound after local static failure; candidate and science parameters unchanged'
    reg['local_transport_helpers'] = {name: sha(name) for name in old}
    reg['prior_static_failure_sha256'] = sha(BASE / 'TRANSPORT_STATIC_FAILURE_V1.json')
    reg_path.write_text(json.dumps(reg, indent=2) + '\n', encoding='utf-8')
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    binding['created_utc'] = datetime.now(timezone.utc).isoformat()
    binding['registration_sha256'] = sha(reg_path)
    binding['prior_static_failure_sha256'] = reg['prior_static_failure_sha256']
    binding_path.write_text(json.dumps(binding, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rebound': len(drift), 'registration_sha256': sha(reg_path), 'server_mutation': False}))


if __name__ == '__main__':
    main()
