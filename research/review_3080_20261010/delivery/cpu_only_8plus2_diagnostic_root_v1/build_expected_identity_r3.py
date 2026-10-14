"""Generate EXPECTED_IDENTITY_r3.json: the fail-closed pre-launch identity manifest.

The manifest is written AFTER every r3 artifact is final, and the supervisor refuses to
launch the child unless every entry matches.  The manifest itself is not hashed (it is the
root of trust for one run); its sha256 is recorded by the supervisor record instead.
"""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[3]

PACKAGE_FILES = [
    'server_diagnostic_r2.py',
    'supervise_diagnostic_r2.py',
    'unified_closure_protocol_r2.py',
    'REGISTRATION_r3.json',
]
ROOT_FILES = [
    'tools/actual_loaded_closure_candidate_r21.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/mpre.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/cvrptw_utils.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/project_paths.py',
    'MASKCO_code/ckpts/cvrp100.ckpt',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {
        'attempt_id': 'delivery_cpu_only_8plus2_engineering_diagnostic_r3',
        'purpose': 'pre-launch identity gate for run_r3; the supervisor refuses to launch on any mismatch',
        'child': 'server_diagnostic_r2.py',
        'files': {},
        'root_files': {},
    }
    for rel in PACKAGE_FILES:
        path = PACKAGE / rel
        assert path.is_file(), path
        manifest['files'][rel] = sha(path)
    for rel in ROOT_FILES:
        path = ROOT / rel
        assert path.is_file(), path
        manifest['root_files'][rel] = sha(path)
    out = PACKAGE / 'EXPECTED_IDENTITY_r3.json'
    assert not out.exists(), out
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8', newline='\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
