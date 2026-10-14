"""Install the reviewed current-source fix, preserving original source/tests."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
LINE = ROOT / 'research/review_3080_20261010/reproduction'
GATE = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/research_claim_gate'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = GATE / 'program_support_audit.py'
    expected = '2871e5b9971c79b3703252b33eb7ec567269786c0f0c1374accec7de9776cfc1'
    if sha(old) != expected:
        raise RuntimeError('Current-source identity changed; do not install stale review')
    backup = LINE / 'installed_v3_source_backup'
    backup.mkdir(exist_ok=False)
    test = GATE / 'test_program_support_audit.py'
    before = {}
    for p in (old, test):
        shutil.copy2(p, backup / p.name)
        before[str(p.relative_to(ROOT))] = sha(p)
    shutil.copy2(LINE / 'program_support_audit_v3.py', old)
    # Assert the checked subset, never the deprecated always-false proof alias.
    source = test.read_text(encoding='utf-8')
    source = source.replace("['necessary_design_conditions_pass']", "['checked_design_subset_pass']")
    test.write_text(source, encoding='utf-8')
    after = {str(p.relative_to(ROOT)): sha(p) for p in (old, test)}
    with (LINE / 'ACTUAL_CURRENT_ENTRY_INSTALL.json').open('x', encoding='utf-8') as f:
        json.dump({'original_sha_expected': expected, 'before': before, 'after': after,
                   'basis': 'latest user four-axis implementation plus narrow tested fixes',
                   'archived_results_modified': False}, f, indent=2)
    print(json.dumps({'installed': True, 'original_preserved': True, 'old_results_modified': False}))


if __name__ == '__main__':
    main()
