"""Independent local audit of downloaded release QA; never a method-gain claim."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from diag_a3_adjudicate import _check_rows


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    release = ROOT / 'results/i3_registered_1008_v3'
    qa, report = read(release / 'QA_PASS.json'), read(release / 'engineering_actual_teacher_training/train_report.json')
    assert release.joinpath('qa.exit').read_text().splitlines() == ['RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1']
    checks = release.joinpath('qa_verify_end.txt').read_text().splitlines()
    assert len(checks) == len(release.joinpath('SOURCE_MANIFEST.sha256').read_text().splitlines())
    assert all(row.endswith(': OK') for row in checks)
    assert 'ALL_PASS_REGISTERED_I3_RELEASE=1' in release.joinpath('qa.log').read_text()
    assert 'empty/duplicate/wrong seeds' in release.joinpath('qa.log').read_text()
    assert report['qualification'] == 'engineering' and report['steps_completed'] == 3
    assert report['source_start'] == report['source_end'] and report['source_stable']
    assert report['checkpoint_reload_verified'] and not report['online_gain_established']
    assert report['initial_trainable_hash'] != report['final_trainable_hash']
    assert sum(row['actor_update'] for row in report['logs']) == 2
    assert sum(not row['teacher_valid'] for row in report['logs']) == 1
    assert qa['model_sha256'] == report['model_bin_sha256'] == sha(release / 'engineering_actual_teacher_training/model.bin')
    assert qa['model_sha256'] == read(ROOT / 'results/i3_registered_1008_v2/QA_PASS.json')['model_sha256']
    assert not qa['formal_training'] and not qa['online_gain_established']
    protocol = ROOT / 'configs/i3_20261008_v1/protocol.json'
    assert sha(protocol) == sha(release / 'protocol.json')
    revision = read(ROOT / 'tools_remote/i3_release_1008_v2/qualification_revision.json')
    assert not revision['scientific_parameters_changed'] and revision['all_actor_teacher_optimizer_online_driver_bytes_unchanged']
    replay_folder = ROOT / 'results/i3_teacher_200_archive_replay_1008'
    replay = read(replay_folder / 'independent_replay.json')
    assert replay['engineering_archive_replay_pass'] and replay['independently_replayed_branches'] == 6
    assert replay_folder.joinpath('run.exit').read_text().splitlines() == ['START_EXIT=0', 'RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1']
    baseline_folder = ROOT / 'results/a3_baseline_full-cond_20260926_1007_230001'
    baseline = read(baseline_folder / 'gate.json')
    assert baseline_folder.joinpath('run.exit').read_text().splitlines() == ['RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1']
    ident = baseline['identity']
    assert ident['source_stable'] and ident['source_sha256'] == ident['source_end_sha256']
    assert ident['seeds']['gate'] == 20260926
    assert ident['shared_source_sha256']['scripts/evaluation/scenario_saa.py'] == '36e08e06ce7b8e44eee2e6ff778904c1f26bd3145eb7d2072fb7360f33732318'
    for tier in baseline['gate'].values():
        assert not _check_rows(tier['per_day']['cond_hist'], 40)
    result = {'engineering_release_verified': True, 'release': str(release.relative_to(ROOT)),
        'qa_end_seal_checks': len(checks), 'actual_teacher_training_steps': 3,
        'actor_updates': 2, 'invalid_teacher_reconstruction_updates': 1,
        'actual_model_bin_sha256': qa['model_sha256'], 'v2_v3_engineering_model_bytes_identical': True,
        'teacher_200_independently_replayed_branches': 6,
        'protocol_sha256': sha(protocol), 'scientific_parameters_changed': False,
        'first_same_source_baseline_complete_hard_finite_days_per_tier': 40,
        'a3_complete_adjudication_still_pending': True, 'formal_i3_training': False,
        'maskco_online_gain_established': False, 'final_test_run': False,
        'limitations': ['Three engineering updates on three fixed states, not formal I3 training',
            'Fixed-state teacher risk is not full online-return gradient',
            'Teacher objective holdout does not imply unseen-day generalization',
            'Full heldout-model risk evaluation and same-weight iteration remain pending']}
    path = ROOT / 'results/a3_repair_chain_1007_2250/I3_REGISTERED_RELEASE_AUDIT_1008.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    audit()
