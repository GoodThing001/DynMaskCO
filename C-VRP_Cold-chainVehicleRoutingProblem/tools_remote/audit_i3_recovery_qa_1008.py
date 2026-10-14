"""Independent audit of downloaded v4 engineering QA; no online gain claim."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/i3_registered_1008_v4'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    qa = read(OUT / 'QA_PASS.json')
    trained = OUT / 'engineering_actual_teacher_training'
    report, card = read(trained / 'train_report.json'), read(trained / 'config.json')
    assert (OUT / 'qa.exit').read_text().splitlines() == ['RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1']
    seal = (OUT / 'qa_verify_end.txt').read_text().splitlines()
    assert len(seal) == len((OUT / 'SOURCE_MANIFEST.sha256').read_text().splitlines()) == 332
    assert all(line.endswith(': OK') for line in seal)
    log = (OUT / 'qa.log').read_text()
    for evidence in ['ALL_PASS_REGISTERED_I3_RELEASE=1', 'empty/duplicate/wrong seeds',
                     'sparse 174-ID frozenset typed roundtrip identity',
                     'real I3/reconstruction-control/warmstart/cond online adapter']:
        assert evidence in log, evidence
    assert qa['engineering_release_pass'] and qa['source_stable']
    assert report['qualification'] == card['qualification'] == 'engineering'
    assert report['steps_completed'] == len(report['logs']) == 3
    assert sorted(row['step'] for row in report['logs']) == [1, 2, 3]
    assert report['source_stable'] and report['source_start'] == report['source_end']
    assert report['checkpoint_reload_verified'] and card['training_complete']
    assert report['initial_trainable_hash'] != report['final_trainable_hash']
    previous = report['initial_trainable_hash']
    for row in report['logs']:
        assert row['trainable_before'] == previous and row['trainable_after'] != previous
        previous = row['trainable_after']
    assert previous == report['final_trainable_hash']
    assert sum(row['actor_update'] for row in report['logs']) == 2
    assert sum(not row['teacher_valid'] for row in report['logs']) == 1
    model_sha = sha(trained / 'model.bin')
    assert model_sha == qa['model_sha256'] == report['model_bin_sha256']
    old = ROOT / 'results/i3_registered_1008_v3/engineering_actual_teacher_training'
    assert model_sha == read(ROOT / 'results/i3_registered_1008_v3/QA_PASS.json')['model_sha256']
    assert (trained / 'model.bin').read_bytes() == (old / 'model.bin').read_bytes()
    assert report['frozen_hash'] == read(old / 'train_report.json')['frozen_hash']
    protocol = ROOT / 'configs/i3_20261008_v1/protocol.json'
    assert protocol.read_bytes() == (OUT / 'protocol.json').read_bytes()
    assert read(OUT / 'teacher_collection_config.json')['collection'] == read(protocol)['teacher_collection']
    risk = read(ROOT / 'results/i3_teacher_risk_1008_v2/ENGINEERING_PASS.json')
    assert risk['real_teacher_diagnostic_pass'] and (risk['states'], risk['evaluated_states']) == (3, 2)
    assert all(not obj['online_gain_established'] for obj in (qa, report, risk))
    result = {'engineering_release_verified': True, 'source_end_seal_items': len(seal),
        'engineering_steps': 3, 'actor_updates': 2, 'invalid_teacher_reconstruction_updates': 1,
        'model_sha256': model_sha, 'model_bytes_identical_to_pre_identity_fix_engineering_qa': True,
        'frozen_branch_hash_matches_previous_independently_verified_qa': True,
        'checkpoint_reload_verified': True, 'protocol_sha256': sha(protocol),
        'formal_training_completed': False, 'online_gain_established': False,
        'limits': ['Three engineering updates, not 192-step formal training',
                   'Serialized model equality and full hash check; no fresh local training performed',
                   'Formal teacher shards still collecting; no completed 128-state teacher audit']}
    with (OUT / 'INDEPENDENT_QA_RELEASE_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))
