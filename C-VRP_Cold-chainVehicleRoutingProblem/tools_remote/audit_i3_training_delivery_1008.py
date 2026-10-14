"""Read-only cross-check of downloaded engineering artifacts, never a paper gate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    optimizer = ROOT / 'results/i3_optimizer_pretrained_probe_1008_v3'
    assert optimizer.joinpath('run.exit').read_text().splitlines() == [
        'RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1']
    checks = optimizer.joinpath('verify_end.txt').read_text().splitlines()
    assert len(checks) == 366 and all(line.endswith(': OK') for line in checks)
    assert 'ALL PASS: actual frozen CVRP checkpoint branch' in optimizer.joinpath('run.log').read_text()
    runner = ROOT / 'results/i3_runner_engineering_server_1008_1110'
    report = read('results/i3_runner_engineering_server_1008_1110/train_report.json')
    card = read('results/i3_runner_engineering_server_1008_1110/config.json')
    assert report['source_start'] == report['source_end'] and report['source_stable']
    assert report['checkpoint_reload_verified'] and report['steps_completed'] == 2
    assert report['initial_trainable_hash'] != report['final_trainable_hash']
    assert report['model_bin_sha256'] == digest(runner / 'model.bin')
    assert card['qualification'] == report['qualification'] == 'engineering'
    assert card['training_complete'] and not report['online_gain_established']
    assert not set(row['day_id'] for row in report['logs']) & set(report['holdout_days'])
    for rel, want in card['model_source_sha256'].items():
        assert digest(ROOT / rel) == want, 'Training delivery changed: ' + rel
    validation = ROOT / 'results/i3_training_runner_probe_1008_1110'
    assert validation.joinpath('run.exit').read_text().splitlines() == ['RUN_EXIT=0', 'DONE=1']
    assert 'ALL_PASS_I3_TRAINING_RUNNER=1' in validation.joinpath('run.log').read_text()
    teacher = ROOT / 'results/i3_teacher_200_archive_1008_1055'
    probe = read('results/i3_teacher_200_archive_1008_1055/probe.json')
    assert digest(teacher / 'private.json') == probe['private_archive']['sha256']
    assert teacher.joinpath('run.exit').read_text().splitlines() == [
        'RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1']
    assert len(teacher.joinpath('verify_end.txt').read_text().splitlines()) == 15
    assert all(line.endswith(': OK') for line in teacher.joinpath('verify_end.txt').read_text().splitlines())
    assert probe['summary']['records'] == 3 and probe['summary']['valid_teachers'] == 2
    assert probe['config']['not_formal_training_config'] is True
    result = {
        'engineering_delivery_verified': True, 'formal_training': False,
        'online_gain_established': False, 'pretrained_probe_sealed_checks': len(checks),
        'training_runner_steps': report['steps_completed'],
        'training_source_stable': True, 'training_checkpoint_reload_verified': True,
        'training_model_sha256': report['model_bin_sha256'],
        'teacher_200_archive_sha256': probe['private_archive']['sha256'],
        'teacher_200_records': 3, 'teacher_200_elapsed_s': probe['elapsed_s'],
        'teacher_200_independent_replay': 'pending; separate finite job',
        'holdout_scope': report['split_scope'],
        'limitations': ['Two training steps, 20-order fixture, random dim=16 actor; not a trained paper model',
                        'Actual frozen CVRP probe uses synthetic decision losses',
                        'Fixed offline teacher-state objective is not full online-return gradient',
                        'Reconstruction control matches steps and initialization, not wall compute',
                        'Formal teacher collection and exact training choices remain unregistered',
                        'Legacy online driver cannot reproduce I3 excluded-day decoding pool; blocked locally'],
    }
    target = ROOT / 'results/a3_repair_chain_1007_2250/I3_TRAINING_DELIVERY_AUDIT_1008.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    audit()
