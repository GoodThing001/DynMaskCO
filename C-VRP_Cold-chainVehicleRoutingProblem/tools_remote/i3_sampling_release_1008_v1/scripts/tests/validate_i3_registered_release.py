"""Finite CPU release QA including real 200-order teachers and actual B weights."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def validate_low_order_collection(out, scripts):
    """Fresh real N=150 four-state engineering capture and terminal re-evaluation."""
    from copy import deepcopy
    import importlib.util
    from collect_i3_teachers import collect, sha
    from run_identity import SHARED_SOURCE_FILES
    from i3_teacher_archive import load_archive
    from scenario_saa import make_c0_contract_v2
    reg = json.loads((out / 'protocol.json').read_text())
    cc = deepcopy(reg['teacher_collection'])
    cc.update(days=[12], K=2, version='engineering_low_actual_count_v2',
              not_formal_training_config=True)
    cc['capture_ordinals_by_day'] = {'12': cc['capture_ordinals_by_day']['12']}
    cc['capture_design']['actual_orders_by_day'] = {'12': 150}
    files = set(SHARED_SOURCE_FILES) | {'scripts/training/collect_i3_teachers.py',
        'scripts/training/i3_capture_schedule.py', 'scripts/training/i3_counterfactual_teacher.py',
        'scripts/training/i3_teacher_archive.py', 'scripts/simulation/recourse_snapshot.py',
        'scripts/evaluation/run_exp_encoder_v3.py', 'scripts/evaluation/run_identity.py'}
    cfg = {'schema': 'i3_teacher_collection_v1', 'qualification': 'engineering',
           'collection': cc, 'source_sha256': {rel: sha(scripts.parent / rel) for rel in files}}
    path = out / 'engineering_low_order_config.json'
    path.write_text(json.dumps(cfg, indent=2, allow_nan=False))
    folder = out / 'engineering_low_order_collection'
    result = collect(path, sha(path), 12, folder)
    expected = [(12, ordinal) for ordinal in [18,56,93,131]]
    assert result['expected_keys'] == expected and len(result['records']) == 4
    assert all(row['same_action_parity'] for row in result['records'])
    restored = load_archive(folder / 'private.json', archive_sha256=result['archive_sha256'],
                            expected_context=result['context'], expected_keys=expected)
    spec = importlib.util.spec_from_file_location('low_order_independent_audit', out / 'teacher_data_audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    backend = audit.attach_backend(scripts.parent)
    audit.expected_capture_positions(cc, 12, 150)
    rows = [audit.check_entry(entry, public, restored['dataset'], make_c0_contract_v2(), cc, backend)
            for entry, public in zip(restored['entries'], result['records'])]
    assert len(rows) == 4 and all(row['terminal_trace_recomputed'] for row in rows)
    out.joinpath('LOW_ORDER_QA_PASS.json').write_text(json.dumps({
        'engineering_low_order_sampling_pass': True, 'day_id': 12, 'actual_customers': 150,
        'expected_keys': expected, 'captured_states': 4, 'original_action_parity_all': True,
        'archive_reload_verified': True, 'all_four_terminal_rows_recomputed': True,
        'collection_report_sha256': sha(folder / 'report.json'),
        'archive_sha256': result['archive_sha256'], 'K': 2,
        'qualification': 'engineering_only_no_formal_state_reuse',
        'online_gain_established': False}, indent=2, allow_nan=False))
    print('ALL_PASS_REAL_N150_FOUR_STATE_CAPTURE_AND_INDEPENDENT_TERMINALS=1', flush=True)


def main(out, project):
    os.environ.update(JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[16:20] if len(allowed) >= 20 else allowed[:4]))
    out, project = Path(out), Path(project)
    scripts = Path(__file__).resolve().parents[1]
    for folder in ('tests', 'training', 'evaluation', 'models'):
        sys.path.insert(0, str(scripts / folder))
    from test_i3_pool_lineage import test as pool_test
    from test_i3_collection import test as collection_test
    from test_i3_online_adapter import test as online_test
    from test_i3_online_adjudication import test as adjudication_test
    from test_i3_teacher_archive import test_archive, test_unordered_identity
    from test_i3_capture_schedule import test as capture_schedule_test
    import diag_i3_teacher_probe as fresh_teacher_probe
    from test_train_i3_runner import configuration
    from train_i3_scenario import run, sha
    from maskco_scenario import check_model_generation
    assert check_model_generation(out / 'warmstart/model.bin') == 'ok'
    pool_test()
    capture_schedule_test()
    adjudication_test()
    test_archive()
    test_unordered_identity()
    collection_test(out / 'engineering_collection')
    # Teacher identity encoding changed. Regenerate engineering fixtures with
    # this release, never rewrite or promote an old formal teacher shard.
    archive = out / 'engineering_small_teacher'
    archive.mkdir()
    small = fresh_teacher_probe.probe(archive / 'private.json')
    archive.joinpath('report.json').write_text(json.dumps(small, indent=2, allow_nan=False), encoding='utf-8')
    online_test(archive, out / 'engineering_online_adapter')
    archive = out / 'engineering_200_teacher'
    archive.mkdir()
    fresh_teacher_probe.CONFIG = fresh_teacher_probe.SCALE200_CONFIG
    real = fresh_teacher_probe.probe(archive / 'private.json')
    archive.joinpath('probe.json').write_text(json.dumps(real, indent=2, allow_nan=False), encoding='utf-8')
    cfg = configuration(archive / 'probe.json', archive / 'private.json', out / 'warmstart/model.bin')
    cfg['holdout_days'] = []
    cfg['actor_pool_excluded_days'] = []
    cfg['model'] = {'arm': 'pretrained', 'dim': 128, 'initialization_seed': 42,
        'warmstart_path': str(out / 'warmstart/model.bin'),
        'warmstart_sha256': sha(out / 'warmstart/model.bin'),
        'warmstart_training_days': list(range(200)),
        'cvrp_path': str(project.parent / 'MASKCO_code/ckpts/cvrp100.ckpt'),
        'cvrp_sha256': sha(project.parent / 'MASKCO_code/ckpts/cvrp100.ckpt')}
    cfg['settings']['steps'] = 3
    cfg['runtime']['one_cpu_affinity'] = False
    path = out / 'engineering_actual_teacher_config.json'
    path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    report = run(path, sha(path), out / 'engineering_actual_teacher_training')
    assert report['steps_completed'] == 3 and report['checkpoint_reload_verified']
    assert sum(row['actor_update'] for row in report['logs']) == 2
    assert sum(not row['teacher_valid'] for row in report['logs']) == 1
    assert report['source_start'] == report['source_end'] and not report['online_gain_established']
    validate_low_order_collection(out, scripts)
    out.joinpath('QA_PASS.json').write_text(json.dumps({
        'engineering_release_pass': True, 'real_200_order_teacher_pretrained_steps': 3,
        'actual_teacher_actor_updates': 2, 'invalid_teacher_reconstruction_updates': 1,
        'model_reload_verified': True, 'formal_training': False, 'online_gain_established': False,
        'source_stable': True, 'model_sha256': report['model_bin_sha256']}, indent=2), encoding='utf-8')
    print('ALL_PASS_REGISTERED_I3_RELEASE=1', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--project', required=True)
    args = ap.parse_args()
    main(args.out, args.project)
