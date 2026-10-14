"""Finite CPU release QA including real 200-order teachers and actual B weights."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


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
    from test_train_i3_runner import configuration
    from train_i3_scenario import run, sha
    from maskco_scenario import check_model_generation
    assert check_model_generation(out / 'warmstart/model.bin') == 'ok'
    pool_test()
    collection_test(out / 'engineering_collection')
    archive = project / 'results/i3_teacher_archive_small_server_1008'
    online_test(archive, out / 'engineering_online_adapter')
    archive = project / 'results/i3_teacher_200_archive_1008_1055'
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
