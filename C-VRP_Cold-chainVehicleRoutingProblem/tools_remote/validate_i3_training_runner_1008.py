"""Finite engineering tests, one CPU away from the pretrained-probe CPU."""
import os
from pathlib import Path
import runpy
import sys

os.environ.update(JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')
allowed = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {allowed[min(1, len(allowed) - 1)]})
root = Path(__file__).resolve().parents[1]
tests = root / 'scripts/tests'
sys.argv = [str(tests / 'test_i3_training_engine.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
archive = root / 'results/i3_teacher_archive_small_server_1008'
out = root / 'results/i3_runner_engineering_server_1008_1110'
sys.argv = [str(tests / 'test_train_i3_runner.py'), '--report', str(archive / 'report.json'),
            '--archive', str(archive / 'private.json'), '--persistent-out', str(out)]
runpy.run_path(sys.argv[0], run_name='__main__')
print('ALL_PASS_I3_TRAINING_RUNNER=1', flush=True)
