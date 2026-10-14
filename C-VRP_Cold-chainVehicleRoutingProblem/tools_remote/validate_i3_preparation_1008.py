"""Finite CPU-only validation of revised I3 preparation, not a training job."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

os.environ.update(JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')
allowed = os.sched_getaffinity(0)
os.sched_setaffinity(0, {min(allowed)})
root = Path(__file__).resolve().parents[1]
tests = ['test_i3_decision_objective.py', 'test_i3_optimizer.py',
         'test_i3_actor_draws.py', 'test_i3_teacher_archive.py', 'test_i3_end_to_end.py']
files = list((root / 'scripts').rglob('*.py'))
before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
for test in tests:
    script = root / 'scripts/tests' / test
    sys.argv = [str(script)]
    print('TEST=' + test, flush=True)
    runpy.run_path(str(script), run_name='__main__')
after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
if before != after:
    raise RuntimeError('Preparation source changed during tests')
out = root / 'results/a3_repair_chain_1007_2250/i3_preparation_revision_validation_1008.json'
if out.exists():
    raise FileExistsError('Do not overwrite an earlier validation attempt')
out.write_text(json.dumps({'purpose': 'engineering tests only, not formal I3 training',
                          'tests': tests, 'all_pass': True, 'source_stable': True,
                          'source_sha256': before}, indent=2), encoding='utf-8')
print('ALL_PASS_SOURCE_STABLE=1', flush=True)
