"""Remote runner: pin BLAS/OMP threads to 1, then execute a target script.

Why: the PROGRAM requires single-thread CPU isolation, and tools/_srv_exec.py builds the remote
command from components so there is no place to inject `set VAR=...` / `export VAR=...`. This
wrapper sets the environment BEFORE the target imports numpy, then hands over argv unchanged.

Usage (from the directory containing this file):
    <python> _remote_run.py <target_script.py> [target args...]
"""
import os
import runpy
import sys

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

if len(sys.argv) < 2:
    raise SystemExit('usage: _remote_run.py <target_script.py> [args...]')
_target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(_target, run_name='__main__')
