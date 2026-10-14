"""Portable single-thread machine benchmark (stdlib + numpy, optional jax).

Purpose: decide where to run the project's CPU-bound panels. The panels are Python-heavy
(the reconstruct / certification code loops in Python) with a modest number of small JAX ops, so
raw core speed, small-array numpy throughput and JAX dispatch overhead matter more than BLAS
throughput or core count.

Threads are pinned to 1 through the environment before numpy is imported, so the numbers are
comparable and consistent with the project's "CPU single-thread isolation" rule.

Usage:
    python tools/_bench.py            # prints one JSON object
"""
import json
import os
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np  # noqa: E402


def timed(fn, repeat=1):
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def bench_py_loop(n=2_000_000):
    s = 0.0
    for i in range(n):
        s += (i % 7) * 0.5
    return s


def bench_numpy_small(ops=60_000):
    rng = np.random.default_rng(0)
    a = rng.random((51, 51))
    b = rng.random((51, 51))
    for _ in range(ops):
        c = a + b
        d = float(np.mean(c))
    return d


def bench_numpy_mid(ops=300):
    rng = np.random.default_rng(0)
    a = rng.random((200, 200))
    b = rng.random((200, 200))
    for _ in range(ops):
        a = (a @ b) * 1e-6
    return float(a[0, 0])


def bench_jax():
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:                      # noqa: BLE001
        return {'available': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    out = {'available': True, 'version': jax.__version__,
           'backend': jax.default_backend()}

    @jax.jit
    def small(x):
        return jnp.tanh(x) * 2.0 + jnp.sum(x) * 0.001

    x = jnp.ones((51, 51))
    small(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(3000):
        small(x).block_until_ready()
    out['dispatch_3000_small_ops_s'] = time.perf_counter() - t0

    @jax.jit
    def mid(a, b):
        return a @ b

    a = jnp.ones((512, 512))
    b = jnp.ones((512, 512))
    mid(a, b).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(300):
        mid(a, b).block_until_ready()
    out['matmul_300x512_s'] = time.perf_counter() - t0
    return out


def main():
    out = {'python': __import__('sys').version.split()[0], 'numpy': np.__version__,
           'numpy_threads_env': {k: os.environ[k] for k in
                                 ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                  'OPENBLAS_NUM_THREADS')}}
    out['py_loop_2M_s'] = timed(lambda: bench_py_loop(2_000_000))
    out['numpy_small_60k_s'] = timed(lambda: bench_numpy_small(60_000))
    out['numpy_mid_300x200matmul_s'] = timed(lambda: bench_numpy_mid(300))
    out['jax'] = bench_jax()
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
