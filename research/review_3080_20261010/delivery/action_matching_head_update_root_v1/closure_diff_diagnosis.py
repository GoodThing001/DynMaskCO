"""Mechanism diagnosis: localize the loaded-source closure change seen by v1.

Read-only with respect to v1 recorded artifacts.  It repeats the v1 warm-up and
work sequence and reports exactly which files/modules appear or change between
the start and end loaded-source closures.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1')

import numpy as np
import psutil

import engineering_stage_closure_v1 as seal
import head_action_matching as ham

REG = json.loads((HERE / 'REGISTRATION.json').read_text(encoding='utf-8'))


def main():
    seal.initialize()
    import jax
    import jax.numpy as jnp

    w = {'k1': jnp.ones((3075, 64), jnp.float32), 'b1': jnp.zeros((64,), jnp.float32),
         'k2': jnp.ones((64, 1), jnp.float32), 'b2': jnp.zeros((1,), jnp.float32)}

    def warm_ce(leaves):
        x = jnp.ones((4, 3075), jnp.float32)
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[0]

    np.asarray(jax.grad(warm_ce)(w))
    ham.construct_head_leaves(seed=ham.HEAD_SEED)
    ham.head_scores_jax(np.zeros((2, 3075), np.float32), {k: np.asarray(v) for k, v in w.items()})

    a = seal.closure(tuple(Path(p) for p in REG['sources']))

    # Reproduce the v1 work sequence on the sealed inputs.
    records = [ham.load_record(n, ham.ARMS[0]) for n in range(ham.RECORDS)]
    options = ham.action_options()
    groups = []
    for n in range(ham.RECORDS):
        g = ham.group_of(n, records[n], options[n])
        g['eligible_concat'] = ham.concat_numpy(records[n])[g['eligible']]
        groups.append(g)
    supervised = [g for g in groups if g['target_local'] is not None]
    rows = {g['index']: (jnp.asarray(g['eligible_concat']), g['target_local']) for g in supervised}

    def group_ce(leaves, index):
        x, t = rows[index]
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[t]

    leaves = {k: jnp.asarray(records[0]['leaves'][k]) for k in ham.LEAF_KEYS}
    grad_fn = jax.grad(group_ce)
    import optax
    opt = optax.adam(learning_rate=0.003)
    opt_state = opt.init(leaves)
    for index in [g['index'] for g in supervised][:3]:
        grads = grad_fn(leaves, index)
        updates, opt_state = opt.update(grads, opt_state)
        leaves = optax.apply_updates(leaves, updates)
    np.savez(HERE / 'CLOSURE_DIAGNOSIS_tmp.npz', **{k: np.asarray(leaves[k]) for k in ham.LEAF_KEYS})

    b = seal.closure(tuple(Path(p) for p in REG['sources']))
    changed = sorted(p for p, h in a['files'].items() if b['files'].get(p) != h)
    added = sorted(set(b['files']) - set(a['files']))
    removed = sorted(set(a['files']) - set(b['files']))
    mod_added = sorted(set(b['modules']) - set(a['modules']))
    mod_removed = sorted(set(a['modules']) - set(b['modules']))
    mod_changed = sorted(m for m in set(a['modules']) & set(b['modules']) if a['modules'][m] != b['modules'][m])
    out = dict(equal=seal.same(a, b), files_start=len(a['files']), files_end=len(b['files']),
               modules_start=len(a['modules']), modules_end=len(b['modules']),
               changed_files=changed, added_files=added, removed_files=removed,
               added_modules=mod_added, removed_modules=mod_removed, changed_modules=mod_changed,
               note='diagnosis only; no v1 artifact modified')
    (HERE / 'CLOSURE_DIAGNOSIS.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n',
                                                 encoding='utf-8', newline='\n')
    os.remove(HERE / 'CLOSURE_DIAGNOSIS_tmp.npz')
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])


if __name__ == '__main__':
    raise SystemExit(main())
