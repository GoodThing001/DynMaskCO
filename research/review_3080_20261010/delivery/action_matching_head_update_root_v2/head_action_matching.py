"""Action-matching scoring-head arithmetic over sealed MaskCO decoder node latents.

Scope: head-only (four leaves) per-customer feasible-action cross-entropy.
No backbone forward, no C0, no solver, no teacher cost input, no optimizer state
of the model.  All representation inputs are the sealed saved latents of the
`actual_action_head_execution_root_v1` pretrained_initial arm.

Upstream MaskCO is unchanged.  Qualification of any trained control is external.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent
SEALED_NPZ = DELIVERY / 'actual_action_head_execution_root_v1' / 'downloaded'
SEALED_MANIFEST = DELIVERY / 'actual_action_head_execution_root_v1' / 'DOWNLOAD_MANIFEST.json'
SEALED_HEAD_SOURCE = DELIVERY / 'head_intermediate_arithmetic_root_v1' / 'delivery_actual_action_head_v1.py'
SEALED_INTERMEDIATES = DELIVERY / 'head_intermediate_arithmetic_root_v1' / 'downloaded' / 'INTERMEDIATES.npz'
ACTION_OPTIONS = DELIVERY / 'feasible_action_options_root_v1' / 'ALL_ACTION_OPTIONS.json'

ARMS = ('pretrained_initial',)
RECORDS = 30
HEAD_SEED = 20261012
LEAF_KEYS = ('k1', 'b1', 'k2', 'b2')
WEIGHT_KEYS = ('route_weights', 'prefix_weights', 'suffix_weights')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sealed_manifest_hashes():
    rows = json.loads(Path(SEALED_MANIFEST).read_text(encoding='utf-8'))
    return {r['name']: r['sha256'] for r in rows}


def action_options():
    return json.loads(Path(ACTION_OPTIONS).read_text(encoding='utf-8'))


def construct_head_leaves(seed=HEAD_SEED):
    """Initial four leaves of the sealed head module, read-only import."""
    spec = importlib.util.spec_from_file_location('sealed_action_head_source', SEALED_HEAD_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    head = mod.construct_head(seed=seed)
    return {'k1': np.asarray(head.input.kernel.value, np.float32),
            'b1': np.asarray(head.input.bias.value, np.float32),
            'k2': np.asarray(head.output.kernel.value, np.float32),
            'b2': np.asarray(head.output.bias.value, np.float32)}


def record_path(index, arm='pretrained_initial'):
    return SEALED_NPZ / ('%s_%02d.npz' % (arm, index))


def load_record(index, arm='pretrained_initial'):
    z = np.load(record_path(index, arm))
    return {'index': index, 'arm': arm,
            'latent': z['latent'], 'actions': z['actions'].astype(np.int32),
            'route_weights': np.asarray(z['route_weights'], np.float32),
            'prefix_weights': np.asarray(z['prefix_weights'], np.float32),
            'suffix_weights': np.asarray(z['suffix_weights'], np.float32),
            'slot_position_length': np.asarray(z['slot_position_length'], np.float32),
            'sealed_scores': np.asarray(z['scores'], np.float32),
            'leaves': {'k1': np.asarray(z['head_input_kernel'], np.float32),
                       'b1': np.asarray(z['head_input_bias'], np.float32),
                       'k2': np.asarray(z['head_output_kernel'], np.float32),
                       'b2': np.asarray(z['head_output_bias'], np.float32)}}


def concat_jax(rec):
    """Rebuild the recorded head input (6*512 + 3) with the sealed binding."""
    import jax.numpy as jnp
    h = jnp.asarray(rec['latent'][0].astype(np.float32))
    a = rec['actions']
    parts = [h[jnp.asarray(a[:, 0])], h[jnp.asarray(a[:, 3])], h[jnp.asarray(a[:, 4])]]
    parts += [jnp.asarray(rec[k]) @ h for k in WEIGHT_KEYS]
    parts.append(jnp.asarray(rec['slot_position_length']))
    return jnp.concatenate(parts, axis=-1)


def concat_numpy(rec):
    h = rec['latent'][0].astype(np.float32)
    a = rec['actions']
    parts = [h[a[:, 0]], h[a[:, 3]], h[a[:, 4]]]
    parts += [rec[k] @ h for k in WEIGHT_KEYS]
    parts.append(rec['slot_position_length'])
    return np.concatenate(parts, axis=-1)


def head_scores_jax(concat, leaves):
    import jax.numpy as jnp
    return (jnp.tanh(jnp.asarray(concat) @ jnp.asarray(leaves['k1']) + jnp.asarray(leaves['b1']))
            @ jnp.asarray(leaves['k2']) + jnp.asarray(leaves['b2']))[:, 0]


def head_scores_numpy64(concat, leaves):
    """Independent float64 reference of the same two-layer head."""
    x = np.asarray(concat, np.float64)
    z1 = x @ np.asarray(leaves['k1'], np.float64) + np.asarray(leaves['b1'], np.float64)
    a = np.tanh(z1)
    return (a @ np.asarray(leaves['k2'], np.float64))[:, 0] + np.asarray(leaves['b2'], np.float64)


def ce_loss64(scores, target_local):
    s = np.asarray(scores, np.float64)
    m = float(s.max())
    return float(m + np.log(np.exp(s - m).sum()) - s[target_local])


def ce_grad64(concat, leaves, group_rows, target_local):
    """Independent analytic gradient of the group CE w.r.t. the four leaves."""
    x = np.asarray(concat, np.float64)[group_rows]
    k1 = np.asarray(leaves['k1'], np.float64)
    b1 = np.asarray(leaves['b1'], np.float64)
    k2 = np.asarray(leaves['k2'], np.float64)
    z1 = x @ k1 + b1
    a = np.tanh(z1)
    s = (a @ k2)[:, 0]
    m = float(s.max())
    p = np.exp(s - m)
    p = p / p.sum()
    ds = p.copy()
    ds[target_local] -= 1.0
    ds = ds[:, None]
    g_k2 = a.T @ ds
    g_b2 = ds.sum(axis=0)
    da = ds @ k2.T
    dz1 = da * (1.0 - a * a)
    g_k1 = x.T @ dz1
    g_b1 = dz1.sum(axis=0)
    return {'k1': g_k1, 'b1': g_b1, 'k2': g_k2, 'b2': g_b2}


def group_of(index, rec, option_record):
    """Candidate rows, sealed-eligible feasible subset and target row of one record."""
    a = rec['actions']
    customer = option_record['customer']
    cand = np.where(a[:, 0] == customer)[0]
    t = option_record['target']
    trow = np.where((a[:, 0] == t['customer']) & (a[:, 1] == t['vehicle']) & (a[:, 2] == t['position'])
                    & (a[:, 3] == t['predecessor']) & (a[:, 4] == t['successor']))[0]
    elig = []
    for row in option_record['rows']:
        if not row['eligible']:
            continue
        hit = np.where((a[:, 0] == row['customer']) & (a[:, 1] == row['vehicle'])
                       & (a[:, 2] == row['position']) & (a[:, 3] == row['predecessor'])
                       & (a[:, 4] == row['successor']))[0]
        elig.extend(int(x) for x in hit)
    elig = np.array(sorted(set(elig)), dtype=np.int64)
    if len(trow) != 1:
        return dict(index=index, key=option_record['key'], customer=int(customer),
                    candidates=cand, eligible=elig, target_row=None, target_local=None,
                    reason='sealed target action absent from the saved head-input action grid')
    target_row = int(trow[0])
    if target_row not in elig.tolist():
        return dict(index=index, key=option_record['key'], customer=int(customer),
                    candidates=cand, eligible=elig, target_row=target_row, target_local=None,
                    reason='sealed target action not inside the sealed eligible subset')
    return dict(index=index, key=option_record['key'], customer=int(customer),
                candidates=cand, eligible=elig, target_row=target_row,
                target_local=elig.tolist().index(target_row), reason=None)
