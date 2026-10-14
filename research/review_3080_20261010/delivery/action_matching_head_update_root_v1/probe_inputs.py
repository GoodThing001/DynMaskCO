"""Read-only probe v3: leaf identity and bit-exact forward anchors."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEL = HERE.parent
SEALED_HEAD = DEL / 'head_intermediate_arithmetic_root_v1' / 'delivery_actual_action_head_v1.py'
NPZ = DEL / 'actual_action_head_execution_root_v1' / 'downloaded'
AO = DEL / 'feasible_action_options_root_v1' / 'ALL_ACTION_OPTIONS.json'
INTER = DEL / 'head_intermediate_arithmetic_root_v1' / 'downloaded' / 'INTERMEDIATES.npz'


def load_head_module():
    spec = importlib.util.spec_from_file_location('sealed_head_source', SEALED_HEAD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import jax.numpy as jnp
    import flax.nnx as nnx
    mod = load_head_module()
    ref = mod.construct_head()
    got = {'k1': np.asarray(ref.input.kernel.value), 'b1': np.asarray(ref.input.bias.value),
           'k2': np.asarray(ref.output.kernel.value), 'b2': np.asarray(ref.output.bias.value)}
    print('construct_head leaves:', {k: (v.shape, str(v.dtype)) for k, v in got.items()})

    # 1) record 0 concat reconstruction, in-JAX vs numpy
    z0 = np.load(NPZ / 'pretrained_initial_00.npz')
    h0 = z0['latent'][0].astype(np.float32)
    a0 = z0['actions'].astype(np.int32)
    W = {k: np.asarray(z0[k], np.float32) for k in ('route_weights', 'prefix_weights', 'suffix_weights')}
    parts_np = [h0[a0[:, 0]], h0[a0[:, 3]], h0[a0[:, 4]]] + [W[k] @ h0 for k in W] + [np.asarray(z0['slot_position_length'], np.float32)]
    cat_np = np.concatenate(parts_np, -1)
    jh = jnp.asarray(h0)
    parts_jx = [jh[jnp.asarray(a0[:, 0])], jh[jnp.asarray(a0[:, 3])], jh[jnp.asarray(a0[:, 4])]] \
        + [jnp.asarray(W[k]) @ jh for k in W] + [jnp.asarray(z0['slot_position_length'])]
    cat_jx = np.asarray(jnp.concatenate(parts_jx, -1))
    print('concat numpy vs in-jax: exact', np.array_equal(cat_np, cat_jx),
          'max|diff|', float(np.max(np.abs(cat_np - cat_jx))))

    inter = np.load(INTER)
    print('sealed eager_concat vs reconstructed(jax): exact', np.array_equal(inter['eager_concat'], cat_jx),
          'max|diff|', float(np.max(np.abs(inter['eager_concat'] - cat_jx))))
    print('sealed eager_concat vs reconstructed(numpy): max|diff|', float(np.max(np.abs(inter['eager_concat'] - cat_np))))

    k1, b1, k2, b2 = (jnp.asarray(got['k1']), jnp.asarray(got['b1']), jnp.asarray(got['k2']), jnp.asarray(got['b2']))

    def fwd(x):
        return np.asarray((jnp.tanh(x @ k1 + b1) @ k2 + b2)[:, 0])

    for name, cat in (('sealed_eager_concat', jnp.asarray(inter['eager_concat'])),
                      ('reconstructed_in_jax', jnp.asarray(cat_jx))):
        out = fwd(cat)
        print(name, 'vs sealed scores: exact', np.array_equal(out, z0['scores']),
              'max|diff|', float(np.max(np.abs(out - z0['scores']))))
        print('   vs sealed eager_output: exact', np.array_equal(out, inter['eager_output']),
              'max|diff|', float(np.max(np.abs(out - inter['eager_output']))))
    print('sealed eager_output vs sealed scores: exact', np.array_equal(inter['eager_output'], z0['scores']),
          '| vs saved_original: exact', np.array_equal(inter['eager_output'], inter['saved_original']))

    # 2) all 30 records: in-jax reconstruction vs sealed scores
    ex = 0
    mx = 0.0
    for n in range(30):
        z = np.load(NPZ / ('pretrained_initial_%02d.npz' % n))
        h = z['latent'][0].astype(np.float32)
        a = z['actions'].astype(np.int32)
        jh = jnp.asarray(h)
        p = [jh[jnp.asarray(a[:, 0])], jh[jnp.asarray(a[:, 3])], jh[jnp.asarray(a[:, 4])]]
        p += [jnp.asarray(np.asarray(z[k], np.float32)) @ jh for k in ('route_weights', 'prefix_weights', 'suffix_weights')]
        p += [jnp.asarray(z['slot_position_length'])]
        cat = jnp.concatenate(p, -1)
        out = fwd(cat)
        ex += int(np.array_equal(out, z['scores']))
        mx = max(mx, float(np.max(np.abs(out - z['scores']))))
    print('all 30 in-jax reconstruction vs sealed scores: exact', ex, '/30 max|diff|', mx)


if __name__ == '__main__':
    main()
