"""真 3b 第一步：冻结 encoder 分布迁移接口检查（服务器 CPU）。

检查 DynamicColdChainModel（step50000.ckpt，256 维，7D）能否在 ~200 节点合成日上
正常编码（训练时是 ~50 节点 + 51×51 padding），并验证 visible_mask 门控未来节点。

7D = [x, y, demand/cap, tw_start/tw_max, tw_end/tw_max, temp_class/2, reveal_time/tw_max]。
"""
import sys
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
for _p in (_MASKCO, _CVRPTW, os.path.join(_CVRPTW, 'scripts', 'models'),
           os.path.join(_CVRPTW, 'scripts', 'simulation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModelConfig
from cvrptw_utils import coord_normalize_visible

CKPT = os.path.join(_CVRPTW, 'ckpts', 'r1_5_baseline', 'typed_v1_edge',
                    'phase3c', 'seed42', 'step50000.ckpt')


def main():
    print('loading', CKPT, flush=True)
    params, _, _, model_config, _, _ = load_ckpt(CKPT)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    if params is None:
        raise RuntimeError('empty params')
    model = nnx.merge(nnx.graphdef(model), params)
    model_in = int(model.init_proj.kernel.shape[0])
    print('encoder_input_dim =', model_in, flush=True)

    for N in (50, 200):
        rng = np.random.default_rng(N)
        coords = rng.uniform(0.0, 1.0, (N + 1, 2))
        coords[0] = [0.5, 0.5]
        demands = rng.uniform(1.0, 3.0, N + 1)
        demands[0] = 0.0
        temp_class = rng.integers(0, 3, N + 1)
        temp_class[0] = 0
        tw_start = rng.uniform(0.0, 10.0, N + 1)
        tw_end = tw_start + rng.uniform(1.0, 3.0, N + 1)
        tw_end[0] = 16.0
        reveal_time = rng.uniform(0.0, 8.0, N + 1)
        reveal_time[0] = 0.0
        cap, tw_max = 50.0, float(tw_end.max())

        f_list = [
            coords.astype(np.float32),
            (demands / cap)[..., None].astype(np.float32),
            (tw_start / tw_max)[..., None].astype(np.float32),
            (tw_end / tw_max)[..., None].astype(np.float32),
            (temp_class.astype(np.float32) / 2.0)[..., None],
            (reveal_time / tw_max)[..., None].astype(np.float32),
        ]
        raw = np.concatenate(f_list, axis=-1)[None]

        current_time = 3.0
        visible = (reveal_time <= current_time).astype(np.float32)
        visible[0] = 1.0
        visible_mask = visible[None]

        @jax.jit
        def encode(raw, vm):
            raw = raw.at[..., :2].set(coord_normalize_visible(raw[..., :2], vm))
            return model.encode(raw[..., :model_in], visible_mask=vm)

        H = np.asarray(encode(jnp.asarray(raw), jnp.asarray(visible_mask)))
        depot_norm = float(np.linalg.norm(H[0, 0]))
        cust_norms = np.linalg.norm(H[0, 1:], axis=-1)
        print(f'N={N}: H shape {H.shape}, finite={bool(np.isfinite(H).all())}, '
              f'depot_norm={depot_norm:.3f}, cust_norm mean={cust_norms.mean():.3f} '
              f'std={cust_norms.std():.3f}', flush=True)


if __name__ == '__main__':
    main()
