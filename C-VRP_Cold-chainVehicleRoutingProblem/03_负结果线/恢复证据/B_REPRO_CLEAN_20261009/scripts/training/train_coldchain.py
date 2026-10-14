"""
冷链训练入口 — 基于 train_cvrptw.py 扩展 6D 输入。
"""

import sys, os, argparse, time
_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.dirname(_BASE)
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)

# GPU 选择（必须在 import jax 之前）
_GPU_ID = None
for _i, _arg in enumerate(sys.argv):
    if _arg == '--gpu_id' and _i + 1 < len(sys.argv):
        _GPU_ID = sys.argv[_i + 1]; break
if _GPU_ID is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_GPU_ID)

sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'data'))

import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from flax.jax_utils import replicate
from functools import partial
import optax

from ColdChainModel import ColdChainModelConfig, ColdChainModel
from ColdChainDataloader import ColdChainDataloader
from modules.functional import coord_normalize
from training.TrainConfig import TrainConfig
from training import load_ckpt, save_ckpt
from helpers import sol2adj, sol2adj_with_mask, with_invalid_kwargs_filtered, maybe_eval


def train_coldchain(
    dataloader, model_config, train_config, model, opt_state,
    save_interval, logdir, savedir, step=0, encoder_input_dim=6,
):
    max_steps = train_config.num_steps
    tx = train_config.init_optimizer()
    graphdef, params = nnx.split(model)
    if opt_state is None:
        opt_state = tx.init(params)
    num_nodes = train_config.num_nodes
    assert train_config.target_disruption is None

    os.makedirs(logdir, exist_ok=True)
    if savedir: os.makedirs(savedir, exist_ok=True)

    num_devices = jax.device_count()
    assert train_config.batch_size % num_devices == 0
    is_replicated = num_devices > 1
    batch_size_per_device = train_config.batch_size // num_devices
    if is_replicated:
        params = replicate(params)
        opt_state = replicate(opt_state)

    @partial(jax.jit, donate_argnums=[1, 2])
    def train_step(graphdef, params, opt_state, raw_features, target, timestep, key):
        raw_features = raw_features[..., :encoder_input_dim]
        raw_features = raw_features.at[..., :2].set(coord_normalize(raw_features[..., :2]))
        route_len = target.shape[-1]
        tgt_adjmat = sol2adj(target, dtype=jnp.float32, is_cvrp=True, num_nodes=num_nodes + 1)
        keep_prob = timestep
        key, subkey = jax.random.split(key)
        mask = jax.random.bernoulli(subkey, keep_prob.reshape(batch_size_per_device, 1),
                                     shape=(batch_size_per_device, route_len))
        cur_adjmat = sol2adj_with_mask(target, mask=mask, dtype=jnp.float16,
                                        is_cvrp=True, num_nodes=num_nodes + 1)

        def loss_fn(params):
            m = nnx.merge(graphdef, params)
            feats = m.encode(raw_features)
            logits = m.decode(feats, timestep, cur_adjmat)
            lp = jax.nn.log_softmax(logits)
            return -(lp[:, 1:] * tgt_adjmat[:, 1:]).mean() * ((num_nodes + 1) / 2)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)
        if is_replicated: grads = jax.lax.pmean(grads, axis_name='data')
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state, key

    if not is_replicated:
        train_step = jax.jit(train_step, donate_argnums=[1, 2])
    else:
        train_step = jax.pmap(train_step, donate_argnums=[1, 2], axis_name='data')

    print(f"Training: max_steps={max_steps}, batch={train_config.batch_size}, lr={train_config.peak_lr}")
    t_start = time.time()

    while True:
        key = jax.random.wrap_key_data(np.random.randint(0, 2**31, size=[2], dtype=np.uint32))
        if num_devices > 1: key = jax.random.split(key, num_devices)
        for raw_features, target, timestep, _visible, _reveal in dataloader:
            step += 1
            if is_replicated:
                raw_features, target, timestep = tuple(map(
                    lambda x: x.reshape((num_devices, batch_size_per_device) + x.shape[1:]),
                    (raw_features, target, timestep)))
            loss, params, opt_state, key = train_step(graphdef, params, opt_state, raw_features, target, timestep, key)

            if step % 100 == 0:
                print(f"  step {step}/{max_steps} | loss={np.mean(loss):.4f} | {time.time()-t_start:.0f}s")

            if savedir and step % save_interval == 0:
                save_ckpt(params, opt_state, np.random.get_state(), model_config, train_config, step, savedir, is_replicated=is_replicated)

            if step >= max_steps: break
        if step >= max_steps: break

    if savedir:
        save_ckpt(params, opt_state, np.random.get_state(), model_config, train_config, step, savedir, is_replicated=is_replicated)
    print(f"Done: {step} steps in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--capacity', type=eval, required=True)
    parser.add_argument('--num_steps', type=int, default=10**6)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--peak_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer_type', type=str, default='adamw')
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--model_config', type=str, default='softcap_fn')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--target_disruption', type=partial(maybe_eval, should_keep=['default']), default=None)
    parser.add_argument('--tw_max', type=float, default=None)
    parser.add_argument('--encoder_input_dim', type=int, default=6)
    args = parser.parse_args()

    if args.logdir is None: args.logdir = os.path.join(_CVRPTW, 'logs', 'train')
    if args.savedir is None: args.savedir = os.path.join(_CVRPTW, 'ckpts', 'train')

    params, opt_state, np_rd_state, model_config, train_config, step = load_ckpt(args.ckpt)
    if args.ignore_ckpt_train_config if hasattr(args, 'ignore_ckpt_train_config') else False:
        opt_state, np_rd_state, train_config, step = None, None, None, 0
    if np_rd_state is not None: np.random.set_state(np_rd_state)
    if train_config is None: train_config = with_invalid_kwargs_filtered(TrainConfig)(**vars(args))
    if model_config is None: model_config = ColdChainModelConfig.get_config(args.model_config)
    if args.encoder_input_dim != model_config.encoder_input_dim:
        print(f"  Overriding encoder_input_dim: {model_config.encoder_input_dim} → {args.encoder_input_dim}")
        model_config.encoder_input_dim = args.encoder_input_dim

    model = model_config.construct_model()
    if params is not None:
        try: model = nnx.merge(nnx.graphdef(model), params)
        except Exception: print("WARNING: training from scratch")

    dataloader = ColdChainDataloader(
        dict(np.load(args.data)), batch_size=train_config.batch_size,
        capacity=args.capacity, tw_max=args.tw_max,
        target_disruption=None, need_current=False, num_workers=2,
    )

    _tw_max = args.tw_max if args.tw_max is not None else float(np.load(args.data)['tw_end'].max())
    print(f"=== ColdChain Training ===")
    print(f"  nodes={train_config.num_nodes}, capacity={args.capacity}, tw_max={_tw_max:.1f}")

    train_coldchain(dataloader, model_config, train_config, model, opt_state,
                    save_interval=args.save_interval, logdir=args.logdir,
                    savedir=args.savedir, step=step,
                    encoder_input_dim=args.encoder_input_dim)
