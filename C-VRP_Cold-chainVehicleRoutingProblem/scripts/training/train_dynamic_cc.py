"""
Phase C: 动态冷链训练 — DynamicColdChainModel + Event-Driven Masking。
"""

import sys, os, argparse, time
_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
_MASKCO = os.path.dirname(_CVRPTW)

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

from DynamicColdChainModel import DynamicColdChainModelConfig, DynamicColdChainModel
from DynamicColdChainModelEdgeState import DynamicColdChainModelEdgeState
from ColdChainDataloader import ColdChainDataloader
from cvrptw_utils import coord_normalize_visible, project_route_to_visible, build_causal_target_adj, build_causal_training_adj
from training.TrainConfig import TrainConfig
from training import load_ckpt, save_ckpt
from helpers import sol2adj, sol2adj_with_mask, with_invalid_kwargs_filtered, maybe_eval


def train_dynamic_cc(
    dataloader, model_config, train_config, model, opt_state,
    save_interval, logdir, savedir, step=0, encoder_input_dim=7,
    masking_mode='random', spoilage_lambda=0.1,
    online_seq_training=False,
    online_seq_steps=5,  # Phase 3c: K timesteps (default 5)
    use_edge_feat=False,  # P0-6: 显式边特征（energy_mat）注入编码器
):
    max_steps = train_config.num_steps
    tx = train_config.init_optimizer()
    graphdef, params = nnx.split(model)
    if opt_state is None:
        opt_state = tx.init(params)
    num_nodes = train_config.num_nodes
    batch_size_per_device = train_config.batch_size // jax.device_count()

    os.makedirs(logdir, exist_ok=True)
    if savedir: os.makedirs(savedir, exist_ok=True)

    is_replicated = jax.device_count() > 1
    if is_replicated:
        params = replicate(params)
        opt_state = replicate(opt_state)

    @partial(jax.jit, donate_argnums=[1, 2])
    def train_step(graphdef, params, opt_state, raw_features, target, timestep,
                   visible_mask, edge_feat, key):
        raw_features = raw_features[..., :encoder_input_dim]
        # 掩码 future 特征（t=0 可见性）：features[2:]→0，coords→depot(0.5)
        if visible_mask is not None:
            vis = visible_mask[..., None]
            raw_features = raw_features.at[..., 2:].set(raw_features[..., 2:] * vis)
            raw_features = raw_features.at[..., :2].set(raw_features[..., :2] * vis + (1.0 - vis) * 0.5)
        raw_features = raw_features.at[..., :2].set(
            coord_normalize_visible(raw_features[..., :2], visible_mask))
        route_len = target.shape[-1]
        # B2：causal 训练 adjacency（production helper，与 C1 测试共用）
        tgt_adjmat = build_causal_target_adj(target, visible_mask, num_nodes)

        # === Event-Driven Masking ===
        key, subkey = jax.random.split(key)
        keep_prob = timestep

        if masking_mode == 'random':
            mask = jax.random.bernoulli(subkey, keep_prob.reshape(batch_size_per_device, 1),
                                         shape=(batch_size_per_device, route_len))
        elif masking_mode == 'spatio_temporal':
            # 时空掩码：动态订单多的batch增加掩码比例
            keep = jnp.clip(keep_prob * 0.8, 0.1, 0.9)
            mask = jax.random.bernoulli(subkey, keep.reshape(-1, 1),
                                         shape=(batch_size_per_device, route_len))
        elif masking_mode == 'predictive':
            # 预判式掩码：降低整体 keep_prob (0.65x)，更多掩码→模型学会留空间
            keep = jnp.clip(keep_prob * 0.65, 0.08, 0.9)
            mask = jax.random.bernoulli(subkey, keep.reshape(-1, 1),
                                         shape=(batch_size_per_device, route_len))
        elif masking_mode == 'targeted':
            keep = jnp.clip(keep_prob * 0.6, 0.1, 0.9)
            mask = jax.random.bernoulli(subkey, keep.reshape(-1, 1),
                                         shape=(batch_size_per_device, route_len))

        elif masking_mode == 'adaptive':
            # D4: per-batch dynamic factor based on reveal_time ratio
            rt_mean = raw_features[..., 6].mean(axis=-1, keepdims=True)  # (B, 1)
            dynamic_factor = jnp.clip(1.0 - 0.3 * (rt_mean > 0.01), 0.7, 1.0)
            keep = jnp.clip(keep_prob.reshape(-1, 1) * dynamic_factor, 0.08, 0.95)
            mask = jax.random.bernoulli(subkey, keep, shape=(batch_size_per_device, route_len))
        else:
            keep = jnp.clip(keep_prob * 0.6, 0.1, 0.9)
            mask = jax.random.bernoulli(subkey, keep.reshape(-1, 1),
                                         shape=(batch_size_per_device, route_len))

        cur_adjmat = build_causal_training_adj(target, mask, visible_mask, num_nodes)

        def loss_fn(params):
            m = nnx.merge(graphdef, params)
            feats = m.encode(raw_features, visible_mask=visible_mask, edge_feat=edge_feat)
            logits = m.decode(feats, timestep, cur_adjmat)
            lp = jax.nn.log_softmax(logits)
            # B2：loss target 已经是 projected + vis×vis 兜底，直接做 visible-only 监督
            ce = -(lp[:, 1:] * tgt_adjmat[:, 1:]).mean() * ((num_nodes + 1) / 2)
            return ce

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

    # === D5 / 3c: K-Timestep Online Sequential Training ===
    if online_seq_training:
        K_STEPS = online_seq_steps  # number of timesteps (default 5)
        # Visibility schedule: t=0 uses actual visible_mask, later steps progressively reveal
        # Mask density schedule: sparse → dense (more edges known at later steps)
        KEEP_SCHEDULE = [0.15, 0.30, 0.50, 0.70, 0.85, 0.90, 0.92, 0.95]
        WEIGHTS = [0.25, 0.20, 0.18, 0.15, 0.12, 0.05, 0.03, 0.02]  # early steps higher weight
        REVEAL_SCHEDULE = [0.0, 0.25, 0.50, 0.75, 1.0]  # 第一个状态 = 真实 t=0 filtration

        @partial(jax.jit, donate_argnums=[1, 2])
        def train_step_online(graphdef, params, opt_state, raw_features, target,
                              timestep, visible_mask, edge_feat, reveal_time, key):
            """K-timestep event-sequence rollout training (Phase 3c)."""
            raw_features = raw_features[..., :encoder_input_dim]
            route_len = target.shape[-1]
            B = batch_size_per_device

            def loss_at_timestep(params, vis_mask_t, cur_adj_t, tgt, timestep_t):
                # 按当前可见性 vis_k 逐步掩码：新揭示订单特征真正被揭示，future 清零。
                # （2026-08-27 修复 zeroed-features bug：原 t=0 掩码导致刚揭示订单的
                #   demand/TW/temp 永不被恢复，动态训练信号退化。）
                # （2026-08-28 修复 timestep mismatch：decoder timestep 必须等于实际
                #   mask density = timestep * KEEP_SCHEDULE[k]，不再是原始 timestep。）
                vis = vis_mask_t[..., None]
                feats_masked = raw_features.at[..., 2:].set(raw_features[..., 2:] * vis)
                feats_masked = feats_masked.at[..., :2].set(
                    feats_masked[..., :2] * vis + (1.0 - vis) * 0.5)
                feats_masked = feats_masked.at[..., :2].set(
                    coord_normalize_visible(feats_masked[..., :2], vis_mask_t))
                m = nnx.merge(graphdef, params)
                feats = m.encode(feats_masked, visible_mask=vis_mask_t, edge_feat=edge_feat)
                logits = m.decode(feats, timestep_t, cur_adj_t)
                lp = jax.nn.log_softmax(logits)
                # B2：tgt 已经是 projected + vis×vis 兜底，直接 visible-only 监督
                return -(lp[:, 1:] * tgt[:, 1:]).mean() * ((num_nodes + 1) / 2)

            def combined_loss(params):
                total = 0.0
                key_iter = key
                for k in range(min(K_STEPS, len(KEEP_SCHEDULE))):
                    key_iter, sk = jax.random.split(key_iter)
                    # P1-G：keep 用固定 KEEP_SCHEDULE[k]（不再 timestep * schedule，
                    # 避免「0.70 阶段」实际 keep = timestep*0.70 的名实不符）
                    keep = jnp.full_like(timestep, KEEP_SCHEDULE[k])
                    mask_k = jax.random.bernoulli(sk, keep.reshape(-1, 1), shape=(B, route_len))
                    # P1-G：reveal 用 REVEAL_SCHEDULE，第一个状态 thr=0.0 = 真实 t=0 filtration
                    thr = REVEAL_SCHEDULE[k] if k < len(REVEAL_SCHEDULE) else 1.0
                    reveal_time_norm = reveal_time  # (B, N)，未掩码 reveal_time（单独 yield，归一化到 [0,1]）
                    vis_k = (reveal_time_norm <= thr).astype(visible_mask.dtype)
                    vis_k = jnp.maximum(visible_mask, vis_k)
                    # B2：causal 训练 adjacency（production helper，与 C1 测试共用）
                    cur_adj = build_causal_training_adj(target, mask_k, vis_k, num_nodes)
                    tgt_k = build_causal_target_adj(target, vis_k, num_nodes)
                    w = WEIGHTS[k] if k < len(WEIGHTS) else 0.02
                    total += w * loss_at_timestep(params, vis_k, cur_adj, tgt_k, keep)
                return total

            grad_fn = jax.value_and_grad(combined_loss)
            loss, grads = grad_fn(params)
            if is_replicated: grads = jax.lax.pmean(grads, axis_name='data')
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_opt_state, key

    print(f"Training: steps={max_steps} batch={train_config.batch_size} "
          f"lr={train_config.peak_lr} mask={masking_mode} "
          f"online_seq={online_seq_training} spoilage_lambda={spoilage_lambda}")
    use_online = online_seq_training
    t_start = time.time()

    while True:
        key = jax.random.wrap_key_data(np.random.randint(0, 2**31, size=[2], dtype=np.uint32))
        if is_replicated: key = jax.random.split(key, jax.device_count())
        for batch in dataloader:
            if use_edge_feat:
                raw_features, target, timestep, visible_mask, edge_feat, reveal_time = batch
            else:
                raw_features, target, timestep, visible_mask, reveal_time = batch
                edge_feat = None
            step += 1
            if is_replicated:
                raw_features, target, timestep = tuple(map(
                    lambda x: x.reshape((jax.device_count(), batch_size_per_device) + x.shape[1:]),
                    (raw_features, target, timestep)))
                if visible_mask is not None:
                    visible_mask = visible_mask.reshape((jax.device_count(), batch_size_per_device) + visible_mask.shape[1:])
                if edge_feat is not None:
                    edge_feat = edge_feat.reshape((jax.device_count(), batch_size_per_device) + edge_feat.shape[1:])
                if reveal_time is not None:
                    reveal_time = reveal_time.reshape((jax.device_count(), batch_size_per_device) + reveal_time.shape[1:])
            if use_online:
                loss, params, opt_state, key = train_step_online(
                    graphdef, params, opt_state, raw_features, target, timestep, visible_mask, edge_feat, reveal_time, key)
            else:
                loss, params, opt_state, key = train_step(
                    graphdef, params, opt_state, raw_features, target, timestep, visible_mask, edge_feat, key)

            if step % 100 == 0:
                print(f"  step {step}/{max_steps} | loss={np.mean(loss):.4f} | {time.time()-t_start:.0f}s")

            if savedir and step % save_interval == 0:
                save_ckpt(params, opt_state, np.random.get_state(),
                          model_config, train_config, step, savedir, is_replicated=is_replicated)
            if step >= max_steps: break
        if step >= max_steps: break

    if savedir:
        save_ckpt(params, opt_state, np.random.get_state(),
                  model_config, train_config, step, savedir, is_replicated=is_replicated)
    print(f"Done: {step} steps in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--capacity', type=eval, required=True)
    parser.add_argument('--seed', type=int, default=42,
                        help='NumPy random seed for training reproducibility')
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
    parser.add_argument('--encoder_input_dim', type=int, default=7)
    parser.add_argument('--masking_mode', type=str, default='random',
                        choices=['random', 'spatio_temporal', 'targeted', 'predictive', 'adaptive'],
                        help='D4: "adaptive" uses tw_slack + reveal_time per-node mask ratio')
    parser.add_argument('--online_seq_training', action='store_true', default=False,
                        help='Phase 3c: K-timestep event-sequence training rollout')
    parser.add_argument('--online_seq_steps', type=int, default=5,
                        help='Phase 3c: number of timesteps (default 5, range 2-8)')
    parser.add_argument('--spoilage_lambda', type=float, default=0.1)
    parser.add_argument('--use_edge_feat', action='store_true', default=False,
                        help='P0-6: 显式边特征（energy_mat）注入编码器注意力偏置')
    parser.add_argument('--use_edge_state', action='store_true', default=False,
                        help='Phase A1: 用 DynamicColdChainModelEdgeState（5D 边特征 + EdgeBiasProjector 多头偏置）')
    args = parser.parse_args()

    if args.logdir is None: args.logdir = os.path.join(_CVRPTW, 'logs', 'train')
    if args.savedir is None: args.savedir = os.path.join(_CVRPTW, 'ckpts', 'train')

    params, opt_state, np_rd_state, model_config, train_config, step = load_ckpt(args.ckpt)
    if np_rd_state is not None:
        np.random.set_state(np_rd_state)
    else:
        np.random.seed(args.seed)  # fresh training with reproducible seed
    if train_config is None: train_config = with_invalid_kwargs_filtered(TrainConfig)(**vars(args))
    if model_config is None: model_config = DynamicColdChainModelConfig.get_config(args.model_config)
    if args.encoder_input_dim != model_config.encoder_input_dim:
        print(f"  encoder_input_dim override: {model_config.encoder_input_dim} → {args.encoder_input_dim}")
        model_config.encoder_input_dim = args.encoder_input_dim

    if args.use_edge_state:
        # Phase A1: EdgeState 模型（5D 边特征 + EdgeBiasProjector 多头偏置）
        model = DynamicColdChainModelEdgeState(**vars(model_config))
    else:
        model = model_config.construct_model()
    if params is not None:
        try:
            model = nnx.merge(nnx.graphdef(model), params)
        except Exception as e:
            # P1-E：checkpoint 与模型结构不匹配时绝不静默从随机模型训练
            raise RuntimeError(
                f"Checkpoint/model mismatch: {args.ckpt}\n  {e}"
            )

    dataloader = ColdChainDataloader(
        dict(np.load(args.data)), batch_size=train_config.batch_size,
        capacity=args.capacity, tw_max=args.tw_max,
        target_disruption=None, need_current=False, num_workers=2,
        need_edge_feat=args.use_edge_feat,
    )
    _tw_max = args.tw_max if args.tw_max is not None else float(np.load(args.data)['tw_end'].max())
    print(f"=== Phase C Training ===")
    print(f"  nodes={train_config.num_nodes}, tw_max={_tw_max:.1f}, "
          f"mask={args.masking_mode}, spoilage_lambda={args.spoilage_lambda}")

    train_dynamic_cc(dataloader, model_config, train_config, model, opt_state,
                     save_interval=args.save_interval, logdir=args.logdir,
                     savedir=args.savedir, step=step,
                     encoder_input_dim=args.encoder_input_dim,
                     masking_mode=args.masking_mode,
                     spoilage_lambda=args.spoilage_lambda,
                     online_seq_training=args.online_seq_training,
                     online_seq_steps=args.online_seq_steps,
                     use_edge_feat=args.use_edge_feat)
