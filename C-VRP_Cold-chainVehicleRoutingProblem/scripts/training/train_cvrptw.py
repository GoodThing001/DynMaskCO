"""
CVRPTW 训练入口 — MaskCO mask-and-reconstruct 训练 for CVRPTW。

用法:
    cd /home/hzeng/project/MASKCO-Main/
    python -u "C-VRP_Cold-chainVehicleRoutingProblem/training/train_cvrptw.py" \
        --num_nodes 50 --capacity 50 --model_config softcap_fn \
        --peak_lr 1e-3 --batch_size 64 --num_steps 100 --save_interval 50 \
        --data "C-VRP_Cold-chain.../data/cvrptw50_r1_train.npz" \
        --logdir "C-VRP_Cold-chain.../logs/test" \
        --savedir "C-VRP_Cold-chain.../ckpts/test" \
        --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
"""

import sys, os

# ── GPU 选择（必须在 import jax 之前） ──
_GPU_ID = None
for _i, _arg in enumerate(sys.argv):
    if _arg == '--gpu_id' and _i + 1 < len(sys.argv):
        _GPU_ID = sys.argv[_i + 1]
        break
if _GPU_ID is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_GPU_ID)
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_GPU_ID}")

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
_MASKCO = os.path.dirname(_CVRPTW)

sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'data'))

import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from flax.jax_utils import replicate
from functools import partial
import optax, argparse, time

from CVRPTWModel import CVRPTWModelConfig, CVRPTWModel
from CVRPTWDataloader import CVRPTWDataloader
from modules.functional import coord_normalize
from training.TrainConfig import TrainConfig
from training import load_ckpt, save_ckpt
from helpers import sol2adj, sol2adj_with_mask, with_invalid_kwargs_filtered, maybe_eval


def _tw_compat_matrix(raw_features, speed):
    """从 5D 归一化特征计算 TW 兼容性矩阵 (JAX, JIT 兼容)。
    raw_features: (batch, nodes, 5) = [x, y, demand, tw_start, tw_end]
    返回: (batch, nodes_from, nodes_to) bool"""
    coords = raw_features[..., :2]
    demand = raw_features[..., 2]
    tw_start = raw_features[..., 3]
    tw_end = raw_features[..., 4]

    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = jnp.sqrt((diff ** 2).sum(axis=-1) + 1e-10)
    service_time = 0.1 + demand * 0.05
    travel_time = dist / speed
    ready_at_j = tw_start[:, :, None] + service_time[:, :, None] + travel_time
    feasible = ready_at_j <= tw_end[:, None, :]
    feasible = feasible.at[:, 0, :].set(True)
    feasible = feasible.at[:, :, 0].set(True)
    return feasible  # (batch, nodes, nodes)


def train_cvrptw(
    dataloader, model_config, train_config, model, opt_state,
    save_interval, logdir, savedir, step=0,
    encoder_input_dim=5,
    tw_loss_lambda=0.0,    # Step 1: TW penalty loss 权重 (0=禁用)
    tw_loss_speed=1.0,     # Step 1: TW 兼容性检查的速度参数
):
    max_steps = train_config.num_steps   # ← 训练总步数，用于停止条件
    tx = train_config.init_optimizer()
    graphdef, params = nnx.split(model)
    if opt_state is None:
        opt_state = tx.init(params)
    num_nodes = train_config.num_nodes
    assert train_config.target_disruption is None

    # 确保日志/保存目录存在
    os.makedirs(logdir, exist_ok=True)
    if savedir:
        os.makedirs(savedir, exist_ok=True)

    num_devices = jax.device_count()
    assert train_config.batch_size % num_devices == 0
    is_replicated = num_devices > 1
    batch_size_per_device = train_config.batch_size // num_devices
    if is_replicated:
        params = replicate(params)
        opt_state = replicate(opt_state)

    @partial(jax.jit, donate_argnums=[1, 2])
    def train_step(graphdef, params, opt_state, raw_features, target, timestep, key):
        # 消融实验：仅使用前 encoder_input_dim 个通道
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
            model_ = nnx.merge(graphdef, params)
            feats = model_.encode(raw_features)
            logits = model_.decode(feats, timestep, cur_adjmat)
            log_probs = jax.nn.log_softmax(logits)
            ce_loss = -(log_probs[:, 1:] * tgt_adjmat[:, 1:]).mean() * ((num_nodes + 1) / 2)
            tw_penalty = jnp.array(0.0)
            if tw_loss_lambda > 0:
                tw_compat = _tw_compat_matrix(raw_features, tw_loss_speed)
                probs = jnp.exp(log_probs)
                tw_penalty = (probs * (1.0 - tw_compat.astype(jnp.float32))).mean()
            total = ce_loss + tw_loss_lambda * tw_penalty
            return total, (ce_loss, tw_penalty)  # aux: 用于日志

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (_ce, _tw)), grads = grad_fn(params)
        if is_replicated:
            grads = jax.lax.pmean(grads, axis_name='data')
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state, key, _ce, _tw

    if not is_replicated:
        train_step = jax.jit(train_step, donate_argnums=[1, 2])
    else:
        train_step = jax.pmap(train_step, donate_argnums=[1, 2], axis_name='data')

    print(f"Training: max_steps={max_steps}, save_interval={save_interval}")
    print(f"  batch_size={train_config.batch_size}, lr={train_config.peak_lr}")
    print(f"  logdir={logdir}, savedir={savedir}")
    t_start = time.time()

    while True:
        key = jax.random.wrap_key_data(
            np.random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max,
                              size=[2], dtype=np.uint32))
        if num_devices > 1:
            key = jax.random.split(key, num_devices)
        for raw_features, target, timestep in dataloader:
            step += 1
            if is_replicated:
                raw_features, target, timestep = tuple(map(
                    lambda x: x.reshape((num_devices, batch_size_per_device) + x.shape[1:]),
                    (raw_features, target, timestep)))
            loss, params, opt_state, key, ce_loss, tw_penalty = train_step(
                graphdef, params, opt_state, raw_features, target, timestep, key)

            # 每 100 步打印一次
            if step % 100 == 0:
                elapsed = time.time() - t_start
                if tw_loss_lambda > 0:
                    print(f"  step {step}/{max_steps} | total={np.mean(loss):.4f} "
                          f"CE={np.mean(ce_loss):.4f} TW={np.mean(tw_penalty):.4f} | "
                          f"{elapsed:.0f}s")
                else:
                    print(f"  step {step}/{max_steps} | loss={np.mean(loss):.4f} | "
                          f"{elapsed:.0f}s elapsed")

            # 保存 checkpoint
            if savedir and step % save_interval == 0:
                save_ckpt(params, opt_state, np.random.get_state(),
                          model_config, train_config, step, savedir,
                          is_replicated=is_replicated)
                print(f"  → checkpoint saved at step {step}")

            # ← 停止条件
            if step >= max_steps:
                break

        if step >= max_steps:
            break

    # 最终保存
    if savedir:
        save_ckpt(params, opt_state, np.random.get_state(),
                  model_config, train_config, step, savedir,
                  is_replicated=is_replicated)
    elapsed = time.time() - t_start
    print(f"Training complete: {step} steps in {elapsed:.0f}s "
          f"({elapsed/step:.2f}s/step)")


if __name__ == '__main__':
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--capacity', type=eval, required=True)
    parser.add_argument('--num_steps', type=int, default=10**6)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--peak_lr', type=float, default=1e-3)
    parser.add_argument('--end_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_type', type=str, default='randperm')
    parser.add_argument('--target_disruption',
                        type=partial(maybe_eval, should_keep=['default']), default=None)
    parser.add_argument('--optimizer_type', type=str, default='adamw')
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--model_config', type=str, default='softcap_fn')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU 设备 ID（如 --gpu_id 1 仅用卡1）')
    parser.add_argument('--data_augment', type=int, default=0)
    parser.add_argument('--ignore_ckpt_train_config', action='store_true', default=False)
    parser.add_argument('--tw_max', type=float, default=None)
    parser.add_argument('--encoder_input_dim', type=int, default=5,
                        help='输入特征维度: 3=CVRP only [x,y,demand], 5=完整CVRPTW [x,y,demand,tw_start,tw_end]')
    parser.add_argument('--tw_loss_lambda', type=float, default=0.0,
                        help='Step 1: TW penalty loss 权重 (0=禁用, 推荐 0.1-1.0)')
    parser.add_argument('--tw_loss_speed', type=float, default=1.0,
                        help='Step 1: TW 兼容性速度参数（归一化空间）')
    args = parser.parse_args()

    # 默认路径放在 CVRPTW 目录下
    if args.logdir is None:
        args.logdir = os.path.join(_CVRPTW, 'logs', 'train')
    if args.savedir is None:
        args.savedir = os.path.join(_CVRPTW, 'ckpts', 'train')

    # 加载或创建模型
    params, opt_state, np_rd_state, model_config, train_config, step = load_ckpt(args.ckpt)
    if args.ignore_ckpt_train_config:
        opt_state, np_rd_state, train_config, step = None, None, None, 0
    if np_rd_state is not None:
        np.random.set_state(np_rd_state)
    if train_config is None:
        train_config = with_invalid_kwargs_filtered(TrainConfig)(**vars(args))
    if model_config is None:
        model_config = CVRPTWModelConfig.get_config(args.model_config)
    # 消融实验：覆盖 encoder_input_dim
    if args.encoder_input_dim != model_config.encoder_input_dim:
        print(f"  Overriding encoder_input_dim: {model_config.encoder_input_dim} → {args.encoder_input_dim}")
        model_config.encoder_input_dim = args.encoder_input_dim

    model = model_config.construct_model()
    if params is not None:
        try:
            model = nnx.merge(nnx.graphdef(model), params)
        except Exception:
            print("WARNING: skipping checkpoint merge, training from scratch.")

    dataloader = CVRPTWDataloader(
        dict(np.load(args.data)),
        batch_size=train_config.batch_size,
        capacity=args.capacity,
        tw_max=args.tw_max,
        target_disruption=None,
        need_current=False,
        data_augment=args.data_augment,
        num_workers=2,
    )
    # 记录实际使用的 tw_max（用于推理时复现）
    _actual_tw_max = args.tw_max if args.tw_max is not None else float(
        np.load(args.data)['tw_end'].max()
    )

    print(f"=== CVRPTW Training ===")
    print(f"  nodes={train_config.num_nodes}, capacity={args.capacity}")
    print(f"  model={args.model_config}, optimizer={args.optimizer_type}")
    print(f"  data={args.data}")
    print(f"  tw_max={_actual_tw_max:.2f}  ← 推理时需保持一致！")
    if args.tw_loss_lambda > 0:
        print(f"  TW penalty: lambda={args.tw_loss_lambda}, speed={args.tw_loss_speed}")

    train_cvrptw(dataloader, model_config, train_config, model, opt_state,
                 save_interval=args.save_interval, logdir=args.logdir,
                 savedir=args.savedir, step=step,
                 encoder_input_dim=args.encoder_input_dim,
                 tw_loss_lambda=args.tw_loss_lambda,
                 tw_loss_speed=args.tw_loss_speed)
