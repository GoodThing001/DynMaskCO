"""
P0-3b v2: Target-Label Leakage Test

验证: 训练时的 target adjacency 中未来节点的边不应影响当前可见节点的 loss。

测试逻辑:
  World A: 原始 target adjacency (含未来节点的 edges)
  World B: 未来节点的所有 incoming/outgoing edges 置零

  若未来节点 target edges 携带信息 → 当前可见节点 loss 会不同 → 泄漏。
  正确结果: loss 在可见节点上完全一致 (未来 edges 不应影响当前 loss)。

用法:
    python test_target_label_leakage.py --data <train.npz> --ckpt <ckpt>
"""
import sys, os, argparse, numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModelConfig
from cvrptw_utils import coord_normalize_visible, build_causal_target_adj, build_causal_training_adj
from helpers import sol2adj, sol2adj_with_mask


def _permute_future_nodes(target, vis, rng):
    """重排 route 里 future 节点的值（visible 节点与 depot 顺序不变）。"""
    out = target.copy()
    for b in range(target.shape[0]):
        route = target[b]
        fut_pos = [p for p in range(route.shape[0])
                   if route[p] > 0 and vis[b, int(route[p])] == 0]
        if len(fut_pos) < 2:
            continue
        vals = [route[p] for p in fut_pos]
        perm = rng.permutation(len(vals))
        for p, i in zip(fut_pos, perm):
            out[b, p] = vals[i]
    return out


def test_target_label_leakage(ckpt_path, data_path):
    print("=" * 60)
    print("P0-3b v2: Target-Label Leakage Test")
    print("=" * 60)

    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model = model_config.construct_model()
    if params is not None:
        model = nnx.merge(nnx.graphdef(model), params)
    encoder_input_dim = int(model.init_proj.kernel.shape[0])
    print(f"  Model: {encoder_input_dim}D")

    data = dict(np.load(data_path))
    B = 4
    rng = np.random.default_rng(42)
    idx = rng.choice(data['coords'].shape[0], size=B, replace=False)

    coords = data['coords'][idx].astype(np.float32)
    demands = data['demands'][idx].astype(np.float32)
    tw_start = data['tw_start'][idx].astype(np.float32)
    tw_end = data['tw_end'][idx].astype(np.float32)
    temp_class = data['temp_class'][idx].astype(np.float32) / 2.0
    target = data['routes'][idx]
    vis = data.get('visible_mask', np.ones_like(demands))[idx].astype(np.float32)
    tw_max = float(tw_end.max())
    nodes = coords.shape[1]  # N+1
    cap = 50

    # Build features with P0-2 masking
    fl = [coords, (demands/cap)[..., None], (tw_start/tw_max)[..., None],
          (tw_end/tw_max)[..., None], temp_class[..., None]]
    if 'reveal_time' in data:
        fl.append((data['reveal_time'][idx].astype(np.float32)/tw_max)[..., None])
    feats = np.concatenate(fl, axis=-1).astype(np.float32)
    vis_3d = vis[..., None]
    feats_masked = feats.copy()
    feats_masked[..., 2:] = feats_masked[..., 2:] * vis_3d
    feats_masked[..., :2] = feats_masked[..., :2] * vis_3d + (1 - vis_3d) * 0.5

    # production helper（与 train_dynamic_cc.py 共用）：projection → sol2adj → vis×vis 兜底
    route_len = target.shape[-1]
    keep_prob = 0.3
    mask = rng.random((B, route_len)) < keep_prob
    n_customers = nodes - 1  # 客户数（helper 的 num_nodes 参数）
    cur_adj = np.array(build_causal_training_adj(jnp.array(target), jnp.array(mask), jnp.array(vis), n_customers))
    tgt_adj = np.array(build_causal_target_adj(jnp.array(target), jnp.array(vis), n_customers))

    # C1-a（判别力强）：future 边必须为 0（future→any、any→future），直接断言 no-leak
    future_leak = 0.0
    for b in range(B):
        fut = np.where(vis[b] == 0)[0]
        if len(fut) == 0:
            continue
        future_leak = max(future_leak,
                          float(np.abs(cur_adj[b, :, fut]).max()),
                          float(np.abs(cur_adj[b, fut, :]).max()),
                          float(np.abs(tgt_adj[b, :, fut]).max()),
                          float(np.abs(tgt_adj[b, fut, :]).max()))
    no_leak = future_leak < 1e-6

    # C1-b（辅助）：World A/B（future 节点重排）→ causalized cur_adj 相同（future 结构被中立化）
    target_b = _permute_future_nodes(target, vis, rng)
    cur_adj_b = np.array(build_causal_training_adj(jnp.array(target_b), jnp.array(mask), jnp.array(vis), n_customers))
    tgt_adj_b = np.array(build_causal_target_adj(jnp.array(target_b), jnp.array(vis), n_customers))
    diff_cur = float(np.abs(cur_adj - cur_adj_b).max())
    ab_inv = diff_cur < 1e-6

    # C1-c：真正跑 decoder + loss（A/B loss 必须相同）
    def compute_loss(cur_adj_enc, tgt_adj_enc):
        f = jnp.array(feats_masked)
        f = f.at[..., :2].set(coord_normalize_visible(f[..., :2], jnp.array(vis)))
        enc = model.encode(f[..., :encoder_input_dim], visible_mask=jnp.array(vis))
        logits = model.decode(enc, jnp.full(B, 0.5), jnp.array(cur_adj_enc).astype(jnp.float32))
        lp = jax.nn.log_softmax(logits)
        return float(-(lp[:, 1:] * jnp.array(tgt_adj_enc)[:, 1:]).mean() * (nodes / 2))

    loss_a = compute_loss(cur_adj, tgt_adj)
    loss_b = compute_loss(cur_adj_b, tgt_adj_b)
    diff_loss = abs(loss_a - loss_b)
    loss_ok = diff_loss < 1e-6

    print(f"  future 边泄漏 (max):     {future_leak:.2e} {'PASS' if no_leak else 'FAIL'} (需 0)")
    print(f"  A/B cur_adj 不变:       {diff_cur:.2e} {'PASS' if ab_inv else 'FAIL'} (需 <1e-6)")
    print(f"  A/B loss 不变:          {diff_loss:.2e} {'PASS' if loss_ok else 'FAIL'} (需 <1e-6)")
    print(f"  Future nodes/batch:     {[(vis[b] == 0).sum() for b in range(B)]}")

    return no_leak and ab_inv and loss_ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    ok = test_target_label_leakage(args.ckpt, args.data)
    print()
    print("=" * 60)
    print(f"RESULT: {'ALL PASS' if ok else 'FAIL — TARGET LEAKAGE'}")
    print("=" * 60)
    sys.exit(0 if ok else 1)
