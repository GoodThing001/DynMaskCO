"""
P0-2/3a: Future Perturbation End-to-End Test（强制测试 #2）

验证「改变所有 future 订单属性 → 当前 action（解码边偏好）不变」。

端到端语义：不是只看 encoder 输出，而是走完整 encode → decode 得到 edge logits，
再检查「可见节点相关的边 logits」在扰动 future 属性前后完全一致。

原理：future 节点经 P0-2 掩码后（坐标→depot、特征→0），无论其原始属性如何，
编码器看到的都是一样的掩码向量；coord_normalize_visible 只用 visible 统计量；
visibility-gated attention 让 future→visible 注意力=0。若这三层任一泄漏，
future 属性扰动会改变 visible 边 logits，本测试 FAIL。

用法（需 checkpoint）:
    python scripts/tests/test_future_perturbation.py \
        --data <dcc_test.npz> --ckpt <step50000.ckpt>
"""

import sys, os, argparse
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
_MASKCO = os.path.dirname(_CVRPTW)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
from cvrptw_utils import coord_normalize_visible


def build_features(data, cap, tw_max, b):
    """构造单实例 7D features（未掩码）。"""
    fl = [
        data['coords'][b:b + 1].astype(np.float32),
        (data['demands'][b:b + 1].astype(np.float32) / cap)[..., None],
        (data['tw_start'][b:b + 1].astype(np.float32) / tw_max)[..., None],
        (data['tw_end'][b:b + 1].astype(np.float32) / tw_max)[..., None],
        (data['temp_class'][b:b + 1].astype(np.float32) / 2.0)[..., None],
    ]
    if 'reveal_time' in data:
        fl.append((data['reveal_time'][b:b + 1].astype(np.float32) / tw_max)[..., None])
    return np.concatenate(fl, axis=-1).astype(np.float32)


def apply_visible_mask(features, vis):
    """P0-2 掩码：future 非坐标特征→0（含 reveal_time 索引 6），坐标→depot (0.5, 0.5)。"""
    f = features.copy()
    v = vis[..., None]
    f[..., 2:] = f[..., 2:] * v
    f[..., :2] = f[..., :2] * v + (1.0 - v) * 0.5
    return f


def test_future_perturbation(ckpt_path, data_path):
    print("=" * 60)
    print("P0-2/3a: Future Perturbation End-to-End Test")
    print("=" * 60)

    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model = model_config.construct_model()
    if params is not None:
        model = nnx.merge(nnx.graphdef(model), params)
    model_in = int(model.init_proj.kernel.shape[0])
    print(f"  Model input dim: {model_in}D")

    data = dict(np.load(data_path))
    if 'visible_mask' not in data:
        print("  SKIP: no visible_mask in dataset")
        return True

    cap = 50.0
    tw_max = float(data['tw_end'].max())
    B = min(4, data['coords'].shape[0])
    nodes = data['coords'].shape[1]

    rng = np.random.default_rng(42)
    all_pass = True

    for b in range(B):
        vis = data['visible_mask'][b].astype(np.float32)
        future = np.where(vis == 0)[0]
        if len(future) == 0:
            continue

        feats_orig = build_features(data, cap, tw_max, b)
        # 扰动 future 节点所有属性（坐标/demand/TW/temp/reveal_time）
        feats_pert = feats_orig.copy()
        feats_pert[0, future, 0] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 1] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 2] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 3] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 4] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 5] = rng.integers(0, 3, size=len(future)).astype(np.float32) / 2.0
        if feats_pert.shape[-1] > 6:
            feats_pert[0, future, 6] = rng.uniform(0, 1, size=len(future))  # reveal_time

        # 两者都做 P0-2 掩码
        a = apply_visible_mask(feats_orig, vis)
        c = apply_visible_mask(feats_pert, vis)

        # 固定一个部分邻接（可见节点构成简单链），供 decode 用
        vis_idx = np.where(vis == 1)[0]
        adjmat = np.zeros((1, nodes, nodes), dtype=np.float32)
        for k in range(len(vis_idx) - 1):
            adjmat[0, vis_idx[k], vis_idx[k + 1]] = 1.0

        def encdec(feats):
            f = jnp.array(feats)
            f = f.at[..., :2].set(coord_normalize_visible(f[..., :2], jnp.array(vis[None])))
            enc = model.encode(f[..., :model_in], visible_mask=jnp.array(vis[None]))
            logits = model.decode(enc, jnp.array([0.5], dtype=jnp.float32),
                                  jnp.array(adjmat))
            return np.array(logits[0])

        logits_a = encdec(a)
        logits_c = encdec(c)

        # 只比较「可见节点相关」的边 logits（future 边被 beam 屏蔽，不进入 action）
        vis_set = set(vis_idx.tolist()) | {0}
        max_diff = 0.0
        for i in range(nodes):
            if i not in vis_set:
                continue
            for j in range(nodes):
                if j not in vis_set:
                    continue
                max_diff = max(max_diff, float(abs(logits_a[i, j] - logits_c[i, j])))

        ok = max_diff < 1e-4
        all_pass = all_pass and ok
        print(f"  batch {b}: future={len(future)} "
              f"max|Δ visible-edge logit|={max_diff:.2e} "
              f"{'PASS' if ok else 'FAIL'}")

    return all_pass


def test_future_perturbation_no_masking(ckpt_path, data_path):
    """Test 2b（更强）: 不掩码特征，仅靠 encoder 的 visible_mask（attention bias=-1e9）
    + coord_normalize_visible 隔离 future。

    这是论文核心「causal encoder 架构」的直接证据：即使 future 特征未被显式清零，
    attention bias 仍应让 future 值不影响 visible 编码。若 FAIL，说明 encoder 层
    存在序列级信息混合（如序列级 LayerNorm），causal 声明不成立。
    """
    print("\n" + "=" * 60)
    print("P0-2: Future Perturbation WITHOUT feature masking (encoder-only causal)")
    print("=" * 60)

    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model = model_config.construct_model()
    if params is not None:
        model = nnx.merge(nnx.graphdef(model), params)
    model_in = int(model.init_proj.kernel.shape[0])

    data = dict(np.load(data_path))
    if 'visible_mask' not in data:
        print("  SKIP: no visible_mask in dataset")
        return True

    cap, tw_max = 50.0, float(data['tw_end'].max())
    B, nodes = min(4, data['coords'].shape[0]), data['coords'].shape[1]
    rng = np.random.default_rng(42)
    all_pass = True

    for b in range(B):
        vis = data['visible_mask'][b].astype(np.float32)
        future = np.where(vis == 0)[0]
        if len(future) == 0:
            continue

        feats_orig = build_features(data, cap, tw_max, b)
        feats_pert = feats_orig.copy()
        feats_pert[0, future, 0] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 1] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 2] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 3] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 4] = rng.uniform(0, 1, size=len(future))
        feats_pert[0, future, 5] = rng.integers(0, 3, size=len(future)).astype(np.float32) / 2.0
        if feats_pert.shape[-1] > 6:
            feats_pert[0, future, 6] = rng.uniform(0, 1, size=len(future))

        # 关键：不掩码，直接 encode（coord_normalize_visible 用 visible 统计量 + attention bias）
        def encode_no_mask(feats):
            f = jnp.array(feats)
            f = f.at[..., :2].set(coord_normalize_visible(f[..., :2], jnp.array(vis[None])))
            return np.array(model.encode(f[..., :model_in],
                                         visible_mask=jnp.array(vis[None]))[0])

        enc_a = encode_no_mask(feats_orig)
        enc_c = encode_no_mask(feats_pert)

        vis_idx = np.where(vis == 1)[0]
        max_diff = float(np.abs(enc_a[vis_idx] - enc_c[vis_idx]).max())
        ok = max_diff < 1e-4
        all_pass = all_pass and ok
        print(f"  batch {b}: future={len(future)} "
              f"max|Δ visible encoding|={max_diff:.2e} "
              f"{'PASS' if ok else 'FAIL'}")

    return all_pass


def test_edge_feat_perturbation(ckpt_path, data_path):
    """Test 2c: edge_feat（energy_mat）future 边扰动 → 当前 visible action 不变。

    验证 P0-7（edge_feat 乘 visible 掩码）：即使 future 相关边（future↔visible、
    future↔future）的 energy_mat 被任意扰动，visible 边 logits 也不应变化。
    """
    print("\n" + "=" * 60)
    print("P0-7: Edge Feature Perturbation (future energy_mat edges)")
    print("=" * 60)

    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model = model_config.construct_model()
    if params is not None:
        model = nnx.merge(nnx.graphdef(model), params)
    model_in = int(model.init_proj.kernel.shape[0])

    data = dict(np.load(data_path))
    if 'visible_mask' not in data or 'energy_mat' not in data:
        print("  SKIP: no visible_mask or energy_mat in dataset")
        return True

    cap, tw_max = 50.0, float(data['tw_end'].max())
    B, nodes = min(4, data['coords'].shape[0]), data['coords'].shape[1]
    energy_mat = data['energy_mat'].astype(np.float32)
    rng = np.random.default_rng(42)
    all_pass = True

    for b in range(B):
        vis = data['visible_mask'][b].astype(np.float32)
        future = np.where(vis == 0)[0]
        if len(future) == 0:
            continue

        feats = build_features(data, cap, tw_max, b)
        feats = apply_visible_mask(feats, vis)  # 掩码节点特征

        # edge_A（原始） vs edge_B（future 相关边全部随机扰动）
        edge_A = energy_mat[b:b + 1].copy()  # (1, N, N)
        edge_B = energy_mat[b:b + 1].copy()
        for fi in future:
            for j in range(nodes):
                edge_B[0, fi, j] = rng.uniform(0, 1)
                edge_B[0, j, fi] = rng.uniform(0, 1)

        vis_idx = np.where(vis == 1)[0]
        adjmat = np.zeros((1, nodes, nodes), dtype=np.float32)
        for k in range(len(vis_idx) - 1):
            adjmat[0, vis_idx[k], vis_idx[k + 1]] = 1.0

        def encdec(edge):
            f = jnp.array(feats)
            f = f.at[..., :2].set(coord_normalize_visible(f[..., :2], jnp.array(vis[None])))
            enc = model.encode(f[..., :model_in], visible_mask=jnp.array(vis[None]),
                               edge_feat=jnp.array(edge))
            logits = model.decode(enc, jnp.array([0.5], dtype=jnp.float32), jnp.array(adjmat))
            return np.array(logits[0])

        logits_A = encdec(edge_A)
        logits_B = encdec(edge_B)

        vis_set = set(vis_idx.tolist()) | {0}
        max_diff = 0.0
        for i in range(nodes):
            if i not in vis_set:
                continue
            for j in range(nodes):
                if j not in vis_set:
                    continue
                max_diff = max(max_diff, float(abs(logits_A[i, j] - logits_B[i, j])))

        ok = max_diff < 1e-4
        all_pass = all_pass and ok
        print(f"  batch {b}: future={len(future)} "
              f"max|Δ visible-edge logit|={max_diff:.2e} "
              f"{'PASS' if ok else 'FAIL'}")

    return all_pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    ok1 = test_future_perturbation(args.ckpt, args.data)
    ok2 = test_future_perturbation_no_masking(args.ckpt, args.data)
    ok3 = test_edge_feat_perturbation(args.ckpt, args.data)
    ok = ok1 and ok2 and ok3
    print("\n" + "=" * 60)
    print(f"RESULT: {'ALL PASS — future perturbation does not change action' if ok else 'FAIL — leakage detected'}")
    print(f"  (with masking: {'PASS' if ok1 else 'FAIL'}, "
          f"without masking: {'PASS' if ok2 else 'FAIL'}, "
          f"edge-feature: {'PASS' if ok3 else 'FAIL'})")
    sys.exit(0 if ok else 1)
