"""M0 utility head 评分级防泄漏 + 训练冒烟 + checkpoint 往返 + 评价指标测试。

工作包 B 的本地最小测试（JAX/Flax NNX + NumPy，合成数据）：

  1. DEFER 伪动作（rolled_out=False）仍是有效监督（读取器 supervision_mask 修复验证）；
  2. 扰动未揭示订单 → 评分不变；
  3. 追加未揭示订单 → 评分不变（隐藏节点不参与 pooling）；
  4. 候选重排 → 评分对应重排（score 是候选的函数，与位置无关）；
  5. 200 步合成训练冒烟：loss 下降、参数更新、预测变化（非全平局标签）；
  6. checkpoint 保存/加载后评分一致；
  7. encoder 表征 Shuffle：Real 与 Shuffle context 不同，且隐藏节点不参与聚合；
  8. 评价指标 sanity（完美评分 → 误差 0 / 排序 1 / 一致 1 / 无失败 / 平局口径正确）。

用法：python scripts/tests/test_coldchain_utility_probe.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)                                      # C-VRP root
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'expert', 'data', 'models',
          'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from coldchain_visible_features import (extract_context_features, extract_action_features,
                                        ACTION_FEAT_DIM)
from coldchain_teacher_dataset import TeacherDataset
from coldchain_utility_head import (UtilityHead, feature_only_context, encoder_context,
                                    shuffle_node_embeddings, FEATURE_CONTEXT_DIM)
from train_coldchain_utility_probe import (train_probe, build_head, save_checkpoint,
                                           load_checkpoint, DEFAULT_HIDDEN)
from run_coldchain_utility_probe import compute_metrics

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# --------------------------------------------------------------------------- #
# 合成 fixture
# --------------------------------------------------------------------------- #
def _tiny_dataset(coords, demands, temp_class=None, reveal=None):
    N = coords.shape[0]
    return {
        'coords': coords[None].astype(np.float32),
        'demands': demands[None].astype(np.float32),
        'tw_start': np.zeros((1, N), np.float32),
        'tw_end': np.full((1, N), 100.0, np.float32),
        'service_time': np.zeros((1, N), np.float32),
        'temp_class': np.asarray(temp_class if temp_class is not None else [0] * N,
                                 np.int32)[None],
        'initial_quality': np.ones((1, N), np.float32),
        'reveal_time': (np.zeros((1, N), np.float32) if reveal is None
                        else np.asarray(reveal, np.float32)[None]),
    }


def _snapshot(num_vehicles=2, visible_mask=None, N=4):
    if visible_mask is None:
        visible_mask = np.ones(N, bool)
    return {
        'num_vehicles': num_vehicles,
        'visible_mask': np.asarray(visible_mask),
        'vehicle_node': np.zeros(num_vehicles, np.int32),
        'vehicle_ready': np.zeros(num_vehicles),
        'vehicle_load': np.zeros(num_vehicles),
        'needs_replan': np.ones(num_vehicles, bool),
        'committed_next': np.full(num_vehicles, -1, np.int32),
        'committed_arrive': np.full(num_vehicles, np.nan),
        'committed_finish': np.full(num_vehicles, np.nan),
        'vehicle_coldchain_state': [None] * num_vehicles,
    }


def _context_vec(dataset, snapshot, inst_idx=0):
    feats = extract_context_features(dataset, inst_idx, snapshot)
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))


def _actions(n=3):
    avs, avds = [], []
    for j in range(n):
        cand = {'action': {'customer': j + 1, 'slot_kind': 'anchored', 'slot_anchor': 0,
                           'position': j, 'predecessor': 0, 'successor': 0,
                           'incumbent': False}, 'is_pseudo': False}
        av, avd = extract_action_features(cand)
        avs.append(av)
        avds.append(avd)
    return np.stack(avs).astype(np.float32), np.stack(avds).astype(bool)


def _head(seed=0):
    head = UtilityHead(FEATURE_CONTEXT_DIM, ACTION_FEAT_DIM, DEFAULT_HIDDEN, rngs=seed)
    return nnx.split(head)


def _score(graphdef, params, ctx, av, avd):
    h = nnx.merge(graphdef, params)
    return np.asarray(h(jnp.asarray(ctx), jnp.asarray(av), jnp.asarray(avd)))


def _score_candidates(graphdef, params, ctx_vec, av, avd):
    return _score(graphdef, params, ctx_vec[None], av[None], avd[None])[0]


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
def test_defer_in_supervision():
    ctx = {'context_id': 'c1'}
    defer = {'context_id': 'c1', 'is_pseudo': True, 'pseudo': 'DEFER', 'rolled_out': False,
             'outcome': {'coldchain_cost': 5.0}, 'service_ok': True, 'protocol_error': False,
             'action': {'customer': 2, 'kind': 'defer'}}
    ds = TeacherDataset({'schema': 'o0cc-teacher-dataset-v1'}, [ctx], [defer])
    ok = ds.supervision_mask([defer]) == [True]
    record('defer_in_supervision', ok, 'rolled_out=False 仍为有效监督')
    return ok


def test_unrevealed_perturb_score_invariant():
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [30., 30.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    reveal = np.array([0., 0., 0., 10.], np.float32)          # 节点 3 未揭示
    ds = _tiny_dataset(coords, demands, reveal=reveal)
    snap = _snapshot(visible_mask=np.array([True, True, True, False]), N=4)
    g, p = _head(0)
    av, avd = _actions(3)
    s1 = _score_candidates(g, p, _context_vec(ds, snap), av, avd)

    ds2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ds.items()}
    ds2['coords'] = ds2['coords'].copy()
    ds2['coords'][0, 3] = [999., 999.]
    s2 = _score_candidates(g, p, _context_vec(ds2, snap), av, avd)

    ok = np.allclose(s1, s2)
    record('unrevealed_perturb_score_invariant', ok, f"s1={s1.round(3).tolist()}")
    return ok


def test_append_unrevealed_score_invariant():
    coords4 = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], np.float32)
    demands4 = np.array([0., 1., 1., 1.], np.float32)
    ds4 = _tiny_dataset(coords4, demands4, reveal=np.zeros(4, np.float32))
    snap4 = _snapshot(visible_mask=np.ones(4, bool), N=4)
    g, p = _head(0)
    av, avd = _actions(3)
    s4 = _score_candidates(g, p, _context_vec(ds4, snap4), av, avd)

    coords5 = np.vstack([coords4, [[50., 50.]]]).astype(np.float32)
    demands5 = np.append(demands4, 1.0).astype(np.float32)
    reveal5 = np.append(np.zeros(4, np.float32), 100.0).astype(np.float32)   # 节点 4 未揭示
    ds5 = _tiny_dataset(coords5, demands5, reveal=reveal5)
    snap5 = _snapshot(visible_mask=np.array([True, True, True, True, False]), N=5)
    s5 = _score_candidates(g, p, _context_vec(ds5, snap5), av, avd)

    ok = np.allclose(s4, s5)
    record('append_unrevealed_score_invariant', ok, f"s4={s4.round(3).tolist()}")
    return ok


def test_candidate_reorder_score_permutes():
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], np.float32)
    demands = np.array([0., 1., 1., 1.], np.float32)
    ds = _tiny_dataset(coords, demands)
    snap = _snapshot(visible_mask=np.ones(4, bool), N=4)
    g, p = _head(0)
    av, avd = _actions(4)
    c = _context_vec(ds, snap)
    s = _score_candidates(g, p, c, av, avd)
    perm = np.array([2, 0, 3, 1])
    s2 = _score_candidates(g, p, c, av[perm], avd[perm])
    ok = np.allclose(s2, s[perm])
    record('candidate_reorder_score_permutes', ok)
    return ok


def test_training_smoke_200_steps():
    C, M = 64, 6
    rng = np.random.default_rng(0)
    ctx = rng.standard_normal((C, FEATURE_CONTEXT_DIM)).astype(np.float32)
    av = rng.standard_normal((C, M, ACTION_FEAT_DIM)).astype(np.float32)
    avd = np.ones((C, M, ACTION_FEAT_DIM), bool)
    # 可学习目标（context 内随候选变化，非全平局）
    y = (2.0 * (av[..., 0] - av[..., 1]) + ctx[:, :1] - 1.0).astype(np.float32)
    sup = np.ones((C, M), bool)
    tensors = {'context_vecs': ctx, 'action_vals': av, 'action_valid': avd,
               'supervision': sup, 'labels': y}
    assert y.std(axis=1).mean() > 0.1, '合成标签全平局，无法证明排序学习'

    g, params = _head(0)
    tx = optax.adamw(1e-3)
    opt_state = tx.init(params)
    init_l1 = np.array(params['l1']['kernel'].value).copy()
    init_score = _score(g, params, ctx, av, avd)

    params, opt_state, losses = train_probe(g, params, tx, opt_state, tensors,
                                            np.arange(C), 200, 8, 0, verbose=False)

    final_score = _score(g, params, ctx, av, avd)
    final_l1 = np.array(params['l1']['kernel'].value)

    ok = losses[-1] < losses[0]
    ok = ok and np.abs(final_l1 - init_l1).sum() > 0
    ok = ok and not np.allclose(init_score, final_score)
    record('training_smoke_200_steps', ok,
           f"loss {losses[0]:.4f}->{losses[-1]:.4f} "
           f"dL1={float(np.abs(final_l1 - init_l1).sum()):.2f}")
    return ok


def test_checkpoint_roundtrip():
    C, M = 8, 4
    rng = np.random.default_rng(1)
    ctx = rng.standard_normal((C, FEATURE_CONTEXT_DIM)).astype(np.float32)
    av = rng.standard_normal((C, M, ACTION_FEAT_DIM)).astype(np.float32)
    avd = np.ones((C, M, ACTION_FEAT_DIM), bool)
    g, params = _head(0)
    config = {'context_dim': FEATURE_CONTEXT_DIM, 'action_dim': ACTION_FEAT_DIM,
              'hidden': list(DEFAULT_HIDDEN), 's': 2.5, 'seed': 0, 'mode': 'feature_only'}
    tmp = tempfile.mkdtemp(prefix='probe_ckpt_')
    try:
        save_checkpoint(tmp, params, config)
        loaded = load_checkpoint(os.path.join(tmp, 'probe.ckpt'))
        g2 = build_head(loaded['config'])
        s1 = _score(g, params, ctx, av, avd)
        s2 = _score(g2, loaded['params'], ctx, av, avd)
        ok = np.allclose(s1, s2) and abs(loaded['config']['s'] - 2.5) < 1e-6
        record('checkpoint_roundtrip', ok)
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_encoder_shuffle_context_differs():
    rng = np.random.default_rng(0)
    H = rng.standard_normal((2, 6, 256)).astype(np.float32)
    vis = np.ones((2, 6), bool)
    vis[:, 4:] = False                                          # 节点 4/5 隐藏
    real = np.asarray(encoder_context(jnp.asarray(H), jnp.asarray(vis)))

    shuf = shuffle_node_embeddings(H, rng)
    shuf_ctx = np.asarray(encoder_context(jnp.asarray(shuf), jnp.asarray(vis)))

    # 追加一个隐藏节点（任意 embedding）→ context 不变（隐藏节点不参与 pooling）
    H2 = np.concatenate([H, rng.standard_normal((2, 1, 256)).astype(np.float32)], axis=1)
    vis2 = np.concatenate([vis, np.zeros((2, 1), bool)], axis=1)
    ctx2 = np.asarray(encoder_context(jnp.asarray(H2), jnp.asarray(vis2)))

    ok = not np.allclose(real, shuf_ctx)
    ok = ok and np.allclose(real, ctx2)
    record('encoder_shuffle_context_differs', ok)
    return ok


def test_eval_metrics_sanity():
    C, M = 3, 4
    scores = np.array([[0.9, 0.5, 0.1, 0.2],
                       [0.1, 0.8, 0.3, 0.2],
                       [0.5, 0.5, 0.5, 0.5]], np.float32)
    legal = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]], bool)
    sup = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], bool)
    labels = np.array([[0.9, 0.5, 0.1, 0.0],
                       [0.1, 0.8, 0.0, 0.0],
                       [0.5, 0.5, 0.5, 0.5]], np.float32)
    service_ok = np.ones((C, M), bool)
    is_pseudo = np.zeros((C, M), bool)
    selected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
    tensors = {'legal': legal, 'supervision': sup, 'labels': labels,
               'service_ok': service_ok, 'is_pseudo': is_pseudo, 'selected': selected}

    m = compute_metrics(scores, tensors)
    ok = abs(m['offline']['utility_mae']) < 1e-5
    ok = ok and abs(m['offline']['rank_corr_spearman'] - 1.0) < 1e-5
    ok = ok and abs(m['online']['top1_agreement'] - 1.0) < 1e-5
    ok = ok and abs(m['online']['keep_decision_error']) < 1e-5
    ok = ok and abs(m['online']['failure_selected_rate']) < 1e-5
    ok = ok and abs(m['online']['tie_fraction'] - 1.0 / 3.0) < 1e-5
    record('eval_metrics_sanity', ok,
           f"mae={m['offline']['utility_mae']:.4f} "
           f"spearman={m['offline']['rank_corr_spearman']:.4f} "
           f"tie={m['online']['tie_fraction']:.3f}")
    return ok


def main():
    ok = [test_defer_in_supervision(), test_unrevealed_perturb_score_invariant(),
          test_append_unrevealed_score_invariant(), test_candidate_reorder_score_permutes(),
          test_training_smoke_200_steps(), test_checkpoint_roundtrip(),
          test_encoder_shuffle_context_differs(), test_eval_metrics_sanity()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}  ({sum(ok)}/{len(ok)})")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
