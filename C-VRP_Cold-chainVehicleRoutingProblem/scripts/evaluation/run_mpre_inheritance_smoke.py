"""M-trained 继承关系 smoke：验证 MpreTrainedModel 正确继承预训练 CVRP 行为并接通梯度。

验收（对应 MaskCO直接目标训练工作包.md 第三步）：
  1. 参数继承：backbone 参数与 M-pre 一致；新增只有 c_adapter/delta_head；无 Param 漏网。
  2. 零残差：C=0、δ=0，且相关输出全部有限。
  3. 原输出继承：decode(target='logit') 与新模型有效非对角 logits 一致。
  4. 动作继承：相同合法动作下，插入分数 / log-prob / argmax 一致（logp_diff 入通过条件）。
  5. 梯度与冻结：诊断更新后 decoder + 残差出口有有效梯度；冻结参数从更新后模型重提取逐元素不变；
     独立 M-pre 不变；应训练模块确实更新（非仅梯度非零）。
  6. 附加特征通路：出口更新后扰动输入特征能改变选择分布（log-prob，非原始分数）；
     第二次反向传播梯度进入适配内部层。
  7. padding 一致性：init 与更新后模型，B=2 不同有效长度 + 不同有限填充内容，有效分数/log-prob/argmax 一致。

失败时返回非零退出码。
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'data'))

from mpre import load_cvrp_model
from mpre_trained import (load_mpre_trained, partition, param_manifest,
                          edge_insertion_scores)
from repair_state import F_NODE, F_ACTION_EXPLICIT

NODE_FEAT_DIM = F_NODE
ACTION_FEAT_DIM = F_ACTION_EXPLICIT


def _make_inputs(N, M, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, (1, N, 2)).astype(np.float32)
    demands = rng.uniform(1, 10, (1, N)).astype(np.float32)
    coords[:, 0] = 0.0
    demands[:, 0] = 0.0
    raw_3d = np.concatenate([coords, (demands / 50.0)[..., None]], axis=-1)
    node_valid = np.ones((1, N), bool)
    node_feats = np.zeros((1, N, NODE_FEAT_DIM), np.float32)
    for i in range(N):
        node_feats[0, i, 0] = rng.uniform(-1, 1)          # tw_start rel
        node_feats[0, i, 1] = rng.uniform(0, 1)           # tw_end rel
        node_feats[0, i, 2] = 0.05                        # service_time
        node_feats[0, i, 3] = float(rng.integers(0, 3))   # temp_class
        node_feats[0, i, 4] = rng.uniform(0, 1)           # initial_quality（冷链品质）
    node_feats[0, 0, 5] = 1.0   # depot
    timestep = np.array([0.6], np.float32)
    adjmat = np.zeros((1, N, N), np.float32)
    for i in range(1, N - 1):
        adjmat[0, i, i + 1] = adjmat[0, i + 1, i] = 1.0
    # 4 个动作：3 个普通插入 + 1 个真实空车开路线 (pred==succ==0)
    cust = np.array([[1, 2, 3, 4]], np.int32)
    pred = np.array([[0, 4, 2, 0]], np.int32)
    succ = np.array([[4, 0, 0, 0]], np.int32)
    action_feats = np.zeros((1, M, ACTION_FEAT_DIM), np.float32)
    action_feats[0, :, 0] = rng.uniform(0, 1, (M,))
    return dict(raw_3d=raw_3d, node_valid=node_valid, node_feats=node_feats,
                timestep=timestep, adjmat=adjmat, cust=cust, pred=pred, succ=succ,
                action_feats=action_feats)


def _flatten(state):
    out = {}
    for p, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        key = '.'.join(str(k.key if hasattr(k, 'key') else k) for k in p)
        out[key] = np.asarray(leaf).copy()
    return out


def _grad_nonzero_count(g, prefix, exclude_exit=False):
    n = 0
    for p, leaf in jax.tree_util.tree_flatten_with_path(g)[0]:
        key = '.'.join(str(k.key if hasattr(k, 'key') else k) for k in p)
        if key.startswith(prefix) and np.asarray(leaf).size > 0 and np.abs(np.asarray(leaf)).max() > 0:
            if exclude_exit and ('.layers.2.' in key or key.endswith('.layers.2.kernel') or key.endswith('.layers.2.bias')):
                continue
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    backbone, cfg, step = load_cvrp_model(args.cvrp_ckpt)
    model, _, _ = load_mpre_trained(args.cvrp_ckpt, NODE_FEAT_DIM, ACTION_FEAT_DIM, seed=0)
    inp = _make_inputs(N=6, M=4)
    R = inp['raw_3d']; V = inp['node_valid']; NF = inp['node_feats']
    ts = inp['timestep']; A = inp['adjmat']
    cust = inp['cust']; pred = inp['pred']; succ = inp['succ']; AF = inp['action_feats']

    report = {}
    Rj = jnp.array(R); Vj = jnp.array(V); NFj = jnp.array(NF)
    tsj = jnp.array(ts); Aj = jnp.array(A[0])
    custj = jnp.array(cust); predj = jnp.array(pred); succj = jnp.array(succ); AFj = jnp.array(AF)

    # --- 1. 参数继承 ---
    gd, train_state, frozen_state, other_state = partition(model)
    leak = sum(1 for _ in jax.tree_util.tree_leaves(other_state))
    report['param_manifest'] = param_manifest(model)
    report['n_train'] = sum(1 for _ in jax.tree_util.tree_leaves(train_state))
    report['n_frozen'] = sum(1 for _ in jax.tree_util.tree_leaves(frozen_state))
    report['n_leak'] = leak
    pre_gd, pre_state = nnx.split(backbone)
    pre_paths = _flatten(pre_state)
    all_bb = {}
    for p, leaf in jax.tree_util.tree_flatten_with_path(train_state)[0]:
        key = '.'.join(str(k.key if hasattr(k, 'key') else k) for k in p)
        if key.startswith('backbone.'):
            all_bb[key[len('backbone.'):]] = np.asarray(leaf)
    for p, leaf in jax.tree_util.tree_flatten_with_path(frozen_state)[0]:
        key = '.'.join(str(k.key if hasattr(k, 'key') else k) for k in p)
        if key.startswith('backbone.'):
            all_bb[key[len('backbone.'):]] = np.asarray(leaf)
    backbone_mismatch = [k for k, l in pre_paths.items()
                         if k not in all_bb or not np.array_equal(all_bb[k], l)]
    report['backbone_param_mismatch'] = backbone_mismatch
    report['check_1_param_inherit'] = (leak == 0 and not backbone_mismatch)

    # --- 2/3. 零残差 + 原输出继承 ---
    H_pre = np.asarray(backbone.encode(jnp.array(R), attn_options={}))
    L_pre = np.asarray(backbone.decode(jnp.array(H_pre), jnp.array(ts), jnp.array(A[0]),
                                       target='logit'))
    H0, H = model.encode_residual(Rj, Vj, NFj)
    C = np.asarray(H - H0)
    Z, L_train = model.decode_logits(H, tsj, Aj, Vj)
    scores_init, aux = model.score_actions(Rj, Vj, NFj, tsj, Aj, custj, predj, succj, AFj)
    delta_init = np.asarray(aux[4])
    report['max_abs_C'] = float(np.abs(C).max())
    report['max_abs_delta'] = float(np.abs(delta_init).max())
    report['max_abs_H_diff'] = float(np.abs(H - H_pre).max())
    off = ~np.eye(R.shape[1], dtype=bool)
    report['max_abs_L_offdiag'] = float(np.abs(L_pre[0] - np.asarray(L_train)[0])[off].max())
    all_finite = all(np.isfinite(x).all() for x in (C, H, Z, L_train, scores_init, delta_init))
    report['output_all_finite'] = bool(all_finite)
    report['check_2_zero_residual'] = (report['max_abs_C'] == 0.0
                                       and report['max_abs_delta'] == 0.0
                                       and report['output_all_finite'])
    report['check_3_output_inherit'] = report['max_abs_L_offdiag'] < 1e-5

    # --- 4. 动作继承（含 logp_diff 入通过条件） ---
    s_edge_pre = np.asarray(edge_insertion_scores(jnp.array(L_pre), custj, predj, succj))
    scores = np.asarray(scores_init)
    report['max_abs_score_diff'] = float(np.abs(scores - s_edge_pre).max())
    logp_pre = np.asarray(jax.nn.log_softmax(jnp.array(s_edge_pre), axis=-1))
    logp_train = np.asarray(jax.nn.log_softmax(jnp.array(scores), axis=-1))
    report['max_abs_logp_diff'] = float(np.abs(logp_pre - logp_train).max())
    report['argmax_pre'] = int(np.argmax(s_edge_pre[0]))
    report['argmax_train'] = int(np.argmax(scores[0]))
    report['check_4_action_inherit'] = (report['max_abs_score_diff'] < 1e-5
                                        and report['max_abs_logp_diff'] < 1e-5
                                        and report['argmax_pre'] == report['argmax_train'])
    # 边界：空车动作 (pred==succ==0) 只含两条新增边，不含删边
    c_empty, p_empty, s_empty = 4, 0, 0
    s_empty_via = float(edge_insertion_scores(jnp.array(L_pre), jnp.array([[c_empty]]),
                                              jnp.array([[p_empty]]), jnp.array([[s_empty]]))[0, 0])
    s_empty_expected = float(L_pre[0, p_empty, c_empty] + L_pre[0, c_empty, s_empty])
    report['empty_route_edge_err'] = abs(s_empty_via - s_empty_expected)
    report['check_4b_empty_route'] = report['empty_route_edge_err'] < 1e-6

    # --- 5. 梯度与冻结 ---
    frozen_before = _flatten(frozen_state)
    train_before = _flatten(train_state)

    def loss_fn(ts_):
        m = nnx.merge(gd, ts_, frozen_state)
        s, _ = m.score_actions(Rj, Vj, NFj, tsj, Aj, custj, predj, succj, AFj)
        lp = jax.nn.log_softmax(s, axis=-1)
        return -lp[0, 2]

    loss0, grads = jax.value_and_grad(loss_fn)(train_state)
    tx = optax.adam(1e-3)
    opt_state = tx.init(train_state)
    updates, opt_state = tx.update(grads, opt_state, train_state)
    train_state_up = optax.apply_updates(train_state, updates)
    report['grad_nonzero_decoder'] = _grad_nonzero_count(grads, 'backbone.decoder')
    report['grad_nonzero_c_adapter'] = _grad_nonzero_count(grads, 'c_adapter')
    report['grad_nonzero_delta_head'] = _grad_nonzero_count(grads, 'delta_head')
    # 从更新后组装的模型重提取冻结参数，逐元素不变
    m_up = nnx.merge(gd, train_state_up, frozen_state)
    _, _, frozen_up, _ = partition(m_up)
    frozen_after = _flatten(frozen_up)
    frozen_changed = [k for k in frozen_before
                      if k not in frozen_after or not np.array_equal(frozen_before[k], frozen_after[k])]
    # 应训练模块确实更新（非仅梯度非零）
    train_after = _flatten(train_state_up)
    train_updated = [k for k in train_before
                     if k in train_after and not np.array_equal(train_before[k], train_after[k])]
    report['frozen_changed'] = frozen_changed
    report['n_train_updated'] = len(train_updated)
    # 独立 M-pre 不变（backbone 是独立对象，从未进优化器）
    pre_paths_after = _flatten(nnx.split(backbone)[1])
    pre_changed = [k for k in pre_paths
                   if k not in pre_paths_after or not np.array_equal(pre_paths[k], pre_paths_after[k])]
    report['mpre_changed'] = pre_changed
    report['check_5_gradient_frozen'] = (report['grad_nonzero_decoder'] > 0
                                         and report['grad_nonzero_c_adapter'] > 0
                                         and report['grad_nonzero_delta_head'] > 0
                                         and not frozen_changed
                                         and len(train_updated) > 0
                                         and not pre_changed)

    # --- 6. 附加特征通路（改查 log-prob / 分数差，非原始分数）+ 第二次反向传播 ---
    m2 = nnx.merge(gd, train_state_up, frozen_state)
    NF_pert = NFj.at[0, 1, 4].add(0.5)   # 扰动冷链品质 initial_quality（连续字段）
    scores_a, _ = m2.score_actions(Rj, Vj, NFj, tsj, Aj, custj, predj, succj, AFj)
    scores_b, _ = m2.score_actions(Rj, Vj, NF_pert, tsj, Aj, custj, predj, succj, AFj)
    lpa = np.asarray(jax.nn.log_softmax(scores_a, axis=-1))
    lpb = np.asarray(jax.nn.log_softmax(scores_b, axis=-1))
    report['aux_feat_logp_diff'] = float(np.abs(lpa - lpb).max())
    report['aux_feat_score_gap_diff'] = float(
        np.abs((np.asarray(scores_a[0]) - np.asarray(scores_a[0])[1])
               - (np.asarray(scores_b[0]) - np.asarray(scores_b[0])[1])).max())
    # 第二次反向传播：确认梯度进入适配内部层（layers.0/1，非仅出口 layers.2）
    def loss_fn2(ts_):
        m = nnx.merge(gd, ts_, frozen_state)
        s, _ = m.score_actions(Rj, Vj, NFj, tsj, Aj, custj, predj, succj, AFj)
        lp = jax.nn.log_softmax(s, axis=-1)
        return -lp[0, 2]
    _, grads2 = jax.value_and_grad(loss_fn2)(train_state_up)
    report['grad_internal_c_adapter'] = _grad_nonzero_count(
        grads2, 'c_adapter', exclude_exit=True)
    report['check_6_aux_pathway'] = (report['aux_feat_logp_diff'] > 1e-6
                                     and report['grad_internal_c_adapter'] > 0)

    # --- 7. padding 一致性（B=2 不同有效长度 + 不同有限内容；init 与更新后模型） ---
    def _run_pad(model_m, raw, valid, feats, a0, custv, predv, succv, afv):
        s, _ = model_m.score_actions(jnp.array(raw), jnp.array(valid), jnp.array(feats),
                                     tsj, jnp.array(a0), jnp.array(custv), jnp.array(predv),
                                     jnp.array(succv), jnp.array(afv))
        return np.asarray(s), np.asarray(jax.nn.log_softmax(s, axis=-1))

    # sample A: 6 有效；sample B: 只 5 有效（mask 节点 5），动作只引用 1..4
    custB = np.array([[1, 2, 3, 4]], np.int32)
    predB = np.array([[0, 4, 2, 0]], np.int32)
    succB = np.array([[4, 0, 0, 0]], np.int32)
    N = 6
    rawA = R.copy(); rawB = R.copy()
    valA = np.ones((1, N), bool); valB = np.ones((1, N), bool); valB[0, 5] = False
    # 单样本 unpadded 基准
    sA0, lpA0 = _run_pad(model, rawA, valA, NF, A[0], cust, pred, succ, AF)
    sB0, lpB0 = _run_pad(model, rawB, valB, NF, A[0], custB, predB, succB, AF)
    # B=2 批（都 pad 到 8；sample B 用有限填充 0.7）
    P = 2
    def _pad(x, val):
        return np.pad(x, ((0, 0), (0, P), (0, 0)), constant_values=val)
    raw_batch = np.concatenate([_pad(rawA, 0.0), _pad(rawB, 0.7)], axis=0)
    val_batch = np.concatenate([np.pad(valA, ((0,0),(0,P)), constant_values=False),
                                np.pad(valB, ((0,0),(0,P)), constant_values=False)], axis=0)
    nf_batch = np.concatenate([_pad(NF, 0.0), _pad(NF, 0.7)], axis=0)
    A_batch = np.concatenate([np.pad(A, ((0,0),(0,P),(0,P)), constant_values=0.0),
                              np.pad(A, ((0,0),(0,P),(0,P)), constant_values=0.0)], axis=0)
    ts_batch = np.array([0.6, 0.6], np.float32)
    cust_batch = np.concatenate([cust, custB], axis=0)
    pred_batch = np.concatenate([pred, predB], axis=0)
    succ_batch = np.concatenate([succ, succB], axis=0)
    af_batch = np.concatenate([AF, AF], axis=0)
    s_batch, lp_batch = _run_pad(model, raw_batch, val_batch, nf_batch, A_batch,  # B=2
                                 cust_batch, pred_batch, succ_batch, af_batch)
    report['pad_score_diff'] = float(np.abs(s_batch - np.concatenate([sA0, sB0], axis=0)).max())
    report['pad_logp_diff'] = float(np.abs(lp_batch - np.concatenate([lpA0, lpB0], axis=0)).max())
    report['pad_argmax_A'] = (int(np.argmax(s_batch[0])) == int(np.argmax(sA0[0])))
    report['pad_argmax_B'] = (int(np.argmax(s_batch[1])) == int(np.argmax(sB0[0])))
    # 更新后模型同样验证
    s_batch2, lp_batch2 = _run_pad(m2, raw_batch, val_batch, nf_batch, A_batch,
                                   cust_batch, pred_batch, succ_batch, af_batch)
    sA2, _ = _run_pad(m2, rawA, valA, NF, A[0], cust, pred, succ, AF)
    sB2, _ = _run_pad(m2, rawB, valB, NF, A[0], custB, predB, succB, AF)
    report['pad_score_diff_upd'] = float(np.abs(s_batch2 - np.concatenate([sA2, sB2], axis=0)).max())
    report['check_7_padding'] = (report['pad_score_diff'] < 1e-4
                                 and report['pad_logp_diff'] < 1e-4
                                 and report['pad_argmax_A'] and report['pad_argmax_B']
                                 and report['pad_score_diff_upd'] < 1e-4)

    # --- 汇总 ---
    checks = {k: v for k, v in report.items() if k.startswith('check_')}
    report['ALL_PASS'] = all(checks.values())
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print('=== MpreTrainedModel inheritance smoke ===')
    for k, v in report.items():
        if isinstance(v, (int, float, str, bool, list)) and not k.startswith('param_manifest'):
            print(f'  {k} = {v}')
    print(f'  ALL_PASS = {report["ALL_PASS"]}')
    print(f'saved: {args.out}/report.json')
    sys.exit(0 if report['ALL_PASS'] else 1)


if __name__ == '__main__':
    main()
