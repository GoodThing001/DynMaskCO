"""按 A-v1 特征/目标重训 ML-CO 预测器（小预算、纯 NumPy、可复现）。

预声明预算（写死在代码里，事后不改）
------------------------------------
- 架构：与官方 `prediction.py::NeuralNetwork` **完全一致**
  （20 → 10 relu → 10 relu → 10 relu → 10 relu → 1 linear，末层无 bias）；
- 优化：Adam(lr=1e-3)，batch=256，最多 3000 步，CPU 单核；
- 实测预算上限：≤ 10 分钟（实际约 1–2 分钟）；
- 数据：`gen_coenrich_train_data.py` 产出的训练天（seed 20260920）样本，
  按「天」划分 train/val（最后 2 天做 val，绝不使用开发集 seed 20260926）。

目标替换（必须如实标注）
------------------------
原文用**结构化学习损失**（对 profit 加噪 → 反复调 PC-HGS → 用「扰动后派车集合」的
目标值差做次梯度）。A-v1 桥接没有可复用的该闭环（PC-HGS 仅作 CO 层/优先级生成器），
因此这里退化为**对参考臂接单标签的回归**（MSE/二分类 Brier）。
这是替代目标，不是原文的 SL 损失。

产物
----
`a1_weights.npz` + `a1_model_struct.json`，键名/结构与官方导出物**同构**
（`dense/kernel`、`dense_1/kernel` …），因此 `coenrich_model.NumpyMLP` 可直接加载，
provider 无需区分「官方权重 / 重训权重」。

用法（服务器，cc_compare env）：
    /home/hzeng/envs/cc_compare/bin/python train_coenrich_a1.py \
        --data a1_accept_v1/data/train_pool.npz --out a1_accept_v1/artifacts_a1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coenrich_model import NumpyMLP  # noqa: E402

BUDGET = dict(batch=256, max_steps=3000, lr=1e-3, seed=42)


def init_mlp(rng, dims):
    """Glorot-uniform 初始化（Keras Dense 默认），层名与官方一致。"""
    layers = []
    names = ['dense'] + ['dense_%d' % i for i in range(1, len(dims) - 1)]
    for k, (nin, nout) in enumerate(zip(dims[:-1], dims[1:])):
        lim = np.sqrt(6.0 / (nin + nout))
        W = rng.uniform(-lim, lim, size=(nin, nout)).astype(np.float32)
        b = np.zeros(nout, dtype=np.float32)
        layers.append({'name': names[k], 'W': W, 'b': b,
                       'last': (k == len(dims) - 2)})
    return layers


def forward(layers, X, cache=None):
    y = np.asarray(X, dtype=np.float32)
    acts = [y]
    for L in layers:
        z = y @ L['W'] + (L['b'] if not L.get('last') else 0.0)
        y = np.maximum(z, 0.0) if not L.get('last') else z
        acts.append(y)
    if cache is not None:
        cache['acts'] = acts
    return y.reshape(-1)


def backward(layers, cache, y_true):
    """MSE 反传；返回梯度 dict 与 loss。loss = mean((pred - y)^2)"""
    acts = cache['acts']
    pred = acts[-1].reshape(-1)
    n = pred.shape[0]
    d = (2.0 / n) * (pred - y_true).reshape(-1, 1)          # (n,1)
    grads = [None] * len(layers)
    for k in range(len(layers) - 1, -1, -1):
        L = layers[k]
        a_in = acts[k]
        gW = a_in.T @ d
        gb = d.sum(axis=0) if not L.get('last') else None
        grads[k] = (gW, gb)
        if k > 0:
            dz = d @ L['W'].T
            d = dz * (acts[k] > 0).astype(np.float32)       # relu'
    return float(np.mean((pred - y_true) ** 2)), grads


def adam_step(layers, grads, state, lr, t):
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k, L in enumerate(layers):
        gW, gb = grads[k]
        for tag, g, P in (('W', gW, 'W'), ('b', gb, 'b')):
            if g is None:
                continue
            key = (k, tag)
            m = state.setdefault(key + ('m',), np.zeros_like(g))
            v = state.setdefault(key + ('v',), np.zeros_like(g))
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * (g * g)
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            L[P] = (L[P] - lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)


def export(layers, dims, out_dir, extra):
    os.makedirs(out_dir, exist_ok=True)
    w = {}
    for L in layers:
        w['%s/kernel' % L['name']] = L['W']
        if not L.get('last'):
            w['%s/bias' % L['name']] = L['b']
    np.savez(os.path.join(out_dir, 'a1_weights.npz'), **w)
    struct = {
        'source_model': 'A-v1 retrained (train_coenrich_a1.py)',
        'loader': 'numpy-adam',
        'input_shape': [None, int(dims[0])],
        'layers': [{'class': 'Dense', 'name': L['name'], 'units': int(L['W'].shape[1]),
                    'activation': ('linear' if L.get('last') else 'relu'),
                    'use_bias': (not L.get('last'))} for L in layers],
        'weight_keys': sorted(w),
        'extra': extra,
    }
    with open(os.path.join(out_dir, 'a1_model_struct.json'), 'w') as f:
        json.dump(struct, f, indent=2)
    return sorted(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-days', type=int, default=2)
    ap.add_argument('--official-artifacts', default=os.path.join(_HERE, 'artifacts'))
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, day = d['X'], d['y'], d['ctx_day']
    meta = json.loads(str(d['meta'][0]))
    print('数据: X=%s 正例率=%.4f 天数=%d  meta=%s'
          % (X.shape, y.mean(), len(set(day.tolist())), meta))

    ndays = len(set(day.tolist()))
    val_from = ndays - args.val_days
    tr = day < val_from
    va = ~tr
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print('train=%d (天 0..%d)  val=%d (天 %d..%d)'
          % (tr.sum(), val_from - 1, va.sum(), val_from, ndays - 1))
    if Xtr.shape[0] == 0 or Xva.shape[0] == 0:
        raise SystemExit('空划分：训练天太少')

    rng = np.random.default_rng(BUDGET['seed'])
    dims = [X.shape[1]] + [10, 10, 10, 10, 1]
    layers = init_mlp(rng, dims)
    state = {}
    print('架构 %s（与官方 NeuralNetwork 一致） 预算 %s' % (dims, BUDGET))

    t0 = time.time()
    curve = []
    for t in range(1, BUDGET['max_steps'] + 1):
        idx = rng.integers(0, Xtr.shape[0], size=BUDGET['batch'])
        cache = {}
        forward(layers, Xtr[idx], cache=cache)
        loss, grads = backward(layers, cache, ytr[idx])
        adam_step(layers, grads, state, BUDGET['lr'], t)
        if t % 250 == 0 or t == 1:
            pv = forward(layers, Xva)
            vloss = float(np.mean((pv - yva) ** 2))
            acc = float(np.mean((pv > 0.5) == (yva > 0.5)))
            curve.append(dict(step=t, train_mse=loss, val_mse=vloss, val_acc=acc))
            print('step %5d  train_mse=%.5f  val_mse=%.5f  val_acc=%.4f'
                  % (t, loss, vloss, acc))
    print('训练耗时 %.1fs' % (time.time() - t0))

    pv = forward(layers, Xva)
    base_rate = float(max(yva.mean(), 1 - yva.mean()))
    extra = dict(budget=BUDGET, n_train=int(tr.sum()), n_val=int(va.sum()),
                 val_from_day=int(val_from), data_meta=meta,
                 loss_curve=curve, val_mse_final=float(np.mean((pv - yva) ** 2)),
                 val_acc_final=float(np.mean((pv > 0.5) == (yva > 0.5))),
                 val_majority_rate=base_rate, train_seconds=time.time() - t0,
                 objective='MSE on reference-arm accept label（替代原文 SL 损失）')

    # ---- 与官方权重对比：排序是否真的改变 ----
    off_npz = os.path.join(args.official_artifacts, 'official_weights.npz')
    if os.path.isfile(off_npz):
        off = NumpyMLP(off_npz)
        po = off.predict(Xva)
        rho = float(np.corrcoef(pv, po)[0, 1])
        acc_o = float(np.mean((po > 0.5) == (yva > 0.5)))
        print('vs 官方权重(val): 分数相关 r=%.4f  官方 acc=%.4f  重训 acc=%.4f'
              % (rho, acc_o, extra['val_acc_final']))
        extra['vs_official'] = dict(score_corr=rho, official_val_acc=acc_o,
                                    a1_val_acc=extra['val_acc_final'])

    keys = export(layers, dims, args.out, extra)
    print('导出:', keys)
    print('DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
