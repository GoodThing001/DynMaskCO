"""把官方 ML-CO 预测器（TensorFlow SavedModel）的权重导出为 npz（+ 层结构 + 前向对账）。

为什么需要这一步
----------------
官方仓库（`CC_Compare/CO-enriched-ML/training/prediction.py` 的 `NeuralNetwork`）是
TF2/Keras SavedModel。A-v1 的在线 provider 必须只依赖 numpy（服务器 `cc_compare`
env 无 TF，且 F1 正式批次正在运行，不能改动该 env）。因此：
  1. 在**独立 venv**（`/home/hzeng/venvs/coenrich`，tensorflow-cpu==2.15.1）里加载官方
     SavedModel；
  2. 导出全部权重到 `official_weights.npz` + 层结构 JSON；
  3. 在同一脚本里用固定随机输入做官方前向，落盘 `(x, y)` 供 `coenrich_model.py` 的
     **NumPy 前向**逐点对账。

加载路径（两条，按可用性自动选择）
--------------------------------
  (a) `tf.keras.models.load_model`：需要 Keras 能解析 `keras_metadata.pb`。
      **本资产由 Keras 2.11 写出，Keras 2.15 解析失败**（`Cannot parse keras metadata`），
      PyPI 也没有 cp311 的 tensorflow 2.11 wheel → 本机走不通。
  (b) `tf.saved_model.load`：不解析 keras_metadata，直接读 SavedModel 的变量与
      `serving_default` 签名 —— **本机实际使用的路径**。层结构由变量路径推得
      （隐藏层 relu、末层无 bias 线性，与官方 `prediction.py::NeuralNetwork.create_model`
      的 4×Dense(10,relu)+Dense(1,use_bias=False) 一致），并在日志中打印核对。

不修改第三方原文件：只读 `training/models/<name>`。

用法（服务器）：
    /home/hzeng/venvs/coenrich/bin/python \
        CC_Compare/CO-enriched-ML/a1_accept_v1/export_official_weights.py \
        --model CC_Compare/CO-enriched-ML/training/models/<name> \
        --out  CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np


def _strip_var(name):
    """变量名归一化为 '<layer>/<kernel|bias>'。

    实测本资产经 `tf.saved_model.load` 暴露的名字形如 `dense/kernel:0`
    （Keras 路径下形如 `sequential/dense/kernel`）；两种都要能解析。
    """
    n = name
    if n.endswith(':0'):
        n = n[:-2]
    n = n.split('/.ATTRIBUTES')[0]
    return n


_LAYER_RE = re.compile(r'^(?:.*/)?([^/]+)/(kernel|bias)$')


def load_via_keras(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path, compile=False)
    layers = []
    for layer in model.layers:
        cfg = layer.get_config()
        layers.append({'class': type(layer).__name__, 'name': cfg.get('name'),
                       'units': cfg.get('units'), 'activation': cfg.get('activation'),
                       'use_bias': cfg.get('use_bias')})
    weights = {}
    for v in model.weights:
        key = getattr(v, 'path', None) or _strip_var(v.name)
        weights[key] = np.asarray(v.numpy())
    inp = model.input_shape
    fwd = lambda x: np.asarray(model(x, training=False)).reshape(-1)  # noqa: E731
    return layers, weights, [int(s) if s is not None else None for s in inp], fwd, 'keras.load_model'


def load_via_savedmodel(path):
    import tensorflow as tf
    obj = tf.saved_model.load(path)
    weights = {}
    for v in obj.variables:
        weights[_strip_var(v.name)] = np.asarray(v.numpy())
    if not weights:
        raise RuntimeError('SavedModel 未暴露任何变量')

    def _layer_key(k):
        m = _LAYER_RE.match(k)
        return (m.group(1), m.group(2)) if m else (None, None)

    order, seen = [], set()
    for k in weights:
        name, kind = _layer_key(k)
        if name and name not in seen:
            seen.add(name)
            order.append(name)

    def _sort_key(name):
        m = re.match(r'^(.*?)(?:_(\d+))?$', name)
        return (m.group(1), int(m.group(2) or 0))

    order.sort(key=_sort_key)
    layers = []
    for i, name in enumerate(order):
        W = [v for k, v in weights.items() if _layer_key(k) == (name, 'kernel')]
        b = [v for k, v in weights.items() if _layer_key(k) == (name, 'bias')]
        if not W:
            continue
        is_last = (i == len(order) - 1)
        layers.append({'class': 'Dense', 'name': name, 'units': int(W[0].shape[1]),
                       'activation': 'linear' if is_last else 'relu',
                       'use_bias': bool(b)})
    # 输入维度 = 首层 kernel 的第 0 维
    first = [weights[k] for k in weights if _layer_key(k) == (order[0], 'kernel')][0]
    in_dim = int(first.shape[0])

    fn = obj.signatures['serving_default']
    in_name = list(fn.structured_input_signature[1])[0]
    print('serving_default 输入参数名 =', in_name)

    def fwd(x):
        out = fn(**{in_name: tf.constant(np.asarray(x, dtype=np.float32))})
        vals = [np.asarray(v) for v in out.values()]
        vals = [v for v in vals if v.size == len(x)]
        if len(vals) != 1:
            raise RuntimeError('signature 输出不唯一: %s'
                               % [np.asarray(v).shape for v in out.values()])
        return vals[0].reshape(-1)

    return layers, weights, [None, in_dim], fwd, 'saved_model.load'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-verify', type=int, default=257)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    import tensorflow as tf

    os.makedirs(args.out, exist_ok=True)
    print('tf', tf.__version__)
    try:
        layers, weights, input_shape, fwd, how = load_via_keras(args.model)
    except Exception as exc:  # noqa: BLE001
        print('keras.load_model 不可用（%s: %s）→ 回退 saved_model.load'
              % (type(exc).__name__, str(exc).splitlines()[0][:160]))
        layers, weights, input_shape, fwd, how = load_via_savedmodel(args.model)

    print('加载路径:', how)
    print('layers:', json.dumps(layers, indent=2))
    print('input_shape:', input_shape)
    for k in sorted(weights):
        print('  weight %-44s %s %s' % (k, weights[k].shape, weights[k].dtype))

    # 官方架构核对（prediction.py::NeuralNetwork.create_model）
    shapes = [weights[k].shape for k in sorted(weights) if k.endswith('/kernel')]
    print('kernel 形状序列:', shapes)

    np.savez(os.path.join(args.out, 'official_weights.npz'), **weights)
    with open(os.path.join(args.out, 'official_model_struct.json'), 'w') as f:
        json.dump({'source_model': os.path.abspath(args.model),
                   'tf_version': tf.__version__, 'loader': how,
                   'input_shape': input_shape, 'layers': layers,
                   'weight_keys': sorted(weights)}, f, indent=2)

    dim = int(input_shape[-1])
    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.n_verify, dim)).astype(np.float32)
    y = np.asarray(fwd(x)).reshape(args.n_verify)
    np.savez(os.path.join(args.out, 'official_forward_check.npz'),
             x=x, y=y.astype(np.float64), seed=np.array([args.seed]))
    print('forward check saved; y[:5] =', y[:5])
    print('OK')


if __name__ == '__main__':
    main()
