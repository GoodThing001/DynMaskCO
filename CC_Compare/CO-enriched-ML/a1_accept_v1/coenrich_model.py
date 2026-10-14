"""官方 ML-CO 预测器的 NumPy 复刻（部署侧零 TF 依赖）。

为什么
------
官方 `training/prediction.py::NeuralNetwork` 是 TF2/Keras SavedModel；A-v1 在线 provider
必须只依赖 numpy（服务器 `cc_compare` env 无 TF，且不能为对比方法改动正在跑正式批次的
env）。因此 `export_official_weights.py` 在**独立 venv**（tensorflow-cpu==2.15.1）里把
权重导出为 npz + 层结构 JSON，本模块按层结构逐层复刻前向。

保真性
------
`export_official_weights.py` 同时落盘固定随机输入的官方前向输出
（`official_forward_check.npz`）；`verify_against_official()` 用本模块前向逐点对账，
max|Δ| 必须 ~0（float32 舍入量级）。烟测脚本会打印该数值。
"""
from __future__ import annotations

import json
import os

import numpy as np


class NumpyMLP:
    """按 Keras Sequential 层顺序复刻的 Dense 前向（float32）。"""

    def __init__(self, npz_path, struct_path=None):
        data = np.load(npz_path, allow_pickle=False)
        self.weights = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        if struct_path is None:
            struct_path = os.path.join(os.path.dirname(os.path.abspath(npz_path)),
                                       'official_model_struct.json')
        with open(struct_path) as f:
            self.struct = json.load(f)
        self.layers = []
        for spec in self.struct['layers']:
            name = spec['name']
            key = self._find(name, 'kernel')
            if key is None:
                raise KeyError('层 %s 找不到 kernel 权重；现有键 %s'
                               % (name, sorted(self.weights)))
            W = self.weights[key]
            bkey = self._find(name, 'bias')
            b = self.weights[bkey] if bkey is not None else None
            self.layers.append({
                'name': name, 'W': W, 'b': b,
                'activation': spec.get('activation'),
                'class': spec.get('class'),
            })
        self.input_dim = int(self.struct['input_shape'][-1])
        if self.layers and self.layers[0]['W'].shape[0] != self.input_dim:
            raise ValueError('输入维度不一致: struct=%d 首层=%d'
                             % (self.input_dim, self.layers[0]['W'].shape[0]))

    def _find(self, layer_name, suffix):
        exact = '%s/%s' % (layer_name, suffix)
        tail = '/%s' % exact
        hits = [k for k in self.weights if k == exact or k.endswith(tail)]
        if len(hits) > 1:
            raise KeyError('权重键不唯一: %s -> %s' % (exact, hits))
        return hits[0] if hits else None

    def predict(self, X):
        """X: (n, input_dim) → (n,) 逐节点 profit 预测（float32）。"""
        y = np.asarray(X, dtype=np.float32)
        if y.ndim == 1:
            y = y[None, :]
        for layer in self.layers:
            if layer['class'] != 'Dense':
                raise NotImplementedError('非 Dense 层未复刻: %s' % layer['class'])
            y = y @ layer['W']
            if layer['b'] is not None:
                y = y + layer['b']
            act = layer['activation']
            if act == 'relu':
                y = np.maximum(y, 0.0)
            elif act in (None, 'linear'):
                pass
            else:
                raise NotImplementedError('未复刻激活: %s' % act)
        return y.reshape(-1)

    def verify_against_official(self, check_npz):
        """对账官方 TF 前向；返回 max|Δ|。"""
        d = np.load(check_npz, allow_pickle=False)
        x, y_ref = d['x'], d['y']
        y = self.predict(x).astype(np.float64)
        return float(np.max(np.abs(y - y_ref.astype(np.float64))))
