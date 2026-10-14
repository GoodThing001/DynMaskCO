"""Learning-to-Delegate（L2D）权重可加载性 / 可推理性探针（可执行的最小验证）。

背景（复核 A1_ASSESSMENT.md）
----------------------------
旧评估称「本地核到 3 个回归权重；LKH/HGS 子求解器未下载；官方 generations/ 未下载」。
**本轮复核发现该记录的资产清单不准确**：
  - `CC_Compare/Learning-to-Delegate/lkh3/LKH-3.0.4/`（完整 LKH-3 源码 + Makefile）
    与 `hgs/HGS-CVRP/Program/`（完整 HGS 源码 + Makefile）**都在本地**；
  - `CC_Compare/Omni-VRP/L2D/generations/` 亦在本地（含 mixed_merge / clustered 等
    问题与 `subproblem_selection_hgs/{val,test}_subproblems.npz`）。
因此「缺件」不是主要障碍，障碍在**机制与规模**（见 RESULTS.md 的可比性判定）。

本脚本要回答的可执行问题
------------------------
1. 3 个官方回归权重（`exps/{regression_model_512,regression_model_2048,
   regression_maml_512}/models/40000.pth`）能否**真实加载**进官方 `SubproblemNetwork`？
   （逐键 strict 加载 + missing/unexpected 清单）
2. 能否**真实推理**？在官方 `subproblem_selection_hgs/val_subproblems.npz` 上跑前向，
   与 LKH/HGS 距离标签比 MAE，并与「不做任何委托」的 `prev_dists` 基线和
   `mean_lkh_dist` 常数基线比较 → 判断模型是否真的携带信号。
3. 逐检查点打印加载/推理证据（供 RESULTS.md 引用）。

用法（服务器，cc_compare env：torch 2.11 + torch_geometric 2.8）：
    cd /home/hzeng/project/MASKCO-Main && /home/hzeng/envs/cc_compare/bin/python \
        CC_Compare/Learning-to-Delegate/a1_accept_v1/probe_l2d_weights.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CC = os.path.dirname(os.path.dirname(_HERE))                 # CC_Compare/
_L2D_CODE = os.path.join(_CC, 'Omni-VRP', 'L2D', 'code')
_L2D_ROOT = os.path.join(_CC, 'Omni-VRP', 'L2D')
if _L2D_CODE not in sys.path:
    sys.path.insert(0, _L2D_CODE)


def _stub_missing(name, probe, attrs):
    """官方 `supervised.py` 顶层 import `torch.utils.tensorboard`（训练期日志用），
    而服务器 `cc_compare` env 未装 tensorboard。**不为对比方法改动正在跑正式批次的
    conda env**（装 tensorboard 会连带升级 protobuf/grpc，风险不可控），因此在本进程内
    用轻量 stub 占位：推理路径完全不使用 SummaryWriter。"""
    import importlib.util
    import types
    try:
        present = importlib.util.find_spec(probe) is not None
    except (ImportError, ValueError):
        present = False
    if present:
        return False
    mod = types.ModuleType(name)
    for a in attrs:
        setattr(mod, a, object)
    sys.modules[name] = mod
    parts = name.split('.')
    for i in range(1, len(parts)):
        parent = '.'.join(parts[:i])
        if parent in sys.modules:
            setattr(sys.modules[parent], parts[i], mod)
    return True


_STUBBED = _stub_missing('torch.utils.tensorboard', 'tensorboard',
                         ['SummaryWriter'])


CKPTS = [
    ('regression_model_512', 'supervised'),
    ('regression_model_2048', 'supervised'),
    ('regression_maml_512', 'supervised_maml'),
]


def build_args(cfg_path, train_dir, dataset_dir, module, device='cpu'):
    """用官方 parser 构造 args（config.yaml 覆盖默认值）。"""
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    argv = [dataset_dir, train_dir,
            '--ptype', 'CVRP', '--device', device,
            '--d_hidden', '128', '--n_layers', '6', '--transformer_heads', '8',
            '--step', '40000', '--loss', 'MAE']
    if cfg.get('fit_subproblem'):
        argv.append('--fit_subproblem')
    for k in ('fc_only', 'use_prev_dist_feature', 'normalize_features',
              'use_layer_norm', 'use_count_feature', 'use_x_fc',
              'augment_rotate', 'augment_flip'):
        if cfg.get(k):
            argv.append('--' + k)
    if cfg.get('n_layers'):
        argv += ['--n_layers', str(cfg['n_layers'])]
    if cfg.get('transformer_heads'):
        argv += ['--transformer_heads', str(cfg['transformer_heads'])]
    if cfg.get('gnn_module'):
        argv += ['--gnn_module', str(cfg['gnn_module'])]
    return module.parser.parse_args(argv)


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    return r


def spearman(a, b):
    x, y = rankdata(np.asarray(a, float)), rankdata(np.asarray(b, float))
    x -= x.mean()
    y -= y.mean()
    d = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=_L2D_ROOT)
    ap.add_argument('--n-eval', type=int, default=256)
    ap.add_argument('--device', default='cpu')
    args_cli = ap.parse_args()

    import torch
    print('torch', torch.__version__)
    print('tensorboard stub 生效:', _STUBBED)
    import supervised
    import supervised_maml
    mods = {'supervised': supervised, 'supervised_maml': supervised_maml}

    val_npz = os.path.join(args_cli.root, 'generations', 'mixed_merge',
                           'subproblem_selection_hgs', 'val_subproblems.npz')
    dataset_dir = os.path.join(args_cli.root, 'generations', 'mixed_merge',
                               'subproblem_selection_hgs')
    print('val_subproblems:', val_npz, os.path.isfile(val_npz))

    rows = []
    for name, modname in CKPTS:
        train_dir = os.path.join(args_cli.root, 'exps', name)
        cfg = os.path.join(train_dir, 'config.yaml')
        ckpt = os.path.join(train_dir, 'models', '40000.pth')
        print('\n===== %s =====' % name)
        print('config=%s exists=%s' % (cfg, os.path.isfile(cfg)))
        print('ckpt  =%s exists=%s size=%s'
              % (ckpt, os.path.isfile(ckpt),
                 os.path.getsize(ckpt) if os.path.isfile(ckpt) else -1))
        mod = mods[modname]
        a = build_args(cfg, train_dir, dataset_dir, mod, device=args_cli.device)
        a.n_steps = 40000
        d = mod.load_subproblem_data(val_npz)
        print('d: N=%d d_node=%d' % (d.N, d.d_node))
        net = mod.SubproblemNetwork(a, d).to(args_cli.device)
        blob = torch.load(ckpt, map_location=args_cli.device, weights_only=False)
        print('ckpt 键: %s' % sorted(blob.keys()))
        sd = blob['net']
        print('ckpt step=%s  参数张量数=%d' % (blob.get('step'), len(sd)))
        # 与官方架构逐键比对
        own = net.state_dict()
        missing = [k for k in own if k not in sd]
        unexpected = [k for k in sd if k not in own]
        shape_mismatch = [(k, tuple(own[k].shape), tuple(sd[k].shape))
                          for k in own if k in sd and tuple(own[k].shape) != tuple(sd[k].shape)]
        print('missing=%d unexpected=%d shape_mismatch=%d'
              % (len(missing), len(unexpected), len(shape_mismatch)))
        if missing:
            print('  missing[:8] =', missing[:8])
        if unexpected:
            print('  unexpected[:8] =', unexpected[:8])
        if shape_mismatch:
            print('  shape_mismatch[:4] =', shape_mismatch[:4])
        info = net.load_state_dict(sd, strict=(not (missing or unexpected or shape_mismatch)))
        print('load_state_dict ->', info)

        # ---------- 真实推理 ----------
        n = min(args_cli.n_eval, d.N)
        idxs = list(range(n))
        prep = mod.get_prepare_subproblem(a, d)
        batch = prep(idxs)
        labels = np.asarray(batch.labels.detach().cpu().numpy(), float).reshape(-1)
        prev = np.asarray(batch.prev_dists.detach().cpu().numpy(), float).reshape(-1)
        batch['labels'] = None                      # 触发 preds 返回分支
        net.eval()
        with torch.no_grad():
            preds = net(batch).detach().cpu().numpy().reshape(-1)
        mae = float(np.mean(np.abs(preds - labels)))
        mae_prev = float(np.mean(np.abs(prev - labels)))
        mae_const = float(np.mean(np.abs(float(d.lkh_dists.mean()) - labels)))
        rho = spearman(preds, labels)
        rho_prev = spearman(prev, labels)
        # 确定性
        batch2 = prep(idxs)
        batch2['labels'] = None
        with torch.no_grad():
            preds2 = net(batch2).detach().cpu().numpy().reshape(-1)
        det = float(np.max(np.abs(preds - preds2)))
        print('推理 n=%d  MAE(model)=%.4f  MAE(prev_dists 基线)=%.4f  MAE(常数均值)=%.4f'
              % (n, mae, mae_prev, mae_const))
        print('       Spearman(model)=%.4f  Spearman(prev_dists)=%.4f  确定性 max|Δ|=%.3e'
              % (rho, rho_prev, det))
        print('       预测/标签样例 preds[:5]=%s labels[:5]=%s'
              % (np.round(preds[:5], 2).tolist(), np.round(labels[:5], 2).tolist()))
        rows.append(dict(name=name, module=modname, loaded=True,
                         missing=len(missing), unexpected=len(unexpected),
                         shape_mismatch=len(shape_mismatch), n=n,
                         mae=mae, mae_prev=mae_prev, mae_const=mae_const,
                         rho=rho, rho_prev=rho_prev, deterministic=det))

    print('\n===== 汇总 =====')
    for r in rows:
        print('%-24s loaded=%s miss=%d unexp=%d  MAE %.3f (prev %.3f, const %.3f)  rho %.3f (prev %.3f)'
              % (r['name'], r['loaded'], r['missing'], r['unexpected'],
                 r['mae'], r['mae_prev'], r['mae_const'], r['rho'], r['rho_prev']))
    print('\nPROBE_DONE loaded=%d/%d' % (sum(1 for r in rows if r['loaded']), len(CKPTS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
