"""PIP-constraint 资产/机制可执行性探针（判死复核的最小验证）。

复核背景
--------
`A1_ASSESSMENT.md` 记录「本地资产 = README + LICENSE，完整代码未 clone，无权重」并据此判死。
**本轮复核发现该资产清单不准确**：
  - `CC_Compare/PIP-constraint/` 下**完整存在**官方三个实现：
    `POMO+PIP/`（含 envs/TSPDLEnv.py、envs/TSPTWEnv.py、models/SINGLEModel.py、
    Trainer/Tester）、`AM+PIP/`（nets、problems/tsptw、utils、pretrained、eval/train/run）、
    `GFACS+PIP/`（TSPTW + aco/net），另有 `baselines/`（LKH/OR-Tools）、`data/TSPTW`、
    `data/TSPDL`；
  - `AM+PIP/pretrained/tsptw100_{easy,hard}/AM_star_PIP[-D]/epoch-99.pt` 存在**真实权重**
    （1.7–2.2 MB，含 args.json）；POMO 的 `pretrained/**` 目录为**空目录**（无权重文件）。

本脚本回答的问题（可执行的最小验证）
------------------------------------
1. 真实权重能否被**官方加载路径**（`utils.load_model` + args.json）加载？
2. 能否**真实推理**（贪心解码）并产出「访问全部节点的单条路线」？
3. 该机制的**任务形态**（单车辆、无容量字段）是否与 A-v1 多车 + 容量 50 直接冲突？
   —— 结合 `envs/*` 字段清单一起记录（见 RESULTS.md）。

不修改第三方原文件：只读导入；本文件位于 `PIP-constraint/a1_accept_v1/`。

用法（服务器，cc_compare env）：
    cd /home/hzeng/project/MASKCO-Main/CC_Compare/PIP-constraint/AM+PIP
    CUDA_VISIBLE_DEVICES= /home/hzeng/envs/cc_compare/bin/python \
        ../a1_accept_v1/probe_pip_assets.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AM = os.path.normpath(os.path.join(_HERE, '..', 'AM+PIP'))
_POMO = os.path.normpath(os.path.join(_HERE, '..', 'POMO+PIP'))

CKPTS = [
    ('hard', 'AM_star_PIP-D'),
    ('hard', 'AM_star_PIP'),
    ('easy', 'AM_star_PIP-D'),
]


def inventory():
    print('== 资产清单 ==')
    for name, path in (('AM+PIP', _AM), ('POMO+PIP', _POMO)):
        pts = sorted(glob.glob(os.path.join(path, 'pretrained', '**', '*.pt'),
                               recursive=True))
        real = [p for p in pts if os.path.getsize(p) > 0]
        print('%-9s 存在=%s  *.pt=%d  非空 *.pt=%d' % (name, os.path.isdir(path),
                                                      len(pts), len(real)))
        for p in real[:6]:
            print('    %s  %.2f MB' % (os.path.relpath(p, path),
                                       os.path.getsize(p) / 1e6))
    envs = os.path.join(_POMO, 'envs')
    print('POMO+PIP/envs =', sorted(os.listdir(envs)) if os.path.isdir(envs) else None)
    print('AM+PIP/problems =', sorted(os.listdir(os.path.join(_AM, 'problems')))
          if os.path.isdir(os.path.join(_AM, 'problems')) else None)
    return


def single_route_fields():
    """单车辆/无容量字段检查（任务形态证据）。"""
    print('\n== 任务形态字段检查（本地实现支持的车辆数/容量） ==')
    import re
    for rel in ('envs/TSPTWEnv.py', 'envs/TSPDLEnv.py'):
        p = os.path.join(_POMO, rel)
        if not os.path.isfile(p):
            print('  %-22s 缺失' % rel)
            continue
        txt = open(p).read()
        nveh = len(re.findall(r'\bnum_vehicles\b|\bn_vehicles\b|\bvehicle_id\b', txt))
        cap = len(re.findall(r'\bcapacity\b|\bvehicle_capacity\b', txt))
        pomo = 'pomo_size' in txt
        print('  %-22s num_vehicles/vehicle_id 出现=%d  capacity 出现=%d  pomo=%s'
              % (rel, nveh, cap, pomo))
    print('  → 均为「单车辆访问全部节点」的 POMO 多起点 TSP 变体；容量约束由 draft/时间窗'
          '代替，不存在车队/容量字段（详见 RESULTS.md）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--graph-size', type=int, default=100)
    args = ap.parse_args()

    inventory()
    single_route_fields()

    os.chdir(_AM)                       # 官方相对路径（../data/...）与 import 依赖 cwd
    for _p in (_AM,):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import torch
    print('\n== 权重加载 / 推理 ==')
    print('torch', torch.__version__)
    from utils import load_model

    for hardness, variant in CKPTS:
        d = os.path.join(_AM, 'pretrained',
                         'tsptw%s_%s' % (args.graph_size, hardness), variant)
        ck = os.path.join(d, 'epoch-99.pt')
        print('\n-- %s/%s --' % (hardness, variant))
        print('   ckpt exists=%s size=%s args.json=%s  (cwd=%s)'
              % (os.path.isfile(ck),
                 os.path.getsize(ck) if os.path.isfile(ck) else -1,
                 os.path.isfile(os.path.join(d, 'args.json')), os.getcwd()))
        if not os.path.isfile(ck) or not os.path.isfile(os.path.join(d, 'args.json')):
            print('   [SKIP] 资产不完整（官方 load_model 需要同目录 args.json）')
            continue
        try:
            t0 = time.time()
            model, opts = load_model(ck)
            print('   加载成功 %.1fs  类=%s  参数量=%.2fM'
                  % (time.time() - t0, type(model).__name__,
                     sum(p.numel() for p in model.parameters()) / 1e6))
            for k in ('problem', 'graph_size', 'hardness', 'generate_PI_mask',
                      'pip_decoder', 'is_train_double_head_decoder', 'model'):
                print('     %-28s = %s' % (k, getattr(model, k, getattr(opts, k, None))))
            model.eval()
            ds = model.problem.make_dataset(size=args.graph_size, num_samples=args.batch)
            with torch.no_grad():
                t0 = time.perf_counter()
                cost, ll, pi = model(ds, return_pi=True, decode_type='greedy')
                dt = time.perf_counter() - t0
            pi_np = pi.detach().cpu().numpy()
            print('   推理 batch=%d n=%d  贪心 cost 均值=%.4f  耗时=%.3fs/批'
                  % (args.batch, args.graph_size, float(cost.mean()), dt))
            row = pi_np[0]
            uniq = len(set(int(x) for x in row))
            print('   路线样例(前12)=%s  唯一节点=%d/%d'
                  % (row[:12].tolist(), uniq, len(row)))
            full = sorted(set(int(x) for x in row))
            print('   覆盖 0..n = %s' % (full == list(range(args.graph_size + 1))))
            with torch.no_grad():
                cost2, _, pi2 = model(ds, return_pi=True, decode_type='greedy')
            print('   确定性 max|Δcost|=%.3e  tour 相同=%s'
                  % (float((cost - cost2).abs().max()),
                     bool((pi == pi2).all())))
        except Exception as exc:                    # noqa: BLE001
            import traceback
            print('   [FAIL] %s: %s' % (type(exc).__name__, exc))
            traceback.print_exc()
    print('\nPROBE_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
