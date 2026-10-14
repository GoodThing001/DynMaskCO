#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读探针：把「同源对照 `cond_hist + SAA`」的**实际落盘配置**取出来。

用途：P1-A §3 要求把主对照从**算法名**固定为**具体配置 + 冻结代码身份**。
本探针不做任何判断，只把 gate 里真实记录的 `config` / `seeds` / `_identity` 字段
原样打印，供人工填表 —— **不发明任何值**。
"""
import glob
import io
import json
import os
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WS = os.path.join(ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')

# 关注这些旋钮（SaaReplanner 的构造参数 + 预算/合同身份）
KNOBS = ('sampler', 'K', 'k', 'time_limit', 'reject_penalty', 'penalty', 'future_policy',
         'shadow_mode', 'energy_pricing', 'overage_price', 'incr_eval', 'arm_seed', 'seed',
         'budget', 'budget_B', 'cooling_share', 'mode', 'contract', 'contract_sha256',
         'data_sha256', 'data_meta_sha256', 'shared_source_sha256', 'source_sha256',
         'source_stable', 'provider', 'solver', 'workers', 'baseline_from', 'dev_instances',
         'n_orders', 'objective')

CASES = [
    'results/a1_c1rr2_reveal_40/gate.json',
    'results/a1_c1fresh_reveal_40/gate.json',
    'results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json',
    'results/i3_online_recovery_1009_v1/online_20260926/gate.json',
]


def scan_text(path, needle):
    """该 gate 文件里是否**出现过**某关键字（判断字段是否被记录）。"""
    try:
        with open(path, encoding='utf-8') as f:
            return needle in f.read()
    except Exception:                                          # noqa: BLE001
        return None


def show(rel):
    p = os.path.join(WS, *rel.split('/'))
    if not os.path.isfile(p):
        print('MISSING %s' % rel)
        return
    g = json.load(open(p, encoding='utf-8'))
    print('=' * 78)
    print('%s' % rel)
    print('  顶层键: %s' % sorted(g.keys()))
    cfg = g.get('config') or {}
    print('  config 键: %s' % sorted(cfg.keys()))
    for k in KNOBS:
        if k in cfg:
            print('    config[%s] = %r' % (k, cfg[k]))
    if not cfg:
        print('    （无 config 块）')
    print('  seeds = %r' % (g.get('seeds'),))
    # 身份：取第一个带 _identity 的 results 子块
    for yr, yb in (g.get('results') or {}).items():
        if isinstance(yb, dict) and isinstance(yb.get('_identity'), dict):
            idt = yb['_identity']
            print('  _identity[%s] 键: %s' % (yr, sorted(idt.keys())))
            for k in KNOBS:
                if k in idt:
                    v = idt[k]
                    print('    _identity[%s] = %r' % (k, (str(v)[:24] + '…')
                                                      if isinstance(v, str) and len(v) > 24 else v))
            break
    # 关键字是否被记录（决定 Role C 能否与之对齐）
    for kw in ('future_policy', 'shadow_mode', 'energy_pricing', 'overage_price',
               'incr_eval', 'arm_seed', 'K'):
        t = scan_text(p, '"%s"' % kw)
        print('  记录 "%s" ? %s' % (kw, {True: '是', False: '否', None: '读取失败'}[t]))


def main():
    print('gate 根: %s' % WS)
    for rel in CASES:
        show(rel)
    # 顺带看一个外部臂 gate 记了哪些旋钮
    pats = glob.glob(os.path.join(WS, 'results', 'a1_ext_formal2', '*', 'gate.json'))
    if pats:
        rel = os.path.relpath(pats[0], WS).replace('\\', '/')
        print('（附）外部臂示例：')
        show(rel)


if __name__ == '__main__':
    main()
