#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读探针：把对照候选 gate 的 `config` / `budget` / `identity` / `seeds` **原样**导出为 JSON。

用于把 P1-A 的「同源 `cond_hist + SAA`」从**算法名**固定成**具体配置 + 冻结身份**。
只读；不改任何文件；不发明任何值。
"""
import io
import json
import os
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WS = os.path.join(ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')

CASES = [
    ('c1rr2', 'results/a1_c1rr2_reveal_40/gate.json'),
    ('c1fresh', 'results/a1_c1fresh_reveal_40/gate.json'),
]


def main():
    out = {}
    for tag, rel in CASES:
        p = os.path.join(WS, *rel.split('/'))
        if not os.path.isfile(p):
            out[tag] = {'MISSING': rel}
            continue
        g = json.load(open(p, encoding='utf-8'))
        rec = {
            'path': rel,
            'top_keys': sorted(g.keys()),
            'config': g.get('config'),
            'budget': g.get('budget'),
            'seeds': g.get('seeds'),
            'main_penalty': g.get('main_penalty'),
            'delta_star': g.get('delta_star'),
            'note': g.get('note'),
        }
        idt = g.get('identity')
        if isinstance(idt, dict):
            # 身份只留**结构 + 关键值**（shared_source_sha256 / source_sha256 可能很长，
            # 这里保留文件数与前两个键，避免刷屏；要全量可另跑）
            rec['identity_keys'] = sorted(idt.keys())
            rec['identity_scalars'] = {k: v for k, v in idt.items()
                                       if not isinstance(v, (dict, list))}
            for k in ('shared_source_sha256', 'source_sha256'):
                v = idt.get(k)
                if isinstance(v, dict):
                    rec['identity_%s' % k] = {'n_files': len(v),
                                              'files': sorted(v.keys())}
                elif v is not None:
                    rec['identity_%s' % k] = v
            rec['shared_config'] = idt.get('shared_config')
        else:
            rec['identity_present'] = False
        out[tag] = rec
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
