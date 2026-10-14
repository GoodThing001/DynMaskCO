"""只读探针：打印 gate 的 seeds / 年份键 / 各档 arms / identity 摘要。

用途：核对「某个 main_model gate 到底是不是 20260926 开发集」—— 渲染器的
`filter_main_model` 以**目录名里是否含 `20260926`** 来筛开发集，若某个 40 天开发集 gate
的目录名不含该串（例如 `a1_a1_formal_40_1004_171907`），它会被**静默剔除**。
本探针把真实 seed 打出来，判断这是「有意排除」还是「被目录名规则误伤」。

用法（扩展根下）：
    python ../CC_Compare/tools/_probe_gate_meta.py <gate.json> [<gate.json> ...]
"""
import json
import sys

for p in sys.argv[1:]:
    try:
        with open(p, encoding='utf-8') as f:
            g = json.load(f)
    except Exception as e:                                   # noqa: BLE001
        print('%-60s ERR %s' % (p, e))
        continue
    print('== %s' % p)
    print('   seeds        : %s' % (g.get('seeds'),))
    ident = g.get('identity') or {}
    print('   source_stable: %s | contract=%s | data=%s' % (
        ident.get('source_stable'),
        str(ident.get('contract_sha256'))[:16], str(ident.get('data_sha256'))[:16]))
    res = g.get('results') or g.get('gate') or {}
    for y, blk in res.items():
        if y.startswith('_') or not isinstance(blk, dict):
            continue
        for pc, b2 in blk.items():
            if not isinstance(b2, dict):
                continue
            arms = b2.get('arms') if isinstance(b2.get('arms'), dict) else {}
            print('   %-22s %-16s arms=%s' % (y, pc, list(arms.keys())))
