"""只读探针：对比 v1 / v2 同名臂 gate 的关键字段，用于解释「同臂效用偏离」的原因。

用法（1003 扩展根下）：
    python ../CC_Compare/tools/_probe_arm_v1v2.py <v2_gate> <v1_gate>
"""
import json
import sys


def block(g):
    r = g.get('results') or g.get('gate') or {}
    ident = None
    for y, blk in r.items():
        if isinstance(blk, dict):
            for k, v in blk.items():
                if k == '_identity':
                    ident = v
    return r, ident


def show(tag, path):
    g = json.load(open(path, encoding='utf-8'))
    res, ident = block(g)
    cfg = g.get('config') or {}
    print('== %s : %s' % (tag, path))
    print('   config : %s' % {k: cfg.get(k) for k in
                              ('solver', 'provider', 'provider_device', 'workers',
                               'time_limit', 'baseline_from')})
    print('   ckpt   : %s' % cfg.get('provider_ckpt'))
    if ident:
        print('   source_stable=%s  shared_src_n=%d' % (
            ident.get('source_stable'), len(ident.get('shared_source_sha256') or {})))
    for y, blk in res.items():
        if y.startswith('_') or not isinstance(blk, dict):
            continue
        for pc, b2 in blk.items():
            if not isinstance(b2, dict):
                continue
            arms = b2.get('arms') or {}
            for an, ab in arms.items():
                if an in ('cond_hist', 'uncond_hist', 'explicit_feat'):
                    continue
                u = ab.get('utility_all_days') or {}
                print('   %-10s %-16s util=%-9s to=%-6s hfr=%-5s served=%-6s rejected=%-6s fail=%s'
                      % (y, pc, ('%.3f' % u['mean']) if u.get('mean') is not None else 'n/a',
                         ab.get('mean_timeouts'), ab.get('hard_feasible_rate'),
                         ab.get('mean_served'), ab.get('mean_rejected'), ab.get('fail_counts')))
            sm = b2.get('solver_meta') or b2.get('arm_meta') or {}
            if sm:
                print('        solver_meta: %s' % json.dumps(sm, ensure_ascii=False)[:220])


for i in range(1, len(sys.argv), 2):
    show(sys.argv[i], sys.argv[i + 1])
