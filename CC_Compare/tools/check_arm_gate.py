"""Print the decision-relevant fields of one arm's gate.json (for verifying arms as they land).

Checks the three things that decide whether an arm is usable in the final table:
  1) identity block present + source_stable
  2) _pairing_identity.identity_verified / formal_adjudication (the v2 whole point)
  3) per-tier utility / mean_timeouts / hard_feasible_rate, and the provider device actually used

Usage:
    python CC_Compare/tools/check_arm_gate.py <gate.json path>
"""
import json
import os
import sys

DEVICE_KEYS = ('provider_device', 'rrnco_device')


def walk(obj, key, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            elif isinstance(v, (dict, list)):
                walk(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, key, out)
    return out


def arm_identity_block(gate):
    """Mirror collect_compare_table.arm_identity_block.

    外部臂的 gate **没有** 顶层 `identity` 键：身份块在 `results[<seed>]['_identity']`。
    （初版本工具只读顶层 `identity`，于是把有身份的臂误报成 source_stable=None。）
    """
    if 'identity' in gate:
        return gate['identity']
    for _yr, yb in (gate.get('results') or {}).items():
        if isinstance(yb, dict) and '_identity' in yb:
            return yb['_identity']
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print('MISSING %s' % path)
        return 1
    g = json.load(open(path, encoding='utf-8'))

    ident = arm_identity_block(g) or {}
    print('== identity ==')
    # 规则臂（run_a1_downstream_ablation.py）**没有** identity/_identity 块，
    # source_stable 落在别处 —— 所以再全局兜底搜一次，避免误报成 None。
    ss = ident.get('source_stable')
    if ss is None:
        vals = sorted({str(v) for v in walk(g, 'source_stable', [])})
        ss = ','.join(vals) if vals else None
    print('  source_stable : %s' % ss)
    print('  contract      : %s' % str(ident.get('contract_sha256'))[:16])
    print('  data          : %s' % str(ident.get('data_sha256'))[:16])
    print('  meta          : %s' % str(ident.get('data_meta_sha256'))[:16])

    # formal_adjudication 落在每个 p_c 块里（driver 逐档写入），单独找。
    fa = sorted({str(v) for v in walk(g, 'formal_adjudication', [])})
    if fa:
        print('  formal_adjudication: %s (×%d)' % (','.join(fa), len(walk(g, 'formal_adjudication', []))))

    for key in DEVICE_KEYS:
        vals = sorted({str(v) for v in walk(g, key, [])})
        if vals:
            print('  %-14s: %s' % (key, ','.join(vals)))

    print('== pairing identity (落盘口径) ==')
    seen = 0
    for pid in walk(g, '_pairing_identity', []):
        if isinstance(pid, dict) and pid and seen < 6:
            seen += 1
            print('  identity_verified=%s formal_adjudication=%s reason=%s problems=%s' % (
                pid.get('identity_verified'), pid.get('formal_adjudication'),
                pid.get('reason'), (pid.get('problems') or [])[:3]))

    print('== tiers ==')
    cont = g.get('results') or g.get('gate') or {}
    for p_c, blk in cont.items():
        if not isinstance(blk, dict) or 'arms' not in blk:
            # nested schema: results[seed][p_c]
            for k2, v2 in blk.items():
                if isinstance(v2, dict) and 'arms' in v2:
                    _tier(k2, v2)
            continue
        _tier(p_c, blk)
    return 0


def _tier(p_c, blk):
    arms = blk.get('arms') or {}
    for an, ab in arms.items():
        u = (ab or {}).get('utility_all_days') or {}
        print('  %-16s %-14s util=%-10s to/day=%-7s hfr=%s' % (
            p_c, an,
            ('%.2f' % u['mean']) if u.get('mean') is not None else 'n/a',
            ab.get('mean_timeouts'), ab.get('hard_feasible_rate')))


if __name__ == '__main__':
    sys.exit(main())
