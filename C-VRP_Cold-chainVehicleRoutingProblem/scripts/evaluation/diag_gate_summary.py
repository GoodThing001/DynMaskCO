# -*- coding: utf-8 -*-
"""步骤 2 门 gate.json 汇总（S3 杠杆用）。用法：python diag_gate_summary.py <gate.json>"""
import json
import sys

import numpy as np


def main():
    path = sys.argv[1]
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    d = json.load(open(path, encoding='utf-8'))
    print('config:', {k: v for k, v in d['config'].items()
                      if k in ('gate_instances', 'shadow_mode', 'energy_pricing',
                               'future_policy', 'standby_orders', 'workers')})
    # 2026-09-30 完整性核验（独立于 runner 自报）：每个臂 day_id 必须恰为 0..expected-1
    for pn in ['p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)']:
        p = d['gate'].get(pn)
        if p is None:
            continue
        for arm, rows in p['per_day'].items():
            ids = sorted(r['i'] for r in rows)
            if ids != list(range(expected)):
                print('  [完整性] %s/%s day_ids 不完整: n=%d 缺=%s'
                      % (pn, arm, len(ids),
                         [i for i in range(expected) if i not in set(ids)][:5]))
    for pn in ['p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)']:
        p = d['gate'].get(pn)
        if p is None:
            continue
        m = p['main_comparison']['cond_minus_uncond']
        arms = p['arms']
        pd = p['per_day']
        print(' ', pn)
        for name in ('uncond_hist', 'cond_hist', 'explicit_feat'):
            a = arms[name]
            rows = pd[name]
            print('   %-14s adj=%s hard=%s served=%s util=%s to=%s' % (
                name, a['adjudicable'], a['hard_feasible_rate'],
                round(float(np.mean([r['n_served'] for r in rows if r['n_served'] is not None])), 1),
                round(float(np.mean([r['utility'] for r in rows if r['utility'] is not None])), 1),
                round(float(np.mean([r['timeouts'] for r in rows])), 2)))
        print('   cond-uncond: mean=%s ci=[%s, %s] adj=%s passed=%s delta_star=%s'
              % (m['mean'], m['ci_lo'], m['ci_hi'], m.get('adjudicable'),
                 p['main_comparison']['passed'], p['main_comparison']['delta_star']))


if __name__ == '__main__':
    main()
