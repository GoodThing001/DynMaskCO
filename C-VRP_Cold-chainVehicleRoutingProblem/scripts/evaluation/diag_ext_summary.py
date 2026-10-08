# -*- coding: utf-8 -*-
"""外部强求解器接单臂 gate.json 汇总（A-05e 后）。用法：python diag_ext_summary.py <gate.json>"""
import json
import sys


def main():
    path = sys.argv[1]
    d = json.load(open(path, encoding='utf-8'))
    for seed_s, r in d['results'].items():
        solver = r.get('_solver', 'ortools')
        key = '%s_accept' % solver
        print('seed', seed_s, 'solver', solver,
              'baseline_from', r.get('_baseline_from'))
        for pn in ['p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)']:
            b = r.get(pn)
            if b is None:
                continue
            a = b['arms'].get(key, {})
            print(' ', pn)
            print('   %s: adj=%s hard=%s miss=%s fails=%s' % (
                solver, a.get('adjudicable'), a.get('hard_feasible_rate'),
                a.get('missing_day_ids'), a.get('fail_counts')))
            u = a.get('utility_all_days', {}) or {}
            print('   utility mean=%s ci=[%s, %s]' % (
                u.get('mean'), u.get('ci_lo'), u.get('ci_hi')))
            for arm in ('cond_hist', 'uncond_hist', 'explicit_feat'):
                p = b.get('%s_accept_minus_%s' % (solver, arm))
                if p is None:
                    continue
                print('   minus %-14s mean=%s ci=[%s, %s] adj=%s n=%s' % (
                    arm, p.get('mean'), p.get('ci_lo'), p.get('ci_hi'),
                    p.get('adjudicable'), p.get('n_paired')))
            sm = b.get('solver_meta') or {}
            print('   meta: n_solves=%s n_fail=%s solve_time_s=%s mean_solve=%s'
                  % (sm.get('n_solves'), sm.get('n_fail'),
                     sm.get('solve_time_s'), sm.get('mean_solve_time_s')))
            rr = sm.get('reject_reasons') or {}
            print('   reject_reasons:', rr)
            rows = b.get('per_day') or []
            if rows:
                import numpy as np
                for fkey in ('n_served', 'n_rejected', 'energy_kwh', 'timeouts'):
                    vals = [r.get(fkey) for r in rows if r.get(fkey) is not None]
                    if vals:
                        print('   %s mean=%s' % (fkey, float(np.mean(vals))))
            # 基线臂分解
            for arm in ('cond_hist', 'uncond_hist', 'explicit_feat'):
                brows = b['arms'].get(arm, {}).get('per_day')
                if not brows:
                    continue
                import numpy as np
                print('   %s: n_served mean=%s n_rejected mean=%s energy mean=%s'
                      % (arm,
                         float(np.mean([r.get('n_served') for r in brows if r.get('n_served') is not None])),
                         float(np.mean([r.get('n_rejected') for r in brows if r.get('n_rejected') is not None])),
                         float(np.mean([r.get('energy_kwh') for r in brows if r.get('energy_kwh') is not None]))))


if __name__ == '__main__':
    main()
