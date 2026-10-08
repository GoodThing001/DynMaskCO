"""A-v1 诊断批次 8 臂 gate.json 关键数字抽取（写 RESULTS.md 用，不入库评估）。"""
import json
import os
import sys

BASE = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_ext_extra'
NAMES = ['attention_0', 'attention_51015', 'deepaco_0', 'deepaco_51015',
         'omnivrp_0', 'omnivrp_51015', 'lih_0', 'lih_51015']
PEN = {'attention_0': 'p_c=0', 'deepaco_0': 'p_c=0', 'omnivrp_0': 'p_c=0',
       'lih_0': 'p_c=0'}
for n in NAMES:
    P = PEN.get(n, 'p_c=(5,10,15)')
    path = os.path.join(BASE, n, 'gate.json')
    if not os.path.isfile(path):
        print('=== %s: MISSING' % n)
        continue
    g = json.load(open(path))
    b = g['results']['20260926'][P]
    a = b['arms']['ordering_accept']
    u = a['utility_all_days']
    print('=== %s (%s)' % (n, P))
    print('utility_all_days: mean=%s ci=[%s, %s] n=%s'
          % (u.get('mean'), u.get('ci_lo'), u.get('ci_hi'), u.get('n')))
    print('n_days=%s adjudicable=%s hard_feasible_rate=%s'
          % (a.get('n_days'), a.get('adjudicable'), a.get('hard_feasible_rate')))
    print('fail_counts=%s mean_timeouts=%s elapsed_total_s=%s'
          % (a.get('fail_counts'), a.get('mean_timeouts'), a.get('elapsed_total_s')))
    for arm in ('cond_hist', 'uncond_hist', 'explicit_feat'):
        k = 'ordering_accept_minus_%s' % arm
        d = b.get(k)
        if d is None:
            continue
        print('  minus_%s: mean=%s ci=[%s, %s] n_paired=%s adjudicable=%s '
              'formal=%s ident=%s'
              % (arm, d.get('mean'), d.get('ci_lo'), d.get('ci_hi'),
                 d.get('n_paired'), d.get('adjudicable'),
                 d.get('formal_adjudication'), d.get('identity_verified')))
    sm = b.get('solver_meta', {})
    print('solver_meta: n_solves=%s n_fail=%s mean_solve_time_s=%s reject_reasons=%s'
          % (sm.get('n_solves'), sm.get('n_fail'),
             sm.get('mean_solve_time_s'), sm.get('reject_reasons')))
