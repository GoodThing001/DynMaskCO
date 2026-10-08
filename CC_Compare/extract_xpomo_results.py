"""读取 POMO 系（pomo/symnco/sgbs）A-v1 臂 gate.json 汇总关键数字（服务器，cc_compare env）。

用法：cd C-VRP_Cold-chainVehicleRoutingProblem && python ../../CC_Compare/extract_xpomo_results.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(
    ROOT, '..', 'C-VRP_Cold-chainVehicleRoutingProblem', 'results', 'a1_ext_extra'))

ARMS = [
    ('pomo', 'pomo_0', 'pomo_0', 'p_c=0'),
    ('pomo', 'pomo_51015', 'pomo_51015', 'p_c=(5,10,15)'),
    ('symnco', 'symnco_0', 'symnco_0', 'p_c=0'),
    ('symnco', 'symnco_51015', 'symnco_51015', 'p_c=(5,10,15)'),
    ('sgbs', 'sgbs_0', 'sgbs_0', 'p_c=0'),
    ('sgbs', 'sgbs_51015', 'sgbs_51015', 'p_c=(5,10,15)'),
]

for name, dname, tag, pen in ARMS:
    p = os.path.join(OUT, dname, 'gate.json')
    if not os.path.exists(p):
        print('== %s [%s]: MISSING (%s)' % (tag, pen, p))
        continue
    d = json.load(open(p))
    r = d['results']['20260926'][pen]
    a = r['arms']['ordering_accept']
    print('== %s [%s]' % (tag, pen))
    u = a['utility_all_days']
    print('  utility_all_days: mean=%.3f ci=[%.3f, %.3f] n=%s' % (
        u.get('mean'), u.get('ci_lo'), u.get('ci_hi'), u.get('n')))
    print('  hard_feasible_rate=%s fail_counts=%s adjudicable=%s n_days=%s' % (
        a['hard_feasible_rate'], a.get('fail_counts'), a['adjudicable'],
        a['n_days']))
    print('  mean_timeouts=%s timeout_days=%s elapsed_total_s=%s' % (
        a['mean_timeouts'], a['timeout_days'], a['elapsed_total_s']))
    for arm in ('cond_hist', 'uncond_hist', 'explicit_feat'):
        k = 'ordering_accept_minus_%s' % arm
        if k in r:
            v = r[k]
            print('  minus_%s: mean=%.3f ci=[%.3f, %.3f] n_paired=%s adj=%s ident=%s' % (
                arm, v.get('mean'), v.get('ci_lo'), v.get('ci_hi'),
                v.get('n_paired'), v.get('adjudicable'),
                v.get('identity_verified')))
    m = r['solver_meta']
    print('  solver_meta: n_solves=%s n_fail=%s mean_solve_time_s=%s reject_reasons=%s' % (
        m['n_solves'], m['n_fail'], m.get('mean_solve_time_s'),
        m.get('reject_reasons')))
    ns = [x['n_served'] for x in r['per_day'] if x.get('n_served') is not None]
    if ns:
        print('  n_served mean=%.1f (n=%d)' % (sum(ns) / len(ns), len(ns)))
    print('  pairing: %s' % json.dumps(
        d['results']['20260926'].get('_pairing_identity'), ensure_ascii=False))
