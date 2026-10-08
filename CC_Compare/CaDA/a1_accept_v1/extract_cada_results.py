"""从 cada_0 / cada_51015 的 gate.json 提取 RESULTS.md 所需数字（只读）。

用法（本地或服务器）：python extract_cada_results.py <path_to_cada_0_gate.json>
                                    <path_to_cada_51015_gate.json>
输出：两臂 utility_all_days（mean/ci/n）、n_days、adjudicable、hard_feasible_rate、
mean_timeouts、timeout_days、fail_counts、solver_meta（n_solves/n_fail/
mean_solve_time_s/reject_reasons）、三个基线配对差（mean/ci/n_paired/adjudicable/
formal_adjudication/identity_verified）。
"""
import json
import sys

BASELINES = ('cond_hist', 'uncond_hist', 'explicit_feat')


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def summarize(path):
    d = load(path)
    r = d['results']['20260926']
    blk = r.get('p_c=0', r.get('p_c=(5,10,15)'))  # 本臂只有一档 p_c
    pname = 'p_c=0' if 'p_c=0' in r else 'p_c=(5,10,15)'
    arm = blk['arms']['ordering_accept']
    meta = blk['solver_meta']
    paired = {}
    for b in BASELINES:
        key = 'ordering_accept_minus_%s' % b
        if key in blk:
            p = blk[key]
            paired[b] = dict(mean=p.get('mean'), ci_lo=p.get('ci_lo'),
                             ci_hi=p.get('ci_hi'), n=p.get('n_paired'),
                             adjudicable=p.get('adjudicable'),
                             formal_adjudication=p.get('formal_adjudication'),
                             identity_verified=p.get('identity_verified'))
    u = arm['utility_all_days']
    failures = [r2 for r2 in blk['per_day'] for f in r2['failures']]
    from collections import Counter
    fc = Counter(failures)
    return {
        'path': path, 'penalty': pname,
        'utility_mean': u.get('mean'), 'utility_ci_lo': u.get('ci_lo'),
        'utility_ci_hi': u.get('ci_hi'), 'utility_n': u.get('n'),
        'n_days': arm['n_days'], 'adjudicable': arm['adjudicable'],
        'hard_feasible_rate': arm['hard_feasible_rate'],
        'mean_timeouts': arm['mean_timeouts'],
        'timeout_days': arm['timeout_days'],
        'elapsed_total_s': arm['elapsed_total_s'],
        'fail_counts': dict(arm['fail_counts']),
        'missing_day_ids': arm['missing_day_ids'],
        'failures_all': dict(fc),
        'n_solves': meta['n_solves'], 'n_fail': meta['n_fail'],
        'solve_time_s': meta['solve_time_s'],
        'mean_solve_time_s': meta['mean_solve_time_s'],
        'reject_reasons': meta.get('reject_reasons', {}),
        'paired': paired,
        'n_rejected_total': sum(r2['n_rejected'] for r2 in blk['per_day']
                                if r2.get('n_rejected') is not None),
        'n_served_total': sum(r2['n_served'] for r2 in blk['per_day']
                              if r2.get('n_served') is not None),
    }


def main():
    paths = sys.argv[1:]
    for p in paths:
        s = summarize(p)
        print('=' * 78)
        print('%s  (%s)' % (s['path'], s['penalty']))
        print('  utility_all_days: mean=%.4f  ci=[%.4f, %.4f]  n=%s'
              % (s['utility_mean'], s['utility_ci_lo'], s['utility_ci_hi'],
                 s['utility_n']))
        print('  n_days=%s adjudicable=%s hard_feasible_rate=%s'
              % (s['n_days'], s['adjudicable'], s['hard_feasible_rate']))
        print('  mean_timeouts=%.3f timeout_days=%d'
              % (s['mean_timeouts'], s['timeout_days']))
        print('  fail_counts=%s  failures_all=%s missing=%s'
              % (s['fail_counts'], s['failures_all'], s['missing_day_ids']))
        print('  n_solves=%d n_fail=%d mean_solve_time_s=%.3f elapsed_total_s=%.0f'
              % (s['n_solves'], s['n_fail'], s['mean_solve_time_s'],
                 s['elapsed_total_s']))
        print('  reject_reasons=%s' % s['reject_reasons'])
        print('  n_rejected_total=%s n_served_total=%s'
              % (s['n_rejected_total'], s['n_served_total']))
        for b, p in s['paired'].items():
            print('  minus_%s: mean=%.4f ci=[%.4f, %.4f] n=%s adjud=%s formal=%s'
                  % (b, p['mean'], p['ci_lo'], p['ci_hi'], p['n'],
                     p['adjudicable'], p['formal_adjudication']))
    print('=' * 78)


if __name__ == '__main__':
    main()
