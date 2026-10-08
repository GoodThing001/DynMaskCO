# -*- coding: utf-8 -*-
"""Synthetic sanity test for diag_a3_adjudicate.py (local, no server)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, 'scripts/evaluation')

tmp = tempfile.mkdtemp()
pen = 'p_c=(5,10,15)'
for arm in 'BCD':
    d = os.path.join(tmp, 'a1_a3_online_density_%s_20260926_40_1003_040454' % arm)
    os.makedirs(d)
    rows = [dict(i=i, utility=float(i + (3 if arm == 'B' else 0))) for i in range(5)]
    base_rows = [dict(i=i, utility=float(i)) for i in range(5)]
    mc = {'maskco_minus_cond_hist': dict(mean=2.0, ci_lo=0.5, ci_hi=3.5, n_paired=5,
                                         adjudicable=True),
          'delta_3': 20.0, 'passed': False, 'violations_present': None,
          'criterion': 'x', 'identity_verified': True}
    blk = {'arms': {'maskco': dict(hard_feasible_rate=1.0, mean_timeouts=3.0),
                    'cond_hist': dict(hard_feasible_rate=1.0, mean_timeouts=4.0)},
           'main_comparison': mc,
           'per_day': {'maskco': rows, 'cond_hist': base_rows,
                       'explicit_feat': base_rows, 'uncond_hist': base_rows},
           'formal_adjudication': True, 'identity_problems': []}
    doc = {'config': {}, 'seeds': {}, 'model_generation_check': 'ok', 'delta_3': 20.0,
           'results': {'20260926': {pen: blk,
                                    '_identity': dict(source_stable=True),
                                    '_pairing_identity': dict(identity_verified=True)}}}
    with open(os.path.join(d, 'gate.json'), 'w') as f:
        json.dump(doc, f)

lockp = os.path.join(tmp, 'lock.json')
with open(lockp, 'w') as f:
    json.dump(dict(baseline_policy='density', caveats=['x']), f)

from diag_a3_adjudicate import main  # noqa: E402

out = main([os.path.join(tmp, 'a1_a3_online_density_%s_20260926_40_1003_040454' % a)
            for a in 'BCD']
           + ['--lock', lockp, '--out', os.path.join(tmp, 'adj.json')])
print('ablations:', {k: {kk: vv for kk, vv in v.items() if kk != 'mean'}
                     for k, v in out['ablations'].items()})
print('per_batch:', out['per_batch'])
print('summary B:', out['summary_by_arm_seed']['B'])
print('SYNTHETIC_OK')
