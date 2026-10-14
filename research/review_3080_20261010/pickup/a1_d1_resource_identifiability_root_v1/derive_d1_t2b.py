"""Derive the D1 T2b verdict with the adjudicated checker (four-level identifiability).

Kept separate from the D1 executor so the T2b row is attributable to the adjudicated
checker (2026-10-14 用户裁定③) rather than to hand-written if/else in the executor.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'CC_Compare' / 'tools'))
import role_c_decision_audit as aud  # noqa: E402

d1 = json.loads((HERE / 'D1_RESOURCE_IDENTIFIABILITY.json').read_text(encoding='utf-8'))
sp = d1['shadow_price_effect']
stats = {
    'n_decisions': d1['n_decisions'],
    'n_raw_positive': sp['n_price_active'],
    'n_lambda_effective_positive': sp['n_lambda_above_grid_floor'],
    'n_score_changed': sp['n_score_changed'],
    'n_ordering_or_action_changed': sp['n_action_changed_by_price'],
}
verdict, row, problems = aud.check_t2b_effect_identifiability(stats)
out = {'stage': 'D1_T2b_effect_identifiability', 'checker': 'role_c_decision_audit',
       'checker_sha256_expected': 'see CC_Compare/P1A_RoleC_四项裁定落库',
       'input_stats': stats, 'verdict': verdict, 'row': row, 'problems': problems,
       'reading': ('n_lambda_effective_positive is the number of decisions where the '
                   'ENDOGENOUSLY chosen lambda differs from the grid floor; the grid floor is '
                   '0.25 (never 0), so the price term is active in all 1000 decisions and 709 '
                   'decisions carry a price above the floor. raw>0 and the clipped effective '
                   'price are therefore distinguished: here raw>0 == 1000 but the informative '
                   'quantity is the 709 above-floor decisions.'),
       'utility_read': False}
(HERE / 'D1_T2B_VERDICT.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(out, ensure_ascii=False, indent=2))
