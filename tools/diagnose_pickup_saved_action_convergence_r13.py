"""Exact saved-only action-ranking diagnosis for the two order-77 model arms."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/review_3080_20261010/goal_round_20261013_r13_model_action_resource'
INPUT = ROOT / 'research/review_3080_20261010/pickup/learned_after_strong_recovery_execution_root_v2/RESULT.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(value):
    if isinstance(value, dict) and '__mapping__' in value:
        return {unpack(k): unpack(v) for k, v in value['__mapping__']}
    if isinstance(value, dict) and '__sequence__' in value:
        return [unpack(x) for x in value['values']]
    if isinstance(value, list):
        return [unpack(x) for x in value]
    return value


def main():
    reg = json.loads((OUT / 'PICKUP_MECHANISM_REGISTRATION.json').read_text(encoding='utf-8'))
    assert sha(__file__) == reg['source_sha256'] and sha(INPUT) == reg['input_sha256']
    result = unpack(json.loads(INPUT.read_text(encoding='utf-8')))
    assert [r['mode'] for r in result['records']] == ['pretrained', 'random']
    rows = []
    for record in result['records']:
        actual = record['actual']; proposal = actual['proposal']
        assert actual['qualified'] and proposal['complete'] and proposal['masked_orders'] == [25, 30]
        audit = proposal['audit']
        assert len(audit) == 10 and all(x['certificate']['qualified'] for x in audit)
        first = sorted(audit[:6], key=lambda x: (x['proposal_score'], x['order'], x['vehicle'], x['position']))
        second = sorted(audit[6:], key=lambda x: (x['proposal_score'], x['order'], x['vehicle'], x['position']))
        assert [(x['order'], x['vehicle'], x['position']) for x in (first[0], second[0])] == [(25, 1, 0), (30, 0, 3)]
        known = proposal['candidate_certificate']['evaluation']['utility'] - proposal['baseline_certificate']['evaluation']['utility']
        scores = actual['common_scores']['candidate_scores']
        shadow = scores['actual_candidate']['mean_shadow_utility'] - scores['own_incumbent']['mean_shadow_utility']
        assert known < 0 < shadow and not proposal['accepted_on_known_score_only'] and actual['accepted_on_common_SAA']
        rows.append({'mode': record['mode'], 'masked_orders': [25, 30], 'qualified_actions_recorded': len(audit),
                     'first_choice': [first[0]['order'], first[0]['vehicle'], first[0]['position']],
                     'first_choice_margin': first[1]['proposal_score'] - first[0]['proposal_score'],
                     'second_choice': [second[0]['order'], second[0]['vehicle'], second[0]['position']],
                     'second_choice_margin': second[1]['proposal_score'] - second[0]['proposal_score'],
                     'known_utility_delta': known, 'common_shadow_delta': shadow,
                     'logits_sha256': proposal['logits_sha256'],
                     'model_parameter_sha256': record['ready']['provider_identity']['actual_param_hash'],
                     'candidate': actual['raw_candidate']})
    assert rows[0]['logits_sha256'] != rows[1]['logits_sha256']
    assert rows[0]['model_parameter_sha256'] != rows[1]['model_parameter_sha256']
    assert rows[0]['candidate'] == rows[1]['candidate']
    report = {'source_sha256': sha(__file__), 'registration_sha256': sha(OUT / 'PICKUP_MECHANISM_REGISTRATION.json'),
              'input_sha256': sha(INPUT), 'rows': rows, 'same_action_sequence': True,
              'same_final_candidate': True, 'distinct_logits': True, 'different_parameters': True,
              'model_or_solver_rerun': False, 'effect_selection': False,
              'learned_attribution_qualified': False, 'realized_online_gain': False}
    with (OUT / 'PICKUP_SAVED_ACTION_MECHANISM.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'arms': 2, 'same_candidate': True,
                      'first_margins': [r['first_choice_margin'] for r in rows],
                      'second_margins': [r['second_choice_margin'] for r in rows],
                      'known_delta': rows[0]['known_utility_delta'],
                      'shadow_delta': rows[0]['common_shadow_delta']}))


if __name__ == '__main__':
    main()
