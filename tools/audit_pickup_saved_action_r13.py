"""Independent saved-record audit of the order-77 decision convergence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'research/review_3080_20261010/goal_round_20261013_r13_model_action_resource'
OLD = ROOT / 'research/review_3080_20261010/pickup/learned_after_strong_recovery_execution_root_v2/RESULT.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_wire(x):
    if isinstance(x, dict):
        if '__mapping__' in x:
            return {read_wire(pair[0]): read_wire(pair[1]) for pair in x['__mapping__']}
        if '__sequence__' in x:
            return [read_wire(y) for y in x['values']]
    return x


def main():
    reg = json.loads((ROUND / 'PICKUP_MECHANISM_AUDIT_REGISTRATION.json').read_text(encoding='utf-8'))
    assert digest(__file__) == reg['audit_sha256'] and digest(OLD) == reg['raw_sha256']
    observed = json.loads((ROUND / 'PICKUP_SAVED_ACTION_MECHANISM.json').read_text(encoding='utf-8'))
    assert digest(ROUND / 'PICKUP_SAVED_ACTION_MECHANISM.json') == reg['mechanism_sha256']
    records = read_wire(json.loads(OLD.read_text(encoding='utf-8')))['records']
    checks = 3
    for i, record in enumerate(records):
        actual = record['actual']; proposal = actual['proposal']; actions = proposal['audit']
        assert record['mode'] == observed['rows'][i]['mode']; checks += 1
        assert sorted(proposal['masked_orders']) == [25, 30] and len(actions) == 10; checks += 1
        stage1 = min(actions[:6], key=lambda x: (x['proposal_score'], x['order'], x['vehicle'], x['position']))
        stage2 = min(actions[6:], key=lambda x: (x['proposal_score'], x['order'], x['vehicle'], x['position']))
        assert [stage1[k] for k in ('order', 'vehicle', 'position')] == observed['rows'][i]['first_choice']; checks += 1
        assert [stage2[k] for k in ('order', 'vehicle', 'position')] == observed['rows'][i]['second_choice']; checks += 1
        margin1 = sorted(x['proposal_score'] for x in actions[:6])[1] - stage1['proposal_score']
        margin2 = sorted(x['proposal_score'] for x in actions[6:])[1] - stage2['proposal_score']
        assert margin1 == observed['rows'][i]['first_choice_margin'] and margin2 == observed['rows'][i]['second_choice_margin']; checks += 1
        c = proposal['candidate_certificate']['evaluation']['utility']
        b = proposal['baseline_certificate']['evaluation']['utility']
        assert c - b == observed['rows'][i]['known_utility_delta'] < 0; checks += 1
        score = actual['common_scores']['candidate_scores']
        assert score['actual_candidate']['mean_shadow_utility'] - score['own_incumbent']['mean_shadow_utility'] == observed['rows'][i]['common_shadow_delta'] > 0; checks += 1
        assert actual['raw_candidate'] == actual['selected_plan'] == observed['rows'][i]['candidate']; checks += 1
        assert not proposal['accepted_on_known_score_only'] and actual['accepted_on_common_SAA']; checks += 1
    assert records[0]['actual']['raw_candidate'] == records[1]['actual']['raw_candidate']; checks += 1
    assert records[0]['actual']['proposal']['logits_sha256'] != records[1]['actual']['proposal']['logits_sha256']; checks += 1
    result = {'checks': checks, 'failures': 0, 'raw_sha256': digest(OLD),
              'saved_only': True, 'no_model_or_solver_rerun': True,
              'learning_attribution_qualified': False, 'stable_online_gain': False}
    with (ROUND / 'INDEPENDENT_PICKUP_ACTION_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'checks': checks, 'failures': 0}))


if __name__ == '__main__':
    main()
