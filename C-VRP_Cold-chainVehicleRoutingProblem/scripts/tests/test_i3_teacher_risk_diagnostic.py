"""Teacher risk is clustered by day, with honest ineligible-state scope."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools_remote'))
from i3_teacher_risk_diagnostic_1008 import aggregate, decision_metrics


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Malformed teacher diagnosis was accepted')


def test():
    positive = {'teacher_valid': True, 'teacher_delta': 60., 'decision_eligible': True}
    a = decision_metrics(positive, [-5., 5.], tau=20., weight_cap=100., supported=True)
    assert a['eligible_teacher_regret'] == 30. and a['direction_accuracy'] == .5
    negative = {**positive, 'teacher_delta': -200.}
    b = decision_metrics(negative, [0., 20.], tau=20., weight_cap=100., supported=True)
    assert b['eligible_teacher_regret'] == 100. and b['direction_accuracy'] == .5
    np.testing.assert_allclose(b['preference_risk'], 100.*np.mean([np.log(2.), np.logaddexp(0., 1.)]))
    invalid = {'teacher_valid': False, 'teacher_delta': None, 'decision_eligible': False}
    zeros = decision_metrics(invalid, None, tau=20., weight_cap=100., supported=True)
    assert zeros['preference_risk'] == 0. and zeros['direction_accuracy'] is None
    rejects(lambda: decision_metrics(positive, [np.nan], tau=20., weight_cap=100., supported=True))
    rejects(lambda: decision_metrics(invalid, [1.], tau=20., weight_cap=100., supported=True))
    # Unequal states per day must not weight a large day more heavily.
    rows = []
    for day, deltas in [(0, [0., 0., 0.]), (1, [100.])]:
        for ordinal, risk in enumerate(deltas):
            rows.append({'day_id': day, 'ordinal': ordinal, **positive,
                'arms': {'maskco': {**a, 'preference_risk': risk},
                         'cond_hist': {**a, 'preference_risk': 0.}}})
    keys = [(r['day_id'], r['ordinal']) for r in rows]
    result = aggregate(list(reversed(rows)), keys, bootstrap_samples=2000, bootstrap_seed=1777)
    assert result['days'] == 2 and result['states'] == 4
    assert result['maskco_minus_controls']['cond_hist']['preference_risk']['mean'] == 50.
    assert result['bootstrap']['unit'] == 'day'
    for malformed in (rows[:-1], rows[:-1] + [rows[0]]):
        rejects(lambda: aggregate(malformed, keys, bootstrap_samples=20, bootstrap_seed=1))
    corrupt = deepcopy(rows)
    corrupt[0]['arms'].pop('cond_hist')
    rejects(lambda: aggregate(corrupt, keys, bootstrap_samples=20, bootstrap_seed=1))
    print('ALL PASS: risk sign/tie/cap, eligible regret, invalid-state scope, equal day weights, exact keys/arms and no nonfinite filtering')


if __name__ == '__main__':
    test()
