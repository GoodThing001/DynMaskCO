"""Detect terminal utility/trace corruption without rerunning a complete teacher collection."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

EXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXT / 'tools_remote'))
sys.path.insert(0, str(EXT / 'scripts/tests'))
from i3_teacher_data_audit_1008 import attach_backend, check_entry, summarize
from test_i3_counterfactual_teacher import fixture, replay, check_raises


def test():
    backend = attach_backend(EXT)
    data, contract, factory, policy, records, original = fixture(infeasible=True)
    rows, entries, public = [], [], []
    collection = {'budget': policy.budget, 'minimum_gap': 5}
    for ordinal, record in enumerate(records):
        branches = [replay(record, data, contract, factory, accept) for accept in (True, False)]
        for branch in branches:
            if 'traces' in branch:
                branch['traces'] = [asdict(trace) for trace in branch['traces']]
        preference = backend['teacher_preference'](*branches, minimum_gap=5)
        margin = 0.
        regret = abs(preference['teacher_delta']) if preference['teacher_valid'] and preference['teacher_delta'] > 0 else \
            (0. if preference['teacher_valid'] else None)
        row = {'day_id': 0, 'ordinal': ordinal, 'same_action_parity': True, **preference,
               'hist_resampled_margin': margin, 'hist_teacher_regret': regret}
        entry = {'day_id': 0, 'ordinal': ordinal, 'record': record, 'branches': branches, 'preference': preference}
        rows.append(check_entry(entry, row, data, contract, collection, backend))
        entries.append(entry)
        public.append(row)
    result = summarize(rows, [(0, i) for i in range(4)])
    assert result['states'] == 4 and result['invalid_teachers'] == 1 and not result['online_gain_established']
    feasible = next(i for i, entry in enumerate(entries) if entry['branches'][0]['action_feasible'])
    altered = deepcopy(entries[feasible])
    altered['branches'][0]['fuel_cost'] += 1
    check_raises(lambda: check_entry(altered, public[feasible], data, contract, collection, backend))
    altered = deepcopy(entries[feasible])
    altered['branches'][0]['traces'][0]['services'][0]['arrival_time'] = 1e6
    check_raises(lambda: check_entry(altered, public[feasible], data, contract, collection, backend))
    altered_row = deepcopy(public[feasible])
    altered_row['hist_teacher_regret'] += 1
    check_raises(lambda: check_entry(entries[feasible], altered_row, data, contract, collection, backend))
    check_raises(lambda: summarize(rows[:-1], [(0, i) for i in range(4)]))
    check_raises(lambda: summarize(rows + rows[:1], [(0, i) for i in range(4)]))
    print('ALL PASS: archived complete terminal trace metrics, infeasible state retained, fuel/TW/regret corruption and exact all-state summaries')


if __name__ == '__main__':
    test()
