"""Bounded 54-case public-only fixture execution of the original pre-filter."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'a1_budget_reserving_admission_root_v4'
sys.path.insert(0, str(HERE))
from trace_reservation import traced_decision


def load_original():
    name = 'frozen_reserving_admission_for_new_fixture'
    spec = importlib.util.spec_from_file_location(name, ORIGINAL / 'reserving_admission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Grid:
    def __init__(self, values): self.values = values
    def __getitem__(self, item): return self.values[item]


class Env:
    def __init__(self, classes, demands):
        self.dataset = {'temp_class': Grid(classes)}
        self.demands = Grid(demands)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reg = json.loads((HERE / 'REGISTRATION.json').read_text(encoding='utf-8'))
    original_reg = json.loads((ORIGINAL / 'REGISTRATION.json').read_text(encoding='utf-8'))
    original_source = ORIGINAL / 'reserving_admission.py'
    assert sha(original_source) == original_reg['sources'][str(original_source)]
    result_path = ORIGINAL / 'RESULT.json'
    source = json.loads(result_path.read_text(encoding='utf-8'))
    objective = original_reg['objective']
    module = load_original()
    B = float(source['budget_B'])
    rows = []
    cases = []
    for cls in reg['fixed_engineering_matrix']['classes']:
        for clock in reg['fixed_engineering_matrix']['clock_hours']:
            for frac in reg['fixed_engineering_matrix']['budget_remaining_fraction_of_B']:
                for accepted in reg['fixed_engineering_matrix']['accepted_preexisting_class_2_order']:
                    cases.append((cls, clock, frac, accepted))
    assert len(cases) == reg['fixed_engineering_matrix']['cases']
    classes = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 2}
    demands = {(0, 0): 0.0, (0, 1): 2.0, (0, 2): 2.0, (0, 3): 2.0}
    env = Env(classes, demands)
    for seq, (cls, clock, frac, accepted) in enumerate(cases):
        classes[(0, 1)] = cls
        policy = module.BudgetReservingAdmission()
        policy.budget = B * frac + (module.e_lower_of(2, 2.0, objective) if accepted else 0.0)
        policy._accepted = {3} if accepted else set()
        policy._reserving_init(source['lambda_grid'], source['clock_grid'], source['g_table'], objective)
        allowed, row = traced_decision(policy, env, 0, clock, 1, seq)
        assert row['reserve_decisions_before'] == 0 and row['reserve_decisions_after'] == 1
        assert row['reserve_rejects_after'] == int(not allowed)
        assert row['accepted_before'] == ([3] if accepted else [])
        rows.append(row)
    report = {'stage':'reservation_trace_public_fixture','cases':len(rows),'rows':rows,
              'source_sha256':{'original_policy':sha(original_source),'original_registration':sha(ORIGINAL/'REGISTRATION.json'),
                               'original_result':sha(result_path),'trace_component':sha(HERE/'trace_reservation.py'),
                               'new_registration':sha(HERE/'REGISTRATION.json')},
              'old_five_day_policy_executed':False,'saa_executed':False,'scientific_data_generated':False,
              'full_live_request_qualified':False,'stable_online_gain':False}
    with (HERE/'FIXTURE_RESULT.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'cases':len(rows),'allowed':sum(r['allowed'] for r in rows),'rejected':sum(not r['allowed'] for r in rows)}))


if __name__ == '__main__': main()
