"""Independent arithmetic and source-observability audit of the frozen 5-day panel."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4'
RESULT = BASE / 'RESULT.json'
SOURCE = BASE / 'run.py'
RUN_ARM = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_a1_step2_gate.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, name, checks):
    checks.append({'name': name, 'pass': bool(condition)})


def main():
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    checks = []
    rows = []
    require(result['panel_days'] == 5 and len(result['paired']) == 5, 'five_fixed_days', checks)
    for p in result['paired']:
        k = int(p['day'])
        c, t = result['arms']['control_frozen_saa'][k], result['arms']['candidate_reserving'][k]
        dr = float(t['revenue'] - c['revenue'])
        df = float(t['fuel_cost'] - c['fuel_cost'])
        dl = float(t['reject_loss'] - c['reject_loss'])
        du = float(t['utility'] - c['utility'])
        rates = {z: float(t['service_rate'][z] - c['service_rate'][z]) for z in ('0', '1', '2')}
        require(abs((dr - df - dl) - du) < 1e-9, 'utility_decomposition_day_%d' % k, checks)
        require(abs(du - p['delta']) < 1e-9, 'frozen_pair_day_%d' % k, checks)
        require(t['hard_feasible'] and c['hard_feasible'] and not t['budget_violated'] and not c['budget_violated'], 'hard_and_budget_day_%d' % k, checks)
        rows.append({'day': k, 'delta_utility': du, 'delta_revenue': dr, 'delta_fuel_cost': df,
                     'delta_reject_loss': dl, 'delta_served': t['n_served'] - c['n_served'],
                     'delta_distance_km': t['distance_km'] - c['distance_km'],
                     'delta_energy_kwh': t['energy_kwh'] - c['energy_kwh'],
                     'delta_service_rate': rates})
    require([r['day'] for r in rows] == list(range(5)), 'day_order', checks)
    require(abs(sum(r['delta_utility'] for r in rows)/5 - result['mean_delta']) < 1e-9, 'mean_delta', checks)
    src = SOURCE.read_text(encoding='utf-8')
    arm = RUN_ARM.read_text(encoding='utf-8')
    ast.parse(src); ast.parse(arm)
    # Source-level observability check: run_arm returns only the saved per-day rows.
    # The new mixin's per-order reserve_log and counters are not put in those rows.
    require('self.reserve_log.append' in src, 'in_memory_reservation_log_exists', checks)
    require("n_decisions=int(getattr(rp, 'n_decisions', 0))" in arm, 'row_decision_counter_is_base_n_decisions', checks)
    require('reserve_log' not in RESULT.read_text(encoding='utf-8'), 'per_order_reservation_log_not_in_result', checks)
    require(all(x['n_decisions'] == 0 for x in result['arms']['candidate_reserving']), 'saved_candidate_n_decisions_zero', checks)
    out = {
        'stage': 'saved_five_day_utility_decomposition', 'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (RESULT, SOURCE, RUN_ARM)},
        'checks_count': len(checks), 'failed': sum(not q['pass'] for q in checks), 'checks': checks,
        'rows': rows,
        'sum_delta': {name: sum(r[name] for r in rows) for name in ('delta_utility','delta_revenue','delta_fuel_cost','delta_reject_loss')},
        'days_positive': [r['day'] for r in rows if r['delta_utility'] > 0],
        'days_zero': [r['day'] for r in rows if r['delta_utility'] == 0],
        'direct_reservation_rejection_count_identified': False,
        'causal_mediation_by_temperature_class_identified': False,
        'interpretation': 'The A/B treatment is the added admission filter and all downstream state changes. Saved rows identify paired utility components and class service rates, not direct per-order filter decisions or class-mediated causal effect. Day 2 improves revenue despite fewer served orders and lower class-0 service rate.',
        'old_audit_failed_preserved': 2, 'stable_online_gain': False,
    }
    target = HERE / 'SAVED_MECHANISM_AUDIT.json'
    with target.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=2, ensure_ascii=False); f.write('\n')
    print(json.dumps({'checks_count': out['checks_count'], 'failed': out['failed'], 'sum_delta': out['sum_delta']}, ensure_ascii=False))
    return int(out['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
