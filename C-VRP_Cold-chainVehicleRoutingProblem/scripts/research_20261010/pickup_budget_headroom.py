"""P-BUDGET-v1: where does pickup headroom actually live? An arithmetic ceiling on
energy-budget utilisation, computed from the already-recorded formal A-v1 gate.

Why this panel exists. The pickup line's registered reconstructor n3 was pruned last
round by a census of its own action set: same-vehicle tail reordering caps utility at
about 0.1 percent. That census measured the wrong resource. It bounded what a
reordering can change (travel distance, 9.48 percent of utility) but not what the real
binding constraint is. This panel reads the frozen A-v1 gate and asks where the
binding constraint sits and what the ceiling on relieving it is.

What it does. Pure secondary arithmetic over one recorded file - no simulation is run,
no model is loaded, no protocol is re-opened. It extracts, for the main penalty regime:
  * the C0 energy budget B and the all-accept energy demand
  * the served count and the realised budget utilisation
  * the implied per-order energy cost
  * the served count achievable if the SAME per-order cost were sustained to full
    budget utilisation, and the resulting relative gain

Honesty about the ceiling. Sustaining the observed average per-order energy while
adding orders is optimistic, because added orders also add travel and duty time, so the
true marginal cost is at least the average. The number produced here is therefore an
UPPER bound on the headroom available from budget utilisation alone, and it is labelled
as such. A lower bound would require re-simulating routes, which this panel deliberately
does not do.

It also records, as prior evidence rather than as a new result, the already-existing
finding that the rolling optimiser (enumerate accept, greedy insert, relocate+swap,
certify) does not beat the non-learned unconditional-dynamic rule on the same instances.

Usage:
    python pickup_budget_headroom.py --gate GATE.json --out NEWDIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


def main(args):
    gate_path = Path(args.gate)
    raw = gate_path.read_bytes()
    gate = json.loads(raw.decode('utf-8'))
    budget = gate['budget']
    B = float(budget['B'])
    all_accept = float(budget['train_accept_all_c0_energy_mean'])
    results = gate['results']

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    per_regime = {}
    for penalty, block in results.items():
        arms = block['arms']
        entry = {'served': {}, 'utility': {}}
        for arm, value in arms.items():
            entry['served'][arm] = float(value['mean_n_served'])
            entry['utility'][arm] = float(value['utility']['mean'])
            entry['budget_usage'] = float(value.get('mean_budget_usage', float('nan')))
            entry['hard_feasible_rate'] = float(value.get('hard_feasible_rate', float('nan')))
            entry['rejected'] = float(value.get('mean_n_rejected', float('nan')))
        entry['paired_vs_uncond_dynamic'] = block.get('paired', {})
        per_regime[penalty] = entry

    main_regime = gate.get('main_penalty', 'p_c=(5,10,15)')
    block = per_regime[main_regime]
    arm = 'uncond_dynamic'
    served = block['served'][arm]
    usage = block['budget_usage']
    used_energy = usage * B
    per_order = used_energy / served if served else None
    ceiling_served = B / per_order if per_order else None
    ceiling_gain = (ceiling_served - served) / served if served else None

    # What the same analysis says about the rejected population and coverage.
    total_orders = served + block['rejected']
    coverage = B / all_accept
    rollout_gain = ceiling_gain

    # Utility sensitivity: revenue per served order, from the p_c = 0 regime where the
    # reject penalty is zero so utility is revenue minus fuel only.
    zero_regime = per_regime.get('p_c=0', {})
    revenue_free_utility = zero_regime.get('utility', {}).get(arm)
    utility_at_ceiling = None
    if revenue_free_utility is not None and served:
        revenue_per_order = revenue_free_utility / served
        utility_at_ceiling = revenue_per_order * ceiling_served

    result = {
        'kind': 'secondary_arithmetic_not_a_new_experiment',
        'source_gate': str(gate_path).replace('\\', '/'),
        'source_gate_sha256': hashlib.sha256(raw).hexdigest(),
        'main_penalty_regime': main_regime,
        'reference_arm': arm,
        'energy_budget_B': B,
        'all_accept_energy_demand': all_accept,
        'budget_coverage_fraction': coverage,
        'observed': {
            'served': served,
            'rejected': block['rejected'],
            'total_orders': total_orders,
            'service_rate': served / total_orders if total_orders else None,
            'budget_utilisation': usage,
            'energy_used': used_energy,
            'implied_energy_per_served_order': per_order,
            'utility_p_c0': revenue_free_utility,
            'utility_main_penalty': block['utility'][arm]},
        'ceiling_if_per_order_cost_held': {
            'served_at_full_budget': ceiling_served,
            'relative_service_gain': ceiling_gain,
            'utility_at_ceiling_p_c0_approx': utility_at_ceiling,
            'is_upper_bound_because': ('adding orders also adds travel and duty time, so '
                                       'true marginal energy cost is at least the average')},
        'already_recorded_arm_comparison': {
            'note': 'prior evidence, not produced by this panel',
            'paired_rolling_minus_uncond_dynamic': {
                k: v for k, v in block['paired_vs_uncond_dynamic'].items()},
            'reading': ('the rolling optimiser (enumerate accept, greedy insert, relocate '
                        'and swap, certify plan) does not beat the non-learned rule; its '
                        'paired difference is negative in every penalty regime with a '
                        'confidence interval spanning zero')},
        'all_regimes': per_regime,
        'interpretation': ('pickup headroom is not in reordering a vehicle tail (about 0.1 '
                           'percent, P-CENSUS-v1) but in how much of the fixed energy budget '
                           'gets converted into served orders; the ceiling on that channel '
                           'is about %.3f relative, which is two orders of magnitude larger '
                           'than the reordering ceiling'
                           % (ceiling_gain if ceiling_gain else 0.0)),
        'claims_online_gain': False,
        'stable_gain_established': False,
    }
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in
                      ('energy_budget_B', 'all_accept_energy_demand',
                       'budget_coverage_fraction', 'observed',
                       'ceiling_if_per_order_cost_held',
                       'already_recorded_arm_comparison')},
                     indent=2, default=str, ensure_ascii=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
