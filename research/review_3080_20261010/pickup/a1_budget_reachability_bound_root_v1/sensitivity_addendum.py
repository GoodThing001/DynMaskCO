"""Addendum: what closes the 800-unit gap between the rigorous bound and the controls.

The registered bound uses only the per-order pre-cooling energy as a lower bound on
the energy needed to serve an order, which is rigorous but loose.  This addendum

  (1) reports a *sensitivity*: recomputes the same knapsack while assuming the true
      per-order energy is the observed population split (pre-cooling ~72 percent of
      energy).  Only share = 1.0 is a bound; every other row is a sensitivity and is
      labelled as such, because "total >= precool / share" is not proved per order;
  (2) reports the class mix the relaxation selects against the mix the recorded
      controls actually serve, which localises the gap to selection rather than to
      arithmetic;
  (3) names the two bounded steps that would turn the bound into a decisive one.

Read-only with respect to the registered stage artifacts.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
sys.path.insert(0, str(ROOT / 'tools'))
for sub in ('scripts', 'scripts/evaluation', 'scripts/coldchain', 'scripts/simulation'):
    sys.path.insert(0, str(EXT / sub))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1')

import numpy as np

REG = json.loads((HERE / 'REGISTRATION.json').read_text(encoding='utf-8'))
R = json.loads((HERE / 'RESULT.json').read_text(encoding='utf-8'))


def frac_knapsack(values, costs, cap):
    items = sorted([(values[i], costs[i]) for i in range(len(values)) if values[i] > 0],
                   key=lambda vc: -(vc[0] / vc[1]))
    total, left = 0.0, float(cap)
    for v, c in items:
        if c <= left:
            total += v
            left -= c
        else:
            total += v * (left / c)
            break
    return total, items


def main():
    import run_exp_reserve as RR
    from run_identity import dataset_identity
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from dataclasses import replace
    from coldchain_contract import ColdChainContractV2, UnitScale
    contract = ColdChainContractV2()
    units = UnitScale(distance_km_per_unit=float(REG['frozen']['km_per_unit']),
                      hours_per_time_unit=1.0, speed_kmph=float(REG['frozen']['speed_kmh']))
    prov = list(contract.parameter_provenance)
    for i, p in enumerate(prov):
        if p.parameter_path == 'units.distance_km_per_unit':
            prov[i] = replace(p, sensitivity_high=50.0)
    contract = replace(contract, units=units, parameter_provenance=tuple(prov))
    ds = add_v2_initial_quality(RR.generate_dataset(REG['frozen']['gate_instances'],
                                                   REG['frozen']['n_orders'],
                                                   REG['frozen']['gate_seed']), contract)
    assert dataset_identity(ds) == REG['frozen']['expected_data_sha256']

    th = REG['objective']
    depot = np.asarray(th['depot_coord'], np.float64)
    kmu, spd = float(REG['frozen']['km_per_unit']), float(REG['frozen']['speed_kmh'])
    horizon, vehicles = float(REG['frozen']['booking_horizon']), int(REG['frozen']['num_vehicles'])
    svc, B = float(REG['frozen']['service_time_h']), float(REG['frozen']['budget_B'])
    rev = {int(k): float(v) for k, v in th['revenue_by_class'].items()}
    controls = json.loads((ROOT / REG['frozen']['recorded_controls']).read_text(encoding='utf-8'))
    per_day = controls['gate'][th['tier']]['per_day']
    arms = list(per_day)

    shares = [1.0, 0.9, 0.8, 0.72, 0.6]
    rows = {s: [] for s in shares}
    mix = {}
    for i in range(REG['frozen']['gate_instances']):
        coords = np.asarray(ds['coords'][i], np.float64)
        q = np.asarray(ds['demands'][i], np.float64)
        cls = np.asarray(ds['temp_class'][i], np.int64)
        reveal = np.asarray(ds['reveal_time'][i], np.float64)
        tw_end = np.asarray(ds['tw_end'][i], np.float64)
        vals, base_e, tau, classes = [], [], [], []
        for o in range(1, len(q)):
            if q[o] <= 0:
                continue
            d = kmu * float(np.linalg.norm(coords[o] - depot))
            t = d / spd
            start = max(t, float(reveal[o]))
            if not (start + svc <= float(tw_end[o]) and start + svc + t <= horizon):
                continue
            delta = max(0.0, th['ambient_c'] - th['target_c'][int(cls[o])])
            vals.append(rev[int(cls[o])])
            base_e.append(float(q[o]) * th['precool_energy_per_unit_per_c'] * delta)
            tau.append(2.0 * t + svc)
            classes.append(int(cls[o]))
        for s in shares:
            be, _ = frac_knapsack(vals, [e / s for e in base_e], B)
            bt, _ = frac_knapsack(vals, tau, vehicles * horizon)
            rows[s].append(min(be, bt))
        if i == 0:
            pairs = sorted([(vals[k], base_e[k], classes[k]) for k in range(len(vals))],
                           key=lambda vc: -(vc[0] / vc[1]))
            used, accum = {0: 0.0, 1: 0.0, 2: 0.0}, 0.0
            for v, c, cl in pairs:
                if accum + c <= B:
                    accum += c
                    used[cl] += 1.0
                else:
                    used[cl] += (B - accum) / c
                    break
            mix['lp_selected_class_counts_day0'] = {str(k): round(v, 3) for k, v in used.items()}
            mix['available_class_counts_day0'] = {str(c): classes.count(c) for c in (0, 1, 2)}
    observed_mix = {}
    for arm in arms:
        r0 = per_day[arm][0]
        observed_mix[arm] = dict(served=float(r0['n_served']), revenue=float(r0['revenue']),
                                 energy_kwh=float(r0['energy_kwh']),
                                 service_rate=r0.get('service_rate'))

    sens = {}
    for s in shares:
        ub = np.asarray(rows[s])
        entry = dict(share_of_precool=float(s), rigorous_bound=bool(s == 1.0),
                     mean_upper_bound=float(ub.mean()), min_upper_bound=float(ub.min()))
        for arm in arms:
            u = np.asarray([float(r['utility']) for r in per_day[arm]])
            entry[arm] = dict(mean_utility=float(u.mean()), mean_gap=float(ub.mean() - u.mean()),
                              min_gap=float(np.min(ub - u)))
        sens['share=%.2f' % s] = entry

    out = dict(
        stage='pickup_a1_budget_reachability_bound_addendum',
        registered_bound_read_only=dict(result_sha256=hashlib.sha256((HERE / 'RESULT.json').read_bytes()).hexdigest(),
                                        registration_sha256=hashlib.sha256(
                                            (HERE / 'REGISTRATION.json').read_bytes()).hexdigest()),
        sensitivity=sens,
        interpretation=(
            'Only share=1.00 is the registered rigorous bound (mean upper bound 2544 on the time-window-plus-horizon '
            'relaxation). The other rows divide the per-order pre-cooling lower bound by the observed population '
            'pre-cooling share; that is NOT proved per order, so those rows are sensitivities and must not be quoted as '
            'bounds. The sensitivity shows that if the true per-order energy were the pre-cooling energy divided by the '
            'recorded 0.72 share, the bound would fall to within a few percent of the strongest recorded control, i.e. '
            'the entire open headroom question is carried by how much of the true per-order energy the relaxation is '
            'allowed to ignore.'),
        mix=mix, observed_mix=observed_mix,
        gap_localisation=(
            'The vehicle-time relaxation is not binding at all (its bound equals the total reachable revenue), so the '
            'gap is produced solely by the energy relaxation. Within the energy relaxation the selected class mix is '
            'dominated by the cheapest-to-cool classes, whereas the recorded controls serve 72/56/26 percent of classes '
            '0/1/2; the gap is therefore a selection effect that the energy-only relaxation cannot exclude.'),
        next_bounded_steps=[
            'derive a rigorous per-order lower bound on the whole segment energy (pre-cooling plus door recovery plus the '
            'minimum cooling for a direct depot-order-depot trip) from coldchain_state.transition_segment, which is a '
            'small bounded derivation and would tighten the rigorous bound directly',
            'keep the joint time-window and fleet coupling in the relaxation instead of dropping it, e.g. one time-indexed '
            'vehicle-hours relaxation per vehicle, which is what prevents a policy from simply serving every '
            'cheap-to-cool order',
        ],
        no_policy_run=True, no_model_loaded=True,
        flags=dict(formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
                   scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
                   stable_online_gain=False),
    )
    (HERE / 'SENSITIVITY_ADDENDUM.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n',
                                                    encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('sensitivity',)}, ensure_ascii=False)[:200])
    for k, v in sens.items():
        print(k, 'mean UB', round(v['mean_upper_bound'], 2), 'rigorous', v['rigorous_bound'],
              'gap vs cond_hist', round(v['cond_hist']['mean_gap'], 2),
              'gap vs explicit_feat', round(v['explicit_feat']['mean_gap'], 2))


if __name__ == '__main__':
    raise SystemExit(main())
