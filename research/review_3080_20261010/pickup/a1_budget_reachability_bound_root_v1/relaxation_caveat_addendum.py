"""Addendum: the vehicle-time relaxation never bound, and its construction is not one-sided.

Read-only with respect to the registered bound artifacts.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'RESULT.json').read_text(encoding='utf-8'))

stats = {}
for variant in ('tw_only', 'tw_and_horizon'):
    stricter, equal, capacity = 0, 0, 0
    for d in R['days']:
        e = d[variant]['lp_energy_bound']
        t = d[variant]['lp_time_bound']
        stricter += int(t < e - 1e-9)
        equal += int(abs(t - e) < 1e-9)
        capacity += int(d[variant]['vehicle_hours_lower_total'] > R['num_vehicles'] * R['horizon'])
    stats[variant] = dict(days=len(R['days']), time_bound_strictly_tighter=stricter,
                          time_and_energy_bounds_equal=equal,
                          days_over_vehicle_hour_capacity=capacity)

out = dict(
    stage='pickup_a1_budget_reachability_bound_addendum_vehicle_time',
    registered_result_sha256=hashlib.sha256((HERE / 'RESULT.json').read_bytes()).hexdigest(),
    registration_sha256=hashlib.sha256((HERE / 'REGISTRATION.json').read_bytes()).hexdigest(),
    finding=stats,
    reading=('The vehicle-time relaxation (R3) never bound on any of the 40 days: its bound is never strictly tighter '
             'than the energy bound, so the reported upper bound equals the energy relaxation everywhere and R3 did not '
             'influence any number in the registered result.'),
    construction_caveat=('R3 as implemented sums one out-and-back trip per served order. That quantity is an UPPER bound '
                         'on the time a chained route needs, not a lower bound, so if R3 had ever been the binding '
                         'relaxation the result would not have been a valid upper bound. R3 never bound, so the '
                         'registered numbers are unaffected; R3 must not be relied on in this form.'),
    valid_tightening_directions=[
        'a rigorous per-order lower bound on the whole segment energy (pre-cooling plus door recovery plus the minimum '
        'cooling for the direct depot-order-depot trip) derived from coldchain_state._advance_temperatures, which with all '
        'three zones active gives at least (0.30*0.143 + 1.80*0.714 + 3.20*0.860) kWh per hour of segment time',
        'a joint time-window and fleet relaxation that keeps the coupling R1 and R3 drop separately',
    ],
    no_artifact_modified=True,
    flags=dict(formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
               scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
               stable_online_gain=False),
)
out_path = HERE / 'RELAXATION_CAVEAT_ADDENDUM.json'
assert not out_path.exists()
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(out['finding'], ensure_ascii=False))
