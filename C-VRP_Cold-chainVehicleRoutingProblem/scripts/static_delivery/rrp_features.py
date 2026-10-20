"""Lossless global source conditions plus node features; no fabricated coords."""
import numpy as np
from rrp_data import source_protocol, PRIMARY_PROFILE, TIME_DOMAINS

SCALARS = ('A', 'B', 'C', 'Tint', 'U', 'surface', 'carb_weight', 'mass', 'days',
           'driver_wage', 'fuel_cost', 'fix_time', 'opening_time', 'row_q', 'row_time',
           'time_up', 'start_time')
PROTOCOL_SCALARS = ('fuel_l_per_kwh', 'pallet_capacity', 'max_tour_hours', 'infiltration_initial_seconds')
FEATURE_DIM = 5 + len(SCALARS) + 24 * 5 + 288 * 9 + 12 + len(PROTOCOL_SCALARS) + 3
SCHEMA = 'rrp-global-source-conditions-v3'


def global_context(instance, objective, profile, time_domain='source_day_domain'):
    profiles = {'paper': 1, 'archive_compatibility': 0, 'archive_pre_service': 2, PRIMARY_PROFILE: 3}
    if objective not in ('with_wage', 'without_wage') or profile not in profiles or time_domain not in TIME_DOMAINS:
        raise ValueError('Unknown objective/profile condition')
    raw, protocol = instance.raw, source_protocol()
    values = np.concatenate((np.array([raw[k] for k in SCALARS]),
                             np.asarray(raw['speed_ranges']).ravel(),
                             np.asarray(raw['temperature_ranges']).ravel(),
                             np.asarray(raw['month_days']),
                             np.array([protocol[k] for k in PROTOCOL_SCALARS]),
                             np.array([objective == 'with_wage', profiles[profile], time_domain == 'operational_tour_9h'])))
    # Fixed, invertible signed-log transform in the documented source units.
    # All month/hour entries are retained, rather than using hindsight summaries.
    return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32)
