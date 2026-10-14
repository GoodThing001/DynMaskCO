"""Explicit old-C0 snapshot features and unvalidated paired route objective.

No online contract or physically calibrated commodity kinetics is implemented.
Future requests are rejected here rather than passing hidden attributes through
an upstream model without visibility attention.
"""
import hashlib
import json
import math
import numpy as np

UNITS = {'coords': 'km', 'times': 'h', 'demands': 'quantity',
         'target_temperature': 'degC', 'quality': 'contract_metric_class_specific'}
FIELDS = ['normalized_x', 'normalized_y', 'demand_over_capacity',
          'tw_start_over_horizon', 'tw_end_over_horizon', 'service_over_horizon',
          'release_over_horizon', 'ready_over_horizon',
          'contract_target_temperature_c_over_20', 'contract_initial_quality_over_9']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def make_features(data, contract, normalized_coords, units, contract_hash, wave_h=0.):
    from delivery_evaluator import validate_instance
    if units != UNITS or any(type(v) is not str for v in units.values()):
        raise ValueError('Explicit exact engineering units required')
    if type(contract_hash) is not str or contract_hash != contract.identity():
        raise ValueError('Contract identity mismatch')
    if type(wave_h) not in (int, float) or not math.isfinite(wave_h) or wave_h < 0:
        raise ValueError('Invalid snapshot time')
    validate_instance(data, contract)
    physics = contract.physics()
    if physics.units.distance_km_per_unit != 1 or physics.units.hours_per_time_unit != 1:
        raise ValueError('This engineering map only accepts physical km/h data')
    n = len(data['demands'])
    coord = np.asarray(normalized_coords, np.float32)
    if coord.shape != (n, 2) or not np.isfinite(coord).all():
        raise ValueError('Invalid visible coordinate normalization')
    # Snapshot-only: feature masking alone cannot establish dynamic causality.
    if np.any(np.asarray(data['reveal_time']) > wave_h) or np.any(np.asarray(data['ready_time']) > wave_h):
        raise ValueError('Unreleased/unready requests: dynamic visibility attention not qualified')
    classes = np.asarray(data['temp_class'], int)
    initial = np.asarray(physics.quality.initial_value, np.float32)[classes]
    if 'initial_quality' in data and not np.array_equal(np.asarray(data['initial_quality'], np.float32), initial):
        raise ValueError('Old C0 evaluator derives class quality; attribute override would disagree')
    x = np.zeros((1, n, 10), np.float32)
    x[0, :, :2] = coord
    x[0, :, 2] = np.asarray(data['demands'], np.float32) / contract.capacity
    for column, field in enumerate(('tw_start', 'tw_end', 'service_time', 'reveal_time', 'ready_time'), 3):
        x[0, :, column] = np.asarray(data[field], np.float32) / contract.horizon_h
    x[0, :, 8] = np.asarray(physics.thermal.target_temperature_c, np.float32)[classes] / 20.
    x[0, :, 9] = initial / 9.
    x[0, 0, 3:] = 0.  # Depot does not represent a cargo commodity.
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite engineering features')
    return x


def extend_projection(kernel):
    """Keep original 3 rows, add 7 zero rows without random initialization."""
    a = np.asarray(kernel)
    if a.shape != (3, 512) or not np.isfinite(a).all():
        raise ValueError('Actual original projection must be 3x512 finite')
    return np.concatenate((a, np.zeros((7, 512), a.dtype)), axis=0)


def pair_label(before, candidate):
    if before['hard'] is not True or candidate['hard'] is not True:
        raise ValueError('Only complete independently hard-certified pairs carry preference')
    if before['complete'] is not True or candidate['complete'] is not True:
        raise ValueError('Incomplete pair')
    delta = float(before['cost']) - float(candidate['cost'])
    if not math.isfinite(delta):
        raise ValueError('Nonfinite certified pair')
    return 0 if abs(delta) <= 1e-8 else (1 if delta > 0 else -1)


def objective_terms(logits, target, alternative, label, coefficient):
    import jax
    import jax.numpy as jnp
    nodes = logits.shape[-1]
    logp = jax.nn.log_softmax(logits, axis=-1)
    ce = -(logp[:, 1:] * target[:, 1:]).mean() * (nodes / 2)
    # Fixed degree-2 customer average, independent of route count/zero padding.
    score_delta = ((logp[:, 1:] * (target[:, 1:] - alternative[:, 1:])).sum()
                   / (2 * (nodes - 1)))
    preference = jnp.where(label == 0, 0., jax.nn.softplus(-label * score_delta))
    return ce + coefficient * preference, (ce, preference, score_delta)
