"""Product-specific literature kinetics; never modifies historical contracts.

Input temperatures are PRODUCT temperatures, not vehicle air setpoints.
Piecewise-constant quadrature is exact for the supplied temperature segments.
This is a literature-parameterized simulator, not business calibration or safety.
"""
from pathlib import Path
import hashlib
import json
import math

PROFILE_FILE = Path(__file__).with_name('literature_quality_profiles_v1.json')
R_J_MOL_K = 8.314


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite numeric '+label+' required')
    return float(value)


def load_profiles():
    raw = PROFILE_FILE.read_bytes()
    data = json.loads(raw)
    return data['profiles'], hashlib.sha256(raw).hexdigest()


def rate_per_hour(profile, product_temperature_c, *, allow_extrapolation=False):
    t = finite(product_temperature_c, 'product_temperature_c')
    if t <= -273.15:
        raise ValueError('nonpositive absolute temperature')
    lo, hi = profile['experimental_temperature_range_c']
    outside = not lo <= t <= hi
    if outside and not allow_extrapolation:
        raise ValueError('temperature outside paper experimental range')
    ea = profile['activation_energy_j_mol']
    if profile['rate_parameterization'] == 'arrhenius_prefactor_per_day':
        k = math.exp(profile['ln_prefactor_per_day']-ea/(R_J_MOL_K*(t+273.15)))/24.
    else:
        k = profile['reference_rate_per_day']/24.*math.exp(
            -ea/R_J_MOL_K*(1./(t+273.15)-1./(profile['reference_temperature_c']+273.15)))
    if not math.isfinite(k):
        raise ValueError('nonfinite rate')
    return k, outside


def evaluate_product(profile_id, initial_value, segments, *, allow_extrapolation=False):
    """Segments: start_h/end_h/product_temperature_c, contiguous absolute clock.

    Returns raw nutrient ratio or sensory score. No silent clipping, no conversion
    of sensory score points to quality fractions or monetary revenue.
    """
    profiles, digest = load_profiles()
    if profile_id not in profiles:
        raise ValueError('explicit commodity profile required')
    p = profiles[profile_id]
    q0 = finite(initial_value, 'initial_value')
    if p['order'] == 1 and q0 <= 0:
        raise ValueError('positive initial nutrient value required')
    if p['order'] == 0 and not 1 <= q0 <= 9:
        raise ValueError('initial sensory score must be within 1..9')
    total = 0.; previous = None; rows = []
    for s in segments:
        start = finite(s['start_h'], 'start_h'); end = finite(s['end_h'], 'end_h')
        if start < 0 or end < start:
            raise ValueError('negative or reversed clock')
        if previous is not None and not math.isclose(start, previous, rel_tol=0., abs_tol=1e-10):
            raise ValueError('gap or overlap in cargo temperature trajectory')
        k, extrapolated = rate_per_hour(p, s['product_temperature_c'], allow_extrapolation=allow_extrapolation)
        decrement = k*(end-start)
        total += decrement
        rows.append(dict(s, rate_per_hour=k, exposure=decrement, extrapolated=extrapolated))
        previous = end
    final = q0*math.exp(-total) if p['order'] == 1 else q0-total
    threshold = p['sensory_rejection_score']
    return {'profile_id':profile_id, 'profile_sha256':digest, 'source_doi':p['source_doi'],
            'initial_value':q0, 'final_value':final, 'loss':q0-final,
            'value_unit':p['value_unit'], 'integrated_exposure':total, 'segments':rows,
            'sensory_below_threshold':None if threshold is None else final < threshold,
            'outside_sensory_scale':p['order']==0 and final < 1,
            'extrapolated':any(x['extrapolated'] for x in rows),
            'rate_evidence':p['rate_evidence'], 'business_calibrated':False,
            'food_safety_certified':False, 'monetary_loss':None}


def evaluate_cargo(profile_id, initial_value, segments, *, direction, begin_h, end_h,
                   allow_extrapolation=False):
    """Caller supplies certified pickup service_finish/return_arrival clocks,
    or delivery load_finish/service_finish clocks. Full interval must be covered.
    Warehouse stages require separate explicitly supplied temperature records.
    """
    if direction not in ('pickup', 'delivery'):
        raise ValueError('pickup or delivery direction required')
    begin = finite(begin_h, 'begin_h'); end = finite(end_h, 'end_h')
    if begin < 0 or end < begin:
        raise ValueError('invalid cargo interval')
    segments = list(segments)
    if not segments:
        if begin != end:
            raise ValueError('nonempty cargo trajectory required')
    elif (not math.isclose(segments[0]['start_h'], begin, rel_tol=0., abs_tol=1e-10)
          or not math.isclose(segments[-1]['end_h'], end, rel_tol=0., abs_tol=1e-10)):
        raise ValueError('cargo lifecycle coverage mismatch')
    result = evaluate_product(profile_id, initial_value, segments, allow_extrapolation=allow_extrapolation)
    result.update(direction=direction, begin_h=begin, end_h=end,
                  cargo_clock_semantics='service_finish_to_return_arrival' if direction=='pickup'
                  else 'load_finish_to_service_finish')
    return result


def electrical_energy_kwh(segments):
    """Integrate electrical input kW × h; cooling output must not be passed here.
    Each segment explicitly labels power_basis ('measurement' or 'simulation').
    No COP, door heat or vehicle heat capacity is inferred from food kinetics.
    """
    total = 0.; previous = None; measured = True
    for s in segments:
        start = finite(s['start_h'], 'start_h'); end = finite(s['end_h'], 'end_h')
        power = finite(s['electrical_input_kw'], 'electrical_input_kw')
        if start < 0 or end < start or power < 0:
            raise ValueError('invalid power trajectory')
        if previous is not None and not math.isclose(start, previous, rel_tol=0., abs_tol=1e-10):
            raise ValueError('power trajectory gap or overlap')
        if s['power_basis'] not in ('measurement', 'simulation'):
            raise ValueError('explicit power provenance required')
        measured = measured and s['power_basis']=='measurement'
        total += power*(end-start); previous=end
    return {'electrical_energy_kwh':total, 'all_segments_labeled_measurement':measured,
            'measurement_provenance_independently_verified':False}
