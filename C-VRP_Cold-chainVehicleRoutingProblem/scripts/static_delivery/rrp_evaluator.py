"""Independent static delivery clocks/load and the sourced RRP cost equations."""
from __future__ import annotations

import math
import numpy as np
from rrp_data import source_protocol, PRIMARY_PROFILE, TIME_DOMAINS

PROFILES = (PRIMARY_PROFILE, 'paper', 'archive_compatibility', 'archive_pre_service')
OBJECTIVES = ('with_wage', 'without_wage')


def speed_at(instance, seconds):
    if not math.isfinite(seconds) or not 0 <= seconds < 86400:
        raise ValueError('clock_outside_day')
    return float(instance.raw['speed_ranges'][int(seconds // 3600)][4])


def service_seconds(instance, quantity, delivered_pallets, profile=PRIMARY_PROFILE):
    if profile not in PROFILES:
        raise ValueError('Unknown service profile')
    r = instance.raw
    factor = 2 if profile in (PRIMARY_PROFILE, 'paper') else 1
    positions = range(int(delivered_pallets), int(delivered_pallets + quantity))
    handling = sum(r['time_up'] + factor * r['row_time'] * (p // r['row_q']) for p in positions)
    return float(r['fix_time'] + 2 * r['opening_time'] + handling)


def evaluate(instance, route, waits_s, objective, *, profile=PRIMARY_PROFILE, time_domain='source_day_domain'):
    if profile not in PROFILES:
        raise ValueError('Unknown service profile')
    if objective not in OBJECTIVES:
        raise ValueError('Unknown objective')
    if time_domain not in TIME_DOMAINS:
        raise ValueError('Unknown time domain')
    p, raw = source_protocol(), instance.raw
    result = {'instance': instance.name, 'source_sha256': instance.source_sha256,
              'profile': profile, 'objective': objective, 'clock': p['clock'], 'time_domain': time_domain,
              'feasible': False, 'complete': False, 'errors': [], 'legs': [],
              'route': list(route), 'waits_s': list(waits_s),
              'initial_load_kg': float(sum(instance.demands) * raw['mass']),
              'terminal_load_kg': None, 'costs': None, 'duration_s': None,
              'qualification': 'MIP_equation_priority_with_disclosed_dimensional_correction_not_all_author_numbers',
              'duration_limit_s': None, 'driving_s': None, 'service_total_s': None,
              'waiting_total_s': None, 'exceeds_operational_9h': None}
    n = instance.nodes
    if len(route) != n or any(type(v) not in (int, np.int32, np.int64) for v in route) or list(route[:1]) != [0] or set(route) != set(range(n)):
        result['errors'] = ['route_coverage']
        return result
    if len(waits_s) != n or any(not isinstance(v, (int, float, np.number)) or not math.isfinite(v) or v < 0 for v in waits_s):
        result['errors'] = ['invalid_waits']
        return result
    if waits_s[0] != 0 and profile != 'archive_compatibility':
        result['errors'] = ['fixed_depot_start']
        return result
    start = float(raw['start_time'] * 3600)
    limit = p['max_tour_hours'] * 3600 if time_domain == 'operational_tour_9h' else 86400 - start
    result['duration_limit_s'] = limit
    t, load = start, result['initial_load_kg']
    delivered_pallets = 0
    traction_l = transmission_kwh = infiltration_kwh = distance_km = 0.0
    for index, current in enumerate(route):
        physical_arrival = t
        wait = float(waits_s[index])
        # Paper ti permits slack before service (Section 3, Eq.26 and CP text).
        # Keep the physical clock separately; the archive diagnostic uses its
        # saved pre-wait arrival convention instead of silently conflating them.
        arrival = physical_arrival if profile == 'archive_compatibility' else physical_arrival + wait
        service = 0.0 if current == 0 else service_seconds(instance, instance.demands[current], delivered_pallets, profile)
        departure = physical_arrival + wait + service
        if departure >= 86400:
            result['errors'].append('clock_outside_day')
            return result
        if time_domain == 'operational_tour_9h' and departure - start > limit:
            result['errors'].append('tour_horizon')
            return result
        if current:
            delivered_pallets += int(instance.demands[current])
            load -= float(instance.demands[current] * raw['mass'])
        target = int(route[index + 1]) if index + 1 < n else 0
        speed = speed_at(instance, departure)
        distance = float(instance.distances[current, target])
        travel = distance * 3600 / speed
        end = departure + travel
        fuel = (raw['A'] * distance * (raw['carb_weight'] + load)
                + raw['B'] * distance / speed + raw['C'] * distance * speed ** 2)
        result['legs'].append({'from': int(current), 'to': target, 'arrival_s': arrival,
                               'physical_arrival_s': physical_arrival,
                               'service_s': service, 'wait_s': wait, 'departure_s': departure,
                               'travel_s': travel, 'next_physical_arrival_s': end, 'speed_kmph': speed,
                               'distance_km': distance, 'load_kg': load,
                               'traction_l_tour': fuel})
        traction_l += fuel
        distance_km += distance
        t = end
    duration = t - start
    if t >= 86400:
        result['errors'].append('clock_outside_day')
    if time_domain == 'operational_tour_9h' and duration > limit:
        result['errors'].append('tour_horizon')
    for index, leg in enumerate(result['legs']):
        end = result['legs'][index + 1]['arrival_s'] if index + 1 < n else t
        leg['next_arrival_s'] = end
        thermal_hour = int(leg['arrival_s'] // 3600)
        trans = inf = 0.0
        for month, days in enumerate(raw['month_days']):
            row = raw['temperature_ranges'][month * 24 + thermal_hour]
            trans += days * row[7] * (end - leg['arrival_s']) / 3600 / row[8]
            if leg['from']:
                if leg['service_s'] <= p['infiltration_initial_seconds']:
                    result['errors'].append('infiltration_outside_equation_domain')
                    return result
                inf += days * (row[5] + row[6] * (leg['service_s'] - p['infiltration_initial_seconds'])) / 3600 / row[8]
        leg.update(transmission_kwh_year=trans, infiltration_kwh_year=inf, temperature_hour=thermal_hour)
        transmission_kwh += trans
        infiltration_kwh += inf
    costs = {
        'traction_eur_year': traction_l * raw['days'] * raw['fuel_cost'],
        'transmission_eur_year': transmission_kwh * p['fuel_l_per_kwh'] * raw['fuel_cost'],
        'infiltration_eur_year': infiltration_kwh * p['fuel_l_per_kwh'] * raw['fuel_cost'],
        'driver_eur_year': raw['driver_wage'] * raw['days'] * duration if objective == 'with_wage' else 0.0}
    costs['total_eur_year'] = sum(costs.values())
    result.update(complete=True, terminal_load_kg=load, duration_s=duration,
                  distance_km=distance_km, costs=costs, return_arrival_s=t,
                  traction_l_tour=traction_l, transmission_kwh_year=transmission_kwh,
                  infiltration_kwh_year=infiltration_kwh,
                  feasible=not result['errors'] and load == 0,
                  driving_s=sum(leg['travel_s'] for leg in result['legs']),
                  service_total_s=sum(leg['service_s'] for leg in result['legs']),
                  waiting_total_s=sum(waits_s),
                  exceeds_operational_9h=duration > p['max_tour_hours'] * 3600)
    return result
