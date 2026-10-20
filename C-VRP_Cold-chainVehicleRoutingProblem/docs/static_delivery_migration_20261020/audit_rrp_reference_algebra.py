"""Exploratory research arithmetic audit of fixed RRP records (no optimization).

This is NOT an independent route evaluator or a formal qualification gate.
It substitutes the saved author arrival/stop/speed vectors into equations;
it does not independently reconstruct those vectors from route and waits.
The 0.1 EUR screen was chosen after inspection, not supplied by the authors.

Source: Ceschia et al., Energies 13(23), 6214 (2020), Table 1,
Equations 2, 3, 14, 15 and public instance/solution files.
The fuel conversion is 0.30 L/kWh (Table 1), multiplied by electrical
energy. The printed division by sigma in Eq. 13 conflicts with its units.
Column meanings for temperature_ranges are an inference checked against
published costs, not a schema supplied by the repository README.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def numeric_line(text, label):
    pattern = re.escape(label) + r':[^\S\r\n]*\r?\n([^\n]+)'
    for match in re.finditer(pattern, text):
        try:
            return [float(value) for value in match.group(1).split()]
        except ValueError:
            continue
    # One author file duplicates its Route header before its actual values.
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == label + ':':
            for candidate in lines[idx + 1:idx + 4]:
                try:
                    values = [float(value) for value in candidate.split()]
                    if values:
                        return values
                except ValueError:
                    pass
    raise ValueError('Missing numeric line: ' + label)


def reported_costs(text):
    costs = {}
    aliases = {'traction': 'Traction', 'transmission': 'Transmission',
               'infiltration': 'Infiltration', 'refrigeration': 'Refrigeration',
               'driver': 'Driver', 'total': 'Total'}
    for key, label in aliases.items():
        pattern = label + r'(?: Consumption| cost| cons)?\s*:\s*([\d.]+)'
        match = re.search(pattern, text)
        if match:
            costs[key] = float(match.group(1))
    return costs


def audit_one(instance_path, solution_path):
    raw = instance_path.read_bytes()
    data = json.loads(raw)
    solution_raw = solution_path.read_bytes()
    text = solution_raw.decode('utf-8-sig')
    route = [int(value) for value in numeric_line(text, 'Route')]
    arrivals = numeric_line(text, 'Arrival times')
    stops = numeric_line(text, 'Stop times')
    speeds = numeric_line(text, 'Speeds')
    if not (len(arrivals) == len(route) + 1 and
            len(stops) == len(speeds) == len(route)):
        raise ValueError('Inconsistent vector lengths: ' + instance_path.name)
    full_route = route + [0]
    load = sum(data['demands']) * data['mass']
    traction_l_per_tour = 0.0
    for idx, speed in enumerate(speeds):
        distance = data['distances'][full_route[idx]][full_route[idx + 1]]
        traction_l_per_tour += (
            data['A'] * distance * (data['carb_weight'] + load)
            + data['B'] * distance / speed
            + data['C'] * distance * speed ** 2)
        load -= data['demands'][full_route[idx + 1]] * data['mass']
    transmission_electric_kwh_year = 0.0
    infiltration_electric_kwh_year = 0.0
    for month, days in enumerate(data['month_days']):
        for idx in range(len(route)):
            hour = int(arrivals[idx] // 3600)
            if hour < 0 or hour >= 24:
                raise ValueError('Arrival outside one-day author model')
            row = data['temperature_ranges'][month * 24 + hour]
            cop = row[8]
            transmission_electric_kwh_year += (
                days * row[7] * (arrivals[idx + 1] - arrivals[idx]) / 3600 / cop)
            if idx:
                if stops[idx] < 40:
                    raise ValueError('Eq. 15 requires opening longer than 40 seconds')
                infiltration_electric_kwh_year += (
                    days * (row[5] + row[6] * (stops[idx] - 40)) / 3600 / cop)
    fuel_l_per_kwh = 0.30  # Source: paper Table 1; do not silently use JSON fuel_cons=3.
    calculated = {
        'traction': traction_l_per_tour * data['days'] * data['fuel_cost'],
        'transmission': transmission_electric_kwh_year * fuel_l_per_kwh * data['fuel_cost'],
        'infiltration': infiltration_electric_kwh_year * fuel_l_per_kwh * data['fuel_cost'],
        'driver': data['driver_wage'] * data['days'] * (arrivals[-1] - arrivals[0]),
    }
    calculated['refrigeration'] = calculated['transmission'] + calculated['infiltration']
    calculated['total'] = calculated['traction'] + calculated['refrigeration'] + calculated['driver']
    reported = reported_costs(text)
    residual = {key: calculated[key] - value for key, value in reported.items()}
    alternate_total_without_driver = calculated['total'] - calculated['driver']
    source_total = (calculated['total'] if instance_path.stem.startswith('D1')
                    else alternate_total_without_driver)
    return {
        'instance': instance_path.stem,
        'instance_sha256': hashlib.sha256(raw).hexdigest(),
        'solution_sha256': hashlib.sha256(solution_raw).hexdigest(),
        'calculated_eur_per_year': calculated,
        'author_reported_eur_per_year': reported,
        'calculated_minus_author_eur': residual,
        'signed_relative_residual_by_reported_component': {
            key: value / reported[key] if reported[key] else None
            for key, value in residual.items()},
        'source_objective_identity': (
            'D1 paper Table 5: fuel plus time-dependent wage'
            if instance_path.stem.startswith('D1')
            else 'D2 paper Table 7(b): fuel only, wage deliberately neglected'),
        'calculated_source_objective_total_eur': source_total,
        'source_objective_total_minus_reported_eur': source_total - reported['total'],
        'source_objective_total_relative_residual': (
            (source_total - reported['total']) / reported['total']),
        'max_abs_residual_eur': max(abs(value) for value in residual.values()),
        'full_total_matches_within_0_1_eur': abs(residual.get('total', math.inf)) <= 0.1,
        'alternate_total_excluding_driver_eur': alternate_total_without_driver,
        'alternate_total_excluding_driver_minus_reported_eur': (
            alternate_total_without_driver - reported['total'] if 'total' in reported else None),
        'json_fuel_cons_raw': data.get('fuel_cons'),
        'fuel_conversion_used_l_per_kwh': fuel_l_per_kwh,
        'route': route, 'arrival_seconds': arrivals,
        'electric_energy_year_kwh': {
            'transmission': transmission_electric_kwh_year,
            'infiltration': infiltration_electric_kwh_year},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    extension = Path(__file__).resolve().parents[2]
    parser.add_argument('--source', type=Path, default=(
        extension / 'data/public_sources_20261020/refrigerated_routing_rrp'))
    parser.add_argument('--output', type=Path, default=(
        Path(__file__).resolve().parent / 'rrp_reference_algebra_audit_v2.json'))
    args = parser.parse_args()
    results = []
    for solution in sorted((args.source / 'Solutions').rglob('sol-*.txt')):
        name = solution.stem[4:]
        instance = args.source / 'Datasets' / name[:2] / (name + '.json')
        results.append(audit_one(instance, solution))
    report = {
        'source': 'https://bitbucket.org/sceschia/the-refrigerated-routing-problem/',
        'author_commit': 'f5c5317a5ecd8fe64df779e79bd0397904bc0f9e',
        'scope': 'Read-only evaluation of fixed public author routes, no solver or training',
        'status': 'exploratory research; not an independent product evaluator or qualification gate',
        'independent_route_timing_reconstruction': False,
        'screen_0_1_eur_status': 'Post-inspection exploratory screen, not author acceptance tolerance',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'units': 'Costs EUR/year; distance km; speed km/h; clocks seconds',
        'formulas': {
            'traction_l_tour': 'sum(A*d*(curb_weight+remaining_load_kg)+B*d/v+C*d*v^2)',
            'transmission_kwh_year': 'sum_month,step(month_days*table[7]*delta_arrival_s/3600/table[8])',
            'infiltration_kwh_year': 'sum_month,customer(month_days*(table[5]+table[6]*(stop_s-40))/3600/table[8])',
            'refrigeration_eur_year': '(transmission_kwh_year+infiltration_kwh_year)*0.30*fuel_cost',
            'driver_eur_year': 'driver_wage*days*(final_arrival_s-first_arrival_s)',
        },
        'temperature_column_interpretation': {
            '0': 'time interval start hour', '1': 'time interval end hour',
            '2': 'temperature slope (inferred, not needed by recovered tabular evaluator)',
            '3': 'temperature intercept (inferred, not needed by recovered tabular evaluator)',
            '4': 'outdoor temperature used for precomputed thermal fields, degC (inferred)',
            '5': 'first-40-seconds infiltration thermal energy, kJ (algebra verified)',
            '6': 'subsequent infiltration thermal power, kW (algebra verified)',
            '7': 'transmission thermal power, kW (algebra verified)',
            '8': 'COP, dimensionless (algebra verified)',
        },
        'limitations': [
            'Column labels inferred from paper units and public outputs; author README has no explicit schema.',
            'Paper Table 1 sigma=0.30 L/kWh conflicts with printed division in Eq. 13.',
            'Raw JSON fuel_cons=3 has no published scaling schema; this audit uses sourced 0.30 L/kWh.',
            'Uses paper arrival-slot lookup, not continuous temperature integration across slots.',
            'D2 public solutions match the deliberately wage-free objective of paper Table 7(b).',
            'Full wage-inclusive costs and source-objective costs are retained separately; objectives are never mixed.',
            'Saved author clocks, service durations and speeds are inputs, not independently certified outputs.',
        ],
        'count': len(results),
        'full_total_match_count_0_1_eur': sum(row['full_total_matches_within_0_1_eur'] for row in results),
        'results': results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('count', 'full_total_match_count_0_1_eur')}, indent=2))
    print('Maximum residual, grouped by dataset:')
    for dataset in ('D1', 'D2'):
        rows = [row for row in results if row['instance'].startswith(dataset)]
        print(dataset, len(rows), max((row['max_abs_residual_eur'] for row in rows), default=0))


if __name__ == '__main__':
    main()
