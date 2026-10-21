"""Source-specific value/unit calculations. No TA-to-price substitution.

The quote module does not infer quality from routes. A missing source-index
mapping is deliberately visible instead of silently accepting normalized TA.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCRIPTS = next(p/'scripts' for p in Path(__file__).resolve().parents if (p/'scripts/project_paths.py').is_file())
sys.path.insert(0, str(SCRIPTS))
from project_paths import EXTENSION_ROOT

CONFIG = EXTENSION_ROOT/'configs/static_delivery/direct_value_evidence_v2.json'


def load():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def finite_nonnegative(x, name):
    if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def quoted_values(c, high_USD_per_pallet, discount_fraction):
    s = c['gaukler_2023_tomato']
    lo, hi = s['high_quality_price_experiment_range_USD_per_pallet']
    dl, dh = s['low_quality_discount_fraction_experiment_range']
    finite_nonnegative(high_USD_per_pallet, 'price')
    finite_nonnegative(discount_fraction, 'discount')
    if not lo <= high_USD_per_pallet <= hi or not dl <= discount_fraction <= dh:
        raise ValueError('outside published numerical experiment bounds')
    p = s['packing']
    kg = p['carton_lb']*p['cartons_per_pallet']*p['lb_to_kg_exact']
    return {'currency': 'USD', 'pallet_kg_derived': kg, 'high_quality_USD_per_pallet': high_USD_per_pallet, 'low_quality_USD_per_pallet': high_USD_per_pallet*discount_fraction, 'high_quality_USD_per_kg': high_USD_per_pallet/kg, 'low_quality_USD_per_kg': high_USD_per_pallet*discount_fraction/kg, 'normalized_high_value': 1, 'normalized_low_value': discount_fraction, 'source_doi': s['doi'], 'quality_from_routes_inferred': False, 'customer_low_grade_acceptance_implied': False}


def source_quality_eligible(c, quality, unit_id):
    s = c['gaukler_2023_tomato']
    if unit_id != s['quality_unit_id']:
        raise ValueError('quality_unit_mismatch: TA retention, mass fraction and fish score are not this source index')
    finite_nonnegative(quality, 'quality')
    return quality >= s['wholesale_quality_threshold']


def normalize_money(amount, currency, reference, reference_currency):
    if currency != reference_currency:
        raise ValueError('currency_mismatch: normalization does not perform foreign exchange')
    if not math.isfinite(amount) or not math.isfinite(reference) or reference <= 0:
        raise ValueError('finite amount and positive reference required')
    return amount/reference


def electrical_energy_from_heat(thermal_kWh, COP):
    finite_nonnegative(thermal_kWh, 'thermal_energy')
    if COP is None or not math.isfinite(COP) or COP <= 0:
        raise ValueError('explicit positive COP required; cooling heat is not electric input')
    return thermal_kWh/COP


def fuel_from_heat(thermal_kWh, COP, electric_efficiency, fuel_kWh_per_L):
    electric = electrical_energy_from_heat(thermal_kWh, COP)
    if electric_efficiency is None or not 0 < electric_efficiency <= 1 or fuel_kWh_per_L is None or not math.isfinite(fuel_kWh_per_L) or fuel_kWh_per_L <= 0:
        raise ValueError('explicit valid electric conversion efficiency and fuel energy perL required')
    return electric/(electric_efficiency*fuel_kWh_per_L)


def require_complete_routing_profile(c):
    if not c['adoption_decision']['complete_tomato_quality_value_route_model_ready']:
        raise ValueError('source_quality_dynamics_and_currency_contract_incomplete; no TA fallback is allowed')


def calculations(c):
    s = c['gaukler_2023_tomato']
    rows = []
    for high in [s['high_quality_price_experiment_range_USD_per_pallet'][0], s['high_quality_reference_USD_per_pallet'], s['high_quality_price_experiment_range_USD_per_pallet'][1]]:
        for point in s['discount_diagnostic_points']:
            rows.append({**quoted_values(c, high, point['value']), 'discount_point_role': point['role']})
    e = c['energy_accounting']
    return {'scope': c['status'], 'source_specific_value_quotes': rows, 'v1_thermal_cost_unit_check': {'EUR_per_thermal_kWh': e['v1_benchmark_heat_cost_coefficient_EUR_per_W_min']*1000*60, 'v1_wall_heat_kW': e['v1_wall_heat_W_at4C_derived']/1000, 'EUR_per_vehicle_hour': e['v1_wall_heat_W_at4C_derived']*60*e['v1_benchmark_heat_cost_coefficient_EUR_per_W_min'], 'is_electricity_tariff': False}, 'small_unit_same_condition_check': {'electric_input_kW': e['small_CO2_unit_same_condition_cooling_kW']/e['small_CO2_unit_same_condition_COP'], 'is20t_benchmark_vehicle': False}, 'quality_mapping_ready': False, 'new_routing_optimization_run': False, 'new_profit_result': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    if a.out.exists():
        raise FileExistsError('new output file required')
    c = load()
    result = calculations(c)
    result['config_sha256'] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_specific_value_quotes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
