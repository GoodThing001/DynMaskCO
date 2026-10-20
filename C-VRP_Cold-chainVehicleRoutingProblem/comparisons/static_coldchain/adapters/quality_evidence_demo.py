"""Sourced quality changes at laboratory temperatures; no market prices or orders."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCRIPTS=next(p/'scripts' for p in Path(__file__).resolve().parents if (p/'scripts/project_paths.py').is_file())
sys.path.insert(0,str(SCRIPTS))
from project_paths import EXTENSION_ROOT

def calculate(config):
    tomato=config['tomato_TA']; fish=config['mullet_overall_impression']; rows=[]
    for temp in tomato['fitted_temperature_C']:
        rate=tomato['A_per_day']*math.exp(-tomato['Ea_J_per_mol']/(tomato['R_J_per_mol_K']*(temp+273.15)))/24
        for hours in (1,4,8):
            rows.append({'product':'cherry_tomato_TA','temperature_C':temp,'hours':hours,'k_per_hour':rate,
                         'TA_remaining_ratio':math.exp(-rate*hours),'relative_TA_decrease':-math.expm1(-rate*hours),
                         'mass_loss':None,'sales_value':None,'source_doi':tomato['doi']})
    for temp in fish['fitted_temperature_C']+[fish['T_ref_C']]:
        rate=fish['k_ref_score_points_per_day']*math.exp(-fish['Ea_J_per_mol']/fish['R_J_per_mol_K']*(1/(temp+273.15)-1/(fish['T_ref_C']+273.15)))/24
        for hours in (1,4,8):
            rows.append({'product':'chilled_mullet_overall_impression','temperature_C':temp,'hours':hours,
                         'rate_score_points_per_hour':rate,'sensory_score_decrease':rate*hours,
                         'absolute_delivered_score':None,'initial_score':None,'mass_loss':None,'sales_value':None,'source_doi':fish['doi']})
    return rows

def require_profit_inputs(config):
    if not config['net_benefit_implementation_ready']:
        raise ValueError('profit_not_configured: price-quality/yield, initial quality, chilled equipment and units not closed; see required_before_profit')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): raise FileExistsError('new output required')
    path=EXTENSION_ROOT/'configs/static_delivery/product_evidence_v1.json'
    config=json.loads(path.read_text(encoding='utf-8'))
    result={'scope':'LITERATURE_CENTRAL_PARAMETER_CALCULATION_NOT_FIELD_DATA_NOT_LOSS_OF_SALEABLE_MASS_NOT_PROFIT',
            'assumptions':'constant product temperature from the source laboratory study; 1/4/8h are diagnostic durations, not calibrated trips; central fitted parameters without uncertainty intervals',
            'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':calculate(config),
            'chilled_unit_reference_energy': {
                'source_doi':config['chilled_CO2_unit_candidate']['doi'],
                'input_power_kW_derived':config['chilled_CO2_unit_candidate']['input_power_kw_derived'],
                'temperature_C':0, 'ambient_C':30,
                'energy_if_continuously_running_for_8h_kWh':config['chilled_CO2_unit_candidate']['input_power_kw_derived']*8,
                'continuous_operation_is_diagnostic_assumption_not_measured_route':True,
                'monetary_cost':None}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('wrote',len(result['rows']),'sourced quality-only calculations')

if __name__=='__main__': main()
