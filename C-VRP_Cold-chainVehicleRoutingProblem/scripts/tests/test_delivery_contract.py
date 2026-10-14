"""Meaningful delivery invariants and real common-incumbent baseline smoke."""
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'delivery'))
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate


def fixture():
    return {'coords': [[0.,0.],[3.,0.],[6.,0.],[0.,3.]], 'demands': [0.,2.,2.,2.],
        'temp_class': [0,0,1,2], 'reveal_time': [0.,0.,0.,0.], 'ready_time': [0.,0.,0.,0.],
        'tw_start': [0.,0.,0.,0.], 'tw_end': [8.,8.,8.,8.],
        'service_time': [0.,0.01,0.01,0.01]}


def tests(out=None, baselines=False):
    contract = DeliveryContract(capacity=6., vehicles=2)
    data = fixture()
    expected = evaluate(data, [[1,2,3]], contract)
    assert expected['hard'] and expected['complete']
    events = expected['traces'][0]['events']
    delivery = [e for e in events if e['kind']=='deliver']
    assert [(e['load_before'],e['load_after']) for e in delivery] == [(6.,4.),(4.,2.),(2.,0.)]
    assert expected['traces'][0]['terminal_manifest'] == []
    coords = np.asarray(data['coords'])
    distance = sum(np.linalg.norm(coords[a]-coords[b]) for a,b in zip([0,1,2,3],[1,2,3,0]))
    assert abs(expected['distance_km']-distance)<1e-12
    p = contract.physics()
    for item in expected['delivered']:
        c = data['temp_class'][item['order_id']]
        exposure = item['delivery_finish']-item['loading_finish']
        q = (p.quality.initial_value[c]*math.exp(-p.quality.rate_per_hour[c]*exposure)
             if p.quality.dynamics_order[c]=='first' else p.quality.initial_value[c]-p.quality.rate_per_hour[c]*exposure)
        assert abs(q-item['quality'])<1e-12
    assert abs(sum(d['quality_loss_value'] for d in expected['delivered'])-expected['quality_loss_value'])<1e-12
    billed_energy = sum(e.get('energy_kwh',0.)+e.get('preconditioning_energy_kwh',0.)+
                        e.get('door_recovery_kwh',0.) for e in events)
    assert abs(billed_energy-expected['energy_kwh'])<1e-12
    assert abs(expected['cost']-(contract.distance_weight*distance+
        contract.energy_weight*billed_energy+contract.quality_weight*expected['quality_loss_value']))<1e-12
    for routes,error in [([[1,1,2,3]],'duplicate_ownership'), ([[1,2]],'incomplete_delivery')]:
        r=evaluate(data,routes,contract);assert not r['hard'] and error in r['errors']
    r=evaluate(data,[[1,2,3]],replace(contract,capacity=5.));assert not r['hard'] and 'capacity' in r['errors']
    r=evaluate(data,[[1],[2],[3]],contract);assert not r['hard'] and 'fleet_limit' in r['errors']
    bad=fixture();bad['reveal_time'][3]=0.1;bad['ready_time'][3]=0.1
    r=evaluate(bad,[[1,2,3]],contract);assert not r['hard'] and 'unreleased_or_unready_loading' in r['errors']
    bad=fixture();bad['tw_end'][2]=0.01
    r=evaluate(bad,[[1,2,3]],contract);assert not r['hard'] and 'time_window' in r['errors']
    r=evaluate(data,[[1,2,3]],replace(contract,horizon_h=0.1));assert not r['hard'] and 'return_horizon' in r['errors']
    r=evaluate(data,[[1,2,3]],contract,deadline=0.,clock=lambda:1.);assert not r['hard'] and 'certification_deadline' in r['errors']
    empty={k:v[:1] for k,v in data.items()}
    r=evaluate(empty,[],contract);assert r['hard'] and r['cost']==r['energy_kwh']==r['distance_km']==0
    repeated=evaluate(data,[[1,2,3]],contract)
    assert expected == repeated
    result={'delivery_contract_pass':True,'manifest_conservation':True,
        'independent_distance_quality_accounting':True,'trace_energy_no_double_billing':True,
        'unreleased_duplicate_missing_capacity_fleet_tw_return_deadline_negatives':True,
        'empty_wave_zero_cost':True,'contract_hash':contract.identity(),
        'qualification':'engineering_only_no_positive_method_evidence','online_gain_established':False}
    if baselines:
        # QA prewarms the solver import only; each call still includes its own
        # instance preparation, initial certification, search and final certification.
        from ortools.constraint_solver import pywrapcp
        from delivery_baselines import improve_ortools,improve_regret_lns
        arms={name:func(data,[[1,2,3]],contract,time_limit_s=1.0) for name,func in
            [('ortools',improve_ortools),('regret_lns',improve_regret_lns)]}
        for name, arm in arms.items():
            check=evaluate(data,arm['routes'],contract)
            assert check['hard'] and check['cost']<=expected['cost']+1e-9
            assert arm['same_initial_assignment'] and not arm['online_gain_established']
            assert arm['wall_s']<=1.1, (name,arm['wall_s'])
        result['baseline_smoke']={name:{k:a[k] for k in ('method','wall_s','timeout','same_initial_assignment')}
            for name,a in arms.items()}
        result['qa_import_prewarm'] = 'ortools static import only; not formal pilot timing policy'
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        (out/'delivery_engineering_report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        (out/'delivery_contract.json').write_text(json.dumps(contract.payload(),indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--baselines',action='store_true')
    a=p.parse_args();tests(a.out,a.baselines)
