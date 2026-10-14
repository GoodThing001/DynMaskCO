"""Saved-output audit without invoking either admission controller or route search."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import qualify_pickup_joint_policy_transaction_root_v3 as old
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = old.P / 'undecided_public_input_root_v2/RESULT.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    reg = json.loads((HERE / 'PICKUP_AUDIT_REGISTRATION.json').read_text(encoding='utf-8'))
    result = json.loads((HERE / 'PICKUP_MIDCOURSE_RESULT.json').read_text(encoding='utf-8'))
    assert sha(__file__) == reg['audit_sha256']
    assert sha(INPUT) == reg['input_sha256']
    assert sha(HERE / 'PICKUP_MIDCOURSE_RESULT.json') == reg['result_sha256']
    _, decode = old.codecs()
    packet = decode(json.loads(INPUT.read_text(encoding='utf-8'))['records'][reg['record_position']]['packet'])
    cert = decode(result['certificate'])
    checks = {}
    def ck(k,v): checks[k] = bool(v)
    current = packet['admission_input']['current_order']
    prior = packet['snapshot']['replanner_state']
    final_a = set(result['accepted']); final_r = set(result['rejected'])
    flat = [(int(v),o) for v,route in result['plan'].items() for o in route]
    ck('bound_record',reg['record_position']==2 and result['record_position']==2)
    ck('bound_input_hash',result['original_input_sha256']==reg['input_sha256'])
    ck('bound_order_clock',result['order']==current and result['clock']==packet['snapshot']['clock'])
    ck('initial_pending',current in packet['admission_input']['pending'] and current not in prior['accepted'] and current not in prior['rejected'])
    ck('previous_accepted',set(result['previous_accepted'])==set(prior['accepted']))
    ck('previous_rejected',set(result['previous_rejected'])==set(prior['rejected']))
    ck('accepted_delta',final_a-set(prior['accepted'])=={current})
    ck('rejected_delta',final_r-set(prior['rejected'])==set())
    ck('disjoint',not(final_a & final_r))
    ck('pending_delta',set(result['pending'])==set(packet['admission_input']['pending'])-{current})
    ck('current_decision',result['current_decision']=='accepted' and current in final_a)
    ck('no_duplicate_route',len(flat)==len({o for _,o in flat}))
    ck('route_exact_outstanding',set(o for _,o in flat)==final_a-set(np.flatnonzero(packet['snapshot']['served_mask'])))
    ck('fleet_keys',set(result['plan'])=={str(i) for i in range(packet['snapshot']['num_vehicles'])})
    ck('original_route_commitment',all(status!='committed' or not int(packet['snapshot']['committed_next'][v]) or
        result['plan'][str(v)][0]==int(packet['snapshot']['committed_next'][v])
        for v,status in enumerate(packet['snapshot']['vehicle_status'])))
    ck('certificate_hard',cert['qualified'] is True and cert['evaluation']['hard_feasible'] is True and not cert['evaluation']['failures'])
    ck('budget',cert['evaluation']['budget_violated'] is False and cert['evaluation']['energy_kwh']<=packet['budget'])
    ck('utility_components',abs(cert['evaluation']['utility']-(cert['evaluation']['revenue']-cert['evaluation']['fuel_cost']-cert['evaluation']['reject_loss']))<1e-10)
    ck('zero_reject_penalty',cert['evaluation']['reject_loss']==0 and result['same_objective_penalties_zero'])
    ck('input_provenance',result['input_unchanged'] and len(result['input_digest'])==64)
    ck('committed_vehicle_unchanged',int(packet['snapshot']['committed_next'][0])==59 and result['plan']['0'][0]==59 and result['committed_original_vehicle0_destination59_preserved'])
    ck('scope',result['first_derived_midcourse_public_packet_only'] and not result['actual_live_owner'] and not result['formal_strong_dynamic_qualified'] and not result['complete_request_budget_qualified'] and not result['scientific_data_generated'] and not result['stable_online_gain'])
    ck('elapsed_positive',0<=result['elapsed_seconds']<8)
    out = dict(checks=[dict(name=k,pass_=v) for k,v in checks.items()],checks_count=len(checks),failed=sum(not v for v in checks.values()),
        independent_admission_component_only=True,formal_strong_dynamic_qualified=False,stable_online_gain=False)
    with (HERE / 'INDEPENDENT_PICKUP_AUDIT.json').open('x',encoding='utf-8') as f: json.dump(out,f,indent=2)
    if out['failed']: raise SystemExit(1)

if __name__=='__main__': main()
