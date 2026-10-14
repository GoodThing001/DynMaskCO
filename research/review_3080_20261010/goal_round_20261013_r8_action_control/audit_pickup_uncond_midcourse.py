"""Saved-only independent audit; no admission or route-search rerun."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import qualify_pickup_joint_policy_transaction_root_v3 as old
import numpy as np

HERE=Path(__file__).resolve().parent
INPUT=old.P/'undecided_public_input_root_v2/RESULT.json'
PRIOR=HERE.parent/'goal_round_20261013_r7_midcourse_control/PICKUP_MIDCOURSE_RESULT.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    reg=json.loads((HERE/'PICKUP_AUDIT_REGISTRATION.json').read_text(encoding='utf-8'))
    result=json.loads((HERE/'PICKUP_UNCOND_MIDCOURSE_RESULT.json').read_text(encoding='utf-8'))
    previous=json.loads(PRIOR.read_text(encoding='utf-8'))
    _,decode=old.codecs()
    packet=decode(json.loads(INPUT.read_text(encoding='utf-8'))['records'][2]['packet'])
    cert=decode(result['certificate']);initial=packet['snapshot']['replanner_state']
    a=set(result['accepted']);r=set(result['rejected']);current=packet['admission_input']['current_order']
    checks={}
    def ck(name,test):checks[name]=bool(test)
    ck('input_identity',sha(INPUT)==reg['input_sha256']==result['input_sha256'])
    ck('history_identity',sha(old.HISTORY)==reg['history_sha256']==result['history_sha256'])
    ck('program_identity',sha(HERE/'pickup_uncond_midcourse.py')==reg['program_sha256'])
    ck('audit_identity',sha(__file__)==reg['audit_sha256'])
    ck('result_identity',sha(HERE/'PICKUP_UNCOND_MIDCOURSE_RESULT.json')==reg['result_sha256'])
    ck('prior_identity',sha(PRIOR)==reg['prior_independent_result_sha256']==result['prior_result_sha256'])
    ck('fixed_case',result['record_position']==2 and current==61 and result['order']==61 and result['clock']==packet['snapshot']['clock'])
    ck('accepted_delta',a-set(initial['accepted'])=={current})
    ck('rejected_delta',r-set(initial['rejected'])==set())
    ck('disjoint',not a.intersection(r))
    ck('pending',set(result['pending'])==set(packet['admission_input']['pending'])-{current})
    ck('route_coverage',set(o for route in result['plan'].values() for o in route)==a-set(np.flatnonzero(packet['snapshot']['served_mask'])))
    ck('commitment',result['plan']['0'][0]==int(packet['snapshot']['committed_next'][0])==59)
    ck('full_c0_partial',cert['qualified'] is True and cert['evaluation']['hard_feasible'] is True and not cert['evaluation']['failures'])
    ck('budget',cert['evaluation']['budget_violated'] is False and cert['evaluation']['energy_kwh']<=packet['budget'])
    ck('zero_penalty',cert['evaluation']['reject_loss']==0)
    ck('utility',abs(cert['evaluation']['utility']-(cert['evaluation']['revenue']-cert['evaluation']['fuel_cost']-cert['evaluation']['reject_loss']))<1e-10)
    ck('prior_comparison',result['decision_same']==(result['current_decision']==previous['current_decision'])
       and result['plan_same']==(result['plan']==previous['plan']))
    ck('scope',result['input_unchanged'] and not result['continuous_owner_qualified']
       and not result['formal_strong_dynamic_qualified'] and not result['complete_request_budget_qualified']
       and not result['scientific_data_generated'] and not result['stable_online_gain'])
    ck('time',0<=result['elapsed_seconds']<8)
    out=dict(checks_count=len(checks),failed=sum(not x for x in checks.values()),checks=checks,
        saved_only_independent=True,no_controller_or_solver_rerun=True,formal_strong_dynamic_qualified=False)
    with (HERE/'INDEPENDENT_PICKUP_UNCOND_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    if out['failed']:raise SystemExit(1)

if __name__=='__main__':main()
