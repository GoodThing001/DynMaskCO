"""One current-only original unconditioned admission call on a fixed old public state."""
import copy
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import qualify_pickup_joint_policy_transaction_root_v3 as old
from run_exp_rule_control import UncondDynamicReplanner
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
from pickup_undecided_public_v1 import validate_undecided
from pickup_partial_validation_v1 import validate_partial
from pickup_pending_transaction_v1 import certificate
import hard_admission as hd

INPUT=old.P/'undecided_public_input_root_v2/RESULT.json'
HISTORY=old.HISTORY
PRIOR=HERE.parent/'goal_round_20261013_r7_midcourse_control/PICKUP_MIDCOURSE_RESULT.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    reg=json.loads((HERE/'PICKUP_EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    assert sha(INPUT)==reg['input_sha256'] and sha(HISTORY)==reg['history_sha256']
    assert sha(PRIOR)==reg['prior_independent_result_sha256']
    assert sha(__file__)==reg['program_sha256']
    _,decode=old.codecs()
    row=json.loads(INPUT.read_text(encoding='utf-8'))['records'][2]
    assert row['kind']=='derived_old_state' and row['index']==9 and row['variant']==0
    packet=decode(row['packet']);validate_undecided(packet)
    old_digest=old.pf.digest(packet)
    order=packet['admission_input']['current_order'];snap=packet['snapshot']
    assert order==61 and int(snap['committed_next'][0])==59
    with HISTORY.open('rb') as f: hist=pickle.load(f)['history_times_classes']
    rp=UncondDynamicReplanner(packet['budget'],packet['capacity'],packet['booking_horizon'],
        packet['contract'],0.,hist,k=10)
    rp.restore_state(copy.deepcopy(snap['replanner_state']))
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],
        snap['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
    env.replan_count=snap['replan_count']
    fleet,traces,served=restore_recourse_snapshot(snap)
    before=copy.deepcopy(rp.export_state())
    t=time.perf_counter();rp.on_reveal(env,0,snap['clock'],fleet,served,[order]);elapsed=time.perf_counter()-t
    state=rp.export_state(); accepted=set(state['accepted']);rejected=set(state['rejected'])
    assert accepted-{order}==set(before['accepted']) and rejected-{order}==set(before['rejected'])
    assert (order in accepted)!=(order in rejected)
    plan=hd.canonical(packet,state['plan']);state['plan']=plan
    current=copy.deepcopy(packet);current['schema']='pickup-public-partial-admissions-v1'
    current['admission_input'].update(pending=sorted(set(packet['admission_input']['pending'])-{order}),
        current_decision='accepted' if order in accepted else 'rejected')
    current['plan']=plan;current['ownership']={str(o):int(v) for v,route in plan.items() for o in route}
    current['snapshot']['replanner_state']=state
    current['snapshot']['state_hash']=snapshot_state_hash(current['snapshot'])
    current['penalties']={0:0.,1:0.,2:0.}
    validate_partial(current)
    cert,raw=certificate(current,plan)
    prior=json.loads(PRIOR.read_text(encoding='utf-8'))
    assert old.pf.digest(packet)==old_digest==prior['input_digest']
    encode,_=old.codecs()
    output=dict(input_sha256=sha(INPUT),history_sha256=sha(HISTORY),prior_result_sha256=sha(PRIOR),
        record_position=2,order=order,clock=float(snap['clock']),
        elapsed_seconds=elapsed,current_decision=current['admission_input']['current_decision'],
        accepted=sorted(accepted),rejected=sorted(rejected),pending=current['admission_input']['pending'],
        plan=plan,certificate=encode(cert),raw_digest=old.pf.digest(raw),
        prior_independent_decision=prior['current_decision'],prior_independent_plan=prior['plan'],
        decision_same=prior['current_decision']==current['admission_input']['current_decision'],
        plan_same=prior['plan']==plan,input_unchanged=True,
        admission_class='UncondDynamicReplanner',history_count=len(hist),
        continuous_owner_qualified=False,formal_strong_dynamic_qualified=False,
        complete_request_budget_qualified=False,scientific_data_generated=False,stable_online_gain=False)
    with (HERE/'PICKUP_UNCOND_MIDCOURSE_RESULT.json').open('x',encoding='utf-8') as f:
        json.dump(output,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
