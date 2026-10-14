"""Undecided input to provisional C0, shared K10 and a compiled joint state."""
import copy,time
import numpy as np
from pickup_partial_validation_v1 import validate_partial
from pickup_partial_known_v1 import evaluate_partial_known
import public_fleet as pf

def certificate(packet,plan):
    validate_partial(packet)
    snap=packet['snapshot'];outstanding=set(snap['replanner_state']['accepted'])-set(np.flatnonzero(snap['served_mask']))
    flat=[o for route in plan.values() for o in route]
    if any(type(o) is not int for o in flat) or len(flat)!=len(set(flat)) or set(flat)!=outstanding:
        raise ValueError('accepted service commitment violated')
    if set(plan)!=set(packet['plan']):raise ValueError('fleet changed')
    for vid,status in enumerate(snap['vehicle_status']):
        cn=int(snap['committed_next'][vid])
        if status=='committed' and cn>0 and (not plan[str(vid)] or plan[str(vid)][0]!=cn):
            raise ValueError('immutable committed owner/destination violated')
        if status in ('returning','closed') and plan[str(vid)]!=packet['plan'][str(vid)]:
            raise ValueError('unavailable vehicle reassigned')
    ev,tr=evaluate_partial_known(packet,plan)
    return dict(qualified=ev['hard_feasible'] is True,evaluation=ev,all_fleet_energy_accounted=True,
                pending_without_rejection_penalty=True,provisional_not_full_day=True),tr

def decide(packet,history,scripts,deadline):
    from pickup_undecided_public_v1 import validate_undecided
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
    import hard_admission as hd
    validate_undecided(packet);o=packet['admission_input']['current_order'];prior=packet['snapshot']['replanner_state']
    config=dict(name='own_uncond_dynamic',budget=packet['budget'],capacity=packet['capacity'],
                booking_horizon=packet['booking_horizon'],contract=packet['contract'],cooling_share=0.,history=history['history_times_classes'])
    rp=hd.base(config,scripts);rp.restore_state(copy.deepcopy(prior))
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
    env.replan_count=packet['snapshot']['replan_count']
    fleet,traces,served=restore_recourse_snapshot(packet['snapshot'])
    if time.perf_counter()>=deadline:raise TimeoutError('admission start')
    rp.on_reveal(env,0,packet['snapshot']['clock'],fleet,served,[o])
    state=rp.export_state();a=set(state['accepted']);r=set(state['rejected'])
    if a-{o}!=set(prior['accepted']) or r-{o}!=set(prior['rejected']) or o not in a|r or a&r:
        raise ValueError('current-only admission transition required')
    plan=hd.canonical(packet,state['plan']);state['plan']=plan
    current=copy.deepcopy(packet);current['schema']='pickup-public-partial-admissions-v1'
    current['admission_input'].update(pending=sorted(set(packet['admission_input']['pending'])-{o}),current_decision='accepted' if o in a else 'rejected')
    current['plan']=plan;current['ownership']={str(o):int(v) for v,route in plan.items() for o in route}
    current['snapshot']['replanner_state']=state;current['snapshot']['state_hash']=snapshot_state_hash(current['snapshot'])
    validate_partial(current);cert,raw=certificate(current,plan)
    return dict(qualified=cert['qualified'] and time.perf_counter()<deadline,state=state,plan=plan,
                certificate=cert,known_raw=raw,current_packet=current)

def compute(packet,history,scripts,deadline):
    from pickup_partial_common_v1 import score_partial_common
    from joint_install import compile_state,digest,state_dict
    t0=time.perf_counter();base=decide(packet,history,scripts,deadline)
    if not base['qualified']:raise ValueError('admitted incumbent physical failure')
    current=base['current_packet'];proposal=copy.deepcopy(current['plan'])
    scores=score_partial_common(current,{'own_incumbent':current['plan'],'candidate':proposal},history,deadline,seed=202610111301)
    own=scores['candidate_scores']['own_incumbent'];cand=scores['candidate_scores']['candidate']
    choose=cand['qualified'] and cand['mean_shadow_utility']>own['mean_shadow_utility']+1e-10
    selected=proposal if choose else current['plan'];cert,raw=certificate(current,selected)
    state=copy.deepcopy(base['state']);state['plan']=selected;compiled=compile_state(state)
    finished=time.perf_counter()
    return dict(kind='response',request_id=str(packet['admission_input']['current_order']),packet_digest=pf.digest(packet),deadline=deadline,
                certified=cert['qualified'] and finished<deadline,compiled_state=compiled,compiled_state_sha256=digest(state_dict(compiled)),
                base_response=base,scoring_packet=current,proposal=proposal,common_scores=scores,selected_certificate=cert,
                selected_raw=raw,accepted_route=choose,worker_finished=finished,worker_started=t0,
                actual_live_owner=False,formal_learned_controls=False,full_live_10s_qualified=False)
