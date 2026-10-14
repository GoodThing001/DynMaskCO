"""Prospective engineering controller: owns current accept/reject and recourse.

Scenario-based planning is an inspiration, not an MSA reproduction. Physical
simulation remains the existing canonical C0 implementation. No learned arm.
"""
import copy, math, time
import numpy as np
from phase_trace import emit

def neighborhood(plan, packet):
    """All single relocations/swaps excluding immutable committed destinations."""
    snap=packet['snapshot']; frozen={str(v):int(snap['committed_next'][v])
        for v,s in enumerate(snap['vehicle_status']) if s=='committed' and int(snap['committed_next'][v])>0}
    active=[str(v) for v,s in enumerate(snap['vehicle_status']) if s in ('idle','ready','committed')]
    starts={v:1 if v in frozen else 0 for v in active}
    items=[(v,i,o) for v in active for i,o in enumerate(plan[v]) if i>=starts[v]]
    seen=set()
    for v,i,o in items:
        for w in active:
            for j in range(starts[w],len(plan[w])+1):
                if v==w and j in (i,i+1): continue
                q=copy.deepcopy(plan);q[v].pop(i);pos=j-(v==w and j>i);q[w].insert(pos,o)
                key=tuple(tuple(q[x]) for x in sorted(q))
                if key not in seen:seen.add(key);yield q
    for a,(v,i,o) in enumerate(items):
        for w,j,p in items[a+1:]:
            q=copy.deepcopy(plan);q[v][i]=p;q[w][j]=o
            key=tuple(tuple(q[x]) for x in sorted(q))
            if key not in seen:seen.add(key);yield q

def distance_recourse(plan,packet,rp,env,fleet,served,deadline):
    """Four best-improvement passes. Fixed order; no duration-chosen passes."""
    clock=packet['snapshot']['clock'];q=copy.deepcopy(plan);log=[]
    def route_states(p):return rp._start_states(env,0,fleet,served,clock,{int(v):r for v,r in p.items()})
    def valid(p):
        st=route_states(p)
        return all(rp._route_feasible(env,0,s['cur'],s['cur_time'],s['load'],s['route']) for s in st.values())
    for iteration in range(4):
        emit('recourse_pass',iteration)
        st=route_states(q);best_distance=rp._total_distance(env,0,st);best=q;trials=[]
        for p in neighborhood(q,packet):
            if time.perf_counter()>=deadline:raise TimeoutError('route recourse')
            ok=valid(p);d=rp._total_distance(env,0,route_states(p)) if ok else None
            trials.append(dict(plan=p,local_feasible=ok,distance=d))
            if ok and d<best_distance-1e-9:best=p;best_distance=d
        log.append(dict(iteration=iteration,before=q,trials=trials,after=best))
        if best==q:break
        q=best
    return q,log

def branch_packet(packet,decision,plan):
    from recourse_snapshot import snapshot_state_hash
    p=copy.deepcopy(packet);o=p['admission_input']['current_order'];st=p['snapshot']['replanner_state']
    st['accepted']=sorted(set(st['accepted'])|({o} if decision=='accept' else set()))
    st['rejected']=sorted(set(st['rejected'])|({o} if decision=='reject' else set()))
    st['plan']=copy.deepcopy(plan);p['schema']='pickup-public-partial-admissions-v1'
    p['admission_input']['pending']=sorted(set(p['admission_input']['pending'])-{o})
    p['admission_input']['current_decision']='accepted' if decision=='accept' else 'rejected'
    p['plan']=copy.deepcopy(plan);p['ownership']={str(o):int(v) for v,r in plan.items() for o in r}
    p['snapshot']['state_hash']=snapshot_state_hash(p['snapshot']);return p

def compute(packet,history,deadline):
    from pickup_undecided_public_v1 import validate_undecided
    from pickup_pending_transaction_v1 import certificate
    from rolling_optimizer import RollingOptimizeReplanner
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot
    from energy_complete_saa import EnergyCompleteSAA
    from scenario_saa import UncondHistoricalSampler
    import public_fleet as pf
    emit('compute_enter')
    validate_undecided(packet);original_digest=pf.digest(packet)
    rp=RollingOptimizeReplanner(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.,packet['penalties'],10.)
    rp.restore_state(copy.deepcopy(packet['snapshot']['replanner_state']))
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
    fleet,traces,served=restore_recourse_snapshot(packet['snapshot']);clock=packet['snapshot']['clock'];o=packet['admission_input']['current_order']
    generated={'reject':[copy.deepcopy(packet['plan'])],'accept':[]}
    for v,s in enumerate(packet['snapshot']['vehicle_status']):
        if s not in ('idle','ready','committed'):continue
        start=int(s=='committed' and int(packet['snapshot']['committed_next'][v])>0)
        for j in range(start,len(packet['plan'][str(v)])+1):
            p=copy.deepcopy(packet['plan']);p[str(v)].insert(j,o);generated['accept'].append(p)
    emit('generated',value0=len(generated['reject']),value1=len(generated['accept']))
    candidates={};selected={}
    for decision in ('reject','accept'):
        candidates[decision]=[];best=None
        for index,plan in enumerate(generated[decision]):
            emit('candidate_begin',index,decision)
            if time.perf_counter()>=deadline:raise TimeoutError('physical branch generation')
            p=branch_packet(packet,decision,plan)
            states=rp._start_states(env,0,fleet,served,clock,{int(v):r for v,r in plan.items()})
            if not all(rp._route_feasible(env,0,s['cur'],s['cur_time'],s['load'],s['route']) for s in states.values()):
                candidates[decision].append(dict(index=index,input_plan=plan,local_feasible=False,qualified=False));continue
            emit('recourse_begin',index,decision)
            final,recourse=distance_recourse(plan,packet,rp,env,fleet,served,deadline);p=branch_packet(packet,decision,final)
            emit('recourse_end',index,decision)
            emit('C0_begin',index,decision)
            cert,raw=certificate(p,final)
            emit('C0_end',index,decision,value0=int(cert['qualified']))
            row=dict(index=index,input_plan=plan,local_feasible=True,recourse=recourse,plan=final,packet=p,certificate=cert,raw=raw,qualified=cert['qualified'])
            candidates[decision].append(row)
            if cert['qualified'] and (best is None or cert['evaluation']['utility']>best['certificate']['evaluation']['utility']):best=row
        selected[decision]=best
        emit('decision_plan',decision=decision,value0=int(best is not None))
    if selected['reject'] is None:raise ValueError('no fully certified reject fallback')
    # Draw unconditional common historical scenarios exactly once, before either
    # decision enters shadow simulation. Current-day private future is absent.
    shadow=EnergyCompleteSAA(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],contract=packet['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=UncondHistoricalSampler(history['history']),reject_penalty=packet['penalties'],K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='ls',energy_pricing='marginal',future_policy='reveal',standby_orders=0)
    shadow.restore_state(copy.deepcopy(packet['snapshot']['replanner_state']));env.replanner=shadow
    shadow._sync_id_map(env,0,clock);observed=shadow._build_snapshot(env,0,clock,fleet,served)
    emit('scenario_draw')
    scenarios=shadow.sampler.sample(observed,np.random.default_rng(202610130901),10);scenario_sha=pf.digest(scenarios)
    emit('scenario_drawn',value0=len(scenarios))
    space,orders=shadow._build_space(env,0,scenarios);lookup={id(x):space.n_real+i for i,x in enumerate(orders)};indices=[[lookup[id(x)] for x in s] for s in scenarios]
    scores={}
    for decision in ('reject','accept'):
        emit('score_branch',decision=decision)
        row=selected[decision]
        if row is None:scores[decision]=dict(qualified=False,reason='no physically certified acceptance');continue
        shadow.restore_state(copy.deepcopy(row['packet']['snapshot']['replanner_state']))
        st=shadow._sim_states(env,0,clock,fleet,served,{int(v):r for v,r in row['plan'].items()});values=[];ledger=copy.deepcopy(shadow.unavailable_ledger)
        for scenario_index,ids in enumerate(indices):
            emit('SAA_begin',scenario_index,decision)
            if time.perf_counter()>=deadline:raise TimeoutError('complete branch comparison required')
            value=float(shadow._sim_scenario(env,0,clock,st,ids,space))
            if decision=='reject':value-=packet['penalties'][int(packet['dataset']['temp_class'][0,o])]
            if not math.isfinite(value):raise ValueError('nonfinite complete branch')
            values.append(value)
            emit('SAA_end',scenario_index,decision)
        scores[decision]=dict(qualified=True,branch_values=values,mean_shadow_utility=float(np.mean(values)),scenario_sha256=scenario_sha,unavailable_ledger=ledger,branch_admission_state=shadow.export_state(),current_reject_charge=packet['penalties'][int(packet['dataset']['temp_class'][0,o])] if decision=='reject' else 0.)
    choice='accept' if scores['accept']['qualified'] and scores['accept']['mean_shadow_utility']>scores['reject']['mean_shadow_utility'] else 'reject'
    if time.perf_counter()>=deadline:raise TimeoutError('late decision not publishable')
    emit('compute_done')
    return dict(candidates=candidates,selected={k:None if v is None else v['index'] for k,v in selected.items()},scores=scores,scenarios=scenarios,scenario_sha256=scenario_sha,scenario_indices=indices,observed_snapshot=observed,choice=choice,output_packet=selected[choice]['packet'],input_unchanged=pf.digest(packet)==original_digest,independent_admission_owner=True,own_uncond_dynamic_called=False,actual_live_owner=False,formal_strong_dynamic_qualified=False,full_request_budget_qualified=False,learned_control=False,stable_online_gain=False)
