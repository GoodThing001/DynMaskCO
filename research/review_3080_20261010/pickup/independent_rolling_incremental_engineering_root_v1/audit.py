"""Fresh original-false reference; not the candidate observer/controller."""
import os,sys,json,copy,pickle,time,hashlib,ctypes,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT/'tools'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import qualify_pickup_joint_policy_transaction_root_v3 as old
import numpy as np,psutil
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import restore_recourse_snapshot
from scenario_saa import UncondHistoricalSampler
from energy_complete_saa import EnergyCompleteSAA
from coldchain_evaluator_a1 import evaluate_trace_a1
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):
    with (HERE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw_states(states):
    result={}
    for v,s in states.items():
        result[str(v)]={'route':[int(x) for x in s['route']],'cur':int(s['cur']),'cur_time':float(s['cur_time']).hex(),'load':float(s['load']).hex(),'frozen':int(s.get('frozen',0))}
    return result
class OriginalReference(EnergyCompleteSAA):
    def _sim_scenario(self,*args,**kwargs):
        self.events=[];self.terminal=[];return super()._sim_scenario(*args,**kwargs)
    def _insert_u(self,space,env,o,state,deadline):
        initial=raw_states(state);accepted=super()._insert_u(space,env,o,state,deadline)
        self.events.append({'kind':'insert','order':int(o),'before':initial,'accepted':bool(accepted),'after':raw_states(state)});return accepted
    def _local_search_u(self,space,env,state,deadline,max_passes=2):
        initial=raw_states(state);result=super()._local_search_u(space,env,state,deadline,max_passes)
        self.events.append({'kind':'local_search','before':initial,'after':raw_states(state),'max_passes':max_passes});return result
    def _complete_u(self,space,env,inst,clock,state,wait_until):
        initial=raw_states(state);energy,distance=super()._complete_u(space,env,inst,clock,state,wait_until)
        self.terminal.append({'states':initial,'wait_until':wait_until,'energy_hex':None if energy is None else float(energy).hex(),'distance_hex':None if distance is None else float(distance).hex()});return energy,distance
def closure():
    paths={Path(sys.executable),Path(__file__)}
    for m in list(sys.modules.values()):
        name=getattr(m,'__file__',None)
        if name and Path(name).is_file():paths.add(Path(name).resolve())
    for m in psutil.Process().memory_maps():
        p=Path(m.path)
        if p.is_file():paths.add(p.resolve())
    return {str(p):sha(p) for p in sorted(paths)}
def main():
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL('SHCore.dll');psutil.Process().memory_maps();encode,decode=old.codecs();old.pf.digest({'prime':'actualdigest imports only'})
    reg=read(HERE/'REGISTRATION.json');receipt=read(HERE/'RUN_RECEIPT.json');context=decode(read(HERE/'SIMULATION_CONTEXT.json'));p=context['packet'];checks=[];results=[]
    log=(HERE/'AUDIT_CHECKS.jsonl').open('x',encoding='utf-8')
    def ck(name,value):
        c=dict(name=name,pass_=bool(value));checks.append(c);log.write(json.dumps(c)+'\n');log.flush()
    loaded_start=closure();save('AUDIT_SOURCE_START.json',loaded_start)
    for path,h in reg['sources'].items():ck('registered_source:'+path,sha(path)==h)
    for path,h in read(HERE/'PARENT_IDENTITIES.json')['prior_files'].items():ck('prior_frozen_file:'+path,sha(path)==h)
    ck('candidate_flag_only',reg['candidate_incr_eval'] and not reg['exact_reference_incr_eval']);ck('K10_seed_preserved',reg['K']==10 and reg['seed']==202610130901)
    ck('untrained_public_engineering',not reg['scientific_data_generated'] and not reg['formal_strong_dynamic_qualified'] and not reg['stable_online_gain'])
    ck('owned_worker_reaped',receipt['worker_reaped']);ck('outer_under10',receipt['elapsed']<10.)
    original_input_path=next(Path(path) for path in reg['sources'] if Path(path).parent.name=='undecided_public_input_root_v2')
    original=decode(read(original_input_path)['records'][reg['chosen']['position']]['packet']);ck('first_eligible_saved_input_exact',old.pf.digest(p)==old.pf.digest(original))
    current=p['admission_input']['current_order'];prior=p['snapshot']['replanner_state'];fleet,_,served=restore_recourse_snapshot(p['snapshot']);clock=p['snapshot']['clock']
    selected={decision:None if index is None else next(row for row in context['candidates'][decision] if row['index']==index) for decision,index in context['selected'].items()}
    # Actual chosen immediate plans: complete canonical evaluator from saved raw
    # event traces. The prior experimental wrappers are not reference evaluators.
    for decision,row in selected.items():
        if row is None:continue
        state=row['packet']['snapshot']['replanner_state'];actual=evaluate_trace_a1(row['raw'],{k:v[0] for k,v in p['dataset'].items()},p['contract'],set(state['accepted']),set(state['rejected']),p['budget'],p['penalties'])
        for key,val in actual.items():ck(decision+':chosen_full_C0:'+key,old.pf.digest(val)==old.pf.digest(row['certificate']['evaluation'][key]))
        ck(decision+':chosen_C0_hard',actual['hard_feasible'] and row['qualified'])
        ck(decision+':prior_admission_preserved',set(state['accepted'])-{current}==set(prior['accepted']) and set(state['rejected'])-{current}==set(prior['rejected']))
        ck(decision+':current_decision_partition',current in state['accepted'] if decision=='accept' else current in state['rejected'])
        actual_served={s.node for t in row['raw'] for s in t.services};ck(decision+':current_service_semantics',current in actual_served if decision=='accept' else current not in actual_served)
        ck(decision+':current_reject_cost_once',actual['reject_loss']==sum(p['penalties'][int(p['dataset']['temp_class'][0,j])] for j in state['rejected']))
        for v,status in enumerate(p['snapshot']['vehicle_status']):
            cn=int(p['snapshot']['committed_next'][v])
            if status=='committed' and cn>0:ck(decision+':frozen_commitment:'+str(v),row['plan'][str(v)][0]==cn)
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    actor=OriginalReference(budget=p['budget'],capacity=p['capacity'],booking_horizon=p['booking_horizon'],contract=p['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=UncondHistoricalSampler(history['history']),reject_penalty=p['penalties'],K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='ls',energy_pricing='marginal',future_policy='reveal',standby_orders=0,incr_eval=False)
    actor.restore_state(copy.deepcopy(prior));env=StrictOnlineEnv(p['dataset'],p['capacity'],p['tw_speed'],p['snapshot']['num_vehicles'],actor,p['contract'],p['booking_horizon']);actor._sync_id_map(env,0,clock);observed=actor._build_snapshot(env,0,clock,fleet,served)
    scenarios=actor.sampler.sample(observed,np.random.default_rng(reg['seed']),10);ck('original_unconditional_common_K10',old.pf.digest(scenarios)==context['scenario_sha256']==old.pf.digest(context['scenarios']))
    space,orders=actor._build_space(env,0,scenarios);index={id(x):space.n_real+j for j,x in enumerate(orders)};indices=[[index[id(x)] for x in s] for s in scenarios];ck('all_common_scenario_ids',indices==context['indices'])
    for decision in ('reject','accept'):
        row=selected[decision];paths=sorted(HERE.glob('ROLLOUT_%s_*.json'%decision));ck(decision+':prefix_no_holes',[int(path.stem.rsplit('_',1)[1]) for path in paths]==list(range(len(paths))))
        if row is None:ck(decision+':no_feasible_no_rollout',not paths);continue
        actor.restore_state(copy.deepcopy(row['packet']['snapshot']['replanner_state']));states=actor._sim_states(env,0,clock,fleet,served,{int(v):route for v,route in row['plan'].items()})
        for path in paths:
            saved=read(path);j=saved['index'];start=time.perf_counter();value=float(actor._sim_scenario(env,0,clock,states,indices[j],space));end=time.perf_counter()
            if decision=='reject':value-=p['penalties'][int(p['dataset']['temp_class'][0,current])]
            ck(decision+':branch_identity:'+str(j),saved['decision']==decision and saved['incr_eval'] is True)
            score_exact=value.hex()==saved['corrected_utility_hex'];events_exact=actor.events==saved['events'];terminal_exact=actor.terminal==saved['terminal']
            ck(decision+':complete_original_utility_hex:'+str(j),score_exact);ck(decision+':all_action_trajectories_exact:'+str(j),events_exact);ck(decision+':complete_original_C0_terminal_exact:'+str(j),terminal_exact)
            if not events_exact:
                for i in range(max(len(actor.events),len(saved['events']))):
                    a=actor.events[i] if i<len(actor.events) else None;b=saved['events'][i] if i<len(saved['events']) else None
                    if a!=b:
                        save('FIRST_ACTION_DIFFERENCE_%s_%02d.json'%(decision,j),dict(event_index=i,reference=a,candidate=b));break
            witness=dict(decision=decision,index=j,reference_value_hex=value.hex(),candidate_value_hex=saved['corrected_utility_hex'],reference_events=actor.events,reference_terminal=actor.terminal,reference_elapsed=end-start,candidate_elapsed=saved['elapsed'],score_exact=score_exact,events_exact=events_exact,terminal_exact=terminal_exact)
            save('REFERENCE_%s_%02d.json'%(decision,j),witness);results.append({k:v for k,v in witness.items() if k not in ('reference_events','reference_terminal')})
    loaded_end=closure();save('AUDIT_SOURCE_END.json',loaded_end);ck('audit_actual_loaded_source_closure',loaded_start==loaded_end)
    complete_counts={d:sum(x['decision']==d for x in results) for d in ('reject','accept')};completeK10=complete_counts=={'reject':10,'accept':10};score_bad=[x['name'] for x in checks if not x['pass_'] and any(k in x['name'] for k in ('utility_hex','action_trajectories','C0_terminal'))];source_bad=[x['name'] for x in checks if not x['pass_'] and 'source' in x['name']]
    out=dict(checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),failure_names=[x['name'] for x in checks if not x['pass_']],strict_semantics_failure_names=score_bad,registered_source_failure_names=source_bad,complete_counts=complete_counts,complete_K10_both_decisions=completeK10,reference_results=results,strict_prefix_semantics_qualified=not score_bad,candidate_eligible_for_install=completeK10 and not any(not c['pass_'] for c in checks) and receipt['qualified'],full_request_budget_qualified=False,formal_strong_dynamic_qualified=False,full_scientific_loaded_closure=False,stable_online_gain=False)
    log.close();save('INDEPENDENT_SEMANTICS_AUDIT.json',out);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(out['failed']>0)));print(json.dumps(out));return int(out['failed']>0)
if __name__=='__main__':raise SystemExit(main())
