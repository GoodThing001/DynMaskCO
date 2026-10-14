"""Independent prepared-state physical replay and direct public fields, no SAA."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_private_public_transport_v2 as fixture
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
ROOT=old.ROOT;P=old.P;OUT=P/'endogenous_owner_transaction_root_v2';SOURCE=P/'observed_day_extension_root_v1';PREFIX=P/'live_event_prefix_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    psutil.Process().cpu_affinity([7]);encode,decode=old.codecs();reg=read(OUT/'REGISTRATION.json');raw=read(OUT/'RESULT.json');actual=decode(read(OUT/'OWNER_RESULT_0.json'));checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    with (SOURCE/'REQUESTS_STARTED.jsonl').open(encoding='utf-8') as f:request=decode(json.loads(next(f)))
    with (SOURCE/'PREPARED_RESPONSES.jsonl').open(encoding='utf-8') as f:sealed_response=decode(json.loads(next(f)))['response']
    response=actual['response']
    with fixture.TRACE.open(encoding='utf-8') as f:dataset=decode(json.loads(next(f))['payload'])['dataset']
    initial=decode(read(PREFIX/'RESULT.json'))['final'];packet=request['packet'];reference={}
    class Stop(Exception):pass
    class Policy(C0ReserveReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):reference.update(clock=clock,visible=visible);raise Stop()
    rp=Policy(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.);rp.restore_state(copy.deepcopy(initial['state']))
    env=StrictOnlineEnv(copy.deepcopy(dataset),packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon']);fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy();clock=float(initial['clock']);env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];customers=[j for j in range(1,env.num_nodes) if env.demands[0,j]>0];reveals=sorted(set(float(env.reveal_time[0,j]) for j in customers if env.reveal_time[0,j]>0))
    try:env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,reveals,float(env.tw_end[0].max()))
    except Stop:pass
    else:raise AssertionError('registered reveal absent')
    clock=reference['clock'];private_clock=clock;private_event=env.event_id;private_replan=env.replan_count;physical_reference=copy.deepcopy((fleet,traces,served))
    def snapshot():return dict(state=rp.export_state(),fleet=fleet,traces=traces,served=served,clock=clock,event_id=env.event_id,replan_count=env.replan_count)
    baseline=old.pf.digest(snapshot());ck('independent_actual_private_before_identity',baseline==actual['baseline'] and clock==packet['snapshot']['clock'] and env.event_id==packet['snapshot']['event_id'])
    known=reference['visible'];projected=actual['public_packet'];ck('original_sealed_public_digest',old.pf.digest(projected)==old.pf.digest(packet)==actual['public_digest'])
    for key,value in projected['dataset'].items():
        expected=np.zeros_like(value);expected[0,0]=dataset[key][0,0]
        for o in known:expected[0,o]=dataset[key][0,o]
        ck('direct_observed_field:'+key,np.array_equal(value,expected))
    ck('actual_pending_not_placeholder',request['order'] in projected['admission_input']['pending'] and request['order'] not in set(rp._accepted)|set(rp._rejected))
    
    from run_exp_rule_control import UncondDynamicReplanner
    from energy_complete_saa import EnergyCompleteSAA
    from scenario_saa import CondHistoricalSampler
    from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
    import pickle
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    prior=packet['snapshot']['replanner_state'];o=request['order'];name='endogenous';a=set(prior['accepted']);r=set(prior['rejected']);pending=set(packet['admission_input']['pending']);current=response['scoring_packet']
    compiled=response['compiled_state'];state=dict(version='c0-reserve-v1',budget=compiled.budget,mode=compiled.mode,accepted=sorted(compiled.accepted),rejected=sorted(compiled.rejected),plan={str(k):list(v) for k,v in compiled.plan.items()},status_inst=compiled.status_inst)
    independent_compiled=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();ck('original_compiled_sha',independent_compiled==response['compiled_state_sha256']==actual['compiled_sha256'])
    rp=UncondDynamicReplanner(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],contract=packet['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10)
    rp.restore_state(copy.deepcopy(prior));env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
    env.replan_count=packet['snapshot']['replan_count'];fleet,traces,served=restore_recourse_snapshot(packet['snapshot'])
    rp.on_reveal(env,0,packet['snapshot']['clock'],fleet,served,[o]);replay=rp.export_state()
    plans={str(v):[int(j) for j in replay['plan'].get(v,replay['plan'].get(str(v),[])) if not served[j]] for v in range(len(fleet))}
    for v in fleet:
        if v.status=='committed' and v.committed_next not in (None,0):plans[str(v.vehicle_id)]=[int(v.committed_next)]+[j for j in plans[str(v.vehicle_id)] if j!=v.committed_next]
    replay['plan']=plans
    ck(name+':actual_original_admission',old.pf.digest(replay)==old.pf.digest(response['base_response']['state']))
    ck(name+':only_current_transition',set(replay['accepted'])-{o}==a and set(replay['rejected'])-{o}==r)
    remaining=pending-{o};ck(name+':explicit_remaining_partition',set(current['admission_input']['pending'])==remaining and not remaining&(set(replay['accepted'])|set(replay['rejected'])))
    ck(name+':physical_snapshot_unchanged',all(old.pf.digest(packet['snapshot'][k])==old.pf.digest(current['snapshot'][k]) for k in packet['snapshot'] if k not in ('state_hash','replanner_state')))
    ck(name+':state_hash',snapshot_state_hash(current['snapshot'])==current['snapshot']['state_hash'])
    for k in packet['dataset']:ck(name+':retained_observed_features:'+k,np.array_equal(packet['dataset'][k],current['dataset'][k]))
    # Original full physical closing run; only full-day coverage refuses pending.
    ev,tr=old.pf.evaluate_known(current,plans);stored=response['selected_certificate']['evaluation']
    ck(name+':full_original_C0_closing_trace',old.pf.digest(tr)==old.pf.digest(response['selected_raw']))
    ck(name+':original_coverage_boundary',ev['failures']==(['unaccounted_order'] if remaining else []))
    for k,v in ev.items():
        if k not in ('hard_feasible','failures'):ck(name+':full_trace_field:'+k,old.pf.digest(v)==old.pf.digest(stored[k]))
    ck(name+':provisional_hard_with_explicit_pending',stored['hard_feasible'] and stored['failures']==[] and stored['pending_count']==len(remaining) and stored['provisional_not_full_day_utility'])
    ck(name+':pending_no_physical_service',not remaining&{s.node for t in tr for s in t.services})
    expected_loss=sum(packet['penalties'][int(packet['dataset']['temp_class'][0,j])] for j in replay['rejected'])
    ck(name+':no_pending_rejection_loss',stored['reject_loss']==expected_loss)
    # Independently assemble original SAA class, sampler and all 20 branches.
    srp=EnergyCompleteSAA(budget=current['budget'],capacity=current['capacity'],booking_horizon=current['booking_horizon'],contract=current['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=CondHistoricalSampler(history['history'],n_neighbors=10),reject_penalty={0:5.,1:10.,2:15.},K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='greedy',energy_pricing='marginal',future_policy='density',standby_orders=0)
    srp.restore_state(copy.deepcopy(replay));senv=StrictOnlineEnv(current['dataset'],current['capacity'],current['tw_speed'],current['snapshot']['num_vehicles'],srp,current['contract'],current['booking_horizon'])
    sfleet,_,sserved=restore_recourse_snapshot(current['snapshot']);clock=current['snapshot']['clock'];srp._sync_id_map(senv,0,clock)
    ssnap=srp._build_snapshot(senv,0,clock,sfleet,sserved);scenarios=srp.sampler.sample(ssnap,np.random.default_rng(reg['common_seed']),10)
    saved=response['common_scores'];ck(name+':actual_observed_sampler_prefix',old.pf.digest(ssnap)==old.pf.digest(saved['observed_snapshot']))
    ck(name+':all_pending_remains_observed',{srp._id_map[j] for j in remaining}<=set(x.oid for x in ssnap.orders) and not {srp._id_map[j] for j in remaining}&(set(ssnap.accepted)|set(ssnap.rejected)))
    ck(name+':common_fixed_scenario_identity',old.pf.digest(scenarios)==saved['scenario_sha256'] and old.pf.digest(scenarios)==old.pf.digest(saved['scenarios']))
    space,orders=srp._build_space(senv,0,scenarios);ids={id(v):space.n_real+j for j,v in enumerate(orders)}
    for label,plan in {'own_incumbent':plans,'candidate':response['proposal']}.items():
        st=srp._sim_states(senv,0,clock,sfleet,sserved,{int(k):v for k,v in plan.items()})
        vals=[float(srp._sim_scenario(senv,0,clock,st,[ids[id(j)] for j in scen],space)) for scen in scenarios]
        c=saved['candidate_scores'][label]
        for j,(v,w) in enumerate(zip(vals,c['branch_values'])):ck(name+':'+label+':branch:'+str(j),v.hex()==w.hex())
        ck(name+':'+label+':all10mean_grouped',len(vals)==len(c['branch_values'])==10 and float(np.mean(vals)).hex()==c['mean_shadow_utility'].hex())
    ck(name+':no_route_gain_or_model',response['proposal']==plans and not response['accepted_route'] and not response['formal_learned_controls'])
    ck('independent_compiled_matches_original_admission',state==replay)
    # Restore the original private physical reference, because admission replay used public copies.
    rp=Policy(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.);rp.restore_state(copy.deepcopy(state))
    env=StrictOnlineEnv(copy.deepcopy(dataset),packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon']);fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy()
    fleet,traces,served=copy.deepcopy(physical_reference);env.event_id=private_event;env.replan_count=private_replan;clock=private_clock
    env._plan_and_commit(0,clock,fleet,traces,served,customers)
    for k,v in snapshot().items():ck('original_actual_plan_commit:'+k,old.pf.digest(v)==old.pf.digest(actual['after_commit'][k]))
    next_clock=float(env.tw_end[0].max());remaining=[t for t in reveals if t>clock+1e-6]
    if remaining:next_clock=min(next_clock,remaining[0])
    if env.booking_horizon>clock+1e-6:next_clock=min(next_clock,env.booking_horizon)
    for v in fleet:
        if v.status=='committed' and v.committed_finish is not None:next_clock=min(next_clock,v.committed_finish)
        elif v.status=='returning' and v.return_finish is not None:next_clock=min(next_clock,v.return_finish)
    before=old.pf.digest(snapshot());clock=next_clock;env.event_id+=1;env._advance_fleet(0,clock,fleet,traces,served)
    ck('original_natural_physical_clock',clock==actual['after_primary_advance']['clock'] and before==actual['primary_advance_before'])
    for k,v in snapshot().items():ck('original_primary_physical_execution:'+k,old.pf.digest(v)==old.pf.digest(actual['after_primary_advance'][k]))
    ck('actual_primary_physical_digest',old.pf.digest(snapshot())==actual['primary_advance_digest'])
    ck('prepared_only_no_new_strategy',not reg['prepared_response_reused'] and reg['new_admission_or_SAA'] and not reg['new_scientific_data'] and actual['new_admission_or_SAA'])
    for p,h in reg['sources'].items():ck('registered:'+p,sha(p)==h)
    names={'qualify_pickup_endogenous_owner_transaction_v2.py'}
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(arg).name in names for arg in p.info['cmdline'] or [])]
    ck('actual_owned_processes_gone',not active)
    for r in raw['records']:
        n=str(r['case']);ck(n+':independent_old_owner_unchanged',r['baseline']==r['old_owner_after']==baseline);ck(n+':actual_timing_and_reap',r['finished']-r['started']==r['wall_s'] and r['wall_s']<r['cap_s'] and r['reaped']);ck(n+':predeclared_budget',r['cap_s']==10. and r['reserve_s']==2.);ck(n+':fixed_host_work',not r['timed_parent_public_projection_or_pickle']);begin=read(OUT/('SOURCE_START_'+n+'.json'));ck(n+':actual_loaded_pid',begin['pid']==r['pid']==r['ready']['pid'])
        for p,h in begin['files'].items():ck(n+':loaded:'+p,sha(p)==h)
        if r['fault']:ck(n+':cancel_actual_stage_without_owner_promotion',r['stage_reached']==3 and r['error']=='advance_cancelled' and not r['actual_process_owner_selected'] and r['prepared'] is not None and r['promotion'] is not None and not (OUT/('OWNER_RESULT_'+n+'.json')).exists())
        else:
            end=read(OUT/'SOURCE_END_0.json');ck(n+':normal_actual_owner_and_progress',r['actual_process_owner_selected'] and r['primary_advance']==actual['primary_advance_digest'] and r['prepared']==independent_compiled and r['started']<=r['promotion']<r['finished']);ck(n+':normal_loaded_file_closure',begin['files']==end['files'] and begin['module_files']==end['module_files'] and actual['loaded_closure_qualified'])
    ck('full_request_scientific_qualification_remains_false',not raw['full_live_request_qualified'] and not raw['full_loaded_dependency_closure_qualified'] and not raw['stable_online_gain'])
    failed=[c['name'] for c in checks if not c['pass_']];result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,registration_sha256=sha(OUT/'REGISTRATION.json'),producer_sha256=sha(OUT/'RESULT.json'),independent_public_fields_and_original_physical_replay=True,new_SAA_or_admission_reexecution=True,full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_PRIVATE_OWNER.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed)));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
