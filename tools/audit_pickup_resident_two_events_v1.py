"""Independent original admission/K10/C0 and actual two-event private progression."""
import copy,json,pickle,hashlib
from pathlib import Path
import numpy as np,psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_private_public_transport_v2 as fixture
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
from run_exp_rule_control import UncondDynamicReplanner
from energy_complete_saa import EnergyCompleteSAA
from scenario_saa import CondHistoricalSampler
from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
from joint_install import state_dict,digest
ROOT=old.ROOT;P=old.P;OUT=P/'resident_two_events_root_v1';SOURCE=P/'endogenous_owner_transaction_root_v2/OWNER_RESULT_0.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();raw=read(OUT/'RESULT.json');reg=read(OUT/'REGISTRATION.json');actual=decode(read(OUT/'RESIDENT_RESULT.json'));initial=decode(read(SOURCE))['after_primary_advance'];checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    with fixture.TRACE.open(encoding='utf-8') as f:dataset=decode(json.loads(next(f))['payload'])['dataset']
    first=actual['records'][0]['packet'];boundary={};reference_events=[]
    class Stop(Exception):pass
    class Policy(C0ReserveReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):boundary.update(clock=clock,visible=visible);raise Stop()
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            value=super()._advance_fleet(i,clock,fleet,traces,served);reference_events.append(dict(sequence=index+1,clock=clock,event_id=self.event_id,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()));return value
    private_rp=Policy(first['budget'],first['capacity'],first['booking_horizon'],first['contract'],0.);private_rp.restore_state(copy.deepcopy(initial['state']));private_env=Env(copy.deepcopy(dataset),first['capacity'],first['tw_speed'],first['snapshot']['num_vehicles'],private_rp,first['contract'],first['booking_horizon']);private_env.event_id=initial['event_id'];private_env.replan_count=initial['replan_count'];private_env.force_suffix={};private_fleet=copy.deepcopy(initial['fleet']);private_traces=copy.deepcopy(initial['traces']);private_served=initial['served'].copy();private_clock=initial['clock'];customers=[int(j) for j in range(1,private_env.num_nodes) if private_env.demands[0,j]>0];reveals=sorted(set(float(private_env.reveal_time[0,j]) for j in customers if private_env.reveal_time[0,j]>0))
    def snapshot():return dict(state=private_rp.export_state(),fleet=private_fleet,traces=private_traces,served=private_served,clock=private_clock,event_id=private_env.event_id,replan_count=private_env.replan_count)
    ck('sealed_actual_post15_physical_start',old.pf.digest(snapshot())==actual['initial_digest'])
    ck('two_new_requests_pinned_environment',len(actual['records'])==2 and np.__version__=='1.26.4')
    for index,row in enumerate(actual['records']):
        name='new:'+str(index+1);packet=row['packet'];response=row['response'];current=response['scoring_packet'];o=row['order'];prior=packet['snapshot']['replanner_state'];a=set(prior['accepted']);r=set(prior['rejected']);pending=set(packet['admission_input']['pending'])
        before=old.pf.digest(snapshot())
        try:private_env._run_loop(0,private_clock,sum(t<=private_clock+1e-6 for t in reveals),private_fleet,private_traces,private_served,customers,reveals,float(private_env.tw_end[0].max()))
        except Stop:private_clock=boundary['clock']
        else:raise AssertionError('new actual reveal missing')
        ck(name+':actual_continuity_before',before==row['resident_before'])
        ck(name+':actual_reveal_clock_event',private_clock==row['clock'] and private_env.event_id==row['event_id'] and row['callback']==16+index)
        expected_pending=sorted((int(j) for j in boundary['visible'] if dataset['demands'][0,j]>0 and not private_served[j] and j not in set(private_rp._accepted)|set(private_rp._rejected)),key=lambda j:(dataset['reveal_time'][0,j],j))
        ck(name+':original_actual_single_pending',expected_pending==[o] and packet['admission_input']['pending']==[o] and o not in set(private_rp._accepted)|set(private_rp._rejected))
        ck(name+':original_private_admission_before',old.pf.digest(private_rp.export_state())==old.pf.digest(row['before'])==old.pf.digest(prior))
        for key,value in packet['dataset'].items():
            target=np.zeros_like(value);target[0,0]=dataset[key][0,0]
            for j in boundary['visible']:target[0,j]=dataset[key][0,j]
            ck(name+':direct_actual_public_field:'+key,np.array_equal(value,target))
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
        ck(name+':original_compiled_identity',state_dict(response['compiled_state'])==replay==row['after'] and digest(replay)==response['compiled_state_sha256'])
        private_rp.restore_state(copy.deepcopy(replay));private_env._plan_and_commit(0,private_clock,private_fleet,private_traces,private_served,customers)
        for k,v in snapshot().items():ck(name+':original_actual_private_commit:'+k,old.pf.digest(v)==old.pf.digest(row['private_after'][k]))
        ck(name+':actual_private_publish_digest',old.pf.digest(snapshot())==row['resident_after'])
        h=raw['host'][index];ck(name+':exact_normal_elapsed',h['finished']-h['started']==h['wall_s'] and h['started']<=row['resident_started']<=row['started']<=response['worker_finished']<=row['committed']<=row['resident_finished']<=h['received']<=h['finished']<h['started']+10. and h['digest']==row['resident_after'])
        begin=read(OUT/'SOURCE_START.json');mid=read(OUT/('SOURCE_COMPUTE_'+str(index+1)+'.json'));end=read(OUT/('SOURCE_END_'+str(index+1)+'.json'))
        ck(name+':loaded_files_and_module_maps',begin['files']==mid['files']==end['files'] and begin['module_files']==mid['module_files']==end['module_files'] and response['source_start']==begin['files'] and response['source_end']==mid['files'] and response['source_qualified'])
        for p,h in end['files'].items():ck(name+':actual_loaded:'+p,old.sha(p)==h)
    ck('all_actual_physical_events_retained',len(reference_events)==len(actual['events']))
    for i,(x,y) in enumerate(zip(reference_events,actual['events'])):
        for k in x:ck('physical:'+str(i)+':'+k,old.pf.digest(x[k])==old.pf.digest(y[k]))
    for k,v in snapshot().items():ck('final_actual_private_identity:'+k,old.pf.digest(v)==old.pf.digest(actual['final'][k]))
    for p,h in reg['sources'].items():ck('registered:'+p,old.sha(p)==h)
    names={'qualify_pickup_resident_two_events_v1.py'}
    ck('actual_owned_resident_gone',not [p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(a).name in names for a in p.info['cmdline'] or [])])
    ck('same_process_and_final_reap',len({h['pid'] for h in raw['host']})==1 and raw['host'][1]['final_reap_included'] and not raw['host'][0]['final_reap_included'])
    ck('no_formal_upgrade',not raw['physical_ack_loss_qualified'] and not raw['full_live_request_qualified'] and not raw['full_loaded_dependency_closure_qualified'] and not raw['stable_online_gain'])
    failed=[c['name'] for c in checks if not c['pass_']];result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,producer_sha256=old.sha(OUT/'RESULT.json'),registration_sha256=old.sha(OUT/'REGISTRATION.json'),two_actual_new_reveals=True,shared_original_physical_implementation=True,full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_RESIDENT_TWO_EVENTS.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed,events=len(reference_events))));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
