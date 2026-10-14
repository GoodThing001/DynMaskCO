"""Independent original-policy prefix execution and direct branch replay."""
import copy,json,pickle,time
from pathlib import Path
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_private_public_transport_v2 as fixture
from strict_online_env import StrictOnlineEnv
from run_exp_rule_control import UncondDynamicReplanner
from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
from energy_complete_saa import EnergyCompleteSAA
from scenario_saa import CondHistoricalSampler
from joint_install import state_dict,digest
import numpy as np
import psutil
ROOT=old.ROOT;P=old.P;OUT=P/'observed_day_extension_root_v1';PREFIX=P/'live_event_prefix_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
class StopReference(Exception):pass
def reference(template,dataset,history):
    events=[];decisions=[];final={}
    class Policy(UncondDynamicReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            env.callbacks+=1
            pending=sorted((int(o) for o in visible if env.demands[i,o]>0 and not served[o] and o not in self._accepted|self._rejected),key=lambda o:(env.reveal_time[i,o],o))
            for o in pending:
                before=self.export_state();super().on_reveal(env,i,clock,fleet,served,[o]);after=self.export_state()
                decisions.append(dict(callback=env.callbacks,clock=clock,event_id=env.event_id,order=o,before=before,after=after))
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            value=super()._advance_fleet(i,clock,fleet,traces,served)
            events.append(dict(event_id=self.event_id,clock=clock,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()))
            return value
        def _plan_and_commit(self,*args):
            value=super()._plan_and_commit(*args)
            return value
    rp=Policy(budget=template['budget'],capacity=template['capacity'],booking_horizon=template['booking_horizon'],contract=template['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10);rp._reset_if_new(0)
    _,decode=old.codecs();initial=decode(read(PREFIX/'RESULT.json'))['final'];rp.restore_state(copy.deepcopy(initial['state']))
    env=Env(copy.deepcopy(dataset),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],rp,template['contract'],template['booking_horizon']);env.callbacks=14
    fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy();clock=initial['clock']
    env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];env.force_suffix={}
    all_customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0]
    reveal_events=sorted(set(float(env.reveal_time[0,j]) for j in all_customers if env.reveal_time[0,j]>0))
    env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveal_events),fleet,traces,served,all_customers,reveal_events,float(env.tw_end[0].max()))
    final.update(clock=events[-1]['clock'],event_id=env.event_id,replan_count=env.replan_count,fleet=fleet,traces=traces,served=served,state=rp.export_state())
    return decisions,events,final
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();raw=decode(read(OUT/'RESULT.json'));reg=read(OUT/'REGISTRATION.json');checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    ck('producer_completed_registered_extension',raw['failed']==0 and len(raw['records'])>0 and not raw['continuous_prefix_only'] and raw['full_day_completed'] and not raw['first14_replayed'])
    ck('registered_environment',np.__version__=='1.26.4')
    for p,h in reg['sources'].items():ck('identity:'+p,old.sha(p)==h)
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    with fixture.TRACE.open(encoding='utf-8') as f:dataset=decode(json.loads(next(f))['payload'])['dataset']
    first=raw['records'][0]['packet'];decisions,events,final=reference(first,dataset,history)
    initial=decode(read(PREFIX/'RESULT.json'))['final']
    initial_digest=old.pf.digest({k:initial[k] for k in ('fleet','traces','served','state','clock','event_id','replan_count')})
    ck('sealed_post_commit_start_exact',initial_digest==raw['final']['initial_post_commit_digest'])
    ck('independent_original_callbacks',len(decisions)==len(raw['records']) and decisions[0]['callback']==15)
    ck('actual_physical_event_count',len(events)==len(raw['events']))
    for i,(got,expected) in enumerate(zip(raw['events'],events)):
        n='event:'+str(i)
        ck(n+':actual_clock_identity',got['event_id']==expected['event_id'] and got['clock']==expected['clock'])
        for k in ('fleet','traces','served'):ck(n+':original_execution:'+k,old.pf.digest(got[k])==old.pf.digest(expected[k]))
    for i,(got,expected) in enumerate(zip(raw['records'],decisions)):
        n='request:'+str(i);packet=got['packet'];response=got['response'];current=response['scoring_packet'];o=got['order'];state=got['after']
        ck(n+':original_reveal_order_clock',all(got[k]==expected[k] for k in ('callback','clock','event_id','order')))
        for k in ('accepted','rejected','budget','mode','status_inst'):ck(n+':original_admission:'+k,got['after'][k]==expected['after'][k])
        known=sorted(int(j) for j in range(1,dataset['demands'].shape[1]) if dataset['demands'][0,j]>0 and dataset['reveal_time'][0,j]<=got['clock']+1e-6)
        ck(n+':actual_public_universe',known==list(packet['snapshot']['customer_universe']))
        for key,value in packet['dataset'].items():
            target=np.zeros_like(value);target[0,0]=dataset[key][0,0]
            for j in known:target[0,j]=dataset[key][0,j]
            ck(n+':direct_observed_projection:'+key,np.array_equal(target,value))
        ck(n+':genuine_live_pending',o not in set(got['before']['accepted'])|set(got['before']['rejected']) and o in packet['admission_input']['pending'])
        ck(n+':snapshot_state_digest',snapshot_state_hash(packet['snapshot'])==packet['snapshot']['state_hash'] and snapshot_state_hash(current['snapshot'])==current['snapshot']['state_hash'])
        ck(n+':no_future_actor_features',all(packet['dataset']['reveal_time'][0,j]<=got['clock']+1e-6 for j in known))
        ev,tr=old.pf.evaluate_known(current,current['plan'])
        ck(n+':original_full_C0_hard',ev['hard_feasible'] and ev['failures']==[])
        ck(n+':raw_closing_trace',old.pf.digest(tr)==old.pf.digest(response['selected_raw']))
        stored=response['selected_certificate']['evaluation']
        for k,v in ev.items():ck(n+':C0_field:'+k,old.pf.digest(v)==old.pf.digest(stored[k]))
        rp=EnergyCompleteSAA(budget=current['budget'],capacity=current['capacity'],booking_horizon=current['booking_horizon'],contract=current['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=CondHistoricalSampler(history['history'],n_neighbors=10),reject_penalty={0:5.,1:10.,2:15.},K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='greedy',energy_pricing='marginal',future_policy='density',standby_orders=0)
        rp.restore_state(copy.deepcopy(state));env=StrictOnlineEnv(current['dataset'],current['capacity'],current['tw_speed'],current['snapshot']['num_vehicles'],rp,current['contract'],current['booking_horizon'])
        fleet,_,served=restore_recourse_snapshot(current['snapshot']);clock=current['snapshot']['clock'];rp._sync_id_map(env,0,clock);snap=rp._build_snapshot(env,0,clock,fleet,served)
        scenarios=rp.sampler.sample(snap,np.random.default_rng(reg['common_seed']),10);saved=response['common_scores']
        ck(n+':fixed_history_scenario_identity',old.pf.digest(scenarios)==saved['scenario_sha256'] and old.pf.digest(snap)==old.pf.digest(saved['observed_snapshot']))
        space,orders=rp._build_space(env,0,scenarios);ids={id(v):space.n_real+j for j,v in enumerate(orders)}
        for label,plan in {'own_incumbent':current['plan'],'candidate':response['proposal']}.items():
            start=rp._sim_states(env,0,clock,fleet,served,{int(k):v for k,v in plan.items()})
            vals=[float(rp._sim_scenario(env,0,clock,start,[ids[id(v)] for v in scen],space)) for scen in scenarios];s=saved['candidate_scores'][label]
            for j,(x,y) in enumerate(zip(vals,s['branch_values'])):ck(n+':'+label+':branch:'+str(j),x.hex()==y.hex())
            ck(n+':'+label+':all10_mean',len(vals)==len(s['branch_values'])==10 and float(np.mean(vals)).hex()==s['mean_shadow_utility'].hex())
        ck(n+':live_compile_publish',state_dict(response['compiled_state'])==got['after'] and digest(got['after'])==response['compiled_state_sha256'] and got['installed'])
        ck(n+':actual_normal_total_and_compute',got['started']<=response['worker_finished']<got['started']+8. and got['finished']<got['started']+10.)
        ck(n+':shared_sequence_and_sources',got['transport']['sequence']==got['callback']-14 and not got['transport']['timed_pipe_IO'] and response['source_end']==raw['ready']['sources'])
    for k in ('clock','event_id','replan_count','fleet','traces','served'):ck('prefix_final:'+k,old.pf.digest(final[k])==old.pf.digest(raw['final'][k]))
    for k in ('accepted','rejected','budget','mode','status_inst'):ck('prefix_final:state:'+k,final['state'][k]==raw['final']['state'][k])
    ck('new_requests_and_prepared_attempts_retained',len((OUT/'REQUESTS_STARTED.jsonl').read_text(encoding='utf-8').splitlines())==len(decisions) and len((OUT/'PREPARED_RESPONSES.jsonl').read_text(encoding='utf-8').splitlines())==len(decisions))
    from coldchain_evaluator_a1 import evaluate_trace_a1
    ev=evaluate_trace_a1(final['traces'],{k:v[0] for k,v in dataset.items()},first['contract'],set(final['state']['accepted']),set(final['state']['rejected']),first['budget'],first['penalties'])
    for k,v in ev.items():ck('actual_day_original_evaluation:'+k,old.pf.digest(v)==old.pf.digest(raw['original_final_evaluation'][k]))
    ck('original_complete_hard_and_coverage',ev['hard_feasible'] and not ev['failures'] and set(np.flatnonzero(final['served']))-{0}==set(final['state']['accepted']))
    ck('original_actual_used_return',all(t.return_arrival is not None for t in final['traces'] if t.services))
    ck('original_day_no_pending',set(np.flatnonzero(dataset['demands'][0]>0))==set(final['state']['accepted'])|set(final['state']['rejected']))
    try:psutil.Process(raw['worker_pid']);gone=False
    except psutil.NoSuchProcess:gone=True
    ck('owned_resident_worker_gone',gone)
    ck('not_full_day_or_preemptible_parent_or_gain',raw['full_day_completed'] and not raw['parent_preemptible'] and not raw['full_live_10s_qualified'] and not raw['full_loaded_dependency_closure_qualified'] and not raw['stable_online_gain'])
    failed=[c['name'] for c in checks if not c['pass_']]
    result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,producer_sha256=old.sha(OUT/'RESULT.json'),registration_sha256=old.sha(OUT/'REGISTRATION.json'),
                original_policy_extension_execution=True,first14_replayed=False,physical_events=len(events),accepted=len(final['state']['accepted']),rejected=len(final['state']['rejected']),services=sum(len(t.services) for t in final['traces']),
                independent_assembly_shares_original_physical_model=True,full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_OBSERVED_DAY_EXTENSION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed,physical_events=len(events),services=result['services'])));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
