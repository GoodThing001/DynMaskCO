"""Independent original admission/K10/C0 and actual two-event private progression."""
import copy,json,pickle,hashlib,ctypes,os,struct
from pathlib import Path
import numpy as np,psutil
from pickup_loaded_closure_v1 import capture,sha
import qualify_pickup_joint_policy_transaction_root_v3 as old
import hard_admission,rolling_optimizer
import qualify_pickup_private_public_transport_v2 as fixture
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
from run_exp_rule_control import UncondDynamicReplanner
from energy_complete_saa import EnergyCompleteSAA
from scenario_saa import CondHistoricalSampler
from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
from joint_install import state_dict,digest
ROOT=old.ROOT;P=old.P;OUT=P/'restore_compute_cancel_root_v1';SOURCE=P/'timed_restore_live_root_v1/ACTUAL_SHARED_PAYLOAD_5.bin'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();reg=read(OUT/'REGISTRATION.json');raw=read(OUT/'RESULT.json');initial=pickle.loads(SOURCE.read_bytes())['state'];checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    witnesses={i:decode(read(OUT/('CASE_'+str(i))/'FAULT_WITNESS.json')) for i in (1,2)}
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    with fixture.TRACE.open(encoding='utf-8') as f:dataset=decode(json.loads(next(f))['payload'])['dataset']
    details=witnesses[2]['details'];first=details['packet'];boundary={};reference_events=[];index=0
    class Stop(Exception):pass
    class Policy(C0ReserveReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):boundary.update(clock=clock,visible=visible);raise Stop()
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            value=super()._advance_fleet(i,clock,fleet,traces,served);reference_events.append(dict(sequence=6,clock=clock,event_id=self.event_id,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()));return value
    private_rp=Policy(first['budget'],first['capacity'],first['booking_horizon'],first['contract'],0.);private_rp.restore_state(copy.deepcopy(initial['state']));private_env=Env(copy.deepcopy(dataset),first['capacity'],first['tw_speed'],first['snapshot']['num_vehicles'],private_rp,first['contract'],first['booking_horizon']);private_env.event_id=initial['event_id'];private_env.replan_count=initial['replan_count'];private_env.force_suffix={};private_fleet=copy.deepcopy(initial['fleet']);private_traces=copy.deepcopy(initial['traces']);private_served=initial['served'].copy();private_clock=initial['clock'];customers=[int(j) for j in range(1,private_env.num_nodes) if private_env.demands[0,j]>0];reveals=sorted(set(float(private_env.reveal_time[0,j]) for j in customers if private_env.reveal_time[0,j]>0))
    def snapshot():return dict(state=private_rp.export_state(),fleet=private_fleet,traces=private_traces,served=private_served,clock=private_clock,event_id=private_env.event_id,replan_count=private_env.replan_count)
    dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name','cmdline']));capture()
    identity={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()};identity.update({str(SOURCE):sha(SOURCE),str(Path(__file__)):sha(Path(__file__)),str(ROOT/'tools/build_pickup_restore_compute_cancel_audit_v1.py'):sha(ROOT/'tools/build_pickup_restore_compute_cancel_audit_v1.py')})
    with (OUT/'AUDIT_REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(dict(inputs=identity,scope='Stored two new stage faults; actual original raw sealed state, cancellation times/processes/loaded identities. Only second new next-reveal original admission/all20K10/C0/private precommit physical reference; no producer/engine/projector/checkpoint execution.'),f,indent=2)
    begin=capture(identity)
    with (OUT/'AUDIT_SOURCE_START.json').open('x',encoding='utf-8') as f:json.dump(begin,f,indent=2)
    partial=witnesses[1]['details']
    for k,v in initial.items():ck('partial_selected_source:'+k,old.pf.digest(v)==old.pf.digest(partial['selected_state'][k]))
    ck('partial_replanner_exact_not_complete_env',old.pf.digest(partial['restored_replanner'])==old.pf.digest(initial['state']) and not partial['environment_restore_complete'] and partial['admission_calls']==partial['physical_calls']==0)
    restore=decode(read(OUT/'CASE_2/RESTORE_IN_REQUEST.json'))
    for k,v in initial.items():ck('compute_selected_restore:'+k,old.pf.digest(v)==old.pf.digest(restore['state'][k]))
    # This view only maps stored fields to the original static reference below.
    row=dict(packet=first,response=details['response'],order=first['admission_input']['current_order'],clock=first['snapshot']['clock'],event_id=first['snapshot']['event_id'],before=details['before'],resident_before=restore['selected_digest'],callback=23)
    for _ in (0,):
        name='new:'+str(index+1);packet=row['packet'];response=row['response'];current=response['scoring_packet'];o=row['order'];prior=packet['snapshot']['replanner_state'];a=set(prior['accepted']);r=set(prior['rejected']);pending=set(packet['admission_input']['pending'])
        before=old.pf.digest(snapshot())
        try:private_env._run_loop(0,private_clock,sum(t<=private_clock+1e-6 for t in reveals),private_fleet,private_traces,private_served,customers,reveals,float(private_env.tw_end[0].max()))
        except Stop:private_clock=boundary['clock']
        else:raise AssertionError('new actual reveal missing')
        ck(name+':actual_continuity_before',before==row['resident_before'])
        ck(name+':actual_reveal_clock_event',private_clock==row['clock'] and private_env.event_id==row['event_id'] and row['callback']==23)
        expected_pending=sorted((int(j) for j in boundary['visible'] if dataset['demands'][0,j]>0 and not private_served[j] and j not in set(private_rp._accepted)|set(private_rp._rejected)),key=lambda j:(dataset['reveal_time'][0,j],j))
        ck(name+':original_actual_single_pending',expected_pending==[o] and packet['admission_input']['pending']==[o] and o not in set(private_rp._accepted)|set(private_rp._rejected))
        ck(name+':original_private_admission_before',old.pf.digest(private_rp.export_state())==old.pf.digest(row['before']))
        suffix={str(v):[int(j) for j in private_rp.export_state()['plan'].get(str(v),private_rp.export_state()['plan'].get(v,[])) if not private_served[j]] for v in range(len(private_fleet))}
        for v in private_fleet:
            if v.status=='committed' and v.committed_next not in (None,0):suffix[str(v.vehicle_id)]=[int(v.committed_next)]+[j for j in suffix[str(v.vehicle_id)] if j!=v.committed_next]
        ck(name+':direct_public_remaining_suffix',old.pf.digest(suffix)==old.pf.digest(prior['plan']))
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
        ck(name+':original_compiled_identity',state_dict(response['compiled_state'])==replay and digest(replay)==response['compiled_state_sha256'])
    for k,v in snapshot().items():ck('uninstalled_private_at_reveal:'+k,old.pf.digest(v)==old.pf.digest(details['private_at_reveal'][k]))
    ck('all_private_precommit_physical_events',len(reference_events)==len(details['events']))
    for i,(x,y) in enumerate(zip(reference_events,details['events'])):
        for k,v in x.items():ck('physical:'+str(i)+':'+k,old.pf.digest(v)==old.pf.digest(y[k]))
    head=struct.Struct('<QQ32s32s32s')
    for i,h in zip((1,2),raw['host']):
        d=OUT/('CASE_'+str(i));w=witnesses[i];b=read(d/'SOURCE_START.json');e=read(d/'SOURCE_AT_FAULT.json');envelope=(d/'RETAINED_SHARED_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope);payload=envelope[head.size:];cp=pickle.loads(payload)
        ck(str(i)+':actual_original_raw_retained',envelope==(d/'INITIAL_SHARED_ENVELOPE.bin').read_bytes() and payload==SOURCE.read_bytes() and len(payload)==size and hashlib.sha256(payload).digest()==ph and epoch==cp['epoch']==5 and bh.hex()==cp['base_digest'] and sh.hex()==old.pf.digest(initial))
        ck(str(i)+':reached_and_cancelled_in10',h['stage']==w['stage']==i and h['sequence']==w['sequence']==6 and h['started']<=w['reached']<h['cutoff']<h['finished']<h['started']+10. and h['wall_s']==h['finished']-h['started'] and w['deadline']==h['cutoff'])
        ck(str(i)+':no_publication_new_owner_reply',not h['publication_signaled'] and h['reply_sequence']==0 and h['reaped'] and h['authority_digest']==old.pf.digest(initial) and not (d/'OWNER_STATE_6.json').exists() and not (d/'ACTUAL_WRITER_PAYLOAD_6.bin').exists())
        ck(str(i)+':fault_compute_segment_closure',b['files']==e['files'] and b['module_files']==e['module_files'] and w['loaded_closure_qualified'])
        for p,v in e['files'].items():ck(str(i)+':loaded:'+p,sha(p)==v)
        ck(str(i)+':actual_worker_gone',not psutil.pid_exists(h['pid']) or Path(psutil.Process(h['pid']).exe()).name.lower()!='python.exe')
    ck('runtime_all_checks_pass',raw['failed']==0 and raw['checks_count']==13)
    for p,v in reg['sources'].items():ck('registered:'+p,sha(p)==v)
    ck('all_actual_inputs_preserved',all(sha(p)==v for p,v in identity.items()))
    end=capture(identity)
    ck('actual_independent_closure',begin['files']==end['files'] and begin['module_files']==end['module_files'])
    for p,v in begin['files'].items():ck('audit_loaded:'+p,end['files'].get(p)==v==sha(p))
    with (OUT/'AUDIT_SOURCE_END.json').open('x',encoding='utf-8') as f:json.dump(end,f,indent=2)
    failed=[x['name'] for x in checks if not x['pass_']];result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,shared_original_physical_and_canonical_implementation=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_RESTORE_COMPUTE_CANCEL.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failure_names=failed,events=len(reference_events),order=row['order'])));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
