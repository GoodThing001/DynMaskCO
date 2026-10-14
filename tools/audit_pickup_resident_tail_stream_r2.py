"""Independent original admission/K10/C0 and actual two-event private progression."""
import copy,json,pickle,hashlib,ctypes,os,struct
from pathlib import Path
import numpy as np,psutil
from pickup_loaded_closure_v1 import capture,sha
import qualify_pickup_joint_policy_transaction_root_v3 as old
import hard_admission,rolling_optimizer
import qualify_pickup_private_public_transport_v2 as fixture
from strict_online_env import StrictOnlineEnv
from coldchain_evaluator_a1 import evaluate_trace_a1
from run_exp_energy_c0 import C0ReserveReplanner
from run_exp_rule_control import UncondDynamicReplanner
from energy_complete_saa import EnergyCompleteSAA
from scenario_saa import CondHistoricalSampler
from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
from joint_install import state_dict,digest
ROOT=old.ROOT;P=old.P;OUT=P/'resident_day_tail_root_v1';SOURCE=P/'lineage_header_cancel_root_v1/RETAINED_PAYLOAD.bin'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();reg=read(OUT/'REGISTRATION.json');initial=pickle.loads(SOURCE.read_bytes())['state'];checks=[]
    class StreamRecords:
        def __len__(self):return 50
        def __getitem__(self,i):
            if i<0:i+=50
            if not 0<=i<50:raise IndexError(i)
            return decode(read(OUT/('REQUEST_'+str(i+14)+'.json')))
        def __iter__(self):
            for i in range(50):yield self[i]
    actual=dict(initial_digest=old.pf.digest(initial),records=StreamRecords(),final=pickle.loads((OUT/'ACTUAL_SHARED_PAYLOAD_63.bin').read_bytes())['state'])
    raw=dict(host=[read(OUT/('HOST_'+str(i)+'.json')) for i in range(14,64)],physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False)

    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    with fixture.TRACE.open(encoding='utf-8') as f:dataset=decode(json.loads(next(f))['payload'])['dataset']
    first=actual['records'][0]['packet'];boundary={};reference_events=[]
    class Stop(Exception):pass
    class Policy(C0ReserveReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):boundary.update(clock=clock,visible=visible);raise Stop()
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            value=super()._advance_fleet(i,clock,fleet,traces,served);reference_events.append(dict(sequence=reference_sequence,clock=clock,event_id=self.event_id,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()));return value
    private_rp=Policy(first['budget'],first['capacity'],first['booking_horizon'],first['contract'],0.);private_rp.restore_state(copy.deepcopy(initial['state']));private_env=Env(copy.deepcopy(dataset),first['capacity'],first['tw_speed'],first['snapshot']['num_vehicles'],private_rp,first['contract'],first['booking_horizon']);private_env.event_id=initial['event_id'];private_env.replan_count=initial['replan_count'];private_env.force_suffix={};private_fleet=copy.deepcopy(initial['fleet']);private_traces=copy.deepcopy(initial['traces']);private_served=initial['served'].copy();private_clock=initial['clock'];customers=[int(j) for j in range(1,private_env.num_nodes) if private_env.demands[0,j]>0];reveals=sorted(set(float(private_env.reveal_time[0,j]) for j in customers if private_env.reveal_time[0,j]>0))
    def snapshot():return dict(state=private_rp.export_state(),fleet=private_fleet,traces=private_traces,served=private_served,clock=private_clock,event_id=private_env.event_id,replan_count=private_env.replan_count)
    ck('selected_epoch13_actual_start',old.pf.digest(snapshot())==actual['initial_digest'])
    ck('two_new_requests_pinned_environment',len(actual['records'])==50 and np.__version__=='1.26.4')
    dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name','cmdline']));capture()
    identity={str(p):sha(p) for p in list(OUT.iterdir())+[SOURCE,Path(__file__)] if p.is_file()}
    with (OUT/'STREAM_AUDIT_REGISTRATION_R2.json').open('x',encoding='utf-8') as f:json.dump(dict(inputs=identity,scope='First independent stream replay of all50 original submitted callbacks31..80. Original producer RUN_EXIT1/terminal64 unpublished and absent cumulative physical archive preserved. All1000K10, original C0, per-request whole private post-commit states/physical suffix and actual raw checkpoints; original per-advance event list unavailable, cannot certify raw intermediate event ledger. Do not rerun production, whole frozen auditor, terminal or old qualified QA.'),f,indent=2)
    audit_begin=capture(identity)
    with (OUT/'STREAM_AUDIT_SOURCE_START_R2.json').open('x',encoding='utf-8') as f:json.dump(audit_begin,f,indent=2)
    reference_sequence=14
    for index,row in enumerate(actual['records']):
        reference_sequence=index+14
        name='new:'+str(index+1);packet=row['packet'];response=row['response'];current=response['scoring_packet'];o=row['order'];prior=packet['snapshot']['replanner_state'];a=set(prior['accepted']);r=set(prior['rejected']);pending=set(packet['admission_input']['pending'])
        before=old.pf.digest(snapshot())
        try:private_env._run_loop(0,private_clock,sum(t<=private_clock+1e-6 for t in reveals),private_fleet,private_traces,private_served,customers,reveals,float(private_env.tw_end[0].max()))
        except Stop:private_clock=boundary['clock']
        else:raise AssertionError('new actual reveal missing')
        ck(name+':actual_continuity_before',before==row['resident_before'])
        ck(name+':actual_reveal_clock_event',private_clock==row['clock'] and private_env.event_id==row['event_id'] and row['callback']==31+index)
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
        ck(name+':original_compiled_identity',state_dict(response['compiled_state'])==replay==row['after'] and digest(replay)==response['compiled_state_sha256'])
        private_rp.restore_state(copy.deepcopy(replay));private_env._plan_and_commit(0,private_clock,private_fleet,private_traces,private_served,customers)
        for k,v in snapshot().items():ck(name+':original_actual_private_commit:'+k,old.pf.digest(v)==old.pf.digest(row['private_after'][k]))
        ck(name+':actual_private_publish_digest',old.pf.digest(snapshot())==row['resident_after'])
        h=raw['host'][index];seq=index+14
        ck(name+':within10_exact_clock',h['finished']-h['started']==h['wall_s'] and h['finished']<h['started']+10. and h['started']<=row['resident_started']<=row['started']<=response['worker_finished']<=row['committed']<=row['resident_finished'])
        payload=(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin')).read_bytes();c=pickle.loads(payload)
        ck(name+':actual_private_writer_checkpoint',len(payload)==row['payload_size'] and hashlib.sha256(payload).hexdigest()==row['payload_sha256'] and c['epoch']==seq and c['base_digest']==row['resident_before'] and old.pf.digest(c['state'])==old.pf.digest(snapshot()))
        seq=index+14;envelope=(OUT/('ACTUAL_SHARED_ENVELOPE_'+str(seq)+'.bin')).read_bytes();head=struct.Struct('<QQ32s32s32s');epoch,size,bh,ph,sh=head.unpack_from(envelope)
        ck(name+':selected_actual_raw_envelope',envelope[head.size:]==payload==(OUT/('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin')).read_bytes() and len(payload)==size and hashlib.sha256(payload).digest()==ph and epoch==seq and bh.hex()==c['base_digest'] and sh.hex()==old.pf.digest(c['state']))
        v=h['validation'];ck(name+':parent_bounded_SHA_before_choice',v['started']<=v['finished']<=h['chosen_at']<h['cutoff'] and v['slot_before']==v['slot_after']==h['selected_slot']==h['ticket']['slot'] and v['payload_sha256']==hashlib.sha256(payload).hexdigest() and v['base_digest']==row['resident_before'] and v['sequence']==seq)
        ck(name+':selected_after_private_before_cutoff',h['started']<=row['resident_finished']<=h['chosen_at']<h['cutoff'] and h['ticket']['digest']==row['resident_after'])
        ck(name+':normal_ACK_alternating_resident',h['selected_slot']==(1-index%2) and h['ack_present'] and h['event_signaled_final'] and not h['reaped'])
        begin=read(OUT/'SOURCE_START.json');mid=read(OUT/('SOURCE_COMPUTE_'+str(index+14)+'.json'));end=read(OUT/('SOURCE_END_'+str(index+14)+'.json'))
        ck(name+':loaded_files_and_module_maps',begin['files']==mid['files']==end['files'] and begin['module_files']==mid['module_files']==end['module_files'] and response['source_start']==begin['files'] and response['source_end']==mid['files'] and response['source_qualified'])
        for p,h in end['files'].items():ck(name+':actual_loaded:'+p,old.sha(p)==h)
    ck('reference_physics_matches_stored_event_increment',len(reference_events)==actual['final']['event_id']-initial['event_id'])
    for k,v in snapshot().items():ck('final_actual_private_identity:'+k,old.pf.digest(v)==old.pf.digest(actual['final'][k]))
    for p,h in reg['sources'].items():ck('registered:'+p,old.sha(p)==h)
    names={'qualify_pickup_resident_day_tail_v1.py'}
    ck('actual_owned_resident_gone',not [p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(a).name in names for a in p.info['cmdline'] or [])])
    ck('same_resident50_original_terminal_failed',len({h['pid'] for h in raw['host']})==1 and all(not h['reaped'] for h in raw['host']) and read(OUT/'RUN_EXIT.json')['exit_code']==1 and not (OUT/'ACTUAL_SHARED_PAYLOAD_64.bin').exists() and not (OUT/'OWNER_STATE_64.json').exists())
    ck('no_formal_upgrade',not raw['physical_ack_loss_qualified'] and not raw['full_live_request_qualified'] and not raw['full_loaded_dependency_closure_qualified'] and not raw['stable_online_gain'])
    restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'));h=raw['host'][0]
    ck('selected13_restore_first_actual31_same_window',h['started']<=restore['started']<=restore['finished']<=actual['records'][0]['started']<=h['finished']<h['cutoff'] and restore['source_epoch']==13 and restore['source_payload_sha256']==sha(SOURCE) and restore['selected_digest']==actual['initial_digest'] and restore['warm_digest']!=restore['selected_digest'])
    for k,v in initial.items():ck('actual_restored_source_field:'+k,old.pf.digest(v)==old.pf.digest(restore['state'][k]))
    ck('restore_no_prior_execution',restore['physical_calls']==restore['admission_calls']==0)
    head=struct.Struct('<QQ32s32s32s');envelope=(OUT/'INITIAL_SLOT0_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope)
    ck('actual_source_retained13_loaded_newslot0',epoch==13 and envelope[head.size:]==SOURCE.read_bytes() and size==len(SOURCE.read_bytes()) and ph.hex()==sha(SOURCE) and bh.hex()==pickle.loads(SOURCE.read_bytes())['base_digest'] and sh.hex()==old.pf.digest(initial))
    ck('all50_continuous_callbacks_no_selection',len(actual['records'])==50 and all(h['sequence']==i+14 for i,h in enumerate(raw['host'])))
    ck('all_actual_audit_inputs_unchanged',all(sha(p)==h for p,h in identity.items()))
    audit_end=capture(identity)
    for p,h in audit_begin['files'].items():ck('audit_loaded:'+p,audit_end['files'].get(p)==h==sha(p))
    ck('audit_actual_loaded_closure',audit_begin['files']==audit_end['files'] and audit_begin['module_files']==audit_end['module_files'])
    with (OUT/'STREAM_AUDIT_SOURCE_END_R2.json').open('x',encoding='utf-8') as f:json.dump(audit_end,f,indent=2)
    failed=[c['name'] for c in checks if not c['pass_']];result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,producer_sha256=old.sha(OUT/'RESULT.json'),registration_sha256=old.sha(OUT/'REGISTRATION.json'),scoped_fifty_submitted_request_replay_qualified=not failed,original_whole_run_PASS=False,registered_barrier_failures=['original_terminal64_not_selected','original_cumulative_physical_event_archive_absent'],raw_replay_failure_names=failed,reference_physical_events=len(reference_events),original_terminal_not_reexecuted=True,shared_original_physical_implementation=True,full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_RESIDENT_TAIL_STREAM_R2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed,events=len(reference_events))));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
