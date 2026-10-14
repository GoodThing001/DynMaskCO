"""Continue the sealed post-commit first14 boundary to original observed-day termination."""
import copy,json,pickle,struct,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_private_public_transport_v2 as fixture
import pickup_pending_transaction_v1 as engine
import pickup_undecided_public_v1 as projector
from pickup_live_shared_transaction_v1 import LiveShared,CONTROL,REPLY,MAX_INPUT,MAX_OUTPUT,SharedMemory
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
from joint_install import transactional_class,JointPolicyOwner,state_dict
from recourse_snapshot import snapshot_state_hash
import numpy as np
import psutil
ROOT=old.ROOT;P=old.P;E=old.E;OUT=P/'observed_day_extension_root_v1';PREFIX=P/'live_event_prefix_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def setup():
    _,decode=old.codecs()
    row=next(r for r in read(P/'undecided_public_input_root_v2/RESULT.json')['records'] if r['kind']=='actual_first_reveal_hook' and r['variant']==0)
    packet=decode(row['packet'])
    with fixture.TRACE.open(encoding='utf-8') as f:dataset=decode(json.loads(next(f))['payload'])['dataset']
    return packet,dataset
def sources():
    manifest=old.pf.source_hashes(E/'scripts')
    names=('qualify_pickup_observed_day_extension_v1.py','audit_pickup_observed_day_extension_v1.py','run_pickup_observed_day_extension_bounded.py','pickup_live_shared_transaction_v1.py','pickup_pending_transaction_v1.py','pickup_partial_common_v1.py','pickup_partial_trace_v1.py','pickup_partial_known_v1.py','pickup_partial_validation_v1.py','pickup_undecided_public_v1.py')
    for p in [ROOT/'tools'/x for x in names]+[PREFIX/'RESULT.json',PREFIX/'REGISTRATION.json',PREFIX/'INDEPENDENT_LIVE_PREFIX.json',PREFIX/'INDEPENDENT_MODE_INITIALIZATION_RECOVERY_R2.json',fixture.TRACE,P/'undecided_public_input_root_v2/RESULT.json',old.HISTORY,Path(sys.executable),Path(np.__file__),Path(np.core._multiarray_umath.__file__)]:manifest[str(p)]=old.sha(p)
    return manifest
def worker():
    psutil.Process().cpu_affinity([7])
    if np.__version__!='1.26.4':raise ValueError('original environment')
    control=SharedMemory(name=sys.argv[2]);incoming=SharedMemory(name=sys.argv[3]);outgoing=SharedMemory(name=sys.argv[4])
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    manifest=sources();print(json.dumps(dict(kind='ready',sources=manifest,cpu=[7],numpy=np.__version__)),flush=True);last=0
    try:
        while True:
            seq,deadline,size,fault=CONTROL.unpack_from(control.buf)
            if not seq or seq==last:time.sleep(.001);continue
            last=seq
            try:
                struct.pack_into('<I',control.buf,32,1)
                if fault==1:time.sleep(1.)
                if size>MAX_INPUT or time.perf_counter()>=deadline:raise TimeoutError('input decode')
                packet=pickle.loads(bytes(incoming.buf[:size]));struct.pack_into('<I',control.buf,32,2)
                response=engine.compute(packet,history,E/'scripts',deadline)
                response['source_end']={p:old.sha(p) for p in manifest};response['source_qualified']=response['source_end']==manifest
                struct.pack_into('<I',control.buf,32,3);data=pickle.dumps(response,protocol=5);kind=1
                if time.perf_counter()>=deadline:raise TimeoutError('late encoding')
            except Exception:data=pickle.dumps(dict(traceback=traceback.format_exc()),protocol=5);kind=2
            if len(data)>MAX_OUTPUT:data=pickle.dumps(dict(error='bounded output exceeded'));kind=2
            outgoing.buf[REPLY.size:REPLY.size+len(data)]=data;REPLY.pack_into(outgoing.buf,0,seq,kind,len(data))
    finally:control.close();incoming.close();outgoing.close()
class PrefixStop(Exception):pass
def run_extension(packet,dataset,w,records,events,context):
    cls=transactional_class(C0ReserveReplanner);encode,_=old.codecs()
    class Policy(cls):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            self._reset_if_new(i);env.callback_count+=1
            pending=sorted((int(o) for o in visible if env.demands[i,o]>0 and not served[o] and o not in self._accepted|self._rejected),key=lambda o:(env.reveal_time[i,o],o))
            if not pending:raise ValueError('empty actual reveal admission event')
            for o in pending:
                started=time.perf_counter();deadline=started+10.;before=self.export_state();pointer=self._policy
                _,_,actual,traces,actual_served=env.context
                if actual is not fleet or actual_served is not served:raise ValueError('live event ownership identity')
                reveal_index=len(set(float(env.reveal_time[i,j]) for j in visible if env.demands[i,j]>0 and env.reveal_time[i,j]>0))
                projected=projector.export_undecided(env,i,clock,env.event_id,reveal_index,fleet,traces,served,range(1,env.num_nodes),o)
                with (OUT/'REQUESTS_STARTED.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(encode(dict(callback=env.callback_count,event_id=env.event_id,clock=clock,order=o,packet=projected)))+'\n')
                owner=JointPolicyOwner(self,fleet,clock,env.event_id);token=owner.begin(str(o),o,deadline,old.pf.digest(projected))
                response,transport=w.ask(projected,deadline,reserve_s=2.)
                with (OUT/'PREPARED_RESPONSES.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(encode(dict(callback=env.callback_count,response=response,transport=transport)))+'\n')
                installed,reason=owner.commit(token,response);finished=time.perf_counter()
                if not installed or finished>=deadline:raise ValueError('live publication failed '+reason)
                if self._policy is pointer:raise ValueError('live actual policy pointer unchanged')
                after=self.export_state();records.append(dict(callback=env.callback_count,event_id=env.event_id,clock=clock,order=o,packet=projected,before=before,after=after,response=response,transport=transport,started=started,finished=finished,wall_s=finished-started,installed=installed,reason=reason))
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            before=old.pf.digest(dict(fleet=fleet,traces=traces,served=served))
            value=super()._advance_fleet(i,clock,fleet,traces,served);self.context=(i,clock,fleet,traces,served)
            events.append(dict(event_id=self.event_id,clock=clock,before=before,after=old.pf.digest(dict(fleet=fleet,traces=traces,served=served)),fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()))
            return value
        def _plan_and_commit(self,*args):
            value=super()._plan_and_commit(*args)
            return value
    rp=Policy(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.);rp._reset_if_new(0)
    _,decode=old.codecs();initial=decode(read(PREFIX/'RESULT.json'))['final']
    rp.restore_state(copy.deepcopy(initial['state']))
    env=Env(copy.deepcopy(dataset),packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon']);env.callback_count=14
    fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy()
    env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];env.force_suffix={}
    clock=initial['clock'];env.context=(0,clock,fleet,traces,served)
    context['initial_post_commit_digest']=old.pf.digest(dict(fleet=fleet,traces=traces,served=served,state=rp.export_state(),clock=clock,event_id=env.event_id,replan_count=env.replan_count))
    all_customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0]
    reveal_events=sorted(set(float(env.reveal_time[0,j]) for j in all_customers if env.reveal_time[0,j]>0))
    reveal_idx=sum(t<=clock+1e-6 for t in reveal_events)
    # The sealed input is AFTER plan_and_commit: do not execute that boundary twice.
    env._run_loop(0,clock,reveal_idx,fleet,traces,served,all_customers,reveal_events,float(env.tw_end[0].max()))
    context.update(clock=env.context[1],event_id=env.event_id,replan_count=env.replan_count,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy(),state=copy.deepcopy(rp.export_state()),visible=all_customers)
    return records,events,context
def main():
    psutil.Process().cpu_affinity([7]);encode,_=old.codecs();packet,dataset=setup();OUT.mkdir(exist_ok=False)
    reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),observed_day_input= str(fixture.TRACE),continuation_rule='sealed first14 post-plan_and_commit boundary, original _run_loop until termination; no prefix replay, no outcome-selected states',prefix_result_sha256=old.sha(PREFIX/'RESULT.json'),supervision_cap_s=900.,
             normal_cap_s=10.,worker_compute_cap_s=8.,cleanup_parent_reserve_s=2.,new_fault_injection=False,
             admission='own_uncond_dynamic',route='own_incumbent',K=10,common_seed=202610111301,cpu=[7],numpy='1.26.4',
             source_full_day_parameters_unchanged=True,uses_final_scientific_seeds=False,new_scientific_data=False,
             parent_projection_encode_decode_commit_preemptible=False,full_live_10s_qualified=False,all_formal_arms=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False)
    save('REGISTRATION.json',reg);checks=[];records=[];events=[];final={};w=None
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    try:
        w=LiveShared(Path(__file__));ready=w.ready;records,events,final=run_extension(packet,dataset,w,records,events,final);w.close();pid=w.process.pid;w=None
        ck('registered_continuation_only',len(records)>0 and min(r['callback'] for r in records)==15 and len(set(r['callback'] for r in records))==len(records))
        ck('actual_parent_fleet_execution_services',sum(len(t.services) for t in final['traces'])>0)
        ck('no_pending_at_prefix_boundary',set(final['visible'])==set(final['state']['accepted'])|set(final['state']['rejected']))
        ck('physical_events_advanced',len(events)>len(records) and all(events[j]['clock']<=events[j+1]['clock'] for j in range(len(events)-1)))
        for r in records:
            n=str(r['callback']);current=r['response']['scoring_packet'];remaining=set(r['packet']['admission_input']['pending'])-{r['order']}
            ck(n+':actual_joint_publish',r['installed'] and state_dict(r['response']['compiled_state'])==r['after'])
            ck(n+':only_current_decision',set(r['after']['accepted'])-{r['order']}==set(r['before']['accepted']) and set(r['after']['rejected'])-{r['order']}==set(r['before']['rejected']))
            ck(n+':remaining_pending_preserved',set(current['admission_input']['pending'])==remaining and not remaining&(set(r['after']['accepted'])|set(r['after']['rejected'])))
            ck(n+':complete20C0branches',r['response']['selected_certificate']['qualified'] and all(c['qualified'] and len(c['branch_values'])==10 for c in r['response']['common_scores']['candidate_scores'].values()))
            ck(n+':charged_normal_wall',r['wall_s']<10. and r['response']['worker_finished']<r['started']+8.)
            ck(n+':new_shared_transfer',r['transport']['timed_pipe_IO'] is False and r['transport']['sequence']==r['callback']-14)
            ck(n+':source_end',r['response']['source_end']==ready['sources'])
        from coldchain_evaluator_a1 import evaluate_trace_a1
        final_ev=evaluate_trace_a1(final['traces'],{k:v[0] for k,v in dataset.items()},packet['contract'],set(final['state']['accepted']),set(final['state']['rejected']),packet['budget'],packet['penalties'])
        ck('actual_full_day_original_C0',final_ev['hard_feasible'] and not final_ev['failures'])
        ck('all_actual_accepted_served',set(np.flatnonzero(final['served']))-{0}==set(final['state']['accepted']))
        ck('all_observed_decided_at_day_end',set(final['visible'])==set(final['state']['accepted'])|set(final['state']['rejected']))
        ck('actual_used_vehicles_returned',all(t.return_arrival is not None for t in final['traces'] if t.services))
        ck('actual_terminal_vehicle_states',all(v.status in ('idle','closed') for v in final['fleet']))
        ck('selected_sources_end',all(old.sha(p)==h for p,h in reg['sources'].items()))
        result=dict(checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),records=records,events=events,final=final,ready=ready,worker_pid=pid,original_final_evaluation=final_ev,
                    continuous_prefix_only=False,full_day_completed=True,first14_replayed=False,full_live_10s_qualified=False,parent_preemptible=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False)
        save('RESULT.json',encode(result));save('RUN_EXIT.json',dict(exit_code=int(result['failed']>0)))
        print(json.dumps(dict(checks=len(checks),failed=result['failed'],requests=len(records),events=len(events),services=sum(len(t.services) for t in final['traces']),walls=[r['wall_s'] for r in records],full_day_completed=True)));return int(result['failed']>0)
    except Exception:
        if w:w.close()
        save('FAILURE.json',dict(traceback=traceback.format_exc(),checks=checks,records=encode(records),events=encode(events)));save('RUN_EXIT.json',dict(exit_code=1));raise
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
