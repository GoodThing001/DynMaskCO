"""Two new observed reveals in one resident private owner, no old request replay."""
import copy,json,os,pickle,struct,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from multiprocessing.shared_memory import SharedMemory
import psutil,numpy as np
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_observed_day_extension_v1 as observed
import pickup_undecided_public_v1 as projector
import pickup_pending_transaction_v1 as engine
import pickup_partial_common_v1,pickup_partial_trace_v1,pickup_partial_known_v1,pickup_partial_validation_v1
import hard_admission,rolling_optimizer
from run_exp_rule_control import UncondDynamicReplanner
from energy_complete_saa import EnergyCompleteSAA
from scenario_saa import CondHistoricalSampler
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
import joint_install as ji
from pickup_loaded_closure_v1 import capture,sha
ROOT=old.ROOT;P=old.P;OUT=P/'resident_two_events_root_v1';SOURCE=P/'endogenous_owner_transaction_root_v2/OWNER_RESULT_0.json'
CONTROL=struct.Struct('<QdI');REPLY=struct.Struct('<QI32s')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def sources():
    files=[Path(__file__),ROOT/'tools/build_pickup_resident_audit_v1.py',ROOT/'tools/audit_pickup_resident_two_events_v1.py',ROOT/'tools/run_pickup_resident_two_events_bounded_v1.py',SOURCE,old.HISTORY,observed.fixture.TRACE,ROOT/'tools/pickup_loaded_closure_v1.py',ROOT/'tools/pickup_pending_transaction_v1.py',observed.P/'live_event_prefix_root_v1/REGISTRATION.json',observed.P/'undecided_public_input_root_v2/RESULT.json']
    return {str(p):sha(p) for p in files}
class Completed(Exception):pass
def worker():
    psutil.Process().cpu_affinity([7]);assert np.__version__=='1.26.4';encode,decode=old.codecs();ctrl=SharedMemory(name=sys.argv[2]);reply=SharedMemory(name=sys.argv[3]);template,dataset=observed.setup();initial=decode(read(SOURCE))['after_primary_advance']
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    records=[];events=[];current={};base=ji.transactional_class(C0ReserveReplanner)
    class Policy(base):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            started=time.perf_counter();deadline=current['deadline'];prior=self.export_state();pending=sorted((int(o) for o in visible if env.demands[i,o]>0 and not served[o] and o not in self._accepted|self._rejected),key=lambda o:(env.reveal_time[i,o],o))
            if len(pending)!=1:raise ValueError('registered single actual reveal boundary; preserve failure rather than silently filter')
            o=pending[0];packet=projector.export_undecided(env,i,clock,env.event_id,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,o)
            payload=pickle.dumps(packet,protocol=5)
            if len(payload)>2*1024*1024:raise ValueError('public packet bound')
            decoded=pickle.loads(payload);owner=ji.JointPolicyOwner(self,fleet,clock,env.event_id);token=owner.begin(str(o),o,deadline,old.pf.digest(decoded));response=engine.compute(decoded,history,old.E/'scripts',deadline)
            end_compute=capture(sources());response['source_start']=begin['files'];response['source_end']=end_compute['files'];response['source_qualified']=begin['files']==end_compute['files'] and begin['module_files']==end_compute['module_files'];save('SOURCE_COMPUTE_'+str(current['seq'])+'.json',end_compute);save('ATTEMPT_'+str(current['seq'])+'.json',encode(dict(packet=packet,response=response,before=prior)))
            installed,reason=owner.commit(token,response)
            if not installed:raise ValueError(reason)
            pointer=self._policy is response['compiled_state'];env._plan_and_commit(i,clock,fleet,traces,served,customers)
            current['clock']=clock
            record=dict(sequence=current['seq'],callback=15+current['seq'],order=o,clock=clock,event_id=env.event_id,packet=packet,before=prior,after=self.export_state(),response=response,actual_policy_pointer=pointer,started=started,committed=time.perf_counter(),private_after=copy.deepcopy(snapshot(clock)))
            records.append(record);save('REQUEST_'+str(current['seq'])+'.json',encode(record));raise Completed()
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            value=super()._advance_fleet(i,clock,fleet,traces,served);events.append(dict(sequence=current['seq'],clock=clock,event_id=self.event_id,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()));return value
    rp=Policy(template['budget'],template['capacity'],template['booking_horizon'],template['contract'],0.);rp.restore_state(copy.deepcopy(initial['state']));env=Env(copy.deepcopy(dataset),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],rp,template['contract'],template['booking_horizon']);env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];env.force_suffix={}
    fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy();clock=initial['clock'];customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0];reveals=sorted(set(float(env.reveal_time[0,j]) for j in customers if env.reveal_time[0,j]>0))
    def snapshot(clock):return dict(state=rp.export_state(),fleet=fleet,traces=traces,served=served,clock=clock,event_id=env.event_id,replan_count=env.replan_count)
    initial_hash=old.pf.digest(snapshot(clock));ji.JointPolicyOwner(rp,fleet,clock,env.event_id).signature();capture();begin=capture(sources());save('SOURCE_START.json',begin);print(json.dumps(dict(ready=True,pid=os.getpid(),baseline=initial_hash,files=len(begin['files']))),flush=True)
    last=0
    try:
        while True:
            seq,deadline,op=CONTROL.unpack_from(ctrl.buf)
            if not seq or seq==last:time.sleep(.001);continue
            last=seq
            if op==2:break
            if seq!=len(records)+1 or seq>2 or op!=1:raise ValueError('single-use ordered request')
            current.update(seq=seq,deadline=deadline);before=old.pf.digest(snapshot(clock));started=time.perf_counter()
            try:env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,reveals,float(env.tw_end[0].max()))
            except Completed:clock=current['clock']
            else:raise ValueError('registered next reveal absent')
            end=capture(sources());record=records[-1];record.update(resident_before=before,resident_after=old.pf.digest(snapshot(clock)),resident_started=started,resident_finished=time.perf_counter(),loaded_closure_qualified=begin['files']==end['files'] and begin['module_files']==end['module_files']);save('SOURCE_END_'+str(seq)+'.json',end);save('ACK_'+str(seq)+'.json',encode(record))
            if not record['loaded_closure_qualified'] or time.perf_counter()>=deadline:raise ValueError('closure or deadline after actual commit; preserve failed private branch')
            REPLY.pack_into(reply.buf,0,seq,1,bytes.fromhex(record['resident_after']))
        save('RESIDENT_RESULT.json',encode(dict(initial_digest=initial_hash,records=records,events=events,final=copy.deepcopy(snapshot(clock)),pid=os.getpid(),same_private_env=True)))
    except Exception:
        save('WORKER_FAILURE.json',dict(traceback=traceback.format_exc()));REPLY.pack_into(reply.buf,0,last,2,bytes(32))
    finally:ctrl.close();reply.close()
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False);save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),selection='only next two actual reveal callbacks after sealed callback15 physical event24; same resident private env continues; no callback15 replay',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,cpu=[7],numpy='1.26.4',initialization_outside_clock=True,method='own_uncond_dynamic current-only; own_incumbent; common original history K10/full C0/compile/guard; original private run_loop and plan_and_commit',faults_this_stage=False,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,formal_learned_controls=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);p=None
    try:
        with (OUT/'worker.stderr.txt').open('x',encoding='utf-8') as err:
            p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,reply.name],stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1'))
            ready=json.loads(p.stdout.readline());assert ready['ready'];host=[]
            for seq in (1,2):
                started=time.perf_counter();deadline=started+10.;CONTROL.pack_into(ctrl.buf,0,seq,deadline-2.,1)
                while time.perf_counter()<deadline-2.:
                    got,kind,digest=REPLY.unpack_from(reply.buf)
                    if got==seq:
                        if kind!=1:raise RuntimeError('resident failed')
                        break
                    if p.poll() is not None:raise RuntimeError('resident exited')
                    time.sleep(.001)
                else:raise TimeoutError('resident request deadline; no qualification')
                received=time.perf_counter();row=dict(sequence=seq,started=started,received=received,finished=received,wall_s=received-started,digest=digest.hex(),pid=p.pid,cap_s=10.,execution_s=8.,reserve_s=2.,same_resident=True,timed_parent_projection_pickle=False)
                if seq==2:
                    CONTROL.pack_into(ctrl.buf,0,3,deadline-2.,2);p.wait(timeout=max(.001,deadline-time.perf_counter()));row['finished']=time.perf_counter();row['wall_s']=row['finished']-started;row['final_reap_included']=True
                else:row['final_reap_included']=False
                host.append(row);save('HOST_'+str(seq)+'.json',row)
            encode,decode=old.codecs();actual=decode(read(OUT/'RESIDENT_RESULT.json'));checks=[]
            def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
            ck('two_actual_new_callbacks_same_owner',len(actual['records'])==2 and [r['callback'] for r in actual['records']]==[16,17] and all(r['pid']==p.pid for r in host) and actual['same_private_env'])
            ck('second_starts_at_first_actual_committed_state',actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
            for h,r in zip(host,actual['records']):
                n=str(h['sequence']);ck(n+':actual_published_private_identity',h['digest']==r['resident_after'] and r['actual_policy_pointer']);ck(n+':source_closure',r['loaded_closure_qualified'] and r['response']['source_qualified']);ck(n+':actual_request_time',h['started']<=r['resident_started']<=r['started']<=r['response']['worker_finished']<=r['committed']<=h['received']<=h['finished']<h['started']+10.);ck(n+':original_C0_20branches',r['response']['certified'] and sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20);ck(n+':no_model_or_route_change',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
            ck('own_resident_reaped',p.poll()==0);ck('registered_sources_unchanged',all(sha(p)==h for p,h in read(OUT/'REGISTRATION.json')['sources'].items()));fail=[c['name'] for c in checks if not c['pass_']]
            save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,ready=ready,two_observed_resident_component_qualified=not fail,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failures=fail,walls=[h['wall_s'] for h in host],orders=[r['order'] for r in actual['records']])));return int(bool(fail))
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc()));save('RUN_EXIT.json',dict(exit_code=1));raise
    finally:
        if p:
            if p.poll() is None:p.kill();p.wait(timeout=5)
            p.stdout.close()
        for m in (ctrl,reply):m.close();m.unlink()
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
