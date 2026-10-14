"""Two new endogenous reveals with archived checkpoint publication and lost ACK."""
import copy,ctypes,json,hashlib,os,pickle,struct,subprocess,sys,time,traceback
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
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
import joint_install as ji
from pickup_loaded_closure_v1 import capture,sha
import qualify_pickup_checkpoint_ack_v1 as events_api
ROOT=old.ROOT;OUT=old.P/'continuous_checkpoint_root_v2';INPUT=old.P/'checkpoint_bytes_retention_root_v1/ACTUAL_SHARED_PAYLOAD.bin'
CONTROL=struct.Struct('<QdII');REPLY=struct.Struct('<Q32s');HEADER=struct.Struct('<QQ32s32s32s');MAX=2*1024*1024
events_api.K.ResetEvent.argtypes=[ctypes.c_void_p];events_api.K.ResetEvent.restype=ctypes.c_int
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def raw_save(n,v):
    with (OUT/n).open('xb') as f:f.write(v)
def sources():
    paths=[ROOT/'tools/recover_pickup_continuous_checkpoint_v2.py',old.P/'continuous_checkpoint_root_v1/REGISTRATION.json',old.P/'continuous_checkpoint_root_v1/WORKER_FAILURE.json',old.P/'event27_settlement_diagnosis_root_v1/REGISTRATION.json',old.P/'event27_settlement_diagnosis_root_v1/RESULT.json',Path(__file__),ROOT/'tools/audit_pickup_continuous_checkpoint_v2.py',ROOT/'tools/pickup_pending_transaction_v1.py',ROOT/'tools/pickup_loaded_closure_v1.py',INPUT,old.HISTORY,observed.fixture.TRACE,old.P/'undecided_public_input_root_v2/RESULT.json',Path(sys.executable)]
    return {str(p):sha(p) for p in paths}
class Done(Exception):pass
def worker():
    psutil.Process().cpu_affinity([7]);assert np.__version__=='1.26.4';dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name','cmdline']))
    ctrl=SharedMemory(name=sys.argv[2]);blob=SharedMemory(name=sys.argv[3]);reply=SharedMemory(name=sys.argv[4]);latch=events_api.event(sys.argv[5]);recover='--recover' in sys.argv
    encode,decode=old.codecs();template,dataset=observed.setup();initial=pickle.loads(INPUT.read_bytes())['state'];history=pickle.loads(old.HISTORY.read_bytes());records=[];physical=[];current={};base=ji.transactional_class(C0ReserveReplanner)
    class Policy(base):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            started=time.perf_counter();prior=self.export_state();pending=sorted((int(o) for o in visible if env.demands[i,o]>0 and not served[o] and o not in self._accepted|self._rejected),key=lambda o:(env.reveal_time[i,o],o))
            if len(pending)!=1:raise ValueError('fixed single actual new reveal; no filtering')
            o=pending[0];packet=projector.export_undecided(env,i,clock,env.event_id,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,o);public=pickle.dumps(packet,protocol=5)
            if len(public)>MAX:raise ValueError('public packet bound')
            decoded=pickle.loads(public);owner=ji.JointPolicyOwner(self,fleet,clock,env.event_id);token=owner.begin(str(o),o,current['deadline'],old.pf.digest(decoded));response=engine.compute(decoded,history,old.E/'scripts',current['deadline'])
            mid=capture(sources());response.update(source_start=begin['files'],source_end=mid['files'],source_qualified=begin['files']==mid['files'] and begin['module_files']==mid['module_files']);save('SOURCE_COMPUTE_'+str(current['seq'])+'.json',mid);save('ATTEMPT_'+str(current['seq'])+'.json',encode(dict(packet=packet,response=response,before=prior)))
            installed,reason=owner.commit(token,response)
            if not installed:raise ValueError(reason)
            pointer=self._policy is response['compiled_state'];env._plan_and_commit(i,clock,fleet,traces,served,customers);current['clock']=clock
            records.append(dict(sequence=current['seq'],callback=17+current['seq'],order=o,clock=clock,event_id=env.event_id,packet=packet,before=prior,after=self.export_state(),response=response,actual_policy_pointer=pointer,started=started,committed=time.perf_counter(),private_after=copy.deepcopy(snapshot(clock))))
            raise Done()
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            v=super()._advance_fleet(i,clock,fleet,traces,served);physical.append(dict(sequence=current['seq'],clock=clock,event_id=self.event_id,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()));return v
    def restore(s):
        nonlocal rp,env,fleet,traces,served,clock,customers,reveals
        rp=Policy(template['budget'],template['capacity'],template['booking_horizon'],template['contract'],0.);rp.restore_state(copy.deepcopy(s['state']));env=Env(copy.deepcopy(dataset),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],rp,template['contract'],template['booking_horizon']);env.event_id=s['event_id'];env.replan_count=s['replan_count'];env.force_suffix={};fleet=copy.deepcopy(s['fleet']);traces=copy.deepcopy(s['traces']);served=s['served'].copy();clock=s['clock'];customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0];reveals=sorted(set(float(env.reveal_time[0,j]) for j in customers if env.reveal_time[0,j]>0))
    def snapshot(c):return dict(state=rp.export_state(),fleet=fleet,traces=traces,served=served,clock=c,event_id=env.event_id,replan_count=env.replan_count)
    rp=env=fleet=traces=served=clock=customers=reveals=None
    # Warm restore/serialization/hash primitives before request closure; no events.
    restore(initial);baseline=old.pf.digest(snapshot(clock));ji.JointPolicyOwner(rp,fleet,clock,env.event_id).signature();pickle.dumps(initial,protocol=5);capture();begin=capture(sources());prefix='RECOVERY_' if recover else '';save(prefix+'SOURCE_START.json',begin);print(json.dumps(dict(ready=True,pid=os.getpid(),baseline=baseline)),flush=True)
    last=0
    try:
        while True:
            seq,deadline,op,fault=CONTROL.unpack_from(ctrl.buf)
            if not seq or seq==last:time.sleep(.001);continue
            last=seq;started=time.perf_counter();current.update(seq=seq,deadline=deadline)
            if recover:
                if op!=3 or seq!=2:raise ValueError('fixed already-committed ticket only')
                epoch,size,basehash,ph,sh=HEADER.unpack_from(blob.buf);payload=bytes(blob.buf[HEADER.size:HEADER.size+size])
                if epoch!=seq or size>MAX or hashlib.sha256(payload).digest()!=ph:raise ValueError('actual checkpoint bytes')
                c=pickle.loads(payload)
                if c['epoch']!=seq or c['base_digest']!=basehash.hex():raise ValueError('checkpoint lineage')
                restore(c['state']);target=old.pf.digest(snapshot(clock))
                if target!=sh.hex():raise ValueError('actual restored owner state')
                end=capture(sources());save('RECOVERY_SOURCE_END.json',end);save('RECOVERY.json',encode(dict(state=snapshot(clock),epoch=seq,digest=target,duplicate_digest=old.pf.digest(snapshot(clock)),physical_calls=0,admission_calls=0,started=started,finished=time.perf_counter(),loaded_closure_qualified=begin['files']==end['files'] and begin['module_files']==end['module_files'],payload_sha256=ph.hex())))
                if time.perf_counter()>=deadline or begin['files']!=end['files'] or begin['module_files']!=end['module_files']:raise ValueError('recovery closure/deadline')
                REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(target));break
            if op!=1 or seq!=len(records)+1 or seq>2:raise ValueError('fixed two single-use new requests')
            before=old.pf.digest(snapshot(clock))
            # Sealed source stops immediately after event27 physical advance.
            # Complete that original boundary once inside first request clock.
            if seq==1:env._plan_and_commit(0,clock,fleet,traces,served,customers)
            try:env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,reveals,float(env.tw_end[0].max()))
            except Done:clock=current['clock']
            else:raise ValueError('new reveal absent')
            after=old.pf.digest(snapshot(clock));checkpoint=dict(epoch=seq,base_digest=before,state=copy.deepcopy(snapshot(clock)));payload=pickle.dumps(checkpoint,protocol=5)
            if len(payload)>MAX:raise ValueError('full checkpoint bound')
            raw_save('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin',payload)
            end=capture(sources());qualified=begin['files']==end['files'] and begin['module_files']==end['module_files'];save('SOURCE_END_'+str(seq)+'.json',end)
            records[-1].update(resident_before=before,resident_after=after,resident_started=started,resident_finished=time.perf_counter(),loaded_closure_qualified=qualified,payload_size=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest());save('REQUEST_'+str(seq)+'.json',encode(records[-1]));save('OWNER_STATE_'+str(seq)+'.json',encode(dict(initial_digest=baseline,records=records,events=physical,final=copy.deepcopy(snapshot(clock)),pid=os.getpid())))
            if not qualified or time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('unqualified private branch not published')
            blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,seq,len(payload),bytes.fromhex(before),hashlib.sha256(payload).digest(),bytes.fromhex(after));events_api.K.SetEvent(latch);struct.pack_into('<I',ctrl.buf,40,2)
            if fault==2:time.sleep(12.)
            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('ACK fenced')
            REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(after))
    except Exception:
        save(prefix+'WORKER_FAILURE.json',dict(traceback=traceback.format_exc()));raise
    finally:
        for m in (ctrl,blob,reply):m.close()
        events_api.K.CloseHandle(latch)
def launch(ctrl,blob,reply,name,recover=False):
    err=(OUT/('recovery.stderr.txt' if recover else 'worker.stderr.txt')).open('x',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,blob.name,reply.name,name]+(['--recover'] if recover else []),stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'));ready=json.loads(p.stdout.readline());assert ready['ready'];return p,err,ready
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False);save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='Engineering recovery: complete original event27 plan_and_commit once inside first request window (source was after physical advance only); preserve v1 failure. Same input/method/K10/budget; only next two actual new reveals callback18/19 from sealed event27, original private run_loop/admission/common K10/C0/compile/guard/plan; same resident before lost ACK after second committed checkpoint publication. Archive actual raw bytes. Restore selected second shared checkpoint in separate10s recovery; recognize duplicate seq2 with zero events/admissions. No old event27/callback15-17 or previous QA replay.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,fault_delay_s=12.,initialization_outside_clock=True,recovery_initial_runtime_outside_clock=True,cpu=[7],numpy='1.26.4',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=HEADER.size+MAX);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);name='Local\\MaskCO_continuous_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        p,err,ready=launch(ctrl,blob,reply,name);authority=ready['baseline'];host=[]
        for seq in (1,2):
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,1,2 if seq==2 else 0);ticket=None;ack=False;chosen_at=None
            while time.perf_counter()<cutoff:
                if ticket is None and events_api.signaled(latch):
                    epoch,size,base,ph,sh=HEADER.unpack_from(blob.buf)
                    if epoch!=seq or size>MAX or base.hex()!=authority:raise ValueError('host epoch/base/bound')
                    ticket=dict(epoch=epoch,size=size,base_digest=base.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter();authority=sh.hex()
                got,dh=REPLY.unpack_from(reply.buf)
                if got==seq:ack=dh.hex()==authority;break
                if p.poll() is not None:raise ValueError('worker unexpectedly exited')
                time.sleep(.001)
            if ticket is None:raise TimeoutError('no qualified checkpoint before cutoff')
            if seq==2:
                struct.pack_into('<I',ctrl.buf,44,1);p.kill();p.wait(timeout=max(.001,deadline-time.perf_counter()))
            # Freeze completed blob bytes after writer stops using this seq.
            raw=bytes(blob.buf[:HEADER.size+ticket['size']]);raw_save('ACTUAL_SHARED_ENVELOPE_'+str(seq)+'.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin',raw[HEADER.size:]);f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,ack_present=ack,final_reap_included=seq==2,reaped=p.poll() is not None,stage=struct.unpack_from('<I',ctrl.buf,40)[0]);host.append(row);save('HOST_'+str(seq)+'.json',row)
        p.stdout.close();err.close();p=err=None;ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);p,err,_=launch(ctrl,blob,reply,name,True)
        s=time.perf_counter();CONTROL.pack_into(ctrl.buf,0,2,s+8.,3,0);p.wait(timeout=8.);f=time.perf_counter();got,dh=REPLY.unpack_from(reply.buf);rh=dict(pid=p.pid,started=s,finished=f,wall_s=f-s,digest=dh.hex(),sequence=got,exit_code=p.returncode,reaped=True);save('RECOVERY_HOST.json',rh)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_2.json'));recovery=decode(read(OUT/'RECOVERY.json'));checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        ck('new_callbacks18_19_same_resident',len(actual['records'])==2 and [x['callback'] for x in actual['records']]==[18,19] and host[0]['pid']==host[1]['pid'])
        ck('second_starts_from_first_committed_state',actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
        for h,r in zip(host,actual['records']):
            seq=h['sequence'];ck(str(seq)+':raw_shared_writer_exact',(OUT/('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin')).read_bytes()==(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin')).read_bytes());ck(str(seq)+':qualified_actual_commit',r['actual_policy_pointer'] and r['loaded_closure_qualified'] and r['response']['source_qualified']);ck(str(seq)+':within_total_and_before_cutoff',h['wall_s']<10. and h['chosen_at']<h['cutoff']);ck(str(seq)+':20_original_branches',sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20 and r['response']['certified']);ck(str(seq)+':no_learned_model',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
        ck('post_commit_lost_ACK_second_killed_reaped',not host[1]['ack_present'] and host[1]['stage']==2 and host[1]['reaped'] and host[0]['ack_present'])
        ck('actual_restored_selected_checkpoint',recovery['digest']==rh['digest']==authority and old.pf.digest(recovery['state'])==authority)
        ck('duplicate_second_ticket_no_event_or_admission',recovery['physical_calls']==recovery['admission_calls']==0 and recovery['duplicate_digest']==authority and rh['sequence']==2)
        ck('recovery_closure_budget_reap',recovery['loaded_closure_qualified'] and rh['wall_s']<10. and rh['exit_code']==0)
        ck('registered_bytes',all(sha(x)==v for x,v in read(OUT/'REGISTRATION.json')['sources'].items()));fail=[x['name'] for x in checks if not x['pass_']]
        save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,recovery_host=rh,new_component_runtime_qualified=not fail,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[x['wall_s'] for x in host],orders=[x['order'] for x in actual['records']],recovery_wall=rh['wall_s'])));return int(bool(fail))
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc()));save('RUN_EXIT.json',dict(exit_code=1));raise
    finally:
        if p:
            if p.poll() is None:p.kill();p.wait(timeout=2.)
            p.stdout.close()
        if err:err.close()
        events_api.K.CloseHandle(latch)
        for m in (ctrl,blob,reply):m.close();m.unlink()
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
