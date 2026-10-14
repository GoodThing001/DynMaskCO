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
ROOT=old.ROOT;BASE_OUT=old.P/'restore_compute_cancel_root_v1';OUT=BASE_OUT;INPUT=old.P/'timed_restore_live_root_v1/ACTUAL_SHARED_PAYLOAD_5.bin';WARM=old.P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin'
CONTROL=struct.Struct('<QdII');REPLY=struct.Struct('<Q32s');HEADER=struct.Struct('<QQ32s32s32s');MAX=2*1024*1024
events_api.K.ResetEvent.argtypes=[ctypes.c_void_p];events_api.K.ResetEvent.restype=ctypes.c_int
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def raw_save(n,v):
    with (OUT/n).open('xb') as f:f.write(v)
def sources():
    paths=[ROOT/'tools/build_pickup_restore_compute_cancel_v1.py',old.P/'timed_restore_live_root_v1/REGISTRATION.json',old.P/'timed_restore_live_root_v1/INDEPENDENT_TIMED_RESTORE_LIVE.json',WARM,old.P/'continuous_checkpoint_root_v1/REGISTRATION.json',old.P/'continuous_checkpoint_root_v1/WORKER_FAILURE.json',old.P/'event27_settlement_diagnosis_root_v1/REGISTRATION.json',old.P/'event27_settlement_diagnosis_root_v1/RESULT.json',Path(__file__),ROOT/'tools/audit_pickup_restore_compute_cancel_v1.py',ROOT/'tools/pickup_pending_transaction_v1.py',ROOT/'tools/pickup_loaded_closure_v1.py',INPUT,old.HISTORY,observed.fixture.TRACE,old.P/'undecided_public_input_root_v2/RESULT.json',Path(sys.executable)]
    return {str(p):sha(p) for p in paths}
class Done(Exception):pass
def worker():
    global OUT
    OUT=BASE_OUT/('CASE_'+sys.argv[sys.argv.index('--case')+1])
    psutil.Process().cpu_affinity([7]);assert np.__version__=='1.26.4';dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name','cmdline']))
    ctrl=SharedMemory(name=sys.argv[2]);blob=SharedMemory(name=sys.argv[3]);reply=SharedMemory(name=sys.argv[4]);latch=events_api.event(sys.argv[5]);recover='--recover' in sys.argv
    encode,decode=old.codecs();template,dataset=observed.setup();initial=pickle.loads(WARM.read_bytes())['state'];history=pickle.loads(old.HISTORY.read_bytes());records=[];physical=[];current={};base=ji.transactional_class(C0ReserveReplanner)
    class Policy(base):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            started=time.perf_counter();prior=self.export_state();pending=sorted((int(o) for o in visible if env.demands[i,o]>0 and not served[o] and o not in self._accepted|self._rejected),key=lambda o:(env.reveal_time[i,o],o))
            if len(pending)!=1:raise ValueError('fixed single actual new reveal; no filtering')
            o=pending[0];packet=projector.export_undecided(env,i,clock,env.event_id,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,o);public=pickle.dumps(packet,protocol=5)
            if len(public)>MAX:raise ValueError('public packet bound')
            decoded=pickle.loads(public);owner=ji.JointPolicyOwner(self,fleet,clock,env.event_id);token=owner.begin(str(o),o,current['deadline'],old.pf.digest(decoded));response=engine.compute(decoded,history,old.E/'scripts',current['deadline'])
            mid=capture(sources());response.update(source_start=begin['files'],source_end=mid['files'],source_qualified=begin['files']==mid['files'] and begin['module_files']==mid['module_files']);save('SOURCE_COMPUTE_'+str(current['seq'])+'.json',mid);save('ATTEMPT_'+str(current['seq'])+'.json',encode(dict(packet=packet,response=response,before=prior)))
            pause(2,dict(packet=packet,response=response,before=prior,private_at_reveal=copy.deepcopy(snapshot(clock)),events=copy.deepcopy(physical)))
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
        rp=Policy(template['budget'],template['capacity'],template['booking_horizon'],template['contract'],0.);rp.restore_state(copy.deepcopy(s['state']));
        if current.get('restoring_selected'):pause(1,dict(selected_state=copy.deepcopy(s),restored_replanner=rp.export_state(),environment_restore_complete=False,physical_calls=0,admission_calls=0))
        env=Env(copy.deepcopy(dataset),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],rp,template['contract'],template['booking_horizon']);env.event_id=s['event_id'];env.replan_count=s['replan_count'];env.force_suffix={};fleet=copy.deepcopy(s['fleet']);traces=copy.deepcopy(s['traces']);served=s['served'].copy();clock=s['clock'];customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0];reveals=sorted(set(float(env.reveal_time[0,j]) for j in customers if env.reveal_time[0,j]>0))
    def snapshot(c):return dict(state=rp.export_state(),fleet=fleet,traces=traces,served=served,clock=c,event_id=env.event_id,replan_count=env.replan_count)
    def pause(stage,details):
        if current.get('fault')!=stage:return
        end=capture(sources());save('SOURCE_AT_FAULT.json',end)
        witness=dict(stage=stage,sequence=current['seq'],deadline=current['deadline'],reached=time.perf_counter(),details=details,loaded_closure_qualified=begin['files']==end['files'] and begin['module_files']==end['module_files'])
        save('FAULT_WITNESS.json',encode(witness));struct.pack_into('<I',ctrl.buf,40,stage);time.sleep(12.)
        raise TimeoutError('registered fault delay ended; publication forbidden')
    rp=env=fleet=traces=served=clock=customers=reveals=None
    # Warm restore/serialization/hash primitives before request closure; no events.
    restore(initial);baseline=old.pf.digest(snapshot(clock));ji.JointPolicyOwner(rp,fleet,clock,env.event_id).signature();pickle.dumps(initial,protocol=5);capture();begin=capture(sources());prefix='RECOVERY_' if recover else '';save(prefix+'SOURCE_START.json',begin);print(json.dumps(dict(ready=True,pid=os.getpid(),baseline=baseline)),flush=True)
    last=0
    try:
        while True:
            seq,deadline,op,fault=CONTROL.unpack_from(ctrl.buf)
            if not seq or seq==last:time.sleep(.001);continue
            last=seq;started=time.perf_counter();current.update(seq=seq,deadline=deadline,fault=fault)
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
            if op!=1 or seq!=6 or fault not in (1,2):raise ValueError('fixed next request and two faults')
            if seq==6:
                rs=time.perf_counter();epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf)
                if epoch!=5 or not 0<size<=MAX:raise ValueError('selected actual epoch3 bound')
                payload=bytes(blob.buf[HEADER.size:HEADER.size+size])
                if hashlib.sha256(payload).digest()!=ph:raise ValueError('selected actual raw SHA')
                cp=pickle.loads(payload)
                if cp['epoch']!=epoch or cp['base_digest']!=bh.hex():raise ValueError('selected actual lineage')
                warm_digest=old.pf.digest(snapshot(clock));current['restoring_selected']=True;restore(cp['state']);current['restoring_selected']=False;target=old.pf.digest(snapshot(clock))
                if target!=sh.hex():raise ValueError('selected actual private restoration')
                baseline=target
                save('RESTORE_IN_REQUEST.json',encode(dict(sequence=seq,source_epoch=epoch,source_payload_sha256=ph.hex(),warm_digest=warm_digest,selected_digest=target,state=copy.deepcopy(snapshot(clock)),started=rs,finished=time.perf_counter(),admission_calls=0,physical_calls=0)))
            before=old.pf.digest(snapshot(clock))
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
            if seq==5:break
    except Exception:
        save(prefix+'WORKER_FAILURE.json',dict(traceback=traceback.format_exc()));raise
    finally:
        for m in (ctrl,blob,reply):m.close()
        events_api.K.CloseHandle(latch)
def launch(ctrl,blob,reply,name,recover=False):
    err=(OUT/('recovery.stderr.txt' if recover else 'worker.stderr.txt')).open('x',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,blob.name,reply.name,name]+(['--recover'] if recover else []),stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'));ready=json.loads(p.stdout.readline());assert ready['ready'];return p,err,ready
def main():
    global OUT
    psutil.Process().cpu_affinity([7]);BASE_OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='Two fixed new copies of selected actual callback22/event40/epoch5 raw checkpoint. First timed restore paused after original replanner restore before actual env/fleet restore completes. Second selected restore completes and original next real reveal callback23 computes current-only admission/commonK10/all20branches/fullC0/compile, pauses before original owner.commit and real plan. Each12s delay,10s total/8s execution/2s reserve, kill/reap retains original sealed ticket/raw bytes. No normal request/old cancellation QA rerun, no checkpoint publication, no new science/models. Only simulation private branches, no external irreversible actions.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,fault_delay_s=12.,stages=[1,2],warm_prior_epoch3_outside_clock=True,selected_restore_inside_clock=True,cpu=[7],numpy='1.26.4',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    payload=INPUT.read_bytes();cp=pickle.loads(payload);assert cp['epoch']==5 and len(payload)<=MAX;authority=old.pf.digest(cp['state']);host=[];checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    for fault in (1,2):
        OUT=BASE_OUT/('CASE_'+str(fault));OUT.mkdir(exist_ok=False);ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=HEADER.size+MAX);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,5,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority));initial=bytes(blob.buf[:HEADER.size+len(payload)]);raw_save('INITIAL_SHARED_ENVELOPE.bin',initial);name='Local\\MaskCO_restore_cancel_'+ctrl.name;latch=events_api.event(name,True);p=err=None
        try:
            err=(OUT/'worker.stderr.txt').open('x',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,blob.name,reply.name,name,'--case',str(fault)],stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'));ready=json.loads(p.stdout.readline());assert ready['ready']
            s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,6,cutoff,1,fault)
            while time.perf_counter()<cutoff:
                if p.poll() is not None:raise ValueError('worker exited before planned cancellation')
                if events_api.signaled(latch) or REPLY.unpack_from(reply.buf)[0]:raise ValueError('unexpected publication')
                time.sleep(.001)
            reached=struct.unpack_from('<I',ctrl.buf,40)[0];struct.pack_into('<I',ctrl.buf,44,1);p.kill();p.wait(timeout=max(.001,deadline-time.perf_counter()));retained=bytes(blob.buf[:HEADER.size+len(payload)]);raw_save('RETAINED_SHARED_ENVELOPE.bin',retained);f=time.perf_counter();row=dict(fault=fault,stage=reached,sequence=6,pid=p.pid,started=s,cutoff=cutoff,finished=f,wall_s=f-s,exit_code=p.returncode,reaped=True,publication_signaled=events_api.signaled(latch),reply_sequence=REPLY.unpack_from(reply.buf)[0],authority_digest=authority,retained_payload_sha256=hashlib.sha256(retained[HEADER.size:]).hexdigest());save('HOST.json',row);host.append(row)
            _,decode=old.codecs();w=decode(read(OUT/'FAULT_WITNESS.json'))
            ck(str(fault)+':actual_fixed_stage_reached',reached==w['stage']==fault and s<=w['reached']<cutoff)
            ck(str(fault)+':original_shared_bytes_and_ticket_preserved',retained==initial and row['authority_digest']==old.pf.digest(cp['state']) and row['retained_payload_sha256']==sha(INPUT))
            ck(str(fault)+':no_publication_reply_or_new_owner',not row['publication_signaled'] and row['reply_sequence']==0 and not (OUT/'OWNER_STATE_6.json').exists())
            ck(str(fault)+':cancelled_reaped_in10',row['wall_s']<10. and row['reaped'] and p.poll() is not None)
            ck(str(fault)+':actual_loaded_fault_segment_identity',w['loaded_closure_qualified'])
            if fault==1:ck('partial_restore_no_admission_physical',w['details']['physical_calls']==w['details']['admission_calls']==0 and not w['details']['environment_restore_complete'])
            else:ck('actual_next_compute20branches_certified',w['details']['response']['certified'] and sum(len(v['branch_values']) for v in w['details']['response']['common_scores']['candidate_scores'].values())==20 and w['details']['packet']['admission_input']['current_order']==w['details']['response']['scoring_packet']['admission_input']['current_order'])
        finally:
            if p:
                if p.poll() is None:p.kill();p.wait(timeout=2.)
                p.stdout.close()
            if err:err.close()
            events_api.K.CloseHandle(latch)
            for m in (ctrl,blob,reply):m.close();m.unlink()
    OUT=BASE_OUT;ck('registered_bytes',all(sha(p)==h for p,h in read(OUT/'REGISTRATION.json')['sources'].items()));fail=[x['name'] for x in checks if not x['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[h['wall_s'] for h in host])));return int(bool(fail))
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
