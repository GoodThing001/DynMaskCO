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
ROOT=old.ROOT;OUT=old.P/'event_loss_resume_root_v1';INPUT=old.P/'selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin';WARM=old.P/'prepared_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_8.bin'
CONTROL=struct.Struct('<QdII');REPLY=struct.Struct('<Q32s');HEADER=struct.Struct('<QQ32s32s32s');MAX=2*1024*1024;SLOT=HEADER.size+MAX
events_api.K.ResetEvent.argtypes=[ctypes.c_void_p];events_api.K.ResetEvent.restype=ctypes.c_int
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def raw_save(n,v):
    with (OUT/n).open('xb') as f:f.write(v)
def sources():
    paths=[ROOT/'tools/build_pickup_event_loss_resume_v1.py',old.P/'checkpoint_resume_cancel_root_v1/REGISTRATION.json',old.P/'checkpoint_resume_cancel_root_v1/INDEPENDENT_CHECKPOINT_RESUME_CANCEL.json',WARM,old.P/'continuous_checkpoint_root_v1/REGISTRATION.json',old.P/'continuous_checkpoint_root_v1/WORKER_FAILURE.json',old.P/'event27_settlement_diagnosis_root_v1/REGISTRATION.json',old.P/'event27_settlement_diagnosis_root_v1/RESULT.json',Path(__file__),ROOT/'tools/audit_pickup_event_loss_resume_v1.py',ROOT/'tools/pickup_pending_transaction_v1.py',ROOT/'tools/pickup_loaded_closure_v1.py',INPUT,old.HISTORY,observed.fixture.TRACE,old.P/'undecided_public_input_root_v2/RESULT.json',Path(sys.executable)]
    return {str(p):sha(p) for p in paths}
class Done(Exception):pass
def worker():
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
    last=0;restored_once=False
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
            if not restored_once:
                rs=time.perf_counter();epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf)
                if epoch!=10 or not 0<size<=MAX:raise ValueError('selected actual epoch10 bound')
                payload=bytes(blob.buf[HEADER.size:HEADER.size+size])
                if hashlib.sha256(payload).digest()!=ph:raise ValueError('selected actual raw SHA')
                cp=pickle.loads(payload)
                if cp['epoch']!=epoch or cp['base_digest']!=bh.hex():raise ValueError('selected actual lineage')
                warm_digest=old.pf.digest(snapshot(clock));restore(cp['state']);target=old.pf.digest(snapshot(clock))
                if target!=sh.hex():raise ValueError('selected actual private restoration')
                baseline=target
                save('RESTORE_IN_REQUEST.json',encode(dict(sequence=seq,source_epoch=epoch,source_payload_sha256=ph.hex(),warm_digest=warm_digest,selected_digest=target,state=copy.deepcopy(snapshot(clock)),started=rs,finished=time.perf_counter(),admission_calls=0,physical_calls=0)))
                restored_once=True
            if op==2:
                if seq!=10 or records or physical:raise ValueError('only restored selected10 duplicate allowed')
                before=old.pf.digest(snapshot(clock));end=capture(sources());qualified=begin['files']==end['files'] and begin['module_files']==end['module_files']
                save('DUPLICATE_SELECTED10.json',encode(dict(epoch=seq,state=copy.deepcopy(snapshot(clock)),digest=before,physical_calls=len(physical),admission_calls=len(records),started=started,finished=time.perf_counter(),loaded_closure_qualified=qualified)))
                save('DUPLICATE_SOURCE_END.json',end)
                if not qualified or time.perf_counter()>=deadline:raise ValueError('duplicate deadline/closure')
                REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(before));continue
            if op!=1 or seq!=len(records)+11 or seq>12:raise ValueError('fixed two new continued requests')
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
            active=struct.unpack_from('<I',ctrl.buf,48)[0]
            if active not in (0,1):raise ValueError('selected slot bound')
            inactive=1-active;offset=inactive*SLOT
            blob.buf[offset+HEADER.size:offset+HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,offset,seq,len(payload),bytes.fromhex(before),hashlib.sha256(payload).digest(),bytes.fromhex(after))
            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('before selector deadline fence')
            selector_started=time.perf_counter();struct.pack_into('<I',ctrl.buf,48,inactive)
            events_api.K.SetEvent(latch);struct.pack_into('<I',ctrl.buf,40,2)
            if fault==2:time.sleep(12.)
            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('ACK fenced')
            REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(after))
            if seq==12:break
    except Exception:
        save(prefix+'WORKER_FAILURE.json',dict(traceback=traceback.format_exc()));raise
    finally:
        for m in (ctrl,blob,reply):m.close()
        events_api.K.CloseHandle(latch)
def launch(ctrl,blob,reply,name,recover=False):
    err=(OUT/('recovery.stderr.txt' if recover else 'worker.stderr.txt')).open('x',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,blob.name,reply.name,name]+(['--recover'] if recover else []),stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'));ready=json.loads(p.stdout.readline());assert ready['ready'];return p,err,ready
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='New engineering: restore actual caller-selected epoch10/callback27 raw35318 checkpoint selected despite absent event/ACK in prior stage. Inside first10s window validate SHA/epoch/base/pickle/all private fields, duplicate epoch10 returns committed digest without new admission/scoring/physics/publishing. Same restored resident continues exactly next callback28 and29 once each, orders learned only at actual reveal, current-only admission/own_incumbent/commonK10/fullC0/compile/guard/plan/checkpoint. No replay callback27 or previous faults. Poll selector with bound/lineage/SHA/stability before8s; normal ACK, final graceful exit/reap in last10s window. No models/new science/external irreversible effects. Initial shared/library/history/data/warm older8 environment outside windows; no arbitrary parent/native/disk/persistence qualification.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,slot_count=2,cpu=[7],numpy='1.26.4',selected_checkpoint_restore_inside_first_request=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=2*SLOT);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[:]=bytes(2*SLOT);name='Local\\MaskCO_resume_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        payload=INPUT.read_bytes();cp=pickle.loads(payload);authority=old.pf.digest(cp['state']);assert cp['epoch']==10 and len(payload)<=MAX
        blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,10,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority));initial=bytes(blob.buf[:HEADER.size+len(payload)]);raw_save('INITIAL_SLOT0_ENVELOPE.bin',initial)
        p,err,ready=launch(ctrl,blob,reply,name)
        s=time.perf_counter();deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,10,s+8.,2,0)
        while time.perf_counter()<s+8.:
            got,dh=REPLY.unpack_from(reply.buf)
            if got==10:break
            if p.poll() is not None:raise ValueError('restore worker unexpectedly exited')
            time.sleep(.001)
        else:raise TimeoutError('selected10 duplicate absent')
        if dh.hex()!=authority or bytes(blob.buf[:len(initial)])!=initial or struct.unpack_from('<I',ctrl.buf,48)[0]!=0:raise ValueError('restored duplicate changed authority')
        restore_host=dict(started=s,finished=time.perf_counter(),cutoff=s+8.,epoch=10,digest=dh.hex(),pid=p.pid,reaped=False,shared_envelope_unchanged=True);restore_host['wall_s']=restore_host['finished']-s;save('RESTORE_HOST.json',restore_host)
        host=[]
        for seq in (11,12):
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,1,0);ticket=None;ack=False;chosen_at=None;chosen_bytes=None;validation=None
            while time.perf_counter()<cutoff:
                if ticket is None:
                    slot=struct.unpack_from('<I',ctrl.buf,48)[0]
                    if slot not in (0,1):raise ValueError('parent slot bound')
                    epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,slot*SLOT)
                    if epoch==seq:
                        if not 0<size<=MAX or bh.hex()!=authority:raise ValueError('host current lineage/size')
                        vs=time.perf_counter();off=slot*SLOT;candidate=bytes(blob.buf[off:off+HEADER.size+size]);cp_sha=hashlib.sha256(candidate[HEADER.size:]).digest();slot_after=struct.unpack_from('<I',ctrl.buf,48)[0];event_present=events_api.signaled(latch);vf=time.perf_counter()
                        if cp_sha!=ph or slot_after!=slot or vf>=cutoff:raise ValueError('parent payload SHA/stable selector/cutoff')
                        validation=dict(started=vs,finished=vf,slot_before=slot,slot_after=slot_after,payload_sha256=cp_sha.hex(),event_present=event_present,sequence=seq,base_digest=bh.hex());ticket=dict(epoch=epoch,size=size,slot=slot,base_digest=bh.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter()
                        if chosen_at>=cutoff:raise ValueError('parent choice cutoff')
                        authority=sh.hex();chosen_bytes=candidate
                got,dh=REPLY.unpack_from(reply.buf)
                if got==seq:ack=ticket is not None and dh.hex()==authority;break
                if p.poll() is not None:raise ValueError('worker unexpectedly exited')
                time.sleep(.001)
            if ticket is None or not ack:raise TimeoutError('normal checkpoint absent')
            if seq==12:p.wait(timeout=max(.001,deadline-time.perf_counter()))
            slot=struct.unpack_from('<I',ctrl.buf,48)[0];off=slot*SLOT;epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,off);raw=bytes(blob.buf[off:off+HEADER.size+size])
            if raw!=chosen_bytes or slot!=ticket['slot']:raise ValueError('chosen raw envelope changed')
            raw_save('ACTUAL_SHARED_ENVELOPE_'+str(seq)+'.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin',raw[HEADER.size:])
            if seq==12:raw_save('RETAINED_ENVELOPE.bin',raw);raw_save('RETAINED_PAYLOAD.bin',raw[HEADER.size:])
            f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,validation=validation,ack_present=ack,event_signaled_final=events_api.signaled(latch),final_reap_included=seq==12,reaped=p.poll() is not None,stage=struct.unpack_from('<I',ctrl.buf,40)[0],selected_slot=slot);host.append(row);save('HOST_'+str(seq)+'.json',row)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_12.json'));restored=decode(read(OUT/'RESTORE_IN_REQUEST.json'));dup=decode(read(OUT/'DUPLICATE_SELECTED10.json'));checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        ck('selected10_restore_duplicate_inside_window',restore_host['started']<=restored['started']<=restored['finished']<=dup['finished']<=restore_host['finished']<restore_host['cutoff'] and restored['selected_digest']==dup['digest']==actual['initial_digest']==old.pf.digest(cp['state']))
        ck('restore_duplicate_zero_execution',restored['admission_calls']==restored['physical_calls']==dup['admission_calls']==dup['physical_calls']==0 and dup['loaded_closure_qualified'])
        ck('duplicate_all_private_fields',old.pf.digest(dup['state'])==old.pf.digest(cp['state']) and restore_host['shared_envelope_unchanged'])
        ck('new28_29_same_restored_resident',[r['callback'] for r in actual['records']]==[28,29] and restore_host['pid']==host[0]['pid']==host[1]['pid'])
        ck('second_starts_first_actual_commit',actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
        for index,r in enumerate(actual['records']):
            h=host[index];seq=index+11
            ck(str(index)+':qualified_C0_install',r['actual_policy_pointer'] and r['loaded_closure_qualified'] and r['response']['source_qualified'])
            ck(str(index)+':all20_branches',sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20 and r['response']['certified'])
            ck(str(index)+':no_model_route_gain',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
            ck(str(index)+':selected_raw_writer_exact',(OUT/('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin')).read_bytes()==(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin')).read_bytes() and h['ticket']['digest']==r['resident_after'])
            ck(str(index)+':normal_bounded_selection',h['ack_present'] and h['event_signaled_final'] and h['validation']['finished']<=h['chosen_at']<h['cutoff'] and h['selected_slot']==1-index)
        ck('normal12_final_graceful_reap',host[1]['reaped'] and p.returncode==0 and not host[0]['reaped'])
        ck('three_separate_windows_under10',restore_host['wall_s']<10. and all(h['wall_s']<10. for h in host))
        st=read(OUT/'SOURCE_START.json');end=read(OUT/'SOURCE_END_12.json');ck('start_to_normal_end_closure',st['files']==end['files'] and st['module_files']==end['module_files'])
        ck('registered_bytes',all(sha(pth)==h for pth,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        fail=[x['name'] for x in checks if not x['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,restore_host=restore_host,actual_worker_exit=p.returncode,actual_worker_terminated=False,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[restore_host['wall_s']]+[h['wall_s'] for h in host],orders=[r['order'] for r in actual['records']])));return int(bool(fail))
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
