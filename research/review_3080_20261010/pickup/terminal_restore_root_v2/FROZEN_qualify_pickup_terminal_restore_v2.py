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
from coldchain_evaluator_a1 import evaluate_trace_a1
from run_exp_energy_c0 import C0ReserveReplanner
import joint_install as ji
from pickup_loaded_closure_v1 import capture,sha
import qualify_pickup_checkpoint_ack_v1 as events_api
ROOT=old.ROOT;OUT=old.P/'terminal_restore_root_v2';INPUT=old.P/'resident_day_tail_root_v1/ACTUAL_SHARED_PAYLOAD_63.bin';WARM=old.P/'lineage_header_cancel_root_v1/RETAINED_PAYLOAD.bin'
CONTROL=struct.Struct('<QdII');REPLY=struct.Struct('<Q32s');HEADER=struct.Struct('<QQ32s32s32s');MAX=2*1024*1024;SLOT=HEADER.size+MAX
events_api.K.ResetEvent.argtypes=[ctypes.c_void_p];events_api.K.ResetEvent.restype=ctypes.c_int
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def raw_save(n,v):
    with (OUT/n).open('xb') as f:f.write(v)
def sources():
    paths=[ROOT/'tools/build_pickup_terminal_restore_v2.py',old.P/'checkpoint_resume_cancel_root_v1/REGISTRATION.json',old.P/'checkpoint_resume_cancel_root_v1/INDEPENDENT_CHECKPOINT_RESUME_CANCEL.json',WARM,old.P/'continuous_checkpoint_root_v1/REGISTRATION.json',old.P/'continuous_checkpoint_root_v1/WORKER_FAILURE.json',old.P/'event27_settlement_diagnosis_root_v1/REGISTRATION.json',old.P/'event27_settlement_diagnosis_root_v1/RESULT.json',Path(__file__),ROOT/'tools/audit_pickup_terminal_restore_v2.py',ROOT/'tools/pickup_pending_transaction_v1.py',ROOT/'tools/pickup_loaded_closure_v1.py',INPUT,old.HISTORY,observed.fixture.TRACE,old.P/'undecided_public_input_root_v2/RESULT.json',Path(sys.executable)]
    paths.extend([old.P/'resident_day_tail_root_v1/REGISTRATION.json',old.P/'resident_day_tail_root_v1/FAILURE.json',old.P/'resident_day_tail_root_v1/RUN_EXIT.json',old.P/'resident_day_tail_root_v1/HOST_63.json'])
    return {str(p):sha(p) for p in paths}
class Done(Exception):pass
def worker():
    psutil.Process().cpu_affinity([7]);assert np.__version__=='1.26.4';dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name','cmdline']))
    ctrl=SharedMemory(name=sys.argv[2]);blob=SharedMemory(name=sys.argv[3]);reply=SharedMemory(name=sys.argv[4]);latch=events_api.event(sys.argv[5]);recover='--recover' in sys.argv
    encode,decode=old.codecs();template,dataset=observed.setup();initial=pickle.loads(WARM.read_bytes())['state'];history=pickle.loads(old.HISTORY.read_bytes());records=[];physical=[];current={};base=ji.transactional_class(C0ReserveReplanner)
    class Policy(base):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            raise ValueError('terminal-only source80 must not reveal a new undecided order')
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            v=super()._advance_fleet(i,clock,fleet,traces,served);current['clock']=clock;physical.append(dict(sequence=current['seq'],clock=clock,event_id=self.event_id,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()));return v
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
                if epoch!=63 or not 0<size<=MAX:raise ValueError('selected actual epoch63 bound')
                payload=bytes(blob.buf[HEADER.size:HEADER.size+size])
                if hashlib.sha256(payload).digest()!=ph:raise ValueError('selected actual raw SHA')
                cp=pickle.loads(payload)
                if cp['epoch']!=epoch or cp['base_digest']!=bh.hex():raise ValueError('selected actual lineage')
                warm_digest=old.pf.digest(snapshot(clock));restore(cp['state']);target=old.pf.digest(snapshot(clock))
                if target!=sh.hex():raise ValueError('selected actual private restoration')
                baseline=target
                save('RESTORE_IN_REQUEST.json',encode(dict(sequence=seq,source_epoch=epoch,source_payload_sha256=ph.hex(),warm_digest=warm_digest,selected_digest=target,state=copy.deepcopy(snapshot(clock)),started=rs,finished=time.perf_counter(),admission_calls=0,physical_calls=0)))
                restored_once=True
            if op!=4 or seq!=64 or records:raise ValueError('sole original terminal loop; zero new admission')
            before=old.pf.digest(snapshot(clock))
            if op==4:
                calls_before=len(records);env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,reveals,float(env.tw_end[0].max()));clock=current.get('clock',clock)
                if len(records)!=calls_before:raise ValueError('unexpected terminal new admission')
                final_ev=evaluate_trace_a1(traces,{k:v[0] for k,v in dataset.items()},template['contract'],set(rp._accepted),set(rp._rejected),template['budget'],template['penalties'])
                if not final_ev['hard_feasible'] or final_ev['failures'] or set(np.flatnonzero(served))-{0}!=set(rp._accepted) or set(customers)!=(set(rp._accepted)|set(rp._rejected)) or any(t.services and t.return_arrival is None for t in traces) or any(v.status not in ('idle','closed') for v in fleet):raise ValueError('original terminal full-day coverage/physical gate')
            else:
                try:env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,reveals,float(env.tw_end[0].max()))
                except Done:clock=current['clock']
                else:raise ValueError('registered next reveal absent; no filtering')
            after=old.pf.digest(snapshot(clock));checkpoint=dict(epoch=seq,base_digest=before,state=copy.deepcopy(snapshot(clock)));payload=pickle.dumps(checkpoint,protocol=5)
            if len(payload)>MAX:raise ValueError('full checkpoint bound')
            raw_save('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin',payload)
            end=capture(sources());qualified=begin['files']==end['files'] and begin['module_files']==end['module_files'];save('SOURCE_END_'+str(seq)+'.json',end)
            info=dict(resident_before=before,resident_after=after,resident_started=started,resident_finished=time.perf_counter(),loaded_closure_qualified=qualified,payload_size=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest())
            if op==4:save('TERMINAL.json',encode(dict(info,state=copy.deepcopy(snapshot(clock)),evaluation=final_ev,admission_calls=0,total_new_requests=len(records),sequence=seq)))
            else:records[-1].update(info);save('REQUEST_'+str(seq)+'.json',encode(records[-1]))
            if seq==64:save('TERMINAL_EVENTS.json',encode(physical))
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
            if seq==64:break
    except Exception:
        save(prefix+'WORKER_FAILURE.json',dict(traceback=traceback.format_exc()));raise
    finally:
        for m in (ctrl,blob,reply):m.close()
        events_api.K.CloseHandle(latch)
def launch(ctrl,blob,reply,name,recover=False):
    err=(OUT/('recovery.stderr.txt' if recover else 'worker.stderr.txt')).open('x',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,blob.name,reply.name,name]+(['--recover'] if recover else []),stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'));ready=json.loads(p.stdout.readline());assert ready['ready'];return p,err,ready
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='New terminal-only engineering recovery after original whole-run FAIL. Sole actually parent-selected63/callback80 raw101462 SHA2fac336f0f8dd683cacc9ca38f3f4c85bf72fb8f1f0573320a96c7dc23ba7ce4; original private unpublished64 never a restoration source. Same original state/parameters/evaluator/10s cap execution8 reserve2. Warm older13/data/libraries/shared setup outside. Inside one10s window actual raw SHA/epoch/base/pickle/all-fields restore63, original loop remaining physical/return, original complete80 evaluator, actual terminal-only10event archive/no cumulative50-request encode, checkpoint payload/header/selector/parent cutoffSHA choiceACKarchive and natural0reap. No admission/SAA/model/fresh day/new science, no callback31..80 or old qualified QA rerun. Original failure/source/logs preserved; cumulative synchronous JSON encoding is a candidate bottleneck, not uniquely profiled. Full-science/whole80frominitial/physicalACK/disk/native guarantees remain false.',source_epoch=63,source_sha256='2fac336f0f8dd683cacc9ca38f3f4c85bf72fb8f1f0573320a96c7dc23ba7ce4',terminal_sequence=64,cap_s=10.,execution_s=8.,reserve_s=2.,expected_new_admissions=0,cpu=[7],numpy='1.26.4',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=2*SLOT);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[:]=bytes(2*SLOT);name='Local\\MaskCO_terminal_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        payload=INPUT.read_bytes();cp=pickle.loads(payload);authority=old.pf.digest(cp['state']);assert cp['epoch']==63 and sha(INPUT)=='2fac336f0f8dd683cacc9ca38f3f4c85bf72fb8f1f0573320a96c7dc23ba7ce4'
        blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,63,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority));raw_save('INITIAL_SLOT0_ENVELOPE.bin',bytes(blob.buf[:HEADER.size+len(payload)]));p,err,ready=launch(ctrl,blob,reply,name)
        s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,64,cutoff,4,0);ticket=None;validation=None;ack=False;chosen_at=None;chosen=None
        while time.perf_counter()<cutoff:
            slot=struct.unpack_from('<I',ctrl.buf,48)[0];epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,slot*SLOT)
            if ticket is None and epoch==64:
                if not 0<size<=MAX or bh.hex()!=authority:raise ValueError('current terminal lineage/size')
                vs=time.perf_counter();off=slot*SLOT;candidate=bytes(blob.buf[off:off+HEADER.size+size]);h=hashlib.sha256(candidate[HEADER.size:]).digest();again=struct.unpack_from('<I',ctrl.buf,48)[0];vf=time.perf_counter()
                if h!=ph or again!=slot or vf>=cutoff:raise ValueError('terminal parentSHA/stable/cutoff')
                validation=dict(started=vs,finished=vf,slot_before=slot,slot_after=again,payload_sha256=h.hex(),base_digest=bh.hex());ticket=dict(epoch=64,slot=slot,payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter()
                if chosen_at>=cutoff:raise ValueError('terminal choice cutoff')
                chosen=candidate
            got,dh=REPLY.unpack_from(reply.buf)
            if got==64:ack=ticket is not None and dh.hex()==ticket['digest'];break
            if p.poll() is not None:raise ValueError('terminal worker exited beforeACK')
            time.sleep(.001)
        if ticket is None or not ack:raise TimeoutError('terminal checkpoint absent within8')
        p.wait(timeout=max(.001,deadline-time.perf_counter()));off=ticket['slot']*SLOT;epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,off);raw=bytes(blob.buf[off:off+HEADER.size+size]);assert raw==chosen
        raw_save('ACTUAL_SHARED_ENVELOPE_64.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_64.bin',raw[HEADER.size:]);raw_save('RETAINED_ENVELOPE.bin',raw);raw_save('RETAINED_PAYLOAD.bin',raw[HEADER.size:]);f=time.perf_counter();host=dict(started=s,cutoff=cutoff,finished=f,wall_s=f-s,ticket=ticket,validation=validation,chosen_at=chosen_at,ack_present=ack,event_present=events_api.signaled(latch),actual_worker_exit=p.returncode,reaped=True);save('HOST.json',host)
        _,decode=old.codecs();t=decode(read(OUT/'TERMINAL.json'));restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'));events=decode(read(OUT/'TERMINAL_EVENTS.json'));checks=[]
        def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
        ck('restore63_all_fields_same_window',s<=t['resident_started']<=restore['started']<=restore['finished']<=t['resident_finished']<=chosen_at<cutoff and old.pf.digest(restore['state'])==old.pf.digest(cp['state']) and restore['selected_digest']==t['resident_before'])
        ck('zero_admission_scoring',restore['admission_calls']==restore['physical_calls']==t['admission_calls']==t['total_new_requests']==0)
        ck('terminal10physical',len(events)==10 and t['state']['event_id']==158 and t['state']['replan_count']==135)
        ck('full80original_C0',t['evaluation']['hard_feasible'] and not t['evaluation']['failures'] and t['evaluation']['n_served']==74 and t['evaluation']['n_rejected']==6)
        ck('complete_decisions_service_return',set(np.flatnonzero(t['state']['served']))-{0}==set(t['state']['state']['accepted']) and set(range(1,81))==set(t['state']['state']['accepted'])|set(t['state']['state']['rejected']) and all(v.status=='closed' for v in t['state']['fleet']))
        ck('terminal_raw_writer_shared_retained',raw[HEADER.size:]==(OUT/'ACTUAL_WRITER_PAYLOAD_64.bin').read_bytes()==(OUT/'RETAINED_PAYLOAD.bin').read_bytes())
        ck('actual_natural0_reap_under10',host['actual_worker_exit']==0 and host['reaped'] and host['ack_present'] and host['event_present'] and host['wall_s']<10.)
        begin=read(OUT/'SOURCE_START.json');end=read(OUT/'SOURCE_END_64.json');ck('loaded_workload_closed',begin['files']==end['files'] and begin['module_files']==end['module_files'] and t['loaded_closure_qualified'])
        ck('registered_bytes',all(sha(pth)==h for pth,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        bad=[c['name'] for c in checks if not c['passed']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(bad),failure_names=bad,host=host,physics=len(events),terminal_payload_sha256=sha(OUT/'RETAINED_PAYLOAD.bin'),terminal_payload_bytes=len(raw)-HEADER.size,actual_worker_exit=p.returncode,whole_previous_run_PASS=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(bad))));print(json.dumps(dict(checks=len(checks),failed=len(bad),failure_names=bad,wall_s=host['wall_s'])));return int(bool(bad))
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
