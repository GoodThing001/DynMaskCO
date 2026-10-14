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
ROOT=old.ROOT;OUT=old.P/'lineage_header_cancel_root_v1';INPUT=old.P/'event_loss_resume_root_v1/ACTUAL_SHARED_PAYLOAD_12.bin';WARM=old.P/'selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin'
CONTROL=struct.Struct('<QdII');REPLY=struct.Struct('<Q32s');HEADER=struct.Struct('<QQ32s32s32s');MAX=2*1024*1024;SLOT=HEADER.size+MAX
events_api.K.ResetEvent.argtypes=[ctypes.c_void_p];events_api.K.ResetEvent.restype=ctypes.c_int
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def raw_save(n,v):
    with (OUT/n).open('xb') as f:f.write(v)
def sources():
    paths=[ROOT/'tools/build_pickup_lineage_header_cancel_v1.py',old.P/'checkpoint_resume_cancel_root_v1/REGISTRATION.json',old.P/'checkpoint_resume_cancel_root_v1/INDEPENDENT_CHECKPOINT_RESUME_CANCEL.json',WARM,old.P/'continuous_checkpoint_root_v1/REGISTRATION.json',old.P/'continuous_checkpoint_root_v1/WORKER_FAILURE.json',old.P/'event27_settlement_diagnosis_root_v1/REGISTRATION.json',old.P/'event27_settlement_diagnosis_root_v1/RESULT.json',Path(__file__),ROOT/'tools/audit_pickup_lineage_header_cancel_v1.py',ROOT/'tools/pickup_pending_transaction_v1.py',ROOT/'tools/pickup_loaded_closure_v1.py',INPUT,old.HISTORY,observed.fixture.TRACE,old.P/'undecided_public_input_root_v2/RESULT.json',Path(sys.executable)]
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
                if epoch!=12 or not 0<size<=MAX:raise ValueError('selected actual epoch12 bound')
                payload=bytes(blob.buf[HEADER.size:HEADER.size+size])
                if hashlib.sha256(payload).digest()!=ph:raise ValueError('selected actual raw SHA')
                cp=pickle.loads(payload)
                if cp['epoch']!=epoch or cp['base_digest']!=bh.hex():raise ValueError('selected actual lineage')
                warm_digest=old.pf.digest(snapshot(clock));restore(cp['state']);target=old.pf.digest(snapshot(clock))
                if target!=sh.hex():raise ValueError('selected actual private restoration')
                baseline=target
                save('RESTORE_IN_REQUEST.json',encode(dict(sequence=seq,source_epoch=epoch,source_payload_sha256=ph.hex(),warm_digest=warm_digest,selected_digest=target,state=copy.deepcopy(snapshot(clock)),started=rs,finished=time.perf_counter(),admission_calls=0,physical_calls=0)))
                restored_once=True
            if op!=1 or seq!=len(records)+13 or seq>14:raise ValueError('fixed two new continued requests')
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
            if fault==7:
                # Fault only the header base. Actual writer payload and state remain intact.
                wrong=bytearray(bytes.fromhex(before));wrong[0]^=1
                HEADER.pack_into(blob.buf,offset,seq,len(payload),bytes(wrong),hashlib.sha256(payload).digest(),bytes.fromhex(after))
                raw_save('FAULT_HEADER_ENVELOPE_'+str(seq)+'.bin',bytes(blob.buf[offset:offset+HEADER.size+len(payload)]))
                save('FAULT_HEADER_'+str(seq)+'.json',dict(sequence=seq,correct_base=before,wrong_base=bytes(wrong).hex(),slot=inactive,created=time.perf_counter(),payload_sha256=hashlib.sha256(payload).hexdigest()))
            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('before selector deadline fence')
            selector_started=time.perf_counter();struct.pack_into('<I',ctrl.buf,48,inactive)
            if fault==7:
                struct.pack_into('<I',ctrl.buf,40,7)
                save('FAULT_SELECTOR_14.json',dict(sequence=seq,slot=inactive,published=time.perf_counter(),selector_started=selector_started,event_sent=False,ack_sent=False))
                time.sleep(12.)
            events_api.K.SetEvent(latch);struct.pack_into('<I',ctrl.buf,40,2)
            if fault==2:time.sleep(12.)
            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('ACK fenced')
            REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(after))
            if seq==14:break
    except Exception:
        save(prefix+'WORKER_FAILURE.json',dict(traceback=traceback.format_exc()));raise
    finally:
        for m in (ctrl,blob,reply):m.close()
        events_api.K.CloseHandle(latch)
def launch(ctrl,blob,reply,name,recover=False):
    err=(OUT/('recovery.stderr.txt' if recover else 'worker.stderr.txt')).open('x',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,blob.name,reply.name,name]+(['--recover'] if recover else []),stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'));ready=json.loads(p.stdout.readline());assert ready['ready'];return p,err,ready
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='New engineering: actual retained selected slot0 epoch12/callback29 raw38929 is sole input; warm older10 outside. First10s window selected SHA/epoch/base/pickle/all-field restoration then genuine next callback30/current-only/own_incumbent/commonK10/fullC0/compile/guard/actual plan/checkpoint and parent cutoff selection. Same resident next callback31, order identity learned only at actual reveal. Fixed fault7 flips one bit of HEADER base only, intact actual payload/header/selector publish before event/ACK then12s delay. Parent rejects mismatched current base, never selects14, cancels/reaps at8+reserve2, retains exact previously chosen13 envelope even though shared selector points at invalid14. No recovery from invalid selector/unselected proposal, no prior duplicate/fault replay, no model/science/new day. No arbitrary parent/native/memory/disk/persistence/external irreversibility claim.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,slot_count=2,cpu=[7],numpy='1.26.4',expected_source_epoch=12,expected_source_sha256='3ae9e8f0348b15c544da7cfccfc8c03ce2e2154f3c7cca84fd4bacc70f1027b2',fixed_callback_indices=[30,31],fault='published_header_base_first_byte_xor1_before_event_ACK',selected_checkpoint_restore_inside_first_request=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    active=[x.pid for x in psutil.process_iter(['name','cmdline']) if x.pid!=os.getpid() and any(Path(t).name in ('qualify_pickup_lineage_header_cancel_v1.py','audit_pickup_lineage_header_cancel_v1.py') for t in x.info['cmdline'] or [])]
    save('RESOURCE_PREFLIGHT.json',dict(available_bytes=psutil.virtual_memory().available,owned_active=active,cpu=psutil.Process().cpu_affinity()))
    if active or psutil.virtual_memory().available<1024**3:raise ValueError('actual resource/owned process barrier')
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=2*SLOT);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[:]=bytes(2*SLOT);name='Local\\MaskCO_lineage_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        payload=INPUT.read_bytes();cp=pickle.loads(payload);authority=old.pf.digest(cp['state']);assert cp['epoch']==12 and sha(INPUT)=='3ae9e8f0348b15c544da7cfccfc8c03ce2e2154f3c7cca84fd4bacc70f1027b2' and len(payload)<=MAX
        blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,12,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority));initial=bytes(blob.buf[:HEADER.size+len(payload)]);raw_save('INITIAL_SLOT0_ENVELOPE.bin',initial)
        p,err,ready=launch(ctrl,blob,reply,name);host=[];retained=None
        for seq,fault in ((13,0),(14,7)):
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,1,fault);ticket=None;ack=False;chosen_at=None;chosen_bytes=None;validation=None;rejection=None
            while time.perf_counter()<cutoff:
                if ticket is None:
                    slot=struct.unpack_from('<I',ctrl.buf,48)[0]
                    if slot not in (0,1):raise ValueError('parent slot bound')
                    epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,slot*SLOT)
                    if epoch==seq:
                        if not 0<size<=MAX or bh.hex()!=authority:
                            if rejection is None:rejection=dict(reason='current_base_mismatch' if bh.hex()!=authority else 'size_bound',seen=time.perf_counter(),sequence=seq,slot=slot,observed_base=bh.hex(),expected_base=authority,size=size)
                        else:
                            vs=time.perf_counter();off=slot*SLOT;candidate=bytes(blob.buf[off:off+HEADER.size+size]);cp_sha=hashlib.sha256(candidate[HEADER.size:]).digest();slot_after=struct.unpack_from('<I',ctrl.buf,48)[0];event_present=events_api.signaled(latch);vf=time.perf_counter()
                            if cp_sha!=ph or slot_after!=slot or vf>=cutoff:raise ValueError('parent payload SHA/stable selector/cutoff')
                            validation=dict(started=vs,finished=vf,slot_before=slot,slot_after=slot_after,payload_sha256=cp_sha.hex(),event_present=event_present,sequence=seq,base_digest=bh.hex());ticket=dict(epoch=epoch,size=size,slot=slot,base_digest=bh.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter()
                            if chosen_at>=cutoff:raise ValueError('parent choice cutoff')
                            authority=sh.hex();chosen_bytes=candidate
                got,dh=REPLY.unpack_from(reply.buf)
                if got==seq:ack=ticket is not None and dh.hex()==authority;break
                if p.poll() is not None:raise ValueError('worker unexpectedly exited')
                time.sleep(.001)
            if fault:
                if ticket is not None or ack or rejection is None:raise ValueError('wrong-lineage selection/fault not reached')
                struct.pack_into('<I',ctrl.buf,44,1);kill_started=time.perf_counter();p.kill();p.wait(timeout=max(.001,deadline-time.perf_counter()));kill_finished=time.perf_counter()
                slot=struct.unpack_from('<I',ctrl.buf,48)[0];off=slot*SLOT;epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,off)
                if epoch!=14 or size>MAX or retained is None:raise ValueError('fault published slot missing')
                raw=bytes(blob.buf[off:off+HEADER.size+size]);raw_save('ACTUAL_FAULT_SHARED_ENVELOPE_14.bin',raw);raw_save('ACTUAL_FAULT_SHARED_PAYLOAD_14.bin',raw[HEADER.size:]);raw_save('RETAINED_ENVELOPE.bin',retained);raw_save('RETAINED_PAYLOAD.bin',retained[HEADER.size:])
            else:
                if ticket is None or not ack:raise TimeoutError('normal checkpoint absent')
                slot=struct.unpack_from('<I',ctrl.buf,48)[0];off=slot*SLOT;epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,off);raw=bytes(blob.buf[off:off+HEADER.size+size])
                if raw!=chosen_bytes or slot!=ticket['slot']:raise ValueError('chosen raw envelope changed')
                retained=chosen_bytes;raw_save('ACTUAL_SHARED_ENVELOPE_13.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_13.bin',raw[HEADER.size:]);kill_started=kill_finished=None
            f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,validation=validation,rejection=rejection,ack_present=ack,event_signaled_final=events_api.signaled(latch),final_reap_included=bool(fault),reaped=p.poll() is not None,stage=struct.unpack_from('<I',ctrl.buf,40)[0],selected_slot=slot,authority_digest=authority,kill_started=kill_started,kill_finished=kill_finished);host.append(row);save('HOST_'+str(seq)+'.json',row)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_14.json'));restored=decode(read(OUT/'RESTORE_IN_REQUEST.json'));checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        ck('restore12_first_normal13_same_window',host[0]['started']<=restored['started']<=restored['finished']<=actual['records'][0]['started']<host[0]['finished']<host[0]['cutoff'] and restored['selected_digest']==actual['initial_digest']==old.pf.digest(cp['state']))
        ck('restore_zero_execution',restored['admission_calls']==restored['physical_calls']==0 and old.pf.digest(restored['state'])==old.pf.digest(cp['state']))
        ck('new30_31_same_restored_resident',[r['callback'] for r in actual['records']]==[30,31] and len({h['pid'] for h in host})==1)
        ck('second_starts_first_actual_commit',actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
        for index,r in enumerate(actual['records']):
            ck(str(index)+':qualified_C0_install',r['actual_policy_pointer'] and r['loaded_closure_qualified'] and r['response']['source_qualified'])
            ck(str(index)+':all20_branches',sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20 and r['response']['certified'])
            ck(str(index)+':no_model_route_gain',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
        ck('normal13_raw_writer_exact',(OUT/'ACTUAL_SHARED_PAYLOAD_13.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_13.bin').read_bytes() and host[0]['ticket']['digest']==actual['records'][0]['resident_after'])
        ck('normal13_cutoff_ACK',host[0]['ack_present'] and host[0]['event_signaled_final'] and host[0]['validation']['finished']<=host[0]['chosen_at']<host[0]['cutoff'] and host[0]['selected_slot']==1)
        h=host[1];bad=read(OUT/'FAULT_HEADER_14.json');fh=HEADER.unpack_from((OUT/'ACTUAL_FAULT_SHARED_ENVELOPE_14.bin').read_bytes())
        ck('fault7_published_bad_header_rejected',h['stage']==7 and h['selected_slot']==0 and h['ticket'] is None and h['chosen_at'] is None and h['rejection']['reason']=='current_base_mismatch' and h['rejection']['seen']<h['cutoff'] and fh[2].hex()==bad['wrong_base']!=bad['correct_base'])
        ck('fault_payload_intact', (OUT/'ACTUAL_FAULT_SHARED_PAYLOAD_14.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_14.bin').read_bytes())
        ck('fault_no_event_ACK',not h['ack_present'] and not h['event_signaled_final'])
        ck('retained13_not_shared_invalid14',(OUT/'RETAINED_ENVELOPE.bin').read_bytes()==retained==(OUT/'ACTUAL_SHARED_ENVELOPE_13.bin').read_bytes() and h['authority_digest']==host[0]['ticket']['digest'] and fh[4].hex()!=h['authority_digest'])
        ck('cancel_reaped_actual_returncode',h['reaped'] and p.returncode is not None and p.returncode!=0 and not host[0]['reaped'] and h['kill_started']>=h['cutoff'])
        ck('two_separate_windows_under10',all(h['wall_s']<10. for h in host))
        st=read(OUT/'SOURCE_START.json');end=read(OUT/'SOURCE_END_14.json');ck('start_to_fault_compute_closure',st['files']==end['files'] and st['module_files']==end['module_files'])
        ck('registered_bytes',all(sha(pth)==h for pth,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        fail=[x['name'] for x in checks if not x['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,actual_worker_exit=p.returncode,actual_worker_terminated=True,authoritative_epoch=13,discarded_callback=31,discarded_order=actual['records'][1]['order'],physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[h['wall_s'] for h in host],orders=[r['order'] for r in actual['records']])));return int(bool(fail))
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
