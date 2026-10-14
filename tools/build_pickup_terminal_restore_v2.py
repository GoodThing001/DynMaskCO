"""New terminal-only recovery; never reexecutes the fifty committed requests."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=(ROOT/'research/review_3080_20261010/pickup/resident_day_tail_root_v1/FROZEN_qualify_pickup_resident_day_tail_v1.py').read_text(encoding='utf-8')
    p=p.replace("OUT=old.P/'resident_day_tail_root_v1';INPUT=old.P/'lineage_header_cancel_root_v1/RETAINED_PAYLOAD.bin';WARM=old.P/'event_loss_resume_root_v1/ACTUAL_SHARED_PAYLOAD_12.bin'","OUT=old.P/'terminal_restore_root_v2';INPUT=old.P/'resident_day_tail_root_v1/ACTUAL_SHARED_PAYLOAD_63.bin';WARM=old.P/'lineage_header_cancel_root_v1/RETAINED_PAYLOAD.bin'")
    p=p.replace('build_pickup_resident_day_tail_v1.py','build_pickup_terminal_restore_v2.py').replace('audit_pickup_resident_day_tail_v1.py','audit_pickup_terminal_restore_v2.py').replace("paths.append(ROOT/'tools/run_pickup_resident_day_tail_bounded_v1.py')", "paths.extend([old.P/'resident_day_tail_root_v1/REGISTRATION.json',old.P/'resident_day_tail_root_v1/FAILURE.json',old.P/'resident_day_tail_root_v1/RUN_EXIT.json',old.P/'resident_day_tail_root_v1/HOST_63.json'])")
    start=p.index('            started=time.perf_counter();prior=self.export_state()');end=p.index('    class Env(StrictOnlineEnv):',start)
    p=p[:start]+"            raise ValueError('terminal-only source80 must not reveal a new undecided order')\n"+p[end:]
    p=p.replace('epoch!=13','epoch!=63').replace('actual epoch13','actual epoch63')
    old="if op not in (1,4) or seq!=len(records)+14 or seq>64 or (op==4)!=(seq==64):raise ValueError('registered50 consecutive new reveals then terminal only')"
    p=p.replace(old,"if op!=4 or seq!=64 or records:raise ValueError('sole original terminal loop; zero new admission')")
    p=p.replace("if seq==64:save('OWNER_STATE_64.json',encode(dict(initial_digest=baseline,records=records,events=physical,final=copy.deepcopy(snapshot(clock)),pid=os.getpid())))", "if seq==64:save('TERMINAL_EVENTS.json',encode(physical))")
    p=p[:p.index('def main():')]+MAIN
    ast.parse(p)
    with (ROOT/'tools/qualify_pickup_terminal_restore_v2.py').open('x',encoding='utf-8',newline='\n') as f:f.write(p)
    print('terminal-only recovery saved; no new execution')
MAIN=r'''def main():
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
'''
if __name__=='__main__':main()
