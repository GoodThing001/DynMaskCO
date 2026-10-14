"""Prospective selected10 restoration/idempotence and new callback28/29, no prior QA rerun."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
def create(n,t):
    ast.parse(t)
    with (ROOT/'tools'/n).open('x',encoding='utf-8',newline='\n') as f:f.write(t)
def main():
    p=(ROOT/'tools/qualify_pickup_selector_event_loss_v1.py').read_text(encoding='utf-8')
    p=p.replace('selector_event_loss','event_loss_resume').replace("prepared_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_8.bin","selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin").replace("inactive_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_7.bin","prepared_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_8.bin")
    p=p.replace('last=0','last=0;restored_once=False')
    start=p.index("            if op!=1 or seq!=len(records)+9")
    end=p.index("            before=old.pf.digest(snapshot(clock))",start)
    restoration=p[p.index('            if seq==9:',start):end].replace('if seq==9:','if not restored_once:').replace('epoch!=8','epoch!=10').replace('actual epoch8','actual epoch10')
    restoration+='                restored_once=True\n'
    p=p[:start]+restoration+DUPLICATE+p[end:]
    p=p.replace('if seq==10:break','if seq==12:break')
    start=p.index('            if fault==6:');end=p.index('            events_api.K.SetEvent(latch)',start)
    p=p[:start]+p[end:]
    start=p.index('def main():');p=p[:start]+MAIN
    create('qualify_pickup_event_loss_resume_v1.py',p)
    a=(ROOT/'tools/audit_pickup_selector_event_loss_v1.py').read_text(encoding='utf-8').replace('selector_event_loss','event_loss_resume').replace('SELECTOR_EVENT_LOSS','EVENT_LOSS_RESUME').replace('prepared_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_8.bin','selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin').replace('OWNER_STATE_10','OWNER_STATE_12').replace('index+9','index+11').replace('26+index','28+index').replace('selected_epoch8_actual_start','selected_epoch10_actual_start')
    a=a.replace("else:ck(name+':selector_only_slot0_commit',h['selected_slot']==0 and h['ticket'] is not None and not h['ack_present'] and not v['event_present'] and not h['event_signaled_final'] and h['stage']==6 and h['reaped'])","else:ck(name+':normal_ACK_slot0_final_reap',h['selected_slot']==0 and h['ticket'] is not None and h['ack_present'] and h['event_signaled_final'] and h['reaped'])")
    start=a.index("    restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'))")
    end=a.index("    ck('all_actual_audit_inputs_unchanged'",start)
    a=a[:start]+AUDIT_RESUME+a[end:]
    a=a.replace('two_authoritative_reveals_with_selector_only_second_publication=True','selected10_duplicate_without_execution_and_two_new_authoritative_reveals=True')
    create('audit_pickup_event_loss_resume_v1.py',a)
    print('new producer/audit saved, syntax parsed; no execution')
DUPLICATE="""            if op==2:
                if seq!=10 or records or physical:raise ValueError('only restored selected10 duplicate allowed')
                before=old.pf.digest(snapshot(clock));end=capture(sources());qualified=begin['files']==end['files'] and begin['module_files']==end['module_files']
                save('DUPLICATE_SELECTED10.json',encode(dict(epoch=seq,state=copy.deepcopy(snapshot(clock)),digest=before,physical_calls=len(physical),admission_calls=len(records),started=started,finished=time.perf_counter(),loaded_closure_qualified=qualified)))
                save('DUPLICATE_SOURCE_END.json',end)
                if not qualified or time.perf_counter()>=deadline:raise ValueError('duplicate deadline/closure')
                REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(before));continue
            if op!=1 or seq!=len(records)+11 or seq>12:raise ValueError('fixed two new continued requests')
"""
MAIN="""def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='New engineering: restore actual caller-selected epoch10/callback27 raw35318 checkpoint selected despite absent event/ACK in prior stage. Inside first10s window validate SHA/epoch/base/pickle/all private fields, duplicate epoch10 returns committed digest without new admission/scoring/physics/publishing. Same restored resident continues exactly next callback28 and29 once each, orders learned only at actual reveal, current-only admission/own_incumbent/commonK10/fullC0/compile/guard/plan/checkpoint. No replay callback27 or previous faults. Poll selector with bound/lineage/SHA/stability before8s; normal ACK, final graceful exit/reap in last10s window. No models/new science/external irreversible effects. Initial shared/library/history/data/warm older8 environment outside windows; no arbitrary parent/native/disk/persistence qualification.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,slot_count=2,cpu=[7],numpy='1.26.4',selected_checkpoint_restore_inside_first_request=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=2*SLOT);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[:]=bytes(2*SLOT);name='Local\\\\MaskCO_resume_'+ctrl.name;latch=events_api.event(name,True);p=err=None
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
"""
AUDIT_RESUME="""    restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'));dup=decode(read(OUT/'DUPLICATE_SELECTED10.json'));h=raw['restore_host']
    ck('selected10_restore_duplicate_separate_window',h['started']<=restore['started']<=restore['finished']<=dup['finished']<=h['finished']<h['cutoff'] and h['wall_s']==h['finished']-h['started'] and h['wall_s']<10. and restore['source_epoch']==10 and restore['source_payload_sha256']==sha(SOURCE) and restore['selected_digest']==dup['digest']==actual['initial_digest'] and restore['warm_digest']!=restore['selected_digest'])
    for k,v in initial.items():
        ck('actual_restored_source_field:'+k,old.pf.digest(v)==old.pf.digest(restore['state'][k]))
        ck('actual_duplicate_source_field:'+k,old.pf.digest(v)==old.pf.digest(dup['state'][k]))
    ck('restore_duplicate_no_prior_execution',restore['physical_calls']==restore['admission_calls']==dup['physical_calls']==dup['admission_calls']==0 and dup['epoch']==10 and h['epoch']==10 and h['digest']==dup['digest'])
    head=struct.Struct('<QQ32s32s32s');envelope=(OUT/'INITIAL_SLOT0_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope)
    ck('actual_initial_selected10_bytes',epoch==10 and envelope[head.size:]==SOURCE.read_bytes() and size==len(SOURCE.read_bytes()) and ph.hex()==sha(SOURCE) and bh.hex()==pickle.loads(SOURCE.read_bytes())['base_digest'] and sh.hex()==old.pf.digest(initial) and h['shared_envelope_unchanged'])
    bs=read(OUT/'SOURCE_START.json');de=read(OUT/'DUPLICATE_SOURCE_END.json');ck('duplicate_actual_loaded_closure',bs['files']==de['files'] and bs['module_files']==de['module_files'] and dup['loaded_closure_qualified'])
    ck('new_continuation_after_duplicate',h['finished']<=raw['host'][0]['started'] and h['pid']==raw['host'][0]['pid']==raw['host'][1]['pid'])
    ck('normal_graceful_exit',not raw['actual_worker_terminated'] and raw['actual_worker_exit']==0 and all(x['ack_present'] for x in raw['host']))
    ck('retained_latest12_actual', (OUT/'RETAINED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_12.bin').read_bytes() and (OUT/'RETAINED_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_12.bin').read_bytes())
"""
if __name__=='__main__':main()
