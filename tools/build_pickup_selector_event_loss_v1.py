"""New prospective publication boundary: selector published, event and ACK absent."""
from pathlib import Path
import ast,re
ROOT=Path(__file__).resolve().parents[1]

def create(n,t):
    ast.parse(t)
    with (ROOT/'tools'/n).open('x',encoding='utf-8',newline='\n') as f:f.write(t)

def advance(t):
    t=t.replace('prepared_slot_cancel','selector_event_loss').replace('PREPARED_SLOT_CANCEL','SELECTOR_EVENT_LOSS')
    t=t.replace('inactive_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_7.bin','prepared_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_8.bin')
    t=t.replace('live_resume_saa_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_6.bin','inactive_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_7.bin')
    d={"'OWNER_STATE_9":"'OWNER_STATE_10","'ACTUAL_WRITER_PAYLOAD_9":"'ACTUAL_WRITER_PAYLOAD_10","'ACTUAL_WRITER_PAYLOAD_8":"'ACTUAL_WRITER_PAYLOAD_9","'ACTUAL_SHARED_ENVELOPE_8":"'ACTUAL_SHARED_ENVELOPE_9","'ACTUAL_SHARED_PAYLOAD_8":"'ACTUAL_SHARED_PAYLOAD_9"}
    return re.sub('|'.join(re.escape(k) for k in d),lambda m:d[m[0]],t)

def main():
    p=advance((ROOT/'tools/qualify_pickup_prepared_slot_cancel_v1.py').read_text(encoding='utf-8'))
    p=p.replace('seq!=len(records)+8 or seq>9','seq!=len(records)+9 or seq>10').replace('if seq==9:break','if seq==10:break').replace('if seq==8:','if seq==9:')
    p=p.replace('epoch!=7 or not 0<size<=MAX','epoch!=8 or not 0<size<=MAX').replace('selected actual epoch7 bound','selected actual epoch8 bound')
    p=p.replace("cp['epoch']==7","cp['epoch']==8").replace('pack_into(blob.buf,0,7,len(payload)','pack_into(blob.buf,0,8,len(payload)')
    start=p.index('            if fault==5:');end=p.index('            if fault==2:',start)
    p=p[:start]+WORKER_PUBLICATION+p[end:]
    start=p.index('def main():');p=p[:start]+MAIN
    create('qualify_pickup_selector_event_loss_v1.py',p)
    a=advance((ROOT/'tools/audit_pickup_prepared_slot_cancel_v1.py').read_text(encoding='utf-8'))
    a=a.replace('index+8','index+9').replace('25+index','26+index').replace('selected_epoch7_actual_start','selected_epoch8_actual_start').replace("restore['source_epoch']==7","restore['source_epoch']==8")
    start=a.index('        if index==0:\n            envelope=');end=a.index("        begin=read(OUT/'SOURCE_START.json')",start)
    a=a[:start]+AUDIT_WINDOW+a[end:]
    start=a.index("    witness=read(OUT/'PREPARED_WRITE_WITNESS.json')");end=a.index("    ck('all_actual_audit_inputs_unchanged'",start)
    a=a[:start]+AUDIT_PUBLICATION+a[end:]
    a=a.replace('one_authoritative_reveal_and_one_discarded_private_commit=True','two_authoritative_reveals_with_selector_only_second_publication=True')
    create('audit_pickup_selector_event_loss_v1.py',a)
    print('new producer/audit saved before execution')

WORKER_PUBLICATION="""            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('before selector deadline fence')
            selector_started=time.perf_counter();struct.pack_into('<I',ctrl.buf,48,inactive)
            if fault==6:
                raw_save('SELECTOR_PUBLISHED_ENVELOPE.bin',bytes(blob.buf[offset:offset+HEADER.size+len(payload)]))
                save('SELECTOR_PUBLICATION_WITNESS.json',dict(sequence=seq,previous_slot=active,selected_slot=inactive,total_payload_bytes=len(payload),selector_started=selector_started,entered=time.perf_counter(),selector_changed=True,new_header_written=True,complete_payload_written=True,checkpoint_event_sent=False,ACK_sent=False))
                at=capture(sources());save('SOURCE_AT_SELECTOR_PUBLICATION.json',at);struct.pack_into('<I',ctrl.buf,40,6);time.sleep(12.)
                raise TimeoutError('selector published but event and ACK suppressed')
            events_api.K.SetEvent(latch);struct.pack_into('<I',ctrl.buf,40,2)
"""

MAIN="""def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='New engineering only: restore caller-selected epoch8/callback25 raw checkpoint inside first request; callback26/order36 first authoritative commit epoch9. Same resident next callback27 fully computes/installs then publishes full valid payload/header and aligned selector to new slot, waits12s before Win32 event/ACK. Parent polls selector regardless of event, bounds epoch/size/base and actual payload SHA, requires unchanged selector and before-cutoff verification then chooses exactly one ticket; selected epoch10 is authoritative despite absent event/ACK. Cancel/reap delayed worker at execution8s/reserve2s/total10s, archive exact selected bytes. No new retries or recovery window, no old pre-selector/ACK/half-write QA. No models/new science, arbitrary memory ordering/torn selector/persistence/external effects remain unqualified.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,delay_s=12.,fault_boundary='selector_published_before_event_ACK',parent_choice_rule='valid_current_sequence_lineage_SHA_stable_selector_before_cutoff_event_independent',slot_count=2,cpu=[7],numpy='1.26.4',selected_checkpoint_restore_inside_first_request=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=2*SLOT);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[:]=bytes(2*SLOT);name='Local\\\\MaskCO_selector_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        payload=INPUT.read_bytes();cp=pickle.loads(payload);authority=old.pf.digest(cp['state']);assert cp['epoch']==8 and len(payload)<=MAX
        blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,8,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority));raw_save('INITIAL_SLOT0_ENVELOPE.bin',bytes(blob.buf[:HEADER.size+len(payload)]))
        p,err,ready=launch(ctrl,blob,reply,name);host=[]
        for seq in (9,10):
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,1,6 if seq==10 else 0);ticket=None;ack=False;chosen_at=None;chosen_bytes=None;validation=None
            while time.perf_counter()<cutoff:
                if ticket is None:
                    slot=struct.unpack_from('<I',ctrl.buf,48)[0]
                    if slot not in (0,1):raise ValueError('parent slot bound')
                    epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,slot*SLOT)
                    if epoch==seq:
                        if not 0<size<=MAX or bh.hex()!=authority:raise ValueError('host current lineage/size')
                        vs=time.perf_counter();off=slot*SLOT;candidate=bytes(blob.buf[off:off+HEADER.size+size]);cp_sha=hashlib.sha256(candidate[HEADER.size:]).digest();slot_after=struct.unpack_from('<I',ctrl.buf,48)[0];event_present=events_api.signaled(latch);vf=time.perf_counter()
                        if cp_sha!=ph or slot_after!=slot or vf>=cutoff:raise ValueError('parent payload SHA/stable selector/cutoff')
                        validation=dict(started=vs,finished=vf,slot_before=slot,slot_after=slot_after,payload_sha256=cp_sha.hex(),event_present=event_present,sequence=seq,base_digest=bh.hex())
                        ticket=dict(epoch=epoch,size=size,slot=slot,base_digest=bh.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter()
                        if chosen_at>=cutoff:raise ValueError('parent choice cutoff')
                        authority=sh.hex();chosen_bytes=candidate
                got,dh=REPLY.unpack_from(reply.buf)
                if got==seq:ack=ticket is not None and dh.hex()==authority;break
                if p.poll() is not None:raise ValueError('worker unexpectedly exited')
                time.sleep(.001)
            stage=struct.unpack_from('<I',ctrl.buf,40)[0]
            if seq==9:
                if ticket is None or not ack:raise TimeoutError('normal checkpoint absent')
            else:
                if ticket is None or ack or stage!=6 or events_api.signaled(latch):raise ValueError('selector-only publication boundary absent')
                struct.pack_into('<I',ctrl.buf,44,1);p.kill();p.wait(timeout=max(.001,deadline-time.perf_counter()))
            slot=struct.unpack_from('<I',ctrl.buf,48)[0];off=slot*SLOT;epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,off);raw=bytes(blob.buf[off:off+HEADER.size+size])
            if raw!=chosen_bytes or slot!=ticket['slot']:raise ValueError('chosen raw envelope changed')
            raw_save('ACTUAL_SHARED_ENVELOPE_'+str(seq)+'.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin',raw[HEADER.size:])
            if seq==10:raw_save('RETAINED_ENVELOPE.bin',raw);raw_save('RETAINED_PAYLOAD.bin',raw[HEADER.size:])
            f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,validation=validation,ack_present=ack,event_signaled_final=events_api.signaled(latch),final_reap_included=seq==10,reaped=p.poll() is not None,stage=stage,selected_slot=slot);host.append(row);save('HOST_'+str(seq)+'.json',row)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_10.json'));restored=decode(read(OUT/'RESTORE_IN_REQUEST.json'));w=read(OUT/'SELECTOR_PUBLICATION_WITNESS.json');checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        ck('timed_selected8_restore',host[0]['started']<=restored['started']<=restored['finished']<=actual['records'][0]['started'] and restored['selected_digest']==actual['records'][0]['resident_before']==old.pf.digest(cp['state']) and restored['warm_digest']!=restored['selected_digest'])
        ck('restore_zero_prior_execution',restored['admission_calls']==restored['physical_calls']==0)
        ck('normal_order36_next27_same_resident',[r['callback'] for r in actual['records']]==[26,27] and actual['records'][0]['order']==36 and host[0]['pid']==host[1]['pid'])
        ck('second_starts_first_actual_commit',actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
        for index,r in enumerate(actual['records']):
            h=host[index];seq=index+9
            ck(str(index)+':qualified_C0_install',r['actual_policy_pointer'] and r['loaded_closure_qualified'] and r['response']['source_qualified'])
            ck(str(index)+':all20_branches',sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20 and r['response']['certified'])
            ck(str(index)+':no_model_route_gain',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
            ck(str(index)+':selected_raw_writer_exact',(OUT/('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin')).read_bytes()==(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin')).read_bytes() and h['ticket']['digest']==r['resident_after'] and h['validation']['payload_sha256']==r['payload_sha256'])
            ck(str(index)+':bounded_before_cutoff_validation',h['validation']['started']<=h['validation']['finished']<=h['chosen_at']<h['cutoff'] and h['validation']['slot_before']==h['validation']['slot_after']==h['selected_slot'])
        ck('normal_slot1_ACK_keeps_resident',host[0]['selected_slot']==1 and host[0]['ack_present'] and not host[0]['reaped'])
        ck('selector10_event_ACK_absent',w['sequence']==10 and w['previous_slot']==1 and w['selected_slot']==host[1]['selected_slot']==0 and w['selector_changed'] and not w['checkpoint_event_sent'] and not w['ACK_sent'] and host[1]['ticket'] is not None and not host[1]['validation']['event_present'] and not host[1]['event_signaled_final'] and not host[1]['ack_present'])
        ck('actual_selector_published_bytes',(OUT/'SELECTOR_PUBLISHED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_10.bin').read_bytes()==(OUT/'RETAINED_ENVELOPE.bin').read_bytes())
        ck('selector_published_before_choice',actual['records'][1]['resident_finished']<=w['selector_started']<=host[1]['chosen_at']<host[1]['cutoff'])
        ck('delayed_worker_reaped_not_natural0',host[1]['stage']==6 and host[1]['reaped'] and p.returncode is not None and p.returncode!=0)
        ck('all_host_windows_under10',all(h['wall_s']<10. for h in host))
        st=read(OUT/'SOURCE_START.json');fault=read(OUT/'SOURCE_AT_SELECTOR_PUBLICATION.json');ck('start_to_fault_closure',st['files']==fault['files'] and st['module_files']==fault['module_files'])
        ck('registered_bytes',all(sha(pth)==h for pth,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        fail=[x['name'] for x in checks if not x['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,actual_worker_exit=p.returncode,actual_worker_terminated=True,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[h['wall_s'] for h in host],orders=[r['order'] for r in actual['records']])));return int(bool(fail))
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

AUDIT_WINDOW="""        seq=index+9;envelope=(OUT/('ACTUAL_SHARED_ENVELOPE_'+str(seq)+'.bin')).read_bytes();head=struct.Struct('<QQ32s32s32s');epoch,size,bh,ph,sh=head.unpack_from(envelope)
        ck(name+':selected_actual_raw_envelope',envelope[head.size:]==payload==(OUT/('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin')).read_bytes() and len(payload)==size and hashlib.sha256(payload).digest()==ph and epoch==seq and bh.hex()==c['base_digest'] and sh.hex()==old.pf.digest(c['state']))
        v=h['validation'];ck(name+':parent_bounded_SHA_before_choice',v['started']<=v['finished']<=h['chosen_at']<h['cutoff'] and v['slot_before']==v['slot_after']==h['selected_slot']==h['ticket']['slot'] and v['payload_sha256']==hashlib.sha256(payload).hexdigest() and v['base_digest']==row['resident_before'] and v['sequence']==seq)
        ck(name+':selected_after_private_before_cutoff',h['started']<=row['resident_finished']<=h['chosen_at']<h['cutoff'] and h['ticket']['digest']==row['resident_after'])
        if index==0:ck(name+':normal_ACK_slot1',h['selected_slot']==1 and h['ack_present'] and h['event_signaled_final'] and not h['reaped'])
        else:ck(name+':selector_only_slot0_commit',h['selected_slot']==0 and h['ticket'] is not None and not h['ack_present'] and not v['event_present'] and not h['event_signaled_final'] and h['stage']==6 and h['reaped'])
"""

AUDIT_PUBLICATION="""    witness=read(OUT/'SELECTOR_PUBLICATION_WITNESS.json');h=raw['host'][1];head=struct.Struct('<QQ32s32s32s')
    envelope=(OUT/'RETAINED_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope);payload=envelope[head.size:];selected=pickle.loads(payload)
    ck('selector:full_actual_envelope',envelope==(OUT/'SELECTOR_PUBLISHED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_10.bin').read_bytes() and payload==(OUT/'ACTUAL_WRITER_PAYLOAD_10.bin').read_bytes()==(OUT/'RETAINED_PAYLOAD.bin').read_bytes() and len(payload)==size==witness['total_payload_bytes'])
    ck('selector:new10_lineage_SHA',epoch==selected['epoch']==10 and hashlib.sha256(payload).digest()==ph and bh.hex()==selected['base_digest']==actual['records'][1]['resident_before'] and sh.hex()==old.pf.digest(selected['state'])==actual['records'][1]['resident_after'])
    for k,v in selected['state'].items():ck('selector:authoritative_field:'+k,old.pf.digest(v)==old.pf.digest(actual['records'][1]['private_after'][k]))
    ck('selector:publication_event_ACK_absent',witness['sequence']==10 and witness['previous_slot']==1 and witness['selected_slot']==0 and witness['selector_changed'] and witness['new_header_written'] and witness['complete_payload_written'] and not witness['checkpoint_event_sent'] and not witness['ACK_sent'])
    ck('selector:before_parent_choice_cutoff',actual['records'][1]['resident_finished']<=witness['selector_started']<=h['validation']['started']<=h['chosen_at']<h['cutoff']<=h['finished'] and witness['entered']<h['cutoff'] and h['wall_s']<10.)
    ck('selector:authority10_not_previous9',sh.hex()!=actual['records'][0]['resident_after'] and envelope!=(OUT/'ACTUAL_SHARED_ENVELOPE_9.bin').read_bytes())
    bs=read(OUT/'SOURCE_START.json');fe=read(OUT/'SOURCE_AT_SELECTOR_PUBLICATION.json');ck('selector:start_to_actual_fault_files_maps',bs['files']==fe['files'] and bs['module_files']==fe['module_files'])
    for p,h in fe['files'].items():ck('selector:loaded:'+p,sha(p)==h)
"""
if __name__=='__main__':main()
