"""New prospective in-memory publication engineering, two independent slots."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def create(n,t):
    with (ROOT/'tools'/n).open('x',encoding='utf-8',newline='\n') as f:f.write(t)
def main():
    p=(ROOT/'tools/qualify_pickup_timed_restore_live_v1.py').read_text(encoding='utf-8')
    p=p.replace("OUT=old.P/'timed_restore_live_root_v1';INPUT=old.P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin';WARM=old.P/'continuous_checkpoint_root_v2/ACTUAL_SHARED_PAYLOAD_2.bin'", "OUT=old.P/'inactive_slot_cancel_root_v1';INPUT=old.P/'live_resume_saa_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_6.bin';WARM=old.P/'timed_restore_live_root_v1/ACTUAL_SHARED_PAYLOAD_5.bin'")
    p=p.replace('build_pickup_timed_restore_live_v1.py','build_pickup_inactive_slot_cancel_v1.py').replace('audit_pickup_timed_restore_live_v1.py','audit_pickup_inactive_slot_cancel_v1.py')
    p=p.replace("MAX=2*1024*1024","MAX=2*1024*1024;SLOT=HEADER.size+MAX")
    p=p.replace("seq!=len(records)+4 or seq>5","seq!=len(records)+7 or seq>8").replace('if seq==4:','if seq==7:').replace('epoch!=3 or not 0<size<=MAX','epoch!=6 or not 0<size<=MAX').replace('selected actual epoch3 bound','selected actual epoch6 bound').replace('if seq==5:break','if seq==8:break')
    original="            blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,seq,len(payload),bytes.fromhex(before),hashlib.sha256(payload).digest(),bytes.fromhex(after));events_api.K.SetEvent(latch);struct.pack_into('<I',ctrl.buf,40,2)"
    replacement="""            active=struct.unpack_from('<I',ctrl.buf,48)[0]
            if active not in (0,1):raise ValueError('selected slot bound')
            inactive=1-active;offset=inactive*SLOT
            if fault==4:
                half=len(payload)//2;blob.buf[offset+HEADER.size:offset+HEADER.size+half]=payload[:half]
                raw_save('PARTIAL_INACTIVE_PREFIX.bin',bytes(blob.buf[offset+HEADER.size:offset+HEADER.size+half]));save('PARTIAL_WRITE_WITNESS.json',dict(sequence=seq,selected_slot=active,inactive_slot=inactive,bytes_written=half,total_payload_bytes=len(payload),entered=time.perf_counter(),old_header_retained=True,selector_changed=False,new_header_written=False,checkpoint_event_sent=False))
                at=capture(sources());save('SOURCE_AT_PARTIAL_WRITE.json',at);struct.pack_into('<I',ctrl.buf,40,4);time.sleep(12.)
                raise TimeoutError('inactive partial write cancelled before header/selector/event')
            blob.buf[offset+HEADER.size:offset+HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,offset,seq,len(payload),bytes.fromhex(before),hashlib.sha256(payload).digest(),bytes.fromhex(after))
            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('before selector deadline fence')
            struct.pack_into('<I',ctrl.buf,48,inactive);events_api.K.SetEvent(latch);struct.pack_into('<I',ctrl.buf,40,2)"""
    assert original in p;p=p.replace(original,replacement)
    p=p[:p.index('def main():')]+MAIN
    create('qualify_pickup_inactive_slot_cancel_v1.py',p)
    a=(ROOT/'tools/audit_pickup_timed_restore_live_v1.py').read_text(encoding='utf-8')
    a=a.replace("OUT=P/'timed_restore_live_root_v1';SOURCE=P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin'", "OUT=P/'inactive_slot_cancel_root_v1';SOURCE=P/'live_resume_saa_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_6.bin'")
    a=a.replace('OWNER_STATE_5.json','OWNER_STATE_8.json').replace('index+4','index+7').replace('21+index','24+index').replace('selected_epoch3_actual_start','selected_epoch6_actual_start').replace("restore['source_epoch']==3","restore['source_epoch']==6").replace('qualify_pickup_timed_restore_live_v1.py','qualify_pickup_inactive_slot_cancel_v1.py').replace('INDEPENDENT_TIMED_RESTORE_LIVE.json','INDEPENDENT_INACTIVE_SLOT_CANCEL.json')
    start=a.index("        h=raw['host'][index];ck(name+':exact_elapsed_and_ticket'");end=a.index("        begin=read(OUT/'SOURCE_START.json')",start)
    a=a[:start]+AUDIT_WINDOW+a[end:]
    a=a.replace("ck('normal_second_reaped_exit0',raw['actual_worker_exit']==0 and all(h['ack_present'] for h in raw['host']))", "ck('cancelled_worker_actual_exit',raw['actual_worker_terminated'] and raw['actual_worker_exit']!=0 and raw['host'][0]['ack_present'] and not raw['host'][1]['ack_present'])")
    pos=a.index("    ck('all_actual_audit_inputs_unchanged'")
    a=a[:pos]+AUDIT_PARTIAL+a[pos:]
    a=a.replace('two_actual_new_reveals=True','one_authoritative_reveal_and_one_discarded_private_commit=True')
    create('audit_pickup_inactive_slot_cancel_v1.py',a)

MAIN='''def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='New engineering only: selected epoch6/callback23 raw checkpoint restored inside first request; callback24/order58 caller commit epoch7 in inactive slot1 then select it. Same resident next callback25 fully computes/privately installs and creates raw checkpoint, writes first half payload into inactive slot0 then12s delay; cancel/reap before new header/selector/event/ACK. Active slot1 envelope must remain exact epoch7. Two slots each2MiB+header, aligned32-bit in-memory selector at control offset48, Win32 event after selector. No claim of arbitrary selector tearing, persistent crash, memory-order proof or external irreversible effects; no models/new science, no repeated old QA.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,delay_s=12.,partial_payload_fraction='floor(len/2)',slot_count=2,cpu=[7],numpy='1.26.4',selected_checkpoint_restore_inside_first_request=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=2*SLOT);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[:]=bytes(2*SLOT);name='Local\\\\MaskCO_slots_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        payload=INPUT.read_bytes();cp=pickle.loads(payload);authority=old.pf.digest(cp['state']);assert cp['epoch']==6 and len(payload)<=MAX
        blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,6,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority));raw_save('INITIAL_SLOT0_ENVELOPE.bin',bytes(blob.buf[:HEADER.size+len(payload)]))
        p,err,ready=launch(ctrl,blob,reply,name);host=[]
        for seq in (7,8):
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,1,4 if seq==8 else 0);ticket=None;ack=False;chosen_at=None
            while time.perf_counter()<cutoff:
                if ticket is None and events_api.signaled(latch):
                    slot=struct.unpack_from('<I',ctrl.buf,48)[0]
                    if slot not in (0,1):raise ValueError('parent slot bound')
                    epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,slot*SLOT)
                    if epoch!=seq or size>MAX or bh.hex()!=authority:raise ValueError('host lineage')
                    ticket=dict(epoch=epoch,size=size,slot=slot,base_digest=bh.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter();authority=sh.hex()
                got,dh=REPLY.unpack_from(reply.buf)
                if got==seq:ack=ticket is not None and dh.hex()==authority;break
                if p.poll() is not None:raise ValueError('worker unexpectedly exited')
                time.sleep(.001)
            stage=struct.unpack_from('<I',ctrl.buf,40)[0]
            if seq==7:
                if ticket is None or not ack:raise TimeoutError('normal live checkpoint absent')
                off=ticket['slot']*SLOT;raw=bytes(blob.buf[off:off+HEADER.size+ticket['size']]);raw_save('ACTUAL_SHARED_ENVELOPE_7.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_7.bin',raw[HEADER.size:])
            else:
                if ticket is not None or ack or stage!=4:raise ValueError('fixed inactive partial-write stage absent / forbidden publish')
                struct.pack_into('<I',ctrl.buf,44,1);p.kill();p.wait(timeout=max(.001,deadline-time.perf_counter()))
                slot=struct.unpack_from('<I',ctrl.buf,48)[0];off=slot*SLOT;epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,off);raw=bytes(blob.buf[off:off+HEADER.size+size]);raw_save('RETAINED_ENVELOPE.bin',raw);raw_save('RETAINED_PAYLOAD.bin',raw[HEADER.size:]);w=read(OUT/'PARTIAL_WRITE_WITNESS.json');off=w['inactive_slot']*SLOT;raw_save('ACTUAL_INACTIVE_HEADER.bin',bytes(blob.buf[off:off+HEADER.size]));raw_save('ACTUAL_INACTIVE_PREFIX.bin',bytes(blob.buf[off+HEADER.size:off+HEADER.size+w['bytes_written']]))
            f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,ack_present=ack,final_reap_included=seq==8,reaped=p.poll() is not None,stage=stage,selected_slot=struct.unpack_from('<I',ctrl.buf,48)[0]);host.append(row);save('HOST_'+str(seq)+'.json',row)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_8.json'));restored=decode(read(OUT/'RESTORE_IN_REQUEST.json'));w=read(OUT/'PARTIAL_WRITE_WITNESS.json');checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        ck('timed_selected6_restore',host[0]['started']<=restored['started']<=restored['finished']<=actual['records'][0]['started'] and restored['selected_digest']==actual['records'][0]['resident_before']==old.pf.digest(cp['state']) and restored['warm_digest']!=restored['selected_digest'])
        ck('restore_zero_prior_execution',restored['admission_calls']==restored['physical_calls']==0)
        ck('normal_order58_next25_same_resident',[r['callback'] for r in actual['records']]==[24,25] and actual['records'][0]['order']==58 and host[0]['pid']==host[1]['pid'])
        ck('second_starts_first_actual_commit',actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
        for index,r in enumerate(actual['records']):
            ck(str(index)+':qualified_private_C0_install',r['actual_policy_pointer'] and r['loaded_closure_qualified'] and r['response']['source_qualified'])
            ck(str(index)+':all20_branches',sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20 and r['response']['certified'])
            ck(str(index)+':no_model_route_gain',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
        ck('normal_new_slot_selected_ACK',host[0]['selected_slot']==host[0]['ticket']['slot']==1 and host[0]['ack_present'] and not host[0]['reaped'] and not host[0]['final_reap_included'])
        ck('normal_writer_exact',(OUT/'ACTUAL_SHARED_PAYLOAD_7.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_7.bin').read_bytes())
        ck('actual_partial_inactive_prefix',(OUT/'ACTUAL_INACTIVE_PREFIX.bin').read_bytes()==(OUT/'PARTIAL_INACTIVE_PREFIX.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_8.bin').read_bytes()[:w['bytes_written']] and w['bytes_written']==w['total_payload_bytes']//2 and w['selected_slot']==1 and w['inactive_slot']==0)
        ck('inactive_old_header_exact',(OUT/'ACTUAL_INACTIVE_HEADER.bin').read_bytes()==(OUT/'INITIAL_SLOT0_ENVELOPE.bin').read_bytes()[:HEADER.size])
        ck('active7_byte_exact_retained',(OUT/'RETAINED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_7.bin').read_bytes() and host[1]['selected_slot']==1)
        ck('no_late_epoch8_select_ack',host[1]['ticket'] is None and not host[1]['ack_present'] and host[1]['stage']==4 and host[1]['reaped'] and p.returncode is not None and p.returncode!=0)
        ck('all_host_windows_under10',all(h['wall_s']<10. for h in host))
        st=read(OUT/'SOURCE_START.json');fault=read(OUT/'SOURCE_AT_PARTIAL_WRITE.json');ck('start_to_fault_closure',st['files']==fault['files'] and st['module_files']==fault['module_files'])
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
'''
AUDIT_WINDOW='''        h=raw['host'][index];seq=index+7
        ck(name+':within10_exact_clock',h['finished']-h['started']==h['wall_s'] and h['finished']<h['started']+10. and h['started']<=row['resident_started']<=row['started']<=response['worker_finished']<=row['committed']<=row['resident_finished'])
        payload=(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin')).read_bytes();c=pickle.loads(payload)
        ck(name+':actual_private_writer_checkpoint',len(payload)==row['payload_size'] and hashlib.sha256(payload).hexdigest()==row['payload_sha256'] and c['epoch']==seq and c['base_digest']==row['resident_before'] and old.pf.digest(c['state'])==old.pf.digest(snapshot()))
        if index==0:
            envelope=(OUT/'ACTUAL_SHARED_ENVELOPE_7.bin').read_bytes();head=struct.Struct('<QQ32s32s32s');epoch,size,bh,ph,sh=head.unpack_from(envelope)
            ck(name+':actual_shared_bytes',envelope[head.size:]==payload==(OUT/'ACTUAL_SHARED_PAYLOAD_7.bin').read_bytes() and len(payload)==size and hashlib.sha256(payload).digest()==ph and epoch==7 and bh.hex()==c['base_digest'] and sh.hex()==old.pf.digest(c['state']))
            ck(name+':selected_slot1_before_cutoff',h['selected_slot']==h['ticket']['slot']==1 and h['started']<=row['resident_finished']<=h['chosen_at']<h['cutoff'] and h['ack_present'] and h['ticket']['digest']==row['resident_after'])
        else:
            ck(name+':inactive_partial_cancel_keeps_selected',h['selected_slot']==1 and h['ticket'] is None and not h['ack_present'] and h['stage']==4 and h['reaped'] and (OUT/'RETAINED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_7.bin').read_bytes())
'''
AUDIT_PARTIAL='''    witness=read(OUT/'PARTIAL_WRITE_WITNESS.json');h=raw['host'][1];head=struct.Struct('<QQ32s32s32s')
    payload=(OUT/'ACTUAL_WRITER_PAYLOAD_8.bin').read_bytes();n=witness['bytes_written']
    ck('partial:exact_half_and_actual_prefix',n==len(payload)//2 and witness['total_payload_bytes']==len(payload) and (OUT/'ACTUAL_INACTIVE_PREFIX.bin').read_bytes()==(OUT/'PARTIAL_INACTIVE_PREFIX.bin').read_bytes()==payload[:n])
    ck('partial:inactive_slot0_old_header',witness['inactive_slot']==0 and witness['selected_slot']==1 and (OUT/'ACTUAL_INACTIVE_HEADER.bin').read_bytes()==(OUT/'INITIAL_SLOT0_ENVELOPE.bin').read_bytes()[:head.size])
    epoch,size,bh,ph,sh=head.unpack((OUT/'ACTUAL_INACTIVE_HEADER.bin').read_bytes());ck('partial:old6_header_not8',epoch==6 and size==len(SOURCE.read_bytes()) and ph.hex()==sha(SOURCE))
    ck('partial:no_header_selector_event',not witness['selector_changed'] and not witness['new_header_written'] and not witness['checkpoint_event_sent'] and witness['old_header_retained'])
    ck('partial:after_private_commit_before_cutoff',actual['records'][1]['resident_finished']<=witness['entered']<h['cutoff']<=h['finished'] and h['wall_s']<10.)
    envelope=(OUT/'RETAINED_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope);payload=envelope[head.size:];c=pickle.loads(payload)
    ck('partial:authority_is_actual7_not_private8',epoch==c['epoch']==7 and sh.hex()==old.pf.digest(c['state'])==actual['records'][0]['resident_after'] and sh.hex()!=actual['records'][1]['resident_after'] and payload==(OUT/'RETAINED_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_SHARED_PAYLOAD_7.bin').read_bytes() and hashlib.sha256(payload).digest()==ph and len(payload)==size)
    bs=read(OUT/'SOURCE_START.json');fe=read(OUT/'SOURCE_AT_PARTIAL_WRITE.json');ck('partial:start_to_actual_fault_files_maps',bs['files']==fe['files'] and bs['module_files']==fe['module_files'])
    for p,h in fe['files'].items():ck('partial:loaded:'+p,sha(p)==h)
'''
if __name__=='__main__':main()
