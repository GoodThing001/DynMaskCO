"""New prospective actual callback30 plus published wrong-lineage header fault."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
def put(name,code):
    ast.parse(code)
    with (ROOT/'tools'/name).open('x',encoding='utf-8',newline='\n') as f:f.write(code)
def main():
    p=(ROOT/'tools/qualify_pickup_event_loss_resume_v1.py').read_text(encoding='utf-8')
    p=p.replace('event_loss_resume_root_v1','lineage_header_cancel_root_v1').replace('build_pickup_event_loss_resume_v1','build_pickup_lineage_header_cancel_v1').replace('audit_pickup_event_loss_resume_v1','audit_pickup_lineage_header_cancel_v1')
    p=p.replace('selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin','event_loss_resume_root_v1/ACTUAL_SHARED_PAYLOAD_12.bin').replace('prepared_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_8.bin','selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin')
    p=p.replace('epoch!=10','epoch!=12').replace('actual epoch10','actual epoch12').replace('seq!=len(records)+11 or seq>12','seq!=len(records)+13 or seq>14').replace('if seq==12:break','if seq==14:break')
    start=p.index('            if op==2:');end=p.index('            if op!=1 or seq!=len(records)+13',start);p=p[:start]+p[end:]
    anchor='            if time.perf_counter()>=deadline or struct.unpack_from(\'<I\',ctrl.buf,44)[0]:raise ValueError(\'before selector deadline fence\')'
    patch="""            if fault==7:
                # Fault only the header base. Actual writer payload and state remain intact.
                wrong=bytearray(bytes.fromhex(before));wrong[0]^=1
                HEADER.pack_into(blob.buf,offset,seq,len(payload),bytes(wrong),hashlib.sha256(payload).digest(),bytes.fromhex(after))
                raw_save('FAULT_HEADER_ENVELOPE_'+str(seq)+'.bin',bytes(blob.buf[offset:offset+HEADER.size+len(payload)]))
                save('FAULT_HEADER_'+str(seq)+'.json',dict(sequence=seq,correct_base=before,wrong_base=bytes(wrong).hex(),slot=inactive,created=time.perf_counter(),payload_sha256=hashlib.sha256(payload).hexdigest()))
"""
    assert p.count(anchor)==1;p=p.replace(anchor,patch+anchor)
    anchor='            events_api.K.SetEvent(latch);struct.pack_into(\'<I\',ctrl.buf,40,2)'
    assert p.count(anchor)==1;p=p.replace(anchor,"""            if fault==7:
                struct.pack_into('<I',ctrl.buf,40,7)
                save('FAULT_SELECTOR_14.json',dict(sequence=seq,slot=inactive,published=time.perf_counter(),selector_started=selector_started,event_sent=False,ack_sent=False))
                time.sleep(12.)
"""+anchor)
    p=p[:p.index('def main():')]+MAIN
    put('qualify_pickup_lineage_header_cancel_v1.py',p)
    a=(ROOT/'tools/audit_pickup_event_loss_resume_v1.py').read_text(encoding='utf-8').replace('event_loss_resume_root_v1','lineage_header_cancel_root_v1').replace('qualify_pickup_event_loss_resume_v1','qualify_pickup_lineage_header_cancel_v1').replace('selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin','event_loss_resume_root_v1/ACTUAL_SHARED_PAYLOAD_12.bin').replace('OWNER_STATE_12','OWNER_STATE_14').replace('index+11','index+13').replace('28+index','30+index').replace('selected_epoch10_actual_start','selected_epoch12_actual_start').replace('INDEPENDENT_EVENT_LOSS_RESUME','INDEPENDENT_LINEAGE_HEADER_CANCEL')
    start=a.index("        seq=index+13;envelope=");end=a.index("        begin=read(OUT/'SOURCE_START.json')",start)
    a=a[:start]+AUDIT_PARENT+a[end:]
    a=a.replace("h['finished']<h['started']+10.","h['finished']<h['started']+10.")
    a=a.replace("ck('same_process_and_final_reap',len({h['pid'] for h in raw['host']})==1 and raw['host'][1]['final_reap_included'] and raw['host'][1]['reaped'] and not raw['host'][0]['final_reap_included'])","ck('same_process_and_final_reap',len({h['pid'] for h in raw['host']})==1 and raw['host'][1]['final_reap_included'] and raw['host'][1]['reaped'] and not raw['host'][0]['final_reap_included'])")
    start=a.index("    restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'))");end=a.index("    ck('all_actual_audit_inputs_unchanged'",start)
    a=a[:start]+AUDIT_RESTORE+a[end:]
    a=a.replace('selected10_duplicate_without_execution_and_two_new_authoritative_reveals=True','one_new_authoritative_reveal_and_one_discarded_wrong_lineage_publication=True')
    put('audit_pickup_lineage_header_cancel_v1.py',a)
    print('prospective new producer/auditor syntax parsed; no execution')

MAIN=r'''def main():
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
'''

AUDIT_PARENT=r'''        head=struct.Struct('<QQ32s32s32s')
        if index==0:
            envelope=(OUT/'ACTUAL_SHARED_ENVELOPE_13.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope)
            ck(name+':selected_actual_raw_envelope',envelope[head.size:]==payload==(OUT/'ACTUAL_SHARED_PAYLOAD_13.bin').read_bytes() and len(payload)==size and hashlib.sha256(payload).digest()==ph and epoch==seq and bh.hex()==c['base_digest'] and sh.hex()==old.pf.digest(c['state']))
            v=h['validation'];ck(name+':parent_SHA_before_choice',v['started']<=v['finished']<=h['chosen_at']<h['cutoff'] and v['slot_before']==v['slot_after']==h['selected_slot']==h['ticket']['slot'] and v['payload_sha256']==hashlib.sha256(payload).hexdigest() and v['base_digest']==row['resident_before'] and v['sequence']==seq)
            ck(name+':normal_ACK_slot1',h['selected_slot']==1 and h['ack_present'] and h['event_signaled_final'] and not h['reaped'] and h['ticket']['digest']==row['resident_after'])
        else:
            envelope=(OUT/'ACTUAL_FAULT_SHARED_ENVELOPE_14.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope);fault=read(OUT/'FAULT_HEADER_14.json');published=read(OUT/'FAULT_SELECTOR_14.json');rejection=h['rejection']
            wrong=bytearray(bytes.fromhex(c['base_digest']));wrong[0]^=1
            ck(name+':fault_only_header_base_bit',bh==bytes(wrong) and fault['wrong_base']==bh.hex() and fault['correct_base']==c['base_digest']==actual['records'][0]['resident_after'] and epoch==14 and sh.hex()==row['resident_after'] and size==len(payload) and ph==hashlib.sha256(payload).digest())
            ck(name+':actual_shared_fault_writer_bytes',envelope[head.size:]==payload==(OUT/'ACTUAL_FAULT_SHARED_PAYLOAD_14.bin').read_bytes() and envelope==(OUT/'FAULT_HEADER_ENVELOPE_14.bin').read_bytes())
            ck(name+':selector_bad14_not_authority',h['selected_slot']==0 and h['stage']==7 and published['slot']==0 and h['ticket'] is None and h['validation'] is None and h['chosen_at'] is None and not h['ack_present'] and not h['event_signaled_final'] and not published['event_sent'] and not published['ack_sent'])
            ck(name+':parent_original_base_rejection',rejection['reason']=='current_base_mismatch' and rejection['expected_base']==actual['records'][0]['resident_after'] and rejection['observed_base']==bh.hex()!=rejection['expected_base'] and rejection['sequence']==14 and rejection['slot']==0 and rejection['size']==len(payload))
            # Header writing and polling are concurrent: first parent sighting may precede worker save time.
            ck(name+':fault_publication_rejection_cancel_clock',row['resident_finished']<=published['selector_started']<=published['published']<h['cutoff'] and h['started']<=rejection['seen']<h['cutoff']<=h['kill_started']<=h['kill_finished']<=h['finished'] and h['reaped'])
            retained=(OUT/'RETAINED_ENVELOPE.bin').read_bytes();e,n,b,p,d=head.unpack_from(retained)
            ck(name+':retain_parent_chosen13_not_bad_selector',e==13 and retained==(OUT/'ACTUAL_SHARED_ENVELOPE_13.bin').read_bytes() and retained[head.size:]==(OUT/'RETAINED_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_13.bin').read_bytes() and h['authority_digest']==d.hex()==actual['records'][0]['resident_after'] and n==len(retained)-head.size and p==hashlib.sha256(retained[head.size:]).digest())
'''

AUDIT_RESTORE=r'''    restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'));h=raw['host'][0]
    ck('selected12_restore_in_normal13_window',h['started']<=restore['started']<=restore['finished']<=actual['records'][0]['started']<=h['finished']<h['cutoff'] and restore['source_epoch']==12 and restore['source_payload_sha256']==sha(SOURCE) and restore['selected_digest']==actual['initial_digest'] and restore['warm_digest']!=restore['selected_digest'])
    for k,v in initial.items():ck('actual_restored_source_field:'+k,old.pf.digest(v)==old.pf.digest(restore['state'][k]))
    ck('restore_no_prior_execution',restore['physical_calls']==restore['admission_calls']==0)
    head=struct.Struct('<QQ32s32s32s');envelope=(OUT/'INITIAL_SLOT0_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope)
    ck('actual_initial_selected12_bytes',epoch==12 and envelope[head.size:]==SOURCE.read_bytes() and size==len(SOURCE.read_bytes()) and ph.hex()==sha(SOURCE) and bh.hex()==pickle.loads(SOURCE.read_bytes())['base_digest'] and sh.hex()==old.pf.digest(initial))
    ck('worker_real_kill_not_natural0',raw['actual_worker_terminated'] and raw['actual_worker_exit']!=0 and raw['host'][1]['reaped'] and raw['authoritative_epoch']==13 and raw['discarded_callback']==31 and raw['discarded_order']==actual['records'][1]['order'])
    ck('one_authority_then_private_discard',raw['host'][0]['ticket']['epoch']==13 and raw['host'][1]['ticket'] is None and actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
'''
if __name__=='__main__':main()
