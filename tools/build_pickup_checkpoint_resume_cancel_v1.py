"""Prospective, exclusive engineering derivative; never edits sealed originals."""
from pathlib import Path
import ast
ROOT=Path('D:/PyCharm_/MASKCO-Main')
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)
s=(ROOT/'tools/qualify_pickup_continuous_checkpoint_v2.py').read_text(encoding='utf-8')
s=replace(s,"OUT=old.P/'continuous_checkpoint_root_v2';INPUT=old.P/'checkpoint_bytes_retention_root_v1/ACTUAL_SHARED_PAYLOAD.bin'","OUT=old.P/'checkpoint_resume_cancel_root_v1';INPUT=old.P/'continuous_checkpoint_root_v2/ACTUAL_SHARED_PAYLOAD_2.bin'")
s=replace(s,"paths=[ROOT/'tools/recover_pickup_continuous_checkpoint_v2.py',","paths=[ROOT/'tools/build_pickup_checkpoint_resume_cancel_v1.py',old.P/'continuous_checkpoint_root_v2/REGISTRATION.json',old.P/'continuous_checkpoint_root_v2/INDEPENDENT_CONTINUOUS_CHECKPOINT.json',")
s=replace(s,"ROOT/'tools/audit_pickup_continuous_checkpoint_v2.py'","ROOT/'tools/audit_pickup_checkpoint_resume_cancel_v1.py'")
s=replace(s,"last=0","last=None;committed_epoch=2;probes=[]")
s=replace(s,"if not seq or seq==last:time.sleep(.001);continue\n            last=seq;started", "if not seq or (seq,op)==last:time.sleep(.001);continue\n            last=(seq,op);started")
s=replace(s,"if op!=3 or seq!=2:","if op!=3 or seq!=3:")
s=replace(s,"if op!=1 or seq!=len(records)+1 or seq>2:raise ValueError('fixed two single-use new requests')", """if op in (4,5):
                target=old.pf.digest(snapshot(clock));valid=op==4 and seq==committed_epoch
                if not valid and not (op==5 and seq<committed_epoch):raise ValueError('fixed duplicate/stale probes')
                probes.append(dict(sequence=seq,op=op,recognized_duplicate=valid,rejected_stale=not valid,digest=target,records=len(records),events=len(physical),started=started,finished=time.perf_counter()))
                save('CONTROL_PROBE_'+str(op)+'.json',encode(probes[-1]));struct.pack_into('<I',ctrl.buf,40,op)
                REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(target) if valid else bytes(32));continue
            if op!=1 or seq!=len(records)+3 or seq>4:raise ValueError('fixed two subsequent requests')""")
s=replace(s,"if seq==1:env._plan_and_commit(0,clock,fleet,traces,served,customers)","# Selected checkpoint already follows full original reveal plan commit; no old boundary replay.")
s=replace(s,"blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into", "if fault==1:\n                struct.pack_into('<I',ctrl.buf,40,3);time.sleep(12.)\n                if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('prepublication fenced')\n            blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into")
s=replace(s,"REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(after))","REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(after));committed_epoch=seq")
start=s.index("    psutil.Process().cpu_affinity([7]);OUT.mkdir",s.index('def main():'))
end=s.index("if __name__=='__main__':")
s=s[:start]+'''    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='New engineering from selected callback19/event34 actual raw committed checkpoint. Next actual callback20 normal selected epoch3; duplicate epoch3 and stale epoch2 perform no admission/physical operation. Next actual callback21 computed/privately committed in same resident then12s delay BEFORE shared publication; cancel kill/reap retains epoch3. Separate10s actual shared epoch3 restore; no callback18/19 or old QA repetition. Private cancelled branch is discarded simulation only, no external irreversible actions.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,fault_delay_s=12.,initialization_outside_clock=True,cpu=[7],numpy='1.26.4',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=HEADER.size+MAX);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);name='Local\\\\MaskCO_resume_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        p,err,ready=launch(ctrl,blob,reply,name);authority=ready['baseline'];host=[];probe_host=[]
        for seq in (3,4):
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,1,1 if seq==4 else 0);ticket=None;ack=False;chosen_at=None
            while time.perf_counter()<cutoff:
                if ticket is None and events_api.signaled(latch):
                    epoch,size,base,ph,sh=HEADER.unpack_from(blob.buf)
                    if epoch!=seq or size>MAX or base.hex()!=authority:raise ValueError('host epoch/base/bound')
                    ticket=dict(epoch=epoch,size=size,base_digest=base.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter();authority=sh.hex()
                got,dh=REPLY.unpack_from(reply.buf)
                if got==seq:ack=ticket is not None and dh.hex()==authority;break
                if p.poll() is not None:raise ValueError('worker unexpectedly exited')
                time.sleep(.001)
            if seq==3 and (ticket is None or not ack):raise TimeoutError('first subsequent reveal not committed')
            if seq==4:
                struct.pack_into('<I',ctrl.buf,44,1);p.kill();p.wait(timeout=max(.001,deadline-time.perf_counter()))
                if ticket is not None or ack:raise ValueError('cancelled branch published')
                size=host[0]['ticket']['size'];raw=bytes(blob.buf[:HEADER.size+size]);raw_save('CANCEL_RETAINED_ENVELOPE.bin',raw)
            else:
                raw=bytes(blob.buf[:HEADER.size+ticket['size']]);raw_save('ACTUAL_SHARED_ENVELOPE_3.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_3.bin',raw[HEADER.size:])
            f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,authority_digest=authority,ack_present=ack,final_reap_included=seq==4,reaped=p.poll() is not None,stage=struct.unpack_from('<I',ctrl.buf,40)[0]);host.append(row);save('HOST_'+str(seq)+'.json',row)
            if seq==3:
                for op,probe_seq in ((4,3),(5,2)):
                    reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);ps=time.perf_counter();CONTROL.pack_into(ctrl.buf,0,probe_seq,ps+8.,op,0)
                    while time.perf_counter()<ps+8.:
                        got,dh=REPLY.unpack_from(reply.buf)
                        if got==probe_seq:break
                        if p.poll() is not None:raise ValueError('probe worker exit')
                        time.sleep(.001)
                    else:raise TimeoutError('probe')
                    probe_host.append(dict(op=op,sequence=got,digest=dh.hex(),started=ps,finished=time.perf_counter(),stage=struct.unpack_from('<I',ctrl.buf,40)[0],authority_digest=authority))
                    if (op==4 and dh.hex()!=authority) or (op==5 and dh!=bytes(32)):raise ValueError('sequence semantics')
                save('CONTROL_HOST.json',probe_host)
        p.stdout.close();err.close();p=err=None;ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);p,err,_=launch(ctrl,blob,reply,name,True)
        s=time.perf_counter();CONTROL.pack_into(ctrl.buf,0,3,s+8.,3,0);p.wait(timeout=8.);f=time.perf_counter();got,dh=REPLY.unpack_from(reply.buf);rh=dict(pid=p.pid,started=s,finished=f,wall_s=f-s,digest=dh.hex(),sequence=got,exit_code=p.returncode,reaped=True);save('RECOVERY_HOST.json',rh)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_4.json'));rec=decode(read(OUT/'RECOVERY.json'));checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        ck('two_new_callbacks20_21_same_resident',[x['callback'] for x in actual['records']]==[20,21] and host[0]['pid']==host[1]['pid'])
        ck('continuous_private_branch',actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
        for r in actual['records']:
            ck(str(r['sequence'])+':qualified_private_compute_commit',r['actual_policy_pointer'] and r['loaded_closure_qualified'] and r['response']['source_qualified'])
            ck(str(r['sequence'])+':20_complete_common_branches',sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20 and r['response']['certified'])
            ck(str(r['sequence'])+':no_learning_or_route_gain',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
        ck('first_exact_actual_payload',(OUT/'ACTUAL_SHARED_PAYLOAD_3.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_3.bin').read_bytes())
        ck('prepublication_cancel_retains_exact_prior_envelope',(OUT/'CANCEL_RETAINED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_3.bin').read_bytes() and host[1]['ticket'] is None and not host[1]['ack_present'] and host[1]['stage']==3 and host[1]['reaped'])
        for op in (4,5):
            probe=decode(read(OUT/('CONTROL_PROBE_'+str(op)+'.json')))
            ck('probe'+str(op)+':no_new_private_execution',probe['records']==1 and probe['digest']==actual['records'][0]['resident_after'] and probe['events']==len([e for e in actual['events'] if e['sequence']==3]))
        ck('actual_restore_prior_committed_not_cancelled',rec['digest']==rh['digest']==authority==actual['records'][0]['resident_after'] and old.pf.digest(rec['state'])==old.pf.digest(actual['records'][0]['private_after']) and rec['digest']!=actual['records'][1]['resident_after'])
        ck('separate_recovery_zero_execution',rec['physical_calls']==rec['admission_calls']==0 and rh['sequence']==3 and rh['exit_code']==0 and rh['wall_s']<10. and rec['loaded_closure_qualified'])
        ck('both_normal_cancel_within10',all(h['wall_s']<10. for h in host) and host[0]['chosen_at']<host[0]['cutoff'])
        ck('registered_bytes',all(sha(x)==v for x,v in read(OUT/'REGISTRATION.json')['sources'].items()))
        fail=[x['name'] for x in checks if not x['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,probe_host=probe_host,recovery_host=rh,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[x['wall_s'] for x in host],orders=[x['order'] for x in actual['records']],recovery_wall=rh['wall_s'])));return int(bool(fail))
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc()));save('RUN_EXIT.json',dict(exit_code=1));raise
    finally:
        if p:
            if p.poll() is None:p.kill();p.wait(timeout=2.)
            p.stdout.close()
        if err:err.close()
        events_api.K.CloseHandle(latch)
        for m in (ctrl,blob,reply):m.close();m.unlink()
''' +s[end:]
ast.parse(s)
with (ROOT/'tools/qualify_pickup_checkpoint_resume_cancel_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
a=(ROOT/'tools/audit_pickup_continuous_checkpoint_v3.py').read_text(encoding='utf-8')
a=replace(a,"OUT=P/'continuous_checkpoint_root_v2';SOURCE=P/'checkpoint_bytes_retention_root_v1/ACTUAL_SHARED_PAYLOAD.bin'","OUT=P/'checkpoint_resume_cancel_root_v1';SOURCE=P/'continuous_checkpoint_root_v2/ACTUAL_SHARED_PAYLOAD_2.bin'")
a=replace(a,"OWNER_STATE_2.json","OWNER_STATE_4.json")
a=replace(a,"sequence=index+1","sequence=index+3")
a=replace(a,"        if index==0:private_env._plan_and_commit(0,private_clock,private_fleet,private_traces,private_served,customers)\n","")
a=replace(a,"row['callback']==18+index","row['callback']==20+index")
# Only first new branch publishes. Both actual private branches are independently replayed.
beg=a.index("        h=raw['host'][index]");end=a.index("        begin=read(OUT/'SOURCE_START.json')",beg)
a=a[:beg]+'''        h=raw['host'][index];seq=index+3
        ck(name+':within10_exact_clock',h['finished']-h['started']==h['wall_s'] and h['finished']<h['started']+10. and h['started']<=row['resident_started']<=row['started']<=response['worker_finished']<=row['committed']<=row['resident_finished'])
        payload=(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin')).read_bytes();c=pickle.loads(payload)
        ck(name+':actual_private_writer_checkpoint',len(payload)==row['payload_size'] and hashlib.sha256(payload).hexdigest()==row['payload_sha256'] and c['epoch']==seq and c['base_digest']==row['resident_before'] and old.pf.digest(c['state'])==old.pf.digest(snapshot()))
        if index==0:
            envelope=(OUT/'ACTUAL_SHARED_ENVELOPE_3.bin').read_bytes();head=struct.Struct('<QQ32s32s32s');epoch,size,bh,ph,sh=head.unpack_from(envelope)
            ck(name+':actual_shared_bytes',envelope[head.size:]==payload==(OUT/'ACTUAL_SHARED_PAYLOAD_3.bin').read_bytes() and len(payload)==size and hashlib.sha256(payload).digest()==ph and epoch==3 and bh.hex()==c['base_digest'] and sh.hex()==old.pf.digest(c['state']))
            ck(name+':selected_before_cutoff',h['started']<=row['resident_finished']<=h['chosen_at']<h['cutoff'] and h['ack_present'] and h['ticket']['digest']==row['resident_after'])
        else:
            ck(name+':cancelled_before_publication',h['ticket'] is None and not h['ack_present'] and h['stage']==3 and h['reaped'] and (OUT/'CANCEL_RETAINED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_3.bin').read_bytes())
''' +a[end:]
a=replace(a,"str(index+1)+'.json'","str(index+3)+'.json'")
a=replace(a,"qualify_pickup_continuous_checkpoint_v2.py","qualify_pickup_checkpoint_resume_cancel_v1.py")
beg=a.index("    rh=raw['recovery_host']");end=a.index("    rs=read(OUT/'RECOVERY_SOURCE_START.json')",beg)
a=a[:beg]+'''    rh=raw['recovery_host'];rec=decode(read(OUT/'RECOVERY.json'));selected=actual['records'][0]
    for k,v in selected['private_after'].items():ck('restored_prior_committed:'+k,old.pf.digest(v)==old.pf.digest(rec['state'][k]))
    ck('separate_recovery_prior_ticket',rec['physical_calls']==rec['admission_calls']==0 and rec['epoch']==rh['sequence']==3 and rec['digest']==rec['duplicate_digest']==rh['digest']==selected['resident_after'] and rh['reaped'] and rh['exit_code']==0 and rh['finished']-rh['started']==rh['wall_s']<10.)
    for op,h in zip((4,5),raw['probe_host']):
        pr=decode(read(OUT/('CONTROL_PROBE_'+str(op)+'.json')))
        ck('control:'+str(op)+':no_execution',pr['records']==1 and pr['events']==len([e for e in actual['events'] if e['sequence']==3]) and pr['digest']==selected['resident_after'] and h['authority_digest']==selected['resident_after'])
        ck('control:'+str(op)+':sequence_result',pr['recognized_duplicate']==(op==4) and pr['rejected_stale']==(op==5) and h['stage']==op and h['sequence']==(3 if op==4 else 2) and h['digest']==(selected['resident_after'] if op==4 else '00'*32))
    ck('cancel_retains_selected_ticket',raw['host'][1]['authority_digest']==selected['resident_after']!=actual['records'][1]['resident_after'])
''' +a[end:]
a=replace(a,"INDEPENDENT_CONTINUOUS_CHECKPOINT.json","INDEPENDENT_CHECKPOINT_RESUME_CANCEL.json")
ast.parse(a)
with (ROOT/'tools/audit_pickup_checkpoint_resume_cancel_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(a)
print('Created prospective producer and independent audit; not executed.')
