"""Exclusive derivative for actual checkpoint restore plus live requests in one window."""
import ast
from pathlib import Path
ROOT=Path('D:/PyCharm_/MASKCO-Main')
def rep(s,a,b):
    assert a in s,a
    return s.replace(a,b)
s=(ROOT/'tools/qualify_pickup_continuous_checkpoint_v2.py').read_text(encoding='utf-8')
s=rep(s,"OUT=old.P/'continuous_checkpoint_root_v2';INPUT=old.P/'checkpoint_bytes_retention_root_v1/ACTUAL_SHARED_PAYLOAD.bin'","OUT=old.P/'timed_restore_live_root_v1';INPUT=old.P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin';WARM=old.P/'continuous_checkpoint_root_v2/ACTUAL_SHARED_PAYLOAD_2.bin'")
s=rep(s,"paths=[ROOT/'tools/recover_pickup_continuous_checkpoint_v2.py',","paths=[ROOT/'tools/build_pickup_timed_restore_live_v1.py',old.P/'checkpoint_resume_cancel_root_v1/REGISTRATION.json',old.P/'checkpoint_resume_cancel_root_v1/INDEPENDENT_CHECKPOINT_RESUME_CANCEL.json',WARM,")
s=rep(s,"ROOT/'tools/audit_pickup_continuous_checkpoint_v2.py'","ROOT/'tools/audit_pickup_timed_restore_live_v1.py'")
s=rep(s,"initial=pickle.loads(INPUT.read_bytes())['state']","initial=pickle.loads(WARM.read_bytes())['state']")
s=rep(s,"if op!=1 or seq!=len(records)+1 or seq>2:raise ValueError('fixed two single-use new requests')\n            before=old.pf.digest(snapshot(clock))\n            # Sealed source stops immediately after event27 physical advance.\n            # Complete that original boundary once inside first request clock.\n            if seq==1:env._plan_and_commit(0,clock,fleet,traces,served,customers)","""if op!=1 or seq!=len(records)+4 or seq>5:raise ValueError('fixed two continued live requests')
            if seq==4:
                rs=time.perf_counter();epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf)
                if epoch!=3 or not 0<size<=MAX:raise ValueError('selected actual epoch3 bound')
                payload=bytes(blob.buf[HEADER.size:HEADER.size+size])
                if hashlib.sha256(payload).digest()!=ph:raise ValueError('selected actual raw SHA')
                cp=pickle.loads(payload)
                if cp['epoch']!=epoch or cp['base_digest']!=bh.hex():raise ValueError('selected actual lineage')
                warm_digest=old.pf.digest(snapshot(clock));restore(cp['state']);target=old.pf.digest(snapshot(clock))
                if target!=sh.hex():raise ValueError('selected actual private restoration')
                baseline=target
                save('RESTORE_IN_REQUEST.json',encode(dict(sequence=seq,source_epoch=epoch,source_payload_sha256=ph.hex(),warm_digest=warm_digest,selected_digest=target,state=copy.deepcopy(snapshot(clock)),started=rs,finished=time.perf_counter(),admission_calls=0,physical_calls=0)))
            before=old.pf.digest(snapshot(clock))""")
s=rep(s,"REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(after))","REPLY.pack_into(reply.buf,0,seq,bytes.fromhex(after))\n            if seq==5:break")
beg=s.index("    psutil.Process().cpu_affinity([7]);OUT.mkdir",s.index('def main():'));end=s.index("if __name__=='__main__':")
s=s[:beg]+'''    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='Restore selected actual epoch3/callback20 raw shared checkpoint INSIDE first timed live request, warm only prior epoch2 env outside. Continue still-uncommitted callback21/order37 from cancelled discarded private branch into first caller-authoritative commit epoch4; then next actual callback22 epoch5 same resident. No effects of cancelled old epoch4 input used. Two normal requests, no old cancellation/duplicate QA repetition or scientific retries; no models/new science. Restore and current-only admission/commonK10/C0/compile/guard/realplan/checkpoint publication share first10s window.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,initial_runtime_and_old_warm_env_outside_clock=True,selected_checkpoint_restore_inside_first_request=True,cpu=[7],numpy='1.26.4',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=HEADER.size+MAX);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);name='Local\\\\MaskCO_timed_restore_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        payload=INPUT.read_bytes();cp=pickle.loads(payload);authority=old.pf.digest(cp['state']);assert cp['epoch']==3 and len(payload)<=MAX
        blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,3,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority))
        p,err,ready=launch(ctrl,blob,reply,name);host=[]
        for seq in (4,5):
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,1,0);ticket=None;ack=False;chosen_at=None
            while time.perf_counter()<cutoff:
                if ticket is None and events_api.signaled(latch):
                    epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf)
                    if epoch!=seq or size>MAX or bh.hex()!=authority:raise ValueError('host lineage')
                    ticket=dict(epoch=epoch,size=size,base_digest=bh.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter();authority=sh.hex()
                got,dh=REPLY.unpack_from(reply.buf)
                if got==seq:ack=ticket is not None and dh.hex()==authority;break
                if p.poll() is not None:raise ValueError('worker unexpectedly exited')
                time.sleep(.001)
            if ticket is None or not ack:raise TimeoutError('qualified normal live checkpoint absent')
            if seq==5:p.wait(timeout=max(.001,deadline-time.perf_counter()))
            raw=bytes(blob.buf[:HEADER.size+ticket['size']]);raw_save('ACTUAL_SHARED_ENVELOPE_'+str(seq)+'.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin',raw[HEADER.size:]);f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,ack_present=ack,final_reap_included=seq==5,reaped=p.poll() is not None,stage=struct.unpack_from('<I',ctrl.buf,40)[0]);host.append(row);save('HOST_'+str(seq)+'.json',row)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_5.json'));restored=decode(read(OUT/'RESTORE_IN_REQUEST.json'));checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        ck('selected_restore_inside_same_first_window',host[0]['started']<=restored['started']<=restored['finished']<=actual['records'][0]['started'] and restored['selected_digest']==actual['records'][0]['resident_before']==old.pf.digest(cp['state']) and restored['warm_digest']!=restored['selected_digest'])
        ck('restore_zero_policy_events',restored['admission_calls']==restored['physical_calls']==0)
        ck('callback21_uncommitted_retry_then22_same_resident',[r['callback'] for r in actual['records']]==[21,22] and actual['records'][0]['order']==37 and host[0]['pid']==host[1]['pid'])
        ck('second_starts_first_actual_commit',actual['records'][1]['resident_before']==actual['records'][0]['resident_after'])
        for h,r in zip(host,actual['records']):
            seq=h['sequence'];ck(str(seq)+':actual_raw_writer_shared_exact',(OUT/('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin')).read_bytes()==(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin')).read_bytes());ck(str(seq)+':qualified_actual_commit',r['actual_policy_pointer'] and r['loaded_closure_qualified'] and r['response']['source_qualified']);ck(str(seq)+':20_complete_original_branches',sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20 and r['response']['certified']);ck(str(seq)+':normal_select_and_ack_in_budget',h['ack_present'] and h['wall_s']<10. and h['chosen_at']<h['cutoff']);ck(str(seq)+':no_model_route_gain',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
        ck('last_normal_worker_reaped_exit0',host[1]['reaped'] and p.returncode==0 and not host[0]['final_reap_included'])
        ck('registered_bytes',all(sha(pth)==h for pth,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        fail=[x['name'] for x in checks if not x['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,actual_worker_exit=p.returncode,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[h['wall_s'] for h in host],orders=[r['order'] for r in actual['records']])));return int(bool(fail))
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
with (ROOT/'tools/qualify_pickup_timed_restore_live_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
a=(ROOT/'tools/audit_pickup_continuous_checkpoint_v3.py').read_text(encoding='utf-8')
a=rep(a,"OUT=P/'continuous_checkpoint_root_v2';SOURCE=P/'checkpoint_bytes_retention_root_v1/ACTUAL_SHARED_PAYLOAD.bin'","OUT=P/'timed_restore_live_root_v1';SOURCE=P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin'")
a=rep(a,'OWNER_STATE_2.json','OWNER_STATE_5.json');a=rep(a,'sequence=index+1','sequence=index+4');a=rep(a,"sealed_event27_actual_start","selected_epoch3_actual_start")
a=rep(a,"        if index==0:private_env._plan_and_commit(0,private_clock,private_fleet,private_traces,private_served,customers)\n",'')
a=rep(a,"row['callback']==18+index","row['callback']==21+index")
a=rep(a,"str(index+1)+'.bin'","str(index+4)+'.bin'");a=rep(a,"str(index+1)+'.json'","str(index+4)+'.json'");a=rep(a,"epoch==actual_checkpoint['epoch']==index+1","epoch==actual_checkpoint['epoch']==index+4")
a=rep(a,'qualify_pickup_continuous_checkpoint_v2.py','qualify_pickup_timed_restore_live_v1.py')
beg=a.index("    rh=raw['recovery_host']");end=a.index("    ck('all_actual_audit_inputs_unchanged'",beg)
a=a[:beg]+'''    restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'));firstrow=actual['records'][0];h=raw['host'][0]
    ck('actual_selected_restore_inside_first_request',h['started']<=firstrow['resident_started']<=restore['started']<=restore['finished']<=firstrow['started'] and restore['source_epoch']==3 and restore['source_payload_sha256']==sha(SOURCE) and restore['selected_digest']==actual['initial_digest']==firstrow['resident_before'] and restore['warm_digest']!=restore['selected_digest'])
    for k,v in initial.items():ck('actual_restored_source_field:'+k,old.pf.digest(v)==old.pf.digest(restore['state'][k]))
    ck('restore_no_prior_admission_or_physics',restore['physical_calls']==restore['admission_calls']==0)
    ck('normal_second_reaped_exit0',raw['actual_worker_exit']==0 and all(h['ack_present'] for h in raw['host']))
''' +a[end:]
a=rep(a,'INDEPENDENT_CONTINUOUS_CHECKPOINT.json','INDEPENDENT_TIMED_RESTORE_LIVE.json')
ast.parse(a)
with (ROOT/'tools/audit_pickup_timed_restore_live_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(a)
print('Prospective sources created; not executed.')
