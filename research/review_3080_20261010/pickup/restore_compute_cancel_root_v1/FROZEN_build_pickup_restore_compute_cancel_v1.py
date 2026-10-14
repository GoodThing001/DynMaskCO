"""Two fixed new cancellation boundaries, no normal request repetition."""
import ast
from pathlib import Path
ROOT=Path('D:/PyCharm_/MASKCO-Main')
def rep(s,a,b):
    assert a in s,a
    return s.replace(a,b)
s=(ROOT/'tools/qualify_pickup_timed_restore_live_v1.py').read_text(encoding='utf-8')
s=rep(s,"OUT=old.P/'timed_restore_live_root_v1';INPUT=old.P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin';WARM=old.P/'continuous_checkpoint_root_v2/ACTUAL_SHARED_PAYLOAD_2.bin'", "BASE_OUT=old.P/'restore_compute_cancel_root_v1';OUT=BASE_OUT;INPUT=old.P/'timed_restore_live_root_v1/ACTUAL_SHARED_PAYLOAD_5.bin';WARM=old.P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin'")
s=rep(s,"paths=[ROOT/'tools/build_pickup_timed_restore_live_v1.py',old.P/'checkpoint_resume_cancel_root_v1/REGISTRATION.json',old.P/'checkpoint_resume_cancel_root_v1/INDEPENDENT_CHECKPOINT_RESUME_CANCEL.json',WARM,", "paths=[ROOT/'tools/build_pickup_restore_compute_cancel_v1.py',old.P/'timed_restore_live_root_v1/REGISTRATION.json',old.P/'timed_restore_live_root_v1/INDEPENDENT_TIMED_RESTORE_LIVE.json',WARM,")
s=rep(s,"ROOT/'tools/audit_pickup_timed_restore_live_v1.py'","ROOT/'tools/audit_pickup_restore_compute_cancel_v1.py'")
s=rep(s,"def worker():\n", "def worker():\n    global OUT\n    OUT=BASE_OUT/('CASE_'+sys.argv[sys.argv.index('--case')+1])\n")
s=rep(s,"installed,reason=owner.commit(token,response)","pause(2,dict(packet=packet,response=response,before=prior,private_at_reveal=copy.deepcopy(snapshot(clock)),events=copy.deepcopy(physical)))\n            installed,reason=owner.commit(token,response)")
s=rep(s,"rp.restore_state(copy.deepcopy(s['state']));env=Env", "rp.restore_state(copy.deepcopy(s['state']));\n        if current.get('restoring_selected'):pause(1,dict(selected_state=copy.deepcopy(s),restored_replanner=rp.export_state(),environment_restore_complete=False,physical_calls=0,admission_calls=0))\n        env=Env")
s=rep(s,"    rp=env=fleet=traces=served=clock=customers=reveals=None", """    def pause(stage,details):
        if current.get('fault')!=stage:return
        end=capture(sources());save('SOURCE_AT_FAULT.json',end)
        witness=dict(stage=stage,sequence=current['seq'],deadline=current['deadline'],reached=time.perf_counter(),details=details,loaded_closure_qualified=begin['files']==end['files'] and begin['module_files']==end['module_files'])
        save('FAULT_WITNESS.json',encode(witness));struct.pack_into('<I',ctrl.buf,40,stage);time.sleep(12.)
        raise TimeoutError('registered fault delay ended; publication forbidden')
    rp=env=fleet=traces=served=clock=customers=reveals=None""")
s=rep(s,"current.update(seq=seq,deadline=deadline)","current.update(seq=seq,deadline=deadline,fault=fault)")
s=rep(s,"if op!=1 or seq!=len(records)+4 or seq>5:raise ValueError('fixed two continued live requests')\n            if seq==4:","if op!=1 or seq!=6 or fault not in (1,2):raise ValueError('fixed next request and two faults')\n            if seq==6:")
s=rep(s,"if epoch!=3 or not 0<size<=MAX", "if epoch!=5 or not 0<size<=MAX")
s=rep(s,"warm_digest=old.pf.digest(snapshot(clock));restore(cp['state']);target", "warm_digest=old.pf.digest(snapshot(clock));current['restoring_selected']=True;restore(cp['state']);current['restoring_selected']=False;target")
beg=s.index("def main():");end=s.index("if __name__=='__main__':")
s=s[:beg]+'''def main():
    global OUT
    psutil.Process().cpu_affinity([7]);BASE_OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='Two fixed new copies of selected actual callback22/event40/epoch5 raw checkpoint. First timed restore paused after original replanner restore before actual env/fleet restore completes. Second selected restore completes and original next real reveal callback23 computes current-only admission/commonK10/all20branches/fullC0/compile, pauses before original owner.commit and real plan. Each12s delay,10s total/8s execution/2s reserve, kill/reap retains original sealed ticket/raw bytes. No normal request/old cancellation QA rerun, no checkpoint publication, no new science/models. Only simulation private branches, no external irreversible actions.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,fault_delay_s=12.,stages=[1,2],warm_prior_epoch3_outside_clock=True,selected_restore_inside_clock=True,cpu=[7],numpy='1.26.4',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    payload=INPUT.read_bytes();cp=pickle.loads(payload);assert cp['epoch']==5 and len(payload)<=MAX;authority=old.pf.digest(cp['state']);host=[];checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    for fault in (1,2):
        OUT=BASE_OUT/('CASE_'+str(fault));OUT.mkdir(exist_ok=False);ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=HEADER.size+MAX);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,5,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority));initial=bytes(blob.buf[:HEADER.size+len(payload)]);raw_save('INITIAL_SHARED_ENVELOPE.bin',initial);name='Local\\\\MaskCO_restore_cancel_'+ctrl.name;latch=events_api.event(name,True);p=err=None
        try:
            err=(OUT/'worker.stderr.txt').open('x',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,blob.name,reply.name,name,'--case',str(fault)],stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'));ready=json.loads(p.stdout.readline());assert ready['ready']
            s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,6,cutoff,1,fault)
            while time.perf_counter()<cutoff:
                if p.poll() is not None:raise ValueError('worker exited before planned cancellation')
                if events_api.signaled(latch) or REPLY.unpack_from(reply.buf)[0]:raise ValueError('unexpected publication')
                time.sleep(.001)
            reached=struct.unpack_from('<I',ctrl.buf,40)[0];struct.pack_into('<I',ctrl.buf,44,1);p.kill();p.wait(timeout=max(.001,deadline-time.perf_counter()));retained=bytes(blob.buf[:HEADER.size+len(payload)]);raw_save('RETAINED_SHARED_ENVELOPE.bin',retained);f=time.perf_counter();row=dict(fault=fault,stage=reached,sequence=6,pid=p.pid,started=s,cutoff=cutoff,finished=f,wall_s=f-s,exit_code=p.returncode,reaped=True,publication_signaled=events_api.signaled(latch),reply_sequence=REPLY.unpack_from(reply.buf)[0],authority_digest=authority,retained_payload_sha256=hashlib.sha256(retained[HEADER.size:]).hexdigest());save('HOST.json',row);host.append(row)
            _,decode=old.codecs();w=decode(read(OUT/'FAULT_WITNESS.json'))
            ck(str(fault)+':actual_fixed_stage_reached',reached==w['stage']==fault and s<=w['reached']<cutoff)
            ck(str(fault)+':original_shared_bytes_and_ticket_preserved',retained==initial and row['authority_digest']==old.pf.digest(cp['state']) and row['retained_payload_sha256']==sha(INPUT))
            ck(str(fault)+':no_publication_reply_or_new_owner',not row['publication_signaled'] and row['reply_sequence']==0 and not (OUT/'OWNER_STATE_6.json').exists())
            ck(str(fault)+':cancelled_reaped_in10',row['wall_s']<10. and row['reaped'] and p.poll() is not None)
            ck(str(fault)+':actual_loaded_fault_segment_identity',w['loaded_closure_qualified'])
            if fault==1:ck('partial_restore_no_admission_physical',w['details']['physical_calls']==w['details']['admission_calls']==0 and not w['details']['environment_restore_complete'])
            else:ck('actual_next_compute20branches_certified',w['details']['response']['certified'] and sum(len(v['branch_values']) for v in w['details']['response']['common_scores']['candidate_scores'].values())==20 and w['details']['packet']['admission_input']['current_order']==w['details']['response']['scoring_packet']['admission_input']['current_order'])
        finally:
            if p:
                if p.poll() is None:p.kill();p.wait(timeout=2.)
                p.stdout.close()
            if err:err.close()
            events_api.K.CloseHandle(latch)
            for m in (ctrl,blob,reply):m.close();m.unlink()
    OUT=BASE_OUT;ck('registered_bytes',all(sha(p)==h for p,h in read(OUT/'REGISTRATION.json')['sources'].items()));fail=[x['name'] for x in checks if not x['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[h['wall_s'] for h in host])));return int(bool(fail))
''' +s[end:]
ast.parse(s)
with (ROOT/'tools/qualify_pickup_restore_compute_cancel_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
print('Prospective producer created; audit source must be sealed before execution.')
