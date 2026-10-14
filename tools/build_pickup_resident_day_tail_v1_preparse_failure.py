"""Prospective continuous callback31..80 and terminal return from actual retained13."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
def create(n,t):
    ast.parse(t)
    with (ROOT/'tools'/n).open('x',encoding='utf-8',newline='\n') as f:f.write(t)
def main():
    p=(ROOT/'tools/qualify_pickup_event_loss_resume_v1.py').read_text(encoding='utf-8')
    p=p.replace('event_loss_resume_root_v1','resident_day_tail_root_v1').replace('build_pickup_event_loss_resume_v1','build_pickup_resident_day_tail_v1').replace('audit_pickup_event_loss_resume_v1','audit_pickup_resident_day_tail_v1')
    p=p.replace('selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin','lineage_header_cancel_root_v1/RETAINED_PAYLOAD.bin').replace('prepared_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_8.bin','event_loss_resume_root_v1/ACTUAL_SHARED_PAYLOAD_12.bin')
    p=p.replace('epoch!=10','epoch!=13').replace('actual epoch10','actual epoch13')
    p=p.replace('from strict_online_env import StrictOnlineEnv','from strict_online_env import StrictOnlineEnv\nfrom coldchain_evaluator_a1 import evaluate_trace_a1')
    p=p.replace("v=super()._advance_fleet(i,clock,fleet,traces,served);physical.append", "v=super()._advance_fleet(i,clock,fleet,traces,served);current['clock']=clock;physical.append")
    start=p.index('            if op==2:');end=p.index("            if not qualified or time.perf_counter()>=deadline",start)
    p=p[:start]+COMPUTE+p[end:]
    p=p.replace('            if seq==12:break','            if seq==64:break')
    p=p[:p.index('def main():')]+MAIN
    create('qualify_pickup_resident_day_tail_v1.py',p)
    a=(ROOT/'tools/audit_pickup_event_loss_resume_v1.py').read_text(encoding='utf-8').replace('event_loss_resume_root_v1','resident_day_tail_root_v1').replace('qualify_pickup_event_loss_resume_v1','qualify_pickup_resident_day_tail_v1').replace('selector_event_loss_root_v1/ACTUAL_SHARED_PAYLOAD_10.bin','lineage_header_cancel_root_v1/RETAINED_PAYLOAD.bin').replace('OWNER_STATE_12','OWNER_STATE_64').replace('index+11','index+14').replace('28+index','31+index').replace('selected_epoch10_actual_start','selected_epoch13_actual_start').replace('INDEPENDENT_EVENT_LOSS_RESUME','INDEPENDENT_RESIDENT_DAY_TAIL')
    a=a.replace('from strict_online_env import StrictOnlineEnv','from strict_online_env import StrictOnlineEnv\nfrom coldchain_evaluator_a1 import evaluate_trace_a1')
    a=a.replace("len(actual['records'])==2", "len(actual['records'])==50")
    a=a.replace("reference_events.append(dict(sequence=index+14", "reference_events.append(dict(sequence=reference_sequence")
    a=a.replace("    for index,row in enumerate(actual['records']):", "    reference_sequence=14\n    for index,row in enumerate(actual['records']):\n        reference_sequence=index+14")
    start=a.index("        if index==0:ck(name+':normal_ACK_slot1'");end=a.index("        begin=read(OUT/'SOURCE_START.json')",start)
    a=a[:start]+"        ck(name+':normal_ACK_alternating_resident',h['selected_slot']==(1-index%2) and h['ack_present'] and h['event_signaled_final'] and not h['reaped'])\n"+a[end:]
    start=a.index("    ck('all_actual_physical_events_retained'");a=a[:start]+AUDIT_TERMINAL+a[start:]
    a=a.replace("ck('same_process_and_final_reap',len({h['pid'] for h in raw['host']})==1 and raw['host'][1]['final_reap_included'] and raw['host'][1]['reaped'] and not raw['host'][0]['final_reap_included'])","ck('same_process_and_terminal_reap',len({h['pid'] for h in raw['host']})==1 and raw['host'][-1]['final_reap_included'] and raw['host'][-1]['reaped'] and all(not h['final_reap_included'] for h in raw['host'][:-1]))")
    start=a.index("    restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'))");end=a.index("    ck('all_actual_audit_inputs_unchanged'",start);a=a[:start]+AUDIT_RESTORE+a[end:]
    a=a.replace('selected10_duplicate_without_execution_and_two_new_authoritative_reveals=True','scoped_continuous_last50_and_terminal_from_actual13=True')
    create('audit_pickup_resident_day_tail_v1.py',a)
    print('new continuous-tail producer/auditor saved; syntax parsed; no execution')

COMPUTE=r'''            if op not in (1,4) or seq!=len(records)+14 or seq>64 or (op==4)!=(seq==64):raise ValueError('registered50 consecutive new reveals then terminal only')
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
            if seq==64:save('OWNER_STATE_64.json',encode(dict(initial_digest=baseline,records=records,events=physical,final=copy.deepcopy(snapshot(clock)),pid=os.getpid())))
'''

MAIN=r'''def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='New engineering continuous controller remainder: sole actual caller retained13/callback30 raw42569; never bad shared14/cancelled31 branch. Warm older12/data/history/libraries/shared preload outside. First10s selected SHA/epoch/base/pickle/all-fields restore then genuine callback31. Same resident/environment continues every actual callback31..80 exactly once without filtering/reload/selection, each current-only own_uncond_dynamic/own_incumbent/commonK10/fullC0/compile/guard/actualplan/checkpoint/parentSHA cutoff/ACK. Separate terminal64 operation resumes original loop from post80 plan to all remaining physical events/return and original complete80 full-day evaluator, raw checkpoint and natural0 reap within10. Each51 separate request10/execution8/reserve2, entire600s supervisor; no new fault/duplicates/prefix30 or old472/4711QA. Only observed-day engineering new controller, no model/science/gain/formal control/equal-time qualification; no arbitrary parent/native/disk/persistence claim.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,supervision_cap_s=600.,first_callback=31,last_callback=80,first_sequence=14,last_admission_sequence=63,terminal_sequence=64,source_epoch=13,source_sha256='5d9aa477e539d0f350e09e96970f288d8e50dac16f7a3ecba5674613ae03b7b7',new_initial_buffer_slot=0,prior_caller_slot=1,cpu=[7],numpy='1.26.4',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and any(Path(t).name in ('qualify_pickup_resident_day_tail_v1.py','audit_pickup_resident_day_tail_v1.py') for t in p.info['cmdline'] or [])];save('RESOURCE_PREFLIGHT.json',dict(available_bytes=psutil.virtual_memory().available,owned_active=active));assert not active and psutil.virtual_memory().available>=1024**3
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=2*SLOT);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);blob.buf[:]=bytes(2*SLOT);name='Local\\MaskCO_tail_'+ctrl.name;latch=events_api.event(name,True);p=err=None;overall=time.perf_counter()
    try:
        payload=INPUT.read_bytes();cp=pickle.loads(payload);authority=old.pf.digest(cp['state']);assert cp['epoch']==13 and sha(INPUT)=='5d9aa477e539d0f350e09e96970f288d8e50dac16f7a3ecba5674613ae03b7b7' and len(payload)<=MAX
        blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,13,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority));raw_save('INITIAL_SLOT0_ENVELOPE.bin',bytes(blob.buf[:HEADER.size+len(payload)]));p,err,ready=launch(ctrl,blob,reply,name);host=[]
        for seq in range(14,65):
            if time.perf_counter()-overall>=600.:raise TimeoutError('registered whole engineering supervision')
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,4 if seq==64 else 1,0);ticket=None;ack=False;chosen_at=None;chosen_bytes=None;validation=None
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
            if ticket is None or not ack:
                struct.pack_into('<I',ctrl.buf,44,1)
                raise TimeoutError('registered request publication absent')
            if seq==64:p.wait(timeout=max(.001,deadline-time.perf_counter()))
            slot=struct.unpack_from('<I',ctrl.buf,48)[0];off=slot*SLOT;epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf,off);raw=bytes(blob.buf[off:off+HEADER.size+size])
            if raw!=chosen_bytes or slot!=ticket['slot']:raise ValueError('chosen raw envelope changed')
            raw_save('ACTUAL_SHARED_ENVELOPE_'+str(seq)+'.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin',raw[HEADER.size:])
            if seq==64:raw_save('RETAINED_ENVELOPE.bin',raw);raw_save('RETAINED_PAYLOAD.bin',raw[HEADER.size:])
            f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,validation=validation,ack_present=ack,event_signaled_final=events_api.signaled(latch),final_reap_included=seq==64,reaped=p.poll() is not None,stage=struct.unpack_from('<I',ctrl.buf,40)[0],selected_slot=slot);host.append(row);save('HOST_'+str(seq)+'.json',row);print(json.dumps(dict(progress=True,sequence=seq,callback=17+seq if seq<64 else None,wall_s=row['wall_s'])),flush=True)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_64.json'));restored=decode(read(OUT/'RESTORE_IN_REQUEST.json'));terminal=decode(read(OUT/'TERMINAL.json'));checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        ck('selected13_restore_first14_inside_window',host[0]['started']<=restored['started']<=restored['finished']<=actual['records'][0]['started']<host[0]['finished']<host[0]['cutoff'] and restored['selected_digest']==actual['initial_digest']==old.pf.digest(cp['state']))
        ck('restored_all_fields_zero_calls',old.pf.digest(restored['state'])==old.pf.digest(cp['state']) and restored['admission_calls']==restored['physical_calls']==0)
        ck('all50_genuine_new_callbacks_same_resident',[r['callback'] for r in actual['records']]==list(range(31,81)) and len({h['pid'] for h in host})==1 and len(host)==51)
        for index,r in enumerate(actual['records']):
            h=host[index];seq=index+14
            ck(str(seq)+':actual_previous_commit_start',r['resident_before']==(actual['records'][index-1]['resident_after'] if index else actual['initial_digest']))
            ck(str(seq)+':qualified_C0_install',r['actual_policy_pointer'] and r['loaded_closure_qualified'] and r['response']['source_qualified'])
            ck(str(seq)+':all20_K10',sum(len(c['branch_values']) for c in r['response']['common_scores']['candidate_scores'].values())==20 and r['response']['certified'])
            ck(str(seq)+':no_model_or_route_gain',not r['response']['formal_learned_controls'] and not r['response']['accepted_route'])
            ck(str(seq)+':raw_writer_shared_exact',(OUT/('ACTUAL_SHARED_PAYLOAD_'+str(seq)+'.bin')).read_bytes()==(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(seq)+'.bin')).read_bytes() and h['ticket']['digest']==r['resident_after'])
            ck(str(seq)+':normal_bound_ACK_resident',h['ack_present'] and h['event_signaled_final'] and h['validation']['finished']<=h['chosen_at']<h['cutoff'] and h['selected_slot']==1-index%2 and not h['reaped'])
        ck('terminal_from_actual80_no_new_admission',terminal['resident_before']==actual['records'][-1]['resident_after'] and terminal['admission_calls']==0 and terminal['total_new_requests']==50)
        ck('original_full80_C0_hard',terminal['evaluation']['hard_feasible'] and not terminal['evaluation']['failures'])
        ck('terminal_accepted_served_partition',set(np.flatnonzero(actual['final']['served']))-{0}==set(actual['final']['state']['accepted']) and set(range(1,81))==set(actual['final']['state']['accepted'])|set(actual['final']['state']['rejected']))
        ck('terminal_return_closed',all(v.status in ('idle','closed') for v in actual['final']['fleet']) and all(not t.services or t.return_arrival is not None for t in actual['final']['traces']))
        ck('terminal64_raw_reap',(OUT/'ACTUAL_SHARED_PAYLOAD_64.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_64.bin').read_bytes()==(OUT/'RETAINED_PAYLOAD.bin').read_bytes() and host[-1]['reaped'] and p.returncode==0)
        ck('all51_separate_windows_under10',all(h['wall_s']<10. for h in host))
        st=read(OUT/'SOURCE_START.json');end=read(OUT/'SOURCE_END_64.json');ck('start_to_terminal_loaded_closure',st['files']==end['files'] and st['module_files']==end['module_files']);ck('registered_bytes',all(sha(pth)==h for pth,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        failed=[c['name'] for c in checks if not c['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,host=host,actual_worker_exit=p.returncode,actual_worker_terminated=False,terminal_epoch=64,remaining_requests=50,physical_events=len(actual['events']),whole_engineering_s=time.perf_counter()-overall,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(failed))));print(json.dumps(dict(checks=len(checks),failed=len(failed),failure_names=failed,requests=50,events=len(actual['events']),walls_range=[min(h['wall_s'] for h in host),max(h['wall_s'] for h in host)])));return int(bool(failed))
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

AUDIT_TERMINAL=r'''    reference_sequence=64;admissions_before=len(actual['records'])
    private_env._run_loop(0,private_clock,sum(t<=private_clock+1e-6 for t in reveals),private_fleet,private_traces,private_served,customers,reveals,float(private_env.tw_end[0].max()))
    if reference_events:private_clock=reference_events[-1]['clock']
    terminal=decode(read(OUT/'TERMINAL.json'));h=raw['host'][-1]
    ev=evaluate_trace_a1(private_traces,{k:v[0] for k,v in dataset.items()},first['contract'],set(private_rp._accepted),set(private_rp._rejected),first['budget'],first['penalties'])
    ck('terminal_original_full80_evaluation',old.pf.digest(ev)==old.pf.digest(terminal['evaluation']) and ev['hard_feasible'] and not ev['failures'])
    ck('terminal_original_end_fields',old.pf.digest(snapshot())==old.pf.digest(terminal['state'])==old.pf.digest(actual['final']))
    ck('terminal_no_new_admission',terminal['admission_calls']==0 and terminal['total_new_requests']==50 and h['sequence']==64)
    ck('terminal_all_decided_and_served',set(customers)==set(private_rp._accepted)|set(private_rp._rejected) and set(np.flatnonzero(private_served))-{0}==set(private_rp._accepted))
    ck('terminal_all_vehicles_closed',all(v.status in ('idle','closed') for v in private_fleet) and all(not t.services or t.return_arrival is not None for t in private_traces))
    payload=(OUT/'ACTUAL_WRITER_PAYLOAD_64.bin').read_bytes();cp=pickle.loads(payload);envelope=(OUT/'ACTUAL_SHARED_ENVELOPE_64.bin').read_bytes();head=struct.Struct('<QQ32s32s32s');epoch,size,bh,ph,sh=head.unpack_from(envelope)
    ck('terminal_original_checkpoint_bytes',payload==envelope[head.size:]==(OUT/'ACTUAL_SHARED_PAYLOAD_64.bin').read_bytes()==(OUT/'RETAINED_PAYLOAD.bin').read_bytes() and epoch==64 and size==len(payload) and ph==hashlib.sha256(payload).digest() and sh.hex()==old.pf.digest(snapshot()) and bh.hex()==actual['records'][-1]['resident_after']==cp['base_digest'] and old.pf.digest(cp['state'])==old.pf.digest(snapshot()))
    ck('terminal_SHA_choice_natural_reap_time',h['started']<=terminal['resident_started']<=terminal['resident_finished']<=h['validation']['started']<=h['validation']['finished']<=h['chosen_at']<h['cutoff'] and h['finished']<h['started']+10. and h['ack_present'] and h['event_signaled_final'] and h['reaped'] and h['selected_slot']==1 and h['finished']-h['started']==h['wall_s'])
    begin=read(OUT/'SOURCE_START.json');end=read(OUT/'SOURCE_END_64.json');ck('terminal_loaded_identity',begin['files']==end['files'] and begin['module_files']==end['module_files'])
'''

AUDIT_RESTORE=r'''    restore=decode(read(OUT/'RESTORE_IN_REQUEST.json'));h=raw['host'][0]
    ck('selected13_restore_first_actual31_same_window',h['started']<=restore['started']<=restore['finished']<=actual['records'][0]['started']<=h['finished']<h['cutoff'] and restore['source_epoch']==13 and restore['source_payload_sha256']==sha(SOURCE) and restore['selected_digest']==actual['initial_digest'] and restore['warm_digest']!=restore['selected_digest'])
    for k,v in initial.items():ck('actual_restored_source_field:'+k,old.pf.digest(v)==old.pf.digest(restore['state'][k]))
    ck('restore_no_prior_execution',restore['physical_calls']==restore['admission_calls']==0)
    head=struct.Struct('<QQ32s32s32s');envelope=(OUT/'INITIAL_SLOT0_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope)
    ck('actual_source_retained13_loaded_newslot0',epoch==13 and envelope[head.size:]==SOURCE.read_bytes() and size==len(SOURCE.read_bytes()) and ph.hex()==sha(SOURCE) and bh.hex()==pickle.loads(SOURCE.read_bytes())['base_digest'] and sh.hex()==old.pf.digest(initial))
    ck('natural_worker0_exit',not raw['actual_worker_terminated'] and raw['actual_worker_exit']==0 and all(h['ack_present'] for h in raw['host']))
    ck('all50_continuous_callbacks_no_selection',len(actual['records'])==50 and [r['callback'] for r in actual['records']]==list(range(31,81)) and [r['sequence'] for r in actual['records']]==list(range(14,64)))
    ck('retained_terminal64_actual',(OUT/'RETAINED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_64.bin').read_bytes())
'''
if __name__=='__main__':main()
