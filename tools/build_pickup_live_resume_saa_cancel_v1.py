"""Freeze a new one-normal / partial common-SAA cancellation engineering scope."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def create(name,source):
    with (ROOT/'tools'/name).open('x',encoding='utf-8',newline='\n') as f:f.write(source)
def main():
    p=(ROOT/'tools/qualify_pickup_timed_restore_live_v1.py').read_text(encoding='utf-8')
    p=p.replace("OUT=old.P/'timed_restore_live_root_v1';INPUT=old.P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin';WARM=old.P/'continuous_checkpoint_root_v2/ACTUAL_SHARED_PAYLOAD_2.bin'", "OUT=old.P/'live_resume_saa_cancel_root_v1';INPUT=old.P/'timed_restore_live_root_v1/ACTUAL_SHARED_PAYLOAD_5.bin';WARM=old.P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin'")
    p=p.replace('build_pickup_timed_restore_live_v1.py','build_pickup_live_resume_saa_cancel_v1.py').replace('audit_pickup_timed_restore_live_v1.py','audit_pickup_live_resume_saa_cancel_v1.py')
    p=p.replace('import hard_admission,rolling_optimizer','import hard_admission,rolling_optimizer\nfrom energy_complete_saa import EnergyCompleteSAA')
    p=p.replace("token=owner.begin(str(o),o,current['deadline'],old.pf.digest(decoded));response=engine.compute(decoded,history,old.E/'scripts',current['deadline'])", "token=owner.begin(str(o),o,current['deadline'],old.pf.digest(decoded));current.update(pending_packet=packet,pending_before=prior,private_at_reveal=copy.deepcopy(snapshot(clock)),physical_at_reveal=copy.deepcopy(physical));response=engine.compute(decoded,history,old.E/'scripts',current['deadline'])")
    marker='    class Env(StrictOnlineEnv):'
    hook='''    original_sim=EnergyCompleteSAA._sim_scenario
    def fault_sim(self,*args,**kwargs):
        if current.get('fault')==3:
            frame=sys._getframe(1).f_locals
            if len(current['completed'])==4:
                if frame['label']!='own_incumbent' or len(frame['values'])!=4:raise ValueError('fixed fifth first-candidate branch entry')
                base_response=sys._getframe(2).f_locals['base']
                witness=dict(sequence=current['seq'],packet=current['pending_packet'],before=current['pending_before'],private_at_reveal=current['private_at_reveal'],events=current['physical_at_reveal'],scoring_packet=frame['packet'],base_response=base_response,plans=frame['plans'],label=frame['label'],plan=frame['plan'],certificate=frame['cert'],observed_snapshot=frame['snapshot'],scenarios=frame['scenarios'],scenario_sha256=frame['scenario_hash'],branch_values=frame['values'],completed_values=current['completed'],branch_index=4,policy_installed=False,current_plan_and_commit=False,checkpoint_created=False,scoring_result_returned=False,entered=time.perf_counter())
                save('SAA_CANCEL_WITNESS.json',encode(witness));at=capture(sources());save('SOURCE_AT_SAA_CANCEL.json',at);struct.pack_into('<I',ctrl.buf,40,3);time.sleep(12.)
                raise TimeoutError('delayed branch fenced; no late score/publication')
            value=original_sim(self,*args,**kwargs);current['completed'].append(float(value));return value
        return original_sim(self,*args,**kwargs)
    EnergyCompleteSAA._sim_scenario=fault_sim
'''
    assert p.count(marker)==1;p=p.replace(marker,hook+marker)
    p=p.replace("current.update(seq=seq,deadline=deadline)","current.update(seq=seq,deadline=deadline,fault=fault,completed=[])")
    p=p.replace("seq!=len(records)+4 or seq>5","seq!=len(records)+6 or seq>7").replace('if seq==4:', 'if seq==6:').replace("epoch!=3 or not 0<size<=MAX","epoch!=5 or not 0<size<=MAX").replace('selected actual epoch3 bound','selected actual epoch5 bound').replace('if seq==5:break','if seq==7:break')
    p=p[:p.index('def main():')]+'''def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='Actual selected epoch5/callback22/event40 raw shared payload restored inside first10s window (warm only older epoch3 outside). Normal caller-authoritative callback23/order7 -> epoch6; same resident next actual callback24 is cancelled at entry of fifth original common-SAA branch after four first-candidate branches have completed. Fixed order/no sample filtering, no reuse of discarded order7 acceptance or old faults. Shared epoch6 remains authoritative; no mean/comparison/final compile/guard installation/current plan/checkpoint publication for cancelled callback24. Original branch implementation and exact values independently replayed; not scientific data or formal models.',K=10,common_seed=202610111301,cap_s=10.,execution_s=8.,reserve_s=2.,cancel_after_completed_first_candidate_branches=4,cancel_before_branch_index_zero_based=4,delay_s=12.,initial_runtime_and_old_warm_env_outside_clock=True,selected_checkpoint_restore_inside_first_request=True,cpu=[7],numpy='1.26.4',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=HEADER.size+MAX);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);name='Local\\\\MaskCO_live_saa_'+ctrl.name;latch=events_api.event(name,True);p=err=None
    try:
        payload=INPUT.read_bytes();cp=pickle.loads(payload);authority=old.pf.digest(cp['state']);assert cp['epoch']==5 and len(payload)<=MAX
        blob.buf[HEADER.size:HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,0,5,len(payload),bytes.fromhex(cp['base_digest']),hashlib.sha256(payload).digest(),bytes.fromhex(authority))
        raw_save('INITIAL_ENVELOPE.bin',bytes(blob.buf[:HEADER.size+len(payload)]));p,err,ready=launch(ctrl,blob,reply,name);host=[]
        for seq in (6,7):
            events_api.K.ResetEvent(latch);reply.buf[:]=bytes(REPLY.size);struct.pack_into('<I',ctrl.buf,40,0);s=time.perf_counter();cutoff=s+8.;deadline=s+10.;CONTROL.pack_into(ctrl.buf,0,seq,cutoff,1,3 if seq==7 else 0);ticket=None;ack=False;chosen_at=None
            while time.perf_counter()<cutoff:
                if ticket is None and events_api.signaled(latch):
                    epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf)
                    if epoch!=seq or size>MAX or bh.hex()!=authority:raise ValueError('host lineage')
                    ticket=dict(epoch=epoch,size=size,base_digest=bh.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen_at=time.perf_counter();authority=sh.hex()
                got,dh=REPLY.unpack_from(reply.buf)
                if got==seq:ack=ticket is not None and dh.hex()==authority;break
                if p.poll() is not None:raise ValueError('worker unexpectedly exited')
                time.sleep(.001)
            stage=struct.unpack_from('<I',ctrl.buf,40)[0]
            if seq==6:
                if ticket is None or not ack:raise TimeoutError('normal live checkpoint absent')
                raw=bytes(blob.buf[:HEADER.size+ticket['size']]);raw_save('ACTUAL_SHARED_ENVELOPE_6.bin',raw);raw_save('ACTUAL_SHARED_PAYLOAD_6.bin',raw[HEADER.size:])
            else:
                if ticket is not None or ack or stage!=3:raise ValueError('fixed SAA entry cancellation not reached / forbidden publish')
                struct.pack_into('<I',ctrl.buf,44,1);p.kill();p.wait(timeout=max(.001,deadline-time.perf_counter()))
                epoch,size,bh,ph,sh=HEADER.unpack_from(blob.buf);raw=bytes(blob.buf[:HEADER.size+size]);raw_save('RETAINED_ENVELOPE.bin',raw);raw_save('RETAINED_PAYLOAD.bin',raw[HEADER.size:])
            f=time.perf_counter();row=dict(sequence=seq,pid=p.pid,started=s,cutoff=cutoff,chosen_at=chosen_at,finished=f,wall_s=f-s,ticket=ticket,ack_present=ack,final_reap_included=seq==7,reaped=p.poll() is not None,stage=stage);host.append(row);save('HOST_'+str(seq)+'.json',row)
        _,decode=old.codecs();actual=decode(read(OUT/'OWNER_STATE_6.json'));restored=decode(read(OUT/'RESTORE_IN_REQUEST.json'));w=decode(read(OUT/'SAA_CANCEL_WITNESS.json'));checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        row=actual['records'][0]
        ck('selected_restore_inside_same_normal_window',host[0]['started']<=restored['started']<=restored['finished']<=row['started'] and restored['selected_digest']==row['resident_before']==old.pf.digest(cp['state']) and restored['warm_digest']!=restored['selected_digest'])
        ck('restore_zero_policy_events',restored['admission_calls']==restored['physical_calls']==0)
        ck('one_authoritative_callback23_order7',len(actual['records'])==1 and row['callback']==23 and row['order']==7)
        ck('normal_actual_raw_writer_shared_exact',(OUT/'ACTUAL_SHARED_PAYLOAD_6.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_6.bin').read_bytes())
        ck('normal_qualified_actual_commit',row['actual_policy_pointer'] and row['loaded_closure_qualified'] and row['response']['source_qualified'])
        ck('normal_all20_branches',sum(len(c['branch_values']) for c in row['response']['common_scores']['candidate_scores'].values())==20 and row['response']['certified'])
        ck('normal_select_ack_no_reap',host[0]['ack_present'] and not host[0]['reaped'] and not host[0]['final_reap_included'] and host[0]['wall_s']<10.)
        ck('same_resident_next_request',host[0]['pid']==host[1]['pid'] and w['sequence']==7)
        ck('four_complete_branches_before_fifth',w['branch_index']==4 and w['branch_values']==w['completed_values'] and len(w['branch_values'])==4 and w['label']=='own_incumbent')
        ck('cancel_before_score_return_install_plan_checkpoint',not any(w[k] for k in ('scoring_result_returned','policy_installed','current_plan_and_commit','checkpoint_created')) and not any((OUT/n).exists() for n in ('REQUEST_7.json','ATTEMPT_7.json','ACTUAL_WRITER_PAYLOAD_7.bin','OWNER_STATE_7.json')))
        ck('cancel_target_reached_killed_reaped_under10',host[1]['stage']==3 and host[1]['reaped'] and p.returncode is not None and p.returncode!=0 and host[1]['wall_s']<10. and host[1]['ticket'] is None and not host[1]['ack_present'])
        ck('authoritative_epoch6_envelope_preserved',(OUT/'RETAINED_ENVELOPE.bin').read_bytes()==(OUT/'ACTUAL_SHARED_ENVELOPE_6.bin').read_bytes() and (OUT/'RETAINED_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_SHARED_PAYLOAD_6.bin').read_bytes())
        start=read(OUT/'SOURCE_START.json');fault=read(OUT/'SOURCE_AT_SAA_CANCEL.json');ck('actual_start_to_fault_loaded_identity',start['files']==fault['files'] and start['module_files']==fault['module_files'])
        ck('no_model_route_gain',not row['response']['formal_learned_controls'] and not row['response']['accepted_route'])
        ck('registered_bytes',all(sha(pth)==h for pth,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        fail=[x['name'] for x in checks if not x['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,host=host,actual_worker_exit=p.returncode,actual_worker_terminated=True,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[h['wall_s'] for h in host],normal_order=row['order'],discarded_order=w['packet']['admission_input']['current_order'])));return int(bool(fail))
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
    create('qualify_pickup_live_resume_saa_cancel_v1.py',p)
    a=(ROOT/'tools/audit_pickup_timed_restore_live_v1.py').read_text(encoding='utf-8')
    a=a.replace("OUT=P/'timed_restore_live_root_v1';SOURCE=P/'checkpoint_resume_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_3.bin'", "OUT=P/'live_resume_saa_cancel_root_v1';SOURCE=P/'timed_restore_live_root_v1/ACTUAL_SHARED_PAYLOAD_5.bin'")
    a=a.replace("OWNER_STATE_5.json","OWNER_STATE_6.json").replace('index+4','index+6').replace('21+index','23+index').replace("len(actual['records'])==2","len(actual['records'])==1").replace('selected_epoch3_actual_start','selected_epoch5_actual_start').replace("restore['source_epoch']==3","restore['source_epoch']==5").replace('qualify_pickup_timed_restore_live_v1.py','qualify_pickup_live_resume_saa_cancel_v1.py').replace('INDEPENDENT_TIMED_RESTORE_LIVE.json','INDEPENDENT_LIVE_RESUME_SAA_CANCEL.json')
    a=a.replace("ck('normal_second_reaped_exit0',raw['actual_worker_exit']==0 and all(h['ack_present'] for h in raw['host']))","ck('cancelled_worker_return_recorded',raw['actual_worker_terminated'] and raw['actual_worker_exit']!=0 and raw['host'][0]['ack_present'] and not raw['host'][1]['ack_present'])")
    at="    ck('all_actual_audit_inputs_unchanged'"
    pos=a.index(at)
    a=a[:pos]+FAULT_AUDIT+a[pos:]
    a=a.replace('two_actual_new_reveals=True','one_authoritative_reveal_and_one_discarded_partial_scoring=True').replace('all40K10','normal20K10 plus cancelled first4').replace('two new actual reveals','one authoritative actual reveal and next discarded branch')
    create('audit_pickup_live_resume_saa_cancel_v1.py',a)

FAULT_AUDIT='''    # Continue original reference from actual selected epoch6, without importing the new engine.
    witness=decode(read(OUT/'SAA_CANCEL_WITNESS.json'));index=1;ticket_before=old.pf.digest(snapshot())
    try:private_env._run_loop(0,private_clock,sum(t<=private_clock+1e-6 for t in reveals),private_fleet,private_traces,private_served,customers,reveals,float(private_env.tw_end[0].max()))
    except Stop:private_clock=boundary['clock']
    else:raise AssertionError('next actual reveal missing')
    packet=witness['packet'];current=witness['scoring_packet'];o=packet['admission_input']['current_order'];prior=packet['snapshot']['replanner_state'];base=witness['base_response']
    ck('cancel:next_actual_pending',witness['sequence']==7 and packet['admission_input']['pending']==[o] and o not in private_rp._accepted|private_rp._rejected and private_clock==packet['snapshot']['clock'])
    for k,v in snapshot().items():ck('cancel:original_private_reveal:'+k,old.pf.digest(v)==old.pf.digest(witness['private_at_reveal'][k]))
    ck('cancel:preinstall_policy_unchanged',old.pf.digest(private_rp.export_state())==old.pf.digest(witness['before']))
    for k,value in packet['dataset'].items():
        target=np.zeros_like(value);target[0,0]=dataset[k][0,0]
        for j in boundary['visible']:target[0,j]=dataset[k][0,j]
        ck('cancel:public_field:'+k,np.array_equal(value,target))
    suffix={str(v):[int(j) for j in private_rp.export_state()['plan'].get(str(v),private_rp.export_state()['plan'].get(v,[])) if not private_served[j]] for v in range(len(private_fleet))}
    for v in private_fleet:
        if v.status=='committed' and v.committed_next not in (None,0):suffix[str(v.vehicle_id)]=[int(v.committed_next)]+[j for j in suffix[str(v.vehicle_id)] if j!=v.committed_next]
    ck('cancel:public_suffix',old.pf.digest(suffix)==old.pf.digest(prior['plan']))
    rp=UncondDynamicReplanner(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],contract=packet['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10)
    rp.restore_state(copy.deepcopy(prior));env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon']);env.replan_count=packet['snapshot']['replan_count'];fleet,traces,served=restore_recourse_snapshot(packet['snapshot']);rp.on_reveal(env,0,packet['snapshot']['clock'],fleet,served,[o]);replay=rp.export_state()
    plans={str(v):[int(j) for j in replay['plan'].get(v,replay['plan'].get(str(v),[])) if not served[j]] for v in range(len(fleet))}
    for v in fleet:
        if v.status=='committed' and v.committed_next not in (None,0):plans[str(v.vehicle_id)]=[int(v.committed_next)]+[j for j in plans[str(v.vehicle_id)] if j!=v.committed_next]
    replay['plan']=plans
    ck('cancel:actual_current_only_original_admission',old.pf.digest(replay)==old.pf.digest(base['state']) and set(replay['accepted'])-{o}==set(prior['accepted']) and set(replay['rejected'])-{o}==set(prior['rejected']))
    ck('cancel:base_plan_and_both_same_proposals',plans==base['plan']==witness['plan']==witness['plans']['own_incumbent']==witness['plans']['candidate'])
    ck('cancel:scoring_state_hash',snapshot_state_hash(current['snapshot'])==current['snapshot']['state_hash'] and old.pf.digest(current['snapshot']['replanner_state'])==old.pf.digest(replay))
    ev,tr=old.pf.evaluate_known(current,plans)
    ck('cancel:original_base_C0_trace',old.pf.digest(tr)==old.pf.digest(base['known_raw']))
    for k,v in ev.items():
        if k not in ('hard_feasible','failures'):ck('cancel:C0_field:'+k,old.pf.digest(v)==old.pf.digest(base['certificate']['evaluation'][k]))
    ck('cancel:base_hard',ev['hard_feasible'] and not ev['failures'] and base['qualified'])
    srp=EnergyCompleteSAA(budget=current['budget'],capacity=current['capacity'],booking_horizon=current['booking_horizon'],contract=current['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=CondHistoricalSampler(history['history'],n_neighbors=10),reject_penalty={0:5.,1:10.,2:15.},K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='greedy',energy_pricing='marginal',future_policy='density',standby_orders=0)
    srp.restore_state(copy.deepcopy(replay));senv=StrictOnlineEnv(current['dataset'],current['capacity'],current['tw_speed'],current['snapshot']['num_vehicles'],srp,current['contract'],current['booking_horizon']);sfleet,_,sserved=restore_recourse_snapshot(current['snapshot']);clock=current['snapshot']['clock'];srp._sync_id_map(senv,0,clock);ssnap=srp._build_snapshot(senv,0,clock,sfleet,sserved);scenarios=srp.sampler.sample(ssnap,np.random.default_rng(reg['common_seed']),10)
    ck('cancel:original_snapshot_scenarios',old.pf.digest(ssnap)==old.pf.digest(witness['observed_snapshot']) and old.pf.digest(scenarios)==witness['scenario_sha256']==old.pf.digest(witness['scenarios']))
    space,orders=srp._build_space(senv,0,scenarios);ids={id(v):space.n_real+j for j,v in enumerate(orders)};st=srp._sim_states(senv,0,clock,sfleet,sserved,{int(k):v for k,v in plans.items()})
    vals=[float(srp._sim_scenario(senv,0,clock,st,[ids[id(j)] for j in scen],space)) for scen in scenarios[:4]]
    ck('cancel:only_first4_completed',len(witness['branch_values'])==len(witness['completed_values'])==4 and witness['branch_index']==4 and witness['label']=='own_incumbent')
    for j,(v,w) in enumerate(zip(vals,witness['branch_values'])):ck('cancel:original_branch_hex:'+str(j),v.hex()==w.hex()==witness['completed_values'][j].hex())
    ck('cancel:no_mean_install_current_plan_checkpoint',not any(witness[k] for k in ('scoring_result_returned','policy_installed','current_plan_and_commit','checkpoint_created')) and not any((OUT/n).exists() for n in ('REQUEST_7.json','ATTEMPT_7.json','ACTUAL_WRITER_PAYLOAD_7.bin','OWNER_STATE_7.json')))
    ck('cancel:original_all_private_physics',len(reference_events)==len(witness['events']))
    for i,(x,y) in enumerate(zip(reference_events,witness['events'])):
        for k in x:ck('cancel:physics:'+str(i)+':'+k,old.pf.digest(x[k])==old.pf.digest(y[k]))
    envelope=(OUT/'RETAINED_ENVELOPE.bin').read_bytes();head=struct.Struct('<QQ32s32s32s');epoch,size,bh,ph,sh=head.unpack_from(envelope);payload=envelope[head.size:];checkpoint=pickle.loads(payload)
    ck('cancel:retained_original_raw_epoch6',envelope==(OUT/'ACTUAL_SHARED_ENVELOPE_6.bin').read_bytes() and payload==(OUT/'RETAINED_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_SHARED_PAYLOAD_6.bin').read_bytes() and hashlib.sha256(payload).digest()==ph and len(payload)==size and epoch==checkpoint['epoch']==6 and sh.hex()==ticket_before==old.pf.digest(actual['final']))
    h=raw['host'][1];ck('cancel:target_elapsed_and_actual_reap',h['stage']==3 and h['ticket'] is None and not h['ack_present'] and h['reaped'] and h['final_reap_included'] and h['finished']-h['started']==h['wall_s'] and h['started']<witness['entered']<h['cutoff']<=h['finished']<h['started']+10.)
    begin=read(OUT/'SOURCE_START.json');fault=read(OUT/'SOURCE_AT_SAA_CANCEL.json');ck('cancel:start_to_actual_fault_closure',begin['files']==fault['files'] and begin['module_files']==fault['module_files'])
    for p,h in fault['files'].items():ck('cancel:actual_loaded:'+p,sha(p)==h)
'''
if __name__=='__main__':main()
