"""Finite integration QA from restored private state through guarded publication.
No continuous live simulator, learned proposer or hard realtime qualification.
"""
import copy,json,pickle,struct,sys,time,os
from pathlib import Path
from datetime import datetime,timezone
import qualify_pickup_private_public_transport_v2 as fixture
import pickup_private_joint_compute_v1 as engine
from pickup_shared_joint_transport_v1 import JointTransport,HEADER,CONTROL,MAX_PAYLOAD
ROOT=fixture.ROOT;P=fixture.P;E=fixture.E;OUT=P/'private_joint_integration_root_v1'
PHASES=['private_projection','admission','proposal','scoring','certification','compile','result_encoding','opaque_transfer']
def sha(p):return fixture.sha(p)
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def worker():
    import numpy as np,psutil,public_state
    from multiprocessing.shared_memory import SharedMemory
    psutil.Process().cpu_affinity([7])
    if np.__version__!='1.26.4':raise RuntimeError('Pinned original NumPy required')
    control=SharedMemory(name=sys.argv[2]);output=SharedMemory(name=sys.argv[4])
    states={i:fixture.state(i,0) for i in (0,9,13)}
    # These are old observed engineering environments; restoration is declared setup.
    manifest=engine.pf.source_hashes(E/'scripts')
    for module in (fixture,engine,sys.modules['pickup_shared_joint_transport_v1'],sys.modules['pickup_interruptible_public_transport'],sys.modules['qualify_pickup_joint_policy_transaction_root_v3']):
        manifest[str(Path(module.__file__).resolve())]=sha(module.__file__)
    for p in (Path(__file__),fixture.FULL,fixture.TRACE,engine.HISTORY,engine.OUT/'joint_install.py',Path(sys.executable),Path(np.__file__),Path(np.core._multiarray_umath.__file__)):
        manifest[str(p)]=sha(p)
    stream=sys.stdout.buffer
    def send(kind,data):stream.write(HEADER.pack(kind,len(data))+data);stream.flush()
    send(0,json.dumps({'sources':manifest,'cpu':psutil.Process().cpu_affinity(),'numpy':np.__version__}).encode())
    while not struct.unpack_from('<Q',control.buf,0)[0]:time.sleep(.001)
    _,index,deadline,fault=CONTROL.unpack_from(control.buf,0)
    def phase(name):
        code=PHASES.index(name)+1;struct.pack_into('<I',control.buf,32,code)
        if fault==code:time.sleep(8.)
        if time.perf_counter()>=deadline:raise TimeoutError(name)
    try:
        row,env,fleet,traces,served=states[index];old=row['public_request'];clock=old['snapshot']['clock']
        phase('private_projection')
        projected=public_state.export_public(env,0,clock,old['snapshot']['event_id'],old['snapshot']['reveal_idx'],fleet,traces,served,list(range(1,env.num_nodes)))
        result=engine.compute(projected,row['order'],deadline,phase,manifest)
        phase('result_encoding');data=pickle.dumps(result,protocol=5)
        if len(data)>MAX_PAYLOAD:raise ValueError('Full engineering response exceeds registered bound')
        phase('opaque_transfer')
        output.buf[HEADER.size:HEADER.size+len(data)]=data
        struct.pack_into('<I',output.buf,1,len(data));output.buf[0]=2
    except Exception as exc:send(3,json.dumps({'error':repr(exc)}).encode())
    finally:control.close();output.close()
def main():
    import psutil,numpy as np
    from run_exp_energy_c0 import C0ReserveReplanner
    from recourse_snapshot import restore_recourse_snapshot
    from joint_install import transactional_class,JointPolicyOwner,state_dict
    psutil.Process().cpu_affinity([7])
    if np.__version__!='1.26.4':raise RuntimeError('Pinned runtime required')
    OUT.mkdir(exist_ok=False);encode,decode=engine.codecs()
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'normal_indices':[0,9,13],'route_mode':'own_incumbent','admission':'own_uncond_dynamic','K':10,'common_seed':202610111301,'normal_cap_s':10.,'fault_cap_s':6.,'delay_s':8.,'phases':PHASES,'CPU':[7],'max_payload_bytes':MAX_PAYLOAD,
         'scope':'Restored old private environments, private projection + admission + incumbent proposal + common K10 + C0 + compile + shared return + parent decode + guarded commit',
         'selection':'old first/mutable-width/outcome-selected rejection engineering only','new_science':False,'formal_arms':False,'parent_decode_preemptible':False,'live_event_ownership_integrated':False,'whole_live_10s_qualified':False,'full_loaded_dependency_closure_qualified':False,
         'inputs':{str(p):sha(p) for p in (fixture.FULL,fixture.TRACE,engine.HISTORY)},'tools':{str(p):sha(p) for p in (Path(__file__),Path(engine.__file__),ROOT/'tools/pickup_shared_joint_transport_v1.py')}}
    save('REGISTRATION.json',reg);records=[];faults=[];checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    cls=transactional_class(C0ReserveReplanner)
    for index in reg['normal_indices']:
        row,env,fleet,traces,served=fixture.state(index,0);packet=row['public_request']
        rp=cls(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.)
        rp.restore_state(packet['snapshot']['replanner_state']);live=JointPolicyOwner(rp,fleet,packet['snapshot']['clock'],packet['snapshot']['event_id'])
        before=rp.export_state();w=JointTransport(Path(__file__))
        try:
            ready=json.loads(w.ready[1]);started=time.perf_counter();deadline=started+10.
            token=live.begin(str(row['order']),row['order'],deadline,engine.pf.digest(packet))
            # Start ask with remaining time; the initial guard/digest is charged.
            got=w.ask(index,max(0.,deadline-time.perf_counter()))
            ck(str(index)+':complete',got['kind']=='opaque_joint_response')
            if got['kind']!='opaque_joint_response':raise RuntimeError(got)
            # Actual receipt deadline is used by both preparation and commit.
            token['deadline']=got['deadline']
            response=pickle.loads(got.pop('bytes'));decoded=time.perf_counter()
            original=engine.pf.certificate
            def forbidden(*a,**kw):raise RuntimeError('No parent physical recertification')
            engine.pf.certificate=forbidden
            try:installed,reason=live.commit(token,response)
            finally:engine.pf.certificate=original
            committed=time.perf_counter();after=rp.export_state();w.close();finished=time.perf_counter()
            ck(str(index)+':published',installed and reason=='published')
            ck(str(index)+':guard_compile_export',after==state_dict(response['compiled_state']))
            ck(str(index)+':all_stage_reached',got['stages']==PHASES)
            ck(str(index)+':decode_commit_cleanup_charged',started<=decoded<=committed<=finished<deadline)
            ck(str(index)+':source_end',response['source_end']==ready['sources'])
            ck(str(index)+':public_identity',engine.pf.digest(response['projected_packet'])==engine.pf.digest(packet))
            ck(str(index)+':K10',all(len(c['branch_values'])==10 for c in response['common_scores']['candidate_scores'].values()))
            records.append({'index':index,'order':row['order'],'packet':packet,'before':before,'after':after,'response':response,'transport':got,'ready':ready,'installed':installed,'reason':reason,'started':started,'decoded':decoded,'committed':committed,'finished':finished,'wall_s':finished-started})
        finally:w.close()
    for code,name in enumerate(PHASES,1):
        w=JointTransport(Path(__file__))
        try:
            got=w.ask(9,6.,code)
            ck(name+':actual_target_reached',name in got['stages'])
            ck(name+':cancel_reaped',got['kind']=='cancelled' and got['worker_reaped'])
            faults.append({'phase':name,'response':got,'publication_attempted':False})
        finally:w.close()
    ck('registered_inputs_tools_end',all(sha(p)==v for section in ('inputs','tools') for p,v in reg[section].items()))
    result={'checks':checks,'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'records':records,'faults':faults,'new_science':False,'whole_live_10s_qualified':False,'parent_decode_preemptible':False,'stable_online_gain':False,'full_loaded_dependency_closure_qualified':False}
    save('RESULT.json',encode(result));save('RUN_EXIT.json',{'exit_code':int(result['failed']>0)})
    print(json.dumps({'checks':len(checks),'failed':result['failed'],'normal':len(records),'faults':len(faults),'walls':[x['wall_s'] for x in records]}))
    return int(result['failed']>0)
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
