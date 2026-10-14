"""Interruptible sealed-response decode/hash/actual guard in a disposable policy replica.

No C0 or SAA replay, no actual simulator owner promotion and no new science.
"""
import copy,json,os,pickle,struct,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
from run_exp_energy_c0 import C0ReserveReplanner
import joint_install as ji
from pickup_loaded_closure_v1 import capture,sha
ROOT=old.ROOT;P=old.P;SOURCE=P/'observed_day_extension_root_v1';OUT=P/'parent_decode_guard_shadow_root_v2'
V1=P/'parent_decode_guard_shadow_root_v1'
CONTROL=struct.Struct('<QdI');REPLY=struct.Struct('<QII32s');CAPACITY=2*1024*1024
def save(name,value):
    data=json.dumps(value,indent=2,allow_nan=False)
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(data)
def endpoint_lines(path):
    first=last=None
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if first is None:first=line
            last=line
    if first is None:raise ValueError('empty sealed input')
    return [json.loads(first),json.loads(last)]
def fixtures():
    _,decode=old.codecs();a=endpoint_lines(SOURCE/'REQUESTS_STARTED.jsonl');b=endpoint_lines(SOURCE/'PREPARED_RESPONSES.jsonl')
    result=[]
    for x,y in zip(a,b):
        x=decode(x);y=decode(y);packet=x['packet'];response=y['response']
        expected=ji.digest(ji.state_dict(response['compiled_state']))
        assert response['compiled_state_sha256']==expected and response['packet_digest']==old.pf.digest(packet)
        result.append(dict(callback=x['callback'],packet=packet,response=response,expected=expected,prior=ji.digest(packet['snapshot']['replanner_state'])))
    assert [r['callback'] for r in result]==[15,80]
    return result
def identities():
    items=[Path(__file__),ROOT/'tools/prepare_pickup_parent_shadow_closure_recovery_v2.py',ROOT/'tools/audit_pickup_parent_decode_guard_shadow_v2.py',ROOT/'tools/pickup_loaded_closure_v1.py',ROOT/'tools/run_pickup_parent_shadow_bounded_v2.py',SOURCE/'REGISTRATION.json',SOURCE/'RUN_EXIT.json',SOURCE/'REQUESTS_STARTED.jsonl',SOURCE/'PREPARED_RESPONSES.jsonl',SOURCE/'INDEPENDENT_OBSERVED_DAY_EXTENSION.json']
    items += sorted(p for p in V1.iterdir() if p.is_file())
    return {str(p):sha(p) for p in items}
def worker():
    psutil.Process().cpu_affinity([7]);control=SharedMemory(name=sys.argv[2]);incoming=SharedMemory(name=sys.argv[3]);outgoing=SharedMemory(name=sys.argv[4]);case=int(sys.argv[5]);size=int(sys.argv[6])
    # Predeclared dependency and replica setup; only a second input decode is timed.
    blob=bytes(incoming.buf[:size]);warm=pickle.loads(blob);packet=warm['packet'];rpclass=ji.transactional_class(C0ReserveReplanner)
    stage=[0];armed=[False]
    def mark(n):
        stage[0]=n;struct.pack_into('<I',control.buf,32,n)
        if fault==n:time.sleep(2.)
    class Replica(rpclass):
        def __setattr__(self,key,value):
            if key=='_policy' and armed[0]:mark(5)
            super().__setattr__(key,value)
    rp=Replica(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.)
    rp.restore_state(copy.deepcopy(packet['snapshot']['replanner_state']));fleet,_,_=restore_recourse_snapshot(packet['snapshot'])
    owner=ji.JointPolicyOwner(rp,fleet,float(packet['snapshot']['clock']),int(packet['snapshot']['event_id']))
    old.codecs();owner.signature();capture();start=capture(identities());save('SOURCE_START_'+str(case)+'.json',start)
    print(json.dumps(dict(ready=True,pid=os.getpid(),loaded_files=len(start['files']))),flush=True)
    try:
        while not CONTROL.unpack_from(control.buf)[0]:time.sleep(.001)
        seq,deadline,fault=CONTROL.unpack_from(control.buf);started=time.perf_counter();mark(1);value=pickle.loads(blob)
        packet=value['packet'];response=copy.copy(value['response']);mark(2)
        if old.pf.digest(packet)!=response['packet_digest'] or snapshot_state_hash(packet['snapshot'])!=packet['snapshot']['state_hash']:raise ValueError('sealed public identity')
        original_signature=owner.signature
        def signature():mark(3);return original_signature()
        owner.signature=signature
        token=owner.begin(str(packet['admission_input']['current_order']),packet['admission_input']['current_order'],deadline,response['packet_digest'])
        response['deadline']=deadline
        original_digest=ji.digest
        def guarded_digest(x):mark(4);return original_digest(x)
        ji.digest=guarded_digest;armed[0]=True
        installed,reason=owner.commit(token,response);armed[0]=False;ji.digest=original_digest
        if not installed:raise ValueError(reason)
        if rp._policy is not response['compiled_state']:raise ValueError('replica atomic pointer')
        mark(6);capsule=bytes.fromhex(original_digest(rp.export_state()))
        end=capture(identities());save('SOURCE_END_'+str(case)+'.json',end)
        added=sorted(set(end['files'])-set(start['files']));changed=[p for p,h in start['files'].items() if end['files'].get(p)!=h]
        save('REPLICA_RESULT_'+str(case)+'.json',dict(installed=True,reason=reason,stage=stage[0],started=started,finished=time.perf_counter(),compiled_sha256=capsule.hex(),source_changed=changed,new_loaded_files=added,loaded_closure_qualified=not added and not changed,actual_simulator_owner_promoted=False))
        REPLY.pack_into(outgoing.buf,0,seq,1,stage[0],capsule)
    except Exception:
        save('WORKER_FAILURE_'+str(case)+'.json',dict(traceback=traceback.format_exc(),stage=stage[0]));REPLY.pack_into(outgoing.buf,0,1,2,stage[0],bytes(32))
    finally:control.close();incoming.close();outgoing.close()
def execute(case,item,fault):
    blob=pickle.dumps(item,protocol=5)
    if len(blob)>CAPACITY:raise ValueError('fixture exceeds fixed shared input')
    control=SharedMemory(create=True,size=64);incoming=SharedMemory(create=True,size=CAPACITY);outgoing=SharedMemory(create=True,size=REPLY.size)
    control.buf[:]=bytes(64);outgoing.buf[:]=bytes(REPLY.size);incoming.buf[:len(blob)]=blob
    cap=10. if not fault else .4;reserve=2. if not fault else .15
    stream=(OUT/('worker_'+str(case)+'.stderr.txt')).open('x',encoding='utf-8')
    process=subprocess.Popen([sys.executable,'-B',str(Path(__file__)), '--worker',control.name,incoming.name,outgoing.name,str(case),str(len(blob))],stdout=subprocess.PIPE,stderr=stream,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    try:
        # Setup is outside component timing, bounded separately by stage supervisor.
        ready=json.loads(process.stdout.readline())
        if not ready.get('ready'):raise ValueError('replica setup')
        started=time.perf_counter();deadline=started+cap;compute_deadline=deadline-reserve
        CONTROL.pack_into(control.buf,0,1,compute_deadline,fault);prepared=None;error=None
        while time.perf_counter()<compute_deadline:
            seq,kind,stage,digest=REPLY.unpack_from(outgoing.buf)
            if seq:
                if kind!=1:error='worker_error'
                else:prepared=digest.hex()
                break
            if process.poll() is not None:error='worker_exited';break
            time.sleep(.001)
        reached=struct.unpack_from('<I',control.buf,32)[0]
        if prepared is None and error is None:error='cancelled'
        # Host only receives a fixed digest; it never decodes or installs a policy.
        host_capsule=item['prior']
        if prepared is not None:
            if prepared!=item['expected']:raise ValueError('fixed prepared capsule identity')
            host_capsule=prepared
        if process.poll() is None:process.kill()
        process.wait(timeout=max(.001,deadline-time.perf_counter()));finished=time.perf_counter()
        return dict(case=case,callback=item['callback'],fault=fault,ready=ready,stage_reached=reached,error=error,prepared=prepared,host_capsule=host_capsule,prior_capsule=item['prior'],expected_capsule=item['expected'],cap_s=cap,reserve_s=reserve,started=started,finished=finished,wall_s=finished-started,reaped=process.poll() is not None,pid=process.pid,timed_parent_pickle_decode=False,actual_simulator_owner_promoted=False)
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=5)
        process.stdout.close();stream.close()
        for memory in (control,incoming,outgoing):memory.close();memory.unlink()
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=identities(),selection='first and last sealed observed-day-extension prepared response: callbacks15/80; v2 only two normal closure recovery; six original v1 stage faults retained without rerun',normal_cap_s=10.,normal_reserve_s=2.,fault_cap_s=.4,fault_reserve_s=.15,delay_s=2.,original_v1_registration_sha256=sha(V1/'REGISTRATION.json'),original_v1_result_sha256=sha(V1/'RESULT.json'),original_failures_preserved=True,preinitialize_actual_audit_io=True,new_C0_or_SAA_execution=False,new_scientific_data=False,one_predeclared_input_decode_warmup_per_replica=True,replica_setup_preloaded=True,scope='prepared response decode/public digest/actual JointPolicyOwner guards/shadow pointer/opaque capsule; not live owner promotion',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False))
    try:
        items=fixtures();records=[execute(0,items[0],0),execute(1,items[1],0)]
        checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        for record in records:
            n=str(record['case']);ck(n+':reserved_cleanup_total',record['wall_s']<record['cap_s'] and record['reaped'])
            ck(n+':fixed_host_receive_no_decode',not record['timed_parent_pickle_decode'] and not record['actual_simulator_owner_promoted'])
            if record['fault']:
                ck(n+':actual_stage_cancelled',record['stage_reached']==record['fault'] and record['error']=='cancelled')
                ck(n+':no_host_capsule_publish',record['prepared'] is None and record['host_capsule']==record['prior_capsule'])
            else:
                ck(n+':exact_prepared_digest',record['prepared']==record['expected_capsule'] and record['error'] is None)
                result=json.loads((OUT/('REPLICA_RESULT_'+n+'.json')).read_text());ck(n+':actual_replica_guard_commit',result['installed'] and result['compiled_sha256']==record['prepared'])
                ck(n+':actual_loaded_component_closure',result['loaded_closure_qualified'])
        reg=json.loads((OUT/'REGISTRATION.json').read_text());ck('all_registered_input_source_identity_end',all(sha(p)==h for p,h in reg['sources'].items()))
        failed=[c['name'] for c in checks if not c['pass_']]
        save('RESULT.json',dict(records=records,checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,prepared_shadow_component_qualified=not failed,live_owner_handoff_qualified=False,parent_public_projection_encode_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(failed))))
        print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed,walls=[r['wall_s'] for r in records])));return int(bool(failed))
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc()));save('RUN_EXIT.json',dict(exit_code=1));raise
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
