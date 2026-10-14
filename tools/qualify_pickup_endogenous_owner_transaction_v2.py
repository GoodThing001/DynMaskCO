"""Endogenous current admission/K10/C0 in the actual private environment owner.

One observed engineering request plus one new post-selection cancellation; no learned proposal or scientific day.
"""
import copy,json,os,pickle,struct,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_observed_day_extension_v1 as observed
import pickup_undecided_public_v1 as projector
import pickup_pending_transaction_v1 as engine
import pickup_partial_common_v1,pickup_partial_trace_v1,pickup_partial_known_v1,pickup_partial_validation_v1
import hard_admission
import rolling_optimizer
from run_exp_rule_control import UncondDynamicReplanner
from energy_complete_saa import EnergyCompleteSAA
from scenario_saa import CondHistoricalSampler
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
import joint_install as ji
from pickup_loaded_closure_v1 import capture,sha
ROOT=old.ROOT;P=old.P;SOURCE=P/'observed_day_extension_root_v1';PREFIX=P/'live_event_prefix_root_v1';OUT=P/'endogenous_owner_transaction_root_v2'
OLD_FAILURE=P/'endogenous_owner_transaction_root_v1/LAZY_IMPORT_DIAGNOSIS.json';OLD_REGISTRATION=P/'endogenous_owner_transaction_root_v1/REGISTRATION.json'
CONTROL=struct.Struct('<QdII');REPLY=struct.Struct('<QII32s')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(name,value):
    text=json.dumps(value,indent=2,allow_nan=False)
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
def sealed():
    _,decode=old.codecs()
    with (SOURCE/'REQUESTS_STARTED.jsonl').open(encoding='utf-8') as f:a=decode(json.loads(next(f)))
    with (SOURCE/'PREPARED_RESPONSES.jsonl').open(encoding='utf-8') as f:b=decode(json.loads(next(f)))
    assert a['callback']==b['callback']==15
    return a,b
class BeforeAdmission(Exception):pass
class Bundle:
    pass
def initialize():
    template,dataset=observed.setup();_,decode=old.codecs();initial=decode(read(PREFIX/'RESULT.json'))['final'];b=Bundle()
    base=ji.transactional_class(C0ReserveReplanner)
    class Policy(base):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            b.clock=clock;b.visible=visible
            raise BeforeAdmission()
    rp=Policy(template['budget'],template['capacity'],template['booking_horizon'],template['contract'],0.);rp.restore_state(copy.deepcopy(initial['state']))
    env=StrictOnlineEnv(copy.deepcopy(dataset),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],rp,template['contract'],template['booking_horizon'])
    fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy();env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];env.force_suffix={}
    customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0];reveals=sorted(set(float(env.reveal_time[0,j]) for j in customers if env.reveal_time[0,j]>0));clock=float(initial['clock'])
    try:env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,reveals,float(env.tw_end[0].max()))
    except BeforeAdmission:pass
    else:raise ValueError('registered next reveal not reached')
    b.env=env;b.rp=rp;b.fleet=fleet;b.traces=traces;b.served=served;b.customers=customers;b.reveals=reveals
    b.current=int(sealed()[0]['order']);b.reveal_index=sum(t<=b.clock+1e-6 for t in reveals)
    return b
def state(b):return dict(state=b.rp.export_state(),fleet=b.fleet,traces=b.traces,served=b.served,clock=b.clock,event_id=b.env.event_id,replan_count=b.env.replan_count)
def sources():
    files=[ROOT/'tools/recover_pickup_endogenous_owner_v2.py',OLD_FAILURE,OLD_REGISTRATION,old.HISTORY,ROOT/'tools/pickup_pending_transaction_v1.py',Path(__file__),ROOT/'tools/audit_pickup_endogenous_owner_transaction_v2.py',ROOT/'tools/run_pickup_endogenous_owner_transaction_bounded_v2.py',ROOT/'tools/pickup_loaded_closure_v1.py',PREFIX/'RESULT.json',SOURCE/'REGISTRATION.json',SOURCE/'RUN_EXIT.json',SOURCE/'REQUESTS_STARTED.jsonl',SOURCE/'PREPARED_RESPONSES.jsonl',observed.fixture.TRACE]
    return {str(p):sha(p) for p in files}
def next_physical_clock(b):
    clocks=[float(b.env.tw_end[0].max())]+[t for t in b.reveals if t>b.clock+1e-6]
    if b.env.booking_horizon>b.clock+1e-6:clocks.append(b.env.booking_horizon)
    for v in b.fleet:
        if v.status=='committed' and v.committed_finish is not None:clocks.append(v.committed_finish)
        elif v.status=='returning' and v.return_finish is not None:clocks.append(v.return_finish)
    value=min(clocks)
    if value<=b.clock:raise ValueError('physical progress')
    return value
def worker():
    psutil.Process().cpu_affinity([7]);case=int(sys.argv[4]);ctrl=SharedMemory(name=sys.argv[2]);reply=SharedMemory(name=sys.argv[3]);b=initialize();a,sealed_response=sealed();history=pickle.loads(old.HISTORY.read_bytes());baseline=old.pf.digest(state(b));owner=ji.JointPolicyOwner(b.rp,b.fleet,b.clock,b.env.event_id);owner.signature();capture();begin=capture(sources());save('SOURCE_START_'+str(case)+'.json',begin)
    print(json.dumps(dict(ready=True,pid=os.getpid(),baseline=baseline,clock=b.clock,event_id=b.env.event_id,loaded_files=len(begin['files']))),flush=True)
    last=0;record=dict(case=case,pid=os.getpid(),baseline=baseline)
    def mark(n,fault):
        struct.pack_into('<I',ctrl.buf,40,n)
        if fault==n:time.sleep(12.)
    try:
        while True:
            seq,deadline,op,fault=CONTROL.unpack_from(ctrl.buf)
            if not seq or seq==last:time.sleep(.001);continue
            last=seq
            if op==3:break
            if time.perf_counter()>=deadline:raise TimeoutError('expired command')
            if op==1:
                started=time.perf_counter();mark(1,fault)
                packet=projector.export_undecided(b.env,0,b.clock,b.env.event_id,b.reveal_index,b.fleet,b.traces,b.served,b.customers,b.current)
                mark(2,fault);payload=pickle.dumps(packet,protocol=5)
                if len(payload)>2*1024*1024:raise ValueError('bounded public encoding')
                decoded=pickle.loads(payload);public_hash=old.pf.digest(decoded)
                if public_hash!=old.pf.digest(a['packet']):raise ValueError('actual next-reveal public identity')
                token=owner.begin(str(b.current),b.current,deadline,public_hash);response=engine.compute(decoded,history,old.E/'scripts',deadline)
                end_compute=capture(sources());response['source_qualified']=begin['files']==end_compute['files'] and begin['module_files']==end_compute['module_files'];save('SOURCE_COMPUTE_END_'+str(case)+'.json',end_compute)
                response['source_start']=begin['files'];response['source_end']=end_compute['files']
                encode,_=old.codecs();save('RESPONSE_ATTEMPT_'+str(case)+'.json',encode(response))
                installed,reason=owner.commit(token,response)
                if not installed:raise ValueError(reason)
                published_pointer=(b.rp._policy is response['compiled_state'])
                b.env._plan_and_commit(0,b.clock,b.fleet,b.traces,b.served,b.customers)
                record.update(public_packet=packet,encoded_bytes=len(payload),public_digest=public_hash,compiled_sha256=response['compiled_state_sha256'],actual_policy_pointer=published_pointer,after_commit=copy.deepcopy(state(b)),component_started=started,commit_finished=time.perf_counter(),prepared_response_reused=False,new_admission_or_SAA=True,response=response)
                encode,_=old.codecs();save('CANDIDATE_RESULT_'+str(case)+'.json',encode(record))
                REPLY.pack_into(reply.buf,0,seq,1,2,bytes.fromhex(response['compiled_state_sha256']))
            elif op==2:
                # Caller has selected this resident process as the actual owner.
                mark(3,fault);before=old.pf.digest(state(b));b.clock=next_physical_clock(b);b.env.event_id+=1;b.env._advance_fleet(0,b.clock,b.fleet,b.traces,b.served)
                after=old.pf.digest(state(b));end=capture(sources());record.update(primary_advance_before=before,after_primary_advance=copy.deepcopy(state(b)),primary_advance_digest=after,loaded_closure_qualified=begin['files']==end['files'],source_changed=[p for p,h in begin['files'].items() if end['files'].get(p)!=h],new_loaded_files=sorted(set(end['files'])-set(begin['files'])))
                encode,_=old.codecs();save('SOURCE_END_'+str(case)+'.json',end);save('OWNER_RESULT_'+str(case)+'.json',encode(record));REPLY.pack_into(reply.buf,0,seq,1,3,bytes.fromhex(after))
            else:raise ValueError('command')
    except Exception:
        save('WORKER_FAILURE_'+str(case)+'.json',dict(traceback=traceback.format_exc()));REPLY.pack_into(reply.buf,0,last,2,0,bytes(32))
    finally:ctrl.close();reply.close()
def execute(case,fault,baseline_owner):
    ctrl=SharedMemory(create=True,size=64);reply=SharedMemory(create=True,size=REPLY.size);ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(REPLY.size);stream=(OUT/('worker_'+str(case)+'.stderr.txt')).open('x',encoding='utf-8')
    process=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,reply.name,str(case)],stdout=subprocess.PIPE,stderr=stream,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    primary=baseline_owner;prior=old.pf.digest(state(baseline_owner));cap=10.;reserve=2.
    try:
        ready=json.loads(process.stdout.readline());assert ready['ready'] and ready['baseline']==prior
        started=time.perf_counter();deadline=started+cap;compute=deadline-reserve;CONTROL.pack_into(ctrl.buf,0,1,compute,1,fault);published=None;advance=None;error=None
        def wait(seq):
            while time.perf_counter()<compute:
                got,kind,stage,data=REPLY.unpack_from(reply.buf)
                if got==seq:
                    if kind!=1:raise RuntimeError('owner worker failed')
                    return data.hex()
                if process.poll() is not None:raise RuntimeError('owner worker exited')
                time.sleep(.001)
            return None
        published=wait(1)
        if published is not None:
            if len(published)!=64:raise ValueError('bounded compiled identity')
            # Real owner selection is one process-reference assignment. The old
            # live bundle is untouched; no parent projection, pickle or guard.
            primary=process;promotion=time.perf_counter();CONTROL.pack_into(ctrl.buf,0,2,compute,2,fault);advance=wait(2)
            if advance is None:primary=baseline_owner;error='advance_cancelled'
            else:CONTROL.pack_into(ctrl.buf,0,3,compute,3,0)
        else:error='cancelled';promotion=None
        reached=struct.unpack_from('<I',ctrl.buf,40)[0]
        if error and process.poll() is None:process.kill()
        process.wait(timeout=max(.001,deadline-time.perf_counter()));finished=time.perf_counter()
        return dict(case=case,fault=fault,pid=process.pid,ready=ready,baseline=prior,old_owner_after=old.pf.digest(state(baseline_owner)),prepared=published,primary_advance=advance,actual_process_owner_selected=primary is process,promotion=promotion,error=error,stage_reached=reached,started=started,finished=finished,wall_s=finished-started,cap_s=cap,reserve_s=reserve,reaped=process.poll() is not None,timed_parent_public_projection_or_pickle=False)
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=5)
        process.stdout.close();stream.close()
        for memory in (ctrl,reply):memory.close();memory.unlink()
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),selection='only original first callback15; one endogenous normal and one post-selection/pre-physical-entry cancellation',prepared_response_reused=False,new_admission_or_SAA=True,new_scientific_data=False,final_scientific_seeds_used=False,normal_cap_s=10.,normal_reserve_s=2.,fault_cap_s=10.,fault_reserve_s=2.,delay_s=12.,common_seed=202610111301,K=10,initialization_outside_component_clock=True,actual_owner_scope='restored original private next-reveal environment; endogenous current-only own_uncond_dynamic, own_incumbent proposal, original K10/full C0/compile, real plan_and_commit and one physical advance; host rollback discards delayed private branch',full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False))
    try:
        baseline=initialize();records=[execute(0,0,baseline),execute(1,3,baseline)];checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        for r in records:
            n=str(r['case']);ck(n+':actual_old_owner_unchanged',r['baseline']==r['old_owner_after']);ck(n+':total_with_cleanup_reserve',r['wall_s']<r['cap_s'] and r['reaped']);ck(n+':no_parent_public_encoding_or_decode',not r['timed_parent_public_projection_or_pickle'])
            if r['fault']:
                ck(n+':post_selection_cancel_rollback',r['stage_reached']==3 and r['error']=='advance_cancelled' and r['prepared'] is not None and r['promotion'] is not None and not r['actual_process_owner_selected'])
            else:
                encode,decode=old.codecs();owner=decode(read(OUT/'OWNER_RESULT_0.json'));ck(n+':actual_owner_and_physical_progress',r['actual_process_owner_selected'] and r['primary_advance']==owner['primary_advance_digest'] and owner['after_primary_advance']['clock']>owner['after_commit']['clock']);ck(n+':actual_compiled_pointer',owner['actual_policy_pointer']);ck(n+':actual_loaded_component_closure',owner['loaded_closure_qualified'])
        reg=read(OUT/'REGISTRATION.json');ck('registered_sources_unchanged',all(sha(p)==h for p,h in reg['sources'].items()));failed=[c['name'] for c in checks if not c['pass_']]
        save('RESULT.json',dict(records=records,checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,endogenous_observed_owner_component_qualified=not failed,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(failed))));print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed,walls=[r['wall_s'] for r in records])));return int(bool(failed))
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc()));save('RUN_EXIT.json',dict(exit_code=1));raise
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
