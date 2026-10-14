"""Bounded private simulation checkpoint publication under lost physical ACK.

Only one next natural physical boundary from the sealed callback17 endpoint.
No admission, route proposal, SAA or scientific input generation.
"""
import copy, ctypes, hashlib, json, os, pickle, struct, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_observed_day_extension_v1 as observed
import qualify_pickup_endogenous_owner_transaction_v2 as prev
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
from pickup_loaded_closure_v1 import capture, sha
ROOT=old.ROOT; OUT=old.P/'checkpoint_ack_root_v1'
SOURCE=old.P/'resident_two_events_root_v1/RESIDENT_RESULT.json'
CONTROL=struct.Struct('<QdII'); HEADER=struct.Struct('<QQ32s32s32s')
MAX=2*1024*1024
K=ctypes.WinDLL('kernel32',use_last_error=True)
K.CreateEventW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_wchar_p];K.CreateEventW.restype=ctypes.c_void_p
K.OpenEventW.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_wchar_p];K.OpenEventW.restype=ctypes.c_void_p
K.SetEvent.argtypes=[ctypes.c_void_p];K.SetEvent.restype=ctypes.c_int
K.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];K.WaitForSingleObject.restype=ctypes.c_uint32
K.CloseHandle.argtypes=[ctypes.c_void_p]
def event(name,create=False):
    h=K.CreateEventW(None,True,False,name) if create else K.OpenEventW(0x100002,False,name)
    if not h:raise ctypes.WinError(ctypes.get_last_error())
    return h
def signaled(h):
    v=K.WaitForSingleObject(h,0)
    if v not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
    return v==0
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def native(v):
    if isinstance(v,np.ndarray):return native(v.tolist())
    if isinstance(v,np.generic):return native(v.item())
    if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [native(x) for x in v]
    return v
def save(n,v):
    s=json.dumps(native(v),indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def sources():
    files=[Path(__file__),SOURCE,observed.fixture.TRACE,old.P/'undecided_public_input_root_v2/RESULT.json',ROOT/'tools/pickup_loaded_closure_v1.py',ROOT/'tools/run_pickup_checkpoint_ack_bounded_v1.py',Path(sys.executable)]
    return {str(p):sha(p) for p in files}
def restore(template,dataset,initial):
    b=prev.Bundle();b.rp=C0ReserveReplanner(template['budget'],template['capacity'],template['booking_horizon'],template['contract'],0.)
    b.rp.restore_state(copy.deepcopy(initial['state']))
    b.env=StrictOnlineEnv(copy.deepcopy(dataset),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],b.rp,template['contract'],template['booking_horizon'])
    b.env.event_id=initial['event_id'];b.env.replan_count=initial['replan_count'];b.env.force_suffix={}
    b.fleet=copy.deepcopy(initial['fleet']);b.traces=copy.deepcopy(initial['traces']);b.served=initial['served'].copy();b.clock=initial['clock']
    b.customers=[int(j) for j in range(1,b.env.num_nodes) if b.env.demands[0,j]>0]
    b.reveals=sorted(set(float(b.env.reveal_time[0,j]) for j in b.customers if b.env.reveal_time[0,j]>0))
    return b
def state(b):return prev.state(b)
def worker():
    psutil.Process().cpu_affinity([7]);assert np.__version__=='1.26.4'
    ctrl=SharedMemory(name=sys.argv[2]);blob=SharedMemory(name=sys.argv[3]);reply=SharedMemory(name=sys.argv[4]);latch=event(sys.argv[5]);case=int(sys.argv[6]);recover='--recover' in sys.argv
    encode,decode=old.codecs();template,dataset=observed.setup();initial=decode(read(SOURCE))['final'];baseline=old.pf.digest(initial)
    b=None if recover else restore(template,dataset,initial)
    # Warm precisely the closure collector and canonical codec before sealing.
    old.pf.digest(initial);capture();begin=capture(sources());save(('RECOVERY_' if recover else '')+'SOURCE_START_'+str(case)+'.json',begin)
    print(json.dumps(dict(ready=True,pid=os.getpid(),baseline=baseline)),flush=True)
    try:
        while not CONTROL.unpack_from(ctrl.buf)[0]:time.sleep(.001)
        seq,deadline,op,fault=CONTROL.unpack_from(ctrl.buf);started=time.perf_counter()
        if recover:
            epoch,size,base_hash,payload_hash,state_hash=HEADER.unpack_from(blob.buf)
            if op!=2 or epoch!=seq or size>MAX:raise ValueError('recovery epoch/size')
            payload=bytes(blob.buf[HEADER.size:HEADER.size+size])
            if hashlib.sha256(payload).digest()!=payload_hash:raise ValueError('checkpoint corruption')
            checkpoint=pickle.loads(payload)
            if checkpoint['epoch']!=epoch or checkpoint['base_digest']!=base_hash.hex():raise ValueError('checkpoint lineage')
            b=restore(template,dataset,checkpoint['state']);actual=old.pf.digest(state(b))
            if actual!=state_hash.hex():raise ValueError('restored private identity')
            # Duplicate ticket recognition: returning the current committed
            # identity does not run physical advance or any admission method.
            end=capture(sources());record=dict(case=case,epoch=epoch,restored=state(b),digest=actual,advance_calls=0,duplicate_digest=old.pf.digest(state(b)),started=started,finished=time.perf_counter(),payload_sha256=payload_hash.hex(),loaded_closure_qualified=begin['files']==end['files'] and begin['module_files']==end['module_files'])
            save('RECOVERY_SOURCE_END_'+str(case)+'.json',end);save('RECOVERY_'+str(case)+'.json',encode(record))
            if time.perf_counter()>=deadline or not record['loaded_closure_qualified']:raise ValueError('recovery closure/deadline')
            reply.buf[:32]=bytes.fromhex(actual)
        else:
            if seq!=1 or op!=1:raise ValueError('fixed new epoch')
            before=old.pf.digest(state(b));b.clock=prev.next_physical_clock(b);b.env.event_id+=1
            b.env._advance_fleet(0,b.clock,b.fleet,b.traces,b.served)
            checkpoint=dict(epoch=seq,base_digest=before,state=copy.deepcopy(state(b)))
            digest=old.pf.digest(checkpoint['state']);payload=pickle.dumps(checkpoint,protocol=5)
            if len(payload)>MAX:raise ValueError('bounded full private checkpoint')
            end=capture(sources());record=dict(case=case,epoch=seq,before=before,after=digest,checkpoint=checkpoint,advance_calls=1,started=started,physical_finished=time.perf_counter(),payload_size=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),loaded_closure_qualified=begin['files']==end['files'] and begin['module_files']==end['module_files'])
            save('SOURCE_END_'+str(case)+'.json',end);save('PHYSICAL_ATTEMPT_'+str(case)+'.json',encode(record))
            struct.pack_into('<I',ctrl.buf,40,1)
            if fault==1:time.sleep(12.)
            if time.perf_counter()>=deadline or not record['loaded_closure_qualified'] or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('publication fenced')
            blob.buf[HEADER.size:HEADER.size+len(payload)]=payload
            HEADER.pack_into(blob.buf,0,seq,len(payload),bytes.fromhex(before),hashlib.sha256(payload).digest(),bytes.fromhex(digest))
            if not K.SetEvent(latch):raise ctypes.WinError(ctypes.get_last_error())
            struct.pack_into('<I',ctrl.buf,40,2)
            if fault==2:time.sleep(12.)
            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('ACK fenced')
            reply.buf[:32]=bytes.fromhex(digest)
    except Exception:
        save(('RECOVERY_' if recover else '')+'WORKER_FAILURE_'+str(case)+'.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        for m in (ctrl,blob,reply):m.close()
        K.CloseHandle(latch)
def launch(ctrl,blob,reply,name,case,recover=False):
    err=(OUT/(('recover_' if recover else 'worker_')+str(case)+'.stderr.txt')).open('x',encoding='utf-8')
    args=[sys.executable,'-B',str(Path(__file__)),'--worker',ctrl.name,blob.name,reply.name,name,str(case)]+(['--recover'] if recover else [])
    p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    ready=json.loads(p.stdout.readline());assert ready['ready'];return p,err,ready
def execute(case,fault,initial_digest):
    ctrl=SharedMemory(create=True,size=64);blob=SharedMemory(create=True,size=HEADER.size+MAX);reply=SharedMemory(create=True,size=32)
    ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(32);name='Local\\MaskCO_checkpoint_'+ctrl.name;latch=event(name,True);p=None;err=None
    try:
        p,err,ready=launch(ctrl,blob,reply,name,case);assert ready['baseline']==initial_digest
        started=time.perf_counter();cutoff=started+8.;deadline=started+10.;CONTROL.pack_into(ctrl.buf,0,1,cutoff,1,fault)
        chosen=initial_digest;ticket=None;chosen_at=None
        while time.perf_counter()<cutoff:
            if ticket is None and signaled(latch):
                epoch,size,base,ph,sh=HEADER.unpack_from(blob.buf)
                if epoch!=1 or size>MAX or base.hex()!=initial_digest:raise ValueError('parent fixed lineage/epoch')
                # This is the only authoritative logical commit. A worker's
                # physical copy and proposal marker alone do not commit it.
                ticket=dict(epoch=epoch,size=size,base=base.hex(),payload_sha256=ph.hex(),digest=sh.hex());chosen=sh.hex();chosen_at=time.perf_counter()
            if bytes(reply.buf)!=bytes(32) or p.poll() is not None:break
            time.sleep(.001)
        struct.pack_into('<I',ctrl.buf,44,1);stage=struct.unpack_from('<I',ctrl.buf,40)[0];ack=bytes(reply.buf).hex()
        if p.poll() is None:p.kill()
        p.wait(timeout=max(.001,deadline-time.perf_counter()));finished=time.perf_counter()
        record=dict(case=case,fault=fault,pid=p.pid,started=started,cutoff=cutoff,finished=finished,wall_s=finished-started,stage=stage,ticket=ticket,chosen=chosen,chosen_at=chosen_at,ack=ack,ack_present=ack!=bytes(32).hex(),old_checkpoint_retained=True,reaped=p.poll() is not None)
        save('HOST_'+str(case)+'.json',record)
        p.stdout.close();err.close();p=None;err=None
        if ticket is not None:
            ctrl.buf[:]=bytes(64);reply.buf[:]=bytes(32);p,err,_=launch(ctrl,blob,reply,name,case,True)
            # Runtime/physical dataset are warm outside the recovery request;
            # checkpoint decode, actual private restore and identity in it.
            s=time.perf_counter();limit=s+10.;CONTROL.pack_into(ctrl.buf,0,1,limit-2.,2,0)
            p.wait(timeout=max(.001,limit-time.perf_counter()));f=time.perf_counter()
            save('RECOVERY_HOST_'+str(case)+'.json',dict(pid=p.pid,started=s,finished=f,wall_s=f-s,exit_code=p.returncode,digest=bytes(reply.buf).hex(),reaped=True))
        return record
    finally:
        if p:
            if p.poll() is None:p.kill();p.wait(timeout=5)
            p.stdout.close()
        if err:err.close()
        K.CloseHandle(latch)
        for m in (ctrl,blob,reply):m.close();m.unlink()
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),fixed_source=str(SOURCE),scope='single next natural physical boundary after callback17/event26, private simulation copy only; three cases normal, post-advance pre-publication delay, post-publication pre-ACK delay; checkpoint restore recognizes duplicate ticket without any physical reexecution',cap_s=10.,execution_s=8.,reserve_s=2.,fault_delay_s=12.,MAX=MAX,cpu=[7],numpy='1.26.4',initialization_outside_clock=True,recovery_warm_runtime_outside_clock=True,full_live_request_qualified=False,external_irreversible_effects_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False,new_scientific_data=False))
    try:
        _,decode=old.codecs();initial=decode(read(SOURCE))['final'];baseline=old.pf.digest(initial);rows=[execute(i,i,baseline) for i in range(3)];checks=[]
        def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
        for r in rows:
            n=str(r['case']);attempt=decode(read(OUT/('PHYSICAL_ATTEMPT_'+n+'.json')))
            ck(n+':physical_once',attempt['advance_calls']==1 and attempt['before']==baseline and attempt['checkpoint']['state']['event_id']==initial['event_id']+1)
            ck(n+':loaded_closure',attempt['loaded_closure_qualified']);ck(n+':cancel_reap_total',r['reaped'] and r['wall_s']<10.)
            ck(n+':old_checkpoint_identity',old.pf.digest(decode(read(SOURCE))['final'])==baseline)
            if r['fault']==1:ck(n+':completed_private_step_discarded_without_publication',r['stage']==1 and r['ticket'] is None and r['chosen']==baseline and not r['ack_present'])
            else:
                rec=decode(read(OUT/('RECOVERY_'+n+'.json')));host=read(OUT/('RECOVERY_HOST_'+n+'.json'))
                ck(n+':published_lineage',r['ticket']['base']==baseline and r['chosen']==attempt['after'] and r['chosen_at']<r['cutoff'])
                ck(n+':exact_restored_private_identity',rec['digest']==r['chosen']==host['digest'] and old.pf.digest(rec['restored'])==attempt['after'])
                ck(n+':duplicate_ticket_no_physical_reexecution',rec['advance_calls']==0 and rec['duplicate_digest']==rec['digest'])
                ck(n+':recovery_closure_and_budget',rec['loaded_closure_qualified'] and host['exit_code']==0 and host['wall_s']<10.)
                ck(n+':expected_ack_state',r['ack_present']==(r['fault']==0))
        ck('registration_input_source_bytes_exact',all(sha(p)==h for p,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        fail=[c['name'] for c in checks if not c['pass_']];save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,records=rows,checkpoint_ack_component_qualified=not fail,full_live_request_qualified=False,external_irreversible_effects_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False));save('RUN_EXIT.json',dict(exit_code=int(bool(fail))))
        print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,walls=[r['wall_s'] for r in rows])));return int(bool(fail))
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc()));save('RUN_EXIT.json',dict(exit_code=1));raise
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
