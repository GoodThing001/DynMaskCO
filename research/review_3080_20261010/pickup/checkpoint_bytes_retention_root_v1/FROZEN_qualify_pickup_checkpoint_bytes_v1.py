"""New byte-retention component; no physical advance/admission/solver replay."""
import ctypes, hashlib, json, os, pickle, struct, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
from pickup_loaded_closure_v1 import capture, sha

OUT=old.P/'checkpoint_bytes_retention_root_v1'
INPUT=old.P/'checkpoint_ack_root_v1/PHYSICAL_ATTEMPT_0.json'
TOOLS=old.ROOT/'tools'
HEADER=struct.Struct('<QQ32s32s32s'); MAX=2*1024*1024

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f: f.write(s)
def warm():
    psutil.Process().cpu_affinity([7]); assert np.__version__=='1.26.4'
    # Explicit pre-load, not an explanation of the old audit's first load.
    dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'))
    list(psutil.process_iter(['name','cmdline'])); capture()
    return dll
def sources():
    paths=[INPUT,Path(__file__),TOOLS/'audit_pickup_checkpoint_bytes_v1.py',TOOLS/'pickup_loaded_closure_v1.py',Path(sys.executable)]
    return {str(p):sha(p) for p in paths}
def worker():
    dll=warm(); mem=SharedMemory(name=sys.argv[2]); _,decode=old.codecs()
    checkpoint=decode(read(INPUT))['checkpoint']
    # This is a new serialization from a sealed value tree, never recovered v1 bytes.
    old.pf.digest(checkpoint['state']); capture(); begin=capture(sources())
    save('SOURCE_START.json',begin); print('READY',flush=True)
    try:
        assert sys.stdin.readline().strip()=='GO'; started=time.perf_counter()
        payload=pickle.dumps(checkpoint,protocol=5); assert len(payload)<=MAX
        with (OUT/'ACTUAL_WRITER_PAYLOAD.bin').open('xb') as f: f.write(payload)
        ph=hashlib.sha256(payload).digest(); sh=bytes.fromhex(old.pf.digest(checkpoint['state']))
        mem.buf[HEADER.size:HEADER.size+len(payload)]=payload
        HEADER.pack_into(mem.buf,0,checkpoint['epoch'],len(payload),bytes.fromhex(checkpoint['base_digest']),ph,sh)
        end=capture(sources()); save('SOURCE_END.json',end)
        save('WRITER.json',dict(pid=os.getpid(),started=started,finished=time.perf_counter(),size=len(payload),sha256=ph.hex(),state_digest=sh.hex(),epoch=checkpoint['epoch'],base_digest=checkpoint['base_digest'],actual_writer_archive_before_publication=True,loaded_closure_qualified=begin['files']==end['files'] and begin['module_files']==end['module_files'],physical_calls=0,admission_calls=0))
        print('PUBLISHED',flush=True)
    finally: mem.close()
def main():
    dll=warm(); OUT.mkdir(exist_ok=False)
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),scope='One new serialization of sealed event27 state; retain actual writer and shared bytes. No physical/admission/SAA/solver replay. Independent byte/lineage/private value audit and byte/epoch corruption rejection only. Original absent payloads remain absent and original 4 FAIL unchanged.',max_bytes=MAX,cpu=[7],numpy='1.26.4',cap_s=10.,execution_s=8.,reserve_s=2.,initialization_outside_clock=True,new_scientific_data=False,full_live_request_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False))
    mem=SharedMemory(create=True,size=HEADER.size+MAX); mem.buf[:HEADER.size]=bytes(HEADER.size); p=None
    try:
        with (OUT/'WORKER.stderr.txt').open('x',encoding='utf-8') as err:
            p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',mem.name],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            assert p.stdout.readline().strip()=='READY'
            started=time.perf_counter(); p.stdin.write('GO\n');p.stdin.flush()
            # Bounded exit before host reads a fixed, completed writer's buffer.
            p.wait(timeout=8.); assert p.returncode==0
            raw=bytes(mem.buf); epoch,size,base,ph,sh=HEADER.unpack_from(raw); assert 0<size<=MAX
            payload=raw[HEADER.size:HEADER.size+size]
            with (OUT/'ACTUAL_SHARED_ENVELOPE.bin').open('xb') as f: f.write(raw[:HEADER.size+size])
            with (OUT/'ACTUAL_SHARED_PAYLOAD.bin').open('xb') as f: f.write(payload)
            finished=time.perf_counter(); p.stdin.close();p.stdout.close()
        w=read(OUT/'WRITER.json'); checks=[]
        def ck(n,v): checks.append(dict(name=n,pass_=bool(v)))
        ck('actual_writer_and_shared_bytes_exact',payload==(OUT/'ACTUAL_WRITER_PAYLOAD.bin').read_bytes())
        ck('actual_byte_SHA_and_size',hashlib.sha256(payload).digest()==ph and len(payload)==w['size'] and ph.hex()==w['sha256'])
        ck('exact_header_lineage',epoch==w['epoch'] and base.hex()==w['base_digest'] and sh.hex()==w['state_digest'])
        ck('archive_and_reap_in_component_window',finished-started<10. and p.poll() is not None)
        ck('writer_workload_loaded_closure',w['loaded_closure_qualified'])
        ck('no_physical_or_admission_calls',w['physical_calls']==w['admission_calls']==0)
        ck('registered_sources_unchanged',all(sha(pth)==h for pth,h in read(OUT/'REGISTRATION.json')['sources'].items()))
        failed=[r['name'] for r in checks if not r['pass_']]
        save('RESULT.json',dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,pid=p.pid,reaped=True,started=started,finished=finished,wall_s=finished-started,bytes_retention_runtime_qualified=not failed,new_payload_not_original_v1=True,full_live_request_qualified=False,physical_ack_loss_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False))
        save('RUN_EXIT.json',dict(exit_code=int(bool(failed))));print(json.dumps(dict(checks=len(checks),failed=len(failed),wall_s=finished-started)));return int(bool(failed))
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc()));save('RUN_EXIT.json',dict(exit_code=1));raise
    finally:
        if p and p.poll() is None: p.kill();p.wait(timeout=2.)
        mem.close();mem.unlink()
if __name__=='__main__':
    if '--worker' in sys.argv: worker()
    else: raise SystemExit(main())
