"""Prepared single-request resident worker. It refuses an unsealed execution plan."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,mmap,struct,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
SIZE=2*2**20;HEADER=64

def main():
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    if not (reg.get('execution_authorized') is True and reg.get('independent_timed_request_checker_sealed') is True):
        raise RuntimeError('Prepared source is not execution-qualified')
    from action_head_runtime_v1 import sha
    from timed_evidence_io_v1 import save,save_bytes
    for n,h in reg['package_files'].items():
        if sha(BASE/n)!=h:raise ValueError('Registered package bytes: '+n)
    spec=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    ctrl=mmap.mmap(-1,128,tagname=spec['control']);inp=mmap.mmap(-1,SIZE,tagname=spec['input']);out=mmap.mmap(-1,SIZE+HEADER,tagname=spec['output'])
    from delivery_phase_actual_engine_v1 import InitialEngine
    from bounded_phase_trace_v1 import PhaseTrace,MAX_RECORDS
    telemetry=mmap.mmap(-1,64+64*MAX_RECORDS,tagname=spec['trace'])
    from startup_native_identity_v1 import prime_native_identity
    native_dependency_handle=prime_native_identity()
    engine=InitialEngine(BASE,reg,spec['arm'])
    root=BASE/spec['output_directory'];root.mkdir(exist_ok=False)
    save(root,'SOURCE_START.json',engine.start)
    save(root,'INITIAL_PARAMETERS.json',dict(backbone=engine.parameter_identity,head=engine.head_identity,source_identity=engine.source_identity))
    struct.pack_into('<I',ctrl,12,1)
    began=time.perf_counter()
    while struct.unpack_from('<Q',ctrl,0)[0]!=1:
        if time.perf_counter()-began>60:raise TimeoutError('No request within bounded ready wait')
        time.sleep(.001)
    seq=struct.unpack_from('<Q',ctrl,0)[0];size=struct.unpack_from('<I',ctrl,8)[0];deadline=struct.unpack_from('<d',ctrl,16)[0]
    if not 0<size<=SIZE:raise ValueError('Bounded public input')
    trace=PhaseTrace(telemetry,seq,MAX_RECORDS)
    def cancelled():return struct.unpack_from('<I',ctrl,24)[0]!=0 or time.perf_counter()>=deadline
    def stage(code,name):
        if cancelled():raise TimeoutError('Deadline before actual stage: '+name)
        evidence={str(p.relative_to(root)):sha(p) for p in sorted((root/'request').glob('*')) if p.is_file()}
        save(root,'STAGE_%d.json'%code,dict(stage=code,name=name,clock=time.perf_counter(),deadline=deadline,evidence_files=evidence))
        struct.pack_into('<I',ctrl,28,code)
        if spec.get('fault_stage')==code:
            # Capture actual loaded files BEFORE entering the artificial delay.
            # This is timed fault instrumentation; no extra time is granted.
            snapshot=trace.measure(14,code,engine.closure,engine.extras,engine.trees)
            trace.measure(16,code,save,root,'FAULT_SOURCE_AT_ENTRY.json',dict(**snapshot,**engine.delta(engine.start,snapshot)))
            save(root,'FAULT_DELAY_ENTERED.json',dict(stage=code,clock=time.perf_counter(),deadline=deadline))
            time.sleep(12.)
        if cancelled():raise TimeoutError('Cancelled at actual stage: '+name)
    try:
        report=engine.request(bytes(inp[:size]),root/'request',stage,cancelled,trace)
        def serialize_publish():
            raw=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
            if len(raw)>SIZE:raise ValueError('Bounded response')
            save_bytes(root/'SERIALIZED_RESPONSE.json',raw)
            stage(7,'serialized_response_before_publication')
            if cancelled():raise TimeoutError('No late publication')
            out[HEADER:HEADER+len(raw)]=raw
            struct.pack_into('<QI4x32sd',out,0,seq,len(raw),hashlib.sha256(raw).digest(),time.perf_counter())
            if cancelled():raise TimeoutError('No publication after response copy')
            struct.pack_into('<Q',ctrl,32,seq)
            save(root,'PUBLICATION.json',dict(sequence=seq,size=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                clock=time.perf_counter(),deadline=deadline))
            # Publication precedes this marker. A fault here may be selected by the
            # caller; it is not a rollback or an unselected private branch.
            stage(8,'published_response_before_worker_exit')
        trace.measure(17,0,serialize_publish)
        save(root,'NATURAL_FINISH.json',dict(sequence=seq,clock=time.perf_counter()))
    except Exception:
        save(root,'FAILURE.json',dict(traceback=traceback.format_exc(),stage=struct.unpack_from('<I',ctrl,28)[0]))
        raise
    finally:
        trace.close()
        trace.storage.release()
        telemetry.close();ctrl.close();inp.close();out.close()
if __name__=='__main__':main()
