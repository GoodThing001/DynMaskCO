"""Single fixed archived-public experiment; 8s execute + 2s reap reserve."""
import os,sys,json,copy,pickle,hashlib,time,traceback,subprocess,threading,queue,ctypes
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];TOOLS=ROOT/'tools';sys.path.insert(0,str(TOOLS))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import qualify_pickup_joint_policy_transaction_root_v3 as old
import numpy as np,psutil
from controller import compute
import phase_trace
INPUT=old.P/'undecided_public_input_root_v2/RESULT.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(name,x):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def packet():
    _,decode=old.codecs()
    for position,row in enumerate(read(INPUT)['records']):
        p=decode(row['packet']);snap=p['snapshot'];count=0
        for v,s in enumerate(snap['vehicle_status']):
            if s in ('idle','ready','committed'):
                frozen=int(snap['committed_next'][v]) if s=='committed' else None
                count+=sum(o!=frozen for o in p['plan'][str(v)])
        if row.get('variant')==0 and count>=2:return p,dict(position=position,kind=row['kind'],index=row.get('index'),mutable_orders=count)
    raise ValueError('fixed domain not eligible')
def initialize():
    import pickup_undecided_public_v1,pickup_pending_transaction_v1,pickup_partial_known_v1,pickup_partial_validation_v1,pickup_partial_trace_v1
    import rolling_optimizer,energy_complete_saa,scenario_saa,strict_online_env,recourse_snapshot,coldchain_evaluator_a1,audit_io
    import observed_saa
    old.codecs();ctypes.WinDLL('SHCore.dll');psutil.Process().memory_maps();hashlib.sha256(b'').hexdigest()
def closure():
    paths={Path(sys.executable).resolve(),Path(__file__).resolve(),HERE/'controller.py',HERE/'audit.py',INPUT,old.HISTORY}
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and Path(p).is_file():
            paths.add(Path(p).resolve())
            try:
                c=Path(__import__('importlib').util.cache_from_source(str(p)))
                if c.is_file():paths.add(c.resolve())
            except (NotImplementedError,ValueError):pass
    for m in psutil.Process().memory_maps():
        p=Path(m.path)
        if p.is_file() and p.resolve()!= (HERE/'PHASE_TRACE.bin').resolve():paths.add(p.resolve())
    return {str(p):old.sha(p) for p in sorted(paths)}
def worker():
    psutil.Process().cpu_affinity([7]);initialize();p,chosen=packet();encode,_=old.codecs();old.pf.digest(p)
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    phase_trace.initialize(HERE/'PHASE_TRACE.bin')
    initial=closure();save('SOURCE_START.json',initial)
    print(json.dumps(dict(kind='ready',pid=os.getpid(),creation=psutil.Process().create_time(),chosen=chosen)),flush=True)
    cmd=json.loads(sys.stdin.readline());started=time.perf_counter();deadline=cmd['deadline']
    try:
        result=compute(p,history,deadline);finished=time.perf_counter()
        phase_trace.emit('result_write_b')
        save('COMPONENT_RESULT.json',encode(dict(packet=p,result=result,started=started,finished=finished,elapsed=finished-started,deadline=deadline)))
        phase_trace.emit('result_write_e')
        phase_trace.emit('source_begin')
        final=closure();save('SOURCE_END.json',final)
        phase_trace.emit('source_end')
        phase_trace.close()
        print(json.dumps(dict(kind='done',qualified=True,elapsed=finished-started,source_same=initial==final)),flush=True)
    except Exception:
        save('WORKER_FAILURE.json',dict(traceback=traceback.format_exc(),started=started,failed_at=time.perf_counter()))
        print(json.dumps(dict(kind='failed')),flush=True)
        raise
def main():
    psutil.Process().cpu_affinity([7]);initialize();p,chosen=packet();encode,_=old.codecs()
    if np.__version__!='1.26.4':raise ValueError('original pinned NumPy required')
    resource=dict(available_ram_gib=psutil.virtual_memory().available/2**30,cpu_affinity=psutil.Process().cpu_affinity(),python=sys.version,numpy=np.__version__)
    if resource['available_ram_gib']<.5:raise ValueError('nomodel component resource gate')
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources={str(q):old.sha(q) for q in (HERE/'controller.py',HERE/'run.py',HERE/'audit.py',HERE/'observed_saa.py',HERE/'phase_trace.py',HERE/'prepare.py',HERE/'supervise_audit.py',INPUT,old.HISTORY)},input_selection='first original record, variant0, with >=2 mutable noncommitted outstanding orders; metadata only, no cost selection',chosen=chosen,current=p['admission_input']['current_order'],diagnostic_goal='candidate existingincr_eval route-summary prefilters and delta-ranking; exact action/terminalC0/float.hex equivalence versus originalfalse is REQUIRED, not assumed; no utility selection or scientific gain',prior_registration_sha256='61d71415311a784bdab5787e30920e32955040baa3890ad7766f20a1e713a474',telemetry_size_bytes=32832,method='current-only independent accept/reject branches; all insertion slots; four best-improvement distance relocate/swap passes; original full C0 for each final candidate; best known utility candidate per branch; unconditional historical K10 common once; complete energy-accounting SAA reveal-order future route local search; strictly higher paired mean accepts, ties reject',candidate_incr_eval=True,exact_reference_incr_eval=False,partial_prefix_only_cannot_release=True,witness_per_file_cap_bytes=2097152,reference_audit_supervision_cap_s=120.,PyVRP_reference='Wouda,Lan,Kool2024 IJOC36(4)943-955 DOI10.1287/ijoc.2023.0055,route move cache and conservative filtering inspiration only; not implementation replication or C0 equivalence',K=10,seed=202610130901,resource=resource,worker_budget=8.,cleanup_reserve=2.,total_cap=10.,failure_policy='keep original certified incumbent; current order remains undecided for caller, no publish; fail closed',library_initialization_input_preparation_and_shared_history_outside_component=True,original_evaluator_byte_changes=False,scientific_data_generated=False,formal_strong_dynamic_qualified=False,full_request_budget_qualified=False,full_scientific_loaded_closure=False,stable_online_gain=False,literature={'title':'Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers','authors':['Russell W. Bent','Pascal Van Hentenryck'],'year':2004,'doi':'10.1287/opre.1040.0124','url':'https://pubsonline.informs.org/doi/10.1287/opre.1040.0124','relationship':'engineering adaptation inspired by future-scenario planning; not consensus MSA replication; our C0 utility differs from served-customer objective'}))
    save('INPUT.json',encode(p))
    with (HERE/'PHASE_TRACE.bin').open('xb') as f:f.write(bytes(32832))
    q=queue.Queue();proc=None;started=None;ready=None;done=None;reason=None
    try:
        proc=subprocess.Popen([sys.executable,'-B',str(HERE/'run.py'),'--worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(HERE/'worker.stderr.txt').open('x',encoding='utf-8'),text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW)
        def collect():
            for line in proc.stdout:
                try:q.put(json.loads(line))
                except Exception:q.put(dict(kind='unparsed',text=line))
        threading.Thread(target=collect,daemon=True).start();ready=q.get(timeout=30.)
        if ready.get('kind')!='ready':raise ValueError('worker setup failed')
        started=time.perf_counter();deadline=started+8.;proc.stdin.write(json.dumps(dict(deadline=deadline))+'\n');proc.stdin.flush()
        try:done=q.get(timeout=max(0,deadline-time.perf_counter()))
        except queue.Empty:reason='worker deadline elapsed'
        if done is None or done.get('kind')!='done':
            reason=reason or 'worker failure';proc.kill()
        proc.wait(timeout=2.)
    except Exception:
        reason=traceback.format_exc()
    finally:
        if proc is not None and proc.poll() is None:proc.kill();proc.wait(timeout=2.)
    finished=time.perf_counter();ok=done is not None and done.get('kind')=='done' and reason is None
    receipt=dict(qualified=ok,ready=ready,done=done,reason=reason,started=started,finished=finished,elapsed=None if started is None else finished-started,worker_returncode=None if proc is None else proc.returncode,worker_reaped=proc is not None and proc.poll() is not None,output_selected=False,actual_live_owner=False,source_gate_passed=ok and done['source_same'],formal_strong_dynamic_qualified=False,full_request_budget_qualified=False,stable_online_gain=False)
    save('RUN_RECEIPT.json',receipt);save('RUN_EXIT.json',dict(exit_code=0 if ok else 1));print(json.dumps(receipt));return 0 if ok else 1
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
