"""One registered trace-only run. Never imports the model engine or its parent."""
import hashlib,json,mmap,os,subprocess,sys,time,traceback,uuid
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'research/review_3080_20261010/delivery/phase_cleanup_boundary_engineering_root_v1'
sys.path.insert(0,str(O))
from bounded_phase_trace_v1 import PhaseTrace,TraceCapacityExceeded
from phase_failure_cleanup_v1 import cleanup
SIZE=64+4096*64

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def closure():
    files={str(Path(sys.executable).resolve()):sha(sys.executable)};modules={}
    for name,module in sorted(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file and Path(file).is_file():
            file=str(Path(file).resolve());modules[name]=file;files[file]=sha(file)
        cached=getattr(module,'__cached__',None)
        if cached and Path(cached).is_file():files[str(Path(cached).resolve())]=sha(cached)
    for item in psutil.Process().memory_maps():
        p=Path(item.path)
        if p.suffix.lower() in ('.dll','.pyd','.exe') and p.is_file():files[str(p.resolve())]=sha(p)
    reg=json.loads((O/'TOOLS_REGISTRATION_R2.json').read_text(encoding='utf-8'))
    for name in reg['files']:files[str((ROOT/name).resolve())]=sha(ROOT/name)
    return dict(files=files,modules=modules,python=sys.version,executable=sys.executable)

def child(tag):
    psutil.Process().cpu_affinity([7])
    buf=mmap.mmap(-1,SIZE,tagname=tag)
    closure();start=closure();write('CHILD_SOURCE_START.json',start)
    p=psutil.Process()
    write('CHILD_OWNER.json',dict(pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline(),username=p.username()))
    trace=PhaseTrace(buf,4,4096)
    def stalled():
        # This is a real open trace operation, not an invented fault label.
        write('CHILD_OPEN_ENTRY.json',dict(clock=time.perf_counter(),source=closure()))
        write('CHILD_READY.json',dict(pid=os.getpid(),clock=time.perf_counter()))
        time.sleep(30)
    trace.measure(7,0,stalled)
    raise AssertionError('Registered caller must terminate this writer')

def producer():
    reg=json.loads((O/'TOOLS_REGISTRATION_R2.json').read_text(encoding='utf-8'))
    if (O/'RUN_STARTED.json').exists():raise RuntimeError('No trace component replay')
    assert reg['execution_authorized_for_trace_only'] is True
    for n,h in reg['files'].items():assert sha(ROOT/n)==h
    psutil.Process().cpu_affinity([7]);closure()
    source=closure();write('SOURCE_START.json',source)
    write('RUN_STARTED.json',dict(clock=time.perf_counter(),pid=os.getpid(),model_executed=False))
    cases=[]
    def finish(name,seq,buf,start,end,meta):
        file=name+'.bin'
        with (O/file).open('xb') as f:f.write(bytes(buf))
        cases.append(dict(name=name,sequence=seq,file=file,sha256=sha(O/file),start_clock=start,end_clock=end,**meta))
    buf=bytearray(64+8*64);start=time.perf_counter();t=PhaseTrace(buf,1,8)
    value=t.measure(1,0,lambda:t.measure(2,0,lambda:7));assert value==7
    t.close();end=time.perf_counter();finish('nested_success',1,buf,start,end,dict(expected_phases=[1,2],expected_statuses=[1,1]))
    buf=bytearray(64+8*64);start=time.perf_counter();t=PhaseTrace(buf,2,8)
    def failure():raise RuntimeError('Registered trace operation exception')
    try:t.measure(3,0,failure)
    except RuntimeError:pass
    else:raise AssertionError('Exception propagation')
    assert t.measure(4,0,lambda:11)==11
    t.close();end=time.perf_counter();finish('recorded_exception',2,buf,start,end,dict(expected_phases=[3,4],expected_statuses=[2,1]))
    buf=bytearray(128);start=time.perf_counter();t=PhaseTrace(buf,3,1);t.measure(5,0,lambda:13)
    try:t.measure(6,0,lambda:17)
    except TraceCapacityExceeded:pass
    else:raise AssertionError('No silent overflow')
    t.close();end=time.perf_counter();finish('capacity_overflow',3,buf,start,end,dict(expected_phases=[5],expected_statuses=[1]))
    tag='maskco_phase_'+uuid.uuid4().hex;shared=mmap.mmap(-1,SIZE,tagname=tag)
    log=(O/'child.stderr.txt').open('xb');out=(O/'child.stdout.txt').open('xb')
    started=time.perf_counter();p=subprocess.Popen([sys.executable,'-B',__file__,'child',tag],stdout=out,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    caller_trace=None
    try:
        owned=psutil.Process(p.pid);identity=dict(pid=p.pid,create_time=owned.create_time(),cmdline=owned.cmdline(),username=owned.username())
        while not (O/'CHILD_READY.json').exists():
            if p.poll() is not None:raise RuntimeError('Trace writer exited before entry')
            if time.perf_counter()-started>5:raise TimeoutError('Bounded trace ready wait')
            time.sleep(.005)
        check=psutil.Process(p.pid)
        assert check.create_time()==identity['create_time'] and check.cmdline()==identity['cmdline'] and check.username()==identity['username']
        caller_buf=bytearray(SIZE);window=time.perf_counter();caller_trace=PhaseTrace(caller_buf,5,4096)
        try:caller_trace.measure(18,0,failure)
        except RuntimeError:pass
        def retain(kind,raw):
            with (O/('cleanup_'+kind+'.bin')).open('xb') as f:f.write(raw)
        receipt=cleanup(p,shared,caller_trace,(shared,log,out),retain,window,window+2.)
        assert receipt['worker_reaped'] and not receipt['cleanup_errors'] and receipt['actual_worker_returncode'] is not None
        assert receipt['within_declared_window']
        write('ACTUAL_CLEANUP_RECEIPT.json',dict(**receipt,child_identity=identity,writer_natural_exit=False,
            writer_end_source_available=False,selected_output=False))
        end=receipt['end_clock']
        cases.append(dict(name='actual_open_writer_kill',sequence=4,file='cleanup_worker.bin',sha256=sha(O/'cleanup_worker.bin'),start_clock=started,end_clock=end,expected_phases=[7],expected_statuses=[0]))
        cases.append(dict(name='parent_exception_cleanup',sequence=5,file='cleanup_caller.bin',sha256=sha(O/'cleanup_caller.bin'),start_clock=window,end_clock=end,expected_phases=[18,19],expected_statuses=[2,1]))
    finally:
        if p.poll() is None:p.kill();p.wait(timeout=2)
        for h in (shared,log,out):
            if not h.closed:h.close()
    endsource=closure();write('SOURCE_END.json',endsource)
    unchanged=source==endsource
    write('RESULT.json',dict(cases=cases,source_unchanged=unchanged,model_executed=False,full_request_budget_qualified=False,
        parent_model_workload_executed=False,old_QA_repeated=False,scientific_family3_started=False))
    write('RUN_EXIT.json',dict(returncode=0 if unchanged else 1,actual_self_pid=os.getpid(),natural_exit=True))
    return 0 if unchanged else 1

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child':child(sys.argv[2])
    else:
        try:raise SystemExit(producer())
        except Exception:
            write('FAILURE.json',dict(traceback=traceback.format_exc(),qualification=False,no_automatic_retry=True));raise
