"""One preregistered lifecycle stage and an independent saved-evidence audit.

The real prepared parent is exercised with an explicitly gated startup fixture.
No model, C0, old trace kernel/parser, solver or old request is executed.
"""
import ast
import hashlib
import importlib.util
import json
import mmap
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/review_3080_20261010/delivery/early_launch_lifecycle_root_v3'
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(OUT))
from pickup_loaded_closure_v1 import capture
from startup_native_identity_v1 import prime_native_identity
import action_head_runtime_v1
import timed_evidence_io_v1
import phase_failure_cleanup_v2
import delivery_phase_shared_parent_v3 as parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def source(reg):
    return capture([ROOT/name for name in reg['files']]+[OUT/'ENGINEERING_REGISTRATION.json'])


def delta(a,b):
    return dict(new_files=sorted(set(b['files'])-set(a['files'])),
        missing_files=sorted(set(a['files'])-set(b['files'])),
        changed_files=sorted(k for k in set(a['files'])&set(b['files']) if a['files'][k]!=b['files'][k]),
        modules_changed=a['module_files']!=b['module_files'])


def run_case(name,reg):
    base=OUT/'cases'/name;base.mkdir(parents=True,exist_ok=False)
    fixture=OUT/'early_launch_fixture_worker_v1.py'
    shutil.copy2(fixture,base/fixture.name)
    r=dict(early_launch_component_only=True,execution_authorized_for_early_launch_component=True,
        model_execution_authorized=False,execution_authorized=False,
        independent_timed_request_checker_sealed=False,
        total_request_seconds=10,child_seconds=8,reap_reserve_seconds=2,
        fixture_worker_sha256=sha(fixture),fixture_worker_path=str(fixture.resolve()))
    timed_evidence_io_v1.save(base,'EXECUTION_REGISTRATION.json',r)
    real_map=mmap.mmap;real_process=psutil.Process;real_popen=subprocess.Popen;real_save=timed_evidence_io_v1.save
    handles=[];children=[];map_calls=0;original=None
    def tracked_map(*a,**kw):
        nonlocal map_calls
        map_calls+=1
        if name=='mapping_failure' and map_calls==3:
            raise OSError('Registered third mapping failure')
        handle=real_map(*a,**kw);handles.append(handle);return handle
    def tracked_popen(*a,**kw):
        handle=real_popen(*a,**kw);children.append(handle);return handle
    def tracked_process(pid=None,*a,**kw):
        if pid is not None and name in ('identity_failure','receipt_archive_failure','identity_interrupt'):
            deadline=time.perf_counter()+4
            while not (base/'CHILD_READY.json').exists():
                if children and children[-1].poll() is not None:
                    raise RuntimeError('Fixture exited before registered identity fault')
                if time.perf_counter()>=deadline:
                    raise TimeoutError('Fixture ready limit before identity fault')
                time.sleep(.005)
            if name=='identity_interrupt':raise SystemExit('Registered startup identity interruption')
            raise PermissionError('Registered child identity query failure')
        return real_process(pid,*a,**kw) if pid is not None else real_process(*a,**kw)
    def tracked_save(b,n,v):
        if name=='receipt_archive_failure' and n.endswith('_CALLER_FAILURE_AFTER_CLEANUP.json'):
            real_save(b,'CLEANUP_RECEIPT_ATTEMPT.json',v)
            raise OSError('Registered receipt archive failure')
        return real_save(b,n,v)
    if name=='log_collision':
        (base/(name+'_fault0_stdout.txt')).write_bytes(b'original-collision-bytes')
    began=time.perf_counter()
    mmap.mmap=tracked_map;psutil.Process=tracked_process;subprocess.Popen=tracked_popen;timed_evidence_io_v1.save=tracked_save
    try:
        python=base/'nonexistent_python.exe' if name=='popen_failure' else Path(sys.executable)
        parent.request(base,python,name,b'{}',[])
    except BaseException as exc:
        original=dict(type=type(exc).__name__,message=str(exc),notes=getattr(exc,'__notes__',[]),traceback=traceback.format_exc())
    finally:
        mmap.mmap=real_map;psutil.Process=real_process;subprocess.Popen=real_popen;timed_evidence_io_v1.save=real_save
    finished=time.perf_counter()
    # Inspect actual returned handles before any emergency supervisor intervention.
    actual=dict(original_exception=original,
        acquired_mapping_count=len(handles),mapping_closed=[h.closed for h in handles],
        launched_child_count=len(children),child_returncodes=[c.poll() for c in children],
        child_pids=[c.pid for c in children],start_clock=began,end_clock=finished,
        elapsed=finished-began,model_executed=False)
    leaked=any(c.poll() is None for c in children)
    actual['child_leaked_after_parent']=leaked
    # Never allow a failed new implementation to leave a fixture behind.
    for child in children:
        if child.poll() is None:
            child.kill();child.wait(timeout=2)
    write(name+'_ACTUAL_HANDLE_OBSERVATION.json',actual)
    return actual


def producer(reg):
    if (OUT/'RUN_STARTED.json').exists():
        raise RuntimeError('No repeat of this registered lifecycle stage')
    write('RUN_STARTED.json',dict(pid=os.getpid(),clock=time.perf_counter(),model_executed=False))
    psutil.Process().cpu_affinity([7]);native=prime_native_identity()
    source(reg);start=source(reg);write('SOURCE_START.json',start)
    results={name:run_case(name,reg) for name in reg['cases']}
    end=source(reg);write('SOURCE_END.json',end)
    changes=delta(start,end)
    failures=[]
    for name,item in results.items():
        if item['original_exception'] is None or not all(item['mapping_closed']) or item['child_leaked_after_parent']:
            failures.append(name)
    if any(changes.values()):failures.append('Actual producer source closure changed')
    write('RESULT.json',dict(cases=results,source_delta=changes,failures=failures,
        model_executed=False,full_request_budget_qualified=False,scientific_family3_started=False,stable_online_gain=False))
    return 1 if failures else 0


def audit(reg):
    if (OUT/'INDEPENDENT_LIFECYCLE_AUDIT.json').exists():
        raise RuntimeError('No repeat audit')
    psutil.Process().cpu_affinity([7]);native=prime_native_identity()
    source(reg);start=source(reg);write('AUDIT_SOURCE_START.json',start)
    checks=[]
    def check(name,v):checks.append(dict(name=name,passed=bool(v)))
    for name,h in reg['files'].items():check('Registered file:'+name,sha(ROOT/name)==h)
    inputs=json.loads((OUT/'AUDIT_INPUT_REGISTRATION.json').read_text(encoding='utf-8'))
    for name,h in inputs['files'].items():check('Saved input:'+name,sha(OUT/name)==h)
    result=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'))
    expected=dict(mapping_failure=('OSError',2,0),log_collision=('FileExistsError',4,0),
        popen_failure=('FileNotFoundError',4,0),identity_failure=('PermissionError',4,1),
        init_exit=('RuntimeError',4,1),startup_endpoint=('RuntimeError',4,1),
        receipt_archive_failure=('PermissionError',4,1),identity_interrupt=('SystemExit',4,1))
    for name in reg['cases']:
        typ,nmaps,nchildren=expected[name]
        base=OUT/'cases'/name
        actual=json.loads((OUT/(name+'_ACTUAL_HANDLE_OBSERVATION.json')).read_text(encoding='utf-8'))
        receipt_path=base/('CLEANUP_RECEIPT_ATTEMPT.json' if name=='receipt_archive_failure' else name+'_fault0_CALLER_FAILURE_AFTER_CLEANUP.json')
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        check(name+': original exception',actual['original_exception']['type']==typ)
        check(name+': exact map count',actual['acquired_mapping_count']==nmaps)
        check(name+': all real maps closed',actual['mapping_closed']==[True]*nmaps)
        check(name+': exact launch count',actual['launched_child_count']==nchildren)
        check(name+': no live child',not actual['child_leaked_after_parent'] and all(c is not None for c in actual['child_returncodes']))
        check(name+': no cleanup error',receipt['cleanup_errors']==[])
        check(name+': startup outside request',receipt['start_clock'] is None and receipt['initialization_outside_request'] is True)
        check(name+': original plan selection unchanged',receipt['selected'] is False and receipt['selected_output_revoked'] is False)
        check(name+': no scientific qualification',receipt['qualification'] is False and receipt['full_request_budget_qualified'] is False)
        check(name+': child absence truthful',receipt['child_launched']==bool(nchildren) and receipt['worker_reaped']==bool(nchildren))
        if nchildren:
            check(name+': exact returned code',receipt['actual_worker_returncode']==actual['child_returncodes'][0])
            check(name+': cleanup reap clock',receipt['reap_clock'] is not None and receipt['reap_clock']<=receipt['end_clock'])
            raw=base/(name+'_fault0_WORKER_TRACE.bin')
            check(name+': original raw hash',sha(raw)==receipt['worker_trace_sha256'])
            ready=json.loads((base/'CHILD_READY.json').read_text(encoding='utf-8'))
            check(name+': no input request',ready['request_sequence']==0)
            check(name+': actual child PID',ready['identity']['pid']==actual['child_pids'][0])
            check(name+': original startup marker',sha(raw)==ready['marker_sha256'])
            child_start=json.loads((base/'CHILD_SOURCE_START.json').read_text(encoding='utf-8'))
            check(name+': native actual start',any(p.lower().endswith('\\shcore.dll') for p in child_start['files']))
            if name=='init_exit':
                child_end=json.loads((base/'CHILD_SOURCE_END.json').read_text(encoding='utf-8'))
                check(name+': child natural closure',not any(delta(child_start,child_end).values()) and actual['child_returncodes']==[7])
        else:
            check(name+': no invented worker bytes',receipt['worker_trace_sha256'] is None and not list(base.glob('*WORKER_TRACE.bin')))
        if name=='log_collision':check('Original log collision retained',(base/(name+'_fault0_stdout.txt')).read_bytes()==b'original-collision-bytes')
        if name=='receipt_archive_failure':check('Receipt failure does not replace original',any('Cleanup receipt failed:' in n for n in actual['original_exception']['notes']))
    a=json.loads((OUT/'SOURCE_START.json').read_text(encoding='utf-8'));b=json.loads((OUT/'SOURCE_END.json').read_text(encoding='utf-8'))
    check('Actual producer full component closure',not any(delta(a,b).values()))
    end=source(reg);write('AUDIT_SOURCE_END.json',end)
    check('Actual audit component closure',not any(delta(start,end).values()))
    for p,h in a['files'].items():check('Producer input bytes retained:'+p,Path(p).is_file() and sha(p)==h)
    failures=[c['name'] for c in checks if not c['passed']]
    write('INDEPENDENT_LIFECYCLE_AUDIT.json',dict(checks=len(checks),failures=failures,items=checks,
        production_rerun=False,model_executed=False,full_request_budget_qualified=False,
        full_scientific_loaded_closure_qualified=False,stable_online_gain=False))
    return 1 if failures else 0


def main():
    stage=sys.argv[1]
    if stage not in ('producer','audit'):raise ValueError('Fixed new lifecycle stage')
    reg=json.loads((OUT/'ENGINEERING_REGISTRATION.json').read_text(encoding='utf-8'))
    assert reg['model_execution_authorized'] is False and reg['execution_authorized_for_early_launch_component'] is True
    for name,h in reg['files'].items():assert sha(ROOT/name)==h,name
    code=producer(reg) if stage=='producer' else audit(reg)
    write(stage+'_RUN_EXIT.json',dict(actual_returncode=code,natural_exit=True))
    print(stage+' lifecycle actual exit '+str(code))
    raise SystemExit(code)


if __name__=='__main__':main()
