"""Independent actual raw trace arithmetic and malformed boundary audit."""
import hashlib,json,math,struct,sys,traceback
from fractions import Fraction
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'research/review_3080_20261010/delivery/phase_cleanup_boundary_engineering_root_v1'
sys.path.insert(0,str(O))
from independent_phase_trace_parser_v1 import parse

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def closure():
    files={str(Path(sys.executable).resolve()):sha(Path(sys.executable))};modules={}
    for name,module in sorted(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file and Path(file).is_file():
            file=str(Path(file).resolve());modules[name]=file;files[file]=sha(Path(file))
        cached=getattr(module,'__cached__',None)
        if cached and Path(cached).is_file():files[str(Path(cached).resolve())]=sha(Path(cached))
    for item in psutil.Process().memory_maps():
        p=Path(item.path)
        if p.suffix.lower() in ('.dll','.pyd','.exe') and p.is_file():files[str(p.resolve())]=sha(p)
    reg=json.loads((O/'TOOLS_REGISTRATION_R2.json').read_text(encoding='utf-8'))
    for name in reg['files']:files[str((ROOT/name).resolve())]=sha(ROOT/name)
    return dict(files=files,modules=modules,python=sys.version,executable=sys.executable)
def main():
    if (O/'INDEPENDENT_TRACE_AUDIT.json').exists():raise RuntimeError('No duplicate trace audit')
    reg=json.loads((O/'TOOLS_REGISTRATION_R2.json').read_text(encoding='utf-8'))
    closure();source=closure()
    with (O/'AUDIT_SOURCE_START.json').open('x',encoding='utf-8') as f:json.dump(source,f,indent=2)
    checks=[]
    def check(name,value):checks.append(dict(name=name,pass_condition=bool(value)))
    for n,h in reg['files'].items():check('file:'+n,sha(ROOT/n)==h)
    input_reg=json.loads((O/'AUDIT_INPUT_REGISTRATION.json').read_text(encoding='utf-8'))
    for n,h in input_reg['files'].items():check('actual input:'+n,sha(O/n)==h)
    result=json.loads((O/'RESULT.json').read_text(encoding='utf-8'))
    for case in result['cases']:
        raw=(O/case['file']).read_bytes();check(case['name']+':raw identity',hashlib.sha256(raw).hexdigest()==case['sha256'])
        parsed=parse(raw,case['sequence'],case['start_clock'],case['end_clock'],True)
        count=int.from_bytes(raw[16:20],'little');reference=[]
        intervals=[]
        for i in range(count):
            offset=64+64*i
            phase=int.from_bytes(raw[offset+8:offset+12],'little');status=int.from_bytes(raw[offset+40:offset+44],'little',signed=True)
            began,start,end=struct.unpack('<3d',raw[offset+16:offset+40])
            check(case['name']+':exact clocks'+str(i),parsed['records'][i]['operation_start_hex']==start.hex() and parsed['records'][i]['operation_end_hex']==end.hex())
            duration=Fraction(end)-Fraction(start)
            check(case['name']+':exact rational'+str(i),parsed['records'][i]['exact_operation_seconds']==([duration.numerator,duration.denominator] if status else None))
            reference.append((phase,status))
            if status:intervals.append((Fraction(start),Fraction(end)))
        check(case['name']+':actual phases',[x[0] for x in reference]==case['expected_phases'])
        check(case['name']+':actual statuses',[x[1] for x in reference]==case['expected_statuses'])
        # Independent elementary endpoints partition: every covered interval is
        # counted once, even when the writer recorded nested spans.
        endpoints=sorted(set(x for interval in intervals for x in interval));union=Fraction(0)
        for a,b in zip(endpoints,endpoints[1:]):
            if any(left<=a and b<=right for left,right in intervals):union+=b-a
        check(case['name']+':union',parsed['exact_operation_union_seconds']==[union.numerator,union.denominator])
        check(case['name']+':not scientific qualification',parsed['trace_validity_is_scientific_qualification'] is False)
        if case['name']=='nested_success':
            base=raw;outer=(case['start_clock'],case['end_clock'])
    mutation=[]
    def mutate(name,offset,format,value):
        r=bytearray(base);struct.pack_into(format,r,offset,value);mutation.append((name,bytes(r)))
    mutate('magic',0,'8s',b'BADTRACE');mutate('version',8,'I',2)
    mutate('capacity',12,'I',4097);mutate('count',16,'I',9);mutate('closed',20,'I',2)
    mutate('header_padding',24,'B',1);mutate('sequence',64,'Q',2)
    mutate('phase',72,'I',20);mutate('status',104,'i',3);mutate('record_padding',108,'B',1)
    mutate('nonfinite_clock',88,'d',float('nan'));mutate('clock_order',88,'d',outer[1]+1.)
    mutate('unused_tail',64+2*64,'B',1)
    check('Exact registered mutation set',set(x[0] for x in mutation)==set(reg['validator_mutations']))
    for name,raw in mutation:
        rejected=False
        try:parse(raw,1,*outer,True)
        except ValueError:rejected=True
        check('malformed:'+name,rejected)
    for label,args in [('not_reaped',(base,1,*outer,False)),('outer_bounds',(base,1,outer[1],outer[1],True))]:
        rejected=False
        try:parse(*args)
        except ValueError:rejected=True
        check(label,rejected)
    receipt=json.loads((O/'ACTUAL_CLEANUP_RECEIPT.json').read_text(encoding='utf-8'))
    check('Actual killed worker return saved',receipt['actual_worker_returncode']!=0 and receipt['worker_reaped'] and receipt['writer_natural_exit'] is False)
    check('Real open entry reached',(O/'CHILD_OPEN_ENTRY.json').exists())
    check('Post cleanup clock',receipt['end_clock']>=receipt['reap_clock']>=receipt['kill_clock'])
    check('Actual raw worker hash after reap',sha(O/'cleanup_worker.bin')==receipt['worker_trace_sha256'])
    check('Actual raw caller hash',sha(O/'cleanup_caller.bin')==receipt['caller_trace_sha256'])
    check('No selected-output revocation',receipt['selected_output_revoked'] is False)
    check('Whole request qualification disabled',receipt['full_request_budget_qualified'] is False)
    check('Producer actual source start/end',result['source_unchanged'] is True)
    child=json.loads((O/'CHILD_OWNER.json').read_text(encoding='utf-8'))
    check('Actual owner identity',child==receipt['child_identity'])
    # Verify our exact child no longer exists; a later reused PID is distinct.
    try:
        p=psutil.Process(child['pid']);exited=p.create_time()!=child['create_time']
    except psutil.NoSuchProcess:exited=True
    check('Actual writer exited',exited)
    endsource=closure()
    with (O/'AUDIT_SOURCE_END.json').open('x',encoding='utf-8') as f:json.dump(endsource,f,indent=2)
    check('Independent actual loaded closure',source==endsource)
    failures=[x['name'] for x in checks if not x['pass_condition']]
    with (O/'INDEPENDENT_TRACE_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(dict(checks=len(checks),failures=failures,items=checks,
        scope='New trace-only kernel and cleanup helper; no model/parent workload replay',full_request_budget_qualified=False,
        full_scientific_loaded_closure_qualified=False,old_QA_repeated=False),f,ensure_ascii=False,indent=2)
    with (O/'AUDIT_RUN_EXIT.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=bool(failures)),f)
    print(json.dumps(dict(checks=len(checks),failures=len(failures),model_executed=False)))
    return bool(failures)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        with (O/'AUDIT_FAILURE.json').open('x',encoding='utf-8') as f:json.dump(dict(traceback=traceback.format_exc()),f)
        raise
