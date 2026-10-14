"""Independent stdlib byte parser and exact timer audit; no actor imports."""
import ast,json,struct,hashlib,sys
from fractions import Fraction
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'independent_rolling_admission_root_v1'
def read(n):return json.loads((HERE/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    with (HERE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    psutil.Process().cpu_affinity([7]);checks=[];notes=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    reg=read('REGISTRATION.json');receipt=read('RUN_RECEIPT.json');oldreg=json.loads((OLD/'REGISTRATION.json').read_text(encoding='utf-8'))
    for p,h in reg['sources'].items():ck('registered_source:'+p,sha(p)==h)
    for p,h in read('PREPARATION_IDENTITY.json')['original_sources'].items():ck('old_frozen_source:'+p,sha(p)==h)
    for key in ('input_selection','chosen','current','method','K','seed','worker_budget','cleanup_reserve','total_cap'):ck('same_registered_method:'+key,reg[key]==oldreg[key])
    ck('old_FAIL_unchanged',json.loads((OLD/'RUN_EXIT.json').read_text(encoding='utf-8'))['exit_code']==1)
    class StripTelemetry(ast.NodeTransformer):
        def visit_ImportFrom(self,n):return None if n.module=='phase_trace' else n
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='emit':return None
            return self.generic_visit(n)
        def visit_For(self,n):
            n=self.generic_visit(n)
            if isinstance(n.target,ast.Tuple) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='enumerate' and ast.unparse(n.iter.args[0])=='indices':
                n.target=ast.Name(id='ids',ctx=ast.Store());n.iter=ast.Name(id='indices',ctx=ast.Load())
            return n
    original=ast.parse((OLD/'controller.py').read_text(encoding='utf-8'));instrumented=StripTelemetry().visit(ast.parse((HERE/'controller.py').read_text(encoding='utf-8')))
    ck('method_AST_identical_after_only_telemetry_removal',ast.dump(original,include_attributes=False)==ast.dump(instrumented,include_attributes=False))
    raw=(HERE/'PHASE_TRACE.bin').read_bytes();ck('fixed_trace_size',len(raw)==32832)
    magic,capacity,count,pid,started,reserved=struct.unpack_from('<8sIIQQ32s',raw,0)
    ck('header_magic',magic==b'PKPHv001');ck('bounded_capacity_count',capacity==512 and 0<count<=512);ck('reserved_header',reserved==bytes(32));ck('writer_identity',pid==receipt['ready']['pid'])
    events=[];previous=started
    for j in range(min(count,512)):
        seq,ns,stage,index,branch,v0,v1,pad=struct.unpack_from('<QQ16siiQQ8s',raw,64+64*j)
        label=stage.rstrip(b'\0').decode('ascii');ck('sequence:'+str(j),seq==j+1);ck('clock:'+str(j),ns>=previous);ck('record_reserved:'+str(j),pad==bytes(8));ck('record_branch:'+str(j),branch in (-1,0,1));previous=ns
        events.append(dict(sequence=seq,clock_ns=ns,stage=label,index=index,branch=branch,value0=v0,value1=v1))
    ck('compute_reached',any(x['stage']=='compute_enter' for x in events));ck('trace_before_reap',Fraction(previous,10**9)<=Fraction.from_float(receipt['finished'])+Fraction(1,10**6))
    elapsed=Fraction.from_float(receipt['finished'])-Fraction.from_float(receipt['started']);ck('exact_outer_clock_subtraction',float(elapsed).hex()==float(receipt['elapsed']).hex());ck('total_observed_under10',elapsed<Fraction(10));ck('no_live_output_selection',receipt['output_selected'] is False);ck('owned_reap',receipt['worker_reaped'])
    alive=None;observation={}
    try:
        proc=psutil.Process(pid);observation=dict(create_time=proc.create_time(),cmdline=proc.cmdline());alive=proc.create_time()==receipt['ready']['creation'] and str(HERE/'run.py') in proc.cmdline()
    except psutil.NoSuchProcess:alive=False;observation=dict(pid=pid,not_present=True)
    except (psutil.AccessDenied,OSError) as exc:observation=dict(error=repr(exc),identity_unknown=True)
    ck('same_owned_writer_exited',alive is False)
    tail=raw[64+64*count:];tail_nonzero=any(tail);notes.append('Uncommitted nonzero tail preserved; no committed-event claim' if tail_nonzero else 'Unused tail zero')
    open_phases={};durations=[]
    pairs={'recourse_begin':'recourse_end','C0_begin':'C0_end','SAA_begin':'SAA_end','result_write_b':'result_write_e','source_begin':'source_end'}
    for e in events:
        key=(e['branch'],e['index'])
        if e['stage'] in pairs:open_phases[(e['stage'],)+key]=e
        for begin,end in pairs.items():
            if e['stage']==end:
                start=open_phases.pop((begin,)+key,None);ck('matched_end:'+str(e['sequence']),start is not None)
                if start is not None:durations.append(dict(stage=begin,branch=e['branch'],index=e['index'],start_ns=start['clock_ns'],end_ns=e['clock_ns'],duration_ns=e['clock_ns']-start['clock_ns']))
    summaries={}
    for branch in (0,1):
        summaries['reject' if branch==0 else 'accept']=dict(complete_C0=sum(e['stage']=='C0_end' and e['branch']==branch for e in events),complete_SAA=sum(e['stage']=='SAA_end' and e['branch']==branch for e in events))
    source_end=(HERE/'SOURCE_END.json').exists();out=dict(checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),events=events,durations=durations,completed_prefix=summaries,open_phases=list(open_phases.values()),last_committed_event=events[-1] if events else None,uncommitted_tail_nonzero=tail_nonzero,source_end_present=source_end,worker_observation=observation,notes=notes,qualification='new actual phase observations only; complete branch/physical independent replay not performed',producer_exit=read('RUN_EXIT.json')['exit_code'],original_prior_FAIL_preserved=True,component_qualified=False,formal_strong_dynamic_qualified=False,full_request_budget_qualified=False,full_scientific_loaded_closure=False,stable_online_gain=False)
    save('INDEPENDENT_PHASE_AUDIT.json',out);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(out['failed']>0)));print(json.dumps({k:v for k,v in out.items() if k not in ('checks','events','durations')}));return int(out['failed']>0)
if __name__=='__main__':raise SystemExit(main())
