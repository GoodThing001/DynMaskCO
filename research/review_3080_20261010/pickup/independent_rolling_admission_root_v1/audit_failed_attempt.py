"""Narrow independent saved-only audit; no controller/simulation imports."""
import ast,json,hashlib,sys,time
from pathlib import Path
from fractions import Fraction
import psutil
HERE=Path(__file__).resolve().parent
def read(n):return json.loads((HERE/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    with (HERE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    psutil.Process().cpu_affinity([7]);checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    prereg=read('FAILED_ATTEMPT_AUDIT_REGISTRATION.json');reg=read('REGISTRATION.json');receipt=read('RUN_RECEIPT.json');initial=read('SOURCE_START.json')
    for p,h in prereg['sources'].items():ck('audit_input:'+p,sha(p)==h)
    for p,h in reg['sources'].items():ck('original_registered_byte:'+p,sha(p)==h)
    for p,h in initial.items():ck('observed_start_byte_still_present:'+p,sha(p)==h)
    ck('producer_original_exit1',read('RUN_EXIT.json')['exit_code']==1)
    ck('no_qualified_response',receipt['qualified'] is False and receipt['done'] is None)
    ck('no_output_selection',receipt['output_selected'] is False)
    ck('actual_deadline_cancel',receipt['reason']=='worker deadline elapsed')
    ck('original_worker_kill_return1',receipt['worker_returncode']==1 and receipt['worker_reaped'])
    ck('source_gate_not_pass',receipt['source_gate_passed'] is False)
    ck('complete_result_not_fabricated',not (HERE/'COMPONENT_RESULT.json').exists())
    ck('natural_source_end_not_fabricated',not (HERE/'SOURCE_END.json').exists())
    elapsed=Fraction.from_float(receipt['finished'])-Fraction.from_float(receipt['started'])
    ck('elapsed_exact_subtraction',float(elapsed).hex()==float(receipt['elapsed']).hex())
    ck('total_under10',elapsed<Fraction(10));ck('execution_atleast8',elapsed>=Fraction(8))
    same_identity_alive=None;observation={}
    try:
        proc=psutil.Process(receipt['ready']['pid']);observation=dict(pid=proc.pid,create_time=proc.create_time(),cmdline=proc.cmdline())
        same_identity_alive=proc.create_time()==receipt['ready']['creation'] and str(HERE/'run.py') in proc.cmdline()
    except psutil.NoSuchProcess:same_identity_alive=False;observation=dict(pid=receipt['ready']['pid'],not_present=True)
    except (psutil.AccessDenied,OSError) as exc:observation=dict(error=repr(exc),identity_unknown=True)
    ck('same_owned_identity_not_alive',same_identity_alive is False)
    tree=ast.parse((HERE/'controller.py').read_text(encoding='utf-8'))
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    ck('source_has_no_old_admission_call',not any('on_reveal' in n or 'UncondDynamic' in n or '_uncond_lookahead_lambda' in n for n in calls))
    ck('source_owns_both_decision_branches',"generated = {'reject':" in ast.unparse(tree) and "'accept': []" in ast.unparse(tree))
    result=dict(checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),scope='saved-only original identities, exact timer, failed cancellation, source architecture; no original main audit/simulation/controller replay',actual_worker_observation=observation,registered_engineering_failure='fixed8s expired before result; stage unknown because this original attempt has no partial trace',full_branch_replay_qualified=False,producer_source_end_qualified=False,component_qualified=False,formal_strong_dynamic_qualified=False,full_request_budget_qualified=False,stable_online_gain=False)
    save('INDEPENDENT_FAILED_ATTEMPT_AUDIT.json',result);save('FAILED_ATTEMPT_AUDIT_EXIT.json',dict(exit_code=int(result['failed']>0)))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
