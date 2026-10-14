"""New live installation component QA from already accepted prototype records."""
import argparse,ctypes,hashlib,json,os,queue,subprocess,sys,threading,time
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import is_dataclass
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';P=ROOT/'research/review_3080_20261010/pickup'
OUT=P/'atomic_plan_installation_root_v1';INPUT=P/'actual_common_candidate_root_v2/RESULT.json'
for folder in (P,OUT):sys.path.insert(0,str(folder))
for folder in ('evaluation','simulation','coldchain','correction_engineering'):sys.path.insert(0,str(E/'scripts'/folder))
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def affinity():
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    h=k.GetCurrentProcess();a,b=ctypes.c_size_t(),ctypes.c_size_t()
    if not k.SetProcessAffinityMask(h,1<<7) or not k.GetProcessAffinityMask(h,ctypes.byref(a),ctypes.byref(b)):raise ctypes.WinError(ctypes.get_last_error())
    if a.value!=1<<7:raise RuntimeError('Actual CPU7 required')
    return [7]
def codecs():
    import coldchain_contract,coldchain_state,strict_online_env,scenario_saa
    from audit_io import encode,decode
    classes={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env,scenario_saa)
        for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
    return encode,lambda x:decode(x,classes)

def worker():
    cpu=affinity();encode,decode=codecs()
    import public_fleet as pf
    from recourse_snapshot import restore_recourse_snapshot
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from atomic_install import plan_hash
    # Import exact certifier before capturing the engineering source inventory.
    start=pf.source_hashes(E/'scripts');start[str(Path(__file__).resolve())]=sha(__file__)
    start[str(OUT/'atomic_install.py')]=sha(OUT/'atomic_install.py')
    print(json.dumps({'kind':'ready','cpu':cpu,'source_start':start}),flush=True)
    for line in sys.stdin:
        q=decode(json.loads(line))
        try:
            if q.get('delay_stage')=='certification':time.sleep(1.)
            cert,raw=pf.certificate(q['packet'],q['plan'])
            if q.get('delay_stage')=='compile':time.sleep(1.)
            compiled={int(k):list(route) for k,route in q['plan'].items()}
            end={p:sha(p) for p in start}
            response={'kind':'response','request_id':q['request_id'],'packet_digest':pf.digest(q['packet']),
                'compiled_plan':compiled,'compiled_plan_sha256':plan_hash(compiled),'certificate':cert,'certified':cert['qualified'] is True,
                'deadline':q['deadline'],'source_end':end,'source_qualified':end==start,
                'worker_finished':time.perf_counter(),'cpu':cpu}
            if response['worker_finished']>=q['deadline']:response['certified']=False
        except Exception as exc:response={'kind':'error','error':type(exc).__name__+':'+str(exc)}
        print(json.dumps(encode(response)),flush=True)

class Worker:
    def __init__(self):
        self.q=queue.Queue();self.errors=[];self.p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker'],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        def read():
            for line in self.p.stdout:
                try:self.q.put(json.loads(line))
                except Exception:self.errors.append(line)
            self.q.put({'kind':'closed'})
        def err():
            for line in self.p.stderr:self.errors.append(line)
        threading.Thread(target=read,daemon=True).start();threading.Thread(target=err,daemon=True).start()
        self.ready=self.q.get(timeout=30)
        if self.ready.get('kind')!='ready':raise RuntimeError('Worker setup failed')
    def ask(self,q):
        encode,decode=codecs();self.p.stdin.write(json.dumps(encode(q))+'\n');self.p.stdin.flush()
        try:return decode(self.q.get(timeout=max(0.,q['deadline']-time.perf_counter())))
        except queue.Empty:self.close();return {'kind':'timeout'}
    def close(self):
        if self.p.poll() is None:self.p.kill()
        self.p.wait(timeout=5)

def main():
    affinity();encode,decode=codecs();import public_fleet as pf
    from run_exp_energy_c0 import C0ReserveReplanner
    from recourse_snapshot import restore_recourse_snapshot
    from atomic_install import AtomicPlanOwner
    registration={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'candidate-only worker preparation and real C0ReserveReplanner plan-pointer publication; fixed old packet/five already produced plans',
        'input_sha256':sha(INPUT),'source_sha256':sha(__file__),'installer_sha256':sha(OUT/'atomic_install.py'),
        'normal_cap_s':10.,'delayed_fault_cap_s':.01,'actual_parent_worker_cpu':[7],
        'normal_modes':['own_incumbent','pretrained','random','discard','strong_rolling'],
        'faults':['certification_delay','compile_delay','changed_plan_pointer','mutated_plan_in_place','changed_admissions','changed_fleet','changed_event','expired','replayed_token','wrong_request_identity','unqualified_worker','tampered_prepared_plan'],
        'new_science_data':False,'all_request_stages_covered':False,'stable_online_gain':False}
    write(OUT/'REGISTRATION.json',registration)
    original=decode(json.loads(INPUT.read_text()));packet=original['public_packet'];checks=[];records=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    def owner():
        rp=C0ReserveReplanner(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.)
        rp.restore_state(packet['snapshot']['replanner_state']);fleet,_,_=restore_recourse_snapshot(packet['snapshot'])
        return AtomicPlanOwner(rp,fleet,packet['snapshot']['clock'],packet['snapshot']['event_id'])
    worker_obj=Worker()
    try:
        ck('actual_worker_affinity',worker_obj.ready['cpu']==[7]);mode_responses={}
        for row in original['records']:
            mode=row['mode'];plan=row['actual']['selected_plan'];live=owner();initial=live.rp.export_state()
            started=time.perf_counter();deadline=started+10.;token=live.begin(mode,deadline,pf.digest(packet))
            answer=worker_obj.ask({'packet':packet,'plan':plan,'deadline':deadline,'request_id':mode});received=time.perf_counter()
            # Convert decoding result to actual Python types before publication.
            normal_kind=answer.get('kind')=='response';ck(mode+':actual_response',normal_kind and received<deadline)
            if not normal_kind:raise RuntimeError('Missing normal prepared plan')
            saved_cert=pf.certificate
            def prohibited(*a,**k):raise RuntimeError('Physical certification must not run in commit parent')
            pf.certificate=prohibited
            try:installed,reason=live.commit(token,answer)
            finally:pf.certificate=saved_cert
            finished=time.perf_counter();expected={int(k):list(v) for k,v in plan.items()}
            ck(mode+':real_live_plan_installed',installed and live.rp._plan==expected)
            ck(mode+':one_pointer_publication',live.rp._plan is answer['compiled_plan'])
            ck(mode+':unchanged_admissions_status_budget',all(initial[k]==live.rp.export_state()[k] for k in ('accepted','rejected','status_inst','budget','mode')))
            ck(mode+':whole_component_in_10s',started<=received<=finished<deadline)
            ck(mode+':source_closed',answer['source_end']==worker_obj.ready['source_start'])
            ck(mode+':actual_full_C0_certificate',answer['certificate']['qualified'] and pf.digest(answer['certificate'])==pf.digest(row['actual']['selected_certificate']))
            mode_responses[mode]=answer
            records.append({'mode':mode,'started':started,'deadline':deadline,'received':received,'finished':finished,'prepared':answer,
                'before':initial,'after':live.rp.export_state(),'installed':installed,'reason':reason,'parent_certification_called':False})
        for case in ('changed_plan_pointer','mutated_plan_in_place','changed_admissions','changed_fleet','changed_event','expired','wrong_request_identity','unqualified_worker','tampered_prepared_plan'):
            live=owner();answer=dict(mode_responses['pretrained']);t=time.perf_counter();deadline=t+10.;answer['deadline']=deadline;answer['request_id']=case
            token=live.begin(case,deadline,pf.digest(packet))
            if case=='changed_plan_pointer':live.rp._plan=dict(live.rp._plan)
            elif case=='mutated_plan_in_place':live.rp._plan[0].append(-99)
            elif case=='changed_admissions':live.rp._rejected.add(-99)
            elif case=='changed_fleet':live.fleet[0].ready_time+=.001
            elif case=='changed_event':live.event+=1
            elif case=='expired':token['deadline']=t-1.;answer['deadline']=token['deadline']
            elif case=='wrong_request_identity':answer['request_id']='other'
            elif case=='unqualified_worker':answer['certified']=False
            elif case=='tampered_prepared_plan':answer['compiled_plan']={k:list(v) for k,v in answer['compiled_plan'].items()};answer['compiled_plan'][0].append(-99)
            before=live.rp._plan;installed,reason=live.commit(token,answer)
            ck(case+':not_installed',not installed and live.rp._plan is before)
            records.append({'fault':case,'installed':installed,'reason':reason,'plan_pointer_unchanged':live.rp._plan is before})
        live=owner();answer=dict(mode_responses['pretrained']);deadline=time.perf_counter()+10.;answer['deadline']=deadline;answer['request_id']='once';token=live.begin('once',deadline,pf.digest(packet))
        first,_=live.commit(token,answer);ptr=live.rp._plan;second,reason=live.commit(token,answer)
        ck('replayed_token_not_installed',first and not second and live.rp._plan is ptr)
        records.append({'fault':'replayed_token','first_installed':first,'second_installed':second,'reason':reason})
    finally:worker_obj.close()
    for stage in ('certification','compile'):
        w=Worker();live=owner();t=time.perf_counter();deadline=t+.01;token=live.begin(stage,deadline,pf.digest(packet));ptr=live.rp._plan
        try:reply=w.ask({'packet':packet,'plan':packet['plan'],'deadline':deadline,'request_id':stage,'delay_stage':stage})
        finally:w.close();live.cancel(token)
        ck(stage+':worker_killed_no_late_install',reply['kind']=='timeout' and w.p.poll() is not None and live.rp._plan is ptr)
        records.append({'fault':stage+'_delay','reply_kind':reply['kind'],'worker_reaped':w.p.poll() is not None,'plan_pointer_unchanged':live.rp._plan is ptr})
    ck('input_and_code_unchanged',sha(INPUT)==registration['input_sha256'] and sha(__file__)==registration['source_sha256'] and sha(OUT/'atomic_install.py')==registration['installer_sha256'])
    out={'checks':checks,'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'records':records,'ready':worker_obj.ready,
        'candidate_installation_component_qualified':all(x['pass'] for x in checks),'public_projection_included':False,'actual_admission_rejection_transaction_included':False,
        'all_five_formal_deployment_qualified':False,'full_day_qualified':False,'stable_online_gain':False,'engineering_only':True,
        'registration_sha256':sha(OUT/'REGISTRATION.json'),'raw_input_sha256':sha(INPUT),
        'limits':['Already exported old public packet; live public export and global caller transaction remain unqualified','Real replanner reconstructed from old snapshot, no new day policy experiment','Atomic in-process plan pointer only; no operating-system hard real-time guarantee','Sources bound for component execution, not full scientific interpreter or training closure']}
    write(OUT/'RESULT.json',encode(out));print(json.dumps({'checks':len(checks),'failed':out['failed'],'candidate_component_qualified':out['candidate_installation_component_qualified'],'full_transaction_qualified':False}));return int(out['failed']>0)

if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
