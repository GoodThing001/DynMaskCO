"""Bounded engineering on two previously observed fixtures, without new science."""
import argparse,hashlib,json,os,queue,subprocess,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';D=ROOT/'research/review_3080_20261010/delivery'
OUT=D/'retained_edge_teacher_engineering_root_v2';FIXTURE=D/'finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def boot():
    for sub in ('delivery','coldchain'):sys.path.insert(0,str(E/'scripts'/sub))
    sys.path.insert(0,str(OUT))
def affinity():
    import ctypes
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetCurrentProcess.restype=ctypes.c_void_p
    api.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    api.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    h=api.GetCurrentProcess()
    if not api.SetProcessAffinityMask(h,1<<7):raise ctypes.WinError(ctypes.get_last_error())
    own=ctypes.c_size_t();system=ctypes.c_size_t()
    if not api.GetProcessAffinityMask(h,ctypes.byref(own),ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
    result=[i for i in range(64) if own.value&(1<<i)]
    if result!=[7]:raise RuntimeError('Actual charged CPU affinity mismatch')
    return result

def worker():
    actual_cpu=affinity();boot()
    from retained_gls import solve
    from delivery_contract import DeliveryContract
    import ortools
    print(json.dumps({'kind':'ready','source_sha256':sha(OUT/'retained_gls.py'),'ortools_version':ortools.__version__,'cpu':actual_cpu}),flush=True)
    for line in sys.stdin:
        q=json.loads(line)
        try:
            if q.get('delay'):time.sleep(1.)
            r=solve(q['data'],q['initial'],q['partial'],DeliveryContract(),q['deadline'])
            print(json.dumps({'kind':'response','qualified':True,'result':r}),flush=True)
        except Exception as exc:print(json.dumps({'kind':'response','qualified':False,'error':type(exc).__name__+':'+str(exc)}),flush=True)
def main():
    actual_cpu=affinity();boot()
    from retained_gls import adjacency,constraints_input
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    import numpy as np
    rows=[json.loads(s) for s in FIXTURE.read_text().splitlines()]
    fixtures=[next(r for r in rows if r['instance_id']==i and r['mask_ordinal']==0) for i in (900000,900001)]
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'old two engineering fixtures only; no new train/development/confirmation data',
      'fixtures_sha256':sha(FIXTURE),'program_sha256':sha(__file__),'teacher_sha256':sha(OUT/'retained_gls.py'),
      'actual_parent_worker_cpu':actual_cpu,'request_cap_s':10.,'solver_search_cap_s':3.,'fault_cap_s':.01,
      'masks':['original_registered_mask_ordinal0','all_edges_retained'],
      'partial_adjacency_semantics':'undirected original customer/customer and any-vehicle depot endpoint edges; no fixed orientation or vehicle',
      'contract_hash':DeliveryContract().identity(),'research_new_days':0,'stable_online_gain':False,
      'references':['https://developers.google.com/optimization/routing/routing_tasks','https://proceedings.neurips.cc/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf']}
    write(OUT/'ENGINEERING_REGISTRATION.json',reg)
    q=queue.Queue();errors=[]
    p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    def read():
        for line in p.stdout:
            try:q.put(json.loads(line))
            except json.JSONDecodeError:errors.append(line)
        q.put({'kind':'closed'})
    def err():
        for line in p.stderr:errors.append(line)
    threading.Thread(target=read,daemon=True).start();threading.Thread(target=err,daemon=True).start()
    checks=[];records=[]
    def ck(k,v):checks.append({'name':k,'pass':bool(v)})
    try:
        ready=q.get(timeout=30);ck('bounded_setup',ready['kind']=='ready');ck('actual_source_ready',ready.get('source_sha256')==reg['teacher_sha256']);ck('actual_worker_cpu7',ready.get('cpu')==[7])
        for f in fixtures:
            for label,a in [('partial',f['partial_adjacency']),('all_edges',adjacency(f['incumbent_routes'],len(f['data']['demands'])).tolist())]:
                t=time.perf_counter();deadline=t+10.
                request={'data':f['data'],'initial':f['incumbent_routes'],'partial':a,'deadline':deadline}
                p.stdin.write(json.dumps(request)+'\n');p.stdin.flush()
                answer=q.get(timeout=max(0.,deadline-time.perf_counter()));received=time.perf_counter()
                ck(str((f['instance_id'],label))+':timely_actual_response',answer.get('qualified') is True and received<deadline)
                record={'instance_id':f['instance_id'],'mask_kind':label,'request':request,'started':t,'received':received,'wall_s':received-t,'answer':answer};records.append(record)
                if not answer.get('qualified'):continue
                r=answer['result'];report=evaluate(f['data'],r['routes'],DeliveryContract())
                ck(str((f['instance_id'],label))+':independent_complete_hard',report['hard'] and report['complete'])
                ck(str((f['instance_id'],label))+':retained_all',np.all(adjacency(r['routes'],len(a))[np.asarray(a)>0]>0))
                ck(str((f['instance_id'],label))+':C0_cost_replay',abs(report['cost']-r['report']['cost'])<1e-9)
                ck(str((f['instance_id'],label))+':own_incumbent_not_worse',report['cost']<=r['initial_report']['cost']+1e-9)
                for j,c in enumerate(r['all_candidates']):
                    ev=evaluate(f['data'],c['routes'],DeliveryContract());ck(str((f['instance_id'],label,j))+':candidate_complete_replay',ev['hard']==c['report']['hard'] and abs(ev['cost']-c['report']['cost'])<1e-9)
                    ck(str((f['instance_id'],label,j))+':candidate_mask_preserved',np.all(adjacency(c['routes'],len(a))[np.asarray(a)>0]>0))
        n=len(fixtures[0]['data']['demands']);good=fixtures[0]['partial_adjacency']
        for label,a in [('asymmetric',np.triu(good,1)),('nonbinary',np.asarray(good)*.5),('diagonal',np.eye(n)),('unowned',np.ones((n,n))-np.eye(n))]:
            rejected=False
            try:constraints_input(fixtures[0]['incumbent_routes'],a,n)
            except ValueError:rejected=True
            ck('reject_'+label,rejected)
        f=fixtures[0];t=time.perf_counter();deadline=t+.01
        p.stdin.write(json.dumps({'data':f['data'],'initial':f['incumbent_routes'],'partial':good,'deadline':deadline,'delay':True})+'\n');p.stdin.flush()
        try:fault=q.get(timeout=max(0.,deadline-time.perf_counter()))
        except queue.Empty:fault={'kind':'timeout'}
        ck('fault_no_reply_or_late_install',fault['kind']=='timeout');records.append({'fault_budget_s':.01,'reply_kind':fault['kind'],'fallback_routes':f['incumbent_routes'],'late_reply_applied':False})
    finally:
        if p.poll() is None:p.kill()
        p.wait(timeout=10)
    ck('own_worker_reaped',p.poll() is not None);ck('actual_source_end',sha(OUT/'retained_gls.py')==reg['teacher_sha256'] and sha(__file__)==reg['program_sha256']);ck('old_fixtures_unchanged',sha(FIXTURE)==reg['fixtures_sha256'])
    failed=sum(not c['pass'] for c in checks)
    write(OUT/'RESULT.json',{'checks':checks,'checks_count':len(checks),'failed':failed,'records':records,'ready':ready,'stderr':errors,'registration_sha256':sha(OUT/'ENGINEERING_REGISTRATION.json'),
      'retained_teacher_engineering_qualified':failed==0,'new_scientific_family_registered':False,'research_training_qualified':False,'stable_online_gain':False,'limits':['Observed fixture engineering only','Distance proposals with full C0 selection, not guaranteed optimal labels','No claim of live dynamic fleet qualification','All candidates retained; no timed science re-run']})
    print(json.dumps({'checks':len(checks),'failed':failed,'candidates_recorded':sum(len(r.get('answer',{}).get('result',{}).get('all_candidates',[])) for r in records),'research_new_days':0}));return int(failed>0)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');a=ap.parse_args()
    if a.worker:worker()
    else:raise SystemExit(main())
